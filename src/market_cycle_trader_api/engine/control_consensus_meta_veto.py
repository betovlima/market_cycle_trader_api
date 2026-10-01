"""v10.8.56 consensus Meta-Veto.

Research-only candidate that preserves v10.8.53 as the primary policy and uses
the frozen v10.8.55 expected-advantage regressor only as a secondary
confirmation layer.

The consensus can never create a new veto:
1. v10.8.53 classifier must first request a veto (P(ROTATE better) <= 0.35).
2. If the frozen Ridge regressor is enabled in that fold, it confirms the veto
   only when predicted_delta_capital_fraction <= 0.0.
3. If the regressor is disabled, fall back to the exact v10.8.53 classifier
   veto.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import FunctionType
from typing import Any, Callable

import pandas as pd

from ..tcc_v106_reference import research_challengers as scientific
from ..tcc_v106_reference.capital_rotation import _utility_policy as frozen_utility_policy
from .control_execution_feasibility import simulate_feasible_control
from .control_liquidity_policy import _CapitalAwareUtilityCache
from .control_reduced_signature_meta_veto import (
    FoldModel,
    VETO_PROBABILITY_MAX,
    build_dynamic_feature_row,
    train_fold_models,
)
from .control_expected_advantage_meta_veto import (
    RegressionFoldModel,
    train_regression_fold_models,
)

MODE = "MCT_RESEARCH_CONSENSUS_META_VETO_V1"


def run_consensus_meta_veto_pair(
    bars: dict[str, pd.DataFrame],
    config: Any,
    fee_calculator: Callable,
    slippage: Callable,
    *,
    source_dataset: pd.DataFrame,
    progress_callback: Callable[[float, str, int], None] | None = None,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    (
        frames,
        _common_dates,
        symbols,
        folds,
        _all_decision_dates,
        _decision_to_fold,
        decision_metadata,
    ) = scientific._build_execution_context(bars, config)
    if len(symbols) != 55 or len(folds) != 3:
        raise ValueError("v10.8.56 expects frozen 55-asset / 3-fold Control context.")

    classifier_models, classifier_training = train_fold_models(
        source_dataset, folds,
    )
    regression_models, regression_training = train_regression_fold_models(
        source_dataset, folds,
    )

    original_runner = scientific._run_lightgbm
    original_policy = original_runner.__globals__.get("_utility_policy")
    original_simulator = original_runner.__globals__.get("_simulate_exact")
    if (
        original_policy is not frozen_utility_policy
        or original_simulator is not scientific._simulate_exact
    ):
        raise RuntimeError("Frozen Control runner bindings changed unexpectedly.")

    account: dict[str, Any] = {"enabled": True, "audit": {}}
    captured: dict[str, Any] = {}
    audit: list[dict[str, Any]] = []

    def prepare_account(date, position, holding, cash, shares, equity):
        account.update({
            "decision_timestamp": pd.Timestamp(date),
            "position": int(position),
            "holding_days": int(holding),
            "cash": float(cash),
            "shares": int(shares),
            "equity": float(equity),
        })

    def policy_wrapper(models, panel, labels, settings, switch_margin, **kwargs):
        utility_cache = kwargs.get("utility_cache")
        if utility_cache is None:
            return original_policy(
                models, panel, labels, settings, switch_margin, **kwargs,
            )
        liquidity_cache = _CapitalAwareUtilityCache(
            utility_cache, panel, labels, account,
        )
        base_kwargs = dict(kwargs)
        base_kwargs["utility_cache"] = liquidity_cache
        return original_policy(
            models, panel, labels, settings, switch_margin, **base_kwargs,
        )

    def simulator_wrapper(*args, **kwargs):
        if captured:
            raise ValueError("v10.8.56 expected exactly one scheduled OOS simulator call.")
        backend, scheduled, panel, labels, dates, settings, fees, slip = args[:8]
        metadata = kwargs.get("decision_metadata") or decision_metadata

        baseline = simulate_feasible_control(
            backend,
            scheduled,
            panel,
            labels,
            dates,
            settings,
            fees,
            slip,
            **{**kwargs, "decision_prepare": prepare_account},
        )
        captured["liquidity_baseline"] = baseline

        force_control_next = [False]

        def consensus_policy(
            timestamp: pd.Timestamp,
            current_position: int,
            holding_days: int,
        ) -> tuple[int, float]:
            control_target, control_score = scheduled(
                timestamp, current_position, holding_days,
            )
            key = pd.Timestamp(timestamp)
            fid = int((metadata or {}).get(key, {}).get("fold_id") or 0)
            classifier = classifier_models.get(
                fid, FoldModel(False, None, "UNKNOWN_FOLD"),
            )
            regressor = regression_models.get(
                fid, RegressionFoldModel(False, None, "UNKNOWN_FOLD"),
            )
            incumbent = (
                labels[current_position - 1] if current_position > 0 else "CASH"
            )
            candidate = (
                labels[control_target - 1] if control_target > 0 else "CASH"
            )

            probability = None
            predicted_delta = None
            classifier_veto_candidate = False
            veto = False
            reason = "CONTROL_DEFAULT"

            if force_control_next[0]:
                force_control_next[0] = False
                reason = "CONTROL_AFTER_ONE_SHOT_VETO"
            elif (
                classifier.enabled
                and current_position > 0
                and control_target > 0
                and current_position != control_target
            ):
                if pd.Timestamp(account.get("decision_timestamp")) != key:
                    raise ValueError(
                        "Consensus Meta-Veto account state is not aligned."
                    )
                feature_row = build_dynamic_feature_row(
                    frames=panel,
                    date=key,
                    incumbent_asset=incumbent,
                    candidate_asset=candidate,
                    shares=int(account["shares"]),
                    equity=float(account["equity"]),
                )
                probability = float(
                    classifier.model.predict_proba(feature_row)[:, 1][0]
                )
                classifier_veto_candidate = (
                    probability <= VETO_PROBABILITY_MAX
                )
                if not classifier_veto_candidate:
                    reason = "CLASSIFIER_PASS"
                elif regressor.enabled:
                    predicted_delta = float(
                        regressor.model.predict(feature_row)[0]
                    )
                    if predicted_delta <= 0.0:
                        veto = True
                        force_control_next[0] = True
                        reason = "CONSENSUS_VETO"
                    else:
                        reason = "REGRESSION_CANCELLED_CLASSIFIER_VETO"
                else:
                    veto = True
                    force_control_next[0] = True
                    reason = "CLASSIFIER_VETO_REGRESSOR_UNAVAILABLE"
            elif not classifier.enabled:
                reason = classifier.disable_reason or "CLASSIFIER_DISABLED"
            elif current_position == 0 or control_target == 0:
                reason = "CASH_TRANSITION_NOT_ELIGIBLE"
            else:
                reason = "NO_ASSET_ROTATION"

            final_target = int(current_position) if veto else int(control_target)
            audit.append({
                "decision_date": key,
                "fold_id": fid,
                "incumbent_asset": incumbent,
                "control_target_asset": candidate,
                "final_target_asset": (
                    labels[final_target - 1] if final_target > 0 else "CASH"
                ),
                "probability_rotate_better": probability,
                "classifier_veto_candidate": classifier_veto_candidate,
                "regressor_enabled": bool(regressor.enabled),
                "predicted_delta_capital_fraction": predicted_delta,
                "veto_applied": veto,
                "reason": reason,
                "state_shares": int(account.get("shares") or 0),
                "state_equity": float(account.get("equity") or 0.0),
            })
            return final_target, float(control_score)

        captured["meta_veto"] = simulate_feasible_control(
            backend,
            consensus_policy,
            panel,
            labels,
            dates,
            settings,
            fees,
            slip,
            **{**kwargs, "decision_prepare": prepare_account},
        )
        return baseline

    isolated_globals = dict(original_runner.__globals__)
    isolated_globals["_utility_policy"] = policy_wrapper
    isolated_globals["_simulate_exact"] = simulator_wrapper
    isolated = FunctionType(
        original_runner.__code__,
        isolated_globals,
        original_runner.__name__,
        original_runner.__defaults__,
        original_runner.__closure__,
    )
    isolated.__kwdefaults__ = dict(original_runner.__kwdefaults__ or {})
    result = isolated(
        bars,
        config,
        fee_calculator,
        slippage,
        progress_callback=progress_callback,
        trade_callback=None,
        progress_detail_callback=None,
        technical_log_callback=None,
    )
    if (
        len(result) != 1
        or result[0] is not captured.get("liquidity_baseline")
        or set(captured) != {"liquidity_baseline", "meta_veto"}
    ):
        raise ValueError("v10.8.56 replay did not produce baseline and consensus.")
    if (
        original_runner.__globals__.get("_utility_policy") is not original_policy
        or original_runner.__globals__.get("_simulate_exact") is not original_simulator
    ):
        raise RuntimeError("Frozen TCC module was mutated by v10.8.56.")

    return captured, classifier_training, regression_training, audit
