"""Integrated v10.8.47 replay: exact Liquidity-Aware Control vs meta-veto.

One frozen LightGBM training/calibration chain builds the original Control
policies.  The same fold models, utility caches and execution assumptions are
then replayed twice:
  1) v10.8.44 Liquidity-Aware baseline;
  2) v10.8.47 Liquidity-Aware + counterfactual advantage meta-veto.

No operational registration and no order path.
"""
from __future__ import annotations

from types import FunctionType
from typing import Any, Callable

import pandas as pd

from ..tcc_v106_reference import research_challengers as scientific
from ..tcc_v106_reference.capital_rotation import _utility_policy as frozen_utility_policy
from .control_execution_feasibility import simulate_feasible_control
from .control_liquidity_policy import _CapitalAwareUtilityCache
from .control_counterfactual_advantage import (
    MODE,
    RANDOM_SEED,
    apply_meta_veto,
    build_cross_sectional_pair_samples,
    fit_meta_veto,
    predict_rotation_advantage_probability,
    refit_meta_veto,
)


def run_counterfactual_meta_veto_pair(
    bars: dict[str, pd.DataFrame],
    config: Any,
    fee_calculator: Callable,
    slippage: Callable,
    *,
    progress_callback: Callable[[float, str, int], None] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    (
        frames, common_dates, symbols, folds, _all_decision_dates,
        _decision_to_fold, _decision_metadata,
    ) = scientific._build_execution_context(bars, config)
    if len(symbols) != 55 or len(folds) != 3:
        raise ValueError("v10.8.47 expects frozen 55-asset / 3-fold Control context.")

    meta_by_fold: dict[int, dict[str, Any]] = {}
    fold_reports: list[dict[str, Any]] = []
    for fold in folds:
        fid = int(fold["fold_id"])
        train_dates = common_dates[:int(fold["train_end_index"])]
        calibration_dates = common_dates[
            int(fold["calibration_start_index"]):int(fold["calibration_end_index"])
        ]
        final_dates = common_dates[:int(fold["final_fit_end_index"])]

        training = build_cross_sectional_pair_samples(
            frames, symbols, train_dates,
            maturity_before=pd.Timestamp(fold["calibration_start"]),
        )
        validation = build_cross_sectional_pair_samples(
            frames, symbols, calibration_dates,
            maturity_before=pd.Timestamp(fold["test_start"]),
            date_stride=1,
        )
        if progress_callback:
            progress_callback(
                2.0 + 5.0 * (fid - 1),
                f"Meta-veto fold {fid}: chronological calibration",
                0,
            )
        _cal_model, _cal_scale, epochs, report = fit_meta_veto(
            frames, training, validation, seed=RANDOM_SEED + fid,
        )
        final_samples = build_cross_sectional_pair_samples(
            frames, symbols, final_dates,
            maturity_before=pd.Timestamp(fold["test_start"]),
        )
        final_model, final_scale = refit_meta_veto(
            frames, final_samples,
            seed=RANDOM_SEED + 100 + fid,
            epochs=epochs,
        )
        meta_by_fold[fid] = {
            "model": final_model,
            "scale": final_scale,
            "enabled": bool(report["model_enabled"]),
        }
        fold_reports.append({
            "fold_id": fid,
            "training_pairs": len(training),
            "calibration_pairs": len(validation),
            "final_training_pairs": len(final_samples),
            "test_start": pd.Timestamp(fold["test_start"]).isoformat(),
            "test_end": pd.Timestamp(fold["test_end"]).isoformat(),
            **report,
        })

    original_runner = scientific._run_lightgbm
    original_policy = original_runner.__globals__.get("_utility_policy")
    original_simulator = original_runner.__globals__.get("_simulate_exact")
    if original_policy is not frozen_utility_policy or original_simulator is not scientific._simulate_exact:
        raise RuntimeError("Frozen Control runner bindings changed unexpectedly.")

    account: dict[str, Any] = {"enabled": True, "mode": "baseline", "audit": {}}
    meta_audit: list[dict[str, Any]] = []

    def policy_wrapper(models, panel, labels, settings, switch_margin, **kwargs):
        utility_cache = kwargs.get("utility_cache")
        fold_id = kwargs.get("fold_id")
        if utility_cache is None or fold_id is None:
            return original_policy(
                models, panel, labels, settings, switch_margin, **kwargs,
            )
        fid = int(fold_id)
        liquidity_cache = _CapitalAwareUtilityCache(
            utility_cache, panel, labels, account,
        )
        base_kwargs = dict(kwargs)
        base_kwargs["utility_cache"] = liquidity_cache
        base_policy = original_policy(
            models, panel, labels, settings, switch_margin, **base_kwargs,
        )
        meta = meta_by_fold[fid]

        def policy(timestamp: pd.Timestamp, current_position: int, holding_days: int):
            control_target, control_score = base_policy(
                timestamp, current_position, holding_days,
            )
            if account.get("mode") != "meta":
                return control_target, control_score
            current_asset = labels[current_position-1] if current_position > 0 else "CASH"
            candidate_asset = labels[control_target-1] if control_target > 0 else "CASH"
            probability = predict_rotation_advantage_probability(
                meta["model"], meta["scale"], panel,
                date=pd.Timestamp(timestamp),
                incumbent=current_asset,
                candidate=candidate_asset,
            )
            final_target, reason = apply_meta_veto(
                current_position=int(current_position),
                control_target=int(control_target),
                probability_positive_advantage=probability,
                model_enabled=bool(meta["enabled"]),
            )
            meta_audit.append({
                "decision_date": pd.Timestamp(timestamp),
                "fold_id": fid,
                "incumbent_asset": current_asset,
                "control_target_asset": candidate_asset,
                "final_target_asset": (
                    labels[final_target-1] if final_target > 0 else "CASH"
                ),
                "probability_positive_advantage": probability,
                "model_enabled": bool(meta["enabled"]),
                "veto_applied": int(final_target) != int(control_target),
                "reason": reason,
            })
            return int(final_target), float(control_score)

        return policy

    captured: dict[str, Any] = {}
    def simulator_wrapper(*args, **kwargs):
        if captured:
            raise ValueError("v10.8.47 expected exactly one scheduled OOS simulation call.")

        def replay(mode: str):
            account.update({"mode": mode, "enabled": True, "audit": {}})
            def prepare(date, position, holding, cash, shares, equity):
                account.update({
                    "decision_timestamp": pd.Timestamp(date),
                    "position": int(position),
                    "holding_days": int(holding),
                    "cash": float(cash),
                    "shares": int(shares),
                    "equity": float(equity),
                })
            return simulate_feasible_control(
                *args,
                **{**kwargs, "decision_prepare": prepare},
            )

        captured["liquidity_baseline"] = replay("baseline")
        captured["meta_veto"] = replay("meta")
        return captured["liquidity_baseline"]

    isolated_globals = dict(original_runner.__globals__)
    isolated_globals["_utility_policy"] = policy_wrapper
    isolated_globals["_simulate_exact"] = simulator_wrapper
    isolated = FunctionType(
        original_runner.__code__, isolated_globals, original_runner.__name__,
        original_runner.__defaults__, original_runner.__closure__,
    )
    isolated.__kwdefaults__ = dict(original_runner.__kwdefaults__ or {})
    result = isolated(
        bars, config, fee_calculator, slippage,
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
        raise ValueError("v10.8.47 paired replay did not return the expected two paths.")
    if (
        original_runner.__globals__.get("_utility_policy") is not original_policy
        or original_runner.__globals__.get("_simulate_exact") is not original_simulator
    ):
        raise RuntimeError("Frozen TCC module was mutated by v10.8.47.")

    return captured, fold_reports, meta_audit
