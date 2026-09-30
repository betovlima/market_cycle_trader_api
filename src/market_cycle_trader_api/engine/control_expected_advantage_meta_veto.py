"""v10.8.55 expected-advantage regression Meta-Veto.

Candidate policy for research only. It predicts the paired rollout economic
advantage directly:

    delta_capital_fraction = (ROTATE ending equity - HOLD ending equity)
                             / initial equity

The candidate differs from v10.8.53 only in the decision model:
- fixed Ridge(alpha=1.0) regression;
- same nine features;
- same causal maturity;
- same one-shot HOLD semantics;
- veto when predicted delta_capital_fraction <= 0.0.

No threshold tuning is possible because zero is the economic break-even point.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import FunctionType
from typing import Any, Callable
import math

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..tcc_v106_reference import research_challengers as scientific
from ..tcc_v106_reference.capital_rotation import _utility_policy as frozen_utility_policy
from .control_execution_feasibility import simulate_feasible_control
from .control_liquidity_policy import _CapitalAwareUtilityCache
from .control_reduced_rollout_signature import REDUCED_FEATURES
from .control_reduced_signature_meta_veto import (
    _chronological_split,
    build_dynamic_feature_row,
)

RIDGE_ALPHA = 1.0
MIN_CALIBRATION_SAMPLES = 20
MIN_CALIBRATION_SIGN_BALANCED_ACCURACY = 0.52
MIN_CALIBRATION_SPEARMAN = 0.0


@dataclass(frozen=True)
class RegressionFoldModel:
    enabled: bool
    model: Pipeline | None
    disable_reason: str | None


def _make_model() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=RIDGE_ALPHA)),
    ])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    left = pd.Series(a, dtype=float).rank(method="average")
    right = pd.Series(b, dtype=float).rank(method="average")
    value = left.corr(right, method="pearson")
    return float(value) if pd.notna(value) else float("nan")


def train_regression_fold_models(
    dataset: pd.DataFrame,
    folds: list[dict[str, Any]],
) -> tuple[dict[int, RegressionFoldModel], list[dict[str, Any]]]:
    required = {
        "decision_date",
        "rollout_end_date",
        "source_fold_id",
        "rotate_better",
        "delta_capital_fraction",
        *REDUCED_FEATURES,
    }
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError(f"v10.8.55 source dataset missing columns: {missing}")

    source = dataset.copy()
    source["decision_date"] = pd.to_datetime(source["decision_date"], utc=True)
    source["rollout_end_date"] = pd.to_datetime(source["rollout_end_date"], utc=True)
    source = source.sort_values("decision_date").reset_index(drop=True)

    models: dict[int, RegressionFoldModel] = {}
    reports: list[dict[str, Any]] = []

    for fold in folds:
        fid = int(fold["fold_id"])
        test_start = pd.Timestamp(fold["test_start"])
        if test_start.tzinfo is None:
            test_start = test_start.tz_localize("UTC")
        else:
            test_start = test_start.tz_convert("UTC")

        prior = source.loc[
            (source["source_fold_id"].astype(int) < fid)
            & (source["rollout_end_date"] < test_start)
        ].copy()

        if fid == 1:
            models[fid] = RegressionFoldModel(False, None, "NO_PRIOR_OOS_FOLD")
            reports.append({
                "fold_id": fid,
                "eligible_prior_rows": 0,
                "training_rows": 0,
                "calibration_rows": 0,
                "model_enabled": False,
                "disable_reason": "NO_PRIOR_OOS_FOLD",
                "test_start": test_start.isoformat(),
            })
            continue

        training, calibration = _chronological_split(prior)
        if training.empty or len(calibration) < MIN_CALIBRATION_SAMPLES:
            models[fid] = RegressionFoldModel(False, None, "INSUFFICIENT_PRIOR_ROWS")
            reports.append({
                "fold_id": fid,
                "eligible_prior_rows": int(len(prior)),
                "training_rows": int(len(training)),
                "calibration_rows": int(len(calibration)),
                "model_enabled": False,
                "disable_reason": "INSUFFICIENT_PRIOR_ROWS",
                "test_start": test_start.isoformat(),
            })
            continue

        model = _make_model()
        model.fit(
            training[list(REDUCED_FEATURES)],
            pd.to_numeric(
                training["delta_capital_fraction"], errors="raise",
            ).astype(float),
        )
        predicted = model.predict(calibration[list(REDUCED_FEATURES)]).astype(float)
        actual = pd.to_numeric(
            calibration["delta_capital_fraction"], errors="raise",
        ).astype(float).to_numpy()
        actual_sign = (actual > 0.0).astype(int)
        predicted_sign = (predicted > 0.0).astype(int)

        sign_ba = float(
            balanced_accuracy_score(actual_sign, predicted_sign)
        )
        spearman = _spearman(predicted, actual)
        enabled = bool(
            len(calibration) >= MIN_CALIBRATION_SAMPLES
            and math.isfinite(sign_ba)
            and sign_ba >= MIN_CALIBRATION_SIGN_BALANCED_ACCURACY
            and math.isfinite(spearman)
            and spearman > MIN_CALIBRATION_SPEARMAN
        )

        final_model = None
        reason = None
        if enabled:
            final_model = _make_model()
            final_model.fit(
                prior[list(REDUCED_FEATURES)],
                pd.to_numeric(
                    prior["delta_capital_fraction"], errors="raise",
                ).astype(float),
            )
        else:
            reason = "CALIBRATION_GATE_FAILED"

        models[fid] = RegressionFoldModel(enabled, final_model, reason)
        reports.append({
            "fold_id": fid,
            "eligible_prior_rows": int(len(prior)),
            "training_rows": int(len(training)),
            "calibration_rows": int(len(calibration)),
            "final_training_rows": int(len(prior)) if enabled else 0,
            "calibration_actual_positive_rate": float(actual_sign.mean()),
            "calibration_predicted_positive_rate": float(predicted_sign.mean()),
            "calibration_sign_balanced_accuracy": sign_ba,
            "calibration_spearman": (
                spearman if math.isfinite(spearman) else None
            ),
            "calibration_prediction_mean": float(np.mean(predicted)),
            "calibration_actual_mean": float(np.mean(actual)),
            "model_enabled": enabled,
            "disable_reason": reason,
            "test_start": test_start.isoformat(),
            "ridge_alpha": RIDGE_ALPHA,
            "veto_break_even": 0.0,
        })

    return models, reports


def run_expected_advantage_meta_veto_pair(
    bars: dict[str, pd.DataFrame],
    config: Any,
    fee_calculator: Callable,
    slippage: Callable,
    *,
    source_dataset: pd.DataFrame,
    progress_callback: Callable[[float, str, int], None] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    (
        frames, _common_dates, symbols, folds, _all_decision_dates,
        _decision_to_fold, decision_metadata,
    ) = scientific._build_execution_context(bars, config)
    if len(symbols) != 55 or len(folds) != 3:
        raise ValueError("v10.8.55 expects frozen 55-asset / 3-fold Control context.")

    fold_models, training_reports = train_regression_fold_models(
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
            raise ValueError("v10.8.55 expected exactly one scheduled OOS simulator call.")
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

        def regression_policy(
            timestamp: pd.Timestamp,
            current_position: int,
            holding_days: int,
        ) -> tuple[int, float]:
            control_target, control_score = scheduled(
                timestamp, current_position, holding_days,
            )
            key = pd.Timestamp(timestamp)
            fid = int((metadata or {}).get(key, {}).get("fold_id") or 0)
            fold_model = fold_models.get(
                fid,
                RegressionFoldModel(False, None, "UNKNOWN_FOLD"),
            )
            incumbent = (
                labels[current_position - 1] if current_position > 0 else "CASH"
            )
            candidate = (
                labels[control_target - 1] if control_target > 0 else "CASH"
            )
            predicted_delta = None
            veto = False
            reason = "CONTROL_DEFAULT"

            if force_control_next[0]:
                force_control_next[0] = False
                reason = "CONTROL_AFTER_ONE_SHOT_VETO"
            elif (
                fold_model.enabled
                and current_position > 0
                and control_target > 0
                and current_position != control_target
            ):
                if pd.Timestamp(account.get("decision_timestamp")) != key:
                    raise ValueError(
                        "Regression Meta-Veto account state is not aligned."
                    )
                feature_row = build_dynamic_feature_row(
                    frames=panel,
                    date=key,
                    incumbent_asset=incumbent,
                    candidate_asset=candidate,
                    shares=int(account["shares"]),
                    equity=float(account["equity"]),
                )
                predicted_delta = float(
                    fold_model.model.predict(feature_row)[0]
                )
                if predicted_delta <= 0.0:
                    veto = True
                    force_control_next[0] = True
                    reason = "VETO_NONPOSITIVE_EXPECTED_ADVANTAGE"
                else:
                    reason = "MODEL_PASS"
            elif not fold_model.enabled:
                reason = fold_model.disable_reason or "MODEL_DISABLED"
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
                "predicted_delta_capital_fraction": predicted_delta,
                "model_enabled": bool(fold_model.enabled),
                "veto_applied": veto,
                "reason": reason,
                "state_shares": int(account.get("shares") or 0),
                "state_equity": float(account.get("equity") or 0.0),
            })
            return final_target, float(control_score)

        captured["meta_veto"] = simulate_feasible_control(
            backend,
            regression_policy,
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
        raise ValueError("v10.8.55 replay did not produce baseline and candidate.")
    if (
        original_runner.__globals__.get("_utility_policy") is not original_policy
        or original_runner.__globals__.get("_simulate_exact") is not original_simulator
    ):
        raise RuntimeError("Frozen TCC module was mutated by v10.8.55.")
    return captured, training_reports, audit
