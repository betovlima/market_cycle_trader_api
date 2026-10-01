"""v10.8.57 temporal-ensemble Meta-Veto.

Research-only candidate that preserves the v10.8.53 model contract but avoids
pooling different prior temporal regimes into a single Logistic Regression.

For each test fold, one component is trained per fully matured prior source
fold. Each component must independently pass the frozen v10.8.53 BA/AUC gate.
The decision probability is the arithmetic mean of enabled component
probabilities. Veto threshold remains 0.35 and one-shot semantics are
unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import FunctionType
from typing import Any, Callable
import math

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from ..tcc_v106_reference import research_challengers as scientific
from ..tcc_v106_reference.capital_rotation import _utility_policy as frozen_utility_policy
from .control_execution_feasibility import simulate_feasible_control
from .control_liquidity_policy import _CapitalAwareUtilityCache
from .control_reduced_rollout_signature import REDUCED_FEATURES
from .control_reduced_signature_meta_veto import (
    MIN_CALIBRATION_SAMPLES,
    MIN_CALIBRATION_BALANCED_ACCURACY,
    MIN_CALIBRATION_ROC_AUC,
    VETO_PROBABILITY_MAX,
    FoldModel,
    _chronological_split,
    _make_model,
    build_dynamic_feature_row,
)

MODE = "MCT_RESEARCH_TEMPORAL_ENSEMBLE_META_VETO_V1"


@dataclass(frozen=True)
class TemporalEnsembleFoldModel:
    enabled: bool
    models: tuple[Any, ...]
    source_fold_ids: tuple[int, ...]
    component_weights: tuple[float, ...]
    disable_reason: str | None


def train_temporal_ensemble_fold_models(
    dataset: pd.DataFrame,
    folds: list[dict[str, Any]],
) -> tuple[dict[int, TemporalEnsembleFoldModel], list[dict[str, Any]]]:
    required = {
        "decision_date",
        "rollout_end_date",
        "source_fold_id",
        "rotate_better",
        *REDUCED_FEATURES,
    }
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError(f"v10.8.57 source dataset missing columns: {missing}")

    source = dataset.copy()
    source["decision_date"] = pd.to_datetime(source["decision_date"], utc=True)
    source["rollout_end_date"] = pd.to_datetime(
        source["rollout_end_date"], utc=True,
    )
    source = source.sort_values("decision_date").reset_index(drop=True)

    models: dict[int, TemporalEnsembleFoldModel] = {}
    reports: list[dict[str, Any]] = []

    for fold in folds:
        fid = int(fold["fold_id"])
        test_start = pd.Timestamp(fold["test_start"])
        if test_start.tzinfo is None:
            test_start = test_start.tz_localize("UTC")
        else:
            test_start = test_start.tz_convert("UTC")

        enabled_models: list[Any] = []
        enabled_sources: list[int] = []
        enabled_weights: list[float] = []
        component_reports: list[dict[str, Any]] = []

        if fid == 1:
            models[fid] = TemporalEnsembleFoldModel(
                False, tuple(), tuple(), tuple(), "NO_PRIOR_OOS_FOLD",
            )
            reports.append({
                "fold_id": fid,
                "test_start": test_start.isoformat(),
                "model_enabled": False,
                "disable_reason": "NO_PRIOR_OOS_FOLD",
                "enabled_component_count": 0,
                "enabled_source_fold_ids": [],
                "components": [],
            })
            continue

        for source_fid in range(1, fid):
            component = source.loc[
                (source["source_fold_id"].astype(int) == source_fid)
                & (source["rollout_end_date"] < test_start)
            ].copy()
            training, calibration = _chronological_split(component)

            component_report: dict[str, Any] = {
                "source_fold_id": source_fid,
                "eligible_rows": int(len(component)),
                "training_rows": int(len(training)),
                "calibration_rows": int(len(calibration)),
                "model_enabled": False,
                "disable_reason": None,
            }

            if (
                training.empty
                or len(calibration) < MIN_CALIBRATION_SAMPLES
            ):
                component_report["disable_reason"] = "INSUFFICIENT_PRIOR_ROWS"
                component_reports.append(component_report)
                continue

            gate_model = _make_model()
            gate_model.fit(
                training[list(REDUCED_FEATURES)],
                training["rotate_better"].astype(int),
            )
            probability = gate_model.predict_proba(
                calibration[list(REDUCED_FEATURES)]
            )[:, 1]
            prediction = (probability >= 0.50).astype(int)
            y_cal = calibration["rotate_better"].astype(int)
            ba = float(balanced_accuracy_score(y_cal, prediction))
            auc = (
                float(roc_auc_score(y_cal, probability))
                if y_cal.nunique() == 2 else float("nan")
            )
            enabled = bool(
                math.isfinite(ba)
                and ba >= MIN_CALIBRATION_BALANCED_ACCURACY
                and math.isfinite(auc)
                and auc >= MIN_CALIBRATION_ROC_AUC
            )

            component_report.update({
                "calibration_positive_rate": float(y_cal.mean()),
                "calibration_balanced_accuracy": ba,
                "calibration_roc_auc": auc if math.isfinite(auc) else None,
                "model_enabled": enabled,
                "disable_reason": None if enabled else "CALIBRATION_GATE_FAILED",
            })

            if enabled:
                final_model = _make_model()
                final_model.fit(
                    component[list(REDUCED_FEATURES)],
                    component["rotate_better"].astype(int),
                )
                skill_weight = float((ba - 0.5) + (auc - 0.5))
                if not math.isfinite(skill_weight) or skill_weight <= 0.0:
                    raise ValueError(
                        "Enabled temporal component must have positive skill weight."
                    )
                enabled_models.append(final_model)
                enabled_sources.append(source_fid)
                enabled_weights.append(skill_weight)
                component_report["skill_weight"] = skill_weight
                component_report["final_training_rows"] = int(len(component))
            else:
                component_report["final_training_rows"] = 0

            component_reports.append(component_report)

        fold_enabled = bool(enabled_models)
        reason = None if fold_enabled else "NO_ENABLED_TEMPORAL_COMPONENT"
        models[fid] = TemporalEnsembleFoldModel(
            fold_enabled,
            tuple(enabled_models),
            tuple(enabled_sources),
            tuple(enabled_weights),
            reason,
        )
        reports.append({
            "fold_id": fid,
            "test_start": test_start.isoformat(),
            "model_enabled": fold_enabled,
            "disable_reason": reason,
            "enabled_component_count": len(enabled_models),
            "enabled_source_fold_ids": list(enabled_sources),
            "components": component_reports,
            "aggregation": "arithmetic_mean_probability",
            "veto_probability_max": VETO_PROBABILITY_MAX,
        })

    return models, reports


def run_temporal_ensemble_meta_veto_pair(
    bars: dict[str, pd.DataFrame],
    config: Any,
    fee_calculator: Callable,
    slippage: Callable,
    *,
    source_dataset: pd.DataFrame,
    aggregation: str = "mean",
    progress_callback: Callable[[float, str, int], None] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
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
        raise ValueError("v10.8.57 expects frozen 55-asset / 3-fold Control context.")

    if aggregation not in {"mean", "min", "max", "skill_weighted"}:
        raise ValueError(
            "Temporal ensemble aggregation must be mean, min, max or skill_weighted."
        )
    fold_models, training_reports = train_temporal_ensemble_fold_models(
        source_dataset, folds,
    )
    for report in training_reports:
        report["aggregation"] = {
            "mean": "arithmetic_mean_probability",
            "min": "minimum_component_probability",
            "max": "maximum_component_probability",
            "skill_weighted": "calibration_skill_weighted_probability",
        }[aggregation]

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
            raise ValueError("v10.8.57 expected exactly one scheduled OOS simulator call.")
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

        def temporal_ensemble_policy(
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
                TemporalEnsembleFoldModel(
                    False, tuple(), tuple(), tuple(), "UNKNOWN_FOLD",
                ),
            )
            incumbent = (
                labels[current_position - 1] if current_position > 0 else "CASH"
            )
            candidate = (
                labels[control_target - 1] if control_target > 0 else "CASH"
            )

            probability = None
            component_probabilities: list[float] = []
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
                        "Temporal Ensemble account state is not aligned."
                    )
                feature_row = build_dynamic_feature_row(
                    frames=panel,
                    date=key,
                    incumbent_asset=incumbent,
                    candidate_asset=candidate,
                    shares=int(account["shares"]),
                    equity=float(account["equity"]),
                )
                component_probabilities = [
                    float(model.predict_proba(feature_row)[:, 1][0])
                    for model in fold_model.models
                ]
                if aggregation == "mean":
                    probability = float(np.mean(component_probabilities))
                elif aggregation == "min":
                    probability = float(np.min(component_probabilities))
                elif aggregation == "max":
                    probability = float(np.max(component_probabilities))
                else:
                    if (
                        len(fold_model.component_weights)
                        != len(component_probabilities)
                        or not fold_model.component_weights
                    ):
                        raise ValueError(
                            "Skill-weighted temporal ensemble requires one "
                            "positive calibration weight per component."
                        )
                    probability = float(np.average(
                        np.asarray(component_probabilities, dtype=float),
                        weights=np.asarray(
                            fold_model.component_weights, dtype=float,
                        ),
                    ))
                if probability <= VETO_PROBABILITY_MAX:
                    veto = True
                    force_control_next[0] = True
                    reason = {
                        "mean": "TEMPORAL_ENSEMBLE_VETO",
                        "min": "WORST_REGIME_VETO",
                        "max": "UNANIMOUS_TEMPORAL_VETO",
                        "skill_weighted": "SKILL_WEIGHTED_TEMPORAL_VETO",
                    }[aggregation]
                else:
                    reason = {
                        "mean": "TEMPORAL_ENSEMBLE_PASS",
                        "min": "WORST_REGIME_PASS",
                        "max": "UNANIMOUS_TEMPORAL_PASS",
                        "skill_weighted": "SKILL_WEIGHTED_TEMPORAL_PASS",
                    }[aggregation]
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
                "probability_rotate_better": probability,
                "component_probabilities": "|".join(
                    f"{value:.17g}" for value in component_probabilities
                ),
                "component_source_fold_ids": "|".join(
                    str(value) for value in fold_model.source_fold_ids
                ),
                "enabled_component_count": len(fold_model.models),
                "component_weights": "|".join(
                    f"{value:.17g}" for value in fold_model.component_weights
                ),
                "aggregation": aggregation,
                "veto_applied": veto,
                "reason": reason,
                "state_shares": int(account.get("shares") or 0),
                "state_equity": float(account.get("equity") or 0.0),
            })
            return final_target, float(control_score)

        captured["meta_veto"] = simulate_feasible_control(
            backend,
            temporal_ensemble_policy,
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
        raise ValueError("v10.8.57 replay did not produce baseline and ensemble.")
    if (
        original_runner.__globals__.get("_utility_policy") is not original_policy
        or original_runner.__globals__.get("_simulate_exact") is not original_simulator
    ):
        raise RuntimeError("Frozen TCC module was mutated by v10.8.57.")

    return captured, training_reports, audit
