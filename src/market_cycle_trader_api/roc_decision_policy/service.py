from __future__ import annotations

from collections import defaultdict
from typing import Any
from uuid import uuid4

import numpy as np

from ..classification_evaluation import roc_curve_payload
from ..infrastructure.persistence.mongo_repository import utc_now
from .calibration import calibrate_fold_horizon
from .config import settings_snapshot
from .errors import RocDecisionPolicyConflict
from .inputs import load_source_run, prepare_inputs
from .metrics import finite
from .persistence import latest_raw, persist, public_summary
from .replay import run_replay
from .validation import (
    build_comparison,
    control_metrics,
    infer_cash_rotation_policy,
    temporal_capital,
    threshold_stability,
    validate_control_parity,
)


ROC_POLICY_SCHEMA_VERSION = 2
ROC_POLICY_ENGINE_VERSION = "7.0.1-temporal-control-overlay"


def _entry_horizons(run: dict[str, Any], training: dict[str, Any]) -> list[int]:
    result = run.get("result") if isinstance(run.get("result"), dict) else {}
    multi = result.get("multi_horizon_metrics") if isinstance(result.get("multi_horizon_metrics"), dict) else {}
    configured = [int(value) for value in (multi.get("entry_horizons") or [])]
    available = {int(value) for value in (training.get("horizons") or [])}
    selected = [value for value in configured if value in available]
    if selected:
        return selected
    raise ValueError("Temporal Intelligence result does not expose the entry horizons required by ROC Decision Policy.")


def _oos_roc_rows(observations: list[dict[str, Any]], calibrations: list[dict[str, Any]], *, max_points: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(lambda: {"labels": [], "probabilities": []})
    thresholds = {(int(row["fold_id"]), int(row["horizon"])): row for row in calibrations if row.get("eligible")}
    for row in observations:
        fold_id = int(row.get("fold_id") or 0)
        for fold_horizon, calibration in thresholds.items():
            current_fold, horizon = fold_horizon
            if current_fold != fold_id:
                continue
            probability = finite(row.get(f"profit_before_loss_probability_h{horizon}"))
            realized = finite(row.get(f"realized_profit_before_loss_h{horizon}"))
            if probability is None or realized is None:
                continue
            grouped[(fold_id, horizon)]["labels"].append(1.0 if realized > 0.0 else 0.0)
            grouped[(fold_id, horizon)]["probabilities"].append(probability)
    rows: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        fold_id, horizon = key
        calibration = thresholds[key]
        labels = np.asarray(values["labels"], dtype=int)
        probabilities = np.asarray(values["probabilities"], dtype=float)
        roc = roc_curve_payload(
            labels,
            probabilities,
            operating_threshold=float(calibration["threshold"]),
            operating_point_role="roc_policy_threshold",
            threshold_origin="chronological_calibration_fold",
            validation_metric_name=str(calibration.get("selection_metric") or ""),
            validation_metric_value=finite(calibration.get("selection_score")),
            max_points=int(max_points),
        )
        rows.append({
            "fold_id": fold_id,
            "horizon": horizon,
            "threshold": float(calibration["threshold"]),
            "calibration_auc": calibration.get("calibration_auc"),
            "calibration_samples": calibration.get("calibration_samples"),
            "oos_auc": roc.get("auc"),
            "oos_samples": int(len(labels)),
            "roc": roc,
        })
    return rows


def run(
    db: Any,
    run_id: str,
    *,
    processing_id: str,
    start_month: str,
    end_month: str,
) -> dict[str, Any]:
    if str(end_month) < str(start_month):
        raise ValueError("ROC Decision Policy end_month must be greater than or equal to start_month.")

    snapshot = settings_snapshot(db)
    existing = latest_raw(db, run_id, processing_id=processing_id, start_month=start_month, end_month=end_month)
    if (
        existing is not None
        and str(existing.get("status") or "").lower() == "completed"
        and int(existing.get("schema_version") or 0) == ROC_POLICY_SCHEMA_VERSION
        and str(existing.get("engine_version") or "") == ROC_POLICY_ENGINE_VERSION
        and str(((existing.get("settings_snapshot") or {}).get("settings_hash") or "")) == str(snapshot.get("settings_hash") or "")
    ):
        return public_summary(existing) or {}

    source = load_source_run(db, run_id, processing_id)
    settings = dict(snapshot["settings"])
    prepared = prepare_inputs(db, source, processing_id)
    request = prepared["request"]
    training = prepared["training"]
    temporal_curve = prepared["temporal_curve"]
    entry_horizons = _entry_horizons(source, training)

    calibrations: list[dict[str, Any]] = []
    for fold in training.get("folds") or []:
        for horizon in entry_horizons:
            calibrations.append(calibrate_fold_horizon(
                training,
                request,
                fold=fold,
                horizon=horizon,
                settings=settings,
            ))
    eligible = [row for row in calibrations if row.get("eligible")]
    expected_pairs = len(training.get("folds") or []) * len(entry_horizons)
    if len(eligible) != expected_pairs:
        unavailable = [row for row in calibrations if not row.get("eligible")]
        raise ValueError(f"ROC calibration is incomplete for the current walk-forward analysis: {unavailable}")

    thresholds = {(int(row["fold_id"]), int(row["horizon"])): float(row["threshold"]) for row in eligible}
    one_side_cost = max(0.0, float(request.slippage_bps) / 10_000.0) + max(0.0, float(request.commission_rate))
    source_capital = temporal_capital(source)
    initial_capital = finite(source_capital.get("initial_capital"))
    if initial_capital is None or initial_capital <= 0:
        raise RocDecisionPolicyConflict("Temporal Intelligence does not expose a valid initial capital for ROC Decision Policy.")
    count_cash_transitions = infer_cash_rotation_policy(source, temporal_curve)

    parity_replay = run_replay(
        observations=prepared["observations"],
        winner_daily=prepared["winner_daily"],
        temporal_curve=temporal_curve,
        thresholds=thresholds,
        entry_horizons=entry_horizons,
        one_side_cost=one_side_cost,
        initial_capital=float(initial_capital),
        count_cash_transitions_as_rotations=count_cash_transitions,
        start_month=start_month,
        end_month=end_month,
        enable_roc=False,
    )
    control_parity = validate_control_parity(
        source,
        temporal_curve,
        parity_replay,
        start_month=start_month,
        end_month=end_month,
    )
    if str(control_parity.get("status") or "").lower() != "pass":
        failed = [key for key, value in (control_parity.get("checks") or {}).items() if not value]
        raise RocDecisionPolicyConflict(
            "ROC Control parity failed before applying ROC decisions: " + ", ".join(failed)
        )

    replay = run_replay(
        observations=prepared["observations"],
        winner_daily=prepared["winner_daily"],
        temporal_curve=temporal_curve,
        thresholds=thresholds,
        entry_horizons=entry_horizons,
        one_side_cost=one_side_cost,
        initial_capital=float(initial_capital),
        count_cash_transitions_as_rotations=count_cash_transitions,
        start_month=start_month,
        end_month=end_month,
        enable_roc=True,
    )
    control = control_metrics(source, temporal_curve, start_month=start_month, end_month=end_month)
    comparison = build_comparison(replay.get("metrics") or {}, control)
    oos_rows = _oos_roc_rows(prepared["observations"], eligible, max_points=int(settings["max_curve_points"]))
    by_key = {(int(row["fold_id"]), int(row["horizon"])): row for row in oos_rows}
    fold_horizon_rows = []
    for calibration in eligible:
        key = (int(calibration["fold_id"]), int(calibration["horizon"]))
        oos = by_key.get(key) or {}
        fold_horizon_rows.append({
            "fold_id": key[0],
            "horizon": key[1],
            "selected_threshold": calibration.get("threshold"),
            "selection_metric": calibration.get("selection_metric"),
            "selection_score": calibration.get("selection_score"),
            "calibration_auc": calibration.get("calibration_auc"),
            "calibration_samples": calibration.get("calibration_samples"),
            "oos_auc": oos.get("oos_auc"),
            "oos_samples": oos.get("oos_samples"),
            "roc": oos.get("roc"),
            "calibration_roc": calibration.get("calibration_roc"),
        })

    now = utc_now()
    document = {
        "schema_version": ROC_POLICY_SCHEMA_VERSION,
        "engine_version": ROC_POLICY_ENGINE_VERSION,
        "id": f"roc-policy-{uuid4().hex}",
        "status": "completed",
        "run_id": str(run_id),
        "processing_id": str(processing_id),
        "period_start": str(start_month),
        "period_end": str(end_month),
        "strategy_profile_id": source.get("strategy_profile_id"),
        "strategy_profile_revision": source.get("strategy_profile_revision"),
        "strategy_configuration_hash": source.get("strategy_configuration_hash"),
        "market_data_snapshot_id": source.get("market_data_snapshot_id"),
        "model_family": source.get("model_family"),
        "model_settings_hash": source.get("model_settings_hash"),
        "settings_snapshot": snapshot,
        "threshold_origin": "chronological_calibration_fold",
        "threshold_is_dynamic": True,
        "oos_used_for_threshold_selection": False,
        "control_source": "frozen_temporal_multi_horizon_equity_curve",
        "control_parity": control_parity,
        "entry_horizons": entry_horizons,
        "fold_horizons": fold_horizon_rows,
        "threshold_stability": threshold_stability(eligible, entry_horizons),
        "control": control,
        "challenger": replay.get("metrics") or {},
        "comparison": comparison,
        "folds": replay.get("folds") or [],
        "decision_diagnostics": replay.get("diagnostics") or [],
        "equity": replay.get("equity") or [],
        "created_at": now,
        "updated_at": now,
    }
    return public_summary(persist(db, document)) or {}
