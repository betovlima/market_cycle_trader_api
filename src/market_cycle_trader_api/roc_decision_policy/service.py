from __future__ import annotations

from collections import defaultdict
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from ..classification_evaluation import roc_curve_payload
from ..infrastructure.persistence.mongo_repository import utc_now
from .calibration import calibrate_fold_horizon
from .config import settings_snapshot
from .errors import RocDecisionPolicyConflict
from .inputs import load_source_run, prepare_inputs
from .metrics import finite
from .persistence import latest_raw, persist, public_summary
from .relative_model import score_pair
from .replay import run_replay
from .validation import (
    build_comparison,
    control_metrics,
    infer_cash_rotation_policy,
    temporal_capital,
    threshold_stability,
    validate_control_parity,
)


ROC_POLICY_SCHEMA_VERSION = 4
ROC_POLICY_ENGINE_VERSION = "7.2.0-relative-rotation-abstention"


def _stamp(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _asset(value: Any) -> str:
    text = str(value or "CASH").strip().upper()
    return text or "CASH"


def _entry_horizons(run: dict[str, Any], training: dict[str, Any]) -> list[int]:
    result = run.get("result") if isinstance(run.get("result"), dict) else {}
    multi = result.get("multi_horizon_metrics") if isinstance(result.get("multi_horizon_metrics"), dict) else {}
    configured = [int(value) for value in (multi.get("entry_horizons") or [])]
    available = {int(value) for value in (training.get("horizons") or [])}
    selected = [value for value in configured if value in available]
    if selected:
        return selected
    raise ValueError("Temporal Intelligence result does not expose the entry horizons required by ROC Decision Policy.")


def _entry_horizon_weights(request: Any, entry_horizons: list[int]) -> dict[int, float]:
    horizons = [int(value) for value in request.rotation_target_horizons]
    weights = [float(value) for value in request.rotation_target_horizon_weights]
    if len(horizons) != len(weights):
        raise RocDecisionPolicyConflict("Strategy horizon weights do not match the configured Temporal horizons.")
    configured = {horizon: weight for horizon, weight in zip(horizons, weights)}
    selected = {int(horizon): max(0.0, float(configured.get(int(horizon), 0.0))) for horizon in entry_horizons}
    total = sum(selected.values())
    if total <= 0.0:
        raise RocDecisionPolicyConflict("Strategy entry-horizon weights are unavailable for ROC Relative Rotation Policy.")
    return {horizon: weight / total for horizon, weight in selected.items()}


def _public_calibration(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not str(key).startswith("_")}


def _build_relative_oos_scores(
    training: dict[str, Any],
    winner_daily: list[dict[str, Any]],
    temporal_curve: list[dict[str, Any]],
    calibrations: list[dict[str, Any]],
    *,
    entry_horizons: list[int],
    horizon_weights: dict[int, float],
    round_trip_cost_rate: float,
    start_month: str,
    end_month: str,
) -> list[dict[str, Any]]:
    calibration_by_key = {
        (int(row["fold_id"]), int(row["horizon"])): row
        for row in calibrations
        if row.get("eligible")
    }
    winner_by_execution = {
        _stamp(row.get("timestamp")): dict(row)
        for row in winner_daily
        if isinstance(row, dict) and _stamp(row.get("timestamp")) is not None
    }
    output: list[dict[str, Any]] = []
    temporal_rows = [dict(row) for row in temporal_curve if isinstance(row, dict)]
    temporal_rows.sort(key=lambda row: _stamp(row.get("decision_timestamp")) or pd.Timestamp.max.tz_localize("UTC"))

    for temporal in temporal_rows:
        decision = _stamp(temporal.get("decision_timestamp"))
        execution = _stamp(temporal.get("execution_date"))
        if decision is None or not (start_month <= decision.strftime("%Y-%m") <= end_month):
            continue
        winner = winner_by_execution.get(execution) or {}
        fold_id = int(temporal.get("fold_id") or winner.get("walk_forward_fold") or winner.get("fold_id") or 0)
        temporal_target = _asset(temporal.get("target_symbol"))
        temporal_override = bool(temporal.get("temporal_timing_override"))
        winner_anchor = _asset(temporal.get("winner_anchor_symbol"))
        top1 = _asset(winner.get("top_1_asset") or winner.get("raw_best_asset") or winner.get("best_asset"))
        challenger = _asset(temporal.get("winner_top2_symbol") or winner.get("top_2_asset") or winner.get("second_asset"))
        action_eligible = bool(
            not temporal_override
            and temporal_target != "CASH"
            and temporal_target == winner_anchor
            and temporal_target == top1
            and challenger not in {"CASH", temporal_target}
        )
        if not action_eligible:
            continue

        details: list[dict[str, Any]] = []
        weighted_probability = 0.0
        weighted_threshold = 0.0
        used_weight = 0.0
        qualified_horizons: list[int] = []
        abstained_horizons: list[int] = []
        for horizon in entry_horizons:
            calibration = calibration_by_key.get((fold_id, int(horizon)))
            if calibration is None:
                continue
            detail = score_pair(
                training,
                calibration,
                decision_timestamp=decision,
                control_symbol=temporal_target,
                challenger_symbol=challenger,
                horizon=int(horizon),
                round_trip_cost_rate=round_trip_cost_rate,
            )
            if detail is None:
                continue
            weight = float(horizon_weights[int(horizon)])
            detail = {**detail, "weight": weight}
            details.append(detail)
            if bool(calibration.get("signal_qualified")):
                qualified_horizons.append(int(horizon))
                weighted_probability += weight * float(detail["probability"])
                weighted_threshold += weight * float(detail["threshold"])
                used_weight += weight
            else:
                abstained_horizons.append(int(horizon))

        if not details:
            continue
        signal_qualified = used_weight > 0.0
        aggregate_probability = (weighted_probability / used_weight) if signal_qualified else None
        aggregate_threshold = (weighted_threshold / used_weight) if signal_qualified else None
        aggregate_margin = (aggregate_probability - aggregate_threshold) if signal_qualified else None
        output.append({
            "fold_id": fold_id,
            "decision_timestamp": decision,
            "execution_date": execution,
            "control_asset": temporal_target,
            "challenger_asset": challenger,
            "signal_qualified": bool(signal_qualified),
            "qualification_status": "qualified" if signal_qualified else "abstain",
            "qualified_horizons": qualified_horizons,
            "abstained_horizons": abstained_horizons,
            "aggregate_probability": aggregate_probability,
            "aggregate_threshold": aggregate_threshold,
            "aggregate_margin": aggregate_margin,
            "horizons": details,
        })
    return output


def _qualification_summary(calibrations: list[dict[str, Any]], relative_scores: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in calibrations if row.get("eligible")]
    qualified = [row for row in eligible if row.get("signal_qualified")]
    abstained = [row for row in eligible if not row.get("signal_qualified")]
    candidate_sessions = len(relative_scores)
    qualified_sessions = sum(1 for row in relative_scores if row.get("signal_qualified"))
    return {
        "method": next((row.get("qualification_method") for row in eligible if row.get("qualification_method")), None),
        "eligible_fold_horizons": len(eligible),
        "qualified_fold_horizons": len(qualified),
        "abstained_fold_horizons": len(abstained),
        "candidate_sessions": candidate_sessions,
        "qualified_candidate_sessions": qualified_sessions,
        "abstained_candidate_sessions": max(0, candidate_sessions - qualified_sessions),
    }

def _oos_roc_rows(
    relative_scores: list[dict[str, Any]],
    calibrations: list[dict[str, Any]],
    *,
    max_points: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(lambda: {"labels": [], "probabilities": []})
    calibration_by_key = {
        (int(row["fold_id"]), int(row["horizon"])): row
        for row in calibrations
        if row.get("eligible")
    }
    for score in relative_scores:
        fold_id = int(score.get("fold_id") or 0)
        for detail in score.get("horizons") or []:
            horizon = int(detail.get("horizon") or 0)
            probability = finite(detail.get("probability"))
            realized = detail.get("realized_outperformance")
            if probability is None or realized is None or (fold_id, horizon) not in calibration_by_key:
                continue
            grouped[(fold_id, horizon)]["labels"].append(1.0 if bool(realized) else 0.0)
            grouped[(fold_id, horizon)]["probabilities"].append(probability)

    rows: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        fold_id, horizon = key
        calibration = calibration_by_key[key]
        labels = np.asarray(values["labels"], dtype=int)
        probabilities = np.asarray(values["probabilities"], dtype=float)
        roc = None
        auc = None
        if len(labels) and len(np.unique(labels)) == 2:
            roc = roc_curve_payload(
                labels,
                probabilities,
                operating_threshold=float(calibration["threshold"]),
                operating_point_role="relative_rotation_threshold",
                threshold_origin="chronological_relative_pair_calibration",
                validation_metric_name=str(calibration.get("selection_metric") or ""),
                validation_metric_value=finite(calibration.get("selection_score")),
                max_points=int(max_points),
            )
            auc = roc.get("auc")
        rows.append({
            "fold_id": fold_id,
            "horizon": horizon,
            "threshold": float(calibration["threshold"]),
            "calibration_auc": calibration.get("calibration_auc"),
            "calibration_samples": calibration.get("calibration_samples"),
            "oos_auc": auc,
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
    horizon_weights = _entry_horizon_weights(request, entry_horizons)
    one_side_cost = max(0.0, float(request.slippage_bps) / 10_000.0) + max(0.0, float(request.commission_rate))
    round_trip_cost_rate = 2.0 * one_side_cost

    calibrations: list[dict[str, Any]] = []
    for fold in training.get("folds") or []:
        for horizon in entry_horizons:
            calibrations.append(calibrate_fold_horizon(
                training,
                request,
                fold=fold,
                horizon=horizon,
                settings=settings,
                round_trip_cost_rate=round_trip_cost_rate,
            ))
    eligible = [row for row in calibrations if row.get("eligible")]
    expected_pairs = len(training.get("folds") or []) * len(entry_horizons)
    if len(eligible) != expected_pairs:
        unavailable = [_public_calibration(row) for row in calibrations if not row.get("eligible")]
        raise ValueError(f"ROC relative calibration is incomplete for the current walk-forward analysis: {unavailable}")

    relative_scores = _build_relative_oos_scores(
        training,
        prepared["winner_daily"],
        temporal_curve,
        eligible,
        entry_horizons=entry_horizons,
        horizon_weights=horizon_weights,
        round_trip_cost_rate=round_trip_cost_rate,
        start_month=start_month,
        end_month=end_month,
    )
    score_map = {
        _stamp(row.get("decision_timestamp")): row
        for row in relative_scores
        if _stamp(row.get("decision_timestamp")) is not None
    }
    signal_qualification = _qualification_summary(eligible, relative_scores)

    source_capital = temporal_capital(source)
    initial_capital = finite(source_capital.get("initial_capital"))
    if initial_capital is None or initial_capital <= 0:
        raise RocDecisionPolicyConflict("Temporal Intelligence does not expose a valid initial capital for ROC Decision Policy.")
    count_cash_transitions = infer_cash_rotation_policy(source, temporal_curve)

    parity_replay = run_replay(
        observations=prepared["observations"],
        winner_daily=prepared["winner_daily"],
        temporal_curve=temporal_curve,
        relative_scores=score_map,
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
        raise RocDecisionPolicyConflict("ROC Control parity failed before applying ROC decisions: " + ", ".join(failed))

    replay = run_replay(
        observations=prepared["observations"],
        winner_daily=prepared["winner_daily"],
        temporal_curve=temporal_curve,
        relative_scores=score_map,
        one_side_cost=one_side_cost,
        initial_capital=float(initial_capital),
        count_cash_transitions_as_rotations=count_cash_transitions,
        start_month=start_month,
        end_month=end_month,
        enable_roc=True,
    )
    control = control_metrics(source, temporal_curve, start_month=start_month, end_month=end_month)
    comparison = build_comparison(replay.get("metrics") or {}, control)
    oos_rows = _oos_roc_rows(relative_scores, eligible, max_points=int(settings["max_curve_points"]))
    by_key = {(int(row["fold_id"]), int(row["horizon"])): row for row in oos_rows}
    fold_horizon_rows = []
    for calibration in eligible:
        key = (int(calibration["fold_id"]), int(calibration["horizon"]))
        oos = by_key.get(key) or {}
        fold_horizon_rows.append({
            **_public_calibration(calibration),
            "selected_threshold": calibration.get("threshold"),
            "oos_auc": oos.get("oos_auc"),
            "oos_samples": oos.get("oos_samples"),
            "roc": oos.get("roc"),
        })

    public_calibrations = [_public_calibration(row) for row in eligible]
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
        "policy_target": "challenger_relative_outperformance_net_rotation_cost",
        "threshold_origin": "chronological_relative_pair_calibration",
        "threshold_is_dynamic": True,
        "signal_qualification": signal_qualification,
        "abstention_enabled": True,
        "oos_used_for_threshold_selection": False,
        "control_source": "frozen_temporal_multi_horizon_equity_curve",
        "control_parity": control_parity,
        "entry_horizons": entry_horizons,
        "entry_horizon_weights": horizon_weights,
        "round_trip_cost_rate": round_trip_cost_rate,
        "fold_horizons": fold_horizon_rows,
        "threshold_stability": threshold_stability(public_calibrations, entry_horizons),
        "control": control,
        "challenger": replay.get("metrics") or {},
        "comparison": comparison,
        "folds": replay.get("folds") or [],
        "relative_oos_scores": relative_scores,
        "decision_diagnostics": replay.get("diagnostics") or [],
        "equity": replay.get("equity") or [],
        "created_at": now,
        "updated_at": now,
    }
    return public_summary(persist(db, document)) or {}
