from __future__ import annotations

import math
import uuid
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..infrastructure.persistence.mongo_repository import (
    TEMPORAL_INTELLIGENCE_RUNS_COLLECTION,
    TEMPORAL_WINNER_TRANSITION_CONFIDENCE_RESEARCH_COLLECTION,
    TEMPORAL_WINNER_TRANSITION_RISK_RESEARCH_COLLECTION,
    TEMPORAL_WINNER_TRANSITION_STATEFUL_RESEARCH_COLLECTION,
    bson_value,
    utc_now,
)
from .temporal_policy_search.policy import filter_observations, filter_winner_rows
from .temporal_policy_search.search_space import base_settings
from .temporal_policy_tuning import _load_artifact_rows, _load_observations
from .temporal_winner_transition_attribution import (
    _asset,
    _difference,
    _temporal_snapshot,
    _window_summary,
    _winner_rank,
    _winner_score,
)
from .temporal_winner_transition_attribution import get_winner_transition_attribution
from .model_research import model_execution_snapshot
from .strategy_lab import (
    StrategyLabConflict,
    StrategyLabError,
    StrategyLabNotFound,
    materialize_temporal_stateful_strategy,
)


FAMILY_FEATURES = {
    "temporal_rejection": (
        "temporal_3d_opportunity_gate_mean",
        "temporal_3d_entry_rank_mean",
        "temporal_3d_short_profit_mean",
    ),
    "fragile_leader": (
        "winner_5d_target_top1_rate",
        "winner_5d_incumbent_top1_rate",
        "winner_5d_target_top1_consecutive",
        "winner_5d_leader_change_count",
        "winner_5d_top1_top2_gap_latest",
        "winner_5d_target_minus_incumbent_score_latest",
        "winner_5d_target_minus_incumbent_score_delta",
        "temporal_3d_long_trend_mean",
        "temporal_3d_horizon_agreement_mean",
    ),
}
FAMILY_FEATURES["combined"] = tuple(dict.fromkeys(FAMILY_FEATURES["temporal_rejection"] + FAMILY_FEATURES["fragile_leader"]))


def _mean(values: list[Any]) -> float | None:
    clean = [_finite(value) for value in values]
    finite = [value for value in clean if value is not None]
    return float(sum(finite) / len(finite)) if finite else None


def _temporal_mean(sessions: list[dict[str, Any]], field: str, window: int = 3) -> float | None:
    values = []
    for session in sessions[-window:]:
        temporal = session.get("temporal") if isinstance(session.get("temporal"), dict) else {}
        delta = temporal.get("target_minus_incumbent") if isinstance(temporal.get("target_minus_incumbent"), dict) else {}
        values.append(delta.get(field))
    return _mean(values)


def _risk_features(transition: dict[str, Any]) -> dict[str, float | None]:
    trajectory = transition.get("trajectory") if isinstance(transition.get("trajectory"), dict) else {}
    sessions = [row for row in trajectory.get("sessions") or [] if isinstance(row, dict)]
    windows = trajectory.get("windows") if isinstance(trajectory.get("windows"), dict) else {}
    winner_5d = windows.get("5") if isinstance(windows.get("5"), dict) else {}
    return {
        "temporal_3d_opportunity_gate_mean": _temporal_mean(sessions, "opportunity_gate_score"),
        "temporal_3d_entry_rank_mean": _temporal_mean(sessions, "entry_rank_score"),
        "temporal_3d_short_profit_mean": _temporal_mean(sessions, "short_profit_consensus"),
        "temporal_3d_long_trend_mean": _temporal_mean(sessions, "long_trend_support"),
        "temporal_3d_horizon_agreement_mean": _temporal_mean(sessions, "horizon_agreement"),
        "winner_5d_target_top1_rate": _finite(winner_5d.get("target_top1_rate")),
        "winner_5d_incumbent_top1_rate": _finite(winner_5d.get("incumbent_top1_rate")),
        "winner_5d_target_top1_consecutive": _finite(winner_5d.get("target_top1_consecutive")),
        "winner_5d_leader_change_count": _finite(winner_5d.get("leader_change_count")),
        "winner_5d_top1_top2_gap_latest": _finite(winner_5d.get("top1_top2_gap_latest")),
        "winner_5d_target_minus_incumbent_score_latest": _finite(winner_5d.get("target_minus_incumbent_score_latest")),
        "winner_5d_target_minus_incumbent_score_delta": _finite(winner_5d.get("target_minus_incumbent_score_delta")),
    }


def _pipeline(seed: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=int(seed))),
    ])


def _transition_key(executed_at: Any, from_asset: Any, to_asset: Any) -> tuple[str, str, str] | None:
    stamp = _timestamp(executed_at)
    if stamp is None:
        return None
    source = str(from_asset or "").strip().upper()
    target = str(to_asset or "").strip().upper()
    if not source or not target:
        return None
    return (stamp.isoformat(), source, target)


def _build_transition_dataset(
    attribution: dict[str, Any],
    rotations: list[dict[str, Any]],
    *,
    severe_threshold: float,
) -> list[dict[str, Any]]:
    rotation_map = {}
    for rotation in rotations:
        key = _transition_key(rotation.get("executed_at"), rotation.get("from_asset"), rotation.get("to_asset"))
        if key is not None:
            rotation_map[key] = rotation
    rows = []
    for transition in attribution.get("items") or []:
        if not isinstance(transition, dict):
            continue
        key = _transition_key(transition.get("execution_at"), transition.get("from_asset"), transition.get("to_asset"))
        if key is None:
            continue
        rotation = rotation_map.get(key)
        holding_interval = transition.get("holding_interval_outcome") if isinstance(transition.get("holding_interval_outcome"), dict) else {}
        rotation_value_added = _finite(rotation.get("rotation_value_added")) if rotation else None
        attributed_value_added = _finite(holding_interval.get("value_added"))
        value_added = rotation_value_added if rotation_value_added is not None else attributed_value_added
        stamp = _timestamp(transition.get("execution_at"))
        if value_added is None or stamp is None:
            continue
        row = {
            "year": int(stamp.year),
            "severe": int(value_added <= float(severe_threshold)),
            "rotation_value_added": value_added,
        }
        row.update(_risk_features(transition))
        rows.append(row)
    return rows


class WinnerTransitionStatefulReplayError(RuntimeError):
    pass


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _timestamp(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp


def _timestamp_key(value: Any) -> str | None:
    stamp = _timestamp(value)
    return stamp.isoformat() if stamp is not None else None


def _month_rows(equity: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not equity:
        return []
    frame = pd.DataFrame(equity)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp")
    frame["month"] = frame["timestamp"].dt.strftime("%Y-%m")
    rows: list[dict[str, Any]] = []
    previous = None
    for month, group in frame.groupby("month", sort=True):
        end_value = float(group.iloc[-1]["simulation_equity"])
        start_value = float(group.iloc[0]["starting_value"]) if previous is None else float(previous)
        rows.append({
            "month": str(month),
            "start_value": start_value,
            "end_value": end_value,
            "return": float(end_value / start_value - 1.0) if start_value else None,
        })
        previous = end_value
    return rows


def _worst_month(equity: list[dict[str, Any]]) -> dict[str, Any] | None:
    rows = [row for row in _month_rows(equity) if _finite(row.get("return")) is not None]
    return min(rows, key=lambda row: float(row["return"]), default=None)


def _policy_target(
    rows_by_symbol: dict[str, dict[str, Any]],
    winner_row: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    top1_value = winner_row.get("top_1_asset") or winner_row.get("raw_best_asset") or winner_row.get("best_asset")
    top2_value = winner_row.get("top_2_asset") or winner_row.get("second_asset")
    base_value = winner_row.get("selected_asset") or winner_row.get("final_action_asset") or top1_value
    base_symbol = str(base_value) if base_value not in {None, "", "CASH"} else None
    top1_symbol = str(top1_value) if top1_value not in {None, "", "CASH"} else None
    challenger_symbol = str(top2_value) if top2_value not in {None, "", "CASH"} else None
    base_row = rows_by_symbol.get(base_symbol) if base_symbol else None
    challenger_row = rows_by_symbol.get(challenger_symbol) if challenger_symbol else None
    base_short = _finite((base_row or {}).get("short_profit_consensus"))
    challenger_short = _finite((challenger_row or {}).get("short_profit_consensus"))
    override = False
    if base_short is not None and challenger_short is not None:
        override = bool(
            base_symbol == top1_symbol
            and challenger_symbol != base_symbol
            and base_short < float(settings["timing_base_weak_threshold"])
            and challenger_short >= float(settings["timing_challenger_minimum"])
            and (challenger_short - base_short) >= float(settings["timing_minimum_advantage"])
            and (challenger_short - base_short) <= float(settings.get("timing_maximum_advantage", 1.0))
        )
    proposed = challenger_symbol if override else base_symbol
    return {
        "proposed_symbol": proposed,
        "base_symbol": base_symbol,
        "top1_symbol": top1_symbol,
        "top2_symbol": challenger_symbol,
        "timing_override": override,
    }


def _dynamic_transition_features(
    *,
    history_rows: list[dict[str, Any]],
    observations: dict[str, dict[str, Any]],
    target_symbol: str,
    incumbent_symbol: str,
) -> dict[str, float | None]:
    sessions: list[dict[str, Any]] = []
    for row in history_rows[-10:]:
        key = _timestamp_key(row.get("decision_date"))
        if not key:
            continue
        payload = observations.get(key) or {}
        rows_by_symbol = payload.get("rows_by_symbol") or {}
        target_row = rows_by_symbol.get(target_symbol)
        incumbent_row = rows_by_symbol.get(incumbent_symbol)
        target_temporal = _temporal_snapshot(target_row)
        incumbent_temporal = _temporal_snapshot(incumbent_row)
        target_score = _winner_score(row, target_symbol)
        incumbent_score = _winner_score(row, incumbent_symbol)
        top1_score = _finite(row.get("top_1_score"))
        top2_score = _finite(row.get("top_2_score"))
        sessions.append({
            "decision_at": row.get("decision_date"),
            "selected_asset": _asset(row, "selected_asset") or _asset(row, "final_action_asset") or None,
            "top1_asset": _asset(row, "top_1_asset") or _asset(row, "raw_best_asset") or _asset(row, "best_asset") or None,
            "top2_asset": _asset(row, "top_2_asset") or _asset(row, "second_asset") or None,
            "target_rank": _winner_rank(row, target_symbol),
            "incumbent_rank": _winner_rank(row, incumbent_symbol),
            "target_score": target_score,
            "incumbent_score": incumbent_score,
            "target_minus_incumbent_score": (
                float(target_score - incumbent_score)
                if target_score is not None and incumbent_score is not None
                else None
            ),
            "top1_top2_gap": (
                float(top1_score - top2_score)
                if top1_score is not None and top2_score is not None
                else None
            ),
            "universe_score_mean": _finite(row.get("universe_score_mean")),
            "universe_score_std": _finite(row.get("universe_score_std")),
            "positive_score_count": _finite(row.get("positive_score_count")),
            "finite_score_count": _finite(row.get("finite_score_count")),
            "temporal": {
                "target": target_temporal,
                "incumbent": incumbent_temporal,
                "target_minus_incumbent": _difference(target_temporal, incumbent_temporal),
            },
        })
    transition = {
        "trajectory": {
            "sessions": sessions,
            "windows": {
                str(window): _window_summary(sessions[-window:], target_symbol, incumbent_symbol)
                for window in (1, 3, 5, 10)
            },
        }
    }
    return _risk_features(transition)


def _risk_models(
    dataset: list[dict[str, Any]],
    risk_search: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    if not dataset:
        return {}
    frame = pd.DataFrame(dataset)
    seed = int(((risk_search.get("protocol") or {}).get("seed")) or 42)
    result: dict[int, dict[str, Any]] = {}
    for outer in risk_search.get("outer_results") or []:
        if not isinstance(outer, dict) or outer.get("test_year") is None:
            continue
        year = int(outer["test_year"])
        family = str(outer.get("selected_family") or "temporal_rejection")
        if family not in FAMILY_FEATURES:
            continue
        train = frame[frame["year"] < year]
        if train.empty or len(train["severe"].unique()) < 2:
            continue
        features = list(FAMILY_FEATURES[family])
        model = _pipeline(seed + year)
        model.fit(train[features], train["severe"].to_numpy(dtype=int))
        result[year] = {
            "family": family,
            "features": features,
            "model": model,
            "risk_threshold": _finite(outer.get("risk_threshold")),
        }
    return result


def _confidence_by_year(confidence: dict[str, Any]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in confidence.get("outer_results") or []:
        if not isinstance(row, dict) or row.get("test_year") is None:
            continue
        year = int(row["test_year"])
        active = str(row.get("selected_mode") or "") == "confidence_calibrated_one_session"
        result[year] = {
            "active": active,
            "margin_threshold": _finite(row.get("selected_margin_threshold")),
            "margin_quantile": _finite(row.get("selected_margin_quantile")),
        }
    return result


def _risk_score(model_payload: dict[str, Any] | None, feature_row: dict[str, Any]) -> float | None:
    if not model_payload:
        return None
    features = list(model_payload.get("features") or [])
    model = model_payload.get("model")
    if not features or model is None:
        return None
    frame = pd.DataFrame([{name: feature_row.get(name) for name in features}])
    return float(model.predict_proba(frame[features])[:, 1][0])


def _serialize_risk_model(model_payload: dict[str, Any]) -> dict[str, Any]:
    model = model_payload.get("model")
    if model is None:
        raise WinnerTransitionStatefulReplayError("Stateful risk model is unavailable.")
    imputer = model.named_steps.get("imputer")
    scaler = model.named_steps.get("scaler")
    classifier = model.named_steps.get("model")
    if imputer is None or scaler is None or classifier is None:
        raise WinnerTransitionStatefulReplayError("Stateful risk model pipeline is incomplete.")
    return {
        "family": model_payload.get("family"),
        "features": list(model_payload.get("features") or []),
        "risk_threshold": _finite(model_payload.get("risk_threshold")),
        "imputer_statistics": [float(value) for value in np.asarray(imputer.statistics_, dtype=float)],
        "scaler_mean": [float(value) for value in np.asarray(scaler.mean_, dtype=float)],
        "scaler_scale": [float(value) for value in np.asarray(scaler.scale_, dtype=float)],
        "coef": [float(value) for value in np.asarray(classifier.coef_[0], dtype=float)],
        "intercept": float(np.asarray(classifier.intercept_, dtype=float)[0]),
    }


def build_stateful_live_runtime_bundle(db: Any, strategy: dict[str, Any]) -> dict[str, Any]:
    policy = strategy.get("temporal_policy_snapshot") if isinstance(strategy.get("temporal_policy_snapshot"), dict) else {}
    stateful = policy.get("stateful_policy") if isinstance(policy.get("stateful_policy"), dict) else {}
    replay_id = str(strategy.get("source_stateful_replay_id") or stateful.get("source_stateful_replay_id") or policy.get("source_stateful_replay_id") or "").strip()
    run_id = str(strategy.get("source_temporal_run_id") or policy.get("source_run_id") or "").strip()
    processing_id = str(strategy.get("source_stateful_processing_id") or policy.get("source_processing_id") or "").strip()
    if not replay_id or not run_id or not processing_id:
        raise WinnerTransitionStatefulReplayError("Stateful Strategy is missing its source replay binding.")
    replay = db[TEMPORAL_WINNER_TRANSITION_STATEFUL_RESEARCH_COLLECTION].find_one(
        {"id": replay_id, "run_id": run_id, "status": "completed"},
        {"_id": 0},
    )
    if replay is None:
        raise WinnerTransitionStatefulReplayError("Stateful source replay is unavailable.")
    parity = replay.get("control_parity") if isinstance(replay.get("control_parity"), dict) else {}
    if str(parity.get("status") or "").lower() != "passed":
        raise WinnerTransitionStatefulReplayError("Stateful source replay does not have Control parity.")
    risk_id = str(replay.get("source_risk_search_id") or stateful.get("source_risk_search_id") or "").strip()
    confidence_id = str(replay.get("source_confidence_calibration_id") or stateful.get("source_confidence_calibration_id") or "").strip()
    risk = db[TEMPORAL_WINNER_TRANSITION_RISK_RESEARCH_COLLECTION].find_one({"id": risk_id}, {"_id": 0}) if risk_id else None
    confidence = db[TEMPORAL_WINNER_TRANSITION_CONFIDENCE_RESEARCH_COLLECTION].find_one({"id": confidence_id}, {"_id": 0}) if confidence_id else None
    run = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find_one({"id": run_id}, {"_id": 0})
    if risk is None or confidence is None or run is None:
        raise WinnerTransitionStatefulReplayError("Stateful source risk, confidence, or Temporal run is unavailable.")
    research_settings = risk.get("research_settings") if isinstance(risk.get("research_settings"), dict) else {}
    settings_payload = research_settings.get("settings") if isinstance(research_settings.get("settings"), dict) else research_settings
    risk_settings = settings_payload.get("risk") if isinstance(settings_payload.get("risk"), dict) else {}
    if "severe_threshold" not in risk_settings:
        raise WinnerTransitionStatefulReplayError("Stateful source risk settings are incomplete.")
    from .analytics import processing_analytics
    start_month = str(replay.get("period_start") or policy.get("period_start") or "").strip()
    end_month = str(replay.get("period_end") or policy.get("period_end") or "").strip()
    attribution = get_winner_transition_attribution(db, run_id, start_month=start_month, end_month=end_month)
    analytics = processing_analytics(db, processing_id)
    dataset = _build_transition_dataset(
        attribution,
        list(analytics.get("rotations") or []),
        severe_threshold=float(risk_settings["severe_threshold"]),
    )
    models = _risk_models(dataset, risk)
    if not models:
        raise WinnerTransitionStatefulReplayError("Stateful source risk models cannot be reconstructed.")
    serialized_models = {str(year): _serialize_risk_model(payload) for year, payload in models.items()}
    confidence_years = _confidence_by_year(confidence)
    return bson_value({
        "schema_version": 1,
        "mode": "conservative_one_session",
        "risk_models": serialized_models,
        "confidence_by_year": {str(year): payload for year, payload in confidence_years.items()},
        "policy_settings": base_settings(run),
        "source_replay_id": replay_id,
        "source_risk_search_id": risk_id,
        "source_confidence_calibration_id": confidence_id,
        "source_processing_id": processing_id,
        "source_run_id": run_id,
        "research_settings": research_settings,
        "control_parity": parity,
    })


def _serialized_risk_score(model_payload: dict[str, Any] | None, feature_row: dict[str, Any]) -> float | None:
    if not isinstance(model_payload, dict):
        return None
    features = list(model_payload.get("features") or [])
    statistics = list(model_payload.get("imputer_statistics") or [])
    means = list(model_payload.get("scaler_mean") or [])
    scales = list(model_payload.get("scaler_scale") or [])
    coefficients = list(model_payload.get("coef") or [])
    if not features or not (len(features) == len(statistics) == len(means) == len(scales) == len(coefficients)):
        return None
    linear = float(model_payload.get("intercept") or 0.0)
    for index, name in enumerate(features):
        value = _finite(feature_row.get(name))
        raw = float(statistics[index]) if value is None else float(value)
        scale = float(scales[index])
        normalized = (raw - float(means[index])) / (scale if abs(scale) > 1e-12 else 1.0)
        linear += float(coefficients[index]) * normalized
    if linear >= 0:
        return float(1.0 / (1.0 + math.exp(-linear)))
    exp_value = math.exp(linear)
    return float(exp_value / (1.0 + exp_value))


def _period_key(value: Any) -> str:
    stamp = _timestamp(value)
    return stamp.strftime("%Y-%m") if stamp is not None else ""


def _asset_name(value: Any) -> str:
    symbol = str(value or "CASH").strip().upper()
    return symbol or "CASH"


def _cost_sides(previous: str, target: str) -> int:
    previous = _asset_name(previous)
    target = _asset_name(target)
    if previous == target:
        return 0
    if previous == "CASH" or target == "CASH":
        return 1
    return 2


def _processing_control_path(
    analytics: dict[str, Any],
    *,
    start_month: str,
    end_month: str,
) -> dict[str, Any]:
    all_equity = [row for row in analytics.get("equity") or [] if isinstance(row, dict) and _timestamp(row.get("timestamp")) is not None]
    all_equity.sort(key=lambda row: _timestamp(row.get("timestamp")))
    equity = [row for row in all_equity if start_month <= _period_key(row.get("timestamp")) <= end_month]
    if len(equity) < 2:
        raise WinnerTransitionStatefulReplayError("The selected period does not contain enough Control equity sessions for a stateful replay.")

    all_rotations = [row for row in analytics.get("rotations") or [] if isinstance(row, dict) and _timestamp(row.get("executed_at")) is not None]
    all_rotations.sort(key=lambda row: _timestamp(row.get("executed_at")))
    first_stamp = _timestamp(equity[0].get("timestamp"))
    last_stamp = _timestamp(equity[-1].get("timestamp"))
    current = "CASH"
    rotation_index = 0
    while rotation_index < len(all_rotations) and _timestamp(all_rotations[rotation_index].get("executed_at")) < first_stamp:
        current = _asset_name(all_rotations[rotation_index].get("to_asset"))
        rotation_index += 1
    asset_before_first = current

    sessions: list[dict[str, Any]] = []
    for equity_row in equity:
        stamp = _timestamp(equity_row.get("timestamp"))
        movements = []
        while rotation_index < len(all_rotations):
            movement_stamp = _timestamp(all_rotations[rotation_index].get("executed_at"))
            if movement_stamp is None or movement_stamp > stamp:
                break
            movement = all_rotations[rotation_index]
            if movement_stamp == stamp:
                movements.append(movement)
            current = _asset_name(movement.get("to_asset"))
            rotation_index += 1
        sessions.append({
            "timestamp": stamp.isoformat(),
            "simulation_equity": float(equity_row.get("simulation_equity") or 0.0),
            "reference_equity": _finite(equity_row.get("reference_equity")),
            "selected_asset": current,
            "movements": movements,
        })

    rotations = [
        row for row in all_rotations
        if first_stamp <= _timestamp(row.get("executed_at")) <= last_stamp
    ]
    return {
        "sessions": sessions,
        "rotations": rotations,
        "asset_before_first": asset_before_first,
        "full_equity_count": len(all_equity),
        "full_scope": len(equity) == len(all_equity),
    }


def _observation_execution_map(observations: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for decision_key, payload in observations.items():
        rows_by_symbol = payload.get("rows_by_symbol") or {}
        sample = next((row for row in rows_by_symbol.values() if isinstance(row, dict)), None)
        execution_key = _timestamp_key((sample or {}).get("execution_date"))
        if execution_key:
            result[execution_key] = {"decision_key": decision_key, "payload": payload}
    return result


def _winner_history_by_decision(winner_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    histories: dict[int, list[dict[str, Any]]] = defaultdict(list)
    result: dict[str, list[dict[str, Any]]] = {}
    for row in sorted(
        [item for item in winner_rows if isinstance(item, dict) and _timestamp(item.get("decision_date")) is not None],
        key=lambda item: _timestamp(item.get("decision_date")),
    ):
        key = _timestamp_key(row.get("decision_date"))
        if not key:
            continue
        fold_id = int(row.get("fold_id") or 0)
        history = histories[fold_id]
        history.append(row)
        if len(history) > 10:
            del history[:-10]
        result[key] = list(history)
    return result


def _path_metrics(
    *,
    equity: list[dict[str, Any]],
    selected_assets: list[str],
    rotations: list[dict[str, Any]],
    initial_capital: float,
    interventions: int = 0,
    deferred_sessions: int = 0,
    repeated_defers: int = 0,
) -> dict[str, Any]:
    if not equity:
        raise WinnerTransitionStatefulReplayError("Stateful replay produced no equity path.")
    values = np.asarray([float(row.get("simulation_equity") or 0.0) for row in equity], dtype=float)
    daily_returns = []
    if initial_capital > 0:
        daily_returns.append(float(values[0] / initial_capital - 1.0))
    daily_returns.extend(float(values[index] / values[index - 1] - 1.0) if values[index - 1] > 0 else 0.0 for index in range(1, len(values)))
    daily = np.asarray(daily_returns, dtype=float)
    peaks = np.maximum.accumulate(values)
    drawdowns = np.divide(values, peaks, out=np.ones_like(values), where=peaks != 0) - 1.0
    years = max(len(daily) / 252.0, 1.0 / 252.0)
    ending_capital = float(values[-1])
    cagr = (ending_capital / initial_capital) ** (1.0 / years) - 1.0 if initial_capital > 0 and ending_capital > 0 else -1.0
    volatility = float(np.std(daily, ddof=1)) if len(daily) > 1 else 0.0
    sharpe = float(np.mean(daily) / volatility * math.sqrt(252.0)) if volatility > 1e-12 else 0.0
    cash_days = sum(_asset_name(asset) == "CASH" for asset in selected_assets)
    capital_rotations = sum(
        1
        for rotation in rotations
        if _asset_name(rotation.get("from_asset")) != "CASH"
        and _asset_name(rotation.get("to_asset")) != "CASH"
        and _asset_name(rotation.get("from_asset")) != _asset_name(rotation.get("to_asset"))
    )
    metrics = {
        "initial_capital": float(initial_capital),
        "ending_capital": ending_capital,
        "strategy_return": float(ending_capital / initial_capital - 1.0) if initial_capital else None,
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "maximum_drawdown": float(np.min(drawdowns)),
        "capital_rotations": int(capital_rotations),
        "average_holding_days": float(len(selected_assets) / max(1, capital_rotations)),
        "market_exposure": float((len(selected_assets) - cash_days) / max(1, len(selected_assets))),
        "cash_days": int(cash_days),
        "interventions": int(interventions),
        "deferred_sessions": int(deferred_sessions),
        "repeated_deferred_sessions": int(repeated_defers),
    }
    metrics["worst_month"] = _worst_month(equity)
    metrics["monthly_returns"] = _month_rows(equity)
    return metrics


def _candidate_rotation(
    *,
    sequence: int,
    executed_at: str,
    previous_asset: str,
    target_asset: str,
    before_capital: float,
    entry_capital: float | None,
    entry_at: str | None,
    reason: str,
    intervention: bool,
    risk_score: float | None,
    risk_threshold: float | None,
    confidence_margin: float | None,
    confidence_threshold: float | None,
    one_side_cost: float,
) -> dict[str, Any]:
    realized = None
    position_return = None
    holding = None
    if previous_asset != "CASH" and entry_capital not in {None, 0.0}:
        realized = float(before_capital - float(entry_capital))
        position_return = float(before_capital / float(entry_capital) - 1.0)
        if entry_at:
            try:
                holding = max(0, (pd.Timestamp(executed_at) - pd.Timestamp(entry_at)).days)
            except Exception:
                holding = None
    return {
        "sequence": int(sequence),
        "executed_at": executed_at,
        "from_asset": previous_asset,
        "to_asset": target_asset,
        "holding_days": holding,
        "position_return": position_return,
        "realized_pnl": realized,
        "transaction_fees": float(before_capital * _cost_sides(previous_asset, target_asset) * one_side_cost),
        "sell_reason": reason if previous_asset != "CASH" else None,
        "buy_reason": reason if target_asset != "CASH" else None,
        "stateful_intervention": bool(intervention),
        "risk_score": risk_score,
        "risk_threshold": risk_threshold,
        "confidence_margin": confidence_margin,
        "confidence_threshold": confidence_threshold,
    }


def _parity_anchored_candidate(
    *,
    control_path: dict[str, Any],
    observations: dict[str, dict[str, Any]],
    winner_rows: list[dict[str, Any]],
    settings: dict[str, Any],
    risk_models: dict[int, dict[str, Any]],
    confidence_by_year: dict[int, dict[str, Any]],
    initial_capital: float,
    one_side_cost: float,
    mode: str,
) -> dict[str, Any]:
    sessions = control_path.get("sessions") or []
    if len(sessions) < 2:
        raise WinnerTransitionStatefulReplayError("Control path has too few sessions.")
    observations_by_execution = _observation_execution_map(observations)
    winner_by_decision = {
        key: row for row in winner_rows
        if isinstance(row, dict) and (key := _timestamp_key(row.get("decision_date")))
    }
    histories = _winner_history_by_decision(winner_rows)

    current_asset = _asset_name(control_path.get("asset_before_first"))
    candidate_capital = float(sessions[0].get("simulation_equity") or 0.0)
    selected_assets: list[str] = []
    equity: list[dict[str, Any]] = []
    peak = candidate_capital
    rotations: list[dict[str, Any]] = []
    entry_capital = candidate_capital if current_asset != "CASH" else None
    entry_at = sessions[0]["timestamp"] if current_asset != "CASH" else None
    cooldown = False
    intervention_count = 0
    deferred_sessions = 0
    repeated_defers = 0
    previous_intervention = False

    for index, session in enumerate(sessions):
        executed_at = str(session.get("timestamp"))
        control_previous = _asset_name(control_path.get("asset_before_first")) if index == 0 else _asset_name(sessions[index - 1].get("selected_asset"))
        control_target = _asset_name(session.get("selected_asset"))
        previous_asset = current_asset
        target_asset = control_target
        intervention = False
        risk_score = None
        risk_threshold = None
        confidence_margin = None
        confidence_threshold = None
        model_family = None
        reason = "control_path"

        observation_link = observations_by_execution.get(executed_at)
        decision_key = (observation_link or {}).get("decision_key")
        payload = (observation_link or {}).get("payload") or {}
        rows_by_symbol = payload.get("rows_by_symbol") or {}
        winner_row = winner_by_decision.get(decision_key) if decision_key else None
        year = int(pd.Timestamp(executed_at).year)
        confidence = confidence_by_year.get(year) or {"active": False, "margin_threshold": None}
        model_payload = risk_models.get(year)

        gate_allowed = bool(
            previous_asset != "CASH"
            and control_target != "CASH"
            and previous_asset != control_target
            and bool(confidence.get("active"))
            and winner_row is not None
            and rows_by_symbol
            and rows_by_symbol.get(previous_asset) is not None
            and rows_by_symbol.get(control_target) is not None
            and _finite((rows_by_symbol.get(previous_asset) or {}).get("open_to_open_return")) is not None
            and _finite((rows_by_symbol.get(control_target) or {}).get("open_to_open_return")) is not None
        )
        if gate_allowed:
            proposed = _policy_target(rows_by_symbol, winner_row, settings)
            gate_allowed = bool(
                not proposed.get("timing_override")
                and control_target == _asset_name(proposed.get("base_symbol"))
                and control_target == _asset_name(proposed.get("top1_symbol"))
            )
        if mode == "conservative_one_session" and cooldown:
            gate_allowed = False

        if gate_allowed and model_payload and decision_key:
            feature_row = _dynamic_transition_features(
                history_rows=histories.get(decision_key) or [winner_row],
                observations=observations,
                target_symbol=control_target,
                incumbent_symbol=previous_asset,
            )
            risk_score = _risk_score(model_payload, feature_row)
            risk_threshold = _finite(model_payload.get("risk_threshold"))
            confidence_threshold = _finite(confidence.get("margin_threshold"))
            if risk_score is not None and risk_threshold is not None:
                confidence_margin = float(risk_score - risk_threshold)
                intervention = bool(confidence_threshold is not None and confidence_margin >= confidence_threshold)
                model_family = model_payload.get("family")

        if intervention:
            target_asset = previous_asset
            intervention_count += 1
            deferred_sessions += 1
            if mode == "adaptive_long" and previous_intervention:
                repeated_defers += 1
            if mode == "conservative_one_session":
                cooldown = True
            reason = "stateful_confidence_defer"
        elif mode == "conservative_one_session" and cooldown:
            cooldown = False

        before_capital = candidate_capital
        if previous_asset != target_asset:
            rotations.append(_candidate_rotation(
                sequence=len(rotations) + 1,
                executed_at=executed_at,
                previous_asset=previous_asset,
                target_asset=target_asset,
                before_capital=before_capital,
                entry_capital=entry_capital,
                entry_at=entry_at,
                reason=reason,
                intervention=intervention,
                risk_score=risk_score,
                risk_threshold=risk_threshold,
                confidence_margin=confidence_margin,
                confidence_threshold=confidence_threshold,
                one_side_cost=one_side_cost,
            ))
            if target_asset != "CASH":
                entry_capital = float(before_capital * max(1e-9, 1.0 - _cost_sides(previous_asset, target_asset) * one_side_cost))
                entry_at = executed_at
            else:
                entry_capital = None
                entry_at = None
        elif target_asset != "CASH" and entry_capital is None:
            entry_capital = before_capital
            entry_at = executed_at

        peak = max(peak, candidate_capital)
        selected_assets.append(target_asset)
        equity.append({
            "timestamp": executed_at,
            "simulation_equity": float(candidate_capital),
            "reference_equity": float(session.get("simulation_equity") or 0.0),
            "starting_value": float(initial_capital),
            "drawdown": float(candidate_capital / peak - 1.0) if peak > 0 else 0.0,
            "selected_asset": target_asset,
            "trade_action": "HOLD" if previous_asset == target_asset else ("CASH" if target_asset == "CASH" else "ROTATE"),
            "stateful_intervention": bool(intervention),
            "risk_family": model_family,
        })

        if index < len(sessions) - 1:
            next_session = sessions[index + 1]
            baseline_before = float(session.get("simulation_equity") or 0.0)
            baseline_after = float(next_session.get("simulation_equity") or 0.0)
            baseline_factor = baseline_after / baseline_before if baseline_before > 0 else 1.0
            control_return = 0.0 if control_target == "CASH" else _finite((rows_by_symbol.get(control_target) or {}).get("open_to_open_return"))
            candidate_return = 0.0 if target_asset == "CASH" else _finite((rows_by_symbol.get(target_asset) or {}).get("open_to_open_return"))
            if target_asset != control_target and candidate_return is None:
                target_asset = control_target
                intervention = False
                reason = "control_path_missing_alternative_return"
                candidate_return = control_return
                selected_assets[-1] = target_asset
                equity[-1]["selected_asset"] = target_asset
                equity[-1]["stateful_intervention"] = False
                if mode == "conservative_one_session":
                    cooldown = False
            if target_asset == control_target and previous_asset == control_previous:
                candidate_factor = baseline_factor
            else:
                control_sides = _cost_sides(control_previous, control_target)
                candidate_sides = _cost_sides(previous_asset, target_asset)
                if control_return is not None:
                    expected_control = max(1e-9, 1.0 - control_sides * one_side_cost) * max(1e-9, 1.0 + float(control_return))
                    residual = baseline_factor / expected_control if expected_control > 0 else 1.0
                else:
                    residual = baseline_factor / max(1e-9, 1.0 - control_sides * one_side_cost)
                if candidate_return is None:
                    candidate_return = control_return if target_asset == control_target else 0.0
                candidate_factor = residual * max(1e-9, 1.0 - candidate_sides * one_side_cost) * max(1e-9, 1.0 + float(candidate_return or 0.0))
            candidate_capital *= max(1e-9, float(candidate_factor))

        current_asset = target_asset
        previous_intervention = intervention

    metrics = _path_metrics(
        equity=equity,
        selected_assets=selected_assets,
        rotations=rotations,
        initial_capital=initial_capital,
        interventions=intervention_count,
        deferred_sessions=deferred_sessions,
        repeated_defers=repeated_defers,
    )
    return {"mode": mode, "analytics": {"equity": equity, "rotations": rotations, "metrics": metrics}}


def _control_analytics_from_path(control_path: dict[str, Any], *, initial_capital: float) -> dict[str, Any]:
    sessions = control_path.get("sessions") or []
    rotations = [bson_value(dict(row)) for row in control_path.get("rotations") or []]
    equity = []
    peak = 0.0
    assets = []
    for session in sessions:
        value = float(session.get("simulation_equity") or 0.0)
        peak = max(peak, value)
        asset = _asset_name(session.get("selected_asset"))
        assets.append(asset)
        equity.append({
            "timestamp": session.get("timestamp"),
            "simulation_equity": value,
            "reference_equity": session.get("reference_equity"),
            "starting_value": float(initial_capital),
            "drawdown": float(value / peak - 1.0) if peak > 0 else 0.0,
            "selected_asset": asset,
            "trade_action": "CASH" if asset == "CASH" else "HOLD",
            "stateful_intervention": False,
        })
    metrics = _path_metrics(
        equity=equity,
        selected_assets=assets,
        rotations=rotations,
        initial_capital=initial_capital,
    )
    return {"equity": equity, "rotations": rotations, "metrics": metrics}


def _control_parity(
    *,
    analytics: dict[str, Any],
    control_path: dict[str, Any],
    replay_analytics: dict[str, Any],
) -> dict[str, Any]:
    source_metrics = analytics.get("metrics") if isinstance(analytics.get("metrics"), dict) else {}
    replay_metrics = replay_analytics.get("metrics") if isinstance(replay_analytics.get("metrics"), dict) else {}
    source_end = _finite(source_metrics.get("ending_capital")) if control_path.get("full_scope") else _finite((control_path.get("sessions") or [{}])[-1].get("simulation_equity"))
    replay_end = _finite(replay_metrics.get("ending_capital"))
    source_cash = int(source_metrics.get("cash_days") or 0) if control_path.get("full_scope") else sum(_asset_name(row.get("selected_asset")) == "CASH" for row in control_path.get("sessions") or [])
    replay_cash = int(replay_metrics.get("cash_days") or 0)
    source_exposure = _finite(source_metrics.get("market_exposure")) if control_path.get("full_scope") else float((len(control_path.get("sessions") or []) - source_cash) / max(1, len(control_path.get("sessions") or [])))
    replay_exposure = _finite(replay_metrics.get("market_exposure"))
    source_rotations = len(control_path.get("rotations") or [])
    replay_rotations = len(replay_analytics.get("rotations") or [])
    capital_delta = float(replay_end / source_end - 1.0) if replay_end is not None and source_end not in {None, 0.0} else None
    exposure_delta = float(replay_exposure - source_exposure) if replay_exposure is not None and source_exposure is not None else None
    checks = {
        "ending_capital": capital_delta is not None and abs(capital_delta) <= 1e-10,
        "cash_days": replay_cash == source_cash,
        "market_exposure": exposure_delta is not None and abs(exposure_delta) <= 1e-12,
        "rotations": replay_rotations == source_rotations,
        "equity_sessions": len(replay_analytics.get("equity") or []) == len(control_path.get("sessions") or []),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "source": {
            "ending_capital": source_end,
            "cash_days": source_cash,
            "market_exposure": source_exposure,
            "rotations": source_rotations,
            "equity_sessions": len(control_path.get("sessions") or []),
        },
        "replay": {
            "ending_capital": replay_end,
            "cash_days": replay_cash,
            "market_exposure": replay_exposure,
            "rotations": replay_rotations,
            "equity_sessions": len(replay_analytics.get("equity") or []),
        },
        "ending_capital_delta_rate": capital_delta,
        "market_exposure_delta": exposure_delta,
    }


def run_stateful_transition_replay_from_payloads(
    *,
    run: dict[str, Any],
    processing_id: str,
    start_month: str,
    end_month: str,
    observations: dict[str, dict[str, Any]],
    winner_rows: list[dict[str, Any]],
    analytics: dict[str, Any],
    transition_attribution: dict[str, Any],
    risk_search: dict[str, Any],
    confidence: dict[str, Any],
) -> dict[str, Any]:
    request = run.get("request") if isinstance(run.get("request"), dict) else {}
    source_metrics = analytics.get("metrics") if isinstance(analytics.get("metrics"), dict) else {}
    initial_capital = float(source_metrics.get("initial_capital") or request.get("initial_capital") or 10_000.0)
    one_side_cost = max(0.0, float(request.get("slippage_bps") or 0.0) / 10_000.0) + max(0.0, float(request.get("commission_rate") or 0.0))
    settings = base_settings(run)
    research_settings = risk_search.get("research_settings") if isinstance(risk_search.get("research_settings"), dict) else {}
    settings_payload = research_settings.get("settings") if isinstance(research_settings.get("settings"), dict) else research_settings
    risk_settings = settings_payload.get("risk") if isinstance(settings_payload.get("risk"), dict) else {}
    if "severe_threshold" not in risk_settings:
        raise WinnerTransitionStatefulReplayError("The source risk search does not contain a frozen temporal research settings snapshot.")
    severe_threshold = float(risk_settings["severe_threshold"])
    dataset = _build_transition_dataset(
        transition_attribution,
        list(analytics.get("rotations") or []),
        severe_threshold=severe_threshold,
    )
    models = _risk_models(dataset, risk_search)
    confidence_years = _confidence_by_year(confidence)
    control_path = _processing_control_path(analytics, start_month=start_month, end_month=end_month)
    baseline_analytics = _control_analytics_from_path(control_path, initial_capital=initial_capital)
    parity = _control_parity(analytics=analytics, control_path=control_path, replay_analytics=baseline_analytics)

    conservative = None
    adaptive = None
    if parity.get("status") == "passed":
        conservative = _parity_anchored_candidate(
            control_path=control_path,
            observations=observations,
            winner_rows=winner_rows,
            settings=settings,
            risk_models=models,
            confidence_by_year=confidence_years,
            initial_capital=initial_capital,
            one_side_cost=one_side_cost,
            mode="conservative_one_session",
        )
        adaptive = _parity_anchored_candidate(
            control_path=control_path,
            observations=observations,
            winner_rows=winner_rows,
            settings=settings,
            risk_models=models,
            confidence_by_year=confidence_years,
            initial_capital=initial_capital,
            one_side_cost=one_side_cost,
            mode="adaptive_long",
        )

    return bson_value({
        "schema_version": 2,
        "id": f"winner-transition-stateful-{utc_now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}",
        "run_id": str(run.get("id") or ""),
        "processing_id": str(processing_id),
        "period_start": start_month,
        "period_end": end_month,
        "created_at": utc_now(),
        "status": "completed" if parity.get("status") == "passed" else "blocked",
        "research_settings": research_settings,
        "protocol": {
            "base_policy": "processing_control_path_parity_anchor",
            "detector": "chronological_winner_transition_risk_models",
            "confidence": "walk_forward_confidence_calibration",
            "candidate_a": "one_consecutive_defer_then_follow_current_control_target",
            "candidate_b": "adaptive_repeated_defer_while_high_confidence",
            "stateful_incumbent": True,
            "base_control_target_read_each_session": True,
            "cash_path_preserved": True,
            "temporal_risk_recomputed_vs_stateful_incumbent": True,
            "confidence_recomputed_each_session": True,
            "future_control_transition_used_for_rejoin": False,
            "frozen_observations": True,
            "parity_required_before_candidates": True,
            "research_only": True,
            "strategy_decisions_changed": False,
        },
        "source_risk_search_id": risk_search.get("id"),
        "source_confidence_calibration_id": confidence.get("id"),
        "control_parity": parity,
        "control_replay": {
            "analytics": baseline_analytics,
            "processing_control_ending_capital": parity.get("source", {}).get("ending_capital"),
            "replay_ending_capital": parity.get("replay", {}).get("ending_capital"),
            "replay_vs_processing_delta_rate": parity.get("ending_capital_delta_rate"),
        },
        "candidate_a": ({"label": "Conservative Stateful", **conservative} if conservative else None),
        "candidate_b": ({"label": "Adaptive Long Stateful", **adaptive} if adaptive else None),
    })



def run_winner_transition_stateful_replay(
    db: Any,
    run_id: str,
    *,
    processing_id: str,
    start_month: str,
    end_month: str,
) -> dict[str, Any]:
    run = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find_one({"id": str(run_id)})
    if run is None or str(run.get("status") or "").lower() != "completed":
        raise WinnerTransitionStatefulReplayError("A completed Temporal Intelligence run is required.")
    risk = db[TEMPORAL_WINNER_TRANSITION_RISK_RESEARCH_COLLECTION].find_one(
        {"run_id": str(run_id), "processing_id": str(processing_id), "period_start": start_month, "period_end": end_month, "status": {"$in": ["completed", "blocked"]}},
        {"_id": 0}, sort=[("created_at", -1)],
    )
    confidence = db[TEMPORAL_WINNER_TRANSITION_CONFIDENCE_RESEARCH_COLLECTION].find_one(
        {"run_id": str(run_id), "processing_id": str(processing_id), "period_start": start_month, "period_end": end_month, "status": "completed"},
        {"_id": 0}, sort=[("created_at", -1)],
    )
    if risk is None:
        raise WinnerTransitionStatefulReplayError("Run Winner transition risk search before the stateful replay.")
    if confidence is None:
        raise WinnerTransitionStatefulReplayError("Run confidence calibration before the stateful replay.")
    observations = filter_observations(_load_observations(db, run_id), start_month=start_month, end_month=end_month)
    winner_rows = filter_winner_rows(_load_artifact_rows(db, run_id, "winner_reference_daily"), observations)
    if not observations or not winner_rows:
        raise WinnerTransitionStatefulReplayError("Frozen Temporal observations or Winner rows are unavailable for the selected period.")
    from .analytics import processing_analytics
    analytics = processing_analytics(db, processing_id)
    attribution = get_winner_transition_attribution(db, run_id, start_month=start_month, end_month=end_month)
    result = run_stateful_transition_replay_from_payloads(
        run=bson_value(run),
        processing_id=processing_id,
        start_month=start_month,
        end_month=end_month,
        observations=observations,
        winner_rows=winner_rows,
        analytics=analytics,
        transition_attribution=attribution,
        risk_search=bson_value(risk),
        confidence=bson_value(confidence),
    )
    db[TEMPORAL_WINNER_TRANSITION_STATEFUL_RESEARCH_COLLECTION].insert_one(dict(result))
    return result


def materialize_winner_transition_stateful_candidate_a_strategy(
    db: Any,
    run_id: str,
    replay_id: str,
    *,
    actor_email: str | None,
) -> dict[str, Any]:
    run = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find_one({"id": str(run_id)})
    if run is None or str(run.get("status") or "").lower() != "completed":
        raise WinnerTransitionStatefulReplayError("A completed Temporal Intelligence run is required.")
    replay = db[TEMPORAL_WINNER_TRANSITION_STATEFUL_RESEARCH_COLLECTION].find_one(
        {"id": str(replay_id), "run_id": str(run_id), "status": "completed"},
        {"_id": 0},
    )
    if replay is None:
        raise WinnerTransitionStatefulReplayError("Completed Stateful transition replay not found.")
    parity = replay.get("control_parity") if isinstance(replay.get("control_parity"), dict) else {}
    if str(parity.get("status") or "").lower() != "passed":
        raise WinnerTransitionStatefulReplayError("Candidate A can create a Strategy only after Control parity passes.")
    candidate = replay.get("candidate_a") if isinstance(replay.get("candidate_a"), dict) else None
    if candidate is None:
        raise WinnerTransitionStatefulReplayError("Candidate A is unavailable in this Stateful replay.")
    if str(candidate.get("mode") or "") != "conservative_one_session":
        raise WinnerTransitionStatefulReplayError("Candidate A does not contain the expected Conservative Stateful policy.")

    from .temporal_intelligence import _temporal_policy_strategy_snapshot

    base_snapshot = _temporal_policy_strategy_snapshot(run)
    candidate_analytics = candidate.get("analytics") if isinstance(candidate.get("analytics"), dict) else {}
    candidate_metrics = candidate_analytics.get("metrics") if isinstance(candidate_analytics.get("metrics"), dict) else {}
    control_replay = replay.get("control_replay") if isinstance(replay.get("control_replay"), dict) else {}
    control_analytics = control_replay.get("analytics") if isinstance(control_replay.get("analytics"), dict) else {}
    control_metrics = control_analytics.get("metrics") if isinstance(control_analytics.get("metrics"), dict) else {}
    control_capital = _finite(control_metrics.get("ending_capital"))
    candidate_capital = _finite(candidate_metrics.get("ending_capital"))
    delta_rate = (
        float(candidate_capital / control_capital - 1.0)
        if candidate_capital is not None and control_capital not in {None, 0.0}
        else None
    )
    research_settings = replay.get("research_settings") if isinstance(replay.get("research_settings"), dict) else {}
    policy_snapshot = {
        **base_snapshot,
        "schema_version": 2,
        "family": "winner_anchored_temporal_stateful",
        "label": "Candidate A — Conservative Stateful",
        "experiment": "winner_transition_stateful_conservative",
        "base_temporal_experiment": base_snapshot.get("experiment"),
        "strategy_variant": "winner_transition_stateful_candidate_a",
        "source_stateful_replay_id": str(replay_id),
        "source_processing_id": str(replay.get("processing_id") or ""),
        "period_start": replay.get("period_start"),
        "period_end": replay.get("period_end"),
        "stateful_policy": {
            "candidate": "a",
            "mode": "conservative_one_session",
            "max_consecutive_defer_sessions": 1,
            "stateful_incumbent": True,
            "control_target_re_evaluated_each_session": True,
            "cash_path_preserved": True,
            "future_control_transition_used_for_rejoin": False,
            "source_risk_search_id": replay.get("source_risk_search_id"),
            "source_confidence_calibration_id": replay.get("source_confidence_calibration_id"),
            "research_settings": bson_value(research_settings),
        },
        "stateful_validation": {
            "control_parity": bson_value(parity),
            "control_metrics": bson_value(control_metrics),
            "candidate_metrics": bson_value(candidate_metrics),
            "candidate_delta_vs_control_rate": delta_rate,
        },
    }

    request_payload = run.get("request") if isinstance(run.get("request"), dict) else {}
    research_model_snapshot = None
    research_model_settings = request_payload.get("research_model_settings") if isinstance(request_payload.get("research_model_settings"), dict) else {}
    model_family = str(run.get("model_family") or request_payload.get("research_model_family") or "")
    if model_family and research_model_settings:
        research_model_snapshot = model_execution_snapshot(model_family, research_model_settings)

    period_end = str(replay.get("period_end") or run.get("analysis_end_date") or "").strip()
    replay_suffix = str(replay_id).split("-")[-1][:8]
    name_suffix = period_end if period_end else str(run_id).split("-")[-1][:8]
    name = f"Candidate A — Conservative Stateful — {name_suffix} — {replay_suffix}"
    description = f"Generated from Stateful transition replay {replay_id} of Temporal Intelligence run {run_id}."
    try:
        materialized = materialize_temporal_stateful_strategy(
            db,
            run_id=str(run_id),
            replay_id=str(replay_id),
            processing_id=str(replay.get("processing_id") or ""),
            candidate_key="a",
            candidate_label="Conservative Stateful",
            source_strategy_id=str(run.get("strategy_profile_id") or ""),
            source_strategy_revision=int(run.get("strategy_profile_revision") or 0) or None,
            source_configuration_hash=str(run.get("strategy_configuration_hash") or "") or None,
            name=name,
            description=description,
            experiment="winner_transition_stateful_conservative",
            policy_snapshot=policy_snapshot,
            actor_email=actor_email,
            research_model_snapshot=research_model_snapshot,
        )
    except (StrategyLabConflict, StrategyLabNotFound, StrategyLabError, ValueError) as exc:
        raise WinnerTransitionStatefulReplayError(str(exc)) from exc

    strategy = materialized["strategy"]
    now = utc_now()
    db[TEMPORAL_WINNER_TRANSITION_STATEFUL_RESEARCH_COLLECTION].update_one(
        {"id": str(replay_id), "run_id": str(run_id)},
        {"$set": {
            "candidate_a_materialized_strategy_id": strategy.get("id"),
            "candidate_a_materialized_strategy_name": strategy.get("name"),
            "candidate_a_materialized_strategy_at": now,
            "updated_at": now,
        }},
    )
    return materialized


def get_latest_winner_transition_stateful_replay(
    db: Any,
    run_id: str,
    *,
    processing_id: str,
    start_month: str,
    end_month: str,
) -> dict[str, Any] | None:
    row = db[TEMPORAL_WINNER_TRANSITION_STATEFUL_RESEARCH_COLLECTION].find_one(
        {"run_id": str(run_id), "processing_id": str(processing_id), "period_start": start_month, "period_end": end_month, "status": {"$in": ["completed", "blocked"]}},
        {"_id": 0}, sort=[("created_at", -1)],
    )
    return bson_value(row) if row is not None else None


def build_stateful_candidate_a_live_decision(
    db: Any,
    *,
    bars_by_symbol: dict[str, pd.DataFrame],
    strategy: Any,
    current_asset: str | None,
    holding_sessions: int,
    winner_profile: dict[str, Any],
    winner_model: dict[str, Any],
    cooldown: bool,
) -> dict[str, Any]:
    policy = (
        winner_profile.get("temporal_policy_snapshot")
        if isinstance(winner_profile.get("temporal_policy_snapshot"), dict)
        else winner_profile.get("temporal_policy")
        if isinstance(winner_profile.get("temporal_policy"), dict)
        else {}
    )
    stateful = policy.get("stateful_policy") if isinstance(policy.get("stateful_policy"), dict) else {}
    bundle = stateful.get("live_runtime") if isinstance(stateful.get("live_runtime"), dict) else None
    if bundle is None:
        raise WinnerTransitionStatefulReplayError("The Stateful Winner does not contain its live runtime bundle.")
    if str(bundle.get("mode") or "") != "conservative_one_session":
        raise WinnerTransitionStatefulReplayError("The installed Stateful runtime supports Candidate A Conservative Stateful only.")

    from ..engine.temporal_intelligence import run_temporal_intelligence
    from ..schemas.requests import BacktestExecutionRequest

    dates = None
    for frame in bars_by_symbol.values():
        index = pd.DatetimeIndex(frame.index)
        dates = index if dates is None else dates.intersection(index)
    if dates is None or len(dates) < 2:
        raise WinnerTransitionStatefulReplayError("Stateful live evaluation requires aligned completed market sessions.")
    latest_date = pd.Timestamp(dates[-1])
    latest_iso = latest_date.date().isoformat()
    execution_request = BacktestExecutionRequest.model_validate({
        **strategy.model_dump(mode="python"),
        "analysis_start_date": strategy.start_date,
        "analysis_end_date": latest_iso,
        "calendar_anchor_assets": list(strategy.assets),
        "research_reference_assets": list(strategy.assets),
        "research_candidate_assets": [],
        "research_model_family": str(winner_model.get("family") or "lightgbm_utility"),
        "research_model_settings": dict(winner_model.get("settings_snapshot") or {}),
        "research_market_data_mode": "database_only",
        "deterministic_execution": True,
        "xgb_n_jobs": 1,
        "numeric_thread_limit": 1,
    })
    temporal = run_temporal_intelligence(bars_by_symbol, execution_request)
    latest_rows = [row for row in temporal.get("multi_horizon_latest_forecasts") or [] if isinstance(row, dict)]
    if not latest_rows:
        raise WinnerTransitionStatefulReplayError("Temporal live evaluation produced no latest forecasts.")
    decision_stamp = max((_timestamp(row.get("as_of")) for row in latest_rows), default=None)
    if decision_stamp is None:
        raise WinnerTransitionStatefulReplayError("Temporal live evaluation did not resolve a decision date.")
    decision_key = decision_stamp.isoformat()
    current_rows = {
        str(row.get("symbol") or "").strip().upper(): row
        for row in latest_rows
        if str(row.get("symbol") or "").strip()
    }
    winner_rows = [row for row in temporal.get("_winner_reference_daily_rows") or [] if isinstance(row, dict)]
    eligible_winner_rows = [
        row for row in winner_rows
        if _timestamp(row.get("decision_date")) is not None and _timestamp(row.get("decision_date")) <= decision_stamp
    ]
    if not eligible_winner_rows:
        raise WinnerTransitionStatefulReplayError("Temporal live evaluation produced no Winner anchor for the latest decision.")
    winner_row = max(eligible_winner_rows, key=lambda row: _timestamp(row.get("decision_date")))
    settings = bundle.get("policy_settings") if isinstance(bundle.get("policy_settings"), dict) else {}
    proposed = _policy_target(current_rows, winner_row, settings)
    control_target = _asset_name(proposed.get("proposed_symbol"))
    incumbent = _asset_name(current_asset)

    next_cooldown = False
    intervention = False
    risk_score = None
    risk_threshold = None
    confidence_margin = None
    confidence_threshold = None
    risk_family = None

    gate_allowed = bool(
        not cooldown
        and incumbent != "CASH"
        and control_target != "CASH"
        and incumbent != control_target
        and not bool(proposed.get("timing_override"))
        and control_target == _asset_name(proposed.get("base_symbol"))
        and control_target == _asset_name(proposed.get("top1_symbol"))
    )
    year = int(decision_stamp.year)
    confidence_by_year = bundle.get("confidence_by_year") if isinstance(bundle.get("confidence_by_year"), dict) else {}
    confidence = confidence_by_year.get(str(year)) if isinstance(confidence_by_year.get(str(year)), dict) else {}
    model_by_year = bundle.get("risk_models") if isinstance(bundle.get("risk_models"), dict) else {}
    model_payload = model_by_year.get(str(year)) if isinstance(model_by_year.get(str(year)), dict) else None
    gate_allowed = bool(gate_allowed and confidence.get("active") and model_payload is not None)

    if gate_allowed:
        observations: dict[str, dict[str, Any]] = {}
        for row in temporal.get("_multi_horizon_observations") or []:
            if not isinstance(row, dict):
                continue
            key = _timestamp_key(row.get("timestamp"))
            symbol = str(row.get("symbol") or "").strip().upper()
            if not key or not symbol:
                continue
            observations.setdefault(key, {"rows_by_symbol": {}})["rows_by_symbol"][symbol] = row
        observations[decision_key] = {"rows_by_symbol": current_rows}
        history_rows = sorted(
            [row for row in eligible_winner_rows if _timestamp(row.get("decision_date")) <= decision_stamp],
            key=lambda row: _timestamp(row.get("decision_date")),
        )[-10:]
        feature_row = _dynamic_transition_features(
            history_rows=history_rows,
            observations=observations,
            target_symbol=control_target,
            incumbent_symbol=incumbent,
        )
        risk_score = _serialized_risk_score(model_payload, feature_row)
        risk_threshold = _finite(model_payload.get("risk_threshold"))
        confidence_threshold = _finite(confidence.get("margin_threshold"))
        if risk_score is not None and risk_threshold is not None:
            confidence_margin = float(risk_score - risk_threshold)
            intervention = bool(
                confidence_threshold is not None
                and confidence_margin >= confidence_threshold
            )
            risk_family = model_payload.get("family")

    target = incumbent if intervention else control_target
    if intervention:
        next_cooldown = True
    elif cooldown:
        next_cooldown = False

    return {
        "decision_date": decision_stamp,
        "current_asset": incumbent,
        "control_target_asset": control_target,
        "target_asset": target,
        "stateful_intervention": intervention,
        "stateful_cooldown_before": bool(cooldown),
        "stateful_cooldown_after": bool(next_cooldown),
        "risk_score": risk_score,
        "risk_threshold": risk_threshold,
        "confidence_margin": confidence_margin,
        "confidence_threshold": confidence_threshold,
        "risk_family": risk_family,
        "temporal_result": temporal,
    }
