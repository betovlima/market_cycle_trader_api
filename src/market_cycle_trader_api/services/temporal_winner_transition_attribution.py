from __future__ import annotations

import json
import math
import zlib
from datetime import datetime, timedelta, timezone
from typing import Any

from ..infrastructure.persistence.mongo_repository import (
    TEMPORAL_INTELLIGENCE_ARTIFACTS_COLLECTION,
    TEMPORAL_INTELLIGENCE_OBSERVATIONS_COLLECTION,
    TEMPORAL_INTELLIGENCE_RUNS_COLLECTION,
    bson_value,
)
from .temporal_decision_context import (
    TemporalDecisionContextError,
    TemporalDecisionContextNotFound,
    get_temporal_decision_context,
)

_LOOKBACK_WINDOWS = (1, 3, 5, 10)
_TEMPORAL_FIELDS = (
    "entry_rank_score",
    "entry_rank_percentile",
    "opportunity_gate_score",
    "risk_adjusted_entry_score",
    "hold_score",
    "incumbent_persistence_score",
    "incumbent_risk_health",
    "short_profit_consensus",
    "short_risk_safety",
    "long_profit_confirmation",
    "long_risk_safety",
    "long_trend_support",
    "cross_horizon_agreement",
    "horizon_agreement",
    "all_horizon_risk_safety",
    "predicted_drawdown",
)


def _as_utc(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _timestamp_key(value: Any) -> str | None:
    timestamp = _as_utc(value)
    return timestamp.isoformat() if timestamp is not None else None


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _decode_rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    rows = document.get("rows") or []
    if document.get("encoding") == "zlib-json-v1" and document.get("payload"):
        rows = json.loads(zlib.decompress(bytes(document["payload"])).decode("utf-8"))
    return [dict(row) for row in rows if isinstance(row, dict)]


def _artifact_rows(db: Any, run_id: str, kind: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cursor = db[TEMPORAL_INTELLIGENCE_ARTIFACTS_COLLECTION].find(
        {"run_id": str(run_id), "kind": str(kind)},
        {"_id": 0, "sequence": 1, "encoding": 1, "payload": 1, "rows": 1},
    ).sort("sequence", 1)
    for document in cursor:
        rows.extend(_decode_rows(document))
    return rows


def _observation_rows(db: Any, run_id: str, start: datetime, end: datetime) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    cursor = db[TEMPORAL_INTELLIGENCE_OBSERVATIONS_COLLECTION].find(
        {
            "run_id": str(run_id),
            "timestamp": {"$gte": start - timedelta(days=45), "$lt": end},
        },
        {"_id": 0, "timestamp": 1, "encoding": 1, "payload": 1, "rows": 1},
    ).sort("timestamp", 1)
    for document in cursor:
        key = _timestamp_key(document.get("timestamp"))
        if key:
            grouped[key] = _decode_rows(document)
    return grouped


def _asset(row: dict[str, Any], field: str) -> str:
    return str(row.get(field) or "").strip().upper()


def _winner_score(row: dict[str, Any], symbol: str) -> float | None:
    symbol = str(symbol or "").upper()
    pairs = (
        ("top_1_asset", "top_1_score"),
        ("top_2_asset", "top_2_score"),
        ("top_3_asset", "top_3_score"),
        ("current_asset", "current_score"),
        ("raw_best_asset", "raw_best_score"),
        ("best_asset", "best_score"),
        ("final_action_asset", "final_action_score"),
        ("selected_asset", "decision_score"),
    )
    for asset_field, score_field in pairs:
        if _asset(row, asset_field) == symbol:
            value = _finite(row.get(score_field))
            if value is not None:
                return value
    return None


def _winner_rank(row: dict[str, Any], symbol: str) -> int | None:
    symbol = str(symbol or "").upper()
    for rank, field in ((1, "top_1_asset"), (2, "top_2_asset"), (3, "top_3_asset")):
        if _asset(row, field) == symbol:
            return rank
    if _asset(row, "current_asset") == symbol:
        value = _finite(row.get("current_asset_rank"))
        return int(value) if value is not None and value >= 1 else None
    if _asset(row, "raw_best_asset") == symbol or _asset(row, "best_asset") == symbol:
        return 1
    return None


def _temporal_row(rows: list[dict[str, Any]], symbol: str) -> dict[str, Any] | None:
    symbol = str(symbol or "").upper()
    return next((row for row in rows if str(row.get("symbol") or "").upper() == symbol), None)


def _temporal_snapshot(row: dict[str, Any] | None) -> dict[str, float | None]:
    if not isinstance(row, dict):
        return {field: None for field in _TEMPORAL_FIELDS}
    return {field: _finite(row.get(field)) for field in _TEMPORAL_FIELDS}


def _difference(target: dict[str, float | None], incumbent: dict[str, float | None]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for field in _TEMPORAL_FIELDS:
        left = target.get(field)
        right = incumbent.get(field)
        result[field] = float(left - right) if left is not None and right is not None else None
    return result


def _mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(sum(finite) / len(finite)) if finite else None


def _delta(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(finite[-1] - finite[0]) if len(finite) >= 2 else None


def _rate(values: list[bool]) -> float | None:
    return float(sum(1 for value in values if value) / len(values)) if values else None


def _leader_changes(sessions: list[dict[str, Any]]) -> int:
    leaders = [str(item.get("top1_asset") or "") for item in sessions if item.get("top1_asset")]
    return int(sum(1 for previous, current in zip(leaders, leaders[1:]) if previous != current))


def _consecutive_target_top1(sessions: list[dict[str, Any]], target_symbol: str) -> int:
    count = 0
    for item in reversed(sessions):
        if str(item.get("top1_asset") or "") != target_symbol:
            break
        count += 1
    return count


def _window_summary(sessions: list[dict[str, Any]], target_symbol: str, incumbent_symbol: str) -> dict[str, Any]:
    if not sessions:
        return {"sessions": 0}
    latest = sessions[-1]
    target_scores = [item.get("target_score") for item in sessions]
    incumbent_scores = [item.get("incumbent_score") for item in sessions]
    score_gaps = [item.get("target_minus_incumbent_score") for item in sessions]
    target_ranks = [item.get("target_rank") for item in sessions]
    incumbent_ranks = [item.get("incumbent_rank") for item in sessions]
    temporal_gap_fields = (
        "short_profit_consensus",
        "all_horizon_risk_safety",
        "predicted_drawdown",
        "horizon_agreement",
        "long_profit_confirmation",
        "long_trend_support",
    )
    temporal = {}
    for field in temporal_gap_fields:
        values = [((item.get("temporal") or {}).get("target_minus_incumbent") or {}).get(field) for item in sessions]
        temporal[field] = {
            "latest": values[-1] if values else None,
            "mean": _mean(values),
            "delta": _delta(values),
        }
    return {
        "sessions": len(sessions),
        "leader_change_count": _leader_changes(sessions),
        "target_top1_rate": _rate([item.get("top1_asset") == target_symbol for item in sessions]),
        "incumbent_top1_rate": _rate([item.get("top1_asset") == incumbent_symbol for item in sessions]),
        "target_top3_rate": _rate([item.get("target_rank") in {1, 2, 3} for item in sessions]),
        "target_top1_consecutive": _consecutive_target_top1(sessions, target_symbol),
        "target_rank_mean": _mean([float(value) if value is not None else None for value in target_ranks]),
        "incumbent_rank_mean": _mean([float(value) if value is not None else None for value in incumbent_ranks]),
        "target_score_latest": latest.get("target_score"),
        "target_score_delta": _delta(target_scores),
        "incumbent_score_latest": latest.get("incumbent_score"),
        "incumbent_score_delta": _delta(incumbent_scores),
        "target_minus_incumbent_score_latest": latest.get("target_minus_incumbent_score"),
        "target_minus_incumbent_score_mean": _mean(score_gaps),
        "target_minus_incumbent_score_delta": _delta(score_gaps),
        "top1_top2_gap_latest": latest.get("top1_top2_gap"),
        "top1_top2_gap_mean": _mean([item.get("top1_top2_gap") for item in sessions]),
        "universe_score_std_latest": latest.get("universe_score_std"),
        "universe_score_std_mean": _mean([item.get("universe_score_std") for item in sessions]),
        "temporal_target_minus_incumbent": temporal,
    }


def _holding_interval_outcome(
    contexts: list[dict[str, Any]],
    start_index: int,
    observations: dict[str, list[dict[str, Any]]],
    *,
    fold_id: Any,
    incumbent_symbol: str,
    target_symbol: str,
) -> dict[str, Any]:
    target_factor = 1.0
    incumbent_factor = 1.0
    holding_sessions = 0
    exit_execution_at = None

    for index in range(start_index, len(contexts)):
        context = contexts[index]
        if index > start_index:
            current_fold = context.get("fold_id")
            if fold_id is not None and current_fold is not None:
                try:
                    if int(current_fold) != int(fold_id):
                        break
                except (TypeError, ValueError):
                    pass
            current_symbol = str(context.get("current_symbol") or "").upper()
            next_symbol = str(context.get("target_symbol") or "").upper()
            action = str(context.get("action") or "").upper()
            if current_symbol == target_symbol and action in {"ROTATE", "SELL"} and next_symbol != target_symbol:
                exit_execution_at = context.get("execution_at")
                break
            if current_symbol not in {"", target_symbol}:
                break

        decision_key = _timestamp_key(context.get("decision_at"))
        rows = observations.get(decision_key or "") or []
        target_row = _temporal_row(rows, target_symbol)
        incumbent_row = _temporal_row(rows, incumbent_symbol)
        target_return = _finite((target_row or {}).get("open_to_open_return"))
        incumbent_return = _finite((incumbent_row or {}).get("open_to_open_return"))
        if target_return is None or incumbent_return is None:
            return {
                "complete": False,
                "holding_sessions": holding_sessions,
                "exit_execution_at": exit_execution_at,
                "target_return": None,
                "incumbent_return": None,
                "value_added": None,
            }
        target_factor *= max(1e-9, 1.0 + target_return)
        incumbent_factor *= max(1e-9, 1.0 + incumbent_return)
        holding_sessions += 1

    if exit_execution_at is None or holding_sessions <= 0:
        return {
            "complete": False,
            "holding_sessions": holding_sessions,
            "exit_execution_at": exit_execution_at,
            "target_return": None,
            "incumbent_return": None,
            "value_added": None,
        }

    target_return = float(target_factor - 1.0)
    incumbent_return = float(incumbent_factor - 1.0)
    return {
        "complete": True,
        "holding_sessions": holding_sessions,
        "exit_execution_at": exit_execution_at,
        "target_return": target_return,
        "incumbent_return": incumbent_return,
        "value_added": float(target_return - incumbent_return),
    }


def _reference_context(row: dict[str, Any]) -> dict[str, Any] | None:
    decision_at = _as_utc(row.get("decision_date"))
    execution_at = _as_utc(row.get("timestamp") or row.get("execution_date"))
    if decision_at is None or execution_at is None:
        return None
    current_symbol = _asset(row, "current_asset") or _asset(row, "previous_asset") or "CASH"
    target_symbol = _asset(row, "selected_asset") or _asset(row, "final_action_asset") or current_symbol
    if current_symbol == target_symbol:
        action = "HOLD"
    elif current_symbol == "CASH" and target_symbol != "CASH":
        action = "BUY"
    elif current_symbol != "CASH" and target_symbol == "CASH":
        action = "SELL"
    else:
        action = "ROTATE"
    return {
        "fold_id": row.get("walk_forward_fold"),
        "decision_at": decision_at,
        "execution_at": execution_at,
        "current_symbol": current_symbol,
        "target_symbol": target_symbol,
        "action": action,
        "reason": str(row.get("trade_reason") or "strategy_research_reference"),
    }


def _reference_contexts(
    winner_rows: list[dict[str, Any]],
    *,
    start: datetime,
    end: datetime,
) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    for row in winner_rows:
        context = _reference_context(row)
        if context is None:
            continue
        execution_at = _as_utc(context.get("execution_at"))
        if execution_at is None or execution_at < start or execution_at >= end:
            continue
        contexts.append(context)
    contexts.sort(key=lambda item: _as_utc(item.get("decision_at")) or datetime.min.replace(tzinfo=timezone.utc))
    return contexts


def _period_bounds(start_month: str, end_month: str) -> tuple[datetime, datetime]:
    try:
        start = datetime.strptime(str(start_month), "%Y-%m").replace(tzinfo=timezone.utc)
        end_start = datetime.strptime(str(end_month), "%Y-%m").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise TemporalDecisionContextError("Period must use YYYY-MM format.") from exc
    if end_start < start:
        raise TemporalDecisionContextError("Period end must be greater than or equal to period start.")
    if end_start.month == 12:
        end = datetime(end_start.year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end = datetime(end_start.year, end_start.month + 1, 1, tzinfo=timezone.utc)
    return start, end


def get_winner_transition_attribution(
    db: Any,
    run_id: str,
    *,
    start_month: str,
    end_month: str,
) -> dict[str, Any]:
    run = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find_one(
        {"id": str(run_id)},
        {"_id": 0, "id": 1, "status": 1},
    )
    if run is None:
        raise TemporalDecisionContextNotFound("Temporal Intelligence run not found.")
    if str(run.get("status") or "").lower() != "completed":
        raise TemporalDecisionContextError("Research transition attribution is available only for a completed Temporal Intelligence run.")

    start, end = _period_bounds(start_month, end_month)
    winner_rows = _artifact_rows(db, run_id, "winner_reference_daily")
    winner_rows = [row for row in winner_rows if _as_utc(row.get("decision_date")) is not None]
    winner_rows.sort(key=lambda row: _as_utc(row.get("decision_date")) or datetime.min.replace(tzinfo=timezone.utc))
    contexts = _reference_contexts(winner_rows, start=start, end=end)

    if not contexts:
        decision_context = get_temporal_decision_context(
            db,
            run_id,
            start_month=start_month,
            end_month=end_month,
        )
        contexts = [item for item in decision_context.get("items") or [] if isinstance(item, dict)]

    decision_times = [_as_utc(item.get("decision_at")) for item in contexts]
    decision_times = [value for value in decision_times if value is not None]
    if not decision_times:
        return bson_value({
            "schema_version": 2,
            "run_id": str(run_id),
            "period_start": start_month,
            "period_end": end_month,
            "source": "strategy_research_reference",
            "lookback_windows": list(_LOOKBACK_WINDOWS),
            "count": 0,
            "items": [],
        })
    observations = _observation_rows(db, run_id, min(decision_times), max(decision_times) + timedelta(days=2))

    items: list[dict[str, Any]] = []
    for context_index, context in enumerate(contexts):
        if str(context.get("action") or "").upper() != "ROTATE":
            continue
        incumbent_symbol = str(context.get("current_symbol") or "").upper()
        target_symbol = str(context.get("target_symbol") or "").upper()
        if incumbent_symbol in {"", "CASH"} or target_symbol in {"", "CASH"} or incumbent_symbol == target_symbol:
            continue
        decision_at = _as_utc(context.get("decision_at"))
        if decision_at is None:
            continue
        fold_id = context.get("fold_id")

        eligible: list[dict[str, Any]] = []
        for row in winner_rows:
            row_time = _as_utc(row.get("decision_date"))
            if row_time is None or row_time > decision_at:
                continue
            row_fold = row.get("walk_forward_fold")
            if fold_id is not None and row_fold is not None:
                try:
                    if int(row_fold) != int(fold_id):
                        continue
                except (TypeError, ValueError):
                    pass
            eligible.append(row)
        history_rows = eligible[-max(_LOOKBACK_WINDOWS):]
        sessions: list[dict[str, Any]] = []
        for row in history_rows:
            row_time = _as_utc(row.get("decision_date"))
            if row_time is None:
                continue
            target_score = _winner_score(row, target_symbol)
            incumbent_score = _winner_score(row, incumbent_symbol)
            top1_score = _finite(row.get("top_1_score"))
            top2_score = _finite(row.get("top_2_score"))
            rows = observations.get(row_time.isoformat()) or []
            target_temporal = _temporal_snapshot(_temporal_row(rows, target_symbol))
            incumbent_temporal = _temporal_snapshot(_temporal_row(rows, incumbent_symbol))
            sessions.append({
                "decision_at": row_time,
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

        windows = {
            str(window): _window_summary(sessions[-window:], target_symbol, incumbent_symbol)
            for window in _LOOKBACK_WINDOWS
        }
        decision_rows = observations.get(decision_at.isoformat()) or []
        target_row = _temporal_row(decision_rows, target_symbol)
        incumbent_row = _temporal_row(decision_rows, incumbent_symbol)
        target_interval = _finite((target_row or {}).get("open_to_open_return"))
        incumbent_interval = _finite((incumbent_row or {}).get("open_to_open_return"))
        holding_interval = _holding_interval_outcome(
            contexts,
            context_index,
            observations,
            fold_id=fold_id,
            incumbent_symbol=incumbent_symbol,
            target_symbol=target_symbol,
        )
        items.append(bson_value({
            "fold_id": fold_id,
            "decision_at": context.get("decision_at"),
            "execution_at": context.get("execution_at"),
            "from_asset": incumbent_symbol,
            "to_asset": target_symbol,
            "reason": str(context.get("reason") or "strategy_research_reference"),
            "winner_top1_top2_score_gap": (
                windows.get("1", {}).get("top1_top2_gap_latest")
                if isinstance(windows.get("1"), dict)
                else None
            ),
            "one_interval_outcome": {
                "target_return": target_interval,
                "incumbent_return": incumbent_interval,
                "value_added": (
                    float(target_interval - incumbent_interval)
                    if target_interval is not None and incumbent_interval is not None
                    else None
                ),
            },
            "holding_interval_outcome": holding_interval,
            "trajectory": {
                "sessions": sessions,
                "windows": windows,
            },
        }))

    return bson_value({
        "schema_version": 2,
        "run_id": str(run_id),
        "period_start": start_month,
        "period_end": end_month,
        "source": "strategy_research_reference",
        "lookback_windows": list(_LOOKBACK_WINDOWS),
        "count": len(items),
        "items": items,
    })
