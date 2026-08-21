from __future__ import annotations

import json
import re
import zlib
from datetime import datetime, timedelta, timezone
from typing import Any

from ..infrastructure.persistence.mongo_repository import (
    TEMPORAL_INTELLIGENCE_ARTIFACTS_COLLECTION,
    TEMPORAL_INTELLIGENCE_OBSERVATIONS_COLLECTION,
    TEMPORAL_INTELLIGENCE_RUNS_COLLECTION,
    bson_value,
)

_MONTH_PATTERN = re.compile(r"^(\d{4})-(\d{2})$")

_AGGREGATE_FIELDS = (
    "asset_rank_score",
    "entry_rank_score",
    "entry_rank_percentile",
    "opportunity_gate_score",
    "risk_adjusted_entry_score",
    "entry_risk_multiplier",
    "hold_score",
    "incumbent_persistence_raw",
    "incumbent_persistence_score",
    "incumbent_risk_health",
    "short_profit_consensus",
    "short_risk_safety",
    "short_bottom_support",
    "short_horizon_agreement",
    "long_profit_confirmation",
    "long_risk_safety",
    "long_trend_support",
    "long_horizon_agreement",
    "cross_horizon_agreement",
    "horizon_agreement",
    "all_horizon_risk_safety",
    "predicted_drawdown",
    "entry_separation_strength",
    "entry_top_gap_strength",
    "short_profit_quality",
    "quality_history_samples",
)

_HORIZON_FIELDS = (
    "profit_before_loss_probability",
    "profit_percentile",
    "profit_before_loss_quality_weight",
    "predicted_drawdown",
    "risk_safety_percentile",
    "drawdown_quality_weight",
    "bottom_probability",
    "bottom_quality_weight",
    "top_probability",
    "top_quality_weight",
    "trend_direction",
    "trend_persistence_probability",
    "trend_persistence_quality_weight",
)

_DIAGNOSTIC_FIELDS = (
    "quality_source",
    "quality_history_samples",
    "asset_rank_score",
    "opportunity_gate_score",
    "entry_score",
    "risk_adjusted_entry_score",
    "entry_risk_multiplier",
    "risk_entry_threshold_penalty",
    "current_hold_score",
    "incumbent_persistence_score",
    "incumbent_risk_health",
    "cash_score",
    "base_entry_threshold",
    "entry_threshold",
    "active_reentry_margin",
    "exit_threshold",
    "rotation_hurdle",
    "dynamic_rotation_hurdle",
    "rotation_advantage",
    "challenger_confirmation",
    "challenger_confirmation_threshold",
    "severe_risk_exit",
    "risk_break_exit",
    "risk_deterioration_exit",
    "position_age_before",
    "cash_age_before",
    "cash_recovery_mode",
    "defensive_exit",
    "opportunity_exit",
    "probability_profit_before_loss",
    "expected_max_drawdown",
    "short_profit_consensus",
    "short_risk_safety",
    "short_bottom_support",
    "short_horizon_agreement",
    "long_profit_confirmation",
    "long_risk_safety",
    "long_trend_support",
    "long_horizon_agreement",
    "cross_horizon_agreement",
    "horizon_agreement",
    "all_horizon_risk_safety",
    "winner_anchor_symbol",
    "winner_top1_symbol",
    "winner_top2_symbol",
    "winner_top1_score",
    "winner_top2_score",
    "winner_anchor_score",
    "temporal_timing_candidate",
    "temporal_timing_override",
    "winner_anchor_short_profit_consensus",
    "winner_top2_short_profit_consensus",
    "temporal_short_profit_advantage",
    "winner_anchor_risk_safety",
    "winner_top2_risk_safety",
    "winner_anchor_predicted_drawdown",
    "winner_top2_predicted_drawdown",
    "timing_base_weak_threshold",
    "timing_challenger_minimum",
    "timing_minimum_advantage",
    "timing_maximum_advantage",
)


class TemporalDecisionContextNotFound(RuntimeError):
    pass


class TemporalDecisionContextError(RuntimeError):
    pass


def _month_start(value: str) -> datetime:
    match = _MONTH_PATTERN.match(str(value or "").strip())
    if not match:
        raise TemporalDecisionContextError("Period must use YYYY-MM format.")
    year, month = int(match.group(1)), int(match.group(2))
    if month < 1 or month > 12:
        raise TemporalDecisionContextError("Period month must be between 01 and 12.")
    return datetime(year, month, 1, tzinfo=timezone.utc)


def _next_month(value: datetime) -> datetime:
    if value.month == 12:
        return datetime(value.year + 1, 1, 1, tzinfo=timezone.utc)
    return datetime(value.year, value.month + 1, 1, tzinfo=timezone.utc)


def _as_utc(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    return None


def _timestamp_key(value: Any) -> str | None:
    timestamp = _as_utc(value)
    return timestamp.isoformat() if timestamp is not None else None


def _decode_rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    rows = document.get("rows") or []
    if document.get("encoding") == "zlib-json-v1" and document.get("payload"):
        rows = json.loads(zlib.decompress(bytes(document["payload"])).decode("utf-8"))
    return [dict(row) for row in rows if isinstance(row, dict)]


def _artifact_rows(db: Any, run_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cursor = db[TEMPORAL_INTELLIGENCE_ARTIFACTS_COLLECTION].find(
        {"run_id": str(run_id), "kind": "decision_diagnostics"},
        {"_id": 0, "sequence": 1, "encoding": 1, "payload": 1, "rows": 1},
    ).sort("sequence", 1)
    for document in cursor:
        rows.extend(_decode_rows(document))
    return rows


def _observation_rows(
    db: Any,
    run_id: str,
    start: datetime,
    end: datetime,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    cursor = db[TEMPORAL_INTELLIGENCE_OBSERVATIONS_COLLECTION].find(
        {
            "run_id": str(run_id),
            "timestamp": {"$gte": start - timedelta(days=10), "$lt": end},
        },
        {"_id": 0, "timestamp": 1, "encoding": 1, "payload": 1, "rows": 1},
    ).sort("timestamp", 1)
    for document in cursor:
        key = _timestamp_key(document.get("timestamp"))
        if key is None:
            continue
        grouped[key] = _decode_rows(document)
    return grouped


def _number(value: Any) -> float | int | None:
    if value is None or isinstance(value, bool):
        return value
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return value
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
        return None
    return int(parsed) if parsed.is_integer() and isinstance(value, int) else parsed


def _asset_context(row: dict[str, Any] | None, horizons: list[int]) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    aggregate = {field: _number(row.get(field)) for field in _AGGREGATE_FIELDS if field in row}
    horizon_context: dict[str, dict[str, Any]] = {}
    for horizon in horizons:
        values = {
            field: _number(row.get(f"{field}_h{horizon}"))
            for field in _HORIZON_FIELDS
            if f"{field}_h{horizon}" in row
        }
        if values:
            horizon_context[str(horizon)] = values
    return {
        "symbol": str(row.get("symbol") or ""),
        "aggregate": aggregate,
        "horizons": horizon_context,
    }


def _rank_value(row: dict[str, Any]) -> float:
    for field in ("asset_rank_score", "entry_rank_score", "risk_adjusted_entry_score"):
        try:
            value = float(row.get(field))
        except (TypeError, ValueError):
            continue
        if value == value:
            return value
    return float("-inf")


def get_temporal_decision_context(
    db: Any,
    run_id: str,
    *,
    start_month: str,
    end_month: str,
) -> dict[str, Any]:
    run = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find_one(
        {"id": str(run_id)},
        {"_id": 0, "id": 1, "status": 1, "horizons": 1, "result.horizons": 1},
    )
    if run is None:
        raise TemporalDecisionContextNotFound("Temporal Intelligence run not found.")
    if str(run.get("status") or "").lower() != "completed":
        raise TemporalDecisionContextError("Decision context is available only for a completed Temporal Intelligence run.")

    start = _month_start(start_month)
    end_start = _month_start(end_month)
    if end_start < start:
        raise TemporalDecisionContextError("Period end must be greater than or equal to period start.")
    end = _next_month(end_start)

    result = run.get("result") if isinstance(run.get("result"), dict) else {}
    horizons = [int(value) for value in (run.get("horizons") or result.get("horizons") or [])]
    observations = _observation_rows(db, run_id, start, end)
    artifact_rows = _artifact_rows(db, run_id)

    execution_by_decision: dict[str, dict[str, Any]] = {}
    diagnostics: list[dict[str, Any]] = []
    for row in artifact_rows:
        kind = str(row.get("artifact_kind") or "")
        if kind == "multi_horizon_equity_curve":
            key = _timestamp_key(row.get("decision_timestamp"))
            if key:
                execution_by_decision[key] = row
        elif kind == "multi_horizon_decision_diagnostics":
            diagnostics.append(row)

    items: list[dict[str, Any]] = []
    for diagnostic in diagnostics:
        decision_key = _timestamp_key(diagnostic.get("timestamp"))
        if decision_key is None:
            continue
        economic = execution_by_decision.get(decision_key) or {}
        execution_at = _as_utc(economic.get("execution_date"))
        if execution_at is None or execution_at < start or execution_at >= end:
            continue

        rows = observations.get(decision_key) or []
        ranked = sorted(rows, key=_rank_value, reverse=True)
        temporal_rank_top1 = ranked[0] if ranked else None
        temporal_rank_top2 = ranked[1] if len(ranked) > 1 else None
        current_symbol = str(diagnostic.get("current_symbol") or "CASH")
        target_symbol = str(diagnostic.get("target_symbol") or "CASH")
        winner_anchor_symbol = str(diagnostic.get("winner_anchor_symbol") or "")
        winner_top1_symbol = str(diagnostic.get("winner_top1_symbol") or "")
        winner_top2_symbol = str(diagnostic.get("winner_top2_symbol") or "")
        incumbent = next((row for row in rows if str(row.get("symbol") or "") == current_symbol), None)
        target = next((row for row in rows if str(row.get("symbol") or "") == target_symbol), None)
        winner_anchor = next((row for row in rows if str(row.get("symbol") or "") == winner_anchor_symbol), None)
        winner_top1 = next((row for row in rows if str(row.get("symbol") or "") == winner_top1_symbol), None)
        winner_top2 = next((row for row in rows if str(row.get("symbol") or "") == winner_top2_symbol), None)
        top1 = winner_top1 or temporal_rank_top1
        top2 = winner_top2 or temporal_rank_top2

        temporal_top1_score = _rank_value(temporal_rank_top1) if temporal_rank_top1 else None
        temporal_top2_score = _rank_value(temporal_rank_top2) if temporal_rank_top2 else None
        rank_gap = (
            temporal_top1_score - temporal_top2_score
            if temporal_top1_score is not None and temporal_top2_score is not None and temporal_top1_score != float("-inf") and temporal_top2_score != float("-inf")
            else None
        )
        try:
            winner_top1_score = float(diagnostic.get("winner_top1_score"))
            winner_top2_score = float(diagnostic.get("winner_top2_score"))
            winner_score_gap = winner_top1_score - winner_top2_score
        except (TypeError, ValueError):
            winner_score_gap = None

        decision_metrics = {
            field: _number(diagnostic.get(field))
            for field in _DIAGNOSTIC_FIELDS
            if field in diagnostic
        }
        items.append(bson_value({
            "fold_id": diagnostic.get("fold_id"),
            "decision_at": diagnostic.get("timestamp"),
            "execution_at": economic.get("execution_date"),
            "next_execution_at": economic.get("next_execution_date"),
            "current_symbol": current_symbol,
            "target_symbol": target_symbol,
            "action": diagnostic.get("action"),
            "reason": diagnostic.get("reason"),
            "outcome": {
                **{
                    key: _number(economic.get(key))
                    for key in (
                        "gross_interval_return",
                        "net_interval_return",
                        "cost_sides",
                        "one_side_cost_rate",
                        "strategy_equity",
                        "strategy_drawdown",
                        "cumulative_return",
                    )
                    if key in economic
                },
                "counterfactual_current_interval_return": (
                    0.0
                    if current_symbol == "CASH"
                    else _number((incumbent or {}).get("open_to_open_return"))
                ),
            },
            "winner_anchor_symbol": winner_anchor_symbol or None,
            "winner_top1_symbol": winner_top1_symbol or None,
            "winner_top2_symbol": winner_top2_symbol or None,
            "winner_top1_score": _number(diagnostic.get("winner_top1_score")),
            "winner_top2_score": _number(diagnostic.get("winner_top2_score")),
            "winner_top1_top2_score_gap": winner_score_gap,
            "top1_top2_asset_rank_gap": rank_gap,
            "decision_metrics": decision_metrics,
            "winner_anchor": _asset_context(winner_anchor, horizons),
            "top1": _asset_context(top1, horizons),
            "top2": _asset_context(top2, horizons),
            "temporal_rank_top1": _asset_context(temporal_rank_top1, horizons),
            "temporal_rank_top2": _asset_context(temporal_rank_top2, horizons),
            "incumbent": _asset_context(incumbent, horizons),
            "target": _asset_context(target, horizons),
        }))

    return bson_value({
        "schema_version": 1,
        "run_id": str(run_id),
        "period_start": start_month,
        "period_end": end_month,
        "horizons": horizons,
        "count": len(items),
        "items": items,
    })
