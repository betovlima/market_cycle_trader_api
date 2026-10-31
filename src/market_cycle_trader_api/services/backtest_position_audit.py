"""Read-only audit of actual positions saved in PORTFOLIO backtest predictions.

A recorded action or a rotation is NOT evidence of the effective position.
In particular, the final liquidation can leave selected_asset stale while
cash_weight=1 and assets_held=0. Never infer missing positions from trades.
"""
from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite
from typing import Any


def _instant(value: Any) -> str | None:
    if isinstance(value, datetime):
        moment = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc).isoformat()
    if not value:
        return None
    try:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError):
        return None


def _day(value: Any) -> str | None:
    instant = _instant(value)
    return instant[:10] if instant else None


def _number(value: Any) -> float | None:
    try:
        result = float(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None
    return result if result is not None and isfinite(result) else None


def _recorded_position(row: dict[str, Any]) -> str | None:
    symbol = str(row.get("selected_asset") or "").strip().upper()
    weight = _number(row.get("cash_weight"))
    held = _number(row.get("assets_held"))
    # Portfolio weights and share counts are recorded AFTER execution.
    # An end-of-backtest FINAL_SELL can leave selected_asset unchanged.
    if weight is not None and weight >= 1 - 1e-9 and (held is None or held == 0):
        return "CASH"
    if held == 0 and symbol == "CASH":
        return "CASH"
    if symbol == "CASH" and held is None and weight is None:
        return "CASH"
    if symbol and symbol != "CASH" and (held is None or held > 0) and (weight is None or weight < 1 - 1e-9):
        return symbol
    return None


def build_position_audit(
    equity: list[dict[str, Any]],
    persisted_predictions: list[dict[str, Any]],
    decision_scores: list[dict[str, Any]],
    rotations: list[dict[str, Any]],
    trades: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Join only sessions that really exist; never forward-fill a missing position."""
    # A matching calendar date alone is insufficient when multiple snapshots
    # happen on the same day; join on the exact persisted execution timestamp.
    by_instant: dict[str, dict[str, Any]] = {}
    for row in persisted_predictions:
        instant = _instant(row.get("timestamp"))
        if instant:
            by_instant[instant] = row

    scores = {date: row for row in decision_scores if (date := _day(row.get("execution_date")))}
    moves: dict[str, list[dict[str, Any]]] = {}
    for row in rotations:
        date = _day(row.get("executed_at"))
        if date:
            moves.setdefault(date, []).append(row)

    terminal_sells = {
        instant
        for row in (trades or [])
        if str(row.get("action") or "").upper() == "FINAL_SELL"
        if (instant := _instant(row.get("timestamp")))
    }
    history: list[dict[str, Any]] = []
    for equity_row in equity:
        date = _day(equity_row.get("timestamp"))
        if date is None:
            continue
        raw = by_instant.get(_instant(equity_row.get("timestamp")))
        position = _recorded_position(raw) if raw is not None else None
        action = str(raw.get("trade_action") or "").upper() if raw else ""
        history.append({
            "timestamp": equity_row["timestamp"],
            "asset": position,
            "cash_weight": _number(raw.get("cash_weight")) if raw else None,
            "assets_held": int(raw["assets_held"]) if raw and _number(raw.get("assets_held")) is not None else None,
            "trade_action": action or None,
            "terminal_liquidation": "FINAL_SELL" in action or _instant(equity_row.get("timestamp")) in terminal_sells,
            "recorded": position is not None,
        })

    episodes: list[dict[str, Any]] = []
    for index, item in enumerate(history):
        # Final sale only makes the account CASH at the terminal close;
        # it does not mean capital stayed in CASH during that session.
        if item["asset"] != "CASH" or item["terminal_liquidation"]:
            continue
        if episodes and index == episodes[-1]["_end_index"] + 1:
            episode = episodes[-1]
            episode["_end_index"] = index
            episode["last_recorded_at"] = item["timestamp"]
            episode["sessions"] += 1
            continue
        episodes.append({
            "_start_index": index, "_end_index": index,
            "start_at": item["timestamp"],
            "last_recorded_at": item["timestamp"],
            "sessions": 1,
        })

    detailed = []
    for episode in episodes:
        start_idx, end_idx = episode["_start_index"], episode["_end_index"]
        start_date = _day(history[start_idx]["timestamp"])
        next_idx = end_idx + 1
        next_date = _day(history[next_idx]["timestamp"]) if next_idx < len(history) else None
        previous = history[start_idx - 1]["asset"] if start_idx else None
        following = history[next_idx]["asset"] if next_date else None
        decision = scores.get(start_date) or {}
        entry = next((row for row in moves.get(start_date, []) if str(row.get("to_asset") or "").upper() == "CASH"), None)
        exit_move = next((row for row in moves.get(next_date, []) if str(row.get("from_asset") or "").upper() == "CASH"), None)
        detailed.append({
            "start_at": episode["start_at"],
            "last_recorded_at": episode["last_recorded_at"],
            "sessions": episode["sessions"],
            "previous_asset": previous,
            "next_asset": following,
            "next_session_at": history[next_idx]["timestamp"] if next_date else None,
            "entry_rotation": entry,
            "exit_rotation": exit_move,
            "decision_date": decision.get("decision_date"),
            "decision_reason": decision.get("decision_reason"),
            "best_asset": decision.get("best_asset"),
            "best_score": decision.get("best_score"),
            "current_score": decision.get("current_score"),
            "finite_score_count": decision.get("finite_score_count"),
            "cash_threshold_guard_applied": decision.get("cash_threshold_guard_applied") if decision else None,
            "minimum_expected_edge_guard_applied": decision.get("minimum_expected_edge_guard_applied") if decision else None,
            "terminal_liquidation": history[start_idx]["terminal_liquidation"],
        })

    total = len(history)
    recorded = sum(1 for row in history if row["recorded"])
    return {
        "position_history": history,
        "position_coverage": {
            "total_sessions": total,
            "recorded_sessions": recorded,
            "missing_sessions": total - recorded,
            "complete": recorded == total,
            "cash_sessions": sum(row["asset"] == "CASH" and not row["terminal_liquidation"] for row in history),
            "terminal_liquidations": sum(row["asset"] == "CASH" and row["terminal_liquidation"] for row in history),
            "cash_episodes": [
                {key: ep[key] for key in ("start_at", "last_recorded_at", "sessions")}
                for ep in episodes
            ],
            "position_source": "persisted_execution_prediction" if recorded else "unavailable",
        },
        "cash_investigation": detailed,
    }
