from __future__ import annotations

from collections import defaultdict
import json
import zlib
from datetime import datetime, timedelta, timezone
from statistics import fmean, median
from typing import Any, Iterable

from fastapi import HTTPException

from ..infrastructure.persistence.mongo_repository import (
    ALPACA_MARKET_BARS_COLLECTION,
    COMPARISONS_COLLECTION,
    JOBS_COLLECTION,
    MARKET_BARS_COLLECTION,
    MODEL_TUNING_VALIDATIONS_COLLECTION,
    PAPER_PORTFOLIO_SNAPSHOTS_COLLECTION,
    PAPER_TRADE_ORDERS_COLLECTION,
    PREDICTIONS_COLLECTION,
    RUNS_COLLECTION,
    STRATEGY_PROFILES_COLLECTION,
    TEMPORAL_INTELLIGENCE_ARTIFACTS_COLLECTION,
    TEMPORAL_INTELLIGENCE_RUNS_COLLECTION,
    TEMPORAL_WINNER_TRANSITION_STATEFUL_RESEARCH_COLLECTION,
    TRADES_COLLECTION,
    bson_value,
    get_alpaca_integration_status,
)
from .admin_rotations import admin_job_rotations
from .dashboard import _public_metrics, _selected_internal_row
from .serialization import iso_value


_STATEFUL_STRATEGY_PROCESSING_PREFIX = "strategy-stateful:"
_TEMPORAL_STRATEGY_PROCESSING_PREFIX = "strategy-temporal:"


def temporal_strategy_processing_id(strategy_id: str) -> str:
    normalized = str(strategy_id or "").strip()
    if not normalized:
        raise ValueError("Temporal Strategy id is required.")
    return f"{_TEMPORAL_STRATEGY_PROCESSING_PREFIX}{normalized}"


def _temporal_strategy_id_from_processing(processing_id: str) -> str | None:
    value = str(processing_id or "").strip()
    if not value.startswith(_TEMPORAL_STRATEGY_PROCESSING_PREFIX):
        return None
    strategy_id = value[len(_TEMPORAL_STRATEGY_PROCESSING_PREFIX):].strip()
    return strategy_id or None


def _decode_temporal_artifact_rows(db: Any, run_id: str, artifact_kind: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cursor = db[TEMPORAL_INTELLIGENCE_ARTIFACTS_COLLECTION].find(
        {"run_id": str(run_id), "kind": "decision_diagnostics"},
        {"_id": 0, "sequence": 1, "encoding": 1, "payload": 1, "rows": 1},
    ).sort("sequence", 1)
    for document in cursor:
        payload = document.get("rows") or []
        if document.get("encoding") == "zlib-json-v1" and document.get("payload"):
            payload = json.loads(zlib.decompress(bytes(document["payload"])).decode("utf-8"))
        for row in payload:
            if isinstance(row, dict) and str(row.get("artifact_kind") or "") == str(artifact_kind):
                rows.append(dict(row))
    return rows


def _temporal_strategy_processing_analytics(db: Any, processing_id: str) -> dict[str, Any] | None:
    strategy_id = _temporal_strategy_id_from_processing(processing_id)
    if not strategy_id:
        return None
    profile = db[STRATEGY_PROFILES_COLLECTION].find_one(
        {"_id": strategy_id},
        {
            "_id": 1, "name": 1, "revision": 1, "configuration_hash": 1,
            "strategy_kind": 1, "tuning_target": 1, "temporal_strategy_variant": 1,
            "source_temporal_run_id": 1, "temporal_policy_snapshot": 1,
        },
    )
    if profile is None:
        raise HTTPException(status_code=404, detail="Strategy Research Temporal processing source was not found.")
    if str(profile.get("strategy_kind") or "") != "temporal_intelligence":
        raise HTTPException(status_code=409, detail="The selected Strategy Research profile is not a Temporal Strategy.")
    is_stateful = (
        str(profile.get("temporal_strategy_variant") or "") == "winner_transition_stateful"
        and str(profile.get("tuning_target") or "") == "stateful_transition"
    )
    if is_stateful:
        raise HTTPException(status_code=409, detail="Stateful Temporal Strategies use the Stateful processing source.")
    policy = profile.get("temporal_policy_snapshot") if isinstance(profile.get("temporal_policy_snapshot"), dict) else {}
    run_id = str(profile.get("source_temporal_run_id") or policy.get("source_run_id") or "").strip()
    if not run_id:
        raise HTTPException(status_code=409, detail="The selected Temporal Strategy does not contain its source run binding.")
    run = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find_one(
        {"id": run_id, "status": "completed"},
        {"_id": 0, "id": 1, "result": 1, "created_at": 1, "finished_at": 1},
    )
    if run is None:
        raise HTTPException(status_code=409, detail="The selected Temporal Strategy source run is unavailable.")
    result = run.get("result") if isinstance(run.get("result"), dict) else {}
    multi = result.get("multi_horizon_metrics") if isinstance(result.get("multi_horizon_metrics"), dict) else {}
    capital = multi.get("shadow_capital") if isinstance(multi.get("shadow_capital"), dict) else {}
    curve = _decode_temporal_artifact_rows(db, run_id, "multi_horizon_equity_curve")
    curve.sort(key=lambda row: str(row.get("execution_date") or row.get("decision_timestamp") or ""))
    if not curve:
        raise HTTPException(status_code=409, detail="The selected Temporal Strategy source run does not contain its exact economic replay curve.")

    initial_capital = _as_float(capital.get("initial_capital")) or 0.0
    equity: list[dict[str, Any]] = []
    rotations: list[dict[str, Any]] = []
    last_rotation_index: int | None = None
    for row in curve:
        timestamp = row.get("execution_date") or row.get("timestamp")
        value = _as_float(row.get("strategy_equity"))
        if timestamp is None or value is None:
            continue
        target = str(row.get("target_symbol") or "CASH").upper() or "CASH"
        current = str(row.get("current_symbol") or "CASH").upper() or "CASH"
        action = str(row.get("action") or "HOLD").upper()
        equity.append({
            "timestamp": iso_value(timestamp),
            "simulation_equity": value,
            "reference_equity": value,
            "starting_value": initial_capital or None,
            "drawdown": _as_float(row.get("strategy_drawdown")),
            "selected_asset": target,
            "trade_action": action,
            "temporal_timing_override": bool(row.get("temporal_timing_override")),
        })
        if current != target:
            rotation = {
                "sequence": len(rotations) + 1,
                "executed_at": iso_value(timestamp),
                "from_asset": current,
                "to_asset": target,
                "holding_days": None,
                "position_return": None,
                "realized_pnl": None,
                "transaction_fees": 0.0,
                "sell_reason": str(row.get("reason") or "temporal_strategy_replay"),
                "buy_reason": str(row.get("reason") or "temporal_strategy_replay"),
                "temporal_timing_override": bool(row.get("temporal_timing_override")),
            }
            if last_rotation_index is not None:
                previous = rotations[last_rotation_index]
                previous["holding_days"] = _duration_days(previous.get("executed_at"), timestamp)
            rotations.append(rotation)
            last_rotation_index = len(rotations) - 1

    monthly = _monthly_returns(equity)
    worst_month = min(
        (row for row in monthly if _as_float(row.get("return")) is not None),
        key=lambda row: float(row["return"]),
        default=None,
    )
    metrics = {
        "initial_capital": _as_float(capital.get("initial_capital")),
        "ending_capital": _as_float(capital.get("ending_capital")),
        "strategy_return": _as_float(capital.get("total_return")),
        "cagr": _as_float(capital.get("cagr")),
        "sharpe": _as_float(capital.get("sharpe")),
        "maximum_drawdown": _as_float(capital.get("max_drawdown")),
        "capital_rotations": int(capital.get("switch_count") or len(rotations)),
        "market_exposure": _as_float(capital.get("exposure")),
        "cash_days": int(capital.get("cash_days") or 0),
        "timing_override_count": int(capital.get("timing_override_count") or 0),
        "monthly_returns": monthly,
        "worst_month": worst_month,
    }
    payload = {
        "job_id": str(processing_id),
        "processing_id": str(processing_id),
        "processing_kind": "strategy_research_temporal",
        "processing_label": "Strategy Research · Temporal",
        "reference_label": "Source Temporal Strategy",
        "created_at": iso_value(run.get("created_at")),
        "finished_at": iso_value(run.get("finished_at")),
        "strategy_profile_id": strategy_id,
        "strategy_profile_name": str(profile.get("name") or "Temporal Strategy"),
        "strategy_profile_revision": int(profile.get("revision") or 1),
        "strategy_configuration_hash": str(profile.get("configuration_hash") or ""),
        "source_temporal_run_id": run_id,
        "metrics": metrics,
        "equity": equity,
        "monthly_returns": monthly,
        "consistency": _consistency(monthly),
        "drawdown_episodes": _drawdown_episodes(equity),
        "rotation_summary": {"rotation_count": len(rotations)},
        "asset_attribution": [],
        "transition_matrix": _transition_matrix(rotations),
        "holding_buckets": [],
        "trade_dependency": _trade_dependency([]),
        "rotations": rotations,
    }
    return bson_value(payload)


def stateful_strategy_processing_id(strategy_id: str) -> str:
    normalized = str(strategy_id or "").strip()
    if not normalized:
        raise ValueError("Stateful Strategy id is required.")
    return f"{_STATEFUL_STRATEGY_PROCESSING_PREFIX}{normalized}"


def _stateful_strategy_id_from_processing(processing_id: str) -> str | None:
    value = str(processing_id or "").strip()
    if not value.startswith(_STATEFUL_STRATEGY_PROCESSING_PREFIX):
        return None
    strategy_id = value[len(_STATEFUL_STRATEGY_PROCESSING_PREFIX):].strip()
    return strategy_id or None


def _stateful_strategy_processing_analytics(db: Any, processing_id: str) -> dict[str, Any] | None:
    strategy_id = _stateful_strategy_id_from_processing(processing_id)
    if not strategy_id:
        return None
    profile = db[STRATEGY_PROFILES_COLLECTION].find_one({"_id": strategy_id}, {"_id": 1, "name": 1, "revision": 1, "configuration_hash": 1, "strategy_kind": 1, "tuning_target": 1, "temporal_strategy_variant": 1, "source_stateful_replay_id": 1, "stateful_candidate_key": 1})
    if profile is None:
        raise HTTPException(status_code=404, detail="Strategy Research Stateful processing source was not found.")
    if (
        str(profile.get("strategy_kind") or "") != "temporal_intelligence"
        or str(profile.get("temporal_strategy_variant") or "") != "winner_transition_stateful"
        or str(profile.get("tuning_target") or "") != "stateful_transition"
    ):
        raise HTTPException(status_code=409, detail="The selected Strategy Research profile is not a Stateful Temporal Strategy.")
    replay_id = str(profile.get("source_stateful_replay_id") or "").strip()
    candidate_key = str(profile.get("stateful_candidate_key") or "a").strip().lower() or "a"
    if not replay_id:
        raise HTTPException(status_code=409, detail="The selected Stateful Strategy does not contain its source replay binding.")
    replay = db[TEMPORAL_WINNER_TRANSITION_STATEFUL_RESEARCH_COLLECTION].find_one({"id": replay_id, "status": "completed"}, {"_id": 0})
    if replay is None:
        raise HTTPException(status_code=409, detail="The selected Stateful Strategy source replay is unavailable.")
    candidate = replay.get(f"candidate_{candidate_key}") if isinstance(replay.get(f"candidate_{candidate_key}"), dict) else None
    analytics = candidate.get("analytics") if isinstance(candidate, dict) and isinstance(candidate.get("analytics"), dict) else None
    if analytics is None:
        raise HTTPException(status_code=409, detail="The selected Stateful Strategy does not contain validated replay analytics.")
    payload = dict(bson_value(analytics))
    payload.update({
        "job_id": str(processing_id),
        "processing_id": str(processing_id),
        "processing_kind": "strategy_research_stateful",
        "processing_label": "Strategy Research · Stateful",
        "reference_label": "Source Control",
        "strategy_profile_id": strategy_id,
        "strategy_profile_name": str(profile.get("name") or "Stateful Strategy"),
        "strategy_profile_revision": int(profile.get("revision") or 1),
        "strategy_configuration_hash": str(profile.get("configuration_hash") or ""),
        "source_stateful_replay_id": replay_id,
    })
    return payload


_FORBIDDEN_OUTPUT_KEYS = frozenset(
    {
        "backend",
        "random_seed",
        "effective_config",
        "strategy_configuration_sha256",
        "q_current_position",
        "q_raw_best",
        "q_final_action",
        "q_delta_final_vs_current",
        "q_gap_best_vs_second",
        "decision_score",
        "model_probability",
        "model_probabilities",
        "top_features",
    }
)


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def _as_utc(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _safe_divide(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _selected_backend(db: Any, job_id: str) -> str | None:
    comparison = db[COMPARISONS_COLLECTION].find_one(
        {"job_id": job_id},
        {"_id": 0, "results": 1},
    )
    selected = _selected_internal_row(comparison)
    backend = str((selected or {}).get("backend") or "").strip()
    if backend:
        return backend
    run = db[RUNS_COLLECTION].find_one(
        {"job_id": job_id, "symbol": "PORTFOLIO"},
        {"_id": 0, "backend": 1},
    )
    fallback = str((run or {}).get("backend") or "").strip()
    return fallback or None


def _sorted_rows(rows: Iterable[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: _as_utc(row.get(key)) or datetime.min.replace(tzinfo=timezone.utc),
    )


def _equity_rows(db: Any, job_id: str, backend: str) -> list[dict[str, Any]]:
    rows = db[PREDICTIONS_COLLECTION].find(
        {"job_id": job_id, "symbol": "PORTFOLIO", "backend": backend},
        {
            "_id": 0,
            "timestamp": 1,
            "strategy_equity": 1,
            "buy_hold_equity": 1,
        },
    )
    output: list[dict[str, Any]] = []
    peak: float | None = None
    for row in _sorted_rows(rows, "timestamp"):
        strategy = _as_float(row.get("strategy_equity"))
        reference = _as_float(row.get("buy_hold_equity"))
        if strategy is None:
            continue
        peak = strategy if peak is None else max(peak, strategy)
        drawdown = strategy / peak - 1.0 if peak else 0.0
        output.append(
            {
                "timestamp": iso_value(row.get("timestamp")),
                "simulation_equity": strategy,
                "reference_equity": reference,
                "drawdown": drawdown,
            }
        )
    return output


def _trade_rows(db: Any, job_id: str, backend: str) -> list[dict[str, Any]]:
    rows = db[TRADES_COLLECTION].find(
        {"job_id": job_id, "symbol": "PORTFOLIO", "backend": backend},
        {
            "_id": 0,
            "timestamp": 1,
            "sequence": 1,
            "action": 1,
            "asset": 1,
            "holding_bars": 1,
            "position_return": 1,
            "realized_pnl": 1,
            "total_fee": 1,
        },
    )
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            _as_utc(row.get("timestamp")) or datetime.min.replace(tzinfo=timezone.utc),
            int(row.get("sequence") or 0),
        ),
    )


def _completed_sells(trades: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in trades
        if str(row.get("action") or "").upper() in {"SELL", "FINAL_SELL"}
    ]


def _asset_attribution(sells: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sells:
        groups[str(row.get("asset") or "UNKNOWN")].append(row)

    result: list[dict[str, Any]] = []
    for asset, rows in groups.items():
        pnl_values = [value for row in rows if (value := _as_float(row.get("realized_pnl"))) is not None]
        returns = [value for row in rows if (value := _as_float(row.get("position_return"))) is not None]
        holdings = [value for row in rows if (value := _as_float(row.get("holding_bars"))) is not None]
        fees = [value for row in rows if (value := _as_float(row.get("total_fee"))) is not None]
        total_pnl = float(sum(pnl_values)) if pnl_values else 0.0
        result.append(
            {
                "asset": asset,
                "closed_positions": len(rows),
                "profitable_positions": sum(value > 0 for value in pnl_values),
                "losing_positions": sum(value < 0 for value in pnl_values),
                "win_rate": _safe_divide(sum(value > 0 for value in pnl_values), len(pnl_values)) if pnl_values else None,
                "total_realized_pnl": total_pnl,
                "average_position_return": float(fmean(returns)) if returns else None,
                "median_position_return": float(median(returns)) if returns else None,
                "average_holding_days": float(fmean(holdings)) if holdings else None,
                "transaction_fees": float(sum(fees)) if fees else 0.0,
            }
        )
    result.sort(key=lambda item: float(item["total_realized_pnl"]), reverse=True)
    total_positive = sum(max(0.0, float(item["total_realized_pnl"])) for item in result)
    for item in result:
        item["positive_profit_share"] = (
            max(0.0, float(item["total_realized_pnl"])) / total_positive
            if total_positive > 0
            else None
        )
    return result


def _transition_matrix(rotations: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rotations:
        groups[(str(row.get("from_asset") or "CASH"), str(row.get("to_asset") or "CASH"))].append(row)

    result: list[dict[str, Any]] = []
    for (source, destination), rows in groups.items():
        pnl = [value for row in rows if (value := _as_float(row.get("realized_pnl"))) is not None]
        returns = [value for row in rows if (value := _as_float(row.get("position_return"))) is not None]
        fees = [value for row in rows if (value := _as_float(row.get("transaction_fees"))) is not None]
        result.append(
            {
                "from_asset": source,
                "to_asset": destination,
                "rotations": len(rows),
                "profitable_rotations": sum(value > 0 for value in pnl),
                "win_rate": _safe_divide(sum(value > 0 for value in pnl), len(pnl)) if pnl else None,
                "total_realized_pnl": float(sum(pnl)) if pnl else 0.0,
                "average_position_return": float(fmean(returns)) if returns else None,
                "transaction_fees": float(sum(fees)) if fees else 0.0,
            }
        )
    result.sort(key=lambda item: (-int(item["rotations"]), str(item["from_asset"]), str(item["to_asset"])))
    return result


def _holding_buckets(sells: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    definitions = [
        ("1–5 days", 0, 5),
        ("6–15 days", 6, 15),
        ("16–30 days", 16, 30),
        ("31+ days", 31, None),
    ]
    rows = list(sells)
    result: list[dict[str, Any]] = []
    for label, minimum, maximum in definitions:
        selected = []
        for row in rows:
            holding = _as_float(row.get("holding_bars"))
            if holding is None:
                continue
            if holding < minimum or (maximum is not None and holding > maximum):
                continue
            selected.append(row)
        pnl = [value for row in selected if (value := _as_float(row.get("realized_pnl"))) is not None]
        returns = [value for row in selected if (value := _as_float(row.get("position_return"))) is not None]
        fees = [value for row in selected if (value := _as_float(row.get("total_fee"))) is not None]
        result.append(
            {
                "bucket": label,
                "positions": len(selected),
                "win_rate": _safe_divide(sum(value > 0 for value in pnl), len(pnl)) if pnl else None,
                "average_position_return": float(fmean(returns)) if returns else None,
                "total_realized_pnl": float(sum(pnl)) if pnl else 0.0,
                "transaction_fees": float(sum(fees)) if fees else 0.0,
            }
        )
    return result


def _monthly_returns(equity: list[dict[str, Any]]) -> list[dict[str, Any]]:
    monthly: dict[str, dict[str, Any]] = {}
    for row in equity:
        timestamp = row.get("timestamp")
        if not timestamp:
            continue
        try:
            month = datetime.fromisoformat(str(timestamp)).strftime("%Y-%m")
        except ValueError:
            continue
        monthly[month] = row
    months = sorted(monthly)
    result: list[dict[str, Any]] = []
    previous_simulation: float | None = None
    previous_reference: float | None = None
    for month in months:
        row = monthly[month]
        simulation = _as_float(row.get("simulation_equity"))
        reference = _as_float(row.get("reference_equity"))
        simulation_return = (
            simulation / previous_simulation - 1.0
            if simulation is not None and previous_simulation not in {None, 0}
            else None
        )
        reference_return = (
            reference / previous_reference - 1.0
            if reference is not None and previous_reference not in {None, 0}
            else None
        )
        result.append(
            {
                "month": month,
                "simulation_return": simulation_return,
                "reference_return": reference_return,
                "excess_return": (
                    simulation_return - reference_return
                    if simulation_return is not None and reference_return is not None
                    else None
                ),
            }
        )
        previous_simulation = simulation
        previous_reference = reference
    return result[1:] if len(result) > 1 else []


def _drawdown_episodes(equity: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    peak_value: float | None = None
    peak_at: str | None = None

    for row in equity:
        value = _as_float(row.get("simulation_equity"))
        timestamp = str(row.get("timestamp") or "")
        if value is None:
            continue
        if peak_value is None or value >= peak_value:
            if active is not None:
                active["recovered_at"] = timestamp
                active["duration_days"] = _duration_days(active.get("started_at"), timestamp)
                episodes.append(active)
                active = None
            peak_value = value
            peak_at = timestamp
            continue
        drawdown = value / peak_value - 1.0 if peak_value else 0.0
        if active is None:
            active = {
                "started_at": peak_at,
                "trough_at": timestamp,
                "recovered_at": None,
                "maximum_drawdown": drawdown,
                "duration_days": None,
            }
        elif drawdown < float(active["maximum_drawdown"]):
            active["maximum_drawdown"] = drawdown
            active["trough_at"] = timestamp

    if active is not None:
        active["duration_days"] = _duration_days(active.get("started_at"), equity[-1].get("timestamp") if equity else None)
        episodes.append(active)

    episodes.sort(key=lambda item: float(item.get("maximum_drawdown") or 0.0))
    return episodes[:limit]


def _duration_days(start: Any, end: Any) -> int | None:
    try:
        start_dt = datetime.fromisoformat(str(start))
        end_dt = datetime.fromisoformat(str(end))
    except (TypeError, ValueError):
        return None
    return max(0, int((end_dt - start_dt).total_seconds() // 86400))


def _consistency(monthly: list[dict[str, Any]]) -> dict[str, Any]:
    returns = [value for row in monthly if (value := _as_float(row.get("simulation_return"))) is not None]
    excess = [value for row in monthly if (value := _as_float(row.get("excess_return"))) is not None]
    longest_positive = 0
    longest_negative = 0
    current_positive = 0
    current_negative = 0
    for value in returns:
        if value > 0:
            current_positive += 1
            current_negative = 0
        elif value < 0:
            current_negative += 1
            current_positive = 0
        else:
            current_positive = current_negative = 0
        longest_positive = max(longest_positive, current_positive)
        longest_negative = max(longest_negative, current_negative)
    return {
        "months": len(returns),
        "positive_months": sum(value > 0 for value in returns),
        "negative_months": sum(value < 0 for value in returns),
        "positive_month_rate": _safe_divide(sum(value > 0 for value in returns), len(returns)) if returns else None,
        "median_monthly_return": float(median(returns)) if returns else None,
        "average_monthly_return": float(fmean(returns)) if returns else None,
        "average_monthly_excess_return": float(fmean(excess)) if excess else None,
        "longest_positive_streak": longest_positive,
        "longest_negative_streak": longest_negative,
    }


def _trade_dependency(sells: list[dict[str, Any]]) -> dict[str, Any]:
    pnl = sorted(
        [value for row in sells if (value := _as_float(row.get("realized_pnl"))) is not None],
        reverse=True,
    )
    total = float(sum(pnl)) if pnl else 0.0

    def without(count: int) -> float:
        return float(sum(pnl[count:])) if pnl else 0.0

    return {
        "closed_positions": len(pnl),
        "total_realized_pnl": total,
        "best_position_pnl": pnl[0] if pnl else None,
        "without_best_position_pnl": without(1),
        "without_top_three_pnl": without(3),
        "without_top_five_pnl": without(5),
        "top_five_profit_share": _safe_divide(sum(max(0.0, value) for value in pnl[:5]), sum(max(0.0, value) for value in pnl)) if pnl and sum(max(0.0, value) for value in pnl) > 0 else None,
    }


def backtest_analytics(db: Any, job_id: str) -> dict[str, Any]:
    job = db[JOBS_COLLECTION].find_one(
        {"id": job_id},
        {"_id": 0, "id": 1, "status": 1, "created_at": 1, "finished_at": 1},
    )
    if job is None:
        raise HTTPException(status_code=404, detail="Backtest job not found.")
    if str(job.get("status") or "").lower() != "completed":
        raise HTTPException(status_code=409, detail="Analytics are available after the backtest completes.")

    backend = _selected_backend(db, job_id)
    comparison = db[COMPARISONS_COLLECTION].find_one(
        {"job_id": job_id},
        {"_id": 0, "results": 1},
    )
    metrics = _public_metrics(comparison) or {}
    if not backend:
        return {
            "job_id": job_id,
            "created_at": iso_value(job.get("created_at")),
            "finished_at": iso_value(job.get("finished_at")),
            "metrics": metrics,
            "rotation_summary": {},
            "equity": [],
            "monthly_returns": [],
            "consistency": _consistency([]),
            "drawdown_episodes": [],
            "asset_attribution": [],
            "transition_matrix": [],
            "holding_buckets": [],
            "trade_dependency": _trade_dependency([]),
            "rotations": [],
        }

    equity = _equity_rows(db, job_id, backend)
    trades = _trade_rows(db, job_id, backend)
    sells = _completed_sells(trades)
    rotation_payload = admin_job_rotations(db, job_id)
    rotations = list(rotation_payload.get("rotations", []))
    monthly = _monthly_returns(equity)
    payload = {
        "job_id": job_id,
        "created_at": iso_value(job.get("created_at")),
        "finished_at": iso_value(job.get("finished_at")),
        "metrics": metrics,
        "rotation_summary": rotation_payload.get("summary", {}),
        "equity": equity,
        "monthly_returns": monthly,
        "consistency": _consistency(monthly),
        "drawdown_episodes": _drawdown_episodes(equity),
        "asset_attribution": _asset_attribution(sells),
        "transition_matrix": _transition_matrix(rotations),
        "holding_buckets": _holding_buckets(sells),
        "trade_dependency": _trade_dependency(sells),
        "rotations": rotations,
    }
    _assert_strategy_neutral(payload)
    return payload



def _month_bounds(year: int, month: int) -> tuple[datetime, datetime]:
    start = datetime(int(year), int(month), 1, tzinfo=timezone.utc)
    if int(month) == 12:
        end = datetime(int(year) + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end = datetime(int(year), int(month) + 1, 1, tzinfo=timezone.utc)
    return start, end


def _parse_utc(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return _as_utc(value)
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return _as_utc(parsed)


def _market_price_rows(
    db: Any,
    symbol: str,
    start: datetime,
    end: datetime,
) -> list[dict[str, Any]]:
    timestamp_filter = {"$gte": start, "$lt": end}
    projection = {
        "_id": 0,
        "timestamp": 1,
        "open": 1,
        "high": 1,
        "low": 1,
        "close": 1,
        "updated_at": 1,
    }
    sources = (
        (
            ALPACA_MARKET_BARS_COLLECTION,
            {"symbol": symbol, "interval": "1Day", "timestamp": timestamp_filter},
        ),
        (
            MARKET_BARS_COLLECTION,
            {"symbol": symbol, "interval": "1d", "timestamp": timestamp_filter},
        ),
    )
    rows: list[dict[str, Any]] = []
    for collection_name, query in sources:
        rows = list(db[collection_name].find(query, projection))
        if rows:
            break

    latest_by_timestamp: dict[datetime, dict[str, Any]] = {}
    rows.sort(
        key=lambda row: (
            _as_utc(row.get("timestamp")) or datetime.min.replace(tzinfo=timezone.utc),
            _as_utc(row.get("updated_at")) or datetime.min.replace(tzinfo=timezone.utc),
        )
    )
    for row in rows:
        timestamp = _as_utc(row.get("timestamp"))
        close = _as_float(row.get("close"))
        if timestamp is None or close is None or not (start <= timestamp < end):
            continue
        latest_by_timestamp[timestamp] = {
            "timestamp": iso_value(timestamp),
            "open": _as_float(row.get("open")),
            "high": _as_float(row.get("high")),
            "low": _as_float(row.get("low")),
            "close": close,
        }
    return [latest_by_timestamp[key] for key in sorted(latest_by_timestamp)]


def _rotation_period_equity_rows(
    db: Any,
    job_id: str,
    backend: str,
    start: datetime,
    end: datetime,
) -> list[dict[str, Any]]:
    rows = list(
        db[PREDICTIONS_COLLECTION].find(
            {
                "job_id": job_id,
                "symbol": "PORTFOLIO",
                "backend": backend,
                "timestamp": {"$gte": start, "$lt": end},
            },
            {"_id": 0, "timestamp": 1, "strategy_equity": 1},
        )
    )
    output: list[dict[str, Any]] = []
    for row in _sorted_rows(rows, "timestamp"):
        timestamp = _as_utc(row.get("timestamp"))
        value = _as_float(row.get("strategy_equity"))
        if timestamp is None or value is None:
            continue
        output.append({"timestamp": iso_value(timestamp), "value": value})
    return output


def rotation_period_analysis(
    db: Any,
    job_id: str,
    *,
    year: int,
    month: int,
) -> dict[str, Any]:
    if month < 1 or month > 12:
        raise HTTPException(status_code=422, detail="Month must be between 1 and 12.")

    job = db[JOBS_COLLECTION].find_one(
        {"id": job_id},
        {"_id": 0, "id": 1, "status": 1},
    )
    if job is None:
        raise HTTPException(status_code=404, detail="Backtest job not found.")
    if str(job.get("status") or "").lower() != "completed":
        raise HTTPException(status_code=409, detail="Analytics are available after the backtest completes.")

    backend = _selected_backend(db, job_id)
    start, end = _month_bounds(year, month)
    if not backend:
        return {
            "job_id": job_id,
            "year": year,
            "month": month,
            "period_start": iso_value(start),
            "period_end": iso_value(end),
            "assets": [],
            "movements": [],
            "position_segments": [],
            "strategy_equity": [],
            "default_asset": None,
        }

    rotation_payload = admin_job_rotations(db, job_id)
    timed_movements: list[tuple[datetime, dict[str, Any]]] = []
    current_asset = "CASH"
    for raw in rotation_payload.get("rotations", []):
        timestamp = _parse_utc(raw.get("executed_at"))
        if timestamp is None:
            continue
        movement = dict(raw)
        timed_movements.append((timestamp, movement))
    timed_movements.sort(key=lambda item: item[0])

    for timestamp, movement in timed_movements:
        if timestamp >= start:
            break
        current_asset = str(movement.get("to_asset") or "CASH").upper()

    monthly_movements = [
        movement
        for timestamp, movement in timed_movements
        if start <= timestamp < end
    ]
    monthly_timed = [
        (timestamp, movement)
        for timestamp, movement in timed_movements
        if start <= timestamp < end
    ]
    if (
        current_asset == "CASH"
        and monthly_movements
        and str(monthly_movements[0].get("from_asset") or "CASH").upper() != "CASH"
    ):
        current_asset = str(monthly_movements[0].get("from_asset") or "CASH").upper()

    strategy_equity = _rotation_period_equity_rows(db, job_id, backend, start, end)
    effective_start = _parse_utc(strategy_equity[0].get("timestamp")) if strategy_equity else start
    effective_end = end
    if strategy_equity:
        last_equity_at = _parse_utc(strategy_equity[-1].get("timestamp"))
        if last_equity_at is not None:
            effective_end = min(end, last_equity_at + timedelta(days=1))
    effective_start = effective_start or start

    position_segments: list[dict[str, Any]] = []
    segment_asset = current_asset
    segment_start = effective_start
    for timestamp, movement in monthly_timed:
        if timestamp < effective_start:
            segment_asset = str(movement.get("to_asset") or "CASH").upper()
            continue
        segment_end = min(timestamp, effective_end)
        if segment_end > segment_start:
            position_segments.append(
                {
                    "asset": segment_asset,
                    "start_at": iso_value(segment_start),
                    "end_at": iso_value(segment_end),
                }
            )
        segment_asset = str(movement.get("to_asset") or "CASH").upper()
        segment_start = max(timestamp, effective_start)
        if segment_start >= effective_end:
            break
    if effective_end > segment_start:
        position_segments.append(
            {
                "asset": segment_asset,
                "start_at": iso_value(segment_start),
                "end_at": iso_value(effective_end),
            }
        )

    asset_symbols = {
        asset
        for segment in position_segments
        if (asset := str(segment.get("asset") or "CASH").upper()) != "CASH"
    }
    for movement in monthly_movements:
        for field in ("from_asset", "to_asset"):
            asset = str(movement.get(field) or "CASH").upper()
            if asset != "CASH":
                asset_symbols.add(asset)

    strategy_sessions = [
        _parse_utc(row.get("timestamp"))
        for row in strategy_equity
        if _parse_utc(row.get("timestamp")) is not None
    ]
    assets: list[dict[str, Any]] = []
    for symbol in sorted(asset_symbols):
        prices = _market_price_rows(db, symbol, start, end)
        buys = [m for m in monthly_movements if str(m.get("to_asset") or "CASH").upper() == symbol]
        sells = [m for m in monthly_movements if str(m.get("from_asset") or "CASH").upper() == symbol]
        realized = [
            value
            for movement in sells
            if (value := _as_float(movement.get("realized_pnl"))) is not None
        ]
        first_close = _as_float(prices[0].get("close")) if prices else None
        last_close = _as_float(prices[-1].get("close")) if prices else None
        held_sessions = 0
        for session_at in strategy_sessions:
            if session_at is None:
                continue
            if any(
                str(segment.get("asset") or "CASH").upper() == symbol
                and (_parse_utc(segment.get("start_at")) or start) <= session_at < (_parse_utc(segment.get("end_at")) or end)
                for segment in position_segments
            ):
                held_sessions += 1
        assets.append(
            {
                "symbol": symbol,
                "prices": prices,
                "first_close": first_close,
                "last_close": last_close,
                "period_return": (last_close / first_close - 1.0) if first_close not in {None, 0} and last_close is not None else None,
                "buy_count": len(buys),
                "sell_count": len(sells),
                "realized_pnl": float(sum(realized)) if realized else 0.0,
                "held_sessions": held_sessions,
            }
        )

    assets.sort(
        key=lambda item: (
            -(int(item.get("buy_count") or 0) + int(item.get("sell_count") or 0)),
            -int(item.get("held_sessions") or 0),
            str(item.get("symbol") or ""),
        )
    )
    strategy_return = None
    if len(strategy_equity) >= 2:
        first_equity = _as_float(strategy_equity[0].get("value"))
        last_equity = _as_float(strategy_equity[-1].get("value"))
        if first_equity not in {None, 0} and last_equity is not None:
            strategy_return = last_equity / first_equity - 1.0

    return {
        "job_id": job_id,
        "year": year,
        "month": month,
        "period_start": iso_value(start),
        "period_end": iso_value(effective_end),
        "assets": assets,
        "movements": monthly_movements,
        "position_segments": position_segments,
        "strategy_equity": strategy_equity,
        "strategy_return": strategy_return,
        "default_asset": assets[0]["symbol"] if assets else None,
    }



def _rotation_summary_from_rows(rotations: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = [dict(row) for row in rotations]
    pnl_values = [value for row in rows if (value := _as_float(row.get("realized_pnl"))) is not None]
    fees = [value for row in rows if (value := _as_float(row.get("transaction_fees"))) is not None]
    holdings = [value for row in rows if (value := _as_float(row.get("holding_days"))) is not None]
    asset_to_asset = sum(
        1 for row in rows
        if str(row.get("from_asset") or "CASH").upper() != "CASH"
        and str(row.get("to_asset") or "CASH").upper() != "CASH"
    )
    market_to_cash = sum(
        1 for row in rows
        if str(row.get("from_asset") or "CASH").upper() != "CASH"
        and str(row.get("to_asset") or "CASH").upper() == "CASH"
    )
    cash_to_market = sum(
        1 for row in rows
        if str(row.get("from_asset") or "CASH").upper() == "CASH"
        and str(row.get("to_asset") or "CASH").upper() != "CASH"
    )
    return {
        "total_rotations": len(rows),
        "asset_to_asset_rotations": asset_to_asset,
        "market_to_cash_rotations": market_to_cash,
        "cash_to_market_rotations": cash_to_market,
        "profitable_rotations": sum(value > 0 for value in pnl_values),
        "losing_rotations": sum(value < 0 for value in pnl_values),
        "flat_rotations": sum(value == 0 for value in pnl_values),
        "total_realized_pnl": float(sum(pnl_values)) if pnl_values else 0.0,
        "total_transaction_fees": float(sum(fees)) if fees else 0.0,
        "average_holding_days": float(fmean(holdings)) if holdings else None,
    }


def analytics_from_equity_rotations(
    *,
    processing_id: str,
    equity: list[dict[str, Any]],
    rotations: list[dict[str, Any]],
    metrics: dict[str, Any],
    created_at: Any = None,
    finished_at: Any = None,
    processing_kind: str = "research_validation",
    processing_label: str | None = None,
    reference_label: str = "Reference",
) -> dict[str, Any]:
    clean_equity = [dict(row) for row in equity]
    clean_rotations = [dict(row) for row in rotations]
    monthly = _monthly_returns(clean_equity)
    payload = {
        "job_id": processing_id,
        "processing_id": processing_id,
        "processing_kind": processing_kind,
        "processing_label": processing_label or processing_id,
        "reference_label": reference_label,
        "created_at": iso_value(created_at),
        "finished_at": iso_value(finished_at),
        "metrics": dict(metrics),
        "rotation_summary": _rotation_summary_from_rows(clean_rotations),
        "equity": clean_equity,
        "monthly_returns": monthly,
        "consistency": _consistency(monthly),
        "drawdown_episodes": _drawdown_episodes(clean_equity),
        "asset_attribution": [],
        "transition_matrix": _transition_matrix(clean_rotations),
        "holding_buckets": [],
        "trade_dependency": {},
        "rotations": clean_rotations,
    }
    _assert_strategy_neutral(payload)
    return bson_value(payload)


def _rotation_period_from_data(
    db: Any,
    processing_id: str,
    *,
    equity: list[dict[str, Any]],
    rotations: list[dict[str, Any]],
    year: int,
    month: int,
) -> dict[str, Any]:
    if month < 1 or month > 12:
        raise HTTPException(status_code=422, detail="Month must be between 1 and 12.")
    start, end = _month_bounds(year, month)
    timed_movements: list[tuple[datetime, dict[str, Any]]] = []
    current_asset = "CASH"
    for raw in rotations:
        timestamp = _parse_utc(raw.get("executed_at"))
        if timestamp is None:
            continue
        timed_movements.append((timestamp, dict(raw)))
    timed_movements.sort(key=lambda item: item[0])

    for timestamp, movement in timed_movements:
        if timestamp >= start:
            break
        current_asset = str(movement.get("to_asset") or "CASH").upper()

    monthly_movements = [movement for timestamp, movement in timed_movements if start <= timestamp < end]
    monthly_timed = [(timestamp, movement) for timestamp, movement in timed_movements if start <= timestamp < end]
    if current_asset == "CASH" and monthly_movements and str(monthly_movements[0].get("from_asset") or "CASH").upper() != "CASH":
        current_asset = str(monthly_movements[0].get("from_asset") or "CASH").upper()

    strategy_equity = []
    for row in equity:
        timestamp = _parse_utc(row.get("timestamp"))
        value = _as_float(row.get("simulation_equity"))
        if timestamp is None or value is None or not (start <= timestamp < end):
            continue
        strategy_equity.append({"timestamp": iso_value(timestamp), "value": value})
    strategy_equity.sort(key=lambda row: _parse_utc(row.get("timestamp")) or start)

    effective_start = _parse_utc(strategy_equity[0].get("timestamp")) if strategy_equity else start
    effective_end = end
    if strategy_equity:
        last_equity_at = _parse_utc(strategy_equity[-1].get("timestamp"))
        if last_equity_at is not None:
            effective_end = min(end, last_equity_at + timedelta(days=1))
    effective_start = effective_start or start

    position_segments: list[dict[str, Any]] = []
    segment_asset = current_asset
    segment_start = effective_start
    for timestamp, movement in monthly_timed:
        if timestamp < effective_start:
            segment_asset = str(movement.get("to_asset") or "CASH").upper()
            continue
        segment_end = min(timestamp, effective_end)
        if segment_end > segment_start:
            position_segments.append({"asset": segment_asset, "start_at": iso_value(segment_start), "end_at": iso_value(segment_end)})
        segment_asset = str(movement.get("to_asset") or "CASH").upper()
        segment_start = max(timestamp, effective_start)
        if segment_start >= effective_end:
            break
    if effective_end > segment_start:
        position_segments.append({"asset": segment_asset, "start_at": iso_value(segment_start), "end_at": iso_value(effective_end)})

    asset_symbols = {
        str(segment.get("asset") or "CASH").upper()
        for segment in position_segments
        if str(segment.get("asset") or "CASH").upper() != "CASH"
    }
    for movement in monthly_movements:
        for field in ("from_asset", "to_asset"):
            asset = str(movement.get(field) or "CASH").upper()
            if asset != "CASH":
                asset_symbols.add(asset)

    strategy_sessions = [_parse_utc(row.get("timestamp")) for row in strategy_equity]
    assets: list[dict[str, Any]] = []
    for symbol in sorted(asset_symbols):
        prices = _market_price_rows(db, symbol, start, end)
        buys = [m for m in monthly_movements if str(m.get("to_asset") or "CASH").upper() == symbol]
        sells = [m for m in monthly_movements if str(m.get("from_asset") or "CASH").upper() == symbol]
        realized = [value for movement in sells if (value := _as_float(movement.get("realized_pnl"))) is not None]
        first_close = _as_float(prices[0].get("close")) if prices else None
        last_close = _as_float(prices[-1].get("close")) if prices else None
        held_sessions = 0
        for session_at in strategy_sessions:
            if session_at is None:
                continue
            if any(
                str(segment.get("asset") or "CASH").upper() == symbol
                and (_parse_utc(segment.get("start_at")) or start) <= session_at < (_parse_utc(segment.get("end_at")) or end)
                for segment in position_segments
            ):
                held_sessions += 1
        assets.append({
            "symbol": symbol,
            "prices": prices,
            "first_close": first_close,
            "last_close": last_close,
            "period_return": (last_close / first_close - 1.0) if first_close not in {None, 0} and last_close is not None else None,
            "buy_count": len(buys),
            "sell_count": len(sells),
            "realized_pnl": float(sum(realized)) if realized else 0.0,
            "held_sessions": held_sessions,
        })
    assets.sort(key=lambda item: (-(int(item.get("buy_count") or 0) + int(item.get("sell_count") or 0)), -int(item.get("held_sessions") or 0), str(item.get("symbol") or "")))
    strategy_return = None
    if len(strategy_equity) >= 2:
        first_equity = _as_float(strategy_equity[0].get("value"))
        last_equity = _as_float(strategy_equity[-1].get("value"))
        if first_equity not in {None, 0} and last_equity is not None:
            strategy_return = last_equity / first_equity - 1.0
    return {
        "job_id": processing_id,
        "processing_id": processing_id,
        "year": year,
        "month": month,
        "period_start": iso_value(start),
        "period_end": iso_value(effective_end),
        "assets": assets,
        "movements": monthly_movements,
        "position_segments": position_segments,
        "strategy_equity": strategy_equity,
        "strategy_return": strategy_return,
        "default_asset": assets[0]["symbol"] if assets else None,
    }


def completed_processings(db: Any, limit: int = 100) -> dict[str, Any]:
    safe_limit = max(1, min(500, int(limit)))
    backtests = completed_backtests(db, limit=safe_limit).get("items", [])
    items = [
        {**row, "processing_kind": "backtest", "processing_label": "Backtest"}
        for row in backtests
    ]
    validation_rows = list(
        db[MODEL_TUNING_VALIDATIONS_COLLECTION].find(
            {"status": "completed"},
            {"_id": 0, "id": 1, "created_at": 1, "finished_at": 1, "status": 1, "candidate_id": 1, "strategy_profile_name": 1, "tuning_scope": 1,
             "validation_fold_count": 1, "validation_passed": 1, "certification_processing_id": 1,
             "certification_fold_count": 1, "certification_passed": 1, "certification_completed_at": 1},
        )
    )
    for row in validation_rows:
        candidate_id = row.get("candidate_id")
        validation_folds = row.get("validation_fold_count")
        items.append({
            "id": str(row.get("id") or ""),
            "status": "completed",
            "created_at": iso_value(row.get("created_at")),
            "finished_at": iso_value(row.get("finished_at")),
            "processing_kind": "caro_validation",
            "processing_label": f"CARO Finalist #{candidate_id} · Validation" + (f" · {validation_folds} folds" if validation_folds else ""),
            "strategy_profile_name": row.get("strategy_profile_name"),
            "candidate_id": candidate_id,
            "tuning_scope": row.get("tuning_scope"),
            "fold_count": validation_folds,
            "passed": row.get("validation_passed"),
        })
        certification_id = str(row.get("certification_processing_id") or "").strip()
        if certification_id:
            certification_folds = row.get("certification_fold_count")
            items.append({
                "id": certification_id,
                "status": "completed",
                "created_at": iso_value(row.get("certification_completed_at") or row.get("finished_at")),
                "finished_at": iso_value(row.get("certification_completed_at") or row.get("finished_at")),
                "processing_kind": "caro_certification",
                "processing_label": f"CARO Candidate #{candidate_id} · Certification" + (f" · {certification_folds} folds" if certification_folds else ""),
                "strategy_profile_name": row.get("strategy_profile_name"),
                "candidate_id": candidate_id,
                "tuning_scope": row.get("tuning_scope"),
                "fold_count": certification_folds,
                "passed": row.get("certification_passed"),
            })
    items.sort(key=lambda row: _parse_utc(row.get("finished_at") or row.get("created_at")) or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return {"items": items[:safe_limit]}


def processing_analytics(db: Any, processing_id: str) -> dict[str, Any]:
    from ..milp_decision.processing import processing_analytics as milp_processing_analytics
    milp = milp_processing_analytics(db, processing_id)
    if milp is not None:
        return milp
    temporal = _temporal_strategy_processing_analytics(db, processing_id)
    if temporal is not None:
        return temporal
    stateful = _stateful_strategy_processing_analytics(db, processing_id)
    if stateful is not None:
        return stateful
    certification = db[MODEL_TUNING_VALIDATIONS_COLLECTION].find_one(
        {"certification_processing_id": str(processing_id)}, {"_id": 0, "certification_analytics": 1}
    )
    if certification is not None:
        analytics = certification.get("certification_analytics") if isinstance(certification.get("certification_analytics"), dict) else None
        if analytics is None:
            raise HTTPException(status_code=409, detail="CARO certification analytics are unavailable.")
        return bson_value(dict(analytics))
    validation = db[MODEL_TUNING_VALIDATIONS_COLLECTION].find_one({"id": str(processing_id)}, {"_id": 0})
    if validation is not None:
        analytics = validation.get("analytics") if isinstance(validation.get("analytics"), dict) else None
        if analytics is None:
            raise HTTPException(status_code=409, detail="CARO validation analytics are unavailable.")
        return bson_value(dict(analytics))
    payload = backtest_analytics(db, processing_id)
    payload["processing_id"] = processing_id
    payload["processing_kind"] = "backtest"
    payload["processing_label"] = "Backtest"
    payload["reference_label"] = "Reference"
    return bson_value(payload)


def processing_rotation_period_analysis(
    db: Any,
    processing_id: str,
    *,
    year: int,
    month: int,
) -> dict[str, Any]:
    from ..milp_decision.processing import processing_analytics as milp_processing_analytics
    milp = milp_processing_analytics(db, processing_id)
    if milp is not None:
        return _rotation_period_from_data(
            db, processing_id, equity=list(milp.get("equity") or []),
            rotations=list(milp.get("rotations") or []), year=year, month=month,
        )
    temporal = _temporal_strategy_processing_analytics(db, processing_id)
    if temporal is not None:
        return _rotation_period_from_data(
            db, processing_id, equity=list(temporal.get("equity") or []),
            rotations=list(temporal.get("rotations") or []), year=year, month=month,
        )
    stateful = _stateful_strategy_processing_analytics(db, processing_id)
    if stateful is not None:
        return _rotation_period_from_data(
            db,
            processing_id,
            equity=list(stateful.get("equity") or []),
            rotations=list(stateful.get("rotations") or []),
            year=year,
            month=month,
        )
    validation = db[MODEL_TUNING_VALIDATIONS_COLLECTION].find_one(
        {"$or": [{"id": str(processing_id)}, {"certification_processing_id": str(processing_id)}]},
        {"_id": 0, "id": 1, "analytics": 1, "certification_processing_id": 1, "certification_analytics": 1},
    )
    if validation is None:
        return rotation_period_analysis(db, processing_id, year=year, month=month)
    if str(validation.get("certification_processing_id") or "") == str(processing_id):
        analytics = validation.get("certification_analytics") if isinstance(validation.get("certification_analytics"), dict) else {}
    else:
        analytics = validation.get("analytics") if isinstance(validation.get("analytics"), dict) else {}
    return _rotation_period_from_data(
        db,
        processing_id,
        equity=list(analytics.get("equity") or []),
        rotations=list(analytics.get("rotations") or []),
        year=year,
        month=month,
    )


def completed_backtests(db: Any, limit: int = 100) -> dict[str, Any]:
    safe_limit = max(1, min(500, int(limit)))
    rows = list(
        db[JOBS_COLLECTION].find(
            {"status": "completed"},
            {"_id": 0, "id": 1, "created_at": 1, "finished_at": 1, "status": 1},
        )
    )
    rows = _sorted_rows(rows, "created_at")
    rows.reverse()
    return {
        "items": [
            {
                "id": str(row.get("id") or ""),
                "status": "completed",
                "created_at": iso_value(row.get("created_at")),
                "finished_at": iso_value(row.get("finished_at")),
            }
            for row in rows[:safe_limit]
        ]
    }


def _portfolio_history(db: Any, limit: int = 5000) -> list[dict[str, Any]]:
    rows = list(
        db[PAPER_PORTFOLIO_SNAPSHOTS_COLLECTION].find(
            {},
            {
                "_id": 0,
                "recorded_at": 1,
                "portfolio_value": 1,
                "strategy_cash": 1,
                "market_value": 1,
                "total_pnl": 1,
                "total_return": 1,
                "managed_symbol": 1,
            },
        )
    )
    rows = _sorted_rows(rows, "recorded_at")[-max(1, int(limit)):]
    peak: float | None = None
    output: list[dict[str, Any]] = []
    for row in rows:
        value = _as_float(row.get("portfolio_value"))
        if value is None:
            continue
        peak = value if peak is None else max(peak, value)
        output.append(
            {
                "recorded_at": iso_value(row.get("recorded_at")),
                "portfolio_value": value,
                "strategy_cash": _as_float(row.get("strategy_cash")),
                "market_value": _as_float(row.get("market_value")),
                "total_pnl": _as_float(row.get("total_pnl")),
                "total_return": _as_float(row.get("total_return")),
                "managed_symbol": row.get("managed_symbol"),
                "drawdown": value / peak - 1.0 if peak else 0.0,
            }
        )
    return output


def _period_return(history: list[dict[str, Any]], days: int) -> float | None:
    if len(history) < 2:
        return None
    latest = history[-1]
    latest_at = _parse_iso(latest.get("recorded_at"))
    latest_value = _as_float(latest.get("portfolio_value"))
    if latest_at is None or latest_value is None:
        return None
    target = latest_at - timedelta(days=days)
    candidates = [row for row in history if (_parse_iso(row.get("recorded_at")) or latest_at) <= target]
    if not candidates:
        return None
    base = _as_float(candidates[-1].get("portfolio_value"))
    return latest_value / base - 1.0 if base not in {None, 0} else None


def _parse_iso(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _daily_portfolio_returns(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    daily: dict[str, dict[str, Any]] = {}
    for row in history:
        parsed = _parse_iso(row.get("recorded_at"))
        if parsed is None:
            continue
        daily[parsed.date().isoformat()] = row
    result: list[dict[str, Any]] = []
    previous: float | None = None
    for day in sorted(daily):
        value = _as_float(daily[day].get("portfolio_value"))
        result.append(
            {
                "date": day,
                "portfolio_return": value / previous - 1.0 if value is not None and previous not in {None, 0} else None,
                "portfolio_value": value,
            }
        )
        previous = value
    return result[1:] if len(result) > 1 else []


def _public_orders(db: Any, limit: int = 500) -> list[dict[str, Any]]:
    rows = list(
        db[PAPER_TRADE_ORDERS_COLLECTION].find(
            {},
            {
                "_id": 0,
                "symbol": 1,
                "side": 1,
                "status": 1,
                "quantity": 1,
                "notional": 1,
                "filled_quantity": 1,
                "filled_average_price": 1,
                "submitted_at": 1,
                "filled_at": 1,
                "created_at": 1,
                "updated_at": 1,
            },
        )
    )
    rows = _sorted_rows(rows, "created_at")[-max(1, int(limit)):]
    rows.reverse()
    return [bson_value(row) for row in rows]


def _order_analytics(orders: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = defaultdict(int)
    side_counts: dict[str, int] = defaultdict(int)
    symbol_counts: dict[str, int] = defaultdict(int)
    fill_delays: list[float] = []
    for row in orders:
        status_counts[str(row.get("status") or "unknown").lower()] += 1
        side_counts[str(row.get("side") or "unknown").lower()] += 1
        symbol_counts[str(row.get("symbol") or "UNKNOWN").upper()] += 1
        submitted = _as_utc(row.get("submitted_at"))
        filled = _as_utc(row.get("filled_at"))
        if submitted and filled and filled >= submitted:
            fill_delays.append((filled - submitted).total_seconds())
    filled_count = sum(count for status, count in status_counts.items() if status in {"filled", "partially_filled"})
    rejected_count = status_counts.get("rejected", 0)
    return {
        "total_orders": len(orders),
        "filled_orders": filled_count,
        "rejected_orders": rejected_count,
        "fill_rate": _safe_divide(filled_count, len(orders)) if orders else None,
        "rejection_rate": _safe_divide(rejected_count, len(orders)) if orders else None,
        "average_fill_delay_seconds": float(fmean(fill_delays)) if fill_delays else None,
        "status_counts": dict(sorted(status_counts.items())),
        "side_counts": dict(sorted(side_counts.items())),
        "symbol_counts": dict(sorted(symbol_counts.items(), key=lambda item: (-item[1], item[0]))),
    }


def portfolio_analytics(db: Any) -> dict[str, Any]:
    live: dict[str, Any] | None = None
    connection = {
        "status": "unavailable",
        "message": "Live Paper portfolio is currently unavailable.",
        "configured": bool(get_alpaca_integration_status(db).get("configured")),
    }
    try:
        from .public_paper_portfolio import public_paper_portfolio_snapshot
        live = public_paper_portfolio_snapshot(db)
        connection = {
            "status": "ready",
            "message": "Alpaca Paper connection is available.",
            "configured": True,
        }
    except Exception as exc:  
        message = str(exc).lower()
        if "unauthorized" in message or "401" in message:
            reason = "Alpaca Paper rejected the configured credentials."
        else:
            reason = "Live Paper portfolio could not be refreshed."
        connection["message"] = reason

    history = _portfolio_history(db)
    orders = _public_orders(db)
    latest = history[-1] if history else None
    live_current = (
        {
            key: live.get(key)
            for key in (
                "status",
                "recorded_at",
                "initial_capital",
                "strategy_cash",
                "market_value",
                "portfolio_value",
                "realized_pnl",
                "unrealized_pnl",
                "total_pnl",
                "total_return",
                "position",
                "last_decision_date",
                "last_execution_session",
                "market_clock",
            )
            if live.get(key) is not None
        }
        if live
        else None
    )
    current = live_current or (
        {
            "status": "historical_only",
            "recorded_at": latest.get("recorded_at"),
            "strategy_cash": latest.get("strategy_cash"),
            "market_value": latest.get("market_value"),
            "portfolio_value": latest.get("portfolio_value"),
            "total_pnl": latest.get("total_pnl"),
            "total_return": latest.get("total_return"),
            "position": None,
            "recent_orders": orders[:20],
        }
        if latest
        else None
    )
    drawdowns = [value for row in history if (value := _as_float(row.get("drawdown"))) is not None]
    portfolio_value = _as_float((current or {}).get("portfolio_value"))
    market_value = _as_float((current or {}).get("market_value"))
    payload = {
        "connection": {
            **connection,
            "last_success_at": latest.get("recorded_at") if latest else None,
        },
        "current": current,
        "summary": {
            "portfolio_value": portfolio_value,
            "strategy_cash": _as_float((current or {}).get("strategy_cash")),
            "market_value": market_value,
            "realized_pnl": _as_float((current or {}).get("realized_pnl")),
            "unrealized_pnl": _as_float((current or {}).get("unrealized_pnl")),
            "total_pnl": _as_float((current or {}).get("total_pnl")),
            "total_return": _as_float((current or {}).get("total_return")),
            "market_exposure": market_value / portfolio_value if market_value is not None and portfolio_value not in {None, 0} else None,
            "current_drawdown": drawdowns[-1] if drawdowns else None,
            "maximum_drawdown": min(drawdowns) if drawdowns else None,
            "return_1_day": _period_return(history, 1),
            "return_7_days": _period_return(history, 7),
            "return_30_days": _period_return(history, 30),
        },
        "history": history,
        "daily_returns": _daily_portfolio_returns(history),
        "orders": orders[:100],
        "order_analytics": _order_analytics(orders),
    }
    _assert_strategy_neutral(payload)
    return payload


def _assert_strategy_neutral(value: Any) -> None:
    if isinstance(value, dict):
        forbidden = _FORBIDDEN_OUTPUT_KEYS.intersection(value)
        if forbidden:
            raise RuntimeError("Analytics payload contains protected strategy fields.")
        for item in value.values():
            _assert_strategy_neutral(item)
    elif isinstance(value, list):
        for item in value:
            _assert_strategy_neutral(item)
