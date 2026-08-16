from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from statistics import fmean, median
from typing import Any, Iterable

from fastapi import HTTPException

from ..infrastructure.persistence.mongo_repository import (
    ALPACA_MARKET_BARS_COLLECTION,
    COMPARISONS_COLLECTION,
    JOBS_COLLECTION,
    MARKET_BARS_COLLECTION,
    PAPER_PORTFOLIO_SNAPSHOTS_COLLECTION,
    PAPER_TRADE_ORDERS_COLLECTION,
    PREDICTIONS_COLLECTION,
    RUNS_COLLECTION,
    TRADES_COLLECTION,
    bson_value,
    get_alpaca_integration_status,
)
from .admin_rotations import admin_job_rotations
from .dashboard import _public_metrics, _selected_internal_row
from .serialization import iso_value


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



def _effective_config(db: Any, job_id: str) -> dict[str, Any]:
    comparison = db[COMPARISONS_COLLECTION].find_one(
        {"job_id": job_id},
        {"_id": 0, "effective_config": 1},
    )
    config = (comparison or {}).get("effective_config")
    return dict(config) if isinstance(config, dict) else {}


def _available_assets(db: Any, job_id: str, trades: Iterable[dict[str, Any]] = ()) -> list[str]:
    config = _effective_config(db, job_id)
    configured = config.get("assets")
    symbols = {
        str(symbol).strip().upper()
        for symbol in (configured if isinstance(configured, list) else [])
        if str(symbol).strip()
    }
    symbols.update(
        str(row.get("asset") or "").strip().upper()
        for row in trades
        if str(row.get("asset") or "").strip()
    )
    return sorted(symbols)


def _legacy_interval(value: Any) -> str:
    normalized = str(value or "1Day").strip().lower()
    aliases = {
        "1day": "1d",
        "1d": "1d",
        "day": "1d",
        "1hour": "1h",
        "1h": "1h",
        "1minute": "1m",
        "1min": "1m",
        "1m": "1m",
    }
    return aliases.get(normalized, normalized)


def _market_close_rows(
    db: Any,
    *,
    asset: str,
    start: datetime,
    end: datetime,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    timestamp_filter = {"$gte": start, "$lte": end}
    interval = str(config.get("timeframe") or "1Day")
    alpaca_query: dict[str, Any] = {
        "symbol": asset,
        "interval": interval,
        "timestamp": timestamp_filter,
    }
    feed = str(config.get("alpaca_historical_feed") or "").strip()
    adjustment = str(config.get("alpaca_adjustment") or "").strip()
    if feed:
        alpaca_query["feed"] = feed
    if adjustment:
        alpaca_query["adjustment"] = adjustment

    projection = {"_id": 0, "timestamp": 1, "close": 1}
    sources = (
        (ALPACA_MARKET_BARS_COLLECTION, alpaca_query),
        (
            MARKET_BARS_COLLECTION,
            {
                "symbol": asset,
                "interval": _legacy_interval(interval),
                "timestamp": timestamp_filter,
            },
        ),
    )
    for collection_name, query in sources:
        rows = list(db[collection_name].find(query, projection))
        normalized: list[dict[str, Any]] = []
        for row in rows:
            timestamp = _as_utc(row.get("timestamp"))
            close = _as_float(row.get("close"))
            if timestamp is None or close is None or close <= 0:
                continue
            normalized.append({"timestamp": timestamp, "close": close})
        normalized.sort(key=lambda row: row["timestamp"])
        if normalized:
            return normalized
    return []


def _asset_exposure_by_day(db: Any, job_id: str, backend: str, asset: str) -> dict[str, float]:
    rows = db[PREDICTIONS_COLLECTION].find(
        {"job_id": job_id, "symbol": "PORTFOLIO", "backend": backend},
        {
            "_id": 0,
            "timestamp": 1,
            "selected_asset": 1,
            "portfolio_weights": 1,
        },
    )
    result: dict[str, float] = {}
    for row in rows:
        timestamp = _as_utc(row.get("timestamp"))
        if timestamp is None:
            continue
        weights = row.get("portfolio_weights")
        weight = _as_float(weights.get(asset)) if isinstance(weights, dict) else None
        if weight is None:
            weight = 1.0 if str(row.get("selected_asset") or "").strip().upper() == asset else 0.0
        result[timestamp.date().isoformat()] = max(0.0, min(1.0, float(weight)))
    return result


def _asset_trade_events(db: Any, job_id: str, backend: str, asset: str) -> list[dict[str, Any]]:
    rows = db[TRADES_COLLECTION].find(
        {"job_id": job_id, "symbol": "PORTFOLIO", "backend": backend, "asset": asset},
        {
            "_id": 0,
            "timestamp": 1,
            "decision_timestamp": 1,
            "sequence": 1,
            "action": 1,
            "execution_price": 1,
            "quantity": 1,
            "holding_bars": 1,
            "position_return": 1,
            "realized_pnl": 1,
            "total_fee": 1,
        },
    )
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            _as_utc(row.get("timestamp")) or datetime.min.replace(tzinfo=timezone.utc),
            int(row.get("sequence") or 0),
        ),
    )
    result: list[dict[str, Any]] = []
    for row in ordered:
        action = str(row.get("action") or "").strip().upper()
        position_return = _as_float(row.get("position_return"))
        realized_pnl = _as_float(row.get("realized_pnl"))
        result_value = realized_pnl if realized_pnl is not None else position_return
        outcome = "entry"
        if action in {"SELL", "FINAL_SELL"}:
            outcome = "positive" if result_value is not None and result_value > 0 else "negative" if result_value is not None and result_value < 0 else "neutral"
        result.append(
            {
                "timestamp": iso_value(row.get("timestamp")),
                "decision_timestamp": iso_value(row.get("decision_timestamp")),
                "action": action,
                "execution_price": _as_float(row.get("execution_price")),
                "quantity": _as_float(row.get("quantity")),
                "holding_days": _as_float(row.get("holding_bars")),
                "position_return": position_return,
                "realized_pnl": realized_pnl,
                "transaction_fee": _as_float(row.get("total_fee")),
                "outcome": outcome,
            }
        )
    return result


def asset_strategy_comparison(db: Any, job_id: str, asset: str) -> dict[str, Any]:
    job = db[JOBS_COLLECTION].find_one(
        {"id": job_id},
        {"_id": 0, "id": 1, "status": 1, "created_at": 1, "finished_at": 1},
    )
    if job is None:
        raise HTTPException(status_code=404, detail="Backtest job not found.")
    if str(job.get("status") or "").lower() != "completed":
        raise HTTPException(status_code=409, detail="Analytics are available after the backtest completes.")

    backend = _selected_backend(db, job_id)
    if not backend:
        raise HTTPException(status_code=409, detail="Selected backtest result is unavailable.")

    trades = _trade_rows(db, job_id, backend)
    available_assets = _available_assets(db, job_id, trades)
    normalized_asset = str(asset or "").strip().upper()
    if not normalized_asset or normalized_asset not in available_assets:
        raise HTTPException(status_code=404, detail="Asset is not part of the selected backtest universe.")

    equity = _equity_rows(db, job_id, backend)
    if len(equity) < 2:
        raise HTTPException(status_code=409, detail="Not enough strategy observations for asset comparison.")
    start = _parse_iso(equity[0].get("timestamp"))
    end = _parse_iso(equity[-1].get("timestamp"))
    if start is None or end is None:
        raise HTTPException(status_code=409, detail="Backtest timestamps are unavailable for asset comparison.")

    market_rows = _market_close_rows(
        db,
        asset=normalized_asset,
        start=start - timedelta(days=2),
        end=end + timedelta(days=2),
        config=_effective_config(db, job_id),
    )
    market_by_day = {row["timestamp"].date().isoformat(): row for row in market_rows}
    exposure_by_day = _asset_exposure_by_day(db, job_id, backend, normalized_asset)

    aligned: list[dict[str, Any]] = []
    for row in equity:
        timestamp = _parse_iso(row.get("timestamp"))
        if timestamp is None:
            continue
        market = market_by_day.get(timestamp.date().isoformat())
        strategy_equity = _as_float(row.get("simulation_equity"))
        if market is None or strategy_equity is None:
            continue
        aligned.append(
            {
                "timestamp": row.get("timestamp"),
                "strategy_equity": strategy_equity,
                "asset_close": float(market["close"]),
                "strategy_weight": float(exposure_by_day.get(timestamp.date().isoformat(), 0.0)),
            }
        )
    if len(aligned) < 2:
        raise HTTPException(status_code=404, detail="No aligned market history is available for this asset.")

    first_strategy = float(aligned[0]["strategy_equity"])
    first_asset = float(aligned[0]["asset_close"])
    for row in aligned:
        row["strategy_index"] = float(row["strategy_equity"]) / first_strategy * 100.0 if first_strategy else None
        row["asset_index"] = float(row["asset_close"]) / first_asset * 100.0 if first_asset else None

    strategy_return = float(aligned[-1]["strategy_equity"]) / first_strategy - 1.0 if first_strategy else None
    asset_return = float(aligned[-1]["asset_close"]) / first_asset - 1.0 if first_asset else None
    exposure_values = [float(row.get("strategy_weight") or 0.0) for row in aligned]
    sells = _completed_sells(trades)
    attribution = next(
        (row for row in _asset_attribution(sells) if str(row.get("asset") or "").upper() == normalized_asset),
        None,
    ) or {
        "asset": normalized_asset,
        "closed_positions": 0,
        "profitable_positions": 0,
        "losing_positions": 0,
        "win_rate": None,
        "total_realized_pnl": 0.0,
        "average_position_return": None,
        "transaction_fees": 0.0,
    }
    events = _asset_trade_events(db, job_id, backend, normalized_asset)
    payload = {
        "job_id": job_id,
        "asset": normalized_asset,
        "available_assets": available_assets,
        "started_at": aligned[0]["timestamp"],
        "ended_at": aligned[-1]["timestamp"],
        "summary": {
            "strategy_return": strategy_return,
            "asset_return": asset_return,
            "relative_return": strategy_return - asset_return if strategy_return is not None and asset_return is not None else None,
            "exposure_rate": _safe_divide(sum(value > 1e-9 for value in exposure_values), len(exposure_values)) if exposure_values else None,
            "average_weight": float(fmean(exposure_values)) if exposure_values else None,
            "closed_positions": attribution.get("closed_positions", 0),
            "profitable_positions": attribution.get("profitable_positions", 0),
            "losing_positions": attribution.get("losing_positions", 0),
            "win_rate": attribution.get("win_rate"),
            "realized_pnl": attribution.get("total_realized_pnl", 0.0),
            "average_position_return": attribution.get("average_position_return"),
            "transaction_fees": attribution.get("transaction_fees", 0.0),
        },
        "series": aligned,
        "events": events,
    }
    _assert_strategy_neutral(payload)
    return payload

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
            "available_assets": _available_assets(db, job_id),
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
        "available_assets": _available_assets(db, job_id, trades),
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
