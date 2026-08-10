from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from ..core.config import API_VERSION
from ..infrastructure.persistence.mongo_repository import (
    ASSET_DISCOVERY_CANDIDATES_COLLECTION,
    ASSET_DISCOVERY_RUNS_COLLECTION,
    ASSET_DISCOVERY_STATE_COLLECTION,
    bson_value,
    utc_now,
)
from .asset_discovery_behavior import ASSET_DISCOVERY_EVALUATION_POLICY_VERSION
from .asset_discovery_market import (
    MarketDataAccessBlocked,
    NoHistoricalMarketData,
    NoRecentMarketData,
    discover_alpaca_symbols,
    market_quality_snapshot,
    resolve_completed_market_data_end,
)
from .asset_discovery_settings import (
    ensure_asset_discovery_settings,
    normalized_asset_discovery_settings,
)
from .asset_discovery_store import append_run_update, finish_run
from .strategy_lab import get_research_strategy_context

STATE_ID = "default"


def _aware_utc(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def cleanup_non_analytical_candidate_records(db: Any) -> None:
    """Keep the Candidate Pool analytical: operational/no-data outcomes are not candidates."""

    db[ASSET_DISCOVERY_CANDIDATES_COLLECTION].delete_many(
        {"status": {"$in": ["failed", "skipped"]}}
    )


def eligible_symbols(
    db: Any,
    universe: list[str],
    current_assets: set[str],
) -> list[str]:
    now = utc_now()
    state = db[ASSET_DISCOVERY_STATE_COLLECTION].find_one({"_id": STATE_ID}) or {}
    cursor_symbol = str(state.get("cursor_symbol") or "")
    ordered = [symbol for symbol in universe if symbol not in current_assets]
    if cursor_symbol and cursor_symbol in ordered:
        index = ordered.index(cursor_symbol) + 1
        ordered = ordered[index:] + ordered[:index]

    existing_by_symbol = {
        str(item.get("symbol") or ""): _aware_utc(item.get("next_evaluation_at"))
        for item in db[ASSET_DISCOVERY_CANDIDATES_COLLECTION].find(
            {}, {"symbol": 1, "next_evaluation_at": 1, "_id": 0}
        )
    }
    return [
        symbol
        for symbol in ordered
        if existing_by_symbol.get(symbol) is None or existing_by_symbol[symbol] <= now
    ]


def evaluate_symbol(
    db: Any,
    run_id: str,
    symbol: str,
    config: Any,
    settings: dict[str, Any],
    recent_end: datetime,
) -> str:
    now = utc_now()
    collection = db[ASSET_DISCOVERY_CANDIDATES_COLLECTION]
    try:
        snapshot = market_quality_snapshot(symbol, config, settings, recent_end=recent_end)
        collection.update_one(
            {"symbol": symbol},
            {
                "$set": bson_value(
                    {
                        "symbol": symbol,
                        **snapshot,
                        "last_evaluated_at": now,
                        "next_evaluation_at": now + timedelta(days=int(settings["recheck_days"])),
                        "last_run_id": run_id,
                        "last_evaluated_api_version": API_VERSION,
                        "evaluation_policy_version": ASSET_DISCOVERY_EVALUATION_POLICY_VERSION,
                        "last_error": None,
                    }
                ),
                "$setOnInsert": {
                    "discovered_at": now,
                    "discovered_api_version": API_VERSION,
                },
                "$inc": {"evaluation_count": 1},
            },
            upsert=True,
        )
        return str(snapshot["status"])
    except NoRecentMarketData:
        append_run_update(
            db,
            run_id,
            message=f"Skipped {symbol}: insufficient recent market data.",
        )
        return "skipped"
    except NoHistoricalMarketData:
        append_run_update(
            db,
            run_id,
            message=f"Skipped {symbol}: insufficient historical market data.",
        )
        return "skipped"
    except MarketDataAccessBlocked:
        # A subscription/auth/feed error is global, not a property of this symbol.
        # Persisting it as a candidate would pollute the Candidate Pool, so let
        # the worker abort the batch and keep the diagnostic only in the run.
        raise
    except Exception as exc:
        # Technical failures are operational diagnostics, not candidate states.
        # Preserve any prior analytical record for the symbol and store the error
        # only in the bounded run log/counter.
        message = str(exc)[:1000]
        append_run_update(
            db,
            run_id,
            message=f"Technical evaluation failure for {symbol}: {message}",
        )
        return "failed"


def run_asset_discovery_worker(db: Any, run_id: str, stop_event: Any) -> None:
    try:
        settings = normalized_asset_discovery_settings(
            ensure_asset_discovery_settings(db).get("settings")
        )
        cleanup_non_analytical_candidate_records(db)
        config, _ = get_research_strategy_context(db)
        market_config = config.model_copy(update={"end_date": None, "mongo_cache_enabled": True})
        recent_end = resolve_completed_market_data_end()
        batch_target = int(settings["batch_size"])
        max_attempts = max(batch_target, int(settings["max_scan_attempts"]))
        append_run_update(
            db,
            run_id,
            message="Loading the Alpaca US equity universe.",
            changes={"status": "running", "phase": "discovering", "started_at": utc_now()},
        )
        universe = discover_alpaca_symbols()
        symbols = eligible_symbols(db, universe, set(config.assets))
        append_run_update(
            db,
            run_id,
            message=f"Searching for up to {batch_target} evaluable assets in this bounded discovery batch.",
            changes={
                "phase": "evaluating",
                "universe_size": len(universe),
                "batch_size": batch_target,
                "scan_limit": max_attempts,
            },
        )
        if not symbols:
            finish_run(db, run_id, status="completed", message="No candidate symbols are due for evaluation in this cycle.")
            return

        processed = 0
        attempted = 0
        for symbol in symbols:
            if processed >= batch_target or attempted >= max_attempts:
                break
            run = db[ASSET_DISCOVERY_RUNS_COLLECTION].find_one({"run_id": run_id}) or {}
            if stop_event.is_set() or bool(run.get("cancel_requested")):
                finish_run(db, run_id, status="stopped", message="Asset Discovery stopped safely before the next symbol.")
                return

            append_run_update(db, run_id, message=f"Evaluating {symbol}.", changes={"current_symbol": symbol})
            try:
                result = evaluate_symbol(db, run_id, symbol, market_config, settings, recent_end)
            except MarketDataAccessBlocked as exc:
                db[ASSET_DISCOVERY_RUNS_COLLECTION].update_one(
                    {"run_id": run_id},
                    {"$inc": {"attempted_count": 1, "failed_count": 1}},
                )
                finish_run(
                    db,
                    run_id,
                    status="failed",
                    message=(
                        "Asset Discovery stopped because Alpaca market-data access is unavailable for "
                        f"the configured feed: {str(exc)[:700]}"
                    ),
                )
                return
            attempted += 1
            counted = result in {"candidate", "watchlist", "rejected"}
            if counted:
                processed += 1
            increment_field = {
                "candidate": "candidate_count",
                "watchlist": "watchlist_count",
                "rejected": "rejected_count",
                "failed": "failed_count",
                "skipped": "skipped_count",
            }.get(result, "skipped_count")
            increments = {"attempted_count": 1, increment_field: 1}
            if counted:
                increments["processed_count"] = 1
            db[ASSET_DISCOVERY_RUNS_COLLECTION].update_one(
                {"run_id": run_id}, {"$inc": increments}
            )
            db[ASSET_DISCOVERY_STATE_COLLECTION].update_one(
                {"_id": STATE_ID},
                {"$set": {"cursor_symbol": symbol, "updated_at": utc_now()}},
                upsert=True,
            )

        if processed >= batch_target:
            message = f"Asset Discovery batch completed with {processed} evaluable assets."
        elif attempted >= max_attempts:
            message = (
                f"Asset Discovery stopped after the scan safety limit: {processed} evaluable assets "
                f"from {attempted} attempted symbols."
            )
        else:
            message = f"Asset Discovery completed with {processed} evaluable assets; no more symbols were due."
        finish_run(db, run_id, status="completed", message=message)
    except Exception as exc:
        finish_run(db, run_id, status="failed", message=f"Asset Discovery failed: {exc}")
    finally:
        stop_event.clear()
