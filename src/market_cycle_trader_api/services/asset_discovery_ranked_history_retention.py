from __future__ import annotations

from typing import Any, Callable


RETENTION_ID = "ranked-history-retention"
_INSTALLED = False
_ORIGINAL_START: Callable[..., Any] | None = None
_ORIGINAL_FINISH: Callable[..., Any] | None = None
_ORIGINAL_PROTECTED_SYMBOLS: Callable[..., set[str]] | None = None


def _result_symbols(document: dict[str, Any] | None) -> list[str]:
    if not isinstance(document, dict):
        return []
    symbols: list[str] = []
    for item in list(document.get("results") or []):
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol") or "").strip().upper()
        if symbol:
            symbols.append(symbol)
    return list(dict.fromkeys(symbols))


def _persist_previous_ranked_set(service: Any, db: Any, document: dict[str, Any]) -> None:
    ranked = _result_symbols(document)
    db[service.COLLECTION].replace_one(
        {"_id": RETENTION_ID},
        {
            "_id": RETENTION_ID,
            "source_run_id": document.get("run_id"),
            "source_status": document.get("status"),
            "ranked_symbols": ranked,
            "ranked_symbol_count": len(ranked),
            "captured_at": service.utc_now(),
            "updated_at": service.utc_now(),
        },
        upsert=True,
    )


def _retained_ranked_symbols(service: Any, db: Any) -> set[str]:
    document = db[service.COLLECTION].find_one({"_id": RETENTION_ID}) or {}
    return {
        str(symbol or "").strip().upper()
        for symbol in list(document.get("ranked_symbols") or [])
        if str(symbol or "").strip()
    }


def _persist_ranked_history_frames(service: Any, history: Any, db: Any, results: list[dict[str, Any]]) -> dict[str, Any]:
    symbols = _result_symbols({"results": results})
    if not symbols:
        return {"ranked_symbol_count": 0, "persisted_symbol_count": 0, "persisted_symbols": []}

    campaign = service._campaign(db) or {}
    baseline = campaign.get("baseline") if isinstance(campaign.get("baseline"), dict) else {}
    research_window = campaign.get("research_window") if isinstance(campaign.get("research_window"), dict) else {}
    end_session = str(
        research_window.get("research_end")
        or baseline.get("market_snapshot_end")
        or ""
    ).strip()
    if not end_session:
        return {"ranked_symbol_count": len(symbols), "persisted_symbol_count": 0, "persisted_symbols": []}

    config, _strategy = service.get_research_strategy_context(db)
    collection = db[service.ALPACA_MARKET_BARS_COLLECTION]
    persisted: list[str] = []
    for symbol in symbols:
        key = history._cache_key(symbol, config, end_session)
        frame, _coverage, failure = history._cache_lookup(key)
        if failure or frame is None or getattr(frame, "empty", True):
            continue
        identity = service._market_data_identity(symbol, config)
        service._upsert_frame(collection, frame, identity, config.mongo_write_batch_size)
        persisted.append(symbol)

    return {
        "ranked_symbol_count": len(symbols),
        "persisted_symbol_count": len(persisted),
        "persisted_symbols": persisted,
    }


def install_asset_discovery_ranked_history_retention() -> None:
    global _INSTALLED, _ORIGINAL_START, _ORIGINAL_FINISH, _ORIGINAL_PROTECTED_SYMBOLS
    if _INSTALLED:
        return

    from . import asset_discovery as service
    from . import asset_discovery_predictive_history_integrity as history

    _ORIGINAL_START = service.start_asset_discovery
    _ORIGINAL_FINISH = service._finish
    _ORIGINAL_PROTECTED_SYMBOLS = history._protected_market_history_symbols

    def start_with_previous_ranked_retention(db: Any, *, research_size: int) -> dict[str, Any]:
        current = service._campaign(db) or {}
        if str(current.get("status") or "").strip().lower() not in service.ACTIVE_STATUSES:
            _persist_previous_ranked_set(service, db, current)
        original = _ORIGINAL_START
        if original is None:
            raise RuntimeError("Asset Discovery start function is unavailable.")
        return original(db, research_size=research_size)

    def protected_symbols_with_previous_ranked(service_arg: Any, db: Any) -> set[str]:
        original = _ORIGINAL_PROTECTED_SYMBOLS
        protected = set(original(service_arg, db)) if original is not None else set()
        protected.update(_retained_ranked_symbols(service_arg, db))
        return protected

    def finish_with_ranked_history_retention(
        db: Any,
        run_id: str,
        status: str,
        message: str,
        *,
        results: list[dict[str, Any]] | None = None,
    ) -> None:
        retention_summary: dict[str, Any] | None = None
        if status == "completed" and results is not None:
            try:
                retention_summary = _persist_ranked_history_frames(service, history, db, results)
                db[service.COLLECTION].update_one(
                    {"_id": service.CURRENT_ID, "run_id": run_id},
                    {"$set": {
                        "ranked_history_retention": service.bson_value(retention_summary),
                        "updated_at": service.utc_now(),
                    }},
                )
            except Exception as exc:
                retention_summary = {"status": "failed", "error": str(exc)[:500]}
                db[service.COLLECTION].update_one(
                    {"_id": service.CURRENT_ID, "run_id": run_id},
                    {"$set": {
                        "ranked_history_retention": service.bson_value(retention_summary),
                        "updated_at": service.utc_now(),
                    }},
                )

        original = _ORIGINAL_FINISH
        if original is None:
            raise RuntimeError("Asset Discovery finish function is unavailable.")
        original(db, run_id, status, message, results=results)

    setattr(start_with_previous_ranked_retention, "_asset_discovery_ranked_history_retention", True)
    setattr(protected_symbols_with_previous_ranked, "_asset_discovery_ranked_history_retention", True)
    setattr(finish_with_ranked_history_retention, "_asset_discovery_ranked_history_retention", True)
    service.start_asset_discovery = start_with_previous_ranked_retention
    history._protected_market_history_symbols = protected_symbols_with_previous_ranked
    service._finish = finish_with_ranked_history_retention
    _INSTALLED = True
