from __future__ import annotations

import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import pandas as pd


POLICY_VERSION = "predictive-full-history-v3-prefetch-cache"
_AUTO_WORKER_NAME = "asset-discovery-ranker"
_HISTORY_CACHE_LIMIT = 512
_INSTALLED = False
_ORIGINAL_MARGINAL_REPLAY: Callable[..., Any] | None = None
_ORIGINAL_GET_STATUS: Callable[..., dict[str, Any]] | None = None
_ORIGINAL_CANDIDATE_FRAME: Callable[..., Any] | None = None
_ORIGINAL_CANDIDATE_HISTORY_COVERAGE: Callable[..., Any] | None = None
_ORIGINAL_IDENTITY_INTEGRITY: Callable[..., Any] | None = None
_HISTORY_CACHE_LOCK = threading.Lock()
_HISTORY_CACHE: OrderedDict[tuple[str, ...], tuple[Any, dict[str, Any]]] = OrderedDict()
_HISTORY_FAILURES: dict[tuple[str, ...], str] = {}


def _automatic_worker() -> bool:
    return threading.current_thread().name == _AUTO_WORKER_NAME


def _symbol(item: Any) -> str:
    return str((item or {}).get("symbol") or "").strip().upper() if isinstance(item, dict) else ""


def _coverage_ok(coverage: Any) -> bool:
    return isinstance(coverage, dict) and bool(coverage.get("history_window_complete"))


def _cache_key(symbol: str, config: Any, end_session: Any) -> tuple[str, ...]:
    return (
        str(symbol or "").strip().upper(),
        str(getattr(config, "start_date", "") or ""),
        str(pd.Timestamp(end_session).date()),
        str(getattr(config, "timeframe", "") or ""),
        str(getattr(config, "alpaca_historical_feed", "") or ""),
        str(getattr(config, "alpaca_adjustment", "") or ""),
        str(int(getattr(config, "market_data_history_start_tolerance_days", 0) or 0)),
    )


def _utc_timestamp(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _copy_frame(frame: Any) -> Any:
    try:
        return frame.copy(deep=False)
    except Exception:
        try:
            return frame.copy()
        except Exception:
            return frame


def _clear_transient_history_cache() -> None:
    with _HISTORY_CACHE_LOCK:
        _HISTORY_CACHE.clear()
        _HISTORY_FAILURES.clear()


def _cache_success(key: tuple[str, ...], frame: Any, coverage: dict[str, Any]) -> None:
    with _HISTORY_CACHE_LOCK:
        _HISTORY_FAILURES.pop(key, None)
        _HISTORY_CACHE[key] = (_copy_frame(frame), dict(coverage or {}))
        _HISTORY_CACHE.move_to_end(key)
        while len(_HISTORY_CACHE) > _HISTORY_CACHE_LIMIT:
            evicted_key, _value = _HISTORY_CACHE.popitem(last=False)
            _HISTORY_FAILURES.pop(evicted_key, None)


def _cache_failure(key: tuple[str, ...], reason: str) -> None:
    with _HISTORY_CACHE_LOCK:
        _HISTORY_CACHE.pop(key, None)
        _HISTORY_FAILURES[key] = str(reason or "candidate_history_prefetch_failed")[:300]


def _cache_lookup(key: tuple[str, ...]) -> tuple[Any | None, dict[str, Any] | None, str | None]:
    with _HISTORY_CACHE_LOCK:
        failure = _HISTORY_FAILURES.get(key)
        if failure:
            return None, None, failure
        cached = _HISTORY_CACHE.get(key)
        if cached is None:
            return None, None, None
        _HISTORY_CACHE.move_to_end(key)
        frame, coverage = cached
        return _copy_frame(frame), dict(coverage or {}), None


def _download_complete_candidate_history(
    service: Any,
    db: Any,
    symbol: str,
    config: Any,
    end_session: Any,
    required_sessions: Any,
) -> tuple[Any, dict[str, Any]]:
    """Download the complete daily Strategy window once and validate it immediately."""
    candidate_config = config.model_copy(
        update={
            "end_date": pd.Timestamp(end_session).date().isoformat(),
            "mongo_cache_enabled": False,
            "market_data_history_backfill_enabled": False,
        }
    )
    credentials = service.get_alpaca_credentials(db)
    start = _utc_timestamp(candidate_config.start_date)
    end = _utc_timestamp(end_session).normalize() + pd.Timedelta(days=1)
    frame = service.download_stock_bars(
        api_key_id=credentials["api_key_id"],
        secret_key=credentials["secret_key"],
        symbol=symbol,
        timeframe=candidate_config.timeframe,
        start=start.to_pydatetime(),
        end=end.to_pydatetime(),
        feed=candidate_config.alpaca_historical_feed,
        adjustment=candidate_config.alpaca_adjustment,
    )
    try:
        cleaned = service.validate_and_clean_bars(frame, candidate_config)
    except ValueError as exc:
        raise RuntimeError("insufficient_history") from exc
    coverage = service._history_coverage_against_baseline(
        symbol,
        cleaned,
        candidate_config,
        required_sessions,
    )
    return cleaned, coverage


def _cached_history_coverage(
    service: Any,
    db: Any,
    symbol: str,
    config: Any,
    end_session: Any,
    required_sessions: Any,
) -> tuple[Any, dict[str, Any]]:
    original = _ORIGINAL_CANDIDATE_HISTORY_COVERAGE
    if original is None:
        raise RuntimeError("Candidate history coverage is not installed.")

    key = _cache_key(symbol, config, end_session)
    frame, coverage, failure = _cache_lookup(key)
    if failure:
        raise RuntimeError(failure)
    if frame is not None and coverage is not None:
        return frame, coverage

    if str(getattr(config, "timeframe", "")) == "1Day":
        try:
            frame, coverage = _download_complete_candidate_history(
                service,
                db,
                symbol,
                config,
                end_session,
                required_sessions,
            )
        except Exception as exc:
            _cache_failure(key, str(exc).strip().lower())
            raise
    else:
        try:
            frame, coverage = original(db, symbol, config, end_session, required_sessions)
        except Exception as exc:
            _cache_failure(key, str(exc).strip().lower())
            raise

    _cache_success(key, frame, dict(coverage or {}))
    return frame, dict(coverage or {})


def _protected_market_history_symbols(service: Any, db: Any) -> set[str]:
    current_config, _current_strategy = service.get_research_strategy_context(db)
    winner_config, _winner_strategy = service.get_trader_winner_context(db)
    return {
        str(symbol or "").strip().upper()
        for symbol in [*(current_config.assets or []), *(winner_config.assets or [])]
        if str(symbol or "").strip()
    }


def _purge_unprotected_market_history(service: Any, db: Any, protected: set[str]) -> dict[str, Any]:
    collection = db[service.ALPACA_MARKET_BARS_COLLECTION]
    existing = {
        str(symbol or "").strip().upper()
        for symbol in collection.distinct("symbol")
        if str(symbol or "").strip()
    }
    stale = sorted(existing - set(protected))
    deleted_rows = 0
    if stale:
        result = collection.delete_many({"symbol": {"$in": stale}})
        deleted_rows = int(getattr(result, "deleted_count", 0) or 0)
    return {
        "protected_symbol_count": len(protected),
        "stale_symbol_count": len(stale),
        "deleted_market_bar_rows": deleted_rows,
        "protected_symbols": sorted(protected),
    }


def _prefetch_complete_histories(
    service: Any,
    db: Any,
    run_id: str,
    symbols: list[str],
    *,
    config: Any,
    end_session: Any,
    required_sessions: Any,
) -> dict[str, Any]:
    normalized = list(dict.fromkeys(str(symbol or "").strip().upper() for symbol in symbols if str(symbol or "").strip()))
    workers = max(1, min(int(service._replay_worker_count()), len(normalized) or 1))
    total = len(normalized)
    completed = 0
    complete_count = 0
    history_rejected_count = 0
    technical_failure_count = 0

    service._event(
        db,
        run_id,
        f"Prefetching the complete Strategy history for {total} external assets using {workers} shared workers.",
        phase="scanning",
        changes={
            "progress_step": "candidate_history_prefetch",
            "stage_progress_percent": 0.0,
            "current_stage": "Downloading complete candidate histories",
            "stage_current": 0,
            "stage_total": total,
            "candidate_history_prefetch": {
                "status": "running",
                "total_count": total,
                "completed_count": 0,
                "complete_count": 0,
                "history_rejected_count": 0,
                "technical_failure_count": 0,
                "workers": workers,
                "cache": "shared_memory",
            },
        },
    )

    def load(symbol: str) -> tuple[str, bool, str | None]:
        key = _cache_key(symbol, config, end_session)
        try:
            frame, coverage = _download_complete_candidate_history(
                service,
                db,
                symbol,
                config,
                end_session,
                required_sessions,
            )
            _cache_success(key, frame, dict(coverage or {}))
            return symbol, True, None
        except Exception as exc:
            reason = str(exc).strip().lower() or "candidate_history_prefetch_failed"
            _cache_failure(key, reason)
            return symbol, False, reason

    if normalized:
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="mct-asset-discovery-history-prefetch",
        ) as executor:
            futures = {executor.submit(load, symbol): symbol for symbol in normalized}
            for future in as_completed(futures):
                _symbol_value, ok, reason = future.result()
                completed += 1
                if ok:
                    complete_count += 1
                elif reason in {"insufficient_history", "discontinuous_history", "ticker_identity_discontinuity"}:
                    history_rejected_count += 1
                else:
                    technical_failure_count += 1
                progress = 100.0 * completed / max(1, total)
                service._set_stage_progress(
                    db,
                    run_id,
                    step="candidate_history_prefetch",
                    percent=progress,
                    label="Downloading complete candidate histories",
                    current=completed,
                    total=total,
                )
                db[service.COLLECTION].update_one(
                    {"_id": service.CURRENT_ID, "run_id": run_id},
                    {"$set": {
                        "candidate_history_prefetch.completed_count": completed,
                        "candidate_history_prefetch.complete_count": complete_count,
                        "candidate_history_prefetch.history_rejected_count": history_rejected_count,
                        "candidate_history_prefetch.technical_failure_count": technical_failure_count,
                        "updated_at": service.utc_now(),
                    }},
                )

    summary = {
        "status": "completed",
        "total_count": total,
        "completed_count": completed,
        "complete_count": complete_count,
        "history_rejected_count": history_rejected_count,
        "technical_failure_count": technical_failure_count,
        "workers": workers,
        "cache": "shared_memory",
        "download_policy": "one_complete_series_per_sampled_asset_before_scoring",
    }
    db[service.COLLECTION].update_one(
        {"_id": service.CURRENT_ID, "run_id": run_id},
        {"$set": {"candidate_history_prefetch": service.bson_value(summary), "updated_at": service.utc_now()}},
    )
    service._event(
        db,
        run_id,
        (
            f"Candidate history prefetch completed: {complete_count}/{total} complete histories cached; "
            f"{history_rejected_count} historical rejections and {technical_failure_count} technical failures."
        ),
        phase="scanning",
    )
    return summary


def _validate_shortlist_history(
    service: Any,
    db: Any,
    run_id: str,
    shortlist: list[dict[str, Any]],
    *,
    config: Any,
    end_session: Any,
    required_sessions: Any,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    if not shortlist:
        return [], {}, []

    workers = max(1, min(int(service._replay_worker_count()), len(shortlist)))
    coverage_by_symbol: dict[str, dict[str, Any]] = {}
    total = len(shortlist)
    completed = 0

    service._event(
        db,
        run_id,
        f"Confirming complete Strategy history from the prefetched cache for {total} predictive candidates.",
        phase="predictive_selection",
        changes={
            "progress_step": "predictive_selection",
            "stage_progress_percent": 0.0,
            "current_stage": "Confirming cached Strategy histories",
            "stage_current": 0,
            "stage_total": total,
        },
    )

    def validate(item: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        symbol = _symbol(item)
        if not symbol:
            return "", {"history_window_complete": False, "reason": "missing_symbol"}
        try:
            _frame, coverage = service._candidate_history_coverage(
                db,
                symbol,
                config,
                end_session,
                required_sessions,
            )
            return symbol, dict(coverage or {})
        except Exception as exc:
            return symbol, {
                "history_window_complete": False,
                "reason": str(exc)[:300],
            }

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mct-predictive-history") as executor:
        futures = {executor.submit(validate, item): item for item in shortlist}
        for future in as_completed(futures):
            symbol, coverage = future.result()
            completed += 1
            if symbol:
                coverage_by_symbol[symbol] = coverage
            service._set_stage_progress(
                db,
                run_id,
                step="predictive_selection",
                percent=(100.0 * completed / max(1, total)),
                label="Confirming cached Strategy histories",
                current=completed,
                total=total,
            )

    retained: list[dict[str, Any]] = []
    rejected: list[str] = []
    for item in shortlist:
        symbol = _symbol(item)
        coverage = coverage_by_symbol.get(symbol) or {"history_window_complete": False}
        if not _coverage_ok(coverage):
            if symbol:
                rejected.append(symbol)
            continue
        row = dict(item)
        row.update(coverage)
        row["history_window_complete"] = True
        row["historical_integrity"] = {
            "status": "passed",
            "policy_version": POLICY_VERSION,
        }
        retained.append(row)
    return retained, coverage_by_symbol, rejected


def _repair_current_campaign_from_validation(service: Any, db: Any, payload: dict[str, Any]) -> dict[str, Any]:
    campaign = payload.get("campaign") if isinstance(payload.get("campaign"), dict) else None
    if not campaign or str(campaign.get("discovery_mode") or "").strip().lower() != "predictive_only":
        return payload
    if str(campaign.get("historical_integrity_policy_version") or "") == POLICY_VERSION:
        return payload

    validation = campaign.get("full_strategy_validation") if isinstance(campaign.get("full_strategy_validation"), dict) else {}
    coverage = validation.get("history_coverage") if isinstance(validation.get("history_coverage"), dict) else {}
    if not coverage:
        return payload

    results = [dict(item) for item in list(campaign.get("results") or []) if isinstance(item, dict)]
    if not results:
        return payload

    retained: list[dict[str, Any]] = []
    rejected: list[str] = []
    for item in results:
        symbol = _symbol(item)
        item_coverage = coverage.get(symbol) if isinstance(coverage.get(symbol), dict) else {}
        if not _coverage_ok(item_coverage):
            if symbol:
                rejected.append(symbol)
            continue
        row = dict(item)
        row.update(item_coverage)
        row["history_window_complete"] = True
        row["historical_integrity"] = {
            "status": "passed",
            "policy_version": POLICY_VERSION,
        }
        retained.append(row)

    now = service.utc_now()
    run_id = str(campaign.get("run_id") or "")
    if rejected:
        db[service.CATALOG_COLLECTION].delete_many({"_id": {"$in": sorted(set(rejected))}})

    update_set: dict[str, Any] = {
        "historical_integrity_policy_version": POLICY_VERSION,
        "historical_integrity_validated_count": len(retained),
        "historical_integrity_rejected_count": len(rejected),
        "updated_at": now,
    }
    update: dict[str, Any] = {"$set": update_set}
    if rejected:
        update_set.update({
            "results": service.bson_value(retained),
            "shortlisted_count": len(retained),
            "full_strategy_validation.status": "invalidated",
            "full_strategy_validation.decision": None,
            "full_strategy_validation.invalidated_reason": "historically_incompatible_candidates_removed",
            "message": (
                f"Removed {len(rejected)} predictive candidates without complete Strategy history. "
                f"{len(retained)} historically compatible candidates remain; validate the remaining selection again."
            ),
        })
        update["$inc"] = {
            "rejected_count": len(rejected),
            "rejection_summary.incomplete_strategy_history": len(rejected),
        }
    db[service.COLLECTION].update_one({"_id": service.CURRENT_ID, "run_id": run_id}, update)

    refreshed = dict(payload)
    refreshed["campaign"] = service._public(service._campaign(db))
    return refreshed


def install_asset_discovery_predictive_history_integrity() -> None:
    global _INSTALLED, _ORIGINAL_MARGINAL_REPLAY, _ORIGINAL_GET_STATUS
    global _ORIGINAL_CANDIDATE_FRAME, _ORIGINAL_CANDIDATE_HISTORY_COVERAGE, _ORIGINAL_IDENTITY_INTEGRITY
    if _INSTALLED:
        return

    from . import asset_discovery as service

    if getattr(service._run_marginal_capital_replay, "_asset_discovery_predictive_history_integrity", False):
        _INSTALLED = True
        return

    _ORIGINAL_MARGINAL_REPLAY = service._run_marginal_capital_replay
    _ORIGINAL_GET_STATUS = service.get_asset_discovery_status
    _ORIGINAL_CANDIDATE_FRAME = service._candidate_frame
    _ORIGINAL_CANDIDATE_HISTORY_COVERAGE = service._candidate_history_coverage
    _ORIGINAL_IDENTITY_INTEGRITY = service._identity_integrity_for_symbols

    def identity_integrity_with_prefetch(
        db: Any,
        symbols: list[str],
        *,
        start_date: str,
        end_date: str,
        asset_metadata: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, dict[str, Any]]:
        original = _ORIGINAL_IDENTITY_INTEGRITY
        if original is None:
            raise RuntimeError("Asset identity integrity is not installed.")
        result = original(
            db,
            symbols,
            start_date=start_date,
            end_date=end_date,
            asset_metadata=asset_metadata,
        )
        if not _automatic_worker():
            return result

        campaign = service._campaign(db) or {}
        run_id = str(campaign.get("run_id") or "").strip()
        config, _strategy = service.get_research_strategy_context(db)
        required_sessions = service._required_xnys_sessions(config, end_date)
        protected = _protected_market_history_symbols(service, db)

        _clear_transient_history_cache()
        cleanup = _purge_unprotected_market_history(service, db, protected)
        db[service.COLLECTION].update_one(
            {"_id": service.CURRENT_ID, "run_id": run_id},
            {"$set": {"market_history_cleanup": service.bson_value(cleanup), "updated_at": service.utc_now()}},
        )
        service._event(
            db,
            run_id,
            (
                f"Market-history cleanup preserved {cleanup['protected_symbol_count']} Strategy/Winner symbols and removed "
                f"{cleanup['deleted_market_bar_rows']} cached rows from {cleanup['stale_symbol_count']} other symbols."
            ),
            phase="scanning",
        )

        eligible = [
            str(symbol or "").strip().upper()
            for symbol in symbols
            if str(symbol or "").strip()
            and str((result.get(str(symbol or "").strip().upper()) or {}).get("status") or "").lower() == "passed"
        ]
        _prefetch_complete_histories(
            service,
            db,
            run_id,
            eligible,
            config=config,
            end_session=end_date,
            required_sessions=required_sessions,
        )
        return result

    def candidate_frame_from_prefetch(db: Any, symbol: str, config: Any, end_session: Any, *args: Any, **kwargs: Any) -> Any:
        key = _cache_key(symbol, config, end_session)
        frame, _coverage, failure = _cache_lookup(key)
        if failure:
            raise RuntimeError(failure)
        if frame is not None:
            return frame
        original = _ORIGINAL_CANDIDATE_FRAME
        if original is None:
            raise RuntimeError("Candidate frame loader is not installed.")
        return original(db, symbol, config, end_session, *args, **kwargs)

    def candidate_history_coverage_cached(db: Any, symbol: str, config: Any, end_session: Any, required_sessions: Any) -> Any:
        return _cached_history_coverage(service, db, symbol, config, end_session, required_sessions)

    def history_filtered_marginal_replay(db: Any, run_id: str, *args: Any, **kwargs: Any) -> Any:
        original = _ORIGINAL_MARGINAL_REPLAY
        if original is None or not _automatic_worker():
            return original(db, run_id, *args, **kwargs) if original is not None else ([], {})

        shortlist = [dict(item) for item in list(kwargs.get("shortlist") or []) if isinstance(item, dict)]
        config = kwargs.get("config")
        end_session = kwargs.get("end_session")
        required_sessions = kwargs.get("required_sessions")
        if not shortlist or config is None or end_session is None or required_sessions is None:
            return original(db, run_id, *args, **kwargs)

        retained, coverage_by_symbol, rejected = _validate_shortlist_history(
            service,
            db,
            run_id,
            shortlist,
            config=config,
            end_session=end_session,
            required_sessions=required_sessions,
        )

        now = service.utc_now()
        update: dict[str, Any] = {
            "$set": {
                "historical_integrity_policy_version": POLICY_VERSION,
                "historical_integrity_validated_count": len(retained),
                "historical_integrity_rejected_count": len(rejected),
                "historical_integrity_coverage": service.bson_value(coverage_by_symbol),
                "updated_at": now,
            }
        }
        if rejected:
            update["$inc"] = {
                "rejected_count": len(rejected),
                "rejection_summary.incomplete_strategy_history": len(rejected),
            }
            db[service.CATALOG_COLLECTION].delete_many({"_id": {"$in": sorted(set(rejected))}})
        db[service.COLLECTION].update_one({"_id": service.CURRENT_ID, "run_id": run_id}, update)

        service._event(
            db,
            run_id,
            (
                f"Predictive historical-integrity validation retained {len(retained)} of {len(shortlist)} candidates from the prefetched cache; "
                f"{len(rejected)} candidates without complete Strategy history were rejected."
            ),
            phase="predictive_selection",
            changes={
                "historical_integrity_policy_version": POLICY_VERSION,
                "historical_integrity_validated_count": len(retained),
                "historical_integrity_rejected_count": len(rejected),
                "validation_candidate_count": len(retained),
                "progress_step": "predictive_selection",
                "stage_progress_percent": 100.0,
                "current_stage": "Complete Strategy history validated from cache",
                "stage_current": len(shortlist),
                "stage_total": len(shortlist),
            },
        )

        next_kwargs = dict(kwargs)
        next_kwargs["shortlist"] = retained
        return original(db, run_id, *args, **next_kwargs)

    def get_status_with_history_integrity(db: Any) -> dict[str, Any]:
        original = _ORIGINAL_GET_STATUS
        payload = dict(original(db)) if original is not None else {}
        payload = _repair_current_campaign_from_validation(service, db, payload)
        policy = dict(payload.get("persistence_policy") or {})
        policy["predictive_history_integrity"] = {
            "version": POLICY_VERSION,
            "rule": "complete_strategy_history_before_persistence",
            "market_history_retention": "selected_strategy_and_winner_only_between_campaigns",
            "candidate_history_download": "one_complete_series_before_candidate_scoring",
            "candidate_history_cache": "shared_memory_reused_by_configured_workers",
            "candidate_history_persistence": "memory_only_until_asset_is_added_to_strategy",
        }
        payload["persistence_policy"] = policy
        return payload

    setattr(identity_integrity_with_prefetch, "_asset_discovery_history_prefetch", True)
    setattr(candidate_frame_from_prefetch, "_asset_discovery_candidate_frame_prefetch", True)
    setattr(candidate_history_coverage_cached, "_asset_discovery_predictive_history_cache", True)
    setattr(history_filtered_marginal_replay, "_asset_discovery_predictive_history_integrity", True)
    service._identity_integrity_for_symbols = identity_integrity_with_prefetch
    service._candidate_frame = candidate_frame_from_prefetch
    service._candidate_history_coverage = candidate_history_coverage_cached
    service._run_marginal_capital_replay = history_filtered_marginal_replay
    service.get_asset_discovery_status = get_status_with_history_integrity
    _INSTALLED = True
