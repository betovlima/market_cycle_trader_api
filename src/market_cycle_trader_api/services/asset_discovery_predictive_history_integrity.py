from __future__ import annotations

import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import pandas as pd


POLICY_VERSION = "predictive-full-history-v2-fast-single-request"
_AUTO_WORKER_NAME = "asset-discovery-ranker"
_HISTORY_CACHE_LIMIT = 64
_INSTALLED = False
_ORIGINAL_MARGINAL_REPLAY: Callable[..., Any] | None = None
_ORIGINAL_GET_STATUS: Callable[..., dict[str, Any]] | None = None
_ORIGINAL_CANDIDATE_FRAME: Callable[..., Any] | None = None
_ORIGINAL_CANDIDATE_HISTORY_COVERAGE: Callable[..., Any] | None = None
_HISTORY_CACHE_LOCK = threading.Lock()
_HISTORY_CACHE: OrderedDict[tuple[str, ...], tuple[Any, dict[str, Any]]] = OrderedDict()


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


def _download_complete_candidate_history_fast(
    service: Any,
    db: Any,
    symbol: str,
    config: Any,
    end_session: Any,
    required_sessions: Any,
) -> tuple[Any, dict[str, Any]]:
    """Load a complete daily candidate history in one Alpaca request.

    The generic market-data helper intentionally chunks long ranges for arbitrary
    timeframes. Asset Discovery uses daily bars only; a Strategy window from 2016
    is roughly 2.7k rows, so a single StockBarsRequest is substantially cheaper
    while preserving the exact same coverage validation.
    """
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
    with _HISTORY_CACHE_LOCK:
        cached = _HISTORY_CACHE.get(key)
        if cached is not None:
            _HISTORY_CACHE.move_to_end(key)
            frame, coverage = cached
            try:
                copied_frame = frame.copy()
            except Exception:
                copied_frame = frame
            return copied_frame, dict(coverage)

    if str(getattr(config, "timeframe", "")) == "1Day":
        frame, coverage = _download_complete_candidate_history_fast(
            service,
            db,
            symbol,
            config,
            end_session,
            required_sessions,
        )
    else:
        frame, coverage = original(db, symbol, config, end_session, required_sessions)

    with _HISTORY_CACHE_LOCK:
        try:
            cached_frame = frame.copy()
        except Exception:
            cached_frame = frame
        _HISTORY_CACHE[key] = (cached_frame, dict(coverage or {}))
        _HISTORY_CACHE.move_to_end(key)
        while len(_HISTORY_CACHE) > _HISTORY_CACHE_LIMIT:
            _HISTORY_CACHE.popitem(last=False)
    return frame, dict(coverage or {})


def _probe_history_start(
    service: Any,
    db: Any,
    symbol: str,
    config: Any,
    *,
    credentials: dict[str, str] | None = None,
) -> dict[str, Any]:
    requested = pd.Timestamp(config.start_date)
    if requested.tzinfo is not None:
        requested = requested.tz_convert("UTC").tz_localize(None)
    requested = requested.normalize()
    tolerance_days = int(getattr(config, "market_data_history_start_tolerance_days", 0) or 0)
    latest_allowed = requested + pd.Timedelta(days=tolerance_days)

    auth = credentials or service.get_alpaca_credentials(db)
    start = requested.tz_localize("UTC")
    end = (latest_allowed + pd.Timedelta(days=1)).tz_localize("UTC")
    frame = service.download_stock_bars(
        api_key_id=auth["api_key_id"],
        secret_key=auth["secret_key"],
        symbol=symbol,
        timeframe=config.timeframe,
        start=start.to_pydatetime(),
        end=end.to_pydatetime(),
        feed=config.alpaca_historical_feed,
        adjustment=config.alpaca_adjustment,
    )
    actual = service._normalized_sessions(frame)
    if actual.empty:
        raise RuntimeError("insufficient_history")
    first = pd.Timestamp(actual.min()).normalize()
    if first > latest_allowed:
        raise RuntimeError("insufficient_history")
    return {
        "history_start_probe": "passed",
        "history_required_start": requested.date().isoformat(),
        "history_start_probe_first_session": first.date().isoformat(),
        "history_start_tolerance_days": tolerance_days,
    }


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

    workers = max(1, min(4, len(shortlist)))
    coverage_by_symbol: dict[str, dict[str, Any]] = {}
    total = len(shortlist)
    completed = 0

    service._event(
        db,
        run_id,
        f"Validating complete Strategy history for {total} predictive candidates.",
        phase="predictive_selection",
        changes={
            "progress_step": "predictive_selection",
            "stage_progress_percent": 0.0,
            "current_stage": "Validating complete Strategy history",
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
                label="Validating complete Strategy history",
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
    global _ORIGINAL_CANDIDATE_FRAME, _ORIGINAL_CANDIDATE_HISTORY_COVERAGE
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

    def candidate_frame_with_start_probe(db: Any, symbol: str, config: Any, end_session: Any, *args: Any, **kwargs: Any) -> Any:
        original = _ORIGINAL_CANDIDATE_FRAME
        if original is None:
            raise RuntimeError("Candidate frame loader is not installed.")
        if _automatic_worker():
            try:
                _probe_history_start(
                    service,
                    db,
                    symbol,
                    config,
                    credentials=kwargs.get("credentials") if isinstance(kwargs.get("credentials"), dict) else None,
                )
                db[service.COLLECTION].update_one(
                    {"_id": service.CURRENT_ID},
                    {"$inc": {"history_start_probe_passed_count": 1}, "$set": {"updated_at": service.utc_now()}},
                )
            except RuntimeError as exc:
                if str(exc).strip().lower() == "insufficient_history":
                    db[service.COLLECTION].update_one(
                        {"_id": service.CURRENT_ID},
                        {"$inc": {"history_start_probe_rejected_count": 1}, "$set": {"updated_at": service.utc_now()}},
                    )
                raise
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
                f"Predictive historical-integrity validation retained {len(retained)} of {len(shortlist)} candidates; "
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
                "current_stage": "Complete Strategy history validated",
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
            "fast_start_probe": True,
            "daily_full_history_download": "single_alpaca_request_per_candidate",
            "full_history_cache": "exact_snapshot_in_process",
        }
        payload["persistence_policy"] = policy
        return payload

    setattr(candidate_frame_with_start_probe, "_asset_discovery_predictive_history_start_probe", True)
    setattr(candidate_history_coverage_cached, "_asset_discovery_predictive_history_cache", True)
    setattr(history_filtered_marginal_replay, "_asset_discovery_predictive_history_integrity", True)
    service._candidate_frame = candidate_frame_with_start_probe
    service._candidate_history_coverage = candidate_history_coverage_cached
    service._run_marginal_capital_replay = history_filtered_marginal_replay
    service.get_asset_discovery_status = get_status_with_history_integrity
    _INSTALLED = True
