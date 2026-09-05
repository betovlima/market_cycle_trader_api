from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable


POLICY_VERSION = "predictive-full-history-v1"
_AUTO_WORKER_NAME = "asset-discovery-ranker"
_INSTALLED = False
_ORIGINAL_MARGINAL_REPLAY: Callable[..., Any] | None = None
_ORIGINAL_GET_STATUS: Callable[..., dict[str, Any]] | None = None


def _automatic_worker() -> bool:
    return threading.current_thread().name == _AUTO_WORKER_NAME


def _symbol(item: Any) -> str:
    return str((item or {}).get("symbol") or "").strip().upper() if isinstance(item, dict) else ""


def _coverage_ok(coverage: Any) -> bool:
    return isinstance(coverage, dict) and bool(coverage.get("history_window_complete"))


def _validate_shortlist_history(
    service: Any,
    db: Any,
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
            if symbol:
                coverage_by_symbol[symbol] = coverage

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
    if _INSTALLED:
        return

    from . import asset_discovery as service

    if getattr(service._run_marginal_capital_replay, "_asset_discovery_predictive_history_integrity", False):
        _INSTALLED = True
        return

    _ORIGINAL_MARGINAL_REPLAY = service._run_marginal_capital_replay
    _ORIGINAL_GET_STATUS = service.get_asset_discovery_status

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
                f"Predictive historical-integrity validation retained {len(retained)} of {len(shortlist)} market-adherent candidates; "
                f"{len(rejected)} candidates without complete Strategy history were rejected."
            ),
            phase="predictive_selection",
            changes={
                "historical_integrity_policy_version": POLICY_VERSION,
                "historical_integrity_validated_count": len(retained),
                "historical_integrity_rejected_count": len(rejected),
                "validation_candidate_count": len(retained),
                "stage_current": len(retained),
                "stage_total": len(retained),
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
        }
        payload["persistence_policy"] = policy
        return payload

    setattr(history_filtered_marginal_replay, "_asset_discovery_predictive_history_integrity", True)
    service._run_marginal_capital_replay = history_filtered_marginal_replay
    service.get_asset_discovery_status = get_status_with_history_integrity
    _INSTALLED = True
