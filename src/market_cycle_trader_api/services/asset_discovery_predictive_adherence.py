from __future__ import annotations

import threading
from typing import Any, Callable

from . import asset_discovery as discovery


POLICY_VERSION = "market-quality-v1-restored"
MIN_LATEST_CLOSE = 5.0
MIN_MEDIAN_DOLLAR_VOLUME = 10_000_000.0
MIN_NONZERO_VOLUME_RATIO = 0.98
_ADHERENCE_REASONS = frozenset({"price_filter", "liquidity_filter", "volume_quality_filter"})
_AUTOMATIC_WORKER_NAME = "asset-discovery-ranker"
_THREAD_STATE = threading.local()
_INSTALLED = False

_ORIGINAL_SCORE: Callable[..., dict[str, Any]] | None = None
_ORIGINAL_INCREMENT: Callable[..., None] | None = None
_ORIGINAL_ITEM_IS_PERSISTENT: Callable[[Any], bool] | None = None
_ORIGINAL_CATALOG_METRICS: Callable[[dict[str, Any]], dict[str, Any]] | None = None
_ORIGINAL_PERSIST_SHORTLIST: Callable[..., None] | None = None
_ORIGINAL_EVENT: Callable[..., None] | None = None
_ORIGINAL_GET_STATUS: Callable[..., dict[str, Any]] | None = None
_ORIGINAL_GET_CATALOG: Callable[..., dict[str, Any]] | None = None


def _automatic_worker() -> bool:
    return threading.current_thread().name == _AUTOMATIC_WORKER_NAME


def _policy_snapshot() -> dict[str, Any]:
    return {
        "version": POLICY_VERSION,
        "latest_close_min": MIN_LATEST_CLOSE,
        "median_dollar_volume_min": MIN_MEDIAN_DOLLAR_VOLUME,
        "nonzero_volume_ratio_min": MIN_NONZERO_VOLUME_RATIO,
    }


def _adherence_status(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    value = item.get("market_adherence")
    return str((value or {}).get("status") or "").strip().lower() if isinstance(value, dict) else ""


def _predictive_item(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    if str(item.get("candidate_stage") or "").strip().lower() == "predictive":
        return True
    economic = item.get("economic_validation") if isinstance(item.get("economic_validation"), dict) else {}
    return str(economic.get("validation_method") or "").strip().lower() == "predictive_state_screening"


def _quality_reason(quality: dict[str, Any]) -> str | None:
    latest_close = float(quality.get("latest_close") or 0.0)
    median_dollar_volume = float(quality.get("median_dollar_volume") or 0.0)
    nonzero_volume_ratio = float(quality.get("nonzero_volume_ratio") or 0.0)
    if latest_close < MIN_LATEST_CLOSE:
        return "price_filter"
    if median_dollar_volume < MIN_MEDIAN_DOLLAR_VOLUME:
        return "liquidity_filter"
    if nonzero_volume_ratio < MIN_NONZERO_VOLUME_RATIO:
        return "volume_quality_filter"
    return None


def _score_with_adherence(bundle: Any, symbol: str, frame: Any, baseline_returns: Any) -> dict[str, Any]:
    original = _ORIGINAL_SCORE
    if original is None:
        raise RuntimeError("Predictive adherence policy is not installed.")

    result = dict(original(bundle, symbol, frame, baseline_returns))
    quality = discovery.market_quality(frame)
    reason = _quality_reason(quality)
    if reason is not None:
        _THREAD_STATE.rejection_reason = reason
        raise RuntimeError(reason)

    result["market_adherence"] = {
        "status": "passed",
        "policy_version": POLICY_VERSION,
        "latest_close": float(quality.get("latest_close") or 0.0),
        "median_dollar_volume": float(quality.get("median_dollar_volume") or 0.0),
        "nonzero_volume_ratio": float(quality.get("nonzero_volume_ratio") or 0.0),
        "thresholds": _policy_snapshot(),
    }
    return result


def _increment_with_adherence_rejection(db: Any, run_id: str, values: dict[str, int]) -> None:
    original = _ORIGINAL_INCREMENT
    if original is None:
        raise RuntimeError("Predictive adherence counter policy is not installed.")

    reason = str(getattr(_THREAD_STATE, "rejection_reason", "") or "").strip().lower()
    adjusted = dict(values or {})
    if reason in _ADHERENCE_REASONS and int(adjusted.get("technical_failure_count") or 0) > 0:
        adjusted.pop("technical_failure_count", None)
        adjusted["rejected_count"] = int(adjusted.get("rejected_count") or 0) + 1
        key = f"rejection_summary.{reason}"
        adjusted[key] = int(adjusted.get(key) or 0) + 1
        _THREAD_STATE.rejection_reason = ""
    original(db, run_id, adjusted)


def _persistent_with_adherence(item: Any) -> bool:
    original = _ORIGINAL_ITEM_IS_PERSISTENT
    if original is None:
        return False
    if _predictive_item(item) and _adherence_status(item) != "passed":
        return False
    return bool(original(item))


def _catalog_metrics_with_adherence(item: dict[str, Any]) -> dict[str, Any]:
    original = _ORIGINAL_CATALOG_METRICS
    if original is None:
        return {}
    metrics = dict(original(item))
    if isinstance(item.get("market_adherence"), dict):
        metrics["market_adherence"] = item.get("market_adherence")
    return metrics


def _persist_shortlist_with_adherence(db: Any, document: dict[str, Any], results: list[dict[str, Any]]) -> None:
    original = _ORIGINAL_PERSIST_SHORTLIST
    if original is None:
        return
    original(db, document, results)
    now = discovery.utc_now()
    for item in results:
        symbol = str(item.get("symbol") or "").strip().upper()
        adherence = item.get("market_adherence") if isinstance(item.get("market_adherence"), dict) else None
        if not symbol or not adherence:
            continue
        db[discovery.CATALOG_COLLECTION].update_one(
            {"_id": symbol},
            {"$set": {"latest_metrics.market_adherence": discovery.bson_value(adherence), "updated_at": now}},
        )


def _event_with_adherence(
    db: Any,
    run_id: str,
    message: str,
    *,
    phase: str | None = None,
    changes: dict[str, Any] | None = None,
) -> None:
    original = _ORIGINAL_EVENT
    if original is None:
        return
    safe_changes = dict(changes or {})
    if _automatic_worker() and str(message or "").startswith("Fast scan ranked "):
        count = int(safe_changes.get("ranked_count") or safe_changes.get("validation_candidate_count") or 0)
        safe_changes["adherence_validated_count"] = count
        safe_changes["adherence_policy"] = _policy_snapshot()
    original(db, run_id, message, phase=phase, changes=safe_changes or None)


def _invalidate_legacy_predictive_campaign(db: Any, payload: dict[str, Any]) -> dict[str, Any]:
    campaign = payload.get("campaign") if isinstance(payload.get("campaign"), dict) else None
    if not campaign or str(campaign.get("discovery_mode") or "").strip().lower() != "predictive_only":
        return payload
    if isinstance(campaign.get("adherence_policy"), dict):
        return payload

    run_id = str(campaign.get("run_id") or "").strip()
    if not run_id:
        return payload
    now = discovery.utc_now()
    db[discovery.COLLECTION].update_one(
        {"_id": discovery.CURRENT_ID, "run_id": run_id},
        {"$set": {
            "results": [],
            "shortlisted_count": 0,
            "adherence_validated_count": 0,
            "adherence_policy_upgrade_required": True,
            "message": "Predictive market-adherence policy changed. Run a new Asset Discovery campaign before selecting or appending assets.",
            "full_strategy_validation.status": "invalidated",
            "full_strategy_validation.decision": None,
            "full_strategy_validation.invalidated_reason": "predictive_adherence_policy_upgrade",
            "updated_at": now,
        }},
    )
    refreshed = dict(payload)
    refreshed["campaign"] = discovery._public(discovery._campaign(db))
    return refreshed


def _get_status_with_adherence(db: Any) -> dict[str, Any]:
    original = _ORIGINAL_GET_STATUS
    if original is None:
        return {}
    payload = _invalidate_legacy_predictive_campaign(db, dict(original(db)))
    policy = dict(payload.get("persistence_policy") or {})
    policy["market_adherence"] = _policy_snapshot()
    policy["low_adherence_assets"] = "filtered_before_predictive_persistence"
    payload["persistence_policy"] = policy
    return payload


def _get_catalog_with_adherence(db: Any) -> dict[str, Any]:
    original = _ORIGINAL_GET_CATALOG
    if original is None:
        return {"count": 0, "assets": []}

    stale_ids: list[str] = []
    for stored in db[discovery.CATALOG_COLLECTION].find({"latest_metrics.candidate_stage": "predictive"}):
        metrics = stored.get("latest_metrics") if isinstance(stored.get("latest_metrics"), dict) else {}
        adherence = metrics.get("market_adherence") if isinstance(metrics.get("market_adherence"), dict) else {}
        if str(adherence.get("status") or "").strip().lower() != "passed":
            symbol = str(stored.get("_id") or "").strip().upper()
            if symbol:
                stale_ids.append(symbol)
    if stale_ids:
        db[discovery.CATALOG_COLLECTION].delete_many({"_id": {"$in": sorted(set(stale_ids))}})

    payload = dict(original(db))
    policy = dict(payload.get("persistence_policy") or {})
    policy["market_adherence"] = _policy_snapshot()
    policy["low_adherence_assets"] = "not_persisted"
    payload["persistence_policy"] = policy
    return payload


def install_asset_discovery_predictive_adherence() -> None:
    global _INSTALLED
    global _ORIGINAL_SCORE, _ORIGINAL_INCREMENT, _ORIGINAL_ITEM_IS_PERSISTENT
    global _ORIGINAL_CATALOG_METRICS, _ORIGINAL_PERSIST_SHORTLIST, _ORIGINAL_EVENT
    global _ORIGINAL_GET_STATUS, _ORIGINAL_GET_CATALOG

    if _INSTALLED:
        return
    if getattr(discovery._score_candidate, "_asset_discovery_predictive_adherence", False):
        _INSTALLED = True
        return

    _ORIGINAL_SCORE = discovery._score_candidate
    _ORIGINAL_INCREMENT = discovery._increment
    _ORIGINAL_ITEM_IS_PERSISTENT = discovery._item_is_persistent_candidate
    _ORIGINAL_CATALOG_METRICS = discovery._catalog_metrics
    _ORIGINAL_PERSIST_SHORTLIST = discovery._persist_shortlist_to_catalog
    _ORIGINAL_EVENT = discovery._event
    _ORIGINAL_GET_STATUS = discovery.get_asset_discovery_status
    _ORIGINAL_GET_CATALOG = discovery.get_discovery_catalog

    setattr(_score_with_adherence, "_asset_discovery_predictive_adherence", True)
    discovery._score_candidate = _score_with_adherence
    discovery._increment = _increment_with_adherence_rejection
    discovery._item_is_persistent_candidate = _persistent_with_adherence
    discovery._catalog_metrics = _catalog_metrics_with_adherence
    discovery._persist_shortlist_to_catalog = _persist_shortlist_with_adherence
    discovery._event = _event_with_adherence
    discovery.get_asset_discovery_status = _get_status_with_adherence
    discovery.get_discovery_catalog = _get_catalog_with_adherence
    _INSTALLED = True
