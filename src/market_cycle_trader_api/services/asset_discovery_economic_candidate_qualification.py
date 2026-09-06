from __future__ import annotations

import threading
from typing import Any, Callable


_AUTO_WORKER_NAME = "asset-discovery-ranker"
_VALIDATION_METHOD = "full_strategy_history_replay"
_POLICY_VERSION = "predictive-economic-qualification-v1"

_BASE_MARGINAL_REPLAY: Callable[..., Any] | None = None
_INSTALLED = False


def _automatic_worker() -> bool:
    return threading.current_thread().name == _AUTO_WORKER_NAME


def capture_asset_discovery_base_marginal_replay() -> None:
    """Capture the original full-history marginal replay before predictive wrappers replace it."""
    global _BASE_MARGINAL_REPLAY
    if _BASE_MARGINAL_REPLAY is not None:
        return
    from . import asset_discovery as service

    _BASE_MARGINAL_REPLAY = service._run_marginal_capital_replay


def _predictive_item(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    if isinstance(item.get("discovery_selection"), dict):
        return True
    return str(item.get("candidate_stage") or "").strip().lower() in {
        "predictive",
        "economic_qualified",
    }


def _economically_persistent(service: Any, item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    replay = item.get("marginal_replay") if isinstance(item.get("marginal_replay"), dict) else item
    if not service._marginal_replay_is_persistent_candidate(replay):
        return False
    adherence = item.get("market_adherence") if isinstance(item.get("market_adherence"), dict) else {}
    if adherence and str(adherence.get("status") or "").strip().lower() != "passed":
        return False
    return True


def _economic_validation_snapshot(item: dict[str, Any]) -> dict[str, Any]:
    replay = item.get("marginal_replay") if isinstance(item.get("marginal_replay"), dict) else {}
    return {
        "status": "completed",
        "automatic": True,
        "validation_method": _VALIDATION_METHOD,
        "ending_capital_delta": replay.get("ending_capital_delta"),
        "ending_capital_delta_rate": replay.get("ending_capital_delta_rate"),
        "candidate_ending_capital": (
            (replay.get("candidate") or {}).get("ending_capital")
            if isinstance(replay.get("candidate"), dict)
            else None
        ),
    }


def install_asset_discovery_economic_candidate_qualification() -> None:
    """Restore the pre-predictive invariant: visible candidates must increase full-history Strategy capital."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import asset_discovery as service

    if _BASE_MARGINAL_REPLAY is None:
        raise RuntimeError("Asset Discovery base marginal replay was not captured before predictive wrappers.")

    filtered_predictive_replay = service._run_marginal_capital_replay
    original_item_is_persistent = service._item_is_persistent_candidate
    original_persist_shortlist = service._persist_shortlist_to_catalog
    original_event = service._event
    original_finish = service._finish
    original_get_status = service.get_asset_discovery_status
    original_get_catalog = service.get_discovery_catalog

    if getattr(filtered_predictive_replay, "_asset_discovery_economic_candidate_qualification", False):
        _INSTALLED = True
        return

    def qualified_marginal_replay(
        db: Any,
        run_id: str,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not _automatic_worker():
            return filtered_predictive_replay(db, run_id, *args, **kwargs)

        # Keep every fast predictive optimization and every historical/adherence filter.
        # The wrapped predictive pipeline returns only the post-filter shortlist here.
        predictive_rows, predictive_summary = filtered_predictive_replay(
            db,
            run_id,
            *args,
            **kwargs,
        )
        shortlist = [dict(item) for item in list(predictive_rows or []) if isinstance(item, dict)]

        if not shortlist:
            summary = dict(predictive_summary or {})
            summary.update({
                "status": "completed",
                "automatic": True,
                "validation_method": _VALIDATION_METHOD,
                "total_count": 0,
                "completed_count": 0,
                "persistent_candidate_count": 0,
                "predictive_candidate_count": 0,
                "reason": "no_historically_compatible_predictive_candidate",
                "policy_version": _POLICY_VERSION,
            })
            db[service.COLLECTION].update_one(
                {"_id": service.CURRENT_ID, "run_id": run_id},
                {"$set": {
                    "automatic_marginal_replay": True,
                    "economic_candidate_qualification_policy": _POLICY_VERSION,
                    "marginal_replay": service.bson_value(summary),
                    "updated_at": service.utc_now(),
                }},
            )
            return [], summary

        service._event(
            db,
            run_id,
            f"Starting automatic full-history economic qualification for {len(shortlist)} predictive candidates.",
            phase="marginal_replay",
            changes={
                "progress_step": "marginal_replay",
                "stage_progress_percent": 0.0,
                "stage_current": 0,
                "stage_total": len(shortlist),
                "economic_candidate_qualification_policy": _POLICY_VERSION,
            },
        )

        replay_kwargs = dict(kwargs)
        replay_kwargs["shortlist"] = shortlist
        retained, summary = _BASE_MARGINAL_REPLAY(
            db,
            run_id,
            *args,
            **replay_kwargs,
        )

        qualified: list[dict[str, Any]] = []
        for item in list(retained or []):
            if not isinstance(item, dict) or not _economically_persistent(service, item):
                continue
            row = dict(item)
            row["candidate_stage"] = "economic_qualified"
            row["economic_validation"] = _economic_validation_snapshot(row)
            qualified.append(row)

        final_summary = dict(summary or {})
        final_summary.update({
            "status": "completed",
            "automatic": True,
            "validation_method": _VALIDATION_METHOD,
            "predictive_candidate_count": len(shortlist),
            "persistent_candidate_count": len(qualified),
            "reason": "automatic_post_predictive_economic_qualification",
            "policy_version": _POLICY_VERSION,
        })

        db[service.COLLECTION].update_one(
            {"_id": service.CURRENT_ID, "run_id": run_id},
            {"$set": {
                "automatic_marginal_replay": True,
                "economic_candidate_qualification_policy": _POLICY_VERSION,
                "marginal_replay": service.bson_value(final_summary),
                "updated_at": service.utc_now(),
            }},
        )
        return qualified, final_summary

    def strict_item_is_persistent(item: Any) -> bool:
        if _predictive_item(item):
            return _economically_persistent(service, item)
        return bool(original_item_is_persistent(item))

    def persist_qualified_shortlist(db: Any, document: dict[str, Any], results: list[dict[str, Any]]) -> None:
        qualified = [item for item in results if strict_item_is_persistent(item)]
        original_persist_shortlist(db, document, qualified)
        now = service.utc_now()
        for item in qualified:
            symbol = str(item.get("symbol") or "").strip().upper()
            if not symbol:
                continue
            db[service.CATALOG_COLLECTION].update_one(
                {"_id": symbol},
                {"$set": {
                    "validation_mode": _VALIDATION_METHOD,
                    "economic_validation_status": "completed",
                    "history_window_complete": True,
                    "latest_metrics.candidate_stage": "economic_qualified",
                    "latest_metrics.economic_validation": service.bson_value(_economic_validation_snapshot(item)),
                    "updated_at": now,
                }},
            )

    def event_with_economic_qualification(
        db: Any,
        run_id: str,
        message: str,
        *,
        phase: str | None = None,
        changes: dict[str, Any] | None = None,
    ) -> None:
        text = str(message or "")
        safe_phase = phase
        safe_changes = dict(changes or {})
        if text.startswith("Fast scan ranked "):
            count = int(safe_changes.get("ranked_count") or safe_changes.get("validation_candidate_count") or 0)
            text = (
                f"Predictive scan ranked {count} candidates. "
                "Historical-integrity filtering and full-history economic qualification follow."
            )
            safe_phase = "predictive_selection"
        elif text.startswith("Predictive candidate ranking completed without automatic economic contribution replay"):
            text = "Predictive candidate ranking completed. Full-history economic qualification follows."
        elif text.startswith("Parallel full-history replay completed"):
            text = "Automatic full-history economic candidate qualification completed."
        original_event(db, run_id, text, phase=safe_phase, changes=safe_changes or None)

    def finish_with_economic_qualification(
        db: Any,
        run_id: str,
        status: str,
        message: str,
        *,
        results: list[dict[str, Any]] | None = None,
    ) -> None:
        qualified = [item for item in list(results or []) if strict_item_is_persistent(item)] if results is not None else None
        original_finish(db, run_id, status, message, results=qualified)
        if _automatic_worker() and status == "completed" and qualified is not None:
            db[service.COLLECTION].update_one(
                {"_id": service.CURRENT_ID, "run_id": run_id},
                {"$set": {
                    "message": (
                        f"Asset Discovery completed with {len(qualified)} candidates that increased "
                        "full-history Strategy capital individually."
                    ),
                    "results": service.bson_value(qualified),
                    "shortlisted_count": len(qualified),
                    "automatic_marginal_replay": True,
                    "economic_candidate_qualification_policy": _POLICY_VERSION,
                    "updated_at": service.utc_now(),
                }},
            )

    def status_with_economic_qualification(db: Any) -> dict[str, Any]:
        payload = dict(original_get_status(db))
        policy = dict(payload.get("persistence_policy") or {})
        policy.update({
            "marginal_replay": "automatic_after_predictive_and_history_filters",
            "discovery_catalog": "economically_qualified_candidates_only",
            "candidate_test_details": _VALIDATION_METHOD,
            "selection_policy": (
                "predictive_discovery_then_positive_individual_full_history_capital_qualification_"
                "then_selected_universe_validation"
            ),
            "economic_contribution_replay_automatic": True,
            "full_history_capital_lift_required": True,
            "economic_candidate_qualification_policy": _POLICY_VERSION,
        })
        payload["persistence_policy"] = policy
        return payload

    def catalog_with_economic_qualification(db: Any) -> dict[str, Any]:
        payload = dict(original_get_catalog(db))
        policy = dict(payload.get("persistence_policy") or {})
        policy.update({
            "validation_method": _VALIDATION_METHOD,
            "full_history_capital_lift_required": True,
            "economic_contribution_replay_automatic": True,
            "economic_candidate_qualification_policy": _POLICY_VERSION,
        })
        payload["persistence_policy"] = policy
        return payload

    setattr(qualified_marginal_replay, "_asset_discovery_economic_candidate_qualification", True)
    service._run_marginal_capital_replay = qualified_marginal_replay
    service._item_is_persistent_candidate = strict_item_is_persistent
    service._persist_shortlist_to_catalog = persist_qualified_shortlist
    service._event = event_with_economic_qualification
    service._finish = finish_with_economic_qualification
    service.get_asset_discovery_status = status_with_economic_qualification
    service.get_discovery_catalog = catalog_with_economic_qualification
    _INSTALLED = True
