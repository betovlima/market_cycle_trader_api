from __future__ import annotations

import threading
from typing import Any

import numpy as np
import pandas as pd


_INSTALLED = False
_AUTO_WORKER_NAME = "asset-discovery-ranker"
_PREDICTIVE_VALIDATION_METHOD = "predictive_state_screening"
_STATE_SIMILARITY_METHOD = "lightgbm_leaf_agreement"


def _automatic_discovery_worker() -> bool:
    return threading.current_thread().name == _AUTO_WORKER_NAME


def _predictive_candidate(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    selection = item.get("discovery_selection") if isinstance(item.get("discovery_selection"), dict) else {}
    identity = item.get("identity_integrity") if isinstance(item.get("identity_integrity"), dict) else {}
    score = selection.get("raw_score") if selection else item.get("raw_score")
    try:
        finite_score = score is not None and bool(np.isfinite(float(score)))
    except (TypeError, ValueError):
        finite_score = False
    return bool(selection.get("available")) and finite_score and str(identity.get("status") or "").lower() == "passed"


def _leaf_vector(model: Any, vector: pd.DataFrame) -> np.ndarray | None:
    try:
        leaves = np.asarray(model.predict(vector, pred_leaf=True))
    except Exception:
        return None
    if leaves.size == 0:
        return None
    return leaves.reshape(-1)


def _attach_baseline_leaf_signatures(service: Any, bundle: Any, frames: dict[str, pd.DataFrame]) -> None:
    signatures: dict[str, np.ndarray] = {}
    for symbol, frame in frames.items():
        try:
            row, _stamp = service.latest_feature_snapshot(frame)
            vector = pd.DataFrame(
                [[float(row[column]) for column in service.FEATURE_COLUMNS]],
                columns=list(service.FEATURE_COLUMNS),
            )
            leaves = _leaf_vector(bundle.model, vector)
            if leaves is not None:
                signatures[str(symbol).upper()] = leaves
        except Exception:
            continue
    try:
        setattr(bundle.model, "_asset_discovery_baseline_leaf_signatures", signatures)
    except Exception:
        pass


def _state_novelty(service: Any, bundle: Any, frame: pd.DataFrame) -> dict[str, Any]:
    signatures = getattr(bundle.model, "_asset_discovery_baseline_leaf_signatures", None)
    if not isinstance(signatures, dict) or not signatures:
        return {
            "state_novelty_score": None,
            "state_max_similarity": None,
            "state_nearest_baseline_symbol": None,
            "state_similarity_method": _STATE_SIMILARITY_METHOD,
            "state_leaf_tree_count": None,
        }
    try:
        row, _stamp = service.latest_feature_snapshot(frame)
        vector = pd.DataFrame(
            [[float(row[column]) for column in service.FEATURE_COLUMNS]],
            columns=list(service.FEATURE_COLUMNS),
        )
    except Exception:
        return {
            "state_novelty_score": None,
            "state_max_similarity": None,
            "state_nearest_baseline_symbol": None,
            "state_similarity_method": _STATE_SIMILARITY_METHOD,
            "state_leaf_tree_count": None,
        }
    candidate = _leaf_vector(bundle.model, vector)
    if candidate is None:
        return {
            "state_novelty_score": None,
            "state_max_similarity": None,
            "state_nearest_baseline_symbol": None,
            "state_similarity_method": _STATE_SIMILARITY_METHOD,
            "state_leaf_tree_count": None,
        }

    nearest_symbol: str | None = None
    maximum_similarity: float | None = None
    for symbol, baseline in signatures.items():
        baseline_array = np.asarray(baseline).reshape(-1)
        if len(baseline_array) != len(candidate) or not len(candidate):
            continue
        similarity = float(np.mean(candidate == baseline_array))
        if maximum_similarity is None or similarity > maximum_similarity:
            maximum_similarity = similarity
            nearest_symbol = str(symbol)

    return {
        "state_novelty_score": None if maximum_similarity is None else float(1.0 - maximum_similarity),
        "state_max_similarity": maximum_similarity,
        "state_nearest_baseline_symbol": nearest_symbol,
        "state_similarity_method": _STATE_SIMILARITY_METHOD,
        "state_leaf_tree_count": int(len(candidate)),
    }


def _configured_symbols(payload: dict[str, Any]) -> set[str]:
    baseline = payload.get("baseline") if isinstance(payload.get("baseline"), dict) else {}
    return {
        str(symbol or "").strip().upper()
        for symbol in list(baseline.get("assets") or [])
        if str(symbol or "").strip()
    }


def install_asset_discovery_predictive_policy() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from . import asset_discovery as service

    original_train_ranker = service.train_ranker
    original_score_candidate = service._score_candidate
    original_item_is_persistent_candidate = service._item_is_persistent_candidate
    original_marginal_replay = service._run_marginal_capital_replay
    original_catalog_metrics = service._catalog_metrics
    original_persist_shortlist = service._persist_shortlist_to_catalog
    original_event = service._event
    original_finish = service._finish
    original_get_status = service.get_asset_discovery_status
    original_get_catalog = service.get_discovery_catalog

    if getattr(original_marginal_replay, "_asset_discovery_predictive_mode", False):
        _INSTALLED = True
        return

    def predictive_train_ranker(frames: dict[str, pd.DataFrame], *args: Any, **kwargs: Any) -> Any:
        bundle = original_train_ranker(frames, *args, **kwargs)
        _attach_baseline_leaf_signatures(service, bundle, frames)
        return bundle

    def predictive_score_candidate(bundle: Any, symbol: str, frame: pd.DataFrame, baseline_returns: pd.DataFrame) -> dict[str, Any]:
        result = dict(original_score_candidate(bundle, symbol, frame, baseline_returns))
        result.update(_state_novelty(service, bundle, frame))
        return result

    def predictive_item_is_persistent_candidate(item: Any) -> bool:
        if original_item_is_persistent_candidate(item):
            return True
        return _predictive_candidate(item)

    def predictive_marginal_replay(
        db: Any,
        run_id: str,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not _automatic_discovery_worker():
            return original_marginal_replay(db, run_id, *args, **kwargs)

        shortlist = [dict(item) for item in list(kwargs.get("shortlist") or []) if isinstance(item, dict)]
        annotated: list[dict[str, Any]] = []
        replay_rows: list[dict[str, Any]] = []
        for item in shortlist:
            row = dict(item)
            row["persistence_eligible"] = True
            row["candidate_stage"] = "predictive"
            row["economic_validation"] = {
                "status": "not_run",
                "automatic": False,
                "validation_method": _PREDICTIVE_VALIDATION_METHOD,
            }
            annotated.append(row)
            replay_rows.append({
                "symbol": row.get("symbol"),
                "status": "completed",
                "persistence_eligible": True,
                "validation_method": _PREDICTIVE_VALIDATION_METHOD,
                "economic_validation": "not_run",
            })

        summary = {
            "status": "not_run",
            "automatic": False,
            "validation_method": _PREDICTIVE_VALIDATION_METHOD,
            "total_count": len(annotated),
            "completed_count": 0,
            "current_symbol": None,
            "current_index": 0,
            "current_stage": "Economic contribution replay not run automatically",
            "progress_percent": 0.0,
            "baseline": None,
            "results": replay_rows,
            "predictive_candidate_count": len(annotated),
            "persistent_candidate_count": None,
            "reason": "disabled_for_fast_predictive_discovery",
        }
        db[service.COLLECTION].update_one(
            {"_id": service.CURRENT_ID, "run_id": run_id},
            {"$set": {
                "discovery_mode": "predictive_only",
                "automatic_marginal_replay": False,
                "marginal_replay": service.bson_value(summary),
                "updated_at": service.utc_now(),
            }},
        )
        return annotated, summary

    def predictive_catalog_metrics(item: dict[str, Any]) -> dict[str, Any]:
        metrics = dict(original_catalog_metrics(item))
        for key in (
            "state_novelty_score",
            "state_max_similarity",
            "state_nearest_baseline_symbol",
            "state_similarity_method",
            "state_leaf_tree_count",
            "candidate_stage",
            "economic_validation",
        ):
            if key in item:
                metrics[key] = item.get(key)
        return metrics

    def predictive_persist_shortlist(db: Any, document: dict[str, Any], results: list[dict[str, Any]]) -> None:
        original_persist_shortlist(db, document, results)
        now = service.utc_now()
        for item in results:
            if not _predictive_candidate(item):
                continue
            symbol = str(item.get("symbol") or "").strip().upper()
            if not symbol:
                continue
            db[service.CATALOG_COLLECTION].update_one(
                {"_id": symbol},
                {"$set": {
                    "validation_mode": _PREDICTIVE_VALIDATION_METHOD,
                    "economic_validation_status": str(((item.get("economic_validation") or {}).get("status") or "not_run")),
                    "history_window_complete": None,
                    "updated_at": now,
                }},
            )

    def predictive_event(
        db: Any,
        run_id: str,
        message: str,
        *,
        phase: str | None = None,
        changes: dict[str, Any] | None = None,
    ) -> None:
        safe_message = str(message or "")
        safe_phase = phase
        safe_changes = dict(changes or {})
        if _automatic_discovery_worker() and safe_message.startswith("Fast scan ranked "):
            count = safe_changes.get("ranked_count") or safe_changes.get("validation_candidate_count") or 0
            safe_message = (
                f"Fast scan ranked {count} predictive candidates. "
                "Automatic full-history economic contribution replay is disabled."
            )
            safe_phase = "predictive_selection"
            safe_changes["progress_step"] = "predictive_selection"
            safe_changes["stage_progress_percent"] = 100.0
            safe_changes["current_stage"] = "Finalizing predictive candidates"
            safe_changes["stage_current"] = int(count or 0)
            safe_changes["stage_total"] = int(count or 0)
        elif _automatic_discovery_worker() and safe_message.startswith("Parallel full-history replay completed"):
            safe_message = "Predictive candidate ranking completed without automatic economic contribution replay."
        original_event(db, run_id, safe_message, phase=safe_phase, changes=safe_changes or None)

    def predictive_finish(
        db: Any,
        run_id: str,
        status: str,
        message: str,
        *,
        results: list[dict[str, Any]] | None = None,
    ) -> None:
        safe_message = str(message or "")
        if _automatic_discovery_worker() and status == "completed" and results is not None:
            safe_message = (
                f"Asset Discovery completed with {len(results)} predictive candidates. "
                "Economic contribution over the complete Strategy history was not run automatically."
            )
            db[service.COLLECTION].update_one(
                {"_id": service.CURRENT_ID, "run_id": run_id},
                {"$set": {
                    "discovery_mode": "predictive_only",
                    "automatic_marginal_replay": False,
                    "updated_at": service.utc_now(),
                }},
            )
        original_finish(db, run_id, status, safe_message, results=results)

    def predictive_get_status(db: Any) -> dict[str, Any]:
        payload = dict(original_get_status(db))
        configured = _configured_symbols(payload)
        campaign = payload.get("campaign") if isinstance(payload.get("campaign"), dict) else None
        if campaign is not None and configured:
            campaign = dict(campaign)
            campaign_results = [
                item for item in list(campaign.get("results") or [])
                if str((item or {}).get("symbol") or "").strip().upper() not in configured
            ]
            campaign["results"] = campaign_results
            campaign["shortlisted_count"] = len(campaign_results)
            payload["campaign"] = campaign
        policy = dict(payload.get("persistence_policy") or {})
        policy.update({
            "marginal_replay": "manual_only_after_predictive_discovery",
            "discovery_catalog": "predictive_candidates_and_explicitly_validated_candidates",
            "low_adherence_assets": "not_applicable_until_optional_economic_replay",
            "candidate_test_details": "predictive_state_screening",
            "selection_policy": "predictive_discovery_then_selected_universe_full_strategy_validation",
            "economic_contribution_replay_automatic": False,
        })
        payload["persistence_policy"] = policy
        return payload

    def predictive_get_catalog(db: Any) -> dict[str, Any]:
        try:
            current_config, _current_strategy = service.get_research_strategy_context(db)
            configured = {
                str(symbol or "").strip().upper()
                for symbol in list(current_config.assets or [])
                if str(symbol or "").strip()
            }
        except Exception:
            configured = set()
        if configured:
            db[service.CATALOG_COLLECTION].delete_many({"_id": {"$in": sorted(configured)}})
        payload = dict(original_get_catalog(db))
        if configured:
            assets = [
                item for item in list(payload.get("assets") or [])
                if str((item or {}).get("symbol") or "").strip().upper() not in configured
            ]
            payload["assets"] = assets
            payload["count"] = len(assets)
        policy = dict(payload.get("persistence_policy") or {})
        policy.update({
            "validation_method": _PREDICTIVE_VALIDATION_METHOD,
            "full_history_capital_lift_required": False,
            "economic_contribution_replay_automatic": False,
            "configured_strategy_assets_excluded": True,
        })
        payload["persistence_policy"] = policy
        return payload

    setattr(predictive_marginal_replay, "_asset_discovery_predictive_mode", True)
    service.train_ranker = predictive_train_ranker
    service._score_candidate = predictive_score_candidate
    service._item_is_persistent_candidate = predictive_item_is_persistent_candidate
    service._run_marginal_capital_replay = predictive_marginal_replay
    service._catalog_metrics = predictive_catalog_metrics
    service._persist_shortlist_to_catalog = predictive_persist_shortlist
    service._event = predictive_event
    service._finish = predictive_finish
    service.get_asset_discovery_status = predictive_get_status
    service.get_discovery_catalog = predictive_get_catalog
    _INSTALLED = True
