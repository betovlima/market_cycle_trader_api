from __future__ import annotations

from typing import Any

import pandas as pd

from ..infrastructure.persistence.mongo_repository import JOBS_COLLECTION


_INSTALLED = False


def _normalized_assets(values: Any) -> list[str]:
    return [
        str(value or "").strip().upper()
        for value in list(values or [])
        if str(value or "").strip()
    ]


def _iso_date(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp):
            return ""
        return stamp.date().isoformat()
    except Exception:
        return str(value).strip()


def _source_id(strategy: dict[str, Any]) -> str:
    return str(strategy.get("_id") or strategy.get("id") or "").strip()


def _certified_job(
    service: Any,
    db: Any,
    strategy: dict[str, Any],
    evaluation_end: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_id = _source_id(strategy)
    source_revision = int(strategy.get("revision") or 1)
    source_hash = str(strategy.get("configuration_hash") or "").strip()
    last_backtest_id = str(strategy.get("last_backtest_id") or "").strip()

    if not source_id or not source_hash or not last_backtest_id:
        raise RuntimeError(
            "Asset Discovery requires an exact completed Strategy backtest snapshot for the current revision before research can start."
        )
    if str(strategy.get("last_backtest_status") or "").strip().lower() != "completed":
        raise RuntimeError("Asset Discovery Strategy backtest snapshot is not completed.")
    if int(strategy.get("last_backtest_revision") or 0) != source_revision:
        raise RuntimeError(
            "Asset Discovery Strategy backtest revision does not match the current Strategy revision."
        )

    job = db[JOBS_COLLECTION].find_one({"id": last_backtest_id}) or {}
    if str(job.get("status") or "").strip().lower() != "completed":
        raise RuntimeError("Asset Discovery could not load the completed Strategy backtest snapshot.")
    if str(job.get("strategy_profile_id") or "").strip() != source_id:
        raise RuntimeError("Asset Discovery backtest snapshot belongs to another Strategy.")
    if int(job.get("strategy_profile_revision") or 0) != source_revision:
        raise RuntimeError("Asset Discovery backtest snapshot revision is stale.")
    if str(job.get("strategy_configuration_hash") or "").strip() != source_hash:
        raise RuntimeError("Asset Discovery backtest snapshot configuration hash is stale.")

    request = job.get("request") if isinstance(job.get("request"), dict) else {}
    if not request:
        raise RuntimeError("Asset Discovery Strategy backtest request snapshot is unavailable.")
    if _iso_date(request.get("analysis_end_date")) != _iso_date(evaluation_end):
        raise RuntimeError(
            "Asset Discovery snapshot end differs from the Strategy backtest snapshot. Run a new Strategy backtest for the current market snapshot first."
        )

    try:
        model_snapshot = service.get_strategy_model_snapshot(db, source_id)
    except Exception as exc:
        raise RuntimeError("Asset Discovery could not resolve the current Strategy model snapshot.") from exc

    current_model_hash = str(model_snapshot.get("settings_hash") or "").strip()
    snapshot_model_hash = str(job.get("research_model_settings_hash") or "").strip()
    if current_model_hash and snapshot_model_hash and current_model_hash != snapshot_model_hash:
        raise RuntimeError(
            "Asset Discovery current model settings differ from the Strategy backtest snapshot. Run a new Strategy backtest first."
        )

    return job, request


def install_asset_discovery_certified_baseline() -> None:
    """Lock Asset Discovery to the exact Strategy execution structure.

    The prior backtest is used only to reconstruct execution semantics
    (revision/hash/assets/anchors/research split/model snapshot). Its ending
    capital is NOT a gate. The economic baseline is always the fresh full-
    history replay produced inside the current Asset Discovery run.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    from . import asset_discovery as service

    original_request = service._marginal_execution_request

    if getattr(original_request, "_asset_discovery_dynamic_baseline", False):
        _INSTALLED = True
        return

    def strategy_snapshot_execution_request(
        db: Any,
        base_config: Any,
        strategy: dict[str, Any],
        winner_config: Any,
        end_session: str,
        *,
        assets: list[str],
        reference_assets: list[str],
        candidate_assets: list[str],
        analysis_start_date: str | None = None,
        analysis_end_date: str | None = None,
    ) -> Any:
        request = original_request(
            db,
            base_config,
            strategy,
            winner_config,
            end_session,
            assets=assets,
            reference_assets=reference_assets,
            candidate_assets=candidate_assets,
            analysis_start_date=analysis_start_date,
            analysis_end_date=analysis_end_date,
        )
        _, snapshot = _certified_job(service, db, strategy, analysis_end_date or end_session)

        snapshot_assets = _normalized_assets(snapshot.get("assets"))
        baseline_assets = _normalized_assets(reference_assets)
        if snapshot_assets != baseline_assets:
            raise RuntimeError(
                "Asset Discovery baseline assets do not match the exact asset universe of the Strategy backtest snapshot."
            )

        snapshot_anchors = _normalized_assets(snapshot.get("calendar_anchor_assets"))
        snapshot_reference = _normalized_assets(snapshot.get("research_reference_assets"))
        snapshot_candidates = _normalized_assets(snapshot.get("research_candidate_assets"))
        new_candidates = _normalized_assets(candidate_assets)

        if not snapshot_anchors:
            snapshot_anchors = list(snapshot_reference or snapshot_assets)
        if not snapshot_reference:
            snapshot_reference = list(snapshot_assets)

        combined_candidates = list(dict.fromkeys([*snapshot_candidates, *new_candidates]))
        snapshot_family = str(snapshot.get("research_model_family") or "").strip()
        snapshot_settings = (
            dict(snapshot.get("research_model_settings") or {})
            if isinstance(snapshot.get("research_model_settings"), dict)
            else None
        )

        updates: dict[str, Any] = {
            "calendar_anchor_assets": snapshot_anchors,
            "research_reference_assets": snapshot_reference,
            "research_candidate_assets": combined_candidates,
        }
        if snapshot_family:
            updates["research_model_family"] = snapshot_family
        if snapshot_settings is not None:
            updates["research_model_settings"] = snapshot_settings

        return request.model_copy(update=updates)

    setattr(strategy_snapshot_execution_request, "_asset_discovery_dynamic_baseline", True)
    service._marginal_execution_request = strategy_snapshot_execution_request
    _INSTALLED = True
