from __future__ import annotations

from typing import Any

from ..asset_state_clustering.analysis import AssetStateClusteringCancelled


ASSET_STATE_STAGE = "asset_state_clustering"
_INSTALLED = False


def install_strategy_research_asset_state_stage() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from . import temporal_intelligence as temporal

    stages = list(temporal.STRATEGY_RESEARCH_PIPELINE_STAGES)
    if ASSET_STATE_STAGE not in stages:
        try:
            temporal_index = stages.index("temporal")
        except ValueError:
            temporal_index = 0
        stages.insert(temporal_index + 1, ASSET_STATE_STAGE)
        temporal.STRATEGY_RESEARCH_PIPELINE_STAGES = tuple(stages)

    original_stage_start = temporal._pipeline_stage_start
    original_snapshot = temporal.get_strategy_research_pipeline_snapshot
    original_delete_run_data = temporal._delete_strategy_research_run_data

    def stage_start(db: Any, run_id: str, stage: str) -> dict[str, Any]:
        if stage != "statistical_ml_control":
            return original_stage_start(db, run_id, stage)

        state = temporal.get_strategy_research_pipeline_state(db, run_id)
        asset_state_status = str(
            ((state.get("stage_states") or {}).get(ASSET_STATE_STAGE) or "waiting")
        ).lower()
        if asset_state_status != "completed":
            original_stage_start(db, run_id, ASSET_STATE_STAGE)
            pipeline = temporal.get_strategy_research_pipeline_state(db, run_id)
            run_document = db[temporal.TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find_one(
                {"id": str(run_id)}, {"_id": 0, "research_processing_id": 1}
            ) or {}
            processing_id = str(run_document.get("research_processing_id") or "").strip()
            start_month = str(pipeline.get("start_month") or "").strip()
            end_month = str(pipeline.get("end_month") or "").strip()
            if not processing_id or not start_month or not end_month:
                raise temporal.TemporalIntelligenceConflict(
                    "Daily Asset State Clustering is missing its processing or period binding."
                )
            try:
                from ..asset_state_clustering.service import build_and_persist

                result = build_and_persist(
                    db,
                    run_id,
                    processing_id=processing_id,
                    start_month=start_month,
                    end_month=end_month,
                    progress_callback=lambda progress: temporal._pipeline_stage_progress(
                        db, run_id, ASSET_STATE_STAGE, progress
                    ),
                    cancel_check=lambda: temporal._pipeline_stop_requested(db, run_id),
                )
                if str((result or {}).get("status") or "").lower() != "completed":
                    raise temporal.TemporalIntelligenceConflict(
                        "Daily Asset State Clustering did not produce a completed result."
                    )
            except AssetStateClusteringCancelled as exc:
                raise temporal.TemporalIntelligenceConflict(str(exc)) from exc
            except Exception as exc:
                if not temporal._pipeline_stop_requested(db, run_id):
                    temporal.control_strategy_research_pipeline(
                        db,
                        run_id,
                        action="stage_failed",
                        stage=ASSET_STATE_STAGE,
                        message=str(exc),
                    )
                raise

            temporal._pipeline_stage_complete(db, run_id, ASSET_STATE_STAGE)
            if temporal._pipeline_stop_requested(db, run_id):
                raise temporal.TemporalIntelligenceConflict(
                    "Strategy Research stop requested after Daily Asset State Clustering."
                )

        return original_stage_start(db, run_id, stage)

    def pipeline_snapshot(db: Any, run_id: str) -> dict[str, Any]:
        payload = original_snapshot(db, run_id)
        processing_id = str(payload.get("processing_id") or "").strip()
        period_start = str(payload.get("period_start") or "").strip()
        period_end = str(payload.get("period_end") or "").strip()
        if not processing_id or not period_start or not period_end:
            return payload

        from ..asset_state_clustering.service import get_persisted, public_summary

        asset_state = public_summary(
            get_persisted(
                db,
                run_id,
                processing_id=processing_id,
                start_month=period_start,
                end_month=period_end,
            )
        )
        if asset_state is None:
            return payload

        payload["asset_state_clustering"] = asset_state
        statistical = payload.get("statistical_ml_control")
        if isinstance(statistical, dict):
            statistical = dict(statistical)
            statistical["asset_state_clustering"] = asset_state
            payload["statistical_ml_control"] = statistical
        else:
            payload["statistical_ml_control"] = {
                "id": asset_state.get("id"),
                "status": "asset_state_only",
                "asset_state_clustering": asset_state,
            }
        return payload

    def delete_run_data(db: Any, run_id: str, *, delete_run: bool) -> dict[str, int]:
        deleted = original_delete_run_data(db, run_id, delete_run=delete_run)
        from ..asset_state_clustering.service import delete_run_results

        deleted["asset_state_clustering"] = delete_run_results(db, str(run_id))
        return deleted

    temporal._pipeline_stage_start = stage_start
    temporal.get_strategy_research_pipeline_snapshot = pipeline_snapshot
    temporal._delete_strategy_research_run_data = delete_run_data
    _INSTALLED = True
