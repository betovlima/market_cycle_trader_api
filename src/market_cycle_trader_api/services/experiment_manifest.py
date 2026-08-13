from __future__ import annotations

from typing import Any

from ..core.config import API_VERSION


def build_experiment_manifest(
    job: dict[str, Any],
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    

    primary_run = runs[0] if runs else {}
    metrics = dict(primary_run.get("metrics") or {})
    request = dict(job.get("request") or {})
    return {
        "schema_version": 2,
        "api_version": API_VERSION,
        "job_id": job.get("id"),
        "job_status": job.get("status"),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "strategy_profile_id": job.get("strategy_profile_id"),
        "strategy_profile_name": job.get("strategy_profile_name"),
        "strategy_profile_revision": job.get("strategy_profile_revision"),
        "strategy_configuration_hash": job.get("strategy_configuration_hash"),
        "winner_engine_compatibility": job.get("winner_engine_compatibility"),
        "trader_winner_strategy_id_at_queue": job.get("trader_winner_strategy_id_at_queue"),
        "trader_winner_strategy_name_at_queue": job.get("trader_winner_strategy_name_at_queue"),
        "trader_winner_configuration_hash_at_queue": job.get("trader_winner_configuration_hash_at_queue"),
        "trader_winner_api_version_at_queue": job.get("trader_winner_api_version_at_queue"),
        "research_reference_strategy_id_at_queue": job.get("research_reference_strategy_id_at_queue"),
        "research_reference_strategy_name_at_queue": job.get("research_reference_strategy_name_at_queue"),
        "research_reference_configuration_hash_at_queue": job.get("research_reference_configuration_hash_at_queue"),
        "research_reference_assets": job.get("research_reference_assets") or metrics.get("research_reference_assets") or [],
        "research_candidate_assets": job.get("research_candidate_assets") or metrics.get("research_candidate_assets") or [],
        "execution_calendar_anchor_assets": request.get("calendar_anchor_assets") or [],
        "market_data_signature_sha256": metrics.get("market_data_signature_sha256"),
        "market_data_signatures": metrics.get("market_data_signatures") or {},
        "market_data_history_complete": metrics.get("market_data_history_complete"),
        "runtime_fingerprint_sha256": metrics.get("runtime_fingerprint_sha256"),
        "git_commit": metrics.get("git_commit"),
        "engine_source_sha256": metrics.get("engine_source_sha256"),
        "package_source_sha256": metrics.get("package_source_sha256"),
        "runtime_versions": metrics.get("runtime_versions") or {},
        "xgboost_build_info": metrics.get("xgboost_build_info"),
        "numeric_thread_environment": metrics.get("numeric_thread_environment"),
        "threadpool_runtime": metrics.get("threadpool_runtime"),
        "deployment_runtime": metrics.get("deployment_runtime"),
        "decision_diagnostics_schema_version": metrics.get("decision_diagnostics_schema_version"),
        "decision_diagnostics_rows": metrics.get("decision_diagnostics_rows"),
        "position_risk_diagnostics_schema_version": metrics.get("position_risk_diagnostics_schema_version"),
        "position_risk_diagnostics_rows": metrics.get("position_risk_diagnostics_rows"),
        "market_regime_diagnostics_schema_version": metrics.get("market_regime_diagnostics_schema_version"),
        "market_regime_diagnostics_rows": metrics.get("market_regime_diagnostics_rows"),
        "walk_forward_fold_count": metrics.get("walk_forward_fold_count"),
        "walk_forward_folds": metrics.get("walk_forward_folds") or [],
        "configuration": request,
    }
