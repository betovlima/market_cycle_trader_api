from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.database import Database

from ...core.environment import load_project_environment

MONGO_URI = str(os.getenv("MONGO_URL") or os.getenv("MONGO_URI") or "").strip()
MONGO_DATABASE = str(os.getenv("MONGO_DATABASE") or "").strip()
SETTINGS_COLLECTION = "backtest_settings"
SETTINGS_HISTORY_COLLECTION = "backtest_settings_history"
STRATEGY_PROFILES_COLLECTION = "strategy_profiles"
STRATEGY_CONTROL_COLLECTION = "strategy_control"
STRATEGY_PROMOTION_HISTORY_COLLECTION = "strategy_promotion_history"
JOBS_COLLECTION = "backtest_jobs"
RUNS_COLLECTION = "backtest_runs"
PREDICTIONS_COLLECTION = "backtest_predictions"
TRADES_COLLECTION = "backtest_trades"
COMPARISONS_COLLECTION = "backtest_comparisons"
FAILURES_COLLECTION = "backtest_failures"
ALPACA_MARKET_BARS_COLLECTION = "alpaca_market_bars"
MARKET_BARS_COLLECTION = "market_bars"
INTEGRATIONS_COLLECTION = "integrations"
ALPACA_INTEGRATION_ID = "alpaca"
PAPER_TRADING_SETTINGS_COLLECTION = "paper_trading_settings"
PAPER_TRADING_SETTINGS_HISTORY_COLLECTION = "paper_trading_settings_history"
PAPER_TRADING_STATE_COLLECTION = "paper_trading_state"
PAPER_TRADE_PLANS_COLLECTION = "paper_trade_plans"
PAPER_TRADE_ORDERS_COLLECTION = "paper_trade_orders"
PAPER_MARKET_RUNS_COLLECTION = "paper_market_runs"
PAPER_MARKET_AUTOMATION_COLLECTION = "paper_market_automation"
ADMIN_OPERATION_LOGS_COLLECTION = "admin_operation_logs"
SYSTEM_SETTINGS_COLLECTION = "system_settings"
SYSTEM_SETTINGS_HISTORY_COLLECTION = "system_settings_history"
PAPER_PORTFOLIO_SNAPSHOTS_COLLECTION = "paper_portfolio_snapshots"
PARAMETER_BOOTSTRAP_RUNS_COLLECTION = "parameter_bootstrap_runs"
MODEL_RESEARCH_SETTINGS_COLLECTION = "model_research_settings"
MODEL_RESEARCH_SETTINGS_HISTORY_COLLECTION = "model_research_settings_history"
MODEL_TUNING_RUNS_COLLECTION = "model_tuning_runs"
MODEL_TUNING_MARKET_SNAPSHOTS_COLLECTION = "model_tuning_market_snapshots"
MODEL_TUNING_VALIDATIONS_COLLECTION = "model_tuning_validations"
TEMPORAL_INTELLIGENCE_RUNS_COLLECTION = "temporal_intelligence_runs"
TEMPORAL_INTELLIGENCE_OBSERVATIONS_COLLECTION = "temporal_intelligence_observations"
TEMPORAL_INTELLIGENCE_ARTIFACTS_COLLECTION = "temporal_intelligence_artifacts"
TEMPORAL_POLICY_SEARCH_COLLECTION = "temporal_policy_search_runs"
TEMPORAL_WINNER_TRANSITION_RISK_RESEARCH_COLLECTION = "temporal_winner_transition_risk_research"
TEMPORAL_WINNER_TRANSITION_INTERVENTION_RESEARCH_COLLECTION = "temporal_winner_transition_intervention_research"
TEMPORAL_WINNER_TRANSITION_CONFIDENCE_RESEARCH_COLLECTION = "temporal_winner_transition_confidence_research"
TEMPORAL_WINNER_TRANSITION_STATEFUL_RESEARCH_COLLECTION = "temporal_winner_transition_stateful_research"
TEMPORAL_LEADERSHIP_REGIME_RESEARCH_COLLECTION = "temporal_leadership_regime_research"
TEMPORAL_OPPORTUNITY_DROUGHT_RESEARCH_COLLECTION = "temporal_opportunity_drought_research"
TEMPORAL_FRAGILE_INCUMBENT_RESEARCH_COLLECTION = "temporal_fragile_incumbent_research"
TEMPORAL_REGIME_CLUSTERING_RESEARCH_COLLECTION = "temporal_regime_clustering_research"
TEMPORAL_EMERGING_TREND_RESEARCH_COLLECTION = "temporal_emerging_trend_research"
TEMPORAL_RISK_AWARE_ALTERNATIVE_ACTION_COLLECTION = "temporal_risk_aware_alternative_action"
TEMPORAL_OPERATIONAL_POLICY_QUALIFICATION_COLLECTION = "temporal_operational_policy_qualification"
TEMPORAL_DECISION_SCIENCE_RESEARCH_COLLECTION = "temporal_decision_science_research"
TEMPORAL_RESEARCH_SETTINGS_COLLECTION = "temporal_research_settings"
TEMPORAL_RESEARCH_SETTINGS_HISTORY_COLLECTION = "temporal_research_settings_history"
TEMPORAL_ROTATION_QUALITY_RESEARCH_COLLECTION = "temporal_rotation_quality_research"
TEMPORAL_ROTATION_QUALITY_VALIDATION_COLLECTION = "temporal_rotation_quality_validations"
TEMPORAL_ROTATION_QUALITY_DIAGNOSTIC_COLLECTION = "temporal_rotation_quality_diagnostics"
TEMPORAL_ROTATION_QUALITY_ANALYTICS_COLLECTION = "temporal_rotation_quality_analytics"
ASSET_DISCOVERY_RESEARCH_COLLECTION = "asset_discovery_research"
ASSET_DISCOVERY_CATALOG_COLLECTION = "asset_discovery_catalog"
SETTINGS_SCHEMA_VERSION = 16
SETTINGS_METADATA_FIELDS = frozenset({
    "_id",
    "created_at",
    "updated_at",
    "schema_version",
    "configuration_name",
    "configuration_note",
    "bootstrap_source",
    "revision",
    "winner_source_file",
    "winner_configuration_hash",
})


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def create_client() -> MongoClient:
    if not MONGO_URI:
        raise RuntimeError("MONGO_URL is required in the server environment.")
    return MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=2000,
        connectTimeoutMS=2000,
        maxPoolSize=30,
        minPoolSize=1,
        retryWrites=True,
    )


def get_database(client: MongoClient | None = None) -> Database:
    if not MONGO_DATABASE:
        raise RuntimeError("MONGO_DATABASE is required in the server environment.")
    active_client = client or create_client()
    active_client.admin.command("ping")
    return active_client[MONGO_DATABASE]


def bson_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        stamp = value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
        return stamp.to_pydatetime()
    if isinstance(value, datetime):
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    if isinstance(value, np.generic):
        return bson_value(value.item())
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, dict):
        return {str(key): bson_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [bson_value(item) for item in value]
    return value


def ensure_database(db: Database) -> None:
    

    db[JOBS_COLLECTION].create_index(
        [("status", ASCENDING), ("created_at", DESCENDING)],
        name="ix_jobs_status_created",
    )
    db[RUNS_COLLECTION].create_index(
        [("job_id", ASCENDING), ("symbol", ASCENDING), ("backend", ASCENDING)],
        unique=True,
        name="uq_backtest_run",
    )
    db[PREDICTIONS_COLLECTION].create_index(
        [("job_id", ASCENDING), ("symbol", ASCENDING), ("backend", ASCENDING), ("timestamp", ASCENDING)],
        unique=True,
        name="uq_backtest_prediction",
    )
    db[TRADES_COLLECTION].create_index(
        [("job_id", ASCENDING), ("symbol", ASCENDING), ("backend", ASCENDING), ("timestamp", ASCENDING), ("sequence", ASCENDING)],
        unique=True,
        name="uq_backtest_trade",
    )
    db[COMPARISONS_COLLECTION].create_index(
        [("job_id", ASCENDING)],
        unique=True,
        name="uq_backtest_comparison",
    )
    db[FAILURES_COLLECTION].create_index(
        [("job_id", ASCENDING), ("symbol", ASCENDING), ("backend", ASCENDING)],
        unique=True,
        name="uq_backtest_failure",
    )
    db[SETTINGS_HISTORY_COLLECTION].create_index(
        [("captured_at", DESCENDING)],
        name="ix_settings_history_captured",
    )
    db[STRATEGY_PROFILES_COLLECTION].create_index(
        [("locked", DESCENDING), ("updated_at", DESCENDING)],
        name="ix_strategy_profiles_locked_updated",
    )
    db[STRATEGY_PROFILES_COLLECTION].create_index(
        [("configuration_hash", ASCENDING)],
        name="ix_strategy_profiles_configuration_hash",
    )
    db[ASSET_DISCOVERY_CATALOG_COLLECTION].create_index(
        [("last_seen_at", DESCENDING)],
        name="ix_asset_discovery_catalog_last_seen",
    )
    db[ASSET_DISCOVERY_CATALOG_COLLECTION].create_index(
        [("times_discovered", DESCENDING), ("best_rank", ASCENDING)],
        name="ix_asset_discovery_catalog_recurrence_rank",
    )
    db[STRATEGY_PROMOTION_HISTORY_COLLECTION].create_index(
        [("promoted_at", DESCENDING), ("created_at", DESCENDING)],
        name="ix_strategy_promotion_history",
    )
    db[PAPER_TRADING_SETTINGS_HISTORY_COLLECTION].create_index(
        [("captured_at", DESCENDING)],
        name="ix_paper_settings_history_captured",
    )
    db[PAPER_TRADE_PLANS_COLLECTION].create_index(
        [("decision_date", DESCENDING)],
        unique=True,
        name="uq_paper_plan_decision_date",
    )
    db[PAPER_TRADE_PLANS_COLLECTION].create_index(
        [("status", ASCENDING), ("expected_market_open", ASCENDING)],
        name="ix_paper_plan_status_open",
    )
    db[PAPER_TRADE_ORDERS_COLLECTION].create_index(
        [("client_order_id", ASCENDING)],
        unique=True,
        name="uq_paper_client_order_id",
    )
    db[PAPER_TRADE_ORDERS_COLLECTION].create_index(
        [("plan_id", ASCENDING), ("created_at", ASCENDING)],
        name="ix_paper_orders_plan_created",
    )
    db[PAPER_MARKET_RUNS_COLLECTION].create_index(
        [("active_key", ASCENDING)],
        unique=True,
        sparse=True,
        name="uq_paper_market_active_key",
    )
    db[PAPER_MARKET_RUNS_COLLECTION].create_index(
        [("created_at", DESCENDING)],
        name="ix_paper_market_created",
    )
    db[PAPER_MARKET_RUNS_COLLECTION].create_index(
        [("status", ASCENDING), ("expected_market_open", ASCENDING)],
        name="ix_paper_market_status_open",
    )
    db[ADMIN_OPERATION_LOGS_COLLECTION].create_index(
        [("created_at", DESCENDING)],
        name="ix_admin_operation_logs_created",
    )
    db[SYSTEM_SETTINGS_COLLECTION].create_index(
        [("updated_at", DESCENDING)],
        name="ix_system_settings_updated",
    )
    db[SYSTEM_SETTINGS_HISTORY_COLLECTION].create_index(
        [("settings_id", ASCENDING), ("updated_at", DESCENDING)],
        name="ix_system_settings_history_updated",
    )
    db[PAPER_PORTFOLIO_SNAPSHOTS_COLLECTION].create_index(
        [("recorded_at", DESCENDING)],
        name="ix_paper_portfolio_recorded",
    )
    db[PARAMETER_BOOTSTRAP_RUNS_COLLECTION].create_index(
        [("finished_at", DESCENDING)],
        name="ix_parameter_bootstrap_finished",
    )
    db[MODEL_TUNING_MARKET_SNAPSHOTS_COLLECTION].create_index(
        [("snapshot_id", ASCENDING), ("kind", ASCENDING)],
        name="ix_model_tuning_snapshot_kind",
    )
    db[MODEL_TUNING_MARKET_SNAPSHOTS_COLLECTION].create_index(
        [("snapshot_id", ASCENDING), ("symbol", ASCENDING)],
        unique=True,
        sparse=True,
        name="uq_model_tuning_snapshot_symbol",
    )
    db[MODEL_RESEARCH_SETTINGS_COLLECTION].create_index(
        [("updated_at", DESCENDING)],
        name="ix_model_research_settings_updated",
    )
    db[MODEL_RESEARCH_SETTINGS_HISTORY_COLLECTION].create_index(
        [("settings_id", ASCENDING), ("updated_at", DESCENDING)],
        name="ix_model_research_settings_history",
    )
    db[TEMPORAL_RESEARCH_SETTINGS_COLLECTION].create_index(
        [("updated_at", DESCENDING)],
        name="ix_temporal_research_settings_updated",
    )
    db[TEMPORAL_RESEARCH_SETTINGS_HISTORY_COLLECTION].create_index(
        [("settings_id", ASCENDING), ("updated_at", DESCENDING)],
        name="ix_temporal_research_settings_history",
    )
    db[MODEL_TUNING_RUNS_COLLECTION].create_index(
        [("status", ASCENDING), ("created_at", DESCENDING)],
        name="ix_model_tuning_status_created",
    )
    db[MODEL_TUNING_RUNS_COLLECTION].create_index(
        [("strategy_profile_id", ASCENDING), ("created_at", DESCENDING)],
        name="ix_model_tuning_strategy_created",
    )
    db[MODEL_TUNING_VALIDATIONS_COLLECTION].create_index(
        [("id", ASCENDING)],
        unique=True,
        name="ux_model_tuning_validations_id",
    )
    db[MODEL_TUNING_VALIDATIONS_COLLECTION].create_index(
        [("tuning_run_id", ASCENDING), ("candidate_id", ASCENDING)],
        unique=True,
        name="ux_model_tuning_validations_candidate",
    )
    db[MODEL_TUNING_VALIDATIONS_COLLECTION].create_index(
        [("created_at", DESCENDING)],
        name="ix_model_tuning_validations_created",
    )
    db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].create_index(
        [("status", ASCENDING), ("created_at", DESCENDING)],
        name="ix_temporal_intelligence_status_created",
    )
    db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].create_index(
        [("strategy_profile_id", ASCENDING), ("created_at", DESCENDING)],
        name="ix_temporal_intelligence_strategy_created",
    )
    db[TEMPORAL_INTELLIGENCE_OBSERVATIONS_COLLECTION].create_index(
        [("run_id", ASCENDING), ("timestamp", ASCENDING)],
        name="ix_temporal_intelligence_observations_run_timestamp",
    )
    db[TEMPORAL_INTELLIGENCE_ARTIFACTS_COLLECTION].create_index(
        [("run_id", ASCENDING), ("kind", ASCENDING), ("sequence", ASCENDING)],
        name="ix_temporal_intelligence_artifacts_run_kind_sequence",
    )

    db[TEMPORAL_POLICY_SEARCH_COLLECTION].create_index(
        [("run_id", ASCENDING), ("created_at", DESCENDING)],
        name="ix_temporal_policy_search_run_created",
    )
    db[TEMPORAL_POLICY_SEARCH_COLLECTION].create_index(
        [("id", ASCENDING)],
        unique=True,
        name="uq_temporal_policy_search_id",
    )
    db[TEMPORAL_WINNER_TRANSITION_RISK_RESEARCH_COLLECTION].create_index(
        [("run_id", ASCENDING), ("processing_id", ASCENDING), ("period_start", ASCENDING), ("period_end", ASCENDING), ("created_at", DESCENDING)],
        name="ix_temporal_winner_transition_risk_scope_created",
    )
    db[TEMPORAL_WINNER_TRANSITION_RISK_RESEARCH_COLLECTION].create_index(
        [("id", ASCENDING)],
        unique=True,
        name="uq_temporal_winner_transition_risk_id",
    )
    db[TEMPORAL_WINNER_TRANSITION_INTERVENTION_RESEARCH_COLLECTION].create_index(
        [("run_id", ASCENDING), ("processing_id", ASCENDING), ("period_start", ASCENDING), ("period_end", ASCENDING), ("created_at", DESCENDING)],
        name="ix_temporal_winner_transition_intervention_scope_created",
    )
    db[TEMPORAL_WINNER_TRANSITION_INTERVENTION_RESEARCH_COLLECTION].create_index(
        [("id", ASCENDING)],
        unique=True,
        name="uq_temporal_winner_transition_intervention_id",
    )
    db[TEMPORAL_WINNER_TRANSITION_CONFIDENCE_RESEARCH_COLLECTION].create_index(
        [("run_id", ASCENDING), ("processing_id", ASCENDING), ("period_start", ASCENDING), ("period_end", ASCENDING), ("created_at", DESCENDING)],
        name="ix_temporal_winner_transition_confidence_scope_created",
    )
    db[TEMPORAL_WINNER_TRANSITION_CONFIDENCE_RESEARCH_COLLECTION].create_index(
        [("id", ASCENDING)],
        unique=True,
        name="uq_temporal_winner_transition_confidence_id",
    )
    db[TEMPORAL_WINNER_TRANSITION_STATEFUL_RESEARCH_COLLECTION].create_index(
        [("run_id", ASCENDING), ("processing_id", ASCENDING), ("period_start", ASCENDING), ("period_end", ASCENDING), ("created_at", DESCENDING)],
        name="ix_temporal_winner_transition_stateful_scope_created",
    )
    db[TEMPORAL_WINNER_TRANSITION_STATEFUL_RESEARCH_COLLECTION].create_index(
        [("id", ASCENDING)],
        unique=True,
        name="uq_temporal_winner_transition_stateful_id",
    )
    db[TEMPORAL_ROTATION_QUALITY_RESEARCH_COLLECTION].create_index(
        [("status", ASCENDING), ("created_at", DESCENDING)],
        name="ix_temporal_rotation_quality_research_status_created",
    )
    db[TEMPORAL_ROTATION_QUALITY_VALIDATION_COLLECTION].create_index(
        [("research_id", ASCENDING), ("created_at", DESCENDING)],
        name="ix_temporal_rotation_quality_validation_research_created",
    )
    db[TEMPORAL_ROTATION_QUALITY_VALIDATION_COLLECTION].create_index(
        [("status", ASCENDING), ("created_at", DESCENDING)],
        name="ix_temporal_rotation_quality_validation_status_created",
    )
    db[TEMPORAL_ROTATION_QUALITY_DIAGNOSTIC_COLLECTION].create_index(
        [("validation_id", ASCENDING), ("created_at", DESCENDING)],
        name="ix_temporal_rotation_quality_diagnostic_validation_created",
    )
    db[TEMPORAL_ROTATION_QUALITY_DIAGNOSTIC_COLLECTION].create_index(
        [("status", ASCENDING), ("created_at", DESCENDING)],
        name="ix_temporal_rotation_quality_diagnostic_status_created",
    )
    db[TEMPORAL_ROTATION_QUALITY_ANALYTICS_COLLECTION].create_index(
        [("processing_id", ASCENDING), ("candidate_id", ASCENDING)],
        unique=True,
        name="uq_temporal_rotation_quality_analytics_processing_candidate",
    )
    db[TEMPORAL_ROTATION_QUALITY_ANALYTICS_COLLECTION].create_index(
        [("finished_at", DESCENDING), ("processing_kind", ASCENDING)],
        name="ix_temporal_rotation_quality_analytics_finished_kind",
    )




def _environment_value(*names: str) -> str:
    load_project_environment()
    for name in names:
        value = str(os.getenv(name) or "").strip()
        if value:
            return value
    return ""


def mask_api_key(value: str | None) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if len(text) <= 8:
        return "*" * len(text)
    return f"{text[:4]}{'*' * max(4, len(text) - 8)}{text[-4:]}"


def get_alpaca_integration_status(db: Database | None = None) -> dict[str, Any]:
    

    del db
    api_key_id = _environment_value("ALPACA_API_KEY_ID", "APCA_API_KEY_ID")
    secret_key = _environment_value(
        "ALPACA_SECRET_KEY",
        "ALPACA_API_SECRET_KEY",
        "APCA_API_SECRET_KEY",
    )
    return {
        "configured": bool(api_key_id and secret_key),
        "api_key_id_masked": mask_api_key(api_key_id),
        "source": "environment",
        "updated_at": None,
    }


def get_alpaca_credentials(db: Database | None = None) -> dict[str, str]:
    

    del db
    api_key_id = _environment_value("ALPACA_API_KEY_ID", "APCA_API_KEY_ID")
    secret_key = _environment_value(
        "ALPACA_SECRET_KEY",
        "ALPACA_API_SECRET_KEY",
        "APCA_API_SECRET_KEY",
    )
    if not api_key_id or not secret_key:
        raise RuntimeError(
            "Alpaca API credentials are not configured in the server environment. "
            "Set ALPACA_API_KEY_ID and ALPACA_SECRET_KEY."
        )
    return {"api_key_id": api_key_id, "secret_key": secret_key}


def save_alpaca_credentials(
    db: Database,
    *,
    api_key_id: str,
    secret_key: str,
) -> dict[str, Any]:
    del db, api_key_id, secret_key
    raise RuntimeError(
        "Alpaca credentials are managed exclusively through server environment variables."
    )


def delete_alpaca_credentials(db: Database) -> None:
    del db
    raise RuntimeError(
        "Alpaca credentials are managed exclusively through server environment variables."
    )


def get_settings(db: Database) -> dict[str, Any]:
    

    document = db[SETTINGS_COLLECTION].find_one({"_id": "default"})
    if document is None:
        raise RuntimeError(
            "Locked strategy configuration was not found in MongoDB. "
            "Call POST /api/admin/parameters/bootstrap."
        )
    return {
        key: bson_value(value)
        for key, value in document.items()
        if key not in SETTINGS_METADATA_FIELDS
    }


def dataframe_documents(frame: pd.DataFrame, *, job_id: str, symbol: str, backend: str) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    reset = frame.reset_index()
    if str(reset.columns[0]) != "timestamp":
        reset = reset.rename(columns={reset.columns[0]: "timestamp"})
    documents: list[dict[str, Any]] = []
    for row in reset.to_dict(orient="records"):
        document = {"job_id": job_id, "symbol": symbol, "backend": backend, **{key: bson_value(value) for key, value in row.items()}}
        if document.get("timestamp") is not None:
            document["timestamp"] = bson_value(pd.Timestamp(document["timestamp"]))
        documents.append(document)
    return documents


def trade_documents(frame: pd.DataFrame, *, job_id: str, symbol: str, backend: str) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    documents: list[dict[str, Any]] = []
    for sequence, row in enumerate(frame.to_dict(orient="records"), start=1):
        document = {"job_id": job_id, "symbol": symbol, "backend": backend, "sequence": sequence, **{key: bson_value(value) for key, value in row.items()}}
        if document.get("timestamp") is not None:
            document["timestamp"] = bson_value(pd.Timestamp(document["timestamp"]))
        documents.append(document)
    return documents


def insert_in_batches(collection: Any, documents: list[dict[str, Any]], *, batch_size: int) -> int:
    inserted = 0
    size = max(1, int(batch_size))
    for start in range(0, len(documents), size):
        result = collection.insert_many(documents[start:start + size], ordered=False)
        inserted += len(result.inserted_ids)
    return inserted


def replace_run_result(db: Database, *, job_id: str, symbol: str, backend: str, metrics: dict[str, Any], summary: str, predictions: pd.DataFrame, trades: pd.DataFrame, batch_size: int) -> None:
    run_filter = {"job_id": job_id, "symbol": symbol, "backend": backend}
    db[PREDICTIONS_COLLECTION].delete_many(run_filter)
    db[TRADES_COLLECTION].delete_many(run_filter)
    prediction_docs = dataframe_documents(predictions, job_id=job_id, symbol=symbol, backend=backend)
    trade_docs = trade_documents(trades, job_id=job_id, symbol=symbol, backend=backend)
    insert_in_batches(db[PREDICTIONS_COLLECTION], prediction_docs, batch_size=batch_size)
    insert_in_batches(db[TRADES_COLLECTION], trade_docs, batch_size=batch_size)
    now = utc_now()
    db[RUNS_COLLECTION].update_one(run_filter, {"$set": {**run_filter, "metrics": bson_value(metrics), "summary": summary, "prediction_count": len(prediction_docs), "trade_count": len(trade_docs), "updated_at": now}, "$setOnInsert": {"created_at": now}}, upsert=True)


def replace_comparison(db: Database, *, job_id: str, comparison: list[dict[str, Any]], failures: list[dict[str, Any]], effective_config: dict[str, Any]) -> None:
    now = utc_now()
    db[COMPARISONS_COLLECTION].update_one({"job_id": job_id}, {"$set": {"job_id": job_id, "results": bson_value(comparison), "failures": bson_value(failures), "effective_config": bson_value(effective_config), "updated_at": now}, "$setOnInsert": {"created_at": now}}, upsert=True)
    db[FAILURES_COLLECTION].delete_many({"job_id": job_id})
    if failures:
        db[FAILURES_COLLECTION].insert_many([{"job_id": job_id, "symbol": str(item.get("symbol", "")).upper(), "backend": str(item.get("backend", "")).lower(), "error": str(item.get("error", "")), "created_at": now} for item in failures], ordered=False)




def get_paper_trading_settings(db: Database) -> dict[str, Any]:
    document = db[PAPER_TRADING_SETTINGS_COLLECTION].find_one({"_id": "default"})
    if document is None:
        raise RuntimeError(
            "Paper-trading settings were not found in MongoDB. "
            "Call POST /api/admin/setup/initialize first."
        )
    return {
        key: bson_value(value)
        for key, value in document.items()
        if key not in SETTINGS_METADATA_FIELDS
    }


def get_paper_trading_state(db: Database) -> dict[str, Any]:
    document = db[PAPER_TRADING_STATE_COLLECTION].find_one({"_id": "default"})
    if document is None:
        raise RuntimeError(
            "Paper-trading state is not initialized. "
            "Call POST /api/admin/setup/initialize first."
        )
    return {
        key: bson_value(value)
        for key, value in document.items()
        if key not in SETTINGS_METADATA_FIELDS
    }


def replace_paper_trading_state(db: Database, state: dict[str, Any]) -> None:
    now = utc_now()
    db[PAPER_TRADING_STATE_COLLECTION].replace_one(
        {"_id": "default"},
        {
            "_id": "default",
            **bson_value(state),
            "schema_version": 1,
            "updated_at": now,
            "created_at": (
                db[PAPER_TRADING_STATE_COLLECTION].find_one(
                    {"_id": "default"}, {"created_at": 1}
                ) or {}
            ).get("created_at", now),
        },
        upsert=True,
    )


def insert_paper_trade_plan(db: Database, plan: dict[str, Any], *, replace: bool = False) -> None:
    document = {**bson_value(plan), "updated_at": utc_now()}
    if replace:
        db[PAPER_TRADE_PLANS_COLLECTION].replace_one(
            {"decision_date": document["decision_date"]},
            document,
            upsert=True,
        )
        return
    db[PAPER_TRADE_PLANS_COLLECTION].insert_one(document)


def update_paper_trade_plan(db: Database, plan_id: str, changes: dict[str, Any]) -> None:
    db[PAPER_TRADE_PLANS_COLLECTION].update_one(
        {"plan_id": plan_id},
        {"$set": {**bson_value(changes), "updated_at": utc_now()}},
    )


def insert_paper_trade_order(db: Database, order: dict[str, Any]) -> None:
    db[PAPER_TRADE_ORDERS_COLLECTION].insert_one(
        {**bson_value(order), "created_at": utc_now(), "updated_at": utc_now()}
    )


def update_paper_trade_order(db: Database, client_order_id: str, changes: dict[str, Any]) -> None:
    db[PAPER_TRADE_ORDERS_COLLECTION].update_one(
        {"client_order_id": client_order_id},
        {"$set": {**bson_value(changes), "updated_at": utc_now()}},
    )
