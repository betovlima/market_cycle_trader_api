from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.database import Database

MONGO_URI = str(os.getenv("MONGO_URL") or os.getenv("MONGO_URI") or "").strip()
MONGO_DATABASE = str(os.getenv("MONGO_DATABASE") or "").strip()
SETTINGS_COLLECTION = "backtest_settings"
SETTINGS_HISTORY_COLLECTION = "backtest_settings_history"
JOBS_COLLECTION = "backtest_jobs"
RUNS_COLLECTION = "backtest_runs"
PREDICTIONS_COLLECTION = "backtest_predictions"
TRADES_COLLECTION = "backtest_trades"
COMPARISONS_COLLECTION = "backtest_comparisons"
FAILURES_COLLECTION = "backtest_failures"
ALPACA_MARKET_BARS_COLLECTION = "alpaca_market_bars"
INTEGRATIONS_COLLECTION = "integrations"
ALPACA_INTEGRATION_ID = "alpaca"
PAPER_TRADING_SETTINGS_COLLECTION = "paper_trading_settings"
PAPER_TRADING_SETTINGS_HISTORY_COLLECTION = "paper_trading_settings_history"
PAPER_TRADING_STATE_COLLECTION = "paper_trading_state"
PAPER_TRADE_PLANS_COLLECTION = "paper_trade_plans"
PAPER_TRADE_ORDERS_COLLECTION = "paper_trade_orders"
PAPER_MARKET_RUNS_COLLECTION = "paper_market_runs"
PAPER_PORTFOLIO_SNAPSHOTS_COLLECTION = "paper_portfolio_snapshots"
PARAMETER_BOOTSTRAP_RUNS_COLLECTION = "parameter_bootstrap_runs"
SETTINGS_SCHEMA_VERSION = 15
SETTINGS_METADATA_FIELDS = frozenset({
    "_id",
    "created_at",
    "updated_at",
    "schema_version",
    "configuration_name",
    "configuration_note",
    "bootstrap_source",
    "revision",
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
    """Create storage indexes without mutating the locked strategy document."""

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
    db[PAPER_PORTFOLIO_SNAPSHOTS_COLLECTION].create_index(
        [("recorded_at", DESCENDING)],
        name="ix_paper_portfolio_recorded",
    )
    db[PARAMETER_BOOTSTRAP_RUNS_COLLECTION].create_index(
        [("finished_at", DESCENDING)],
        name="ix_parameter_bootstrap_finished",
    )



def _environment_value(*names: str) -> str:
    for name in names:
        value = str(os.getenv(name) or "").strip()
        if value:
            return value
    return ""

def get_alpaca_credentials() -> dict[str, str]:
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


def get_settings(db: Database) -> dict[str, Any]:
    """Return the complete locked document without applying code defaults."""

    document = db[SETTINGS_COLLECTION].find_one({"_id": "default"})
    if document is None:
        raise RuntimeError(
            "Locked strategy configuration was not found in MongoDB. "
            "Run scripts/apply_locked_config.py with a complete JSON configuration."
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
        if key not in {"_id", "created_at", "updated_at", "schema_version", "configuration_name", "configuration_note", "bootstrap_source"}
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
        if key not in {"_id", "created_at", "updated_at", "schema_version", "configuration_name", "configuration_note", "bootstrap_source"}
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
