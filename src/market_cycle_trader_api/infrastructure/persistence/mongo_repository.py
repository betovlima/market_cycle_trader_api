from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.database import Database

MONGO_URI = os.getenv("MONGO_URL") or os.getenv("MONGO_URI") or "mongodb://127.0.0.1:27017"
MONGO_DATABASE = os.getenv("MONGO_DATABASE", "extrema_backtest")
SETTINGS_COLLECTION = "backtest_settings"
JOBS_COLLECTION = "backtest_jobs"
RUNS_COLLECTION = "backtest_runs"
PREDICTIONS_COLLECTION = "backtest_predictions"
TRADES_COLLECTION = "backtest_trades"
COMPARISONS_COLLECTION = "backtest_comparisons"
FAILURES_COLLECTION = "backtest_failures"
MARKET_BARS_COLLECTION = "market_bars"
ALPACA_MARKET_BARS_COLLECTION = "alpaca_market_bars"
INTEGRATIONS_COLLECTION = "integrations"
ALPACA_INTEGRATION_ID = "alpaca"
SETTINGS_SCHEMA_VERSION = 7
ACTIVE_STRATEGY_MODES = {
    "COMPOUND_ROTATION_SWING_XGBOOST",
    "COMPOUND_ROTATION_SWING_QRDQN",
    "COMPOUND_ROTATION_DAY_TRADE_OPEN_CLOSE",
}

DEFAULT_SETTINGS: dict[str, Any] = {
    "assets": ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AMD", "JPM", "SPY"],
    "strategy_mode": "COMPOUND_ROTATION_SWING_XGBOOST",
    "start_date": "2016-01-01",
    "end_date": None,
    "timeframe": "1Day",
    "market_data_provider": "alpaca",
    "alpaca_feed": "iex",
    "alpaca_adjustment": "all",
    "rotation_models": ["xgboost_utility"],
    "rotation_horizon_days": 40,
    "rotation_minimum_training_rows": 700,
    "rotation_walk_forward_enabled": True,
    "rotation_walk_forward_calibration_days": 126,
    "rotation_walk_forward_test_days": 504,
    "rotation_walk_forward_min_test_days": 126,
    "rotation_purge_days": 60,
    "rotation_downside_penalty": 0.20,
    "rotation_drawdown_penalty": 0.35,
    "rotation_min_holding_days": 2,
    "rotation_min_expected_edge": 0.001,
    "rotation_cash_threshold": 0.0,
    "rotation_switch_margin": 0.005,
    "rotation_switch_margin_candidates": [0.0, 0.0025, 0.005, 0.01],
    "rotation_xgb_n_estimators": 300,
    "rotation_xgb_learning_rate": 0.035,
    "rotation_xgb_max_depth": 3,
    "rotation_accelerator": "auto",
    "rotation_allow_cpu_fallback": True,
    "rotation_parallel_models": True,
    "rotation_xgb_repetitions": 1,
    "rotation_qrdqn_repetitions": 1,
    "rotation_seed_step": 1000,
    "qrdqn_training_steps": 15000,
    "qrdqn_parallel_folds": 2,
    "qrdqn_early_stopping_enabled": False,
    "qrdqn_early_stopping_patience": 4,
    "qrdqn_min_training_steps": 5000,
    "qrdqn_episode_days": 252,
    "qrdqn_replay_size": 30000,
    "qrdqn_learning_starts": 750,
    "qrdqn_batch_size": 128,
    "qrdqn_learning_rate": 0.0003,
    "qrdqn_gamma": 0.99,
    "qrdqn_n_step": 10,
    "qrdqn_n_quantiles": 25,
    "qrdqn_hidden_dim": 128,
    "qrdqn_target_update_steps": 250,
    "qrdqn_eval_every_steps": 1000,
    "qrdqn_epsilon_start": 1.0,
    "qrdqn_epsilon_end": 0.05,
    "initial_capital": 10000.0,
    "whole_shares": False,
    "slippage_bps": 0.0,
    "commission_rate": 0.0,
    "sec_fee_rate": 0.0000206,
    "taf_fee_per_share": 0.000195,
    "taf_fee_cap": 9.79,
    "cat_fee_per_share": 0.000003,
    "xgb_min_child_weight": 5.0,
    "xgb_subsample": 0.85,
    "xgb_colsample_bytree": 0.85,
    "xgb_reg_alpha": 0.10,
    "xgb_reg_lambda": 2.0,
    "xgb_n_jobs": -1,
    "yfinance_auto_adjust": True,
    "yfinance_repair": False,
    "yfinance_timeout": 30,
    "yfinance_fallback_period": "max",
    "mongo_cache_enabled": True,
    "mongo_refresh_overlap_days": 7,
    "mongo_write_batch_size": 1000,
    "random_state": 42,
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def create_client() -> MongoClient:
    return MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000, connectTimeoutMS=2000, maxPoolSize=30, minPoolSize=1, retryWrites=True)


def get_database(client: MongoClient | None = None) -> Database:
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
    settings = db[SETTINGS_COLLECTION]
    existing = settings.find_one({"_id": "default"})
    now = utc_now()
    if existing is None:
        settings.insert_one({"_id": "default", **DEFAULT_SETTINGS, "created_at": now, "updated_at": now, "schema_version": SETTINGS_SCHEMA_VERSION})
    else:
        set_values: dict[str, Any] = {"updated_at": now, "schema_version": SETTINGS_SCHEMA_VERSION}
        if existing.get("strategy_mode") not in ACTIVE_STRATEGY_MODES:
            set_values.update({
                "strategy_mode": DEFAULT_SETTINGS["strategy_mode"],
                "rotation_models": DEFAULT_SETTINGS["rotation_models"],
                "rotation_horizon_days": DEFAULT_SETTINGS["rotation_horizon_days"],
                "rotation_purge_days": DEFAULT_SETTINGS["rotation_purge_days"],
            })
        for key, value in DEFAULT_SETTINGS.items():
            if key not in existing:
                set_values[key] = value
        keep = set(DEFAULT_SETTINGS) | {"_id", "created_at", "updated_at", "schema_version"}
        unset_values = {key: "" for key in existing if key not in keep}
        update: dict[str, Any] = {"$set": set_values}
        if unset_values:
            update["$unset"] = unset_values
        settings.update_one({"_id": "default"}, update)

    db[JOBS_COLLECTION].create_index([("status", ASCENDING), ("created_at", DESCENDING)], name="ix_jobs_status_created")
    db[RUNS_COLLECTION].create_index([("job_id", ASCENDING), ("symbol", ASCENDING), ("backend", ASCENDING)], unique=True, name="uq_backtest_run")
    db[PREDICTIONS_COLLECTION].create_index([("job_id", ASCENDING), ("symbol", ASCENDING), ("backend", ASCENDING), ("timestamp", ASCENDING)], unique=True, name="uq_backtest_prediction")
    db[TRADES_COLLECTION].create_index([("job_id", ASCENDING), ("symbol", ASCENDING), ("backend", ASCENDING), ("timestamp", ASCENDING), ("sequence", ASCENDING)], unique=True, name="uq_backtest_trade")
    db[COMPARISONS_COLLECTION].create_index([("job_id", ASCENDING)], unique=True, name="uq_backtest_comparison")
    db[FAILURES_COLLECTION].create_index([("job_id", ASCENDING), ("symbol", ASCENDING), ("backend", ASCENDING)], unique=True, name="uq_backtest_failure")


def mask_api_key(value: str | None) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if len(text) <= 8:
        return "*" * len(text)
    return f"{text[:4]}{'*' * max(4, len(text) - 8)}{text[-4:]}"


def get_alpaca_integration_status(db: Database) -> dict[str, Any]:
    ensure_database(db)
    document = db[INTEGRATIONS_COLLECTION].find_one({"_id": ALPACA_INTEGRATION_ID}, {"secret_key": 0}) or {}
    api_key_id = str(document.get("api_key_id") or "").strip()
    return {"configured": bool(api_key_id), "api_key_id_masked": mask_api_key(api_key_id), "updated_at": bson_value(document.get("updated_at"))}


def get_alpaca_credentials(db: Database) -> dict[str, str]:
    ensure_database(db)
    document = db[INTEGRATIONS_COLLECTION].find_one({"_id": ALPACA_INTEGRATION_ID}) or {}
    api_key_id = str(document.get("api_key_id") or "").strip()
    secret_key = str(document.get("secret_key") or "").strip()
    if not api_key_id or not secret_key:
        raise RuntimeError("Alpaca API credentials are not configured.")
    return {"api_key_id": api_key_id, "secret_key": secret_key}


def save_alpaca_credentials(db: Database, *, api_key_id: str, secret_key: str) -> dict[str, Any]:
    ensure_database(db)
    api_key_id = str(api_key_id or "").strip()
    secret_key = str(secret_key or "").strip()
    if not api_key_id or not secret_key:
        raise ValueError("Both Alpaca API Key ID and Secret Key are required.")
    now = utc_now()
    db[INTEGRATIONS_COLLECTION].update_one({"_id": ALPACA_INTEGRATION_ID}, {"$set": {"api_key_id": api_key_id, "secret_key": secret_key, "updated_at": now}, "$setOnInsert": {"created_at": now}}, upsert=True)
    return get_alpaca_integration_status(db)


def delete_alpaca_credentials(db: Database) -> None:
    ensure_database(db)
    db[INTEGRATIONS_COLLECTION].delete_one({"_id": ALPACA_INTEGRATION_ID})


def get_settings(db: Database) -> dict[str, Any]:
    ensure_database(db)
    document = db[SETTINGS_COLLECTION].find_one({"_id": "default"}) or {}
    stored = {key: bson_value(value) for key, value in document.items() if key in DEFAULT_SETTINGS}
    return {**DEFAULT_SETTINGS, **stored}


def update_settings(db: Database, changes: dict[str, Any]) -> dict[str, Any]:
    cleaned = {key: bson_value(value) for key, value in changes.items() if key in DEFAULT_SETTINGS}
    if cleaned:
        db[SETTINGS_COLLECTION].update_one({"_id": "default"}, {"$set": {**cleaned, "updated_at": utc_now()}}, upsert=True)
    return get_settings(db)


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


def insert_in_batches(collection: Any, documents: list[dict[str, Any]], *, batch_size: int = 1000) -> int:
    inserted = 0
    size = max(1, int(batch_size))
    for start in range(0, len(documents), size):
        result = collection.insert_many(documents[start:start + size], ordered=False)
        inserted += len(result.inserted_ids)
    return inserted


def replace_run_result(db: Database, *, job_id: str, symbol: str, backend: str, metrics: dict[str, Any], summary: str, predictions: pd.DataFrame, trades: pd.DataFrame, batch_size: int = 1000) -> None:
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


