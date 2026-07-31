from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np
import pandas as pd
from pymongo import ASCENDING, DESCENDING, MongoClient, UpdateOne
from pymongo.database import Database


# Local defaults keep development zero-config. Railway can inject MONGO_URL
# from its MongoDB service; MONGO_URI remains a compatible explicit override.
MONGO_URI = (
    os.getenv("MONGO_URL")
    or os.getenv("MONGO_URI")
    or "mongodb://127.0.0.1:27017"
)
MONGO_DATABASE = os.getenv("MONGO_DATABASE", "extrema_backtest")

SETTINGS_COLLECTION = "backtest_settings"
JOBS_COLLECTION = "backtest_jobs"
RUNS_COLLECTION = "backtest_runs"
PREDICTIONS_COLLECTION = "backtest_predictions"
TRADES_COLLECTION = "backtest_trades"
COMPARISONS_COLLECTION = "backtest_comparisons"
FAILURES_COLLECTION = "backtest_failures"
MARKET_BARS_COLLECTION = "market_bars"
PARAMETER_PROFILES_COLLECTION = "backtest_parameter_profiles"
ALPACA_MARKET_BARS_COLLECTION = "alpaca_market_bars"
INTEGRATIONS_COLLECTION = "integrations"
ALPACA_INTEGRATION_ID = "alpaca"
SETTINGS_SCHEMA_VERSION = 3


DEFAULT_SETTINGS: dict[str, Any] = {
    "assets": ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AMD", "JPM", "SPY"],
    "model_backends": ["histgb"],
    "parameter_mode": "general",
    "asset_overrides": {},
    "strategy_mode": "COMPOUND_ROTATION_SWING_1W",
    "start_date": "2016-01-01",
    "end_date": None,
    "timeframe": "1Day",
    "market_data_provider": "alpaca",
    "alpaca_feed": "iex",
    "alpaca_adjustment": "all",

    "future_horizon": 5,
    "extrema_lookback": 10,
    "reversal_threshold": 0.03,
    "extrema_tolerance": 0.01,
    "event_tolerance_bars": 2,

    "calibration_fraction": 0.15,
    "test_fraction": 0.20,
    "retrain_every_bars": 63,
    "minimum_training_rows": 500,

    "threshold_min": 0.25,
    "threshold_max": 0.85,
    "threshold_step": 0.025,
    "bottom_threshold_max": 0.75,
    "top_threshold_max": 0.85,
    "bottom_min_precision": 0.60,
    "bottom_min_recall": 0.30,
    "top_min_precision": 0.45,
    "top_min_recall": 0.00,
    "minimum_calibration_signals": 3,
    "bottom_min_calibration_signals": 3,
    "top_min_calibration_signals": 3,

    "entry_max_rsi": 60.0,
    "entry_require_above_ema50": False,
    "entry_cooldown_bars": 3,

    "trend_pullback_entry_enabled": True,
    "trend_pullback_ema": 20,
    "trend_pullback_rsi_min": 40.0,
    "trend_pullback_rsi_max": 65.0,
    "trend_pullback_touch_tolerance": 0.02,
    "trend_pullback_require_positive_return": True,

    "adaptive_bull_regime_enabled": True,
    "bull_regime_ema_fast": 20,
    "bull_regime_ema_slow": 50,
    "bull_regime_require_price_above_slow": True,
    "bull_regime_require_slow_ema_rising": True,
    "bull_regime_entry_enabled": False,
    "bull_regime_entry_confirmation_bars": 3,

    "exit_top_probability": False,
    "exit_trend_breakdown": True,
    "exit_atr_trailing_stop": True,
    "minimum_holding_bars": 3,
    "atr_trailing_multiplier": 3.0,
    "top_tighten_trailing": True,
    "tightened_atr_multiplier": 1.5,
    "trend_exit_ema_fast": 5,
    "trend_exit_ema_slow": 20,
    "trend_breakdown_confirmation_bars": 2,
    "trend_breakdown_require_slow_ema_decline": True,

    "bull_exit_ema_fast": 20,
    "bull_exit_ema_slow": 50,
    "bull_exit_confirmation_bars": 3,
    "bull_exit_require_slow_ema_decline": True,

    "exit_fibonacci_target": True,
    "fibonacci_target_ratio": 1.618,
    "fibonacci_swing_lookback": 50,
    "fibonacci_low_lookback": 5,

    "mtf_top_signal_timeframe": "1Week",
    "mtf_top_confirmation_timeframe": "1Day",
    "mtf_top_future_horizon": 4,
    "mtf_top_extrema_lookback": 10,
    "mtf_top_reversal_threshold": 0.10,
    "mtf_top_extrema_tolerance": 0.03,
    "mtf_top_probability_floor": 0.60,
    "mtf_top_retrain_every_bars": 13,
    "mtf_top_minimum_training_rows": 500,
    "mtf_daily_confirmation_ema": 20,
    "mtf_daily_confirmation_bars": 2,
    "mtf_daily_require_negative_return": True,
    "mtf_daily_require_ema_decline": True,
    "mtf_daily_require_lower_high": False,
    "mtf_top_signal_valid_days": 20,
    "mtf_top_min_position_return": 0.0,
    "mtf_top_high_lookback_weeks": 26,
    "mtf_top_max_distance_from_high": 0.10,
    "mtf_exit_quality_horizon_days": 20,
    "exit_risk_model_backend": "xgboost",
    "exit_risk_compare_models": True,
    "exit_risk_model_backends": ["xgboost", "histgb", "catboost"],
    "exit_risk_signal_timeframe": "1Week",
    "exit_risk_horizon_weeks": 8,
    "exit_risk_event_tolerance_weeks": 2,
    "exit_risk_down_barrier": 0.12,
    "exit_risk_up_barrier": 0.08,
    "exit_risk_probability_floor": 0.60,
    "exit_risk_threshold_max": 0.85,
    "exit_risk_min_precision": 0.55,
    "exit_risk_min_recall": 0.20,
    "exit_risk_min_calibration_signals": 5,
    "exit_risk_hard_calibration_gate": True,
    "exit_risk_retrain_every_bars": 26,
    "exit_risk_minimum_training_rows": 300,
    "exit_risk_reentry_enabled": True,
    "exit_risk_reentry_cooldown_days": 5,

    "swing_exit_horizon_days": 10,
    "swing_exit_event_tolerance_days": 3,
    "swing_exit_down_barrier": 0.06,
    "swing_exit_up_barrier": 0.04,
    "swing_exit_retrain_every_bars": 20,
    "swing_exit_minimum_training_rows": 500,

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
    "rotation_seed_step": 1_000,

    "qrdqn_training_steps": 15_000,
    "qrdqn_parallel_folds": 2,
    "qrdqn_early_stopping_enabled": False,
    "qrdqn_early_stopping_patience": 4,
    "qrdqn_min_training_steps": 5_000,
    "qrdqn_episode_days": 252,
    "qrdqn_replay_size": 30_000,
    "qrdqn_learning_starts": 750,
    "qrdqn_batch_size": 128,
    "qrdqn_learning_rate": 0.0003,
    "qrdqn_gamma": 0.99,
    "qrdqn_n_quantiles": 25,
    "qrdqn_hidden_dim": 128,
    "qrdqn_target_update_steps": 250,
    "qrdqn_eval_every_steps": 1000,
    "qrdqn_epsilon_start": 1.0,
    "qrdqn_epsilon_end": 0.05,
    "qrdqn_device": "cpu",

    "initial_capital": 10_000.0,
    "whole_shares": False,
    "slippage_bps": 0.0,

    "commission_rate": 0.0,
    "sec_fee_rate": 0.0000206,
    "taf_fee_per_share": 0.000195,
    "taf_fee_cap": 9.79,
    "cat_fee_per_share": 0.000003,

    "hist_max_iter": 300,
    "hist_learning_rate": 0.04,
    "hist_max_leaf_nodes": 15,
    "hist_min_samples_leaf": 25,
    "hist_l2_regularization": 2.0,

    "xgb_n_estimators": 350,
    "xgb_learning_rate": 0.035,
    "xgb_max_depth": 3,
    "xgb_min_child_weight": 5.0,
    "xgb_subsample": 0.85,
    "xgb_colsample_bytree": 0.85,
    "xgb_gamma": 0.0,
    "xgb_reg_alpha": 0.10,
    "xgb_reg_lambda": 2.0,
    "xgb_n_jobs": -1,
    "xgb_device": "cpu",

    "catboost_iterations": 350,
    "catboost_learning_rate": 0.035,
    "catboost_depth": 6,
    "catboost_l2_leaf_reg": 3.0,
    "catboost_random_strength": 1.0,
    "catboost_thread_count": -1,

    "max_parallel_workers": 3,
    "cuda_parallel_workers": 1,

    "yfinance_auto_adjust": True,
    "yfinance_repair": False,
    "yfinance_timeout": 30,
    "yfinance_fallback_period": "max",

    "mongo_cache_enabled": True,
    "mongo_collection": MARKET_BARS_COLLECTION,
    "mongo_refresh_overlap_days": 7,
    "mongo_server_timeout_ms": 2_000,
    "mongo_write_batch_size": 1_000,

    "random_state": 42,
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def create_client() -> MongoClient:
    return MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=2_000,
        connectTimeoutMS=2_000,
        maxPoolSize=30,
        minPoolSize=1,
        retryWrites=True,
    )


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
        stamp = value
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("UTC")
        else:
            stamp = stamp.tz_convert("UTC")
        return stamp.to_pydatetime()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, np.generic):
        return bson_value(value.item())
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, dict):
        return {str(key): bson_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [bson_value(item) for item in value]
    return value


def public_document(document: dict[str, Any] | None) -> dict[str, Any] | None:
    if document is None:
        return None
    return {
        key: bson_value(value)
        for key, value in document.items()
        if key != "_id"
    }


def ensure_database(db: Database) -> None:
    settings = db[SETTINGS_COLLECTION]
    existing = settings.find_one({"_id": "default"})
    if existing is None:
        settings.insert_one(
            {
                "_id": "default",
                **DEFAULT_SETTINGS,
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "schema_version": SETTINGS_SCHEMA_VERSION,
            }
        )
    else:
        schema_version = int(existing.get("schema_version", 1) or 1)
        migration: dict[str, Any] = {}
        if schema_version < 2:
            # v1.9.12 makes Alpaca the system-wide default. Migrate the old
            # persisted Yahoo default once; users can explicitly select Yahoo
            # again after this migration and that choice will then persist.
            migration["market_data_provider"] = "alpaca"
            if (
                existing.get("strategy_mode") == "COMPOUND_ROTATION_SWING_1W"
                and existing.get("start_date") == "2000-01-01"
            ):
                migration["start_date"] = "2016-01-01"
        if schema_version < 3:
            # v1.9.18 freezes the current official Swing configuration at the
            # selected XGBoost H40 baseline. QR-DQN remains available as an
            # explicit experimental model, but is no longer selected by default.
            if existing.get("strategy_mode") == "COMPOUND_ROTATION_SWING_1W":
                migration["rotation_models"] = ["xgboost_utility"]
                migration["rotation_horizon_days"] = 40
                migration["rotation_purge_days"] = 60
        settings.update_one(
            {"_id": "default"},
            {
                "$set": {
                    **migration,
                    "updated_at": utc_now(),
                    "schema_version": SETTINGS_SCHEMA_VERSION,
                }
            },
        )

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
        [
            ("job_id", ASCENDING),
            ("symbol", ASCENDING),
            ("backend", ASCENDING),
            ("timestamp", ASCENDING),
        ],
        unique=True,
        name="uq_backtest_prediction",
    )
    db[TRADES_COLLECTION].create_index(
        [
            ("job_id", ASCENDING),
            ("symbol", ASCENDING),
            ("backend", ASCENDING),
            ("timestamp", ASCENDING),
            ("sequence", ASCENDING),
        ],
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
    db[PARAMETER_PROFILES_COLLECTION].create_index(
        [
            ("symbol", ASCENDING),
            ("timeframe", ASCENDING),
        ],
        unique=True,
        name="uq_parameter_profile_symbol_timeframe",
    )



def mask_api_key(value: str | None) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if len(text) <= 8:
        return "*" * len(text)
    return f"{text[:4]}{'*' * max(4, len(text) - 8)}{text[-4:]}"


def get_alpaca_integration_status(db: Database) -> dict[str, Any]:
    ensure_database(db)
    document = db[INTEGRATIONS_COLLECTION].find_one(
        {"_id": ALPACA_INTEGRATION_ID},
        {"secret_key": 0},
    ) or {}
    api_key_id = str(document.get("api_key_id") or "").strip()
    return {
        "configured": bool(api_key_id),
        "api_key_id_masked": mask_api_key(api_key_id),
        "updated_at": bson_value(document.get("updated_at")),
    }


def get_alpaca_credentials(db: Database) -> dict[str, str]:
    ensure_database(db)
    document = db[INTEGRATIONS_COLLECTION].find_one(
        {"_id": ALPACA_INTEGRATION_ID}
    ) or {}
    api_key_id = str(document.get("api_key_id") or "").strip()
    secret_key = str(document.get("secret_key") or "").strip()
    if not api_key_id or not secret_key:
        raise RuntimeError(
            "Alpaca API credentials are not configured. Save the API Key ID and Secret Key in the Alpaca integration panel before starting Day Trade training."
        )
    return {"api_key_id": api_key_id, "secret_key": secret_key}


def save_alpaca_credentials(
    db: Database,
    *,
    api_key_id: str,
    secret_key: str,
) -> dict[str, Any]:
    ensure_database(db)
    api_key_id = str(api_key_id or "").strip()
    secret_key = str(secret_key or "").strip()
    if not api_key_id or not secret_key:
        raise ValueError("Both Alpaca API Key ID and Secret Key are required.")
    now = utc_now()
    db[INTEGRATIONS_COLLECTION].update_one(
        {"_id": ALPACA_INTEGRATION_ID},
        {
            "$set": {
                "api_key_id": api_key_id,
                "secret_key": secret_key,
                "updated_at": now,
            },
            "$setOnInsert": {"created_at": now},
        },
        upsert=True,
    )
    return get_alpaca_integration_status(db)


def delete_alpaca_credentials(db: Database) -> None:
    ensure_database(db)
    db[INTEGRATIONS_COLLECTION].delete_one({"_id": ALPACA_INTEGRATION_ID})

def get_settings(db: Database) -> dict[str, Any]:
    ensure_database(db)
    document = db[SETTINGS_COLLECTION].find_one({"_id": "default"}) or {}

    stored = {
        key: bson_value(value)
        for key, value in document.items()
        if key not in {"_id", "created_at", "updated_at", "schema_version"}
    }
    merged = {**DEFAULT_SETTINGS, **stored}

    missing = {
        key: bson_value(value)
        for key, value in DEFAULT_SETTINGS.items()
        if key not in stored
    }
    if missing:
        db[SETTINGS_COLLECTION].update_one(
            {"_id": "default"},
            {
                "$set": {
                    **missing,
                    "updated_at": utc_now(),
                }
            },
            upsert=True,
        )

    return merged


def update_settings(db: Database, changes: dict[str, Any]) -> dict[str, Any]:
    allowed = set(DEFAULT_SETTINGS)
    cleaned = {
        key: bson_value(value)
        for key, value in changes.items()
        if key in allowed
    }
    if cleaned:
        db[SETTINGS_COLLECTION].update_one(
            {"_id": "default"},
            {
                "$set": {
                    **cleaned,
                    "updated_at": utc_now(),
                }
            },
            upsert=True,
        )
    return get_settings(db)


def merge_job_configuration(
    settings: dict[str, Any],
    request: dict[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(settings)
    if request:
        merged.update(
            {
                key: value
                for key, value in request.items()
                if value is not None and key in DEFAULT_SETTINGS
            }
        )
    return merged


def dataframe_documents(
    frame: pd.DataFrame,
    *,
    job_id: str,
    symbol: str,
    backend: str,
) -> list[dict[str, Any]]:
    if frame.empty:
        return []

    reset = frame.reset_index()
    first_column = str(reset.columns[0])
    if first_column not in {"timestamp", "date", "datetime"}:
        reset = reset.rename(columns={reset.columns[0]: "timestamp"})
    elif first_column != "timestamp":
        reset = reset.rename(columns={reset.columns[0]: "timestamp"})

    documents: list[dict[str, Any]] = []
    for row in reset.to_dict(orient="records"):
        document = {
            "job_id": job_id,
            "symbol": symbol,
            "backend": backend,
            **{key: bson_value(value) for key, value in row.items()},
        }
        timestamp = document.get("timestamp")
        if timestamp is not None:
            document["timestamp"] = bson_value(pd.Timestamp(timestamp))
        documents.append(document)
    return documents


def trade_documents(
    frame: pd.DataFrame,
    *,
    job_id: str,
    symbol: str,
    backend: str,
) -> list[dict[str, Any]]:
    if frame.empty:
        return []

    documents: list[dict[str, Any]] = []
    for sequence, row in enumerate(frame.to_dict(orient="records"), start=1):
        document = {
            "job_id": job_id,
            "symbol": symbol,
            "backend": backend,
            "sequence": sequence,
            **{key: bson_value(value) for key, value in row.items()},
        }
        timestamp = document.get("timestamp")
        if timestamp is not None:
            document["timestamp"] = bson_value(pd.Timestamp(timestamp))
        documents.append(document)
    return documents


def insert_in_batches(
    collection: Any,
    documents: list[dict[str, Any]],
    *,
    batch_size: int = 1_000,
) -> int:
    if not documents:
        return 0

    inserted = 0
    size = max(1, int(batch_size))
    for start in range(0, len(documents), size):
        batch = documents[start : start + size]
        result = collection.insert_many(batch, ordered=False)
        inserted += len(result.inserted_ids)
    return inserted


def replace_run_result(
    db: Database,
    *,
    job_id: str,
    symbol: str,
    backend: str,
    metrics: dict[str, Any],
    summary: str,
    predictions: pd.DataFrame,
    trades: pd.DataFrame,
    batch_size: int = 1_000,
) -> None:
    run_filter = {
        "job_id": job_id,
        "symbol": symbol,
        "backend": backend,
    }

    db[PREDICTIONS_COLLECTION].delete_many(run_filter)
    db[TRADES_COLLECTION].delete_many(run_filter)

    prediction_docs = dataframe_documents(
        predictions,
        job_id=job_id,
        symbol=symbol,
        backend=backend,
    )
    trade_docs = trade_documents(
        trades,
        job_id=job_id,
        symbol=symbol,
        backend=backend,
    )

    insert_in_batches(
        db[PREDICTIONS_COLLECTION],
        prediction_docs,
        batch_size=batch_size,
    )
    insert_in_batches(
        db[TRADES_COLLECTION],
        trade_docs,
        batch_size=batch_size,
    )

    db[RUNS_COLLECTION].update_one(
        run_filter,
        {
            "$set": {
                **run_filter,
                "metrics": bson_value(metrics),
                "summary": summary,
                "prediction_count": len(prediction_docs),
                "trade_count": len(trade_docs),
                "updated_at": utc_now(),
            },
            "$setOnInsert": {
                "created_at": utc_now(),
            },
        },
        upsert=True,
    )


def replace_comparison(
    db: Database,
    *,
    job_id: str,
    comparison: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    effective_config: dict[str, Any],
) -> None:
    db[COMPARISONS_COLLECTION].update_one(
        {"job_id": job_id},
        {
            "$set": {
                "job_id": job_id,
                "results": bson_value(comparison),
                "failures": bson_value(failures),
                "effective_config": bson_value(effective_config),
                "updated_at": utc_now(),
            },
            "$setOnInsert": {
                "created_at": utc_now(),
            },
        },
        upsert=True,
    )

    db[FAILURES_COLLECTION].delete_many({"job_id": job_id})
    if failures:
        documents = [
            {
                "job_id": job_id,
                "symbol": str(item.get("symbol", "")).upper(),
                "backend": str(item.get("backend", "")).lower(),
                "error": str(item.get("error", "")),
                "created_at": utc_now(),
            }
            for item in failures
        ]
        db[FAILURES_COLLECTION].insert_many(documents, ordered=False)


def append_job_log(db: Database, job_id: str, line: str, maximum: int = 400) -> None:
    db[JOBS_COLLECTION].update_one(
        {"id": job_id},
        {
            "$push": {
                "logs": {
                    "$each": [line],
                    "$slice": -abs(int(maximum)),
                }
            },
            "$set": {"updated_at": utc_now()},
        },
    )


def profile_key(symbol: str, timeframe: str) -> str:
    return f"{str(symbol).upper()}__{timeframe}"


def get_parameter_profiles(
    db: Database,
    *,
    symbols: Iterable[str] | None = None,
    timeframe: str | None = None,
) -> list[dict[str, Any]]:
    ensure_database(db)
    query: dict[str, Any] = {}
    if symbols:
        query["symbol"] = {
            "$in": [str(item).upper() for item in symbols]
        }
    if timeframe:
        query["timeframe"] = timeframe

    documents = db[PARAMETER_PROFILES_COLLECTION].find(
        query,
        {"_id": 0},
        sort=[("symbol", ASCENDING), ("timeframe", ASCENDING)],
    )
    return [public_document(document) or {} for document in documents]


def get_parameter_profile(
    db: Database,
    *,
    symbol: str,
    timeframe: str,
) -> dict[str, Any] | None:
    ensure_database(db)
    document = db[PARAMETER_PROFILES_COLLECTION].find_one(
        {
            "symbol": str(symbol).upper(),
            "timeframe": timeframe,
        }
    )
    return public_document(document)


def save_parameter_profile(
    db: Database,
    *,
    symbol: str,
    timeframe: str,
    parameters: dict[str, Any],
    profile_name: str | None = None,
    source_job_id: str | None = None,
    validation_status: str = "candidate",
) -> dict[str, Any]:
    ensure_database(db)
    normalized_symbol = str(symbol).upper()
    allowed = set(DEFAULT_SETTINGS)
    cleaned = {
        key: bson_value(value)
        for key, value in parameters.items()
        if key in allowed
    }
    cleaned["timeframe"] = timeframe

    document = {
        "profile_id": profile_key(normalized_symbol, timeframe),
        "profile_name": (
            profile_name
            or f"{normalized_symbol} {timeframe} profile"
        ),
        "scope": "asset_timeframe",
        "symbol": normalized_symbol,
        "timeframe": timeframe,
        "parameters": cleaned,
        "source_job_id": source_job_id,
        "validation_status": validation_status,
        "updated_at": utc_now(),
    }
    db[PARAMETER_PROFILES_COLLECTION].update_one(
        {
            "symbol": normalized_symbol,
            "timeframe": timeframe,
        },
        {
            "$set": document,
            "$setOnInsert": {"created_at": utc_now()},
        },
        upsert=True,
    )
    return (
        get_parameter_profile(
            db,
            symbol=normalized_symbol,
            timeframe=timeframe,
        )
        or document
    )


def delete_parameter_profile(
    db: Database,
    *,
    symbol: str,
    timeframe: str,
) -> bool:
    result = db[PARAMETER_PROFILES_COLLECTION].delete_one(
        {
            "symbol": str(symbol).upper(),
            "timeframe": timeframe,
        }
    )
    return result.deleted_count > 0
