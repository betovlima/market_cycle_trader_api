from __future__ import annotations

import hashlib
import json
import re
import uuid
from typing import Any

import exchange_calendars as xcals
import pandas as pd
from pymongo import ReturnDocument

from ..core.config import API_VERSION
from ..infrastructure.persistence.mongo_repository import (
    JOBS_COLLECTION,
    PAPER_MARKET_AUTOMATION_COLLECTION,
    PAPER_MARKET_RUNS_COLLECTION,
    PAPER_TRADE_PLANS_COLLECTION,
    PAPER_TRADING_STATE_COLLECTION,
    SETTINGS_COLLECTION,
    SETTINGS_METADATA_FIELDS,
    SETTINGS_SCHEMA_VERSION,
    STRATEGY_CONTROL_COLLECTION,
    STRATEGY_PROFILES_COLLECTION,
    STRATEGY_PROMOTION_HISTORY_COLLECTION,
    bson_value,
    utc_now,
)
from ..schemas.requests import BacktestRequest

CONTROL_ID = "default"
BUNDLED_WINNER_ID = "winner-v1-13-2"
BUNDLED_WINNER_HASH = "22a4193fbb30de33d75864fc28c3b1923e4dedd4970b14f9537f793bccf18953"
ACTIVE_PAPER_KEY = "alpaca-paper-next-session"

STRATEGY_PARAMETER_GROUPS: tuple[dict[str, Any], ...] = (
    {
        "id": "identity",
        "label": "Identity and market data",
        "fields": (
            "assets",
            "strategy_mode",
            "start_date",
            "end_date",
            "timeframe",
            "market_data_provider",
            "alpaca_historical_feed",
            "alpaca_live_feed",
            "alpaca_adjustment",
            "market_data_history_backfill_enabled",
            "market_data_history_backfill_provider",
            "market_data_history_start_tolerance_days",
            "market_data_require_complete_history",
        ),
    },
    {
        "id": "targets",
        "label": "Targets and utility",
        "fields": (
            "rotation_horizon_days",
            "rotation_target_horizons",
            "rotation_target_horizon_weights",
            "rotation_movement_capture_weight",
            "rotation_trend_persistence_weight",
        ),
    },
    {
        "id": "validation",
        "label": "Walk-forward validation",
        "fields": (
            "rotation_minimum_training_rows",
            "rotation_walk_forward_enabled",
            "rotation_walk_forward_calibration_days",
            "rotation_walk_forward_test_days",
            "rotation_walk_forward_min_test_days",
            "rotation_purge_days",
        ),
    },
    {
        "id": "decision",
        "label": "Rotation policy",
        "fields": (
            "rotation_downside_penalty",
            "rotation_drawdown_penalty",
            "rotation_min_holding_days",
            "rotation_min_expected_edge",
            "rotation_cash_threshold",
            "rotation_switch_margin",
            "rotation_switch_margin_candidates",
        ),
    },
    {
        "id": "model",
        "label": "XGBoost",
        "fields": (
            "rotation_models",
            "rotation_xgb_n_estimators",
            "rotation_xgb_learning_rate",
            "rotation_xgb_max_depth",
            "rotation_accelerator",
            "rotation_allow_cpu_fallback",
            "rotation_xgb_repetitions",
            "rotation_seed_step",
            "xgb_min_child_weight",
            "xgb_subsample",
            "xgb_colsample_bytree",
            "xgb_reg_alpha",
            "xgb_reg_lambda",
            "xgb_n_jobs",
            "deterministic_execution",
            "numeric_thread_limit",
            "random_state",
        ),
    },
    {
        "id": "capital",
        "label": "Capital and transaction costs",
        "fields": (
            "initial_capital",
            "whole_shares",
            "slippage_bps",
            "commission_rate",
            "sec_fee_rate",
            "taf_fee_per_share",
            "taf_fee_cap",
            "cat_fee_per_share",
        ),
    },
    {
        "id": "storage",
        "label": "Storage and cache",
        "fields": (
            "mongo_cache_enabled",
            "mongo_refresh_overlap_days",
            "mongo_write_batch_size",
        ),
    },
)


STRATEGY_PARAMETER_DESCRIPTIONS: dict[str, str] = {
    'assets': 'Universe of symbols evaluated together by this strategy. The backend validates and normalizes the symbol list before execution.',
    'strategy_mode': 'Selects the supported strategy execution mode for this configuration.',
    'start_date': 'Earliest market date available to the strategy analysis and backtest dataset.',
    'end_date': 'Optional final market date for the analysis. Leaving it empty allows the backend to resolve the latest permitted date.',
    'timeframe': 'Bar interval used when loading and evaluating market data for this strategy.',
    'market_data_provider': 'Backend provider used to obtain the market data required by the strategy.',
    'alpaca_historical_feed': 'Alpaca feed used for historical bars during research and backtests.',
    'alpaca_live_feed': 'Alpaca feed selected for live or near-live Paper market data.',
    'alpaca_adjustment': 'Corporate-action adjustment mode applied to Alpaca historical market data.',
    'market_data_history_backfill_enabled': 'Controls whether missing historical coverage may be backfilled before analysis begins.',
    'market_data_history_backfill_provider': 'Provider authorized to supply missing historical data when backfill is enabled.',
    'market_data_history_start_tolerance_days': 'Maximum tolerated difference between the requested history start and the first available market-data date.',
    'market_data_require_complete_history': 'When enabled, the analysis requires the configured historical coverage instead of silently accepting an incomplete range.',
    'rotation_horizon_days': 'Primary forward horizon, measured in trading sessions, used by the rotation decision model.',
    'rotation_target_horizons': 'Set of forward trading-session horizons evaluated by the target construction process.',
    'rotation_target_horizon_weights': 'Relative contribution assigned to each configured target horizon. The list must correspond to the target-horizon list.',
    'rotation_movement_capture_weight': 'Weight assigned to the component that rewards capturing favorable forward price movement.',
    'rotation_trend_persistence_weight': 'Weight assigned to the component that rewards persistence of the projected directional trend.',
    'rotation_minimum_training_rows': 'Minimum number of valid training observations required before a model can be fitted for an evaluation segment.',
    'rotation_walk_forward_enabled': 'Controls expanding walk-forward validation, preserving chronological separation between training and evaluation periods.',
    'rotation_walk_forward_calibration_days': 'Number of trading sessions reserved for calibration inside each walk-forward cycle.',
    'rotation_walk_forward_test_days': 'Target number of trading sessions evaluated in each walk-forward test segment.',
    'rotation_walk_forward_min_test_days': 'Minimum acceptable number of test sessions for a walk-forward segment to be considered valid.',
    'rotation_purge_days': 'Trading-session gap used to separate training observations from later evaluation data and reduce temporal leakage.',
    'rotation_downside_penalty': 'Penalty weight applied when the utility calculation identifies unfavorable downside behavior.',
    'rotation_drawdown_penalty': 'Penalty weight applied to drawdown-related behavior in the rotation utility calculation.',
    'rotation_min_holding_days': 'Minimum number of trading sessions a selected asset should remain held before ordinary rotation is allowed.',
    'rotation_min_expected_edge': 'Minimum modeled advantage required before the rotation policy treats an opportunity as sufficiently attractive.',
    'rotation_cash_threshold': 'Decision threshold used when comparing an investable opportunity with remaining in cash.',
    'rotation_switch_margin': 'Additional advantage required before replacing the currently selected asset with another candidate.',
    'rotation_switch_margin_candidates': 'Candidate switch-margin values evaluated during calibration to select the operating margin.',
    'rotation_models': 'Model family enabled for the rotation stage. This release accepts only the model family supported by the backend.',
    'rotation_xgb_n_estimators': 'Maximum number of boosting trees configured for the rotation XGBoost model.',
    'rotation_xgb_learning_rate': 'Boosting step size controlling how strongly each new XGBoost tree contributes to the model.',
    'rotation_xgb_max_depth': 'Maximum depth allowed for individual rotation XGBoost trees.',
    'rotation_accelerator': 'Requested execution device for rotation-model training, such as CPU, CUDA or automatic selection.',
    'rotation_allow_cpu_fallback': 'Allows model training to continue on CPU when the requested accelerator cannot be used.',
    'rotation_xgb_repetitions': 'Number of repeated XGBoost training passes used by the rotation evaluation process.',
    'rotation_seed_step': 'Increment applied between repeated model seeds so repeated training runs use distinct reproducible seeds.',
    'xgb_min_child_weight': 'XGBoost minimum child-weight regularization parameter used when deciding whether a tree split has enough supporting weight.',
    'xgb_subsample': 'Fraction of training rows sampled for each boosting tree.',
    'xgb_colsample_bytree': 'Fraction of available features sampled when building each boosting tree.',
    'xgb_reg_alpha': 'L1 regularization strength applied to XGBoost leaf weights.',
    'xgb_reg_lambda': 'L2 regularization strength applied to XGBoost leaf weights.',
    'xgb_n_jobs': 'Number of CPU worker threads made available to XGBoost. The backend validates special values and deterministic-mode requirements.',
    'deterministic_execution': 'Forces the supported deterministic execution constraints so repeated runs are less affected by thread scheduling.',
    'numeric_thread_limit': 'Maximum thread count allowed for supporting numeric libraries used by model training.',
    'random_state': 'Base random seed used to make supported stochastic model operations reproducible.',
    'initial_capital': 'Starting portfolio capital used by the backtest simulation.',
    'whole_shares': 'Controls whether simulated positions must use whole-share quantities instead of fractional shares.',
    'slippage_bps': 'Simulated execution slippage expressed in basis points and applied to modeled trade prices.',
    'commission_rate': 'Commission rate included in simulated transaction costs.',
    'sec_fee_rate': 'SEC fee rate included in applicable simulated sell-side transaction costs.',
    'taf_fee_per_share': 'Trading Activity Fee amount applied per share where the simulation models that fee.',
    'taf_fee_cap': 'Maximum Trading Activity Fee charged to one simulated transaction.',
    'cat_fee_per_share': 'Consolidated Audit Trail fee amount modeled per share when applicable.',
    'mongo_cache_enabled': 'Controls use of the MongoDB-backed market-data cache for repeated research and backtest requests.',
    'mongo_refresh_overlap_days': 'Number of previously cached days re-requested during refresh so recent records can be reconciled.',
    'mongo_write_batch_size': 'Maximum number of cache records grouped into one MongoDB write batch.',
}


def _strategy_parameter_schema() -> dict[str, Any]:
    schema = BacktestRequest.model_json_schema()
    properties = schema.get("properties", {})
    for name, description in STRATEGY_PARAMETER_DESCRIPTIONS.items():
        if name in properties:
            properties[name]["description"] = description
    return schema


class StrategyLabError(RuntimeError):
    pass


class StrategyLabConflict(StrategyLabError):
    pass


class StrategyLabNotFound(StrategyLabError):
    pass


def _configuration_hash(configuration: dict[str, Any]) -> str:
    encoded = json.dumps(
        bson_value(configuration),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _configuration_from_legacy(document: dict[str, Any]) -> BacktestRequest:
    payload = {
        key: value
        for key, value in document.items()
        if key not in SETTINGS_METADATA_FIELDS
    }
    return BacktestRequest.model_validate(payload)


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return cleaned[:80] or "strategy"


def _profile_id(name: str) -> str:
    return f"{_slug(name)}-{uuid.uuid4().hex[:8]}"


def _public_profile(document: dict[str, Any], *, include_configuration: bool = True) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": str(document.get("_id")),
        "name": str(document.get("name") or "Unnamed strategy"),
        "description": str(document.get("description") or ""),
        "status": str(document.get("status") or "draft"),
        "locked": bool(document.get("locked")),
        "revision": int(document.get("revision") or 1),
        "configuration_hash": str(document.get("configuration_hash") or ""),
        "source_strategy_id": document.get("source_strategy_id"),
        "source_strategy_revision": document.get("source_strategy_revision"),
        "created_at": bson_value(document.get("created_at")),
        "updated_at": bson_value(document.get("updated_at")),
        "promoted_at": bson_value(document.get("promoted_at")),
        "last_backtest_id": document.get("last_backtest_id"),
        "last_backtest_status": document.get("last_backtest_status"),
        "last_backtest_revision": document.get("last_backtest_revision"),
        "candidate_at": bson_value(document.get("candidate_at")),
        "candidate_by": document.get("candidate_by"),
        "candidate_note": document.get("candidate_note"),
        "candidate_revision": document.get("candidate_revision"),
        "candidate_backtest_id": document.get("candidate_backtest_id"),
        "superseded_at": bson_value(document.get("superseded_at")),
        "superseded_by_strategy_id": document.get("superseded_by_strategy_id"),
        "supersession_note": document.get("supersession_note"),
        "last_promoted_winner_strategy_id": document.get("last_promoted_winner_strategy_id"),
        "last_promoted_at": bson_value(document.get("last_promoted_at")),
        "source_candidate_backtest_id": document.get("source_candidate_backtest_id"),
        "winner_api_version": document.get("winner_api_version"),
        "promotion_mode": document.get("promotion_mode"),
        "operational_state_preserved": document.get("operational_state_preserved"),
        "broker_interaction_performed": document.get("broker_interaction_performed"),
        "origin": {
            "configuration_name": document.get("origin_configuration_name"),
            "winner_source_file": document.get("origin_winner_source_file"),
            "winner_configuration_hash": document.get("origin_winner_configuration_hash"),
            "bootstrap_source": document.get("origin_bootstrap_source"),
            "schema_version": document.get("origin_schema_version"),
            "revision": document.get("origin_revision"),
        },
    }
    if include_configuration:
        configuration = BacktestRequest.model_validate(
            document.get("configuration") or {}
        ).model_dump(mode="json")
        payload["configuration"] = configuration
    return payload


def _control_response(db: Any, control: dict[str, Any]) -> dict[str, Any]:
    research_id = str(control.get("research_strategy_id") or "")
    winner_id = str(control.get("trader_winner_strategy_id") or "")
    candidate_id = str(control.get("candidate_strategy_id") or "")
    research = db[STRATEGY_PROFILES_COLLECTION].find_one({"_id": research_id})
    winner = db[STRATEGY_PROFILES_COLLECTION].find_one({"_id": winner_id})
    candidate = (
        db[STRATEGY_PROFILES_COLLECTION].find_one({"_id": candidate_id})
        if candidate_id
        else None
    )
    if research is None or winner is None:
        raise StrategyLabError("Strategy selection references a missing strategy profile.")
    return {
        "revision": int(control.get("revision") or 1),
        "research_strategy_id": research_id,
        "candidate_strategy_id": candidate_id or None,
        "trader_winner_strategy_id": winner_id,
        "research_strategy": _public_profile(research, include_configuration=False),
        "candidate_strategy": (
            _public_profile(candidate, include_configuration=False)
            if candidate is not None
            else None
        ),
        "trader_winner": _public_profile(winner, include_configuration=False),
        "updated_at": bson_value(control.get("updated_at")),
        "updated_by": control.get("updated_by"),
        "paper_state_reinitialization_required": bool(
            control.get("paper_state_reinitialization_required")
        ),
        "last_promotion_mode": control.get("last_promotion_mode"),
        "last_promoted_api_version": control.get("last_promoted_api_version"),
        "last_promoted_configuration_hash": control.get("last_promoted_configuration_hash"),
        "last_promoted_assets_count": control.get("last_promoted_assets_count"),
    }


def _legacy_winner_name(document: dict[str, Any]) -> str:
    raw = str(
        document.get("configuration_name")
        or document.get("winner_source_file")
        or "Imported production winner"
    ).strip()
    if raw.lower().endswith(".json"):
        raw = raw[:-5]
    if raw.lower().startswith("winner-v"):
        return "Winner " + raw[len("winner-"):]
    return raw


def _legacy_winner_id(document: dict[str, Any], configuration_hash: str) -> str:
    raw = str(
        document.get("winner_source_file")
        or document.get("configuration_name")
        or f"winner-{configuration_hash[:8]}"
    ).strip()
    if raw.lower().endswith(".json"):
        raw = raw[:-5]
    slug = _slug(raw)
    return slug if slug.startswith("winner-") else f"initial-winner-{slug}"



def _normalize_single_candidate_and_winner(
    db: Any,
    control: dict[str, Any],
) -> dict[str, Any]:
    """Migrate older catalogs to one active Candidate and one active Winner."""
    now = utc_now()
    winner_id = str(control.get("trader_winner_strategy_id") or "")
    winner = db[STRATEGY_PROFILES_COLLECTION].find_one({"_id": winner_id})
    if winner is not None and (
        str(winner.get("status") or "") != "winner" or not bool(winner.get("locked"))
    ):
        db[STRATEGY_PROFILES_COLLECTION].update_one(
            {"_id": winner_id},
            {"$set": {"status": "winner", "locked": True, "updated_at": now}},
        )
    db[STRATEGY_PROFILES_COLLECTION].update_many(
        {"_id": {"$ne": winner_id}, "status": "winner"},
        {"$set": {"status": "former_winner", "locked": True, "updated_at": now}},
    )

    candidate_id = str(control.get("candidate_strategy_id") or "")
    candidate = (
        db[STRATEGY_PROFILES_COLLECTION].find_one(
            {"_id": candidate_id, "status": "candidate"}
        )
        if candidate_id
        else None
    )
    candidates = list(
        db[STRATEGY_PROFILES_COLLECTION].find({"status": "candidate"})
    )
    if candidate is None:
        candidate_id = ""
    if candidate is None and candidates:
        candidates.sort(
            key=lambda item: str(
                item.get("candidate_at") or item.get("updated_at") or item.get("created_at") or ""
            ),
            reverse=True,
        )
        candidate = candidates[0]
        candidate_id = str(candidate.get("_id") or "")

    for profile in candidates:
        profile_id = str(profile.get("_id") or "")
        if profile_id and profile_id != candidate_id:
            db[STRATEGY_PROFILES_COLLECTION].update_one(
                {"_id": profile_id, "status": "candidate"},
                {
                    "$set": {
                        "status": "superseded_candidate",
                        "locked": True,
                        "superseded_at": now,
                        "superseded_by_strategy_id": candidate_id or None,
                        "updated_at": now,
                    }
                },
            )

    normalized_candidate_id = candidate_id or None
    if control.get("candidate_strategy_id") != normalized_candidate_id:
        db[STRATEGY_CONTROL_COLLECTION].update_one(
            {"_id": CONTROL_ID},
            {
                "$set": {
                    "candidate_strategy_id": normalized_candidate_id,
                    "updated_at": now,
                }
            },
        )
        control = db[STRATEGY_CONTROL_COLLECTION].find_one({"_id": CONTROL_ID}) or control
    return control

def ensure_strategy_catalog(db: Any) -> dict[str, Any]:
    control = db[STRATEGY_CONTROL_COLLECTION].find_one({"_id": CONTROL_ID})
    if control is not None:
        research = db[STRATEGY_PROFILES_COLLECTION].find_one(
            {"_id": str(control.get("research_strategy_id") or "")}
        )
        winner = db[STRATEGY_PROFILES_COLLECTION].find_one(
            {"_id": str(control.get("trader_winner_strategy_id") or "")}
        )
        if research is not None and winner is not None:
            return _normalize_single_candidate_and_winner(db, control)

    # First migration from API v1.13.16: preserve the production winner exactly
    # as it exists in backtest_settings/default. The catalog is additive and does
    # not rewrite or rename that historical document.
    legacy = db[SETTINGS_COLLECTION].find_one({"_id": "default"})
    if legacy is None:
        raise StrategyLabNotFound(
            "No strategy configuration exists. Install a protected winner first."
        )
    configuration = _configuration_from_legacy(legacy)
    payload = configuration.model_dump(mode="json")
    configuration_hash = _configuration_hash(payload)
    strategy_id = _legacy_winner_id(legacy, configuration_hash)
    now = utc_now()
    legacy_revision = max(1, int(legacy.get("revision") or 1))
    profile = {
        "_id": strategy_id,
        "name": _legacy_winner_name(legacy),
        "description": "Protected Trader winner imported without changing the production configuration.",
        "status": "winner",
        "locked": True,
        "revision": legacy_revision,
        "configuration": bson_value(payload),
        "configuration_hash": configuration_hash,
        "source_strategy_id": None,
        "source_strategy_revision": None,
        "created_at": legacy.get("created_at") or now,
        "updated_at": legacy.get("updated_at") or now,
        "promoted_at": legacy.get("updated_at") or legacy.get("created_at") or now,
        "origin_configuration_name": legacy.get("configuration_name"),
        "origin_winner_source_file": legacy.get("winner_source_file"),
        "origin_winner_configuration_hash": legacy.get("winner_configuration_hash"),
        "origin_bootstrap_source": legacy.get("bootstrap_source"),
        "origin_schema_version": legacy.get("schema_version"),
        "origin_revision": legacy_revision,
    }
    db[STRATEGY_PROFILES_COLLECTION].replace_one(
        {"_id": strategy_id}, profile, upsert=True
    )
    control = {
        "_id": CONTROL_ID,
        "revision": 1,
        "research_strategy_id": strategy_id,
        "candidate_strategy_id": None,
        "trader_winner_strategy_id": strategy_id,
        "created_at": now,
        "updated_at": now,
        "updated_by": None,
        "paper_state_reinitialization_required": False,
        "catalog_migration_source": "api-v1.13.16-production-winner",
    }
    db[STRATEGY_CONTROL_COLLECTION].replace_one(
        {"_id": CONTROL_ID}, control, upsert=True
    )
    return control


def synchronize_bundled_winner_installation(
    db: Any,
    configuration: BacktestRequest,
    *,
    note: str,
    source: str,
) -> None:
    payload = configuration.model_dump(mode="json")
    configuration_hash = _configuration_hash(payload)
    if configuration_hash != BUNDLED_WINNER_HASH:
        raise StrategyLabError("Bundled winner synchronization received an invalid hash.")
    now = utc_now()
    profile = {
        "_id": BUNDLED_WINNER_ID,
        "name": "Winner v1.13.2",
        "description": "Protected bundled XGBoost winner.",
        "status": "winner",
        "locked": True,
        "revision": 1,
        "configuration": bson_value(payload),
        "configuration_hash": configuration_hash,
        "source_strategy_id": None,
        "source_strategy_revision": None,
        "created_at": now,
        "updated_at": now,
        "promoted_at": now,
        "note": note,
        "source": source,
    }
    db[STRATEGY_PROFILES_COLLECTION].replace_one(
        {"_id": BUNDLED_WINNER_ID}, profile, upsert=True
    )
    for previous in db[STRATEGY_PROFILES_COLLECTION].find(
        {"_id": {"$ne": BUNDLED_WINNER_ID}, "status": "winner"},
        {"_id": 1},
    ):
        db[STRATEGY_PROFILES_COLLECTION].update_one(
            {"_id": previous["_id"]},
            {"$set": {"status": "former_winner", "locked": True, "updated_at": now}},
        )
    control = {
        "_id": CONTROL_ID,
        "revision": 1,
        "research_strategy_id": BUNDLED_WINNER_ID,
        "candidate_strategy_id": None,
        "trader_winner_strategy_id": BUNDLED_WINNER_ID,
        "created_at": now,
        "updated_at": now,
        "updated_by": None,
        "paper_state_reinitialization_required": True,
    }
    db[STRATEGY_CONTROL_COLLECTION].replace_one(
        {"_id": CONTROL_ID}, control, upsert=True
    )


def list_strategies(db: Any) -> dict[str, Any]:
    control = ensure_strategy_catalog(db)
    items = [
        _public_profile(item, include_configuration=False)
        for item in db[STRATEGY_PROFILES_COLLECTION]
        .find({})
        .sort([("locked", -1), ("updated_at", -1), ("name", 1)])
    ]
    return {
        "control": _control_response(db, control),
        "count": len(items),
        "items": items,
        "parameter_order": list(BacktestRequest.model_fields),
        "parameter_groups": [
            {"id": item["id"], "label": item["label"], "fields": list(item["fields"])}
            for item in STRATEGY_PARAMETER_GROUPS
        ],
        "parameter_schema": _strategy_parameter_schema(),
    }


def get_strategy(db: Any, strategy_id: str) -> dict[str, Any]:
    ensure_strategy_catalog(db)
    profile = db[STRATEGY_PROFILES_COLLECTION].find_one({"_id": strategy_id})
    if profile is None:
        raise StrategyLabNotFound("Strategy profile not found.")
    return _public_profile(profile)


def get_strategy_control(db: Any) -> dict[str, Any]:
    return _control_response(db, ensure_strategy_catalog(db))


def get_research_strategy_context(db: Any) -> tuple[BacktestRequest, dict[str, Any]]:
    control = ensure_strategy_catalog(db)
    strategy_id = str(control["research_strategy_id"])
    profile = db[STRATEGY_PROFILES_COLLECTION].find_one({"_id": strategy_id})
    if profile is None:
        raise StrategyLabNotFound("Selected backtest strategy does not exist.")
    configuration = BacktestRequest.model_validate(profile.get("configuration") or {})
    return configuration, _public_profile(profile, include_configuration=False)


def get_trader_winner_context(db: Any) -> tuple[BacktestRequest, dict[str, Any]]:
    control = ensure_strategy_catalog(db)
    strategy_id = str(control["trader_winner_strategy_id"])
    profile = db[STRATEGY_PROFILES_COLLECTION].find_one({"_id": strategy_id})
    if profile is None:
        raise StrategyLabNotFound("Trader winner strategy does not exist.")
    if not bool(profile.get("locked")):
        raise StrategyLabError("Trader winner strategy must be an immutable snapshot.")
    configuration = BacktestRequest.model_validate(profile.get("configuration") or {})
    return configuration, _public_profile(profile, include_configuration=False)


def get_trader_winner_summary(db: Any) -> dict[str, Any]:
    _, profile = get_trader_winner_context(db)
    return profile


def create_strategy(
    db: Any,
    *,
    name: str,
    description: str,
    clone_from_strategy_id: str | None,
    actor_email: str | None,
) -> dict[str, Any]:
    control = ensure_strategy_catalog(db)
    source_id = clone_from_strategy_id or str(control["research_strategy_id"])
    source = db[STRATEGY_PROFILES_COLLECTION].find_one({"_id": source_id})
    if source is None:
        raise StrategyLabNotFound("Source strategy profile not found.")
    configuration = BacktestRequest.model_validate(source.get("configuration") or {})
    now = utc_now()
    strategy_id = _profile_id(name)
    profile = {
        "_id": strategy_id,
        "name": name,
        "description": description,
        "status": "draft",
        "locked": False,
        "revision": 1,
        "configuration": bson_value(configuration.model_dump(mode="json")),
        "configuration_hash": _configuration_hash(configuration.model_dump(mode="json")),
        "source_strategy_id": source_id,
        "source_strategy_revision": int(source.get("revision") or 1),
        "created_at": now,
        "updated_at": now,
        "created_by": (actor_email or "").strip().lower() or None,
        "updated_by": (actor_email or "").strip().lower() or None,
    }
    db[STRATEGY_PROFILES_COLLECTION].insert_one(profile)
    return _public_profile(profile)


def update_strategy(
    db: Any,
    strategy_id: str,
    *,
    configuration: BacktestRequest,
    name: str,
    description: str,
    note: str,
    expected_revision: int,
    actor_email: str | None,
) -> dict[str, Any]:
    ensure_strategy_catalog(db)
    current = db[STRATEGY_PROFILES_COLLECTION].find_one({"_id": strategy_id})
    if current is None:
        raise StrategyLabNotFound("Strategy profile not found.")
    if bool(current.get("locked")):
        raise StrategyLabConflict(
            "Protected winner snapshots cannot be edited. Clone the strategy to create a test version."
        )
    current_revision = int(current.get("revision") or 1)
    if current_revision != expected_revision:
        raise StrategyLabConflict(
            f"Expected strategy revision {expected_revision}, current revision {current_revision}."
        )
    payload = configuration.model_dump(mode="json")
    now = utc_now()
    updated = db[STRATEGY_PROFILES_COLLECTION].find_one_and_update(
        {"_id": strategy_id, "revision": current_revision, "locked": {"$ne": True}},
        {
            "$set": {
                "name": name,
                "description": description,
                "configuration": bson_value(payload),
                "configuration_hash": _configuration_hash(payload),
                "status": "draft",
                "updated_at": now,
                "updated_by": (actor_email or "").strip().lower() or None,
                "last_change_note": note,
                "last_backtest_id": None,
                "last_backtest_status": None,
                "last_backtest_at": None,
                "last_backtest_revision": None,
                "candidate_at": None,
                "candidate_by": None,
                "candidate_note": None,
                "candidate_revision": None,
                "candidate_backtest_id": None,
            },
            "$inc": {"revision": 1},
        },
        return_document=ReturnDocument.AFTER,
    )
    if updated is None:
        raise StrategyLabConflict("Strategy changed before this update was applied.")

    if str(current.get("status") or "draft") == "candidate":
        db[STRATEGY_CONTROL_COLLECTION].update_one(
            {"_id": CONTROL_ID, "candidate_strategy_id": strategy_id},
            {
                "$set": {
                    "candidate_strategy_id": None,
                    "updated_at": now,
                    "updated_by": (actor_email or "").strip().lower() or None,
                },
                "$inc": {"revision": 1},
            },
        )
    return _public_profile(updated)


def _assert_no_active_backtest(db: Any) -> None:
    active = db[JOBS_COLLECTION].find_one(
        {"status": {"$in": ["queued", "running"]}}, {"_id": 0, "id": 1}
    )
    if active is not None:
        raise StrategyLabConflict(
            f"Wait for backtest {active.get('id', 'unknown')} to finish before changing strategy selection."
        )


def select_research_strategy(
    db: Any,
    strategy_id: str,
    *,
    expected_control_revision: int,
    note: str,
    actor_email: str | None,
) -> dict[str, Any]:
    _assert_no_active_backtest(db)
    control = ensure_strategy_catalog(db)
    current_revision = int(control.get("revision") or 1)
    if current_revision != expected_control_revision:
        raise StrategyLabConflict(
            f"Expected selection revision {expected_control_revision}, current revision {current_revision}."
        )
    profile = db[STRATEGY_PROFILES_COLLECTION].find_one({"_id": strategy_id})
    if profile is None:
        raise StrategyLabNotFound("Strategy profile not found.")
    now = utc_now()
    updated_control = db[STRATEGY_CONTROL_COLLECTION].find_one_and_update(
        {"_id": CONTROL_ID, "revision": current_revision},
        {
            "$set": {
                "research_strategy_id": strategy_id,
                "updated_at": now,
                "updated_by": (actor_email or "").strip().lower() or None,
                "last_selection_note": note,
            },
            "$inc": {"revision": 1},
        },
        return_document=ReturnDocument.AFTER,
    )
    if updated_control is None:
        raise StrategyLabConflict("Strategy selection changed before this update was applied.")
    return _control_response(db, updated_control)


def mark_strategy_as_candidate(
    db: Any,
    strategy_id: str,
    *,
    expected_strategy_revision: int,
    note: str,
    actor_email: str | None,
) -> dict[str, Any]:
    control = ensure_strategy_catalog(db)
    profile = db[STRATEGY_PROFILES_COLLECTION].find_one({"_id": strategy_id})
    if profile is None:
        raise StrategyLabNotFound("Strategy profile not found.")
    if bool(profile.get("locked")):
        raise StrategyLabConflict(
            "Protected lifecycle snapshots cannot be marked as candidates. Clone the strategy first."
        )
    current_revision = int(profile.get("revision") or 1)
    if current_revision != expected_strategy_revision:
        raise StrategyLabConflict(
            f"Expected strategy revision {expected_strategy_revision}, current revision {current_revision}."
        )
    current_candidate_id = str(control.get("candidate_strategy_id") or "")
    if current_candidate_id == strategy_id and str(profile.get("status") or "") == "candidate":
        raise StrategyLabConflict("This exact strategy revision is already the active candidate.")

    completed_job_id = str(profile.get("last_backtest_id") or "")
    completed_job = (
        db[JOBS_COLLECTION].find_one(
            {
                "id": completed_job_id,
                "status": "completed",
                "strategy_profile_id": strategy_id,
                "strategy_profile_revision": current_revision,
            },
            {"_id": 0, "id": 1},
        )
        if completed_job_id
        else None
    )
    if (
        str(profile.get("last_backtest_status") or "") != "completed"
        or int(profile.get("last_backtest_revision") or 0) != current_revision
        or completed_job is None
    ):
        raise StrategyLabConflict(
            "Run and complete a backtest with the current strategy revision before marking it as a candidate."
        )

    now = utc_now()
    actor = (actor_email or "").strip().lower() or None
    control_revision = int(control.get("revision") or 1)
    updated_control = db[STRATEGY_CONTROL_COLLECTION].find_one_and_update(
        {"_id": CONTROL_ID, "revision": control_revision},
        {
            "$set": {
                "candidate_strategy_id": strategy_id,
                "updated_at": now,
                "updated_by": actor,
                "last_candidate_note": note,
            },
            "$inc": {"revision": 1},
        },
        return_document=ReturnDocument.AFTER,
    )
    if updated_control is None:
        raise StrategyLabConflict("Candidate selection changed before this update was applied.")

    db[STRATEGY_PROFILES_COLLECTION].update_many(
        {"_id": {"$ne": strategy_id}, "status": "candidate"},
        {
            "$set": {
                "status": "superseded_candidate",
                "locked": True,
                "superseded_at": now,
                "superseded_by_strategy_id": strategy_id,
                "superseded_by": actor,
                "supersession_note": note,
                "updated_at": now,
                "updated_by": actor,
            }
        },
    )

    updated = db[STRATEGY_PROFILES_COLLECTION].find_one_and_update(
        {"_id": strategy_id, "revision": current_revision, "locked": {"$ne": True}},
        {
            "$set": {
                "status": "candidate",
                "candidate_at": now,
                "candidate_by": actor,
                "candidate_note": note,
                "candidate_revision": current_revision,
                "candidate_backtest_id": completed_job_id,
                "updated_at": now,
                "updated_by": actor,
            }
        },
        return_document=ReturnDocument.AFTER,
    )
    if updated is None:
        db[STRATEGY_CONTROL_COLLECTION].update_one(
            {
                "_id": CONTROL_ID,
                "revision": control_revision + 1,
                "candidate_strategy_id": strategy_id,
            },
            {
                "$set": {
                    "candidate_strategy_id": current_candidate_id or None,
                    "updated_at": utc_now(),
                },
                "$inc": {"revision": 1},
            },
        )
        raise StrategyLabConflict("Strategy changed before candidate status was applied.")

    db[STRATEGY_PROMOTION_HISTORY_COLLECTION].insert_one(
        bson_value(
            {
                "action": "candidate_replaced" if current_candidate_id else "candidate_marked",
                "previous_candidate_strategy_id": current_candidate_id or None,
                "new_candidate_strategy_id": strategy_id,
                "strategy_revision": current_revision,
                "backtest_id": completed_job_id,
                "note": note,
                "created_at": now,
                "actor_email": actor,
            }
        )
    )
    return _public_profile(updated)



def _acquire_winner_promotion_lock(
    db: Any,
    *,
    expected_control_revision: int,
    actor_email: str | None,
) -> dict[str, Any]:
    now = utc_now()
    actor = (actor_email or "").strip().lower() or None
    locked = db[STRATEGY_CONTROL_COLLECTION].find_one_and_update(
        {
            "_id": CONTROL_ID,
            "revision": expected_control_revision,
            "winner_promotion_in_progress": {"$ne": True},
        },
        {
            "$set": {
                "winner_promotion_in_progress": True,
                "winner_promotion_started_at": now,
                "winner_promotion_started_by": actor,
            }
        },
        return_document=ReturnDocument.AFTER,
    )
    if locked is None:
        raise StrategyLabConflict(
            "Strategy selection changed or another Winner promotion is already in progress."
        )
    return locked


def _release_winner_promotion_lock(
    db: Any,
    *,
    expected_control_revision: int,
) -> None:
    db[STRATEGY_CONTROL_COLLECTION].update_one(
        {"_id": CONTROL_ID, "revision": expected_control_revision},
        {
            "$set": {
                "winner_promotion_in_progress": False,
                "winner_promotion_started_at": None,
                "winner_promotion_started_by": None,
            }
        },
    )


def _regular_market_is_open() -> bool:
    """Return XNYS regular-session state without contacting the broker."""

    stamp = pd.Timestamp(utc_now())
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    calendar = xcals.get_calendar("XNYS")
    return bool(calendar.is_open_on_minute(stamp.floor("min"), ignore_breaks=True))


def _assert_trader_safe_for_promotion(
    db: Any,
    *,
    candidate_assets: list[str],
) -> dict[str, Any]:
    """Validate a metadata-only Winner handoff without contacting Alpaca.

    An armed run that is only waiting for the configured pre-market analysis window
    is intentionally preserved. The next preparation cycle loads the then-current
    immutable Trader Winner, so every asset in the promoted snapshot participates.
    """

    if _regular_market_is_open():
        raise StrategyLabConflict(
            "Winner promotion is allowed only while the XNYS regular market is closed. "
            "No broker request was performed."
        )

    _assert_no_active_backtest(db)

    active_run = db[PAPER_MARKET_RUNS_COLLECTION].find_one(
        {"active_key": ACTIVE_PAPER_KEY},
        {
            "_id": 0,
            "run_id": 1,
            "status": 1,
            "phase": 1,
            "plan_id": 1,
            "execution_session": 1,
            "premarket_analysis_at": 1,
        },
    )
    if active_run is not None:
        status = str(active_run.get("status") or "").strip().lower()
        if status != "armed":
            raise StrategyLabConflict(
                "Winner promotion requires the Paper pipeline to be idle before calibration, "
                "prediction or order execution. Current run status: "
                f"{status or 'unknown'}."
            )
        if active_run.get("plan_id"):
            raise StrategyLabConflict(
                "A Paper plan already exists for the next session. Promote only before the "
                "scheduled pre-market calibration and prediction cycle starts."
            )

    pending_plan = db[PAPER_TRADE_PLANS_COLLECTION].find_one(
        {"status": {"$in": ["prepared", "executing", "submitted", "pending"]}},
        {"_id": 0, "plan_id": 1, "status": 1},
    )
    if pending_plan is not None:
        raise StrategyLabConflict(
            "A Paper plan is already pending or executing. Promote only before the next "
            "scheduled calibration and prediction cycle."
        )

    state = db[PAPER_TRADING_STATE_COLLECTION].find_one({"_id": "default"}) or {}
    managed_symbol = str(state.get("managed_symbol") or "").strip().upper() or None
    normalized_assets = {str(symbol).strip().upper() for symbol in candidate_assets}
    if managed_symbol and managed_symbol not in normalized_assets:
        raise StrategyLabConflict(
            "The currently managed position is not part of the Candidate asset universe: "
            f"{managed_symbol}. Promotion was blocked without contacting Alpaca or changing the position."
        )

    controller = db[PAPER_MARKET_AUTOMATION_COLLECTION].find_one({"_id": "default"}) or {}
    return {
        "trader_control_mode": str(controller.get("control_mode") or "stopped").strip().lower(),
        "active_run_id": (active_run or {}).get("run_id"),
        "active_run_status": (active_run or {}).get("status"),
        "active_run_phase": (active_run or {}).get("phase"),
        "active_run_execution_session": (active_run or {}).get("execution_session"),
        "active_run_premarket_analysis_at": bson_value(
            (active_run or {}).get("premarket_analysis_at")
        ),
        "managed_symbol": managed_symbol,
        "managed_quantity": float(state.get("managed_quantity") or 0.0),
        "strategy_cash": float(state.get("strategy_cash") or 0.0),
        "holding_sessions": int(state.get("holding_sessions") or 0),
    }

def promote_strategy_to_trader(
    db: Any,
    strategy_id: str,
    *,
    expected_control_revision: int,
    expected_strategy_revision: int,
    note: str,
    actor_email: str | None,
) -> dict[str, Any]:
    control = ensure_strategy_catalog(db)
    control_revision = int(control.get("revision") or 1)
    if control_revision != expected_control_revision:
        raise StrategyLabConflict(
            f"Expected selection revision {expected_control_revision}, current revision {control_revision}."
        )
    if strategy_id == str(control.get("trader_winner_strategy_id") or ""):
        raise StrategyLabConflict("This strategy is already the active Trader winner.")

    source = db[STRATEGY_PROFILES_COLLECTION].find_one({"_id": strategy_id})
    if source is None:
        raise StrategyLabNotFound("Strategy profile not found.")
    source_revision = int(source.get("revision") or 1)
    if source_revision != expected_strategy_revision:
        raise StrategyLabConflict(
            f"Expected strategy revision {expected_strategy_revision}, current revision {source_revision}."
        )
    if str(source.get("status") or "draft") != "candidate":
        raise StrategyLabConflict(
            "Mark the validated strategy revision as a candidate before promotion."
        )
    if str(control.get("candidate_strategy_id") or "") != strategy_id:
        raise StrategyLabConflict("Only the single active candidate can be promoted.")

    candidate_backtest_id = str(source.get("candidate_backtest_id") or "")
    completed_job = (
        db[JOBS_COLLECTION].find_one(
            {
                "id": candidate_backtest_id,
                "status": "completed",
                "strategy_profile_id": strategy_id,
                "strategy_profile_revision": source_revision,
            },
            {
                "_id": 0,
                "id": 1,
                "strategy_configuration_hash": 1,
            },
        )
        if candidate_backtest_id
        else None
    )
    if int(source.get("candidate_revision") or 0) != source_revision or completed_job is None:
        raise StrategyLabConflict(
            "Candidate certification does not match the current strategy revision."
        )

    configuration = BacktestRequest.model_validate(source.get("configuration") or {})
    payload = configuration.model_dump(mode="json")
    configuration_hash = _configuration_hash(payload)
    stored_hash = str(source.get("configuration_hash") or "")
    if stored_hash and stored_hash != configuration_hash:
        raise StrategyLabConflict(
            "Candidate configuration hash does not match its stored immutable revision."
        )
    job_hash = str((completed_job or {}).get("strategy_configuration_hash") or "")
    if job_hash and job_hash != configuration_hash:
        raise StrategyLabConflict(
            "Candidate backtest hash does not match the configuration being promoted."
        )

    winner_name = f"Winner v{API_VERSION}"
    existing_release_winner = db[STRATEGY_PROFILES_COLLECTION].find_one(
        {"name": winner_name}
    )
    if existing_release_winner is not None:
        raise StrategyLabConflict(
            f"{winner_name} already exists. Each API release can officialize only one Winner snapshot."
        )

    actor = (actor_email or "").strip().lower() or None
    _acquire_winner_promotion_lock(
        db,
        expected_control_revision=control_revision,
        actor_email=actor,
    )
    winner_id: str | None = None
    promotion_history_id: str | None = None
    previous_winner_id: str | None = None
    previous_transitioned = False
    source_transitioned = False
    promotion_completed = False
    try:
        operational_snapshot = _assert_trader_safe_for_promotion(
            db,
            candidate_assets=list(configuration.assets),
        )
        now = utc_now()
        winner_id = (
            f"winner-v{API_VERSION.replace('.', '-')}-"
            f"{configuration_hash[:8]}-{uuid.uuid4().hex[:6]}"
        )
        winner = {
            "_id": winner_id,
            "name": winner_name,
            "description": str(source.get("description") or ""),
            "status": "winner",
            "locked": True,
            "revision": 1,
            "configuration": bson_value(payload),
            "configuration_hash": configuration_hash,
            "source_strategy_id": strategy_id,
            "source_strategy_revision": source_revision,
            "source_candidate_backtest_id": candidate_backtest_id,
            "winner_api_version": API_VERSION,
            "created_at": now,
            "updated_at": now,
            "promoted_at": now,
            "promoted_by": actor,
            "promotion_note": note,
            "promotion_mode": "metadata_only_operational_state_preserved",
            "market_closed_confirmed": True,
            "broker_interaction_performed": False,
            "operational_state_preserved": True,
        }
        db[STRATEGY_PROFILES_COLLECTION].insert_one(winner)

        previous_winner_id = str(control.get("trader_winner_strategy_id") or "")
        promotion_history_id = f"promotion-{uuid.uuid4().hex}"
        previous_updated = db[STRATEGY_PROFILES_COLLECTION].update_one(
            {"_id": previous_winner_id, "status": "winner"},
            {
                "$set": {
                    "status": "former_winner",
                    "locked": True,
                    "updated_at": now,
                    "superseded_by_winner_strategy_id": winner_id,
                }
            },
        )
        previous_transitioned = previous_updated.matched_count == 1
        source_updated = db[STRATEGY_PROFILES_COLLECTION].update_one(
            {
                "_id": strategy_id,
                "revision": source_revision,
                "status": "candidate",
            },
            {
                "$set": {
                    "status": "promoted_candidate",
                    "locked": True,
                    "last_promoted_winner_strategy_id": winner_id,
                    "last_promoted_at": now,
                    "last_promoted_by": actor,
                    "updated_at": now,
                    "updated_by": actor,
                }
            },
        )
        source_transitioned = source_updated.matched_count == 1
        if not previous_transitioned or not source_transitioned:
            raise StrategyLabConflict(
                "Winner or Candidate lifecycle changed before promotion could be committed."
            )

        db[STRATEGY_PROMOTION_HISTORY_COLLECTION].insert_one(
            bson_value(
                {
                    "_id": promotion_history_id,
                    "status": "pending_control_commit",
                    "action": "winner_promoted_preserving_operational_state",
                    "previous_winner_strategy_id": previous_winner_id,
                    "new_winner_strategy_id": winner_id,
                    "new_winner_name": winner_name,
                    "winner_api_version": API_VERSION,
                    "source_strategy_id": strategy_id,
                    "source_strategy_revision": source_revision,
                    "candidate_backtest_id": candidate_backtest_id,
                    "configuration_hash": configuration_hash,
                    "assets_count": len(configuration.assets),
                    "note": note,
                    "promoted_at": now,
                    "promoted_by": actor,
                    "market_closed_confirmed": True,
                    "broker_interaction_performed": False,
                    "operational_state_preserved": True,
                    "next_scheduled_evaluation_uses_new_winner": True,
                    "operational_snapshot": operational_snapshot,
                }
            )
        )

        updated_control = db[STRATEGY_CONTROL_COLLECTION].find_one_and_update(
            {
                "_id": CONTROL_ID,
                "revision": control_revision,
                "winner_promotion_in_progress": True,
            },
            {
                "$set": {
                    "candidate_strategy_id": None,
                    "trader_winner_strategy_id": winner_id,
                    "updated_at": now,
                    "updated_by": actor,
                    "last_promotion_note": note,
                    "last_promotion_mode": "metadata_only_operational_state_preserved",
                    "last_promoted_api_version": API_VERSION,
                    "last_promoted_configuration_hash": configuration_hash,
                    "last_promoted_assets_count": len(configuration.assets),
                    "paper_state_reinitialization_required": False,
                    "winner_promotion_in_progress": False,
                    "winner_promotion_started_at": None,
                    "winner_promotion_started_by": None,
                },
                "$inc": {"revision": 1},
            },
            return_document=ReturnDocument.AFTER,
        )
        if updated_control is None:
            raise StrategyLabConflict(
                "Winner selection changed before the metadata-only promotion completed."
            )

        # The compare-and-set above is the logical commit point. From this line on,
        # operational state and broker state remain untouched and the new immutable
        # Winner is authoritative for the next scheduled model cycle.
        promotion_completed = True
        try:
            db[STRATEGY_PROMOTION_HISTORY_COLLECTION].update_one(
                {"_id": promotion_history_id},
                {
                    "$set": {
                        "status": "completed",
                        "control_revision_after": int(updated_control.get("revision") or 0),
                        "completed_at": utc_now(),
                    }
                },
            )
        except Exception:
            # The control pointer has already committed. Keep the pending audit row
            # instead of reporting a false promotion failure after a successful handoff.
            pass
        return {
            "status": "promoted",
            "winner": _public_profile(winner),
            "control": _control_response(db, updated_control),
            "promotion": {
                "mode": "metadata_only_operational_state_preserved",
                "market_closed_confirmed": True,
                "broker_interaction_performed": False,
                "operational_state_preserved": True,
                "current_position_preserved": True,
                "paper_pipeline_preserved": True,
                "next_scheduled_evaluation_uses_new_winner": True,
                "next_scheduled_evaluation_assets_count": len(configuration.assets),
                **operational_snapshot,
            },
        }
    except Exception:
        if not promotion_completed:
            if previous_transitioned and previous_winner_id:
                db[STRATEGY_PROFILES_COLLECTION].update_one(
                    {"_id": previous_winner_id},
                    {
                        "$set": {
                            "status": "winner",
                            "locked": True,
                            "updated_at": utc_now(),
                            "superseded_by_winner_strategy_id": None,
                        }
                    },
                )
            if source_transitioned:
                db[STRATEGY_PROFILES_COLLECTION].update_one(
                    {"_id": strategy_id, "revision": source_revision},
                    {
                        "$set": {
                            "status": "candidate",
                            "locked": False,
                            "last_promoted_winner_strategy_id": None,
                            "last_promoted_at": None,
                            "last_promoted_by": None,
                            "updated_at": utc_now(),
                        }
                    },
                )
            if promotion_history_id:
                db[STRATEGY_PROMOTION_HISTORY_COLLECTION].delete_one(
                    {"_id": promotion_history_id}
                )
            if winner_id:
                db[STRATEGY_PROFILES_COLLECTION].delete_one({"_id": winner_id})
        raise
    finally:
        if not promotion_completed:
            _release_winner_promotion_lock(
                db,
                expected_control_revision=control_revision,
            )

def delete_strategy(
    db: Any,
    strategy_id: str,
    *,
    note: str,
    actor_email: str | None,
) -> dict[str, Any]:
    control = ensure_strategy_catalog(db)
    if strategy_id in {
        str(control.get("research_strategy_id")),
        str(control.get("trader_winner_strategy_id")),
    }:
        raise StrategyLabConflict(
            "A selected backtest strategy or Trader winner cannot be deleted."
        )
    profile = db[STRATEGY_PROFILES_COLLECTION].find_one({"_id": strategy_id})
    if profile is None:
        raise StrategyLabNotFound("Strategy profile not found.")
    if bool(profile.get("locked")):
        raise StrategyLabConflict("Protected lifecycle snapshots cannot be deleted.")
    if str(profile.get("status") or "draft") != "draft":
        raise StrategyLabConflict(
            "Only draft strategies can be deleted. Candidate and winner history is preserved for audit."
        )
    active_job = db[JOBS_COLLECTION].find_one(
        {
            "status": {"$in": ["queued", "running"]},
            "strategy_profile_id": strategy_id,
        },
        {"_id": 0, "id": 1},
    )
    if active_job is not None:
        raise StrategyLabConflict(
            f"Wait for backtest {active_job.get('id', 'unknown')} to finish before deleting this strategy."
        )
    result = db[STRATEGY_PROFILES_COLLECTION].delete_one({"_id": strategy_id})
    if result.deleted_count != 1:
        raise StrategyLabConflict("Strategy was not deleted.")
    db[STRATEGY_PROMOTION_HISTORY_COLLECTION].insert_one(
        bson_value(
            {
                "action": "research_strategy_deleted",
                "strategy_status": profile.get("status") or "draft",
                "strategy_id": strategy_id,
                "strategy_name": profile.get("name"),
                "note": note,
                "created_at": utc_now(),
                "actor_email": (actor_email or "").strip().lower() or None,
            }
        )
    )
    return {"status": "deleted", "strategy_id": strategy_id}


def trader_winner_requires_state_reinitialization(db: Any) -> bool:
    control = ensure_strategy_catalog(db)
    return bool(control.get("paper_state_reinitialization_required"))


def mark_trader_winner_state_initialized(db: Any) -> None:
    control = ensure_strategy_catalog(db)
    winner_id = str(control.get("trader_winner_strategy_id") or "")
    winner = db[STRATEGY_PROFILES_COLLECTION].find_one({"_id": winner_id}) or {}
    db[STRATEGY_CONTROL_COLLECTION].update_one(
        {"_id": CONTROL_ID},
        {
            "$set": {
                "paper_state_reinitialization_required": False,
                "paper_state_winner_strategy_id": winner_id,
                "paper_state_winner_configuration_hash": winner.get("configuration_hash"),
                "paper_state_initialized_at": utc_now(),
            }
        },
    )


def mark_strategy_backtest(
    db: Any,
    *,
    strategy_id: str | None,
    strategy_revision: int | None,
    job_id: str,
    status: str,
) -> None:
    if not strategy_id or not strategy_revision:
        return
    # Update only the exact revision that produced the job. If an Administrator
    # edits the draft while the backtest is running, the old result must not
    # certify the newer revision for Trader promotion.
    db[STRATEGY_PROFILES_COLLECTION].update_one(
        {"_id": strategy_id, "revision": int(strategy_revision)},
        {
            "$set": {
                "last_backtest_id": job_id,
                "last_backtest_status": status,
                "last_backtest_at": utc_now(),
                "last_backtest_revision": int(strategy_revision),
            }
        },
    )
