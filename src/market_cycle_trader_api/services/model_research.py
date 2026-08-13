from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any

from pydantic import BaseModel
from pymongo import ReturnDocument

from ..core.config import API_VERSION
from ..schemas.model_research import (
    IQNResearchSettings,
    LightGBMResearchSettings,
    XGBoostResearchSettings,
    ModelResearchSettingsUpdateRequest,
)
from ..infrastructure.persistence.mongo_repository import (
    JOBS_COLLECTION,
    MODEL_RESEARCH_SETTINGS_COLLECTION,
    MODEL_RESEARCH_SETTINGS_HISTORY_COLLECTION,
    bson_value,
    utc_now,
)

_SETTINGS_ID = "default"
_SETTINGS_SCHEMA_VERSION = 3
_PROFILE_ID = "baseline"

_XGBOOST_DEFAULTS: dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.035,
    "max_depth": 3,
    "min_child_weight": 5.0,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 2.0,
    "n_jobs": -1,
    "repetitions": 1,
    "seed_step": 1000,
    "random_state": 42,
}




_LIGHTGBM_DEFAULTS: dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.035,
    "max_depth": 3,
    "num_leaves": 8,
    "min_child_samples": 20,
    "min_child_weight": 5.0,
    "subsample": 0.85,
    "subsample_freq": 0,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 2.0,
    "max_bin": 255,
    "n_jobs": -1,
    "repetitions": 1,
    "seed_step": 1000,
    "random_state": 42,
}

_IQN_DEFAULTS: dict[str, Any] = {
    "training_steps": 15000,
    "episode_days": 252,
    "replay_size": 30000,
    "learning_starts": 750,
    "batch_size": 128,
    "learning_rate": 0.0003,
    "gamma": 0.99,
    "n_step": 5,
    "quantile_samples": 32,
    "target_quantile_samples": 32,
    "action_quantile_samples": 32,
    "evaluation_quantiles": 64,
    "hidden_dim": 128,
    "cosine_embedding_dim": 64,
    "target_update_steps": 250,
    "eval_every_steps": 1000,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "early_stopping_enabled": False,
    "early_stopping_patience": 4,
    "minimum_training_steps": 5000,
    "gradient_clip_norm": 10.0,
    "huber_kappa": 1.0,
    "repetitions": 1,
    "seed_step": 1000,
    "random_state": 42,
}

_FIELD_METADATA: dict[str, dict[str, dict[str, Any]]] = {
    "xgboost_utility": {
        "n_estimators": {"label": "Estimators", "step": 1, "description": "Maximum number of XGBoost trees trained for each asset model."},
        "learning_rate": {"label": "Learning rate", "step": 0.001, "description": "Boosting shrinkage applied to each new XGBoost tree."},
        "max_depth": {"label": "Maximum depth", "step": 1, "description": "Maximum depth of each XGBoost tree."},
        "min_child_weight": {"label": "Minimum child weight", "step": 0.1, "description": "Minimum child-weight regularization used by XGBoost splits."},
        "subsample": {"label": "Row subsample", "step": 0.01, "description": "Fraction of training rows sampled for each XGBoost tree."},
        "colsample_bytree": {"label": "Feature fraction", "step": 0.01, "description": "Fraction of features sampled for each XGBoost tree."},
        "reg_alpha": {"label": "L1 regularization", "step": 0.01, "description": "L1 regularization strength applied to XGBoost leaf weights."},
        "reg_lambda": {"label": "L2 regularization", "step": 0.01, "description": "L2 regularization strength applied to XGBoost leaf weights."},
        "n_jobs": {"label": "CPU workers", "step": 1, "description": "CPU workers used by XGBoost. -1 uses all CPUs visible to the runtime."},
        "repetitions": {"label": "Repetitions", "step": 1, "description": "Independent XGBoost training repetitions used by this model profile."},
        "seed_step": {"label": "Seed step", "step": 1, "description": "Increment between XGBoost repetition seeds."},
        "random_state": {"label": "Random state", "step": 1, "description": "Base random seed used by this XGBoost research profile."},
    },
    "lightgbm_utility": {
        "n_estimators": {"label": "Estimators", "step": 1, "description": "Maximum number of boosting trees trained for each asset model."},
        "learning_rate": {"label": "Learning rate", "step": 0.001, "description": "Boosting shrinkage applied to each new tree."},
        "max_depth": {"label": "Maximum depth", "step": 1, "description": "Maximum tree depth. -1 allows unrestricted depth subject to the other leaf controls."},
        "num_leaves": {"label": "Number of leaves", "step": 1, "description": "Maximum leaves per tree; a primary LightGBM complexity control."},
        "min_child_samples": {"label": "Minimum child samples", "step": 1, "description": "Minimum training rows required in a leaf."},
        "min_child_weight": {"label": "Minimum child weight", "step": 0.1, "description": "Minimum Hessian weight required in a leaf."},
        "subsample": {"label": "Row subsample", "step": 0.01, "description": "Fraction of training rows available to a boosting iteration when bagging is enabled."},
        "subsample_freq": {"label": "Subsample frequency", "step": 1, "description": "Bagging frequency. 0 preserves the original v1.13.39 baseline with bagging disabled."},
        "colsample_bytree": {"label": "Feature fraction", "step": 0.01, "description": "Fraction of available features sampled for each tree."},
        "reg_alpha": {"label": "L1 regularization", "step": 0.01, "description": "L1 regularization strength on leaf weights."},
        "reg_lambda": {"label": "L2 regularization", "step": 0.01, "description": "L2 regularization strength on leaf weights."},
        "max_bin": {"label": "Maximum bins", "step": 1, "description": "Maximum number of histogram bins used for numeric features."},
        "n_jobs": {"label": "CPU workers", "step": 1, "description": "CPU workers used by LightGBM. -1 uses all CPUs visible to the Railway/container runtime."},
        "repetitions": {"label": "Repetitions", "step": 1, "description": "Independent LightGBM training repetitions used by this model profile."},
        "seed_step": {"label": "Seed step", "step": 1, "description": "Increment between LightGBM repetition seeds."},
        "random_state": {"label": "Random state", "step": 1, "description": "Base random seed used by this LightGBM research profile."},
    },
    "iqn": {
        "training_steps": {"label": "Training steps", "step": 500, "description": "Maximum optimizer/environment steps for each walk-forward fold."},
        "episode_days": {"label": "Episode days", "step": 1, "description": "Maximum trading sessions sampled for a training episode."},
        "replay_size": {"label": "Replay capacity", "step": 1000, "description": "Maximum transitions retained in replay memory."},
        "learning_starts": {"label": "Learning starts", "step": 50, "description": "Transitions collected before gradient updates begin."},
        "batch_size": {"label": "Batch size", "step": 16, "description": "Replay transitions used in one optimizer update."},
        "learning_rate": {"label": "Learning rate", "step": 0.00001, "description": "Adam optimizer learning rate."},
        "gamma": {"label": "Gamma", "step": 0.001, "description": "Discount factor applied to future n-step rewards."},
        "n_step": {"label": "N-step", "step": 1, "description": "Number of transitions accumulated into each return target."},
        "quantile_samples": {"label": "Online quantiles", "step": 4, "description": "Quantile samples used by the online network during training."},
        "target_quantile_samples": {"label": "Target quantiles", "step": 4, "description": "Quantile samples used to build target distributions."},
        "action_quantile_samples": {"label": "Action quantiles", "step": 4, "description": "Quantile samples used for action selection during learning."},
        "evaluation_quantiles": {"label": "Evaluation quantiles", "step": 8, "description": "Quantiles used when evaluating a trained policy."},
        "hidden_dim": {"label": "Hidden dimension", "step": 16, "description": "Width of the IQN hidden representation."},
        "cosine_embedding_dim": {"label": "Cosine embedding", "step": 8, "description": "Dimension of the cosine quantile embedding."},
        "target_update_steps": {"label": "Target update interval", "step": 10, "description": "Optimizer steps between target-network synchronizations."},
        "eval_every_steps": {"label": "Evaluation interval", "step": 100, "description": "Training steps between validation evaluations/checkpoints."},
        "epsilon_start": {"label": "Epsilon start", "step": 0.01, "description": "Initial epsilon used by exploration during training."},
        "epsilon_end": {"label": "Epsilon end", "step": 0.01, "description": "Final epsilon after exploration decay."},
        "early_stopping_enabled": {"label": "Early stopping", "description": "Stop after the validation score stops improving, but only after the configured minimum training steps."},
        "early_stopping_patience": {"label": "Early stopping patience", "step": 1, "description": "Number of validation checks without improvement allowed before stopping."},
        "minimum_training_steps": {"label": "Minimum training steps", "step": 500, "description": "Minimum steps that must run before early stopping may finish a fold."},
        "gradient_clip_norm": {"label": "Gradient clip norm", "step": 0.5, "description": "Maximum gradient norm used to stabilize neural-network updates."},
        "huber_kappa": {"label": "Huber kappa", "step": 0.1, "description": "Huber-loss transition point used by quantile regression."},
        "repetitions": {"label": "Repetitions", "step": 1, "description": "Independent IQN training repetitions used by this model profile."},
        "seed_step": {"label": "Seed step", "step": 1, "description": "Increment between IQN repetition seeds."},
        "random_state": {"label": "Random state", "step": 1, "description": "Base random seed used by this IQN research profile."},
    },
}


class ModelResearchSettingsConflict(RuntimeError):
    pass


def _default_document() -> dict[str, Any]:
    now = utc_now()
    return {
        "_id": _SETTINGS_ID,
        "schema_version": _SETTINGS_SCHEMA_VERSION,
        "revision": 1,
        "created_at": now,
        "updated_at": now,
        "updated_by": None,
        "seeded_api_version": API_VERSION,
        "profiles": {
            "xgboost_utility": {"active_profile_id": _PROFILE_ID, "profiles": {_PROFILE_ID: deepcopy(_XGBOOST_DEFAULTS)}},
            "lightgbm_utility": {"active_profile_id": _PROFILE_ID, "profiles": {_PROFILE_ID: deepcopy(_LIGHTGBM_DEFAULTS)}},
            "iqn": {"active_profile_id": _PROFILE_ID, "profiles": {_PROFILE_ID: deepcopy(_IQN_DEFAULTS)}},
        },
    }


def _validate_values(model_family: str, values: dict[str, Any]) -> BaseModel:
    if model_family == "xgboost_utility":
        return XGBoostResearchSettings.model_validate(values)
    if model_family == "lightgbm_utility":
        return LightGBMResearchSettings.model_validate(values)
    if model_family == "iqn":
        return IQNResearchSettings.model_validate(values)
    raise ValueError(f"Unsupported editable research model: {model_family}")


def _normalized_document(document: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(document)
    normalized["schema_version"] = _SETTINGS_SCHEMA_VERSION
    normalized["revision"] = max(1, int(normalized.get("revision") or 1))

    
    
    legacy_iqn = normalized.get("iqn") if isinstance(normalized.get("iqn"), dict) else {}
    raw_profiles = normalized.get("profiles") if isinstance(normalized.get("profiles"), dict) else {}

    result_profiles: dict[str, Any] = {}
    for family, defaults in (("xgboost_utility", _XGBOOST_DEFAULTS), ("lightgbm_utility", _LIGHTGBM_DEFAULTS), ("iqn", _IQN_DEFAULTS)):
        raw_family = raw_profiles.get(family) if isinstance(raw_profiles.get(family), dict) else {}
        active_profile_id = str(raw_family.get("active_profile_id") or _PROFILE_ID).strip() or _PROFILE_ID
        raw_family_profiles = raw_family.get("profiles") if isinstance(raw_family.get("profiles"), dict) else {}
        raw_baseline = raw_family_profiles.get(_PROFILE_ID) if isinstance(raw_family_profiles.get(_PROFILE_ID), dict) else {}
        if family == "iqn" and not raw_baseline:
            raw_baseline = legacy_iqn
        baseline = _validate_values(family, {**defaults, **raw_baseline}).model_dump(mode="python")
        result_profiles[family] = {
            "active_profile_id": _PROFILE_ID if active_profile_id not in raw_family_profiles else active_profile_id,
            "profiles": {_PROFILE_ID: baseline},
        }
    normalized["profiles"] = result_profiles
    normalized.pop("iqn", None)
    return normalized


def get_model_research_settings(db: Any) -> dict[str, Any]:
    collection = db[MODEL_RESEARCH_SETTINGS_COLLECTION]
    document = collection.find_one({"_id": _SETTINGS_ID})
    if document is None:
        document = _default_document()
        collection.insert_one(deepcopy(document))
        return deepcopy(document)

    normalized = _normalized_document(document)
    comparable_original = deepcopy(document)
    comparable_original.pop("updated_at", None)
    comparable_normalized = deepcopy(normalized)
    comparable_normalized.pop("updated_at", None)
    if comparable_original != comparable_normalized:
        now = utc_now()
        normalized["updated_at"] = now
        collection.replace_one({"_id": _SETTINGS_ID}, deepcopy(normalized), upsert=True)
    return deepcopy(normalized)


def _active_profile(document: dict[str, Any], model_family: str) -> tuple[str, dict[str, Any]]:
    family = document.get("profiles", {}).get(model_family)
    if not isinstance(family, dict):
        raise ValueError(f"No research settings are stored for {model_family}.")
    active_profile_id = str(family.get("active_profile_id") or _PROFILE_ID)
    profiles = family.get("profiles") if isinstance(family.get("profiles"), dict) else {}
    values = profiles.get(active_profile_id)
    if not isinstance(values, dict):
        raise ValueError(f"Active profile {active_profile_id!r} is missing for {model_family}.")
    return active_profile_id, _validate_values(model_family, values).model_dump(mode="python")


def execution_settings_for(db: Any, model_family: str) -> dict[str, Any]:
    if model_family not in {"xgboost_utility", "lightgbm_utility", "iqn"}:
        return {}
    document = get_model_research_settings(db)
    profile_id, values = _active_profile(document, model_family)
    key = {"xgboost_utility": "xgboost", "lightgbm_utility": "lightgbm", "iqn": "iqn"}[model_family]
    return {
        "schema_version": int(document.get("schema_version") or _SETTINGS_SCHEMA_VERSION),
        "settings_revision": int(document.get("revision") or 1),
        "profile_id": profile_id,
        key: values,
    }


def execution_settings_from_values(
    model_family: str,
    values: dict[str, Any],
    *,
    settings_revision: int,
    profile_id: str = "strategy",
) -> dict[str, Any]:
    
    if model_family not in {"xgboost_utility", "lightgbm_utility", "iqn"}:
        raise ValueError(f"Unsupported research model: {model_family}")
    validated = _validate_values(model_family, values).model_dump(mode="python")
    key = {"xgboost_utility": "xgboost", "lightgbm_utility": "lightgbm", "iqn": "iqn"}[model_family]
    return {
        "schema_version": _SETTINGS_SCHEMA_VERSION,
        "settings_revision": max(1, int(settings_revision)),
        "profile_id": str(profile_id or "strategy"),
        key: validated,
    }


def model_values_from_snapshot(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    raw = snapshot if isinstance(snapshot, dict) else {}
    family = str(raw.get("family") or "")
    key = {"xgboost_utility": "xgboost", "lightgbm_utility": "lightgbm", "iqn": "iqn"}.get(family)
    settings = raw.get("settings_snapshot") if isinstance(raw.get("settings_snapshot"), dict) else {}
    values = settings.get(key) if key and isinstance(settings.get(key), dict) else {}
    return deepcopy(values)


def model_execution_snapshot(
    model_family: str,
    settings_snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    







    if model_family not in {"xgboost_utility", "lightgbm_utility", "iqn"}:
        raise ValueError(f"Unsupported research model: {model_family}")
    raw = deepcopy(settings_snapshot) if isinstance(settings_snapshot, dict) else {}
    key = {"xgboost_utility": "xgboost", "lightgbm_utility": "lightgbm", "iqn": "iqn"}[model_family]
    values = raw.get(key) if isinstance(raw.get(key), dict) else {}
    if values:
        validated = _validate_values(model_family, values).model_dump(mode="python")
        raw[key] = validated
        source = "model_research_profile"
    elif model_family == "xgboost_utility":
        
        raw = {}
        source = "legacy_strategy_owned"
    else:
        raise ValueError(
            f"{model_label(model_family)} requires an immutable model settings snapshot."
        )

    encoded = json.dumps(raw, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return {
        "family": model_family,
        "label": model_label(model_family),
        "profile_id": str(raw.get("profile_id") or "legacy") if raw else "legacy",
        "settings_revision": int(raw.get("settings_revision") or 0) if raw else 0,
        "schema_version": int(raw.get("schema_version") or 0) if raw else 0,
        "settings_hash": hashlib.sha256(encoded).hexdigest(),
        "settings_snapshot": raw,
        "source": source,
    }


def public_model_snapshot(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    raw = snapshot if isinstance(snapshot, dict) else {}
    family = str(raw.get("family") or "xgboost_utility")
    return {
        "family": family,
        "label": str(raw.get("label") or model_label(family)),
        "profile_id": str(raw.get("profile_id") or "legacy"),
        "settings_revision": int(raw.get("settings_revision") or 0),
        "settings_hash": str(raw.get("settings_hash") or ""),
        "source": str(raw.get("source") or "legacy_strategy_owned"),
    }


def apply_execution_profile(config: Any, model_family: str, settings_snapshot: dict[str, Any] | None) -> Any:
    




    snapshot = settings_snapshot if isinstance(settings_snapshot, dict) else {}
    key = {"xgboost_utility": "xgboost", "lightgbm_utility": "lightgbm", "iqn": "iqn"}.get(model_family)
    values = snapshot.get(key) if key and isinstance(snapshot.get(key), dict) else {}
    if not values:
        return config

    updates: dict[str, Any] = {}
    if "repetitions" in values:
        updates["rotation_xgb_repetitions"] = int(values["repetitions"])
    if "seed_step" in values:
        updates["rotation_seed_step"] = int(values["seed_step"])
    if "random_state" in values:
        updates["random_state"] = int(values["random_state"])

    if model_family == "xgboost_utility":
        mapping = {
            "n_estimators": "rotation_xgb_n_estimators",
            "learning_rate": "rotation_xgb_learning_rate",
            "max_depth": "rotation_xgb_max_depth",
            "min_child_weight": "xgb_min_child_weight",
            "subsample": "xgb_subsample",
            "colsample_bytree": "xgb_colsample_bytree",
            "reg_alpha": "xgb_reg_alpha",
            "reg_lambda": "xgb_reg_lambda",
            "n_jobs": "xgb_n_jobs",
        }
        for source, target in mapping.items():
            if source in values:
                updates[target] = values[source]

    return config.model_copy(update=updates) if updates else config


def model_label(model_family: str) -> str:
    labels = {
        "xgboost_utility": "XGBoost Utility",
        "lightgbm_utility": "LightGBM Utility",
        "iqn": "IQN",
    }
    try:
        return labels[model_family]
    except KeyError as exc:
        raise ValueError(f"Unsupported research model: {model_family}") from exc


def public_model_research_catalog(db: Any) -> dict[str, Any]:
    document = get_model_research_settings(db)
    return {
        "models": [
            {"id": "xgboost_utility", "label": "XGBoost Utility", "role": "baseline"},
            {"id": "lightgbm_utility", "label": "LightGBM Utility", "role": "challenger"},
            {"id": "iqn", "label": "IQN", "role": "challenger"},
        ],
        "settings_revision": int(document.get("revision") or 1),
        "comparison_contract": {
            "same_strategy_snapshot": True,
            "same_asset_universe": True,
            "same_calendar_and_folds": True,
            "same_costs_and_execution_rules": True,
            "same_repetition_count_and_seed_schedule": True,
            "trader_winner_unchanged": True,
        },
    }


def _field_descriptor(model: BaseModel, name: str, value: Any, metadata: dict[str, Any]) -> dict[str, Any]:
    field = type(model).model_fields[name]
    descriptor: dict[str, Any] = {
        "name": name,
        "label": str(metadata.get("label") or name.replace("_", " ").title()),
        "description": str(metadata.get("description") or ""),
        "value": value,
        "type": "boolean" if isinstance(value, bool) else "integer" if isinstance(value, int) else "number",
    }
    if "step" in metadata:
        descriptor["step"] = metadata["step"]
    for item in field.metadata:
        for attr, output_name in (("ge", "min"), ("gt", "exclusive_min"), ("le", "max"), ("lt", "exclusive_max")):
            candidate = getattr(item, attr, None)
            if candidate is not None:
                descriptor[output_name] = candidate
    return descriptor


def _editable_model_payload(document: dict[str, Any], model_family: str) -> dict[str, Any]:
    profile_id, values = _active_profile(document, model_family)
    validated = _validate_values(model_family, values)
    metadata = _FIELD_METADATA[model_family]
    fields = [
        _field_descriptor(validated, name, values[name], metadata.get(name, {}))
        for name in type(validated).model_fields
    ]
    return {
        "id": model_family,
        "label": model_label(model_family),
        "role": "baseline" if model_family == "xgboost_utility" else "challenger",
        "configuration_owner": "model_research",
        "editable": True,
        "active_profile_id": profile_id,
        "profiles": [{"id": profile_id, "label": "Baseline"}],
        "fields": fields,
    }


def public_model_research_settings(db: Any) -> dict[str, Any]:
    document = get_model_research_settings(db)
    return {
        "revision": int(document.get("revision") or 1),
        "schema_version": int(document.get("schema_version") or _SETTINGS_SCHEMA_VERSION),
        "updated_at": bson_value(document.get("updated_at")),
        "updated_by": document.get("updated_by"),
        "models": [
            _editable_model_payload(document, "xgboost_utility"),
            _editable_model_payload(document, "lightgbm_utility"),
            _editable_model_payload(document, "iqn"),
        ],
    }


def update_model_research_settings(
    db: Any,
    model_family: str,
    payload: ModelResearchSettingsUpdateRequest,
    *,
    actor_email: str | None,
) -> dict[str, Any]:
    if model_family not in {"xgboost_utility", "lightgbm_utility", "iqn"}:
        raise ValueError("Unsupported research model settings.")

    previous = get_model_research_settings(db)
    current_revision = int(previous.get("revision") or 1)
    if payload.expected_revision != current_revision:
        raise ModelResearchSettingsConflict(
            f"Expected revision {payload.expected_revision}, current revision {current_revision}."
        )

    profile_id, current_values = _active_profile(previous, model_family)
    merged_values = {**current_values, **payload.values}
    validated = _validate_values(model_family, merged_values).model_dump(mode="python")
    updated_document = deepcopy(previous)
    updated_document["profiles"][model_family]["profiles"][profile_id] = validated
    now = utc_now()
    actor = str(actor_email or "").strip().lower() or None
    updated_document["updated_at"] = now
    updated_document["updated_by"] = actor
    updated_document["seeded_api_version"] = API_VERSION

    collection = db[MODEL_RESEARCH_SETTINGS_COLLECTION]
    updated = collection.find_one_and_update(
        {"_id": _SETTINGS_ID, "revision": current_revision},
        {
            "$set": {
                "schema_version": _SETTINGS_SCHEMA_VERSION,
                "profiles": updated_document["profiles"],
                "updated_at": now,
                "updated_by": actor,
                "seeded_api_version": API_VERSION,
            },
            "$inc": {"revision": 1},
            "$unset": {"iqn": ""},
        },
        return_document=ReturnDocument.AFTER,
    )
    if updated is None:
        raise ModelResearchSettingsConflict("Model research settings changed before this update was applied.")

    db[MODEL_RESEARCH_SETTINGS_HISTORY_COLLECTION].insert_one(
        bson_value({
            "settings_id": _SETTINGS_ID,
            "previous_revision": current_revision,
            "revision": current_revision + 1,
            "model_family": model_family,
            "profile_id": profile_id,
            "reason": payload.reason,
            "updated_at": now,
            "updated_by": actor,
            "values": validated,
        })
    )
    return public_model_research_settings(db)


def list_model_research_settings_history(db: Any, *, limit: int = 50) -> dict[str, Any]:
    cursor = (
        db[MODEL_RESEARCH_SETTINGS_HISTORY_COLLECTION]
        .find({"settings_id": _SETTINGS_ID})
        .sort("updated_at", -1)
        .limit(max(1, min(int(limit), 200)))
    )
    items = [
        {
            "previous_revision": int(item.get("previous_revision") or 1),
            "revision": int(item.get("revision") or 1),
            "model_family": str(item.get("model_family") or ""),
            "model_label": model_label(str(item.get("model_family") or "xgboost_utility")),
            "profile_id": str(item.get("profile_id") or _PROFILE_ID),
            "reason": str(item.get("reason") or "Model settings updated"),
            "updated_at": bson_value(item.get("updated_at")),
            "updated_by": item.get("updated_by"),
        }
        for item in cursor
    ]
    return {"count": len(items), "items": items}


def list_model_research_executions(db: Any, *, limit: int = 50) -> dict[str, Any]:
    
    safe_limit = max(1, min(100, int(limit)))
    documents = (
        db[JOBS_COLLECTION]
        .find(
            {"internal_job": {"$ne": True}},
            {
                "_id": 0,
                "id": 1,
                "research_model_family": 1,
                "research_model_label": 1,
                "created_at": 1,
            },
        )
        .sort("created_at", -1)
        .limit(safe_limit)
    )
    items: list[dict[str, Any]] = []
    for document in documents:
        job_id = str(document.get("id") or "").strip()
        if not job_id:
            continue
        family = str(document.get("research_model_family") or "xgboost_utility")
        items.append({
            "id": job_id,
            "model_family": family,
            "model_label": str(document.get("research_model_label") or model_label(family)),
        })
    return {"items": items}
