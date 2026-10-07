from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..infrastructure.persistence.mongo_repository import (
    STRATEGY_PROFILES_COLLECTION,
    utc_now,
)
from ..schemas.model_research import LightGBMResearchSettings
from ..schemas.requests import BacktestRequest
from ..tcc_u67_v1210_reference.configuracao import (
    CONFIG as TCC_U67_CONFIG,
    construir_configuracao_controle,
)
from ..tcc_u67_v1210_reference.contract import (
    EXPECTED_ENDING_CAPITAL,
    REPRODUCTION_VERSION,
    SOURCE_COMMIT,
    SOURCE_REPOSITORY,
    U67_REQUESTED_ASSETS,
)
from ..engine.tcc_u67_operational_runtime import (
    tcc_u67_strategy_updates,
)
from .strategy_lab import (
    create_strategy,
    get_research_strategy_context,
    get_strategy,
    get_strategy_control,
    select_research_strategy_only,
    update_strategy,
    update_strategy_model,
)

BACKTEST_ENGINE_BINDING = "tcc_u67_v1210_operational_backtest"
REFERENCE_ENGINE_ID = "tcc-main-u67-v1.21.0-control"


def _tcc_lightgbm_values() -> dict[str, Any]:
    control = construir_configuracao_controle(
        TCC_U67_CONFIG,
        assets=U67_REQUESTED_ASSETS,
    )
    raw = dict(
        (control.research_model_settings or {}).get("lightgbm") or {}
    )
    allowed = set(LightGBMResearchSettings.model_fields)
    return {
        key: deepcopy(value)
        for key, value in raw.items()
        if key in allowed
    }


def _mct_configuration_for_u67(
    source: BacktestRequest,
) -> BacktestRequest:
    payload = source.model_dump(mode="python")
    lightgbm = _tcc_lightgbm_values()
    payload.update(tcc_u67_strategy_updates())
    payload.update(
        {
            # Historical MCT schema token for LightGBM Utility.
            "rotation_models": ["xgboost_utility"],
            "rotation_xgb_n_estimators": int(
                lightgbm["n_estimators"]
            ),
            "rotation_xgb_learning_rate": float(
                lightgbm["learning_rate"]
            ),
            "rotation_xgb_max_depth": int(lightgbm["max_depth"]),
            "xgb_min_child_weight": float(
                lightgbm["min_child_weight"]
            ),
            "xgb_subsample": float(lightgbm["subsample"]),
            "xgb_colsample_bytree": float(
                lightgbm["colsample_bytree"]
            ),
            "xgb_reg_alpha": float(lightgbm["reg_alpha"]),
            "xgb_reg_lambda": float(lightgbm["reg_lambda"]),
            "xgb_n_jobs": int(lightgbm["n_jobs"]),
        }
    )
    return BacktestRequest.model_validate(payload)


def _existing_profile(db: Any) -> dict[str, Any] | None:
    return db[STRATEGY_PROFILES_COLLECTION].find_one(
        {
            "backtest_engine_binding": BACKTEST_ENGINE_BINDING,
            "reference_source_commit": SOURCE_COMMIT,
        }
    )


def install_tcc_u67_operational_strategy(
    db: Any,
    *,
    actor_email: str | None,
) -> dict[str, Any]:
    """Create/reuse U67 in Research only; never touch Model Tuning or Winner."""
    actor = (actor_email or "").strip().lower() or None
    existing = _existing_profile(db)
    if existing is not None:
        now = utc_now()
        db[STRATEGY_PROFILES_COLLECTION].update_one(
            {"_id": existing["_id"]},
            {
                "$set": {
                    "operational_stage": "protected_live_runtime",
                    "live_trader_eligible": True,
                    "live_runtime_version": "10.8.86",
                    "updated_at": now,
                    "updated_by": actor,
                }
            },
        )
        control = get_strategy_control(db)
        selected = select_research_strategy_only(
            db,
            str(existing["_id"]),
            expected_control_revision=int(control["revision"]),
            note=(
                "Select TCC main U67 v1.21.0 operational Strategy "
                "with protected live runtime"
            ),
            actor_email=actor,
        )
        return {
            "created": False,
            "strategy": get_strategy(db, str(existing["_id"])),
            "control": selected,
        }

    source_config, source_profile = get_research_strategy_context(db)
    created = create_strategy(
        db,
        name="TCC U67 v1.21.0 Operational",
        description=(
            "TCC main U67 v1.21.0 Control migrated for protected "
            "MCT operational backtesting"
        ),
        clone_from_strategy_id=str(source_profile["id"]),
        actor_email=actor,
    )
    strategy_id = str(created["id"])

    model_updated = update_strategy_model(
        db,
        strategy_id,
        model_family="lightgbm_utility",
        values=_tcc_lightgbm_values(),
        note="Bind exact TCC main U67 LightGBM Control parameters",
        expected_strategy_revision=int(created["revision"]),
        actor_email=actor,
    )

    config = _mct_configuration_for_u67(source_config)
    updated = update_strategy(
        db,
        strategy_id,
        configuration=config,
        name=str(created.get("name") or ""),
        description=(
            "TCC main U67 v1.21.0 Control "
            f"({SOURCE_COMMIT[:12]})"
        ),
        note="Bind exact TCC main U67 operational configuration",
        expected_revision=int(model_updated["revision"]),
        actor_email=actor,
    )

    now = utc_now()
    db[STRATEGY_PROFILES_COLLECTION].update_one(
        {
            "_id": strategy_id,
            "revision": int(updated["revision"]),
        },
        {
            "$set": {
                "backtest_engine_binding": BACKTEST_ENGINE_BINDING,
                "strategy_kind": "standard",
                "tuning_target": "protected_external_policy",
                "research_reference_assets": list(TCC_U67_CONFIG.assets),
                "reference_engine_id": REFERENCE_ENGINE_ID,
                "reference_source_repository": SOURCE_REPOSITORY,
                "reference_source_branch": "main",
                "reference_source_commit": SOURCE_COMMIT,
                "reference_reproduction_version": REPRODUCTION_VERSION,
                "reference_control_variant": "CONTROL",
                "reference_checkpoint_capital": EXPECTED_ENDING_CAPITAL,
                "reference_checkpoint_is_optimization_target": False,
                "operational_stage": "protected_live_runtime",
                "live_trader_eligible": True,
                "live_runtime_version": "10.8.86",
                "shadow_validated": False,
                "source_git_commit_message": (
                    "TCC main U67 v1.21.0 Control migration "
                    f"({SOURCE_COMMIT[:12]})"
                ),
                "updated_at": now,
                "updated_by": actor,
            },
            "$unset": {
                "source_temporal_run_id": "",
                "source_temporal_experiment": "",
                "temporal_strategy_variant": "",
                "source_stateful_replay_id": "",
                "source_stateful_processing_id": "",
                "stateful_candidate_key": "",
                "stateful_candidate_label": "",
                "temporal_policy_revision": "",
                "temporal_policy_snapshot": "",
                "derived_policy_invalidated_at": "",
                "derived_policy_invalidated_reason": "",
            },
        },
    )

    control = get_strategy_control(db)
    selected = select_research_strategy_only(
        db,
        strategy_id,
        expected_control_revision=int(control["revision"]),
        note=(
            "Create and select TCC main U67 v1.21.0 "
            "with protected live runtime"
        ),
        actor_email=actor,
    )
    return {
        "created": True,
        "strategy": get_strategy(db, strategy_id),
        "control": selected,
    }
