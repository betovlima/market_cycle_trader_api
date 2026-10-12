"""Idempotent Strategy Catalog installer for the fixed 77-asset Challenger."""
from __future__ import annotations

from typing import Any

from ..engine.tcc_u77_static_challenger import (
    BACKTEST_ENGINE_BINDING,
    REFERENCE_CAPITAL,
    REFERENCE_END_DATE,
    REFERENCE_ENGINE_ID,
    SOURCE_COMMIT,
    SOURCE_REPOSITORY,
    SOURCE_TAG,
    SOURCE_VERSION,
    U77_REQUESTED_ASSETS,
    tcc_u77_strategy_updates,
)
from ..infrastructure.persistence.mongo_repository import (
    STRATEGY_PROFILES_COLLECTION,
    utc_now,
)
from ..schemas.requests import BacktestRequest
from .strategy_lab import (
    create_strategy,
    get_research_strategy_context,
    get_strategy,
    get_strategy_control,
    select_research_strategy_only,
    update_strategy,
    update_strategy_model,
)
from .tcc_u67_operational_strategy import (
    _mct_configuration_for_u67,
    _tcc_lightgbm_values,
)


def _strategy_configuration(source: BacktestRequest) -> BacktestRequest:
    baseline = _mct_configuration_for_u67(source)
    updates = tcc_u77_strategy_updates()
    return BacktestRequest.model_validate(
        {**baseline.model_dump(mode="python"), **updates}
    )


def install_tcc_u77_fixed_challenger(
    db: Any,
    *,
    actor_email: str | None,
) -> dict[str, Any]:
    """Install once, then select the Challenger as the research Strategy."""
    actor = str(actor_email or "").strip().lower() or None
    existing = db[STRATEGY_PROFILES_COLLECTION].find_one(
        {
            "backtest_engine_binding": BACKTEST_ENGINE_BINDING,
            "reference_source_commit": SOURCE_COMMIT,
        }
    )
    if existing is not None:
        db[STRATEGY_PROFILES_COLLECTION].update_one(
            {"_id": existing["_id"]},
            {
                "$set": {
                    "operational_stage": "strategy_catalog",
                    "live_trader_eligible": True,
                    "updated_at": utc_now(),
                    "updated_by": actor,
                },
                "$unset": {"shadow_validated": ""},
            },
        )
        control = get_strategy_control(db)
        selected = select_research_strategy_only(
            db,
            str(existing["_id"]),
            expected_control_revision=int(control["revision"]),
            note="Select frozen U67+10 Challenger for Backtest",
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
        name="TCC U67+10 Static Challenger v1.4.1",
        description=(
            "77 assets requested, fixed universe, LightGBM Control, "
            "fresh Alpaca RAW/SIP and structural exclusions"
        ),
        clone_from_strategy_id=str(source_profile["id"]),
        actor_email=actor,
    )
    strategy_id = str(created["id"])
    model = update_strategy_model(
        db,
        strategy_id,
        model_family="lightgbm_utility",
        values=_tcc_lightgbm_values(),
        note="Preserve exact scientific LightGBM model parameters",
        expected_strategy_revision=int(created["revision"]),
        actor_email=actor,
    )
    config = _strategy_configuration(source_config)
    updated = update_strategy(
        db,
        strategy_id,
        configuration=config,
        name=str(created.get("name") or ""),
        description="Fixed U67+10 Challenger from the frozen v1.4.1 research",
        note="Create isolated U77 static Challenger (standard strategy)",
        expected_revision=int(model["revision"]),
        actor_email=actor,
    )

    db[STRATEGY_PROFILES_COLLECTION].update_one(
        {"_id": strategy_id, "revision": int(updated["revision"])},
        {
            "$set": {
                "backtest_engine_binding": BACKTEST_ENGINE_BINDING,
                "reference_engine_id": REFERENCE_ENGINE_ID,
                "strategy_kind": "standard",
                "research_reference_assets": list(U77_REQUESTED_ASSETS),
                "reference_source_repository": SOURCE_REPOSITORY,
                "reference_source_branch": "research/u67-mct-asset-expansion-v1",
                "reference_source_tag": SOURCE_TAG,
                "reference_source_commit": SOURCE_COMMIT,
                "reference_reproduction_version": SOURCE_VERSION,
                "reference_control_variant": "CONTROL_U77_FIXED",
                "reference_checkpoint_capital": REFERENCE_CAPITAL,
                "reference_checkpoint_date": REFERENCE_END_DATE,
                "reference_checkpoint_is_optimization_target": False,
                "operational_stage": "strategy_catalog",
                "live_trader_eligible": True,
                "source_git_commit_message": (
                    "Frozen U67+10 Challenger research " + SOURCE_COMMIT[:12]
                ),
                "updated_at": utc_now(),
                "updated_by": actor,
            }
        },
    )
    control = get_strategy_control(db)
    selected = select_research_strategy_only(
        db,
        strategy_id,
        expected_control_revision=int(control["revision"]),
        note="Select frozen U67+10 static Challenger for Backtest",
        actor_email=actor,
    )
    return {
        "created": True,
        "strategy": get_strategy(db, strategy_id),
        "control": selected,
    }
