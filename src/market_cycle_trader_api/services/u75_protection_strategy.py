"""Install/reuse the U75 daily cash Strategy in the existing catalog.

Winner selection, broker accounts, Paper state and orders remain untouched.
"""
from __future__ import annotations

from typing import Any

from ..engine.u75_market_protection import (
    ENGINE_BINDING, ENGINE_VERSION, REFERENCE_CONTROL_CAPITAL,
    SOURCE_BRANCH, SOURCE_COMMIT, SOURCE_REPOSITORY, SOURCE_VERSION,
    u75_strategy_updates,
)
from ..infrastructure.persistence.mongo_repository import (
    STRATEGY_PROFILES_COLLECTION, utc_now,
)
from ..schemas.requests import BacktestRequest
from .strategy_lab import (
    create_strategy, get_research_strategy_context, get_strategy,
    get_strategy_control, select_research_strategy_only,
    update_strategy, update_strategy_model,
)
from .tcc_u67_operational_strategy import _mct_configuration_for_u67, _tcc_lightgbm_values


def install_u75_protection_strategy(db: Any, *, actor_email: str | None) -> dict:
    actor = str(actor_email or "").strip().lower() or None
    coll = db[STRATEGY_PROFILES_COLLECTION]
    existing = coll.find_one({
        "backtest_engine_binding": ENGINE_BINDING,
        "reference_source_commit": SOURCE_COMMIT,
    })
    if existing:
        sid = str(existing["_id"])
        control = get_strategy_control(db)
        if str(control.get("research_strategy_id") or "") != sid:
            selected = select_research_strategy_only(
                db, sid, expected_control_revision=int(control["revision"]),
                note="U75 collective protection: select for Backtest",
                actor_email=actor,
            )
        else:
            selected = control
        return {"created": False, "strategy": get_strategy(db, sid), "control": selected}

    # Validate the complete target contract BEFORE mutating the catalog.
    # A failed request must not create a half-configured Strategy profile.
    source, current = get_research_strategy_context(db)
    baseline = _mct_configuration_for_u67(source)
    configuration = BacktestRequest.model_validate({
        **baseline.model_dump(mode="python"), **u75_strategy_updates(),
    })
    created = create_strategy(
        db, name="U75 Daily Market Protection v1.0.1",
        description=(
            "75 effective assets, collective market-stress CASH guard; "
            "daily reassessment, no fixed holding period; exploratory thresholds"
        ),
        clone_from_strategy_id=str(current["id"]),
        actor_email=actor,
    )
    sid = str(created["id"])
    model = update_strategy_model(
        db, sid, model_family="lightgbm_utility",
        values=_tcc_lightgbm_values(),
        note="Frozen LightGBM Control settings; market protection is separate",
        expected_strategy_revision=int(created["revision"]), actor_email=actor,
    )
    updated = update_strategy(
        db, sid, configuration=configuration,
        name=str(created.get("name") or ""),
        description=(
            "U75 whole-universe daily protection, based on frozen "
            "v1.0.1 experimental research; not a validated Winner"
        ),
        note="Install isolated U75 daily protection with full Strategy lifecycle",
        expected_revision=int(model["revision"]), actor_email=actor,
    )
    changed = coll.update_one(
        {"_id": sid, "revision": int(updated["revision"])},
        {"$set": {
            "backtest_engine_binding": ENGINE_BINDING,
            "strategy_kind": "standard",
            "reference_engine_id": ENGINE_VERSION,
            "research_reference_assets": list(configuration.assets),
            "reference_source_repository": SOURCE_REPOSITORY,
            "reference_source_branch": SOURCE_BRANCH,
            "reference_source_commit": SOURCE_COMMIT,
            "reference_reproduction_version": SOURCE_VERSION,
            "reference_checkpoint_capital": REFERENCE_CONTROL_CAPITAL,
            "reference_checkpoint_is_optimization_target": False,
            "u75_cash_signal_settings": {
                "status": "unvalidated_exploratory_thresholds",
                "cash_min_sessions": None,
                "cash_max_sessions": None,
                "evaluated_once_per_market_session": True,
                "cash_interest_assumed_zero": True,
            },
            "operational_stage": "strategy_catalog",
            "live_trader_eligible": True,
            "live_runtime_version": "10.8.133",
            "updated_at": utc_now(), "updated_by": actor,
        }},
    )
    if changed.matched_count != 1:
        raise RuntimeError("U75 strategy changed concurrently during installation.")
    control = get_strategy_control(db)
    selected = select_research_strategy_only(
        db, sid, expected_control_revision=int(control["revision"]),
        note="Install and select U75 daily protection for Backtest",
        actor_email=actor,
    )
    return {"created": True, "strategy": get_strategy(db, sid), "control": selected}
