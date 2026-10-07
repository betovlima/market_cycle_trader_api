from __future__ import annotations

import threading
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException

from ...auth.security import SessionIdentity, require_admin_session
from ...core.runtime import database
from ...infrastructure.persistence.mongo_repository import (
    JOBS_COLLECTION,
    utc_now,
)
from ...services.jobs import public_job, run_job
from ...services.strategy_lab import (
    StrategyLabConflict,
    StrategyLabError,
    StrategyLabNotFound,
    get_research_strategy_context,
)
from ...services.tcc_u67_operational_strategy import (
    BACKTEST_ENGINE_BINDING,
    REFERENCE_ENGINE_ID,
    install_tcc_u67_operational_strategy,
    recover_stale_live_market_refresh_lock,
)
from ...tcc_u67_v1210_reference.contract import (
    EXPECTED_ENDING_CAPITAL,
    REPRODUCTION_VERSION,
    SOURCE_COMMIT,
    SOURCE_REPOSITORY,
    U67_REQUESTED_ASSETS,
)
from .jobs import queue_backtest_job

router = APIRouter(
    prefix="/api/research/tcc-u67",
    tags=["tcc-u67-operational"],
)
AdminIdentity = Annotated[SessionIdentity, Depends(require_admin_session)]


@router.get("")
def get_u67_operational_engine() -> dict[str, Any]:
    return {
        "id": REFERENCE_ENGINE_ID,
        "source_repository": SOURCE_REPOSITORY,
        "source_branch": "main",
        "source_commit": SOURCE_COMMIT,
        "reproduction_version": REPRODUCTION_VERSION,
        "vendored_engine": True,
        "requested_asset_count": len(U67_REQUESTED_ASSETS),
        "reference_checkpoint_capital": EXPECTED_ENDING_CAPITAL,
        "checkpoint_is_optimization_target": False,
        "market_data_contract": (
            "fresh Alpaca RAW/SIP + Corporate Actions + "
            "causal split normalization"
        ),
        "structural_identity_policy": (
            "exclude; never bridge or reconstruct"
        ),
        "stage": "protected_live_runtime",
        "trader_eligible": True,
        "winner_promotion_preserves_operational_state": True,
        "backtest_order_submission": "never",
    }


@router.post("/strategy")
def install_u67_strategy(
    identity: AdminIdentity,
) -> dict[str, Any]:
    """Create/reuse U67 and select it for Research with protected live runtime."""
    try:
        return install_tcc_u67_operational_strategy(
            database(),
            actor_email=identity.email,
        )
    except StrategyLabConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except StrategyLabNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except StrategyLabError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/recover-stale-live-market-lock")
def recover_u67_stale_live_market_lock(
    _: AdminIdentity,
) -> dict[str, Any]:
    """Clear only an orphaned refresh lock after safety checks."""
    return recover_stale_live_market_refresh_lock(database())


@router.post("/jobs", status_code=202)
def create_u67_operational_backtest() -> dict[str, Any]:
    """Run fresh-data U67 backtest; this endpoint never submits orders."""
    db = database()
    configuration, selected_strategy = get_research_strategy_context(db)
    if (
        str(selected_strategy.get("backtest_engine_binding") or "")
        != BACKTEST_ENGINE_BINDING
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                "Install/select the protected TCC U67 Strategy before "
                "starting this job."
            ),
        )
    if len(configuration.assets) != len(U67_REQUESTED_ASSETS):
        raise HTTPException(
            status_code=409,
            detail=(
                "The selected U67 Strategy no longer has the exact "
                "67-asset requested universe."
            ),
        )

    queued = queue_backtest_job(
        start_thread=False,
        certify_strategy=False,
    )
    job_id = str(queued["id"])
    db[JOBS_COLLECTION].update_one(
        {"id": job_id},
        {
            "$set": {
                "reference_engine_id": REFERENCE_ENGINE_ID,
                "reference_source_repository": SOURCE_REPOSITORY,
                "reference_source_branch": "main",
                "reference_source_commit": SOURCE_COMMIT,
                "reference_reproduction_version": REPRODUCTION_VERSION,
                "reference_checkpoint_capital": EXPECTED_ENDING_CAPITAL,
                "reference_checkpoint_is_optimization_target": False,
                "certifies_strategy": False,
                "research_model_label": (
                    "TCC main U67 v1.21.0 Control Operational"
                ),
                "order_submission": "never",
                "updated_at": utc_now(),
            },
            "$push": {
                "logs": {
                    "$each": [
                        "TCC U67 operational backtest queued.",
                        (
                            "Fresh Alpaca RAW/SIP + Corporate Actions will "
                            "be downloaded for this job."
                        ),
                        (
                            "Structural identity breaks are excluded; "
                            "Winner, portfolio and orders are untouched."
                        ),
                    ],
                    "$slice": -400,
                }
            },
        },
    )

    thread = threading.Thread(
        target=run_job,
        args=(job_id,),
        daemon=True,
        name=f"tcc-u67-{job_id}",
    )
    thread.start()
    document = db[JOBS_COLLECTION].find_one({"id": job_id})
    return public_job(document) or {}
