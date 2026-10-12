"""Research-only control surface for frozen U67+10 static Challenger."""
from __future__ import annotations

import threading
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException

from ...auth.security import SessionIdentity, require_admin_session
from ...core.runtime import database
from ...engine.tcc_u77_static_challenger import (
    BACKTEST_ENGINE_BINDING,
    REFERENCE_CAPITAL,
    REFERENCE_END_DATE,
    REFERENCE_ENGINE_ID,
    SOURCE_COMMIT,
    SOURCE_REPOSITORY,
    SOURCE_TAG,
    SOURCE_VERSION,
    U77_REQUESTED_ASSETS,
)
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
from ...services.tcc_u77_static_challenger import (
    install_tcc_u77_fixed_challenger,
)
from .jobs import queue_backtest_job

router = APIRouter(prefix="/api/research/tcc-u77-fixed", tags=["tcc-u77-fixed-challenger"])
AdminIdentity = Annotated[SessionIdentity, Depends(require_admin_session)]


@router.get("")
def read_static_challenger_contract() -> dict[str, Any]:
    return {
        "reference_engine_id": REFERENCE_ENGINE_ID,
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "source_tag": SOURCE_TAG,
        "version": SOURCE_VERSION,
        "checkpoint_date": REFERENCE_END_DATE,
        "checkpoint_capital": REFERENCE_CAPITAL,
        "checkpoint_is_optimization_target": False,
        "configured_assets": list(U77_REQUESTED_ASSETS),
        "configured_asset_count": len(U77_REQUESTED_ASSETS),
        "stage": "backtest_only",
        "trader_eligible": False,
        "order_submission": "never",
    }


@router.post("/strategy")
def install_static_challenger(identity: AdminIdentity) -> dict[str, Any]:
    try:
        return install_tcc_u77_fixed_challenger(
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


@router.post("/jobs", status_code=202)
def create_static_challenger_backtest(
    _: AdminIdentity,
) -> dict[str, Any]:
    """Enqueue fresh Alpaca replay; no orders and no Winner changes."""
    db = database()
    configuration, selected = get_research_strategy_context(db)
    if str(selected.get("backtest_engine_binding") or "") != BACKTEST_ENGINE_BINDING:
        raise HTTPException(
            status_code=409,
            detail="Install/select the frozen U77 Strategy before starting the job.",
        )
    if tuple(configuration.assets) != U77_REQUESTED_ASSETS:
        raise HTTPException(
            status_code=409,
            detail="The selected Strategy no longer matches the frozen U77 universe.",
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
                "reference_source_commit": SOURCE_COMMIT,
                "reference_source_tag": SOURCE_TAG,
                "reference_reproduction_version": SOURCE_VERSION,
                "reference_checkpoint_capital": REFERENCE_CAPITAL,
                "reference_checkpoint_is_optimization_target": False,
                "certifies_strategy": False,
                "order_submission": "never",
                "updated_at": utc_now(),
            },
            "$push": {
                "logs": {
                    "$each": [
                        "Frozen U67+10 static Challenger queued, no broker orders.",
                        "Fresh Alpaca RAW/SIP + corporate actions; structural identity exclusions.",
                        "Existing Winner and operational Paper State preserved.",
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
        name=f"tcc-u77-{job_id}",
    )
    thread.start()
    document = db[JOBS_COLLECTION].find_one({"id": job_id})
    return public_job(document) or {}
