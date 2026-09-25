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
from ...services.jobs import (
    TCC_V106_REFERENCE_ENGINE_MODULE,
    public_job,
    run_job,
)
from ...services.strategy_lab import (
    StrategyLabConflict,
    StrategyLabError,
    StrategyLabNotFound,
    get_research_strategy_context,
    get_research_strategy_model_snapshot,
)
from ...services.tcc_v106_reference_strategy import (
    install_tcc_v106_research_strategy,
)
from ...tcc_v106_reference.config import ASSETS as TCC_V106_ASSETS
from .jobs import queue_backtest_job

router = APIRouter(prefix="/api/research/tcc-v106", tags=["tcc-v106-reference"])
AdminIdentity = Annotated[SessionIdentity, Depends(require_admin_session)]

SOURCE_REPOSITORY = "betovlima/tcc_mba_usp_data_science_analytics"
SOURCE_TAG = "v1.0.6"
SOURCE_COMMIT = "c0d71772092f0c26c9f28f0211b9933e9c396b95"
REFERENCE_CONTROL_CAPITAL = 10_094_316.30
REFERENCE_SOFT_CAPITAL = 9_851_632.93


@router.get("")
def get_reference_engine() -> dict[str, Any]:
    return {
        "id": "tcc-v1.0.6-verbatim",
        "source_repository": SOURCE_REPOSITORY,
        "source_tag": SOURCE_TAG,
        "source_commit": SOURCE_COMMIT,
        "vendored_engine": True,
        "vendored_engine_modified": False,
        "mct_data_transport_only": True,
        "tcc_frozen_snapshot_used": False,
        "variants": ["CONTROL", "SOFT_HORIZON_CONSENSUS"],
        "reference_control_ending_capital": REFERENCE_CONTROL_CAPITAL,
        "reference_soft_ending_capital": REFERENCE_SOFT_CAPITAL,
        "asset_count": len(TCC_V106_ASSETS),
    }


@router.post("/strategy")
def install_reference_strategy(
    identity: AdminIdentity,
) -> dict[str, Any]:
    """Create/reuse the TCC reference Strategy and select it for Research."""
    try:
        return install_tcc_v106_research_strategy(
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
def create_reference_job() -> dict[str, Any]:
    db = database()
    selected_configuration, selected_strategy = get_research_strategy_context(db)
    configured_assets = tuple(
        str(symbol).strip().upper()
        for symbol in selected_configuration.assets
    )
    expected_assets = tuple(TCC_V106_ASSETS)

    if set(configured_assets) != set(expected_assets):
        missing = sorted(set(expected_assets).difference(configured_assets))
        extra = sorted(set(configured_assets).difference(expected_assets))
        raise HTTPException(
            status_code=409,
            detail=(
                "TCC v1.0.6 reference API requires the exact 56-asset "
                f"universe. missing={missing or 'none'} "
                f"extra={extra or 'none'}"
            ),
        )

    selected_model = get_research_strategy_model_snapshot(db)
    if str(selected_model.get("family") or "") != "lightgbm_utility":
        raise HTTPException(
            status_code=409,
            detail=(
                "Select a LightGBM Utility Strategy before starting the "
                "TCC v1.0.6 reference API."
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
                "engine_module_override": TCC_V106_REFERENCE_ENGINE_MODULE,
                "reference_engine_id": "tcc-v1.0.6-verbatim",
                "reference_source_repository": SOURCE_REPOSITORY,
                "reference_source_tag": SOURCE_TAG,
                "reference_source_commit": SOURCE_COMMIT,
                "reference_selected_strategy_id": selected_strategy.get("id"),
                "reference_selected_strategy_revision": selected_strategy.get(
                    "revision"
                ),
                "certifies_strategy": False,
                "total_runs": 2,
                "research_model_label": (
                    "TCC v1.0.6 Control + Soft Horizon Consensus"
                ),
                "updated_at": utc_now(),
            },
            "$push": {
                "logs": {
                    "$each": [
                        "TCC v1.0.6 reference job queued.",
                        (
                            "The vendored TCC engine is used verbatim; "
                            "MCT provides only current research market data."
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
        name=f"tcc-v106-{job_id}",
    )
    thread.start()
    document = db[JOBS_COLLECTION].find_one({"id": job_id})
    return public_job(document) or {}
