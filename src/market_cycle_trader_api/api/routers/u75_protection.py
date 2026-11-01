"""One-call Strategy creation + Backtest + Annual Replay integration.

POST is administrative; Winner is NEVER promoted here. Backtest uses
the existing job pipeline, Mongo persistence and Annual Replay endpoints.
"""
from __future__ import annotations

import threading
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException

from ...auth.security import SessionIdentity, require_admin_session
from ...core.runtime import database
from ...engine.u75_market_protection import (
    ENGINE_BINDING, ENGINE_VERSION, REFERENCE_CONTROL_CAPITAL,
    SOURCE_BRANCH, SOURCE_COMMIT, SOURCE_REPOSITORY, SOURCE_VERSION,
)
from ...infrastructure.persistence.mongo_repository import JOBS_COLLECTION, utc_now
from ...services.jobs import public_job, run_job
from ...services.strategy_lab import (
    StrategyLabConflict, StrategyLabError, StrategyLabNotFound,
)
from ...services.u75_protection_strategy import install_u75_protection_strategy
from .jobs import queue_backtest_job

router = APIRouter(prefix="/api/admin/strategies/u75-protection", tags=["u75-market-protection"])
Admin = Annotated[SessionIdentity, Depends(require_admin_session)]


@router.get("")
def describe_u75_strategy(_: Admin) -> dict[str, Any]:
    return {
        "version": ENGINE_VERSION,
        "source_commit": SOURCE_COMMIT,
        "source_branch": SOURCE_BRANCH,
        "reference_control_capital_usd": REFERENCE_CONTROL_CAPITAL,
        "reference_is_optimization_target": False,
        "cash_reassessed_daily": True,
        "fixed_cash_period": False,
        "thresholds_statistically_validated": False,
        "cash_balance_yield": "zero (nominal USD)",
        "trader_automatic_promotion": False,
        "replay_uses": "persisted_standard_backtest_analytics",
    }


@router.post("/install-and-backtest", status_code=202)
def install_and_backtest(identity: Admin) -> dict[str, Any]:
    """Install or reuse, select, enqueue one standard Backtest. No broker orders."""
    db = database()
    running = db[JOBS_COLLECTION].find_one({
        "status": {"$in": ["queued", "running"]},
        "internal_job": {"$ne": True},
    })
    if running:
        if str(running.get("backtest_engine_binding") or "") == ENGINE_BINDING:
            job = public_job(running) or {"id": str(running.get("id") or "")}
            return {
                "created": False, "already_running": True,
                "job": job, "job_id": str(job["id"]),
                "results_path": f"/api/jobs/{job['id']}/results",
                "annual_replay_path": f"/api/analytics/backtests/{job['id']}",
                "annual_replay_files_path": f"/api/analytics/backtests/{job['id']}/files",
                "winner_changed": False, "order_submission": "never",
            }
        raise HTTPException(status_code=409, detail="Another Backtest is active.")
    try:
        installed = install_u75_protection_strategy(
            db, actor_email=identity.email,
        )
        queued = queue_backtest_job(start_thread=False, certify_strategy=True)
        jid = str(queued["id"])
        db[JOBS_COLLECTION].update_one({"id": jid}, {
            "$set": {
                "reference_engine_id": ENGINE_VERSION,
                "reference_source_repository": SOURCE_REPOSITORY,
                "reference_source_branch": SOURCE_BRANCH,
                "reference_source_commit": SOURCE_COMMIT,
                "reference_reproduction_version": SOURCE_VERSION,
                "reference_checkpoint_capital": REFERENCE_CONTROL_CAPITAL,
                "reference_checkpoint_is_optimization_target": False,
                "backtest_engine_binding": ENGINE_BINDING,
                "order_submission": "never",
                "u75_thresholds_validated": False,
                "updated_at": utc_now(),
            },
            "$push": {"logs": {"$each": [
                "U75 strategy installed/selected via one administrative endpoint.",
                "Downloading fresh Alpaca RAW/SIP and Corporate Actions for audit.",
                "Backtest and Annual Replay use standard persistent results.",
                "No trading, no automatic Winner promotion.",
            ], "$slice": -400}},
        })
        threading.Thread(target=run_job, args=(jid,), daemon=True, name=f"u75-{jid}").start()
        return {
            "created": bool(installed["created"]),
            "strategy": installed["strategy"],
            "job_id": jid,
            "job": public_job(db[JOBS_COLLECTION].find_one({"id": jid})) or queued,
            "results_path": f"/api/jobs/{jid}/results",
            "annual_replay_path": f"/api/analytics/backtests/{jid}",
            "annual_replay_files_path": f"/api/analytics/backtests/{jid}/files",
            "winner_changed": False,
            "order_submission": "never",
            "winner_process": "standard_candidate_and_promote_after_certified_backtest",
        }
    except StrategyLabConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except StrategyLabNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (StrategyLabError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
