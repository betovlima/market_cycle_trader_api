from __future__ import annotations

import os
import threading
import uuid
from datetime import timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

from ..core.runtime import database
from ..infrastructure.persistence.mongo_repository import (
    PAPER_MARKET_AUTOMATION_COLLECTION,
    PAPER_MARKET_RUNS_COLLECTION,
    PAPER_TRADE_PLANS_COLLECTION,
    bson_value,
    update_paper_trade_plan,
    utc_now,
)
from ..infrastructure.trading.alpaca_paper import (
    clock_snapshot,
    create_paper_trading_client,
)
from ..schemas.paper_trading import PaperTradingSettings
from .paper_trading import (
    execute_prepared_paper_plan,
    paper_market_readiness,
    prepare_next_paper_plan,
)

EASTERN = ZoneInfo("America/New_York")
ACTIVE_KEY = "alpaca-paper-next-session"
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
AUTOMATION_ID = "default"
AUTOMATION_MODE = "continuous_regular_sessions"
def _positive_float_env(name: str, default: float, *, minimum: float) -> float:
    raw = str(os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be numeric.") from exc
    if value < minimum:
        raise RuntimeError(f"{name} must be at least {minimum}.")
    return value


def scheduler_poll_seconds() -> float:
    return _positive_float_env("PAPER_MARKET_POLL_SECONDS", 10.0, minimum=1.0)


def preparation_retry_seconds() -> float:
    return _positive_float_env(
        "PAPER_MARKET_PREPARE_RETRY_SECONDS",
        60.0,
        minimum=10.0,
    )


def _utc_stamp(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _public_run(document: dict[str, Any] | None) -> dict[str, Any] | None:
    if document is None:
        return None
    hidden = {"_id", "active_key", "worker_id", "lease_expires_at"}
    return {
        key: bson_value(value)
        for key, value in document.items()
        if key not in hidden
    }


def _log_update(message: str) -> dict[str, Any]:
    return {
        "$push": {
            "logs": {
                "$each": [f"{utc_now().isoformat()} — {message}"],
                "$slice": -100,
            }
        },
        "$set": {"updated_at": utc_now(), "last_message": message},
    }


def _append_log(db: Any, run_id: str, message: str) -> None:
    db[PAPER_MARKET_RUNS_COLLECTION].update_one(
        {"run_id": run_id},
        _log_update(message),
    )


def _finish_run(
    db: Any,
    run_id: str,
    *,
    status: str,
    message: str,
    extra: dict[str, Any] | None = None,
) -> None:
    now = utc_now()
    changes = {
        "status": status,
        "phase": status,
        "updated_at": now,
        "finished_at": now,
        "last_message": message,
        **(extra or {}),
    }
    db[PAPER_MARKET_RUNS_COLLECTION].update_one(
        {"run_id": run_id},
        {
            "$set": bson_value(changes),
            "$unset": {
                "active_key": "",
                "worker_id": "",
                "lease_expires_at": "",
            },
            "$push": {
                "logs": {
                    "$each": [f"{now.isoformat()} — {message}"],
                    "$slice": -100,
                }
            },
        },
    )


def _mark_review_required(
    db: Any,
    run_id: str,
    *,
    message: str,
) -> None:
    now = utc_now()
    db[PAPER_MARKET_RUNS_COLLECTION].update_one(
        {"run_id": run_id},
        {
            "$set": {
                "status": "review_required",
                "phase": "review_required",
                "updated_at": now,
                "last_message": message,
            },
            "$unset": {"worker_id": "", "lease_expires_at": ""},
            "$push": {
                "logs": {
                    "$each": [f"{now.isoformat()} — {message}"],
                    "$slice": -100,
                }
            },
        },
    )


def _automation_document(db: Any) -> dict[str, Any] | None:
    return db[PAPER_MARKET_AUTOMATION_COLLECTION].find_one({"_id": AUTOMATION_ID})


def _update_automation(db: Any, changes: dict[str, Any], *, unset: tuple[str, ...] = ()) -> None:
    now = utc_now()
    update: dict[str, Any] = {
        "$set": bson_value({
            "mode": AUTOMATION_MODE,
            "updated_at": now,
            **changes,
        }),
        "$setOnInsert": {"created_at": now},
    }
    if unset:
        update["$unset"] = {field: "" for field in unset}
    db[PAPER_MARKET_AUTOMATION_COLLECTION].update_one(
        {"_id": AUTOMATION_ID},
        update,
        upsert=True,
    )


def _public_robot_run(document: dict[str, Any] | None) -> dict[str, Any] | None:
    if document is None:
        return None
    allowed = (
        "run_id",
        "status",
        "phase",
        "created_at",
        "updated_at",
        "expected_market_open",
        "execution_session",
        "preparation_attempts",
        "finished_at",
    )
    return {key: bson_value(document.get(key)) for key in allowed if document.get(key) is not None}


def latest_paper_market_run(db: Any) -> dict[str, Any] | None:
    document = db[PAPER_MARKET_RUNS_COLLECTION].find_one(
        {},
        sort=[("created_at", -1)],
    )
    return _public_run(document)


def _create_next_session_run(db: Any) -> dict[str, Any]:
    readiness = paper_market_readiness(db)
    clock = readiness["clock"]
    expected_open = _utc_stamp(clock["next_open"])
    execution_session = expected_open.tz_convert(EASTERN).date().isoformat()
    run_id = f"paper-market-{execution_session}-{uuid.uuid4().hex[:8]}"
    now = utc_now()
    phase = "waiting_for_market_close" if bool(clock["is_open"]) else "ready_to_prepare"
    first_message = (
        "Paper market run armed for the next Alpaca regular session. "
        f"Expected open: {expected_open.isoformat()}."
    )
    document = {
        "run_id": run_id,
        "active_key": ACTIVE_KEY,
        "status": "armed",
        "phase": phase,
        "created_at": now,
        "updated_at": now,
        "requested_at": now,
        "expected_market_open": expected_open.to_pydatetime(),
        "execution_session": execution_session,
        "requested_market_was_open": bool(clock["is_open"]),
        "plan_id": None,
        "action": None,
        "target_asset": None,
        "preparation_attempts": 0,
        "cancel_requested": False,
        "strategy_cash": readiness["strategy_cash"],
        "managed_symbol": readiness["managed_symbol"],
        "last_message": first_message,
        "logs": [f"{now.isoformat()} — {first_message}"],
    }
    try:
        db[PAPER_MARKET_RUNS_COLLECTION].insert_one(document)
    except DuplicateKeyError as exc:
        active = db[PAPER_MARKET_RUNS_COLLECTION].find_one(
            {"active_key": ACTIVE_KEY}
        )
        active_id = str((active or {}).get("run_id") or "unknown")
        raise RuntimeError(
            f"Another paper-market run is already active: {active_id}."
        ) from exc
    return _public_run(document) or {}


def arm_next_session(db: Any) -> dict[str, Any]:
    """Enable the continuous Paper robot and ensure the next session is armed."""

    active = db[PAPER_MARKET_RUNS_COLLECTION].find_one({"active_key": ACTIVE_KEY})
    now = utc_now()
    _update_automation(
        db,
        {
            "enabled": True,
            "status": "active",
            "phase": "ensuring_next_regular_session",
            "activated_at": now,
            "stopped_at": None,
            "last_error": None,
        },
    )
    if active is None:
        run = _create_next_session_run(db)
        active_run_id = run.get("run_id")
    else:
        run = _public_run(active) or {}
        active_run_id = run.get("run_id")
    _update_automation(
        db,
        {
            "enabled": True,
            "status": "active",
            "phase": str(run.get("phase") or "armed"),
            "active_run_id": active_run_id,
            "next_market_open": run.get("expected_market_open"),
            "next_execution_session": run.get("execution_session"),
        },
        unset=("last_error",),
    )
    return {
        **run,
        "automation_enabled": True,
        "automation_mode": AUTOMATION_MODE,
    }


def stop_continuous_robot(db: Any, *, cancel_pending_run: bool = True) -> dict[str, Any]:
    """Disable future sessions and safely cancel a non-executing active run."""

    now = utc_now()
    _update_automation(
        db,
        {
            "enabled": False,
            "status": "stopped",
            "phase": "stopped_by_administrator",
            "stopped_at": now,
        },
    )
    active = db[PAPER_MARKET_RUNS_COLLECTION].find_one({"active_key": ACTIVE_KEY})
    if active is not None and cancel_pending_run:
        status = str(active.get("status") or "")
        if status not in {"executing", "review_required"}:
            cancel_paper_market_run(db, str(active["run_id"]))
            active = db[PAPER_MARKET_RUNS_COLLECTION].find_one({"run_id": active["run_id"]})
    return paper_market_robot_status(db)


def cancel_paper_market_run(db: Any, run_id: str) -> dict[str, Any]:
    run = db[PAPER_MARKET_RUNS_COLLECTION].find_one({"run_id": run_id})
    if run is None:
        raise RuntimeError(f"Paper-market run was not found: {run_id}.")
    status = str(run.get("status") or "")
    if status in TERMINAL_STATUSES:
        return _public_run(run) or {}
    if status == "executing":
        raise RuntimeError(
            "The paper-market run is already executing and cannot be cancelled safely."
        )
    if status == "review_required":
        raise RuntimeError(
            "The run requires manual order/state review before it can be released."
        )
    if status == "preparing":
        db[PAPER_MARKET_RUNS_COLLECTION].update_one(
            {"run_id": run_id, "status": "preparing"},
            {
                "$set": {
                    "cancel_requested": True,
                    "updated_at": utc_now(),
                    "last_message": "Cancellation requested; waiting for model preparation to finish.",
                },
                "$push": {
                    "logs": {
                        "$each": [
                            f"{utc_now().isoformat()} — Cancellation requested; waiting for model preparation to finish."
                        ],
                        "$slice": -100,
                    }
                },
            },
        )
        return _public_run(
            db[PAPER_MARKET_RUNS_COLLECTION].find_one({"run_id": run_id})
        ) or {}

    plan_id = str(run.get("plan_id") or "").strip()
    if plan_id:
        update_paper_trade_plan(
            db,
            plan_id,
            {
                "status": "cancelled",
                "cancelled_at": utc_now(),
                "cancel_reason": "Paper-market API run cancelled before execution.",
            },
        )
    _finish_run(
        db,
        run_id,
        status="cancelled",
        message="Paper-market run cancelled before order execution.",
    )
    return _public_run(
        db[PAPER_MARKET_RUNS_COLLECTION].find_one({"run_id": run_id})
    ) or {}


def _recover_stale_runs(db: Any) -> None:
    now = utc_now()
    stale = db[PAPER_MARKET_RUNS_COLLECTION].find(
        {
            "active_key": ACTIVE_KEY,
            "status": {"$in": ["preparing", "executing"]},
        }
    )
    for run in stale:
        run_id = str(run["run_id"])
        status = str(run.get("status") or "")
        plan_id = str(run.get("plan_id") or "").strip()
        plan = (
            db[PAPER_TRADE_PLANS_COLLECTION].find_one({"plan_id": plan_id})
            if plan_id
            else None
        )
        plan_status = str((plan or {}).get("status") or "")
        if status == "preparing":
            db[PAPER_MARKET_RUNS_COLLECTION].update_one(
                {"run_id": run_id, "status": "preparing"},
                {
                    "$set": {
                        "status": "armed",
                        "phase": "recovered_after_interrupted_preparation",
                        "updated_at": now,
                        "next_retry_at": now,
                    },
                    "$unset": {"worker_id": "", "lease_expires_at": ""},
                },
            )
            _append_log(db, run_id, "Recovered an interrupted plan-preparation lease.")
        elif plan_status == "executed":
            _finish_run(
                db,
                run_id,
                status="completed",
                message="Recovered completed Alpaca paper execution after API restart.",
                extra={"execution_result": bson_value(plan.get("final_state"))},
            )
        elif plan_status == "prepared":
            db[PAPER_MARKET_RUNS_COLLECTION].update_one(
                {"run_id": run_id, "status": "executing"},
                {
                    "$set": {
                        "status": "prepared",
                        "phase": "recovered_before_order_submission",
                        "updated_at": now,
                    },
                    "$unset": {"worker_id": "", "lease_expires_at": ""},
                },
            )
            _append_log(db, run_id, "Recovered before the Alpaca plan was claimed.")
        else:
            _mark_review_required(
                db,
                run_id,
                message=(
                    "Execution was interrupted after the plan may have been claimed. "
                    "Review Alpaca open orders, positions and MongoDB paper state before continuing."
                ),
            )


def _claim_for_preparation(db: Any, run_id: str, worker_id: str) -> dict[str, Any] | None:
    now = utc_now()
    return db[PAPER_MARKET_RUNS_COLLECTION].find_one_and_update(
        {
            "run_id": run_id,
            "status": "armed",
            "cancel_requested": {"$ne": True},
        },
        {
            "$set": {
                "status": "preparing",
                "phase": "training_xgboost_and_preparing_plan",
                "worker_id": worker_id,
                "lease_expires_at": now + timedelta(hours=6),
                "updated_at": now,
            },
            "$inc": {"preparation_attempts": 1},
        },
        return_document=ReturnDocument.AFTER,
    )


def _prepare_run(db: Any, run: dict[str, Any], worker_id: str) -> None:
    run_id = str(run["run_id"])
    expected_open = _utc_stamp(run["expected_market_open"])
    next_retry = run.get("next_retry_at")
    if next_retry is not None and _utc_stamp(next_retry) > pd.Timestamp.now(tz="UTC"):
        return

    client = create_paper_trading_client(db)
    clock = clock_snapshot(client)
    now = _utc_stamp(clock["timestamp"])
    current_session = now.tz_convert(EASTERN).date().isoformat()

    if now >= expected_open:
        _finish_run(
            db,
            run_id,
            status="failed",
            message="The expected market open arrived before a valid XGBoost plan was prepared.",
        )
        return

    if bool(clock["is_open"]):
        db[PAPER_MARKET_RUNS_COLLECTION].update_one(
            {"run_id": run_id, "status": "armed"},
            {
                "$set": {
                    "phase": "waiting_for_market_close",
                    "updated_at": utc_now(),
                    "last_message": (
                        "Current regular session is open; the model will be trained after the daily candle completes."
                    ),
                }
            },
        )
        return

    if current_session > str(run["execution_session"]):
        _finish_run(
            db,
            run_id,
            status="failed",
            message="The requested Alpaca execution session is stale.",
        )
        return

    claimed = _claim_for_preparation(db, run_id, worker_id)
    if claimed is None:
        return
    _append_log(db, run_id, "Preparing the locked XGBoost decision for the next regular session.")
    try:
        existing = db[PAPER_TRADE_PLANS_COLLECTION].find_one(
            {"execution_session": str(run["execution_session"])},
            sort=[("created_at", -1)],
        )
        if existing is not None and str(existing.get("status") or "") in {
            "prepared",
            "executing",
            "executed",
        }:
            plan = existing
        else:
            plan = prepare_next_paper_plan(db, replace=existing is not None)

        if str(plan.get("execution_session")) != str(run["execution_session"]):
            raise RuntimeError(
                "Prepared plan session does not match the API-armed next session: "
                f"run={run['execution_session']}, plan={plan.get('execution_session')}."
            )
        plan_open = _utc_stamp(plan.get("expected_market_open"))
        if plan_open != expected_open:
            raise RuntimeError(
                "Prepared plan market-open timestamp does not match the API-armed next open: "
                f"run={expected_open.isoformat()}, plan={plan_open.isoformat()}."
            )
        refreshed = db[PAPER_MARKET_RUNS_COLLECTION].find_one({"run_id": run_id}) or {}
        if bool(refreshed.get("cancel_requested")):
            update_paper_trade_plan(
                db,
                str(plan["plan_id"]),
                {
                    "status": "cancelled",
                    "cancelled_at": utc_now(),
                    "cancel_reason": "Cancellation requested while the XGBoost plan was being prepared.",
                },
            )
            _finish_run(
                db,
                run_id,
                status="cancelled",
                message="Paper-market run cancelled after plan preparation and before execution.",
            )
            return

        now_utc = utc_now()
        db[PAPER_MARKET_RUNS_COLLECTION].update_one(
            {"run_id": run_id, "status": "preparing", "worker_id": worker_id},
            {
                "$set": {
                    "status": "prepared",
                    "phase": "waiting_for_next_market_open",
                    "plan_id": str(plan["plan_id"]),
                    "action": str(plan["action"]),
                    "target_asset": str(plan["target_asset"]),
                    "decision_date": str(plan["decision_date"]),
                    "updated_at": now_utc,
                    "last_message": (
                        f"XGBoost plan prepared: action={plan['action']}, target={plan['target_asset']}."
                    ),
                },
                "$unset": {
                    "worker_id": "",
                    "lease_expires_at": "",
                    "next_retry_at": "",
                    "last_error": "",
                },
                "$push": {
                    "logs": {
                        "$each": [
                            f"{now_utc.isoformat()} — XGBoost plan prepared: action={plan['action']}, target={plan['target_asset']}."
                        ],
                        "$slice": -100,
                    }
                },
            },
        )
    except Exception as exc:
        now_utc = utc_now()
        if _utc_stamp(now_utc) < expected_open:
            retry_at = now_utc + timedelta(seconds=preparation_retry_seconds())
            db[PAPER_MARKET_RUNS_COLLECTION].update_one(
                {"run_id": run_id, "status": "preparing", "worker_id": worker_id},
                {
                    "$set": {
                        "status": "armed",
                        "phase": "waiting_to_retry_plan_preparation",
                        "updated_at": now_utc,
                        "next_retry_at": retry_at,
                        "last_error": str(exc),
                        "last_message": f"Plan preparation failed and will be retried: {exc}",
                    },
                    "$unset": {"worker_id": "", "lease_expires_at": ""},
                    "$push": {
                        "logs": {
                            "$each": [
                                f"{now_utc.isoformat()} — Plan preparation failed and will be retried: {exc}"
                            ],
                            "$slice": -100,
                        }
                    },
                },
            )
        else:
            _finish_run(
                db,
                run_id,
                status="failed",
                message=f"Could not prepare the plan before market open: {exc}",
            )


def _claim_for_execution(db: Any, run_id: str, worker_id: str) -> dict[str, Any] | None:
    now = utc_now()
    return db[PAPER_MARKET_RUNS_COLLECTION].find_one_and_update(
        {"run_id": run_id, "status": "prepared", "cancel_requested": {"$ne": True}},
        {
            "$set": {
                "status": "executing",
                "phase": "submitting_alpaca_paper_orders",
                "worker_id": worker_id,
                "lease_expires_at": now + timedelta(minutes=30),
                "execution_started_at": now,
                "updated_at": now,
            }
        },
        return_document=ReturnDocument.AFTER,
    )


def _execute_run(db: Any, run: dict[str, Any], worker_id: str) -> None:
    run_id = str(run["run_id"])
    plan_id = str(run.get("plan_id") or "").strip()
    if not plan_id:
        _finish_run(
            db,
            run_id,
            status="failed",
            message="Prepared paper-market run has no plan_id.",
        )
        return

    readiness = paper_market_readiness(db)
    settings = PaperTradingSettings.model_validate(readiness["settings"])
    clock = readiness["clock"]
    now = _utc_stamp(clock["timestamp"])
    expected_open = _utc_stamp(run["expected_market_open"])
    earliest = expected_open + pd.Timedelta(
        settings.market_open_delay_seconds,
        unit="s",
    )
    latest = expected_open + pd.Timedelta(
        settings.market_execution_window_seconds,
        unit="s",
    )
    current_session = now.tz_convert(EASTERN).date().isoformat()

    if now > latest:
        _finish_run(
            db,
            run_id,
            status="failed",
            message=(
                "The safe next-open execution window expired without submitting orders. "
                f"Latest allowed time: {latest.isoformat()}."
            ),
        )
        return
    if current_session > str(run["execution_session"]):
        _finish_run(
            db,
            run_id,
            status="failed",
            message="The prepared Alpaca plan is stale for the current session.",
        )
        return
    if not bool(clock["is_open"]) or now < earliest:
        return
    if current_session != str(run["execution_session"]):
        return

    claimed = _claim_for_execution(db, run_id, worker_id)
    if claimed is None:
        return
    _append_log(
        db,
        run_id,
        "The regular market is open and the configured delay elapsed; executing the paper plan.",
    )
    try:
        result = execute_prepared_paper_plan(db, plan_id=plan_id)
        _finish_run(
            db,
            run_id,
            status="completed",
            message=(
                f"Alpaca paper plan executed: action={result.get('action')}, "
                f"target={result.get('target_asset')}."
            ),
            extra={"execution_result": bson_value(result)},
        )
    except Exception as exc:
        plan = db[PAPER_TRADE_PLANS_COLLECTION].find_one({"plan_id": plan_id}) or {}
        plan_status = str(plan.get("status") or "")
        if plan_status == "executing":
            _mark_review_required(
                db,
                run_id,
                message=(
                    "Execution failed after the Alpaca plan was claimed. "
                    f"Manual review is required: {exc}"
                ),
            )
        else:
            _finish_run(
                db,
                run_id,
                status="failed",
                message=f"Alpaca paper execution failed: {exc}",
            )


def _ensure_continuous_run(db: Any) -> dict[str, Any] | None:
    controller = _automation_document(db)
    active = db[PAPER_MARKET_RUNS_COLLECTION].find_one({"active_key": ACTIVE_KEY})

    # Upgrade an already armed one-shot run to the continuous controller on first
    # startup after this release, preserving the currently prepared plan.
    if controller is None and active is not None:
        _update_automation(
            db,
            {
                "enabled": True,
                "status": "active",
                "phase": str(active.get("phase") or active.get("status") or "armed"),
                "activated_at": active.get("requested_at") or utc_now(),
                "active_run_id": active.get("run_id"),
                "next_market_open": active.get("expected_market_open"),
                "next_execution_session": active.get("execution_session"),
                "adopted_existing_run": True,
            },
        )
        controller = _automation_document(db)

    if not bool((controller or {}).get("enabled")):
        return active

    if active is not None:
        _update_automation(
            db,
            {
                "status": "active",
                "phase": str(active.get("phase") or active.get("status") or "active"),
                "active_run_id": active.get("run_id"),
                "next_market_open": active.get("expected_market_open"),
                "next_execution_session": active.get("execution_session"),
            },
        )
        return active

    latest = db[PAPER_MARKET_RUNS_COLLECTION].find_one({}, sort=[("created_at", -1)])
    if latest is not None and str(latest.get("status") or "") == "review_required":
        _update_automation(
            db,
            {
                "enabled": False,
                "status": "blocked",
                "phase": "manual_review_required",
                "last_error": "The previous execution requires manual review.",
                "last_terminal_run_id": latest.get("run_id"),
                "last_terminal_status": latest.get("status"),
            },
        )
        return None

    retry_at = (controller or {}).get("next_retry_at")
    if retry_at is not None and _utc_stamp(retry_at) > pd.Timestamp.now(tz="UTC"):
        return None

    try:
        created = _create_next_session_run(db)
        active = db[PAPER_MARKET_RUNS_COLLECTION].find_one({"run_id": created.get("run_id")})
        _update_automation(
            db,
            {
                "enabled": True,
                "status": "active",
                "phase": str(created.get("phase") or "armed"),
                "active_run_id": created.get("run_id"),
                "next_market_open": created.get("expected_market_open"),
                "next_execution_session": created.get("execution_session"),
                "last_terminal_run_id": (latest or {}).get("run_id"),
                "last_terminal_status": (latest or {}).get("status"),
            },
            unset=("last_error", "next_retry_at"),
        )
        return active
    except Exception as exc:
        retry = utc_now() + timedelta(seconds=preparation_retry_seconds())
        _update_automation(
            db,
            {
                "enabled": True,
                "status": "degraded",
                "phase": "waiting_to_retry_next_session_scheduling",
                "last_error": str(exc),
                "next_retry_at": retry,
            },
        )
        return None


def _advance_one(db: Any, worker_id: str) -> None:
    run = _ensure_continuous_run(db)
    if run is None:
        return
    status = str(run.get("status") or "")
    if status == "armed":
        _prepare_run(db, run, worker_id)
    elif status == "prepared":
        _execute_run(db, run, worker_id)


class PaperMarketScheduler:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._worker_id = f"paper-market-worker-{uuid.uuid4().hex[:10]}"
        self._lock = threading.Lock()
        self._started_at = None
        self._last_tick_at = None
        self._last_successful_tick_at = None
        self._last_error = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._started_at = utc_now()
        try:
            _recover_stale_runs(database())
        except Exception as exc:
            self._last_error = str(exc)
        self._thread = threading.Thread(
            target=self._run,
            name="alpaca-paper-market-scheduler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._thread = None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "scheduler_alive": bool(self._thread is not None and self._thread.is_alive()),
                "scheduler_worker_id": self._worker_id,
                "scheduler_started_at": bson_value(self._started_at),
                "last_scheduler_tick_at": bson_value(self._last_tick_at),
                "last_successful_scheduler_tick_at": bson_value(self._last_successful_tick_at),
                "scheduler_last_error": self._last_error,
                "poll_seconds": scheduler_poll_seconds(),
            }

    def _run(self) -> None:
        while not self._stop.is_set():
            tick = utc_now()
            with self._lock:
                self._last_tick_at = tick
            try:
                db = database()
                _recover_stale_runs(db)
                _advance_one(db, self._worker_id)
                with self._lock:
                    self._last_successful_tick_at = utc_now()
                    self._last_error = None
            except Exception as exc:
                with self._lock:
                    self._last_error = str(exc)
            self._stop.wait(scheduler_poll_seconds())


_SCHEDULER = PaperMarketScheduler()


def paper_market_robot_status(db: Any, *, public: bool = False) -> dict[str, Any]:
    controller = _automation_document(db) or {}
    active = db[PAPER_MARKET_RUNS_COLLECTION].find_one({"active_key": ACTIVE_KEY})
    latest = active or db[PAPER_MARKET_RUNS_COLLECTION].find_one({}, sort=[("created_at", -1)])
    runtime = _SCHEDULER.snapshot()
    enabled = bool(controller.get("enabled"))
    status = str(controller.get("status") or ("active" if enabled else "stopped"))
    output: dict[str, Any] = {
        "enabled": enabled,
        "mode": str(controller.get("mode") or AUTOMATION_MODE),
        "status": status,
        "phase": str(controller.get("phase") or (latest or {}).get("phase") or status),
        "scheduler_alive": bool(runtime.get("scheduler_alive")),
        "scheduler_started_at": runtime.get("scheduler_started_at"),
        "last_scheduler_tick_at": runtime.get("last_scheduler_tick_at"),
        "last_successful_scheduler_tick_at": runtime.get("last_successful_scheduler_tick_at"),
        "next_market_open": bson_value((active or {}).get("expected_market_open") or controller.get("next_market_open")),
        "next_execution_session": (active or {}).get("execution_session") or controller.get("next_execution_session"),
        "active_run": _public_robot_run(active),
        "latest_run": _public_robot_run(latest),
        "updated_at": bson_value(controller.get("updated_at")),
    }
    if not public:
        output.update({
            "scheduler_worker_id": runtime.get("scheduler_worker_id"),
            "scheduler_last_error": runtime.get("scheduler_last_error"),
            "poll_seconds": runtime.get("poll_seconds"),
            "last_error": controller.get("last_error"),
            "next_retry_at": bson_value(controller.get("next_retry_at")),
        })
    return output


def start_paper_market_scheduler() -> None:
    _SCHEDULER.start()


def stop_paper_market_scheduler() -> None:
    _SCHEDULER.stop()
