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
    ADMIN_OPERATION_LOGS_COLLECTION,
    PAPER_MARKET_AUTOMATION_COLLECTION,
    PAPER_MARKET_RUNS_COLLECTION,
    PAPER_TRADE_PLANS_COLLECTION,
    STRATEGY_CONTROL_COLLECTION,
    bson_value,
    get_paper_trading_settings,
    update_paper_trade_plan,
    utc_now,
)
from ..infrastructure.trading.alpaca_paper import (
    clock_snapshot,
    create_paper_trading_client,
)
from ..schemas.paper_trading import PaperTradingSettings
from .system_settings import get_system_settings
from .strategy_lab import get_trader_winner_summary
from .paper_trading import (
    execute_prepared_paper_plan,
    paper_market_readiness,
    prepare_next_paper_plan,
    refresh_trader_live_market_data,
)

EASTERN = ZoneInfo("America/New_York")
ACTIVE_KEY = "alpaca-paper-next-session"
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
AUTOMATION_ID = "default"
AUTOMATION_MODE = "continuous_regular_sessions"
PREMARKET_ANALYSIS_POLICY = "mandatory_premarket_refresh_v1"
TRADER_CONTROL_MODES = frozenset({"active", "paused", "exit_only", "stopped"})
EXIT_ONLY_ALLOWED_ACTIONS = frozenset({"sell_to_cash", "stay_in_cash", "hold"})


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


def _paper_settings(db: Any) -> PaperTradingSettings:
    return PaperTradingSettings.model_validate(get_paper_trading_settings(db))


def _premarket_analysis_at(
    expected_open: Any,
    settings: PaperTradingSettings,
) -> pd.Timestamp:
    return _utc_stamp(expected_open) - pd.Timedelta(
        int(settings.premarket_analysis_minutes),
        unit="m",
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




def _control_mode(controller: dict[str, Any] | None) -> str:
    document = controller or {}
    explicit = str(document.get("control_mode") or "").strip().lower()
    if explicit in TRADER_CONTROL_MODES:
        return explicit
    return "active" if bool(document.get("enabled")) else "stopped"


def _record_admin_operation(
    db: Any,
    *,
    action: str,
    previous_mode: str,
    new_mode: str,
    reason: str | None,
    actor_email: str | None,
    success: bool = True,
) -> None:
    db[ADMIN_OPERATION_LOGS_COLLECTION].insert_one(
        bson_value({
            "action": action,
            "previous_mode": previous_mode,
            "new_mode": new_mode,
            "reason": (reason or "").strip() or None,
            "actor_email": (actor_email or "").strip().lower() or None,
            "created_at": utc_now(),
            "success": bool(success),
        })
    )


def set_trader_control_mode(
    db: Any,
    *,
    mode: str,
    reason: str | None = None,
    actor_email: str | None = None,
    cancel_pending_run: bool = False,
) -> dict[str, Any]:
    requested = str(mode or "").strip().lower()
    if requested not in TRADER_CONTROL_MODES:
        raise RuntimeError(f"Unsupported Trader control mode: {requested}.")

    controller = _automation_document(db) or {}
    previous = _control_mode(controller)

    if requested == "active":
        arm_next_session(db)
        _update_automation(db, {"control_mode": "active", "status": "active", "phase": "active"})
    elif requested == "stopped":
        stop_continuous_robot(db, cancel_pending_run=cancel_pending_run)
        _update_automation(db, {"control_mode": "stopped", "status": "stopped", "phase": "stopped_by_administrator"})
    else:
        now = utc_now()
        _update_automation(
            db,
            {
                "enabled": requested == "exit_only",
                "control_mode": requested,
                "status": requested,
                "phase": "new_entries_paused" if requested == "paused" else "exit_only",
                "mode_changed_at": now,
            },
        )
        if requested == "paused" and cancel_pending_run:
            active = db[PAPER_MARKET_RUNS_COLLECTION].find_one({"active_key": ACTIVE_KEY})
            if active is not None and str(active.get("status") or "") not in {"executing", "review_required"}:
                cancel_paper_market_run(db, str(active["run_id"]))

    _record_admin_operation(
        db,
        action="trader_control_mode_changed",
        previous_mode=previous,
        new_mode=requested,
        reason=reason,
        actor_email=actor_email,
    )
    return paper_market_robot_status(db)


def list_admin_operation_logs(db: Any, *, limit: int = 50) -> list[dict[str, Any]]:
    cursor = db[ADMIN_OPERATION_LOGS_COLLECTION].find({}).sort("created_at", -1).limit(max(1, min(int(limit), 200)))
    return [
        {key: bson_value(value) for key, value in item.items() if key != "_id"}
        for item in cursor
    ]



def _manual_recovery_context(db: Any) -> dict[str, Any]:
    readiness = paper_market_readiness(db)
    clock = readiness["clock"]
    timestamp = _utc_stamp(clock["timestamp"])
    current_session = timestamp.tz_convert(EASTERN).date().isoformat()
    run = db[PAPER_MARKET_RUNS_COLLECTION].find_one(
        {"execution_session": current_session},
        sort=[("created_at", -1)],
    )
    plan = db[PAPER_TRADE_PLANS_COLLECTION].find_one(
        {"execution_session": current_session},
        sort=[("created_at", -1)],
    )
    return {
        "readiness": readiness,
        "clock": clock,
        "timestamp": timestamp,
        "current_session": current_session,
        "run": run,
        "plan": plan,
    }


def paper_market_manual_recovery_status(db: Any) -> dict[str, Any]:
    try:
        context = _manual_recovery_context(db)
    except Exception as exc:
        return {
            "available": False,
            "can_prepare": False,
            "can_execute": False,
            "reason": str(exc),
        }

    clock = context["clock"]
    run = context["run"] or {}
    plan = context["plan"] or {}
    market_open = bool(clock.get("is_open"))
    plan_status = str(plan.get("status") or "")
    run_present = bool(run)
    blocked_plan = plan_status == "executing"
    can_prepare = market_open and run_present and not blocked_plan
    can_execute = market_open and plan_status == "prepared"

    if not market_open:
        prepare_reason = "Manual same-session recovery is available only while the regular market is open."
    elif not run_present:
        prepare_reason = "No scheduled Paper run exists for the current regular session."
    elif blocked_plan:
        prepare_reason = f"The current-session Paper plan cannot be replaced while status={plan_status}."
    else:
        prepare_reason = None

    if not market_open:
        execute_reason = "Manual execution is available only while the regular market is open."
    elif plan_status != "prepared":
        execute_reason = "Run manual analysis first to create a fresh prepared plan for the current session."
    else:
        execute_reason = None

    return {
        "available": bool(market_open and run_present),
        "market_open": market_open,
        "current_session": context["current_session"],
        "can_prepare": can_prepare,
        "prepare_reason": prepare_reason,
        "can_execute": can_execute,
        "execute_reason": execute_reason,
        "run_id": run.get("run_id"),
        "run_status": run.get("status"),
        "run_phase": run.get("phase"),
        "expected_market_open": bson_value(run.get("expected_market_open")),
        "plan_id": plan.get("plan_id"),
        "plan_status": plan_status or None,
        "decision_date": plan.get("decision_date"),
        "current_asset": plan.get("current_asset"),
        "target_asset": plan.get("target_asset"),
        "action": plan.get("action"),
        "manual_current_session_recovery": bool(plan.get("manual_current_session_recovery")),
    }


def prepare_manual_current_session_plan(
    db: Any,
    *,
    actor_email: str | None = None,
) -> dict[str, Any]:
    context = _manual_recovery_context(db)
    clock = context["clock"]
    run = context["run"]
    existing = context["plan"]
    if not bool(clock.get("is_open")):
        raise RuntimeError("Manual same-session analysis requires the regular market to be open.")
    if run is None:
        raise RuntimeError("No scheduled Paper run exists for the current regular session.")
    existing_status = str((existing or {}).get("status") or "")
    if existing_status == "executing":
        raise RuntimeError(
            f"The current-session Paper plan cannot be replaced while status={existing_status}."
        )
    expected_open = run.get("expected_market_open")
    if expected_open is None:
        raise RuntimeError("The scheduled Paper run does not contain expected_market_open.")

    plan = prepare_next_paper_plan(
        db,
        replace=existing is not None,
        allow_open_market=True,
        execution_session_override=context["current_session"],
        expected_market_open_override=expected_open,
        refresh_source="manual_current_session_recovery",
    )
    now = utc_now()
    db[PAPER_MARKET_RUNS_COLLECTION].update_one(
        {"run_id": str(run["run_id"])},
        {"$set": {
            "manual_recovery_prepared_at": now,
            "manual_recovery_plan_id": str(plan["plan_id"]),
            "manual_recovery_actor_email": (actor_email or "").strip().lower() or None,
            "updated_at": now,
        }},
    )
    mode = _control_mode(_automation_document(db))
    _record_admin_operation(
        db,
        action="manual_current_session_analysis",
        previous_mode=mode,
        new_mode=mode,
        reason=f"Prepared manual recovery plan {plan['plan_id']} for {context['current_session']}.",
        actor_email=actor_email,
    )
    return {
        "status": "prepared",
        "plan": {key: bson_value(value) for key, value in plan.items() if key != "_id"},
        "manual_recovery": paper_market_manual_recovery_status(db),
    }


def execute_manual_current_session_plan(
    db: Any,
    *,
    plan_id: str | None = None,
    actor_email: str | None = None,
) -> dict[str, Any]:
    context = _manual_recovery_context(db)
    clock = context["clock"]
    if not bool(clock.get("is_open")):
        raise RuntimeError("Manual same-session execution requires the regular market to be open.")
    mode = _control_mode(_automation_document(db))
    if mode in {"paused", "stopped"}:
        raise RuntimeError(
            f"Manual execution is blocked while Trader control mode is {mode}. Start the Trader first."
        )

    query: dict[str, Any] = {
        "execution_session": context["current_session"],
        "status": "prepared",
    }
    if plan_id:
        query["plan_id"] = str(plan_id)
    plan = db[PAPER_TRADE_PLANS_COLLECTION].find_one(query, sort=[("created_at", -1)])
    if plan is None:
        raise RuntimeError("No prepared manual-recovery Paper plan was found for the current session.")
    action = str(plan.get("action") or "").strip().lower()
    if mode == "exit_only" and action not in EXIT_ONLY_ALLOWED_ACTIONS:
        raise RuntimeError(
            f"Manual action {action!r} is blocked by exit-only Trader mode."
        )

    try:
        result = execute_prepared_paper_plan(db, plan_id=str(plan["plan_id"]))
    except Exception as exc:
        _record_admin_operation(
            db,
            action="manual_current_session_execution",
            previous_mode=mode,
            new_mode=mode,
            reason=str(exc),
            actor_email=actor_email,
            success=False,
        )
        raise

    run = context["run"] or {}
    if run.get("run_id"):
        now = utc_now()
        db[PAPER_MARKET_RUNS_COLLECTION].update_one(
            {"run_id": str(run["run_id"])},
            {"$set": {
                "manual_recovery_executed_at": now,
                "manual_recovery_execution_result": bson_value(result),
                "manual_recovery_actor_email": (actor_email or "").strip().lower() or None,
                "updated_at": now,
            }},
        )
    _record_admin_operation(
        db,
        action="manual_current_session_execution",
        previous_mode=mode,
        new_mode=mode,
        reason=f"Executed manual recovery plan {plan['plan_id']} for {context['current_session']}.",
        actor_email=actor_email,
    )
    return {
        "status": "executed",
        "result": bson_value(result),
        "manual_recovery": paper_market_manual_recovery_status(db),
    }


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
        "analysis_policy",
        "premarket_analysis_minutes",
        "premarket_analysis_at",
        "premarket_analysis_started_at",
        "premarket_analysis_completed_at",
        "winner_strategy_id",
        "winner_strategy_name",
        "winner_strategy_revision",
        "winner_configuration_hash",
        "winner_assets_count",
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
    settings = PaperTradingSettings.model_validate(readiness["settings"])
    expected_open = _utc_stamp(clock["next_open"])
    analysis_at = _premarket_analysis_at(expected_open, settings)
    execution_session = expected_open.tz_convert(EASTERN).date().isoformat()
    run_id = f"paper-market-{execution_session}-{uuid.uuid4().hex[:8]}"
    now = utc_now()
    market_timestamp = _utc_stamp(clock["timestamp"])
    phase = (
        "ready_for_premarket_analysis"
        if market_timestamp >= analysis_at and not bool(clock["is_open"])
        else "waiting_for_premarket_analysis"
    )
    first_message = (
        "Paper market run armed for mandatory pre-market analysis. "
        f"Analysis begins at or after {analysis_at.isoformat()}; "
        f"expected open: {expected_open.isoformat()}."
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
        "analysis_policy": PREMARKET_ANALYSIS_POLICY,
        "premarket_analysis_minutes": int(settings.premarket_analysis_minutes),
        "premarket_analysis_at": analysis_at.to_pydatetime(),
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
                "phase": "refreshing_market_data_and_preparing_premarket_plan",
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
    runtime_training = get_system_settings(db)["training"]
    if not bool(runtime_training["enabled"]) or not bool(runtime_training["automatic_training_enabled"]):
        phase = "training_disabled" if not bool(runtime_training["enabled"]) else "automatic_training_disabled"
        message = "Waiting for training to be enabled in System Settings."
        db[PAPER_MARKET_RUNS_COLLECTION].update_one(
            {"run_id": run_id, "status": "armed"},
            {
                "$set": {
                    "phase": phase,
                    "updated_at": utc_now(),
                    "last_message": message,
                }
            },
        )
        return
    expected_open = _utc_stamp(run["expected_market_open"])
    settings = _paper_settings(db)
    analysis_at = _premarket_analysis_at(expected_open, settings)
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
            message="The expected market open arrived before mandatory pre-market analysis completed.",
        )
        return

    if now < analysis_at:
        if (
            str(run.get("phase") or "") != "waiting_for_premarket_analysis"
            or str(run.get("analysis_policy") or "") != PREMARKET_ANALYSIS_POLICY
            or run.get("premarket_analysis_at") is None
            or int(run.get("premarket_analysis_minutes") or 0) != int(settings.premarket_analysis_minutes)
        ):
            db[PAPER_MARKET_RUNS_COLLECTION].update_one(
                {"run_id": run_id, "status": "armed"},
                {
                    "$set": {
                        "phase": "waiting_for_premarket_analysis",
                        "analysis_policy": PREMARKET_ANALYSIS_POLICY,
                        "premarket_analysis_minutes": int(settings.premarket_analysis_minutes),
                        "premarket_analysis_at": analysis_at.to_pydatetime(),
                        "updated_at": utc_now(),
                        "last_message": (
                            "Waiting for the mandatory pre-market analysis window before refreshing data and retraining XGBoost."
                        ),
                    }
                },
            )
        return

    if bool(clock["is_open"]):
        _finish_run(
            db,
            run_id,
            status="failed",
            message="The regular market opened before mandatory pre-market analysis completed.",
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
    started_at = utc_now()
    db[PAPER_MARKET_RUNS_COLLECTION].update_one(
        {"run_id": run_id, "status": "preparing", "worker_id": worker_id},
        {
            "$set": {
                "analysis_policy": PREMARKET_ANALYSIS_POLICY,
                "premarket_analysis_minutes": int(settings.premarket_analysis_minutes),
                "premarket_analysis_at": analysis_at.to_pydatetime(),
                "premarket_analysis_started_at": started_at,
            }
        },
    )
    _append_log(
        db,
        run_id,
        "Refreshing completed daily data and training the locked XGBoost decision during the mandatory pre-market window.",
    )
    try:
        existing = db[PAPER_TRADE_PLANS_COLLECTION].find_one(
            {"execution_session": str(run["execution_session"])},
            sort=[("created_at", -1)],
        )
        if existing is not None and str(existing.get("status") or "") in {"executing", "executed"}:
            raise RuntimeError(
                "A paper plan for this execution session has already entered execution and cannot be replaced safely."
            )
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
                    "cancel_reason": "Cancellation requested during mandatory pre-market analysis.",
                },
            )
            _finish_run(
                db,
                run_id,
                status="cancelled",
                message="Paper-market run cancelled after pre-market analysis and before execution.",
            )
            return

        completed_at = utc_now()
        db[PAPER_MARKET_RUNS_COLLECTION].update_one(
            {"run_id": run_id, "status": "preparing", "worker_id": worker_id},
            {
                "$set": {
                    "status": "prepared",
                    "phase": "waiting_for_next_market_open",
                    "analysis_policy": PREMARKET_ANALYSIS_POLICY,
                    "premarket_analysis_minutes": int(settings.premarket_analysis_minutes),
                    "premarket_analysis_at": analysis_at.to_pydatetime(),
                    "premarket_analysis_completed_at": completed_at,
                    "plan_id": str(plan["plan_id"]),
                    "action": str(plan["action"]),
                    "target_asset": str(plan["target_asset"]),
                    "decision_date": str(plan["decision_date"]),
                    "winner_strategy_id": str(plan["winner_strategy_id"]),
                    "winner_strategy_name": str(plan["winner_strategy_name"]),
                    "winner_strategy_revision": int(plan["winner_strategy_revision"]),
                    "winner_configuration_hash": str(plan["winner_configuration_hash"]),
                    "winner_assets_count": len(plan["winner_assets"]),
                    "updated_at": completed_at,
                    "last_message": (
                        f"Mandatory pre-market XGBoost plan prepared: action={plan['action']}, target={plan['target_asset']}."
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
                            f"{completed_at.isoformat()} — Mandatory pre-market XGBoost plan prepared: action={plan['action']}, target={plan['target_asset']}."
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
                        "phase": "waiting_to_retry_premarket_analysis",
                        "updated_at": now_utc,
                        "next_retry_at": retry_at,
                        "last_error": str(exc),
                        "last_message": f"Pre-market analysis failed and will be retried: {exc}",
                    },
                    "$unset": {"worker_id": "", "lease_expires_at": ""},
                    "$push": {
                        "logs": {
                            "$each": [
                                f"{now_utc.isoformat()} — Pre-market analysis failed and will be retried: {exc}"
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
                message=f"Could not complete mandatory pre-market analysis before market open: {exc}",
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

    controller = _automation_document(db) or {}
    control_mode = _control_mode(controller)
    if control_mode == "paused":
        return
    if control_mode == "stopped":
        return
    if control_mode == "exit_only":
        action = str(run.get("action") or "").strip().lower()
        if action not in EXIT_ONLY_ALLOWED_ACTIONS:
            plan_id = str(run.get("plan_id") or "").strip()
            if plan_id:
                update_paper_trade_plan(
                    db,
                    plan_id,
                    {
                        "status": "cancelled",
                        "cancelled_at": utc_now(),
                        "cancel_reason": "Blocked by exit-only Trader mode.",
                    },
                )
            _finish_run(
                db,
                run_id,
                status="completed",
                message="The prepared plan was not executed because exit-only mode blocks new entries and rotations.",
                extra={"execution_result": {"action": action, "blocked_by_control_mode": "exit_only"}},
            )
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


def _prepared_run_has_valid_premarket_analysis(run: dict[str, Any]) -> bool:
    if str(run.get("analysis_policy") or "") != PREMARKET_ANALYSIS_POLICY:
        return False
    completed = run.get("premarket_analysis_completed_at")
    analysis_at = run.get("premarket_analysis_at")
    expected_open = run.get("expected_market_open")
    if completed is None or analysis_at is None or expected_open is None:
        return False
    completed_at = _utc_stamp(completed)
    return _utc_stamp(analysis_at) <= completed_at < _utc_stamp(expected_open)


def _rearm_prepared_run_for_premarket_analysis(
    db: Any,
    run: dict[str, Any],
) -> dict[str, Any]:
    if _prepared_run_has_valid_premarket_analysis(run):
        return run
    run_id = str(run["run_id"])
    plan_id = str(run.get("plan_id") or "").strip()
    if plan_id:
        update_paper_trade_plan(
            db,
            plan_id,
            {
                "status": "cancelled",
                "cancelled_at": utc_now(),
                "cancel_reason": "Superseded by mandatory pre-market reanalysis before execution.",
            },
        )
    now = utc_now()
    db[PAPER_MARKET_RUNS_COLLECTION].update_one(
        {"run_id": run_id, "status": "prepared"},
        {
            "$set": {
                "status": "armed",
                "phase": "waiting_for_premarket_analysis",
                "analysis_policy": PREMARKET_ANALYSIS_POLICY,
                "updated_at": now,
                "next_retry_at": now,
                "last_message": (
                    "A plan created before the mandatory pre-market policy was discarded; current data will be refreshed and XGBoost retrained before the next open."
                ),
            },
            "$unset": {
                "plan_id": "",
                "action": "",
                "target_asset": "",
                "decision_date": "",
                "premarket_analysis_started_at": "",
                "premarket_analysis_completed_at": "",
                "worker_id": "",
                "lease_expires_at": "",
            },
            "$push": {
                "logs": {
                    "$each": [
                        f"{now.isoformat()} — Legacy prepared plan discarded for mandatory pre-market reanalysis."
                    ],
                    "$slice": -100,
                }
            },
        },
    )
    return db[PAPER_MARKET_RUNS_COLLECTION].find_one({"run_id": run_id}) or run


def _ensure_continuous_run(db: Any) -> dict[str, Any] | None:
    controller = _automation_document(db)
    active = db[PAPER_MARKET_RUNS_COLLECTION].find_one({"active_key": ACTIVE_KEY})

    
    
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
    controller = _automation_document(db) or {}
    if _control_mode(controller) in {"paused", "stopped"}:
        return

    run = _ensure_continuous_run(db)
    if run is None:
        return

    # Keep the immutable Winner's mutable market-data position current only when
    # the Paper robot is actually armed/continuous. During the session this is a
    # no-op (cutoff remains D-1); after close plus the daily-bar safety buffer it
    # advances once to D.
    live_market = refresh_trader_live_market_data(db, source="paper_scheduler_live_refresh")
    if bool(live_market.get("pending_retry")):
        return
    status = str(run.get("status") or "")
    if status == "prepared" and not _prepared_run_has_valid_premarket_analysis(run):
        run = _rearm_prepared_run_for_premarket_analysis(db, run)
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
    strategy_control = db[STRATEGY_CONTROL_COLLECTION].find_one({"_id": "default"}) or {}
    active = db[PAPER_MARKET_RUNS_COLLECTION].find_one({"active_key": ACTIVE_KEY})
    latest = active or db[PAPER_MARKET_RUNS_COLLECTION].find_one({}, sort=[("created_at", -1)])
    runtime = _SCHEDULER.snapshot()
    enabled = bool(controller.get("enabled"))
    control_mode = _control_mode(controller)
    status = str(controller.get("status") or control_mode)
    try:
        trader_winner = get_trader_winner_summary(db)
        if public and trader_winner is not None:
            trader_winner = {"name": trader_winner.get("name")}
    except Exception:
        trader_winner = None
    output: dict[str, Any] = {
        "enabled": enabled,
        "control_mode": control_mode,
        "mode": str(controller.get("mode") or AUTOMATION_MODE),
        "status": status,
        "phase": str(controller.get("phase") or (latest or {}).get("phase") or status),
        "scheduler_alive": bool(runtime.get("scheduler_alive")),
        "scheduler_started_at": runtime.get("scheduler_started_at"),
        "last_scheduler_tick_at": runtime.get("last_scheduler_tick_at"),
        "last_successful_scheduler_tick_at": runtime.get("last_successful_scheduler_tick_at"),
        "next_market_open": bson_value((active or {}).get("expected_market_open") or controller.get("next_market_open")),
        "next_execution_session": (active or {}).get("execution_session") or controller.get("next_execution_session"),
        "next_premarket_analysis_at": bson_value((active or {}).get("premarket_analysis_at")),
        "premarket_analysis_minutes": (active or {}).get("premarket_analysis_minutes"),
        "active_run": _public_robot_run(active),
        "latest_run": _public_robot_run(latest),
        "updated_at": bson_value(controller.get("updated_at")),
        "trader_winner": trader_winner,
        "live_market_cutoff": strategy_control.get("live_market_cutoff"),
        "live_market_cutoff_updated_at": bson_value(strategy_control.get("live_market_cutoff_updated_at")),
    }
    if not public:
        output.update({
            "scheduler_worker_id": runtime.get("scheduler_worker_id"),
            "scheduler_last_error": runtime.get("scheduler_last_error"),
            "poll_seconds": runtime.get("poll_seconds"),
            "last_error": controller.get("last_error"),
            "next_retry_at": bson_value(controller.get("next_retry_at")),
            "live_market_refresh_target": strategy_control.get("live_market_refresh_target"),
            "live_market_refresh_next_retry_at": bson_value(strategy_control.get("live_market_refresh_next_retry_at")),
            "live_market_refresh_last_error": strategy_control.get("live_market_refresh_last_error"),
        })
    return output


def start_paper_market_scheduler() -> None:
    _SCHEDULER.start()


def stop_paper_market_scheduler() -> None:
    _SCHEDULER.stop()
