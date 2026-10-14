"""Conservative, auditable reconciliation of abandoned Paper HOLD plans.

Never infer that a submitted order was not filled. Only past-session,
non-trading plans without any order or live scheduler reference are eligible.
The original plan is retained, with an atomic status transition and evidence.
"""
from __future__ import annotations

import uuid
from datetime import date
from typing import Any
from zoneinfo import ZoneInfo

from pymongo import ReturnDocument

from ..infrastructure.persistence.mongo_repository import (
    ADMIN_OPERATION_LOGS_COLLECTION,
    PAPER_MARKET_RUNS_COLLECTION,
    PAPER_TRADE_ORDERS_COLLECTION,
    PAPER_TRADE_PLANS_COLLECTION,
    bson_value,
    utc_now,
)

EASTERN = ZoneInfo("America/New_York")
TERMINAL_RUN_STATUSES = frozenset({"completed", "failed", "cancelled"})
SCAN_LIMIT = 250
RECONCILIATION_REASON = "Past-session, non-trading plan expired without execution."


def _market_today() -> date:
    return utc_now().astimezone(EASTERN).date()


def _linked_runs(db: Any, plan_id: str) -> list[dict[str, Any]]:
    return list(
        db[PAPER_MARKET_RUNS_COLLECTION].find(
            {"$or": [{"plan_id": plan_id}, {"manual_recovery_plan_id": plan_id}]},
            {
                "_id": 0, "run_id": 1, "status": 1, "active_key": 1,
                "execution_session": 1,
            },
        ).limit(20)
    )


def inspect_plan(db: Any, plan: dict[str, Any], *, today: date) -> dict[str, Any]:
    """Return explicit disqualifiers rather than guessing a historical fill."""
    plan_id = str(plan.get("plan_id") or "")
    status = str(plan.get("status") or "")
    action = str(plan.get("action") or "")
    current = str(plan.get("current_asset") or "").upper()
    target = str(plan.get("target_asset") or "").upper()
    session = str(plan.get("execution_session") or "")
    blockers: list[str] = []

    if not plan_id:
        blockers.append("missing_plan_id")
    if status != "prepared":
        blockers.append("status_is_not_prepared")
    try:
        execution_day = date.fromisoformat(session)
    except ValueError:
        execution_day = None
        blockers.append("invalid_execution_session")
    if execution_day is not None and execution_day >= today:
        blockers.append("session_not_in_past")
    if action not in {"hold", "stay_in_cash"} or current != target:
        blockers.append("not_a_non_trading_decision")
    if action == "stay_in_cash" and current != "CASH":
        blockers.append("invalid_stay_in_cash_state")
    if any(plan.get(key) for key in ("execution_started_at", "filled_orders", "final_state")):
        blockers.append("execution_evidence_on_plan")
    if plan.get("execution_stage") not in (None, "", "prepared"):
        blockers.append("execution_stage_requires_review")

    orders_exist = False
    runs: list[dict[str, Any]] = []
    if plan_id:
        orders_exist = db[PAPER_TRADE_ORDERS_COLLECTION].find_one(
            {"plan_id": plan_id}, {"_id": 1}
        ) is not None
        if orders_exist:
            blockers.append("associated_order_requires_review")
        runs = _linked_runs(db, plan_id)
        if len(runs) == 20:
            blockers.append("run_history_truncated")
        if any(
            bool(run.get("active_key"))
            or str(run.get("status") or "") not in TERMINAL_RUN_STATUSES
            for run in runs
        ):
            blockers.append("linked_run_not_terminal")

    return {
        "plan_id": plan_id,
        "status": status,
        "execution_session": session,
        "action": action,
        "current_asset": current,
        "target_asset": target,
        "eligible": not blockers,
        "blockers": blockers,
        "associated_orders": orders_exist,
        "linked_runs": [
            {
                "run_id": str(run.get("run_id") or ""),
                "status": str(run.get("status") or ""),
            }
            for run in runs
        ],
    }


def preview_paper_plan_reconciliation(db: Any) -> dict[str, Any]:
    """Read-only. Never changes Paper state, history, or orders."""
    today = _market_today()
    plans = list(
        db[PAPER_TRADE_PLANS_COLLECTION]
        .find({"status": "prepared"})
        .sort([("created_at", 1)])
        .limit(SCAN_LIMIT + 1)
    )
    truncated = len(plans) > SCAN_LIMIT
    items = [inspect_plan(db, plan, today=today) for plan in plans[:SCAN_LIMIT]]
    return {
        "market_date": today.isoformat(),
        "scanned": len(items),
        "eligible_count": sum(int(item["eligible"]) for item in items),
        "blocked_count": sum(int(not item["eligible"]) for item in items),
        "truncated": truncated,
        "items": items,
    }


def close_obsolete_paper_plans(
    db: Any, *, plan_ids: list[str], actor_email: str,
) -> dict[str, Any]:
    """Explicit admin-only request. CAS prevents overwriting newer decisions.

    For each requested plan, confirm it is stale and non-trading, has NO
    linked orders and has no active/incomplete run, then record the previous
    status and reason in the SAME atomic Mongo update as the status change.
    """
    if not plan_ids or len(plan_ids) > 100:
        raise ValueError("Provide between 1 and 100 plan IDs.")
    if len(set(plan_ids)) != len(plan_ids) or any(not p or len(p) > 160 for p in plan_ids):
        raise ValueError("Plan IDs must be distinct, non-empty and at most 160 characters.")

    today = _market_today()
    # Preflight all IDs before changing anything. Repeat per ID just before CAS.
    checked: list[tuple[dict[str, Any] | None, dict[str, Any]]] = []
    for plan_id in plan_ids:
        plan = db[PAPER_TRADE_PLANS_COLLECTION].find_one({"plan_id": plan_id})
        if plan is None:
            inspection = {
                "plan_id": plan_id, "eligible": False, "blockers": ["plan_not_found"]
            }
        else:
            inspection = inspect_plan(db, plan, today=today)
        checked.append((plan, inspection))

    if any(not item["eligible"] for _, item in checked):
        return {
            "status": "blocked",
            "changed_count": 0,
            "results": [
                {"plan_id": item["plan_id"], "status": "blocked" if not item["eligible"] else "not_changed",
                 "blockers": item["blockers"]}
                for _, item in checked
            ],
        }

    results: list[dict[str, Any]] = []
    for initial_plan, _ in checked:
        assert initial_plan is not None
        plan_id = str(initial_plan["plan_id"])
        plan = db[PAPER_TRADE_PLANS_COLLECTION].find_one({"plan_id": plan_id})
        if plan is None:
            results.append({"plan_id": plan_id, "status": "concurrent_change"})
            continue
        inspection = inspect_plan(db, plan, today=today)
        if not inspection["eligible"]:
            results.append({
                "plan_id": plan_id, "status": "concurrent_change",
                "blockers": inspection["blockers"],
            })
            continue
        # Protect against a replaced/updated plan between inspection and write.
        # Every eligibility-critical field is part of this atomic compare-and-set.
        guard: dict[str, Any] = {
            "_id": plan["_id"],
            "plan_id": plan_id,
            "status": "prepared",
            "execution_session": plan.get("execution_session"),
            "action": plan.get("action"),
            "current_asset": plan.get("current_asset"),
            "target_asset": plan.get("target_asset"),
        }
        if "updated_at" in plan:
            guard["updated_at"] = plan["updated_at"]
        else:
            guard["updated_at"] = {"$exists": False}
        now = utc_now()
        audit = {
            "previous_status": "prepared",
            "new_status": "cancelled",
            "reason_code": "expired_unexecuted_non_trading_plan",
            "reason": RECONCILIATION_REASON,
            "actor_email": actor_email,
            "at": now,
            "operation_id": uuid.uuid4().hex,
            "linked_runs_at_check": inspection["linked_runs"],
        }
        updated = db[PAPER_TRADE_PLANS_COLLECTION].find_one_and_update(
            guard,
            {"$set": {
                "status": "cancelled",
                "cancelled_at": now,
                "cancel_reason": RECONCILIATION_REASON,
                "status_reconciliation": audit,
                "updated_at": now,
            }},
            return_document=ReturnDocument.AFTER,
        )
        if updated is None:
            results.append({"plan_id": plan_id, "status": "concurrent_change"})
            continue

        # The on-plan audit survives even if the separate admin log is unavailable.
        try:
            db[ADMIN_OPERATION_LOGS_COLLECTION].insert_one({
                "action": "paper_plan_status_reconciled",
                "plan_id": plan_id,
                "actor_email": actor_email,
                "previous_status": "prepared",
                "new_status": "cancelled",
                "reason_code": audit["reason_code"],
                "operation_id": audit["operation_id"],
                "created_at": now,
                "success": True,
            })
            log_status = "recorded"
        except Exception:
            log_status = "embedded_audit_only"
        results.append({
            "plan_id": plan_id, "status": "cancelled",
            "audit_log": log_status,
        })

    changed = sum(int(item["status"] == "cancelled") for item in results)
    return {
        "status": "completed" if changed == len(plan_ids) else "partial",
        "changed_count": changed,
        "results": bson_value(results),
    }
