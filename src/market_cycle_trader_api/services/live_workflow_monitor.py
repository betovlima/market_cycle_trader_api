"""Read-only Live workflow monitoring and per-cycle calculation supervision.

MongoDB's existing Live session state and broker ledger remain authoritative.
Tenacity scopes one computation invocation; the existing scheduler retries the
*entire* calculation on subsequent eligible observations, with no global
attempt budget. Never apply Tenacity to Alpaca order submissions.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4

from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_none
from transitions import Machine

from ..infrastructure.trading.alpaca_live import create_live_readonly_client
from .live_winner_session import LIVE_RUNS

LIVE_ATTEMPTS = "live_cycle_attempts"
ORDER_LEDGER = "live_winner_broker_orders"

# These are *observation* states, not a second source of trading authority.
STATES = (
    "pending", "running", "completed", "waiting",
    "retry_pending", "reconciliation_required",
)


class LiveStageMachine:
    """Validate transitions within one scheduler invocation."""

    def __init__(self) -> None:
        self.machine = Machine(
            model=self, states=STATES, initial="pending",
            auto_transitions=False, send_event=False,
            transitions=[
                {"trigger": "start", "source": "pending", "dest": "running"},
                {"trigger": "complete", "source": "running", "dest": "completed"},
                {"trigger": "wait_for_data", "source": "running", "dest": "waiting"},
                {"trigger": "retry_from_start", "source": "running", "dest": "retry_pending"},
                {"trigger": "reconcile", "source": "running", "dest": "reconciliation_required"},
            ],
        )


def _timestamp(value: Any) -> str | None:
    return value.isoformat() if hasattr(value, "isoformat") else None


def _persist_best_effort(db: Any, item_id: str, updates: dict[str, Any]) -> None:
    """A diagnostic database failure must never trigger or duplicate a trade."""
    try:
        db[LIVE_ATTEMPTS].update_one({"_id": item_id}, {"$set": updates})
    except Exception:
        pass


def _outcome_state(result: Any, *, brokerage: bool) -> str:
    status = str(result.get("status") or "") if isinstance(result, dict) else ""
    if brokerage:
        if status == "completed":
            return "completed"
        return "reconciliation_required"
    if status in {"waiting_for_open_retry", "waiting_for_post_close_retry"}:
        return "retry_pending"
    if status.startswith("waiting_") or status in {
        "live_open_recalibration_in_progress",
        "post_close_training_in_progress",
        "recalibrated_no_orders",
        "not_eligible_for_live_open_refresh",
        "open_recalibration_claimed_elsewhere",
        "stale_signal_replacement_claimed_elsewhere",
    }:
        # A prepared signal is a valid calculated result; another stage
        # (broker execution) decides if any real orders are permitted.
        return "completed" if status == "recalibrated_no_orders" else "waiting"
    return "completed"


def execute_monitored_live_stage(
    db: Any, *, session: str, stage: str,
    operation: Callable[[], Any], brokerage: bool = False,
) -> Any:
    """Record a single real processing attempt, never impose a time limit.

    Retry scope: one full calculation invocation per scheduler observation.
    There is NO cumulative retry cap. If a calculation fails, existing Live
    services persist a waiting state and the next eligible observation
    re-runs from their first download. No Tenacity retries around broker work.
    """
    flow = LiveStageMachine()
    flow.start()
    run = db[LIVE_RUNS].find_one({"execution_session": session}) or {}
    account_id = str(run.get("live_account_id") or "")
    attempt_id = uuid4().hex
    key = f"{account_id}:{session}:{attempt_id}"
    started = datetime.now(timezone.utc)
    doc = {
        "_id": key, "account_id": account_id,
        "execution_session": session, "stage": stage,
        "attempt_id": attempt_id, "status": flow.state,
        "started_at": started, "brokerage": brokerage,
        "error_type": None,
    }
    if account_id:
        try:
            db[LIVE_ATTEMPTS].insert_one(doc)
        except Exception:
            pass
    try:
        if brokerage:
            # Broker requests are never re-sent by a computation retry policy.
            result = operation()
        else:
            # One invocation per scheduler event, NOT a maximum of one attempt
            # per session. The scheduler re-enters the complete flow after a
            # recoverable failure, without spinning on unavailable providers.
            policy = Retrying(
                retry=retry_if_exception_type(Exception),
                stop=stop_after_attempt(1),
                wait=wait_none(),
                reraise=True,
            )
            result = policy(operation)

        outcome = _outcome_state(result, brokerage=brokerage)
        if outcome == "waiting":
            flow.wait_for_data()
        elif outcome == "retry_pending":
            flow.retry_from_start()
        elif outcome == "reconciliation_required":
            flow.reconcile()
        else:
            flow.complete()
        if account_id:
            _persist_best_effort(db, key, {
                "status": flow.state, "finished_at": datetime.now(timezone.utc),
                "result_status": result.get("status") if isinstance(result, dict) else None,
            })
        return result
    except Exception as exc:
        if brokerage:
            flow.reconcile()
        else:
            flow.retry_from_start()
        if account_id:
            _persist_best_effort(db, key, {
                "status": flow.state, "finished_at": datetime.now(timezone.utc),
                "error_type": type(exc).__name__,
            })
        raise


def live_workflow_monitor(
    db: Any, *, session: str | None = None, limit: int = 20,
) -> dict[str, Any]:
    """Account-scoped, read-only monitoring. No order submissions or retries."""
    broker = create_live_readonly_client()
    account_id = str(getattr(broker.get_account(), "id", "") or "")
    if not account_id:
        raise RuntimeError("Alpaca Live account identity is unavailable.")
    if session is None:
        run = db[LIVE_RUNS].find_one(
            {"live_account_id": account_id},
            sort=[("execution_session", -1)],
        )
    else:
        run = db[LIVE_RUNS].find_one({
            "live_account_id": account_id, "execution_session": session,
        })
    if not run:
        return {
            "execution_session": session, "status": "no_live_session",
            "attempts": [], "orders_requiring_reconciliation": 0,
        }
    session = str(run["execution_session"])
    max_items = min(max(int(limit), 1), 100)
    cursor = db[LIVE_ATTEMPTS].find(
        {"account_id": account_id, "execution_session": session},
        {"_id": 0, "account_id": 0},
    ).sort("started_at", -1).limit(max_items)
    attempts = []
    for attempt in cursor:
        attempts.append({
            "attempt_id": attempt.get("attempt_id"),
            "stage": attempt.get("stage"),
            "status": attempt.get("status"),
            "brokerage": bool(attempt.get("brokerage")),
            "started_at": _timestamp(attempt.get("started_at")),
            "finished_at": _timestamp(attempt.get("finished_at")),
            "error_type": attempt.get("error_type"),
            "result_status": attempt.get("result_status"),
        })
    unresolved = db[ORDER_LEDGER].count_documents({
        "broker_account_id": account_id,
        "execution_session": session,
        "$or": [
            {"requires_reconciliation": True},
            {"status": {"$in": [
                "intent_persisted", "submitted", "partially_filled",
            ]}},
        ],
    })
    status = str(run.get("status") or "unknown")
    if unresolved or status in {"executing", "review_required"}:
        recovery = "broker_reconciliation_required"
    elif status in {"preparing_signal", "open_recalibration"}:
        recovery = "processing"
    elif status in {"waiting_for_winner_signal", "waiting_for_open_recalibration"}:
        recovery = "eligible_for_full_recalculation"
    else:
        recovery = "not_required"
    return {
        "execution_session": session,
        "status": status,
        "phase": run.get("phase"),
        "recovery": recovery,
        "preparation_attempts": int(run.get("preparation_attempts") or 0),
        "open_preparation_attempts": int(run.get("open_preparation_attempts") or 0),
        "last_preparation_error_type": run.get("last_preparation_error_type"),
        "last_open_calibration_error_type": (
            "RuntimeError" if run.get("last_open_calibration_error") else None
        ),
        "orders_requiring_reconciliation": unresolved,
        "attempts": attempts,
        "limit": max_items,
    }
