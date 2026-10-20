"""Fail-closed Live order executor for independently recalibrated Winner sessions.

Real submissions follow the existing Live Start/Pause/Stop operating mode. It cannot use a Paper account or Paper order/state collections.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import exchange_calendars as xcals
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

from ..infrastructure.trading.alpaca_live import create_live_execution_client
from ..infrastructure.trading.alpaca_live_orders import (
    asset_snapshot, submit_market_buy_notional, submit_market_sell,
    wait_for_order,
)
from .live_control import live_control_status
from .live_trader_state import verified_live_state, confirm_live_fill_state
from .live_winner_session import LIVE_RUNS
from .live_execution_window import live_execution_window_deadline
from .strategy_lab import get_trader_winner_summary

ORDER_LEDGER = "live_winner_broker_orders"
EASTERN = ZoneInfo("America/New_York")
LIVE_FILL_TIMEOUT_SECONDS = 300
LIVE_FILL_POLL_SECONDS = 2.0


def _assert_live_order_window(broker: Any, session: str, deadline: pd.Timestamp) -> pd.Timestamp:
    """Recheck market/date/deadline directly at the broker before EACH leg."""
    clock = broker.get_clock()
    if not bool(getattr(clock, "is_open", False)):
        raise RuntimeError("LiveExecutionWindowExpired: broker reports market closed.")
    now = pd.Timestamp(clock.timestamp)
    if now.tzinfo is None:
        raise RuntimeError("LiveExecutionWindowExpired: broker timestamp has no timezone.")
    now = now.tz_convert("UTC")
    if now.tz_convert(EASTERN).date().isoformat() != session or now >= deadline:
        raise RuntimeError("LiveExecutionWindowExpired: real order deadline has passed.")
    return now


def _full_live_cash_notional(account: Any) -> float:
    """Use broker-reported cash to size Winner entries without MCT reserve or caps."""
    try:
        cash = Decimal(str(getattr(account, "cash", None)))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise RuntimeError("Invalid Alpaca Live cash balance.") from exc
    if not cash.is_finite() or cash < Decimal("1.00"):
        raise RuntimeError("Alpaca Live cash cannot fund a notional buy.")
    return float(cash.quantize(Decimal("0.01"), rounding=ROUND_DOWN))


def _unique_order_id(account: str, session: str, side: str, asset: str) -> str:
    digest = hashlib.sha256(f"{account}|{session}|{side}|{asset}".encode("utf-8")).hexdigest()[:20]
    return f"mct-live-{side}-{digest}"[:48]


def _execute_leg(
    db: Any, broker: Any, *, account: str, session: str,
    side: str, symbol: str, quantity: float | None, notional: float | None,
    winner_hash: str,
) -> Any:
    client_order_id = _unique_order_id(account, session, side, symbol)
    record = {
        "_id": client_order_id, "market": "live", "broker_account_id": account,
        "execution_session": session, "symbol": symbol, "side": side,
        "winner_configuration_hash": winner_hash,
        "requested_quantity": quantity, "requested_notional": notional,
        "status": "intent_persisted", "created_at": datetime.now(timezone.utc),
    }
    # The unique ledger key prevents duplicate submissions after any restart.
    try:
        db[ORDER_LEDGER].insert_one(record)
    except DuplicateKeyError as exc:
        raise RuntimeError("Live order intent already exists; reconcile it before continuing.") from exc
    try:
        if side == "sell":
            submitted = submit_market_sell(
                broker, symbol=symbol, quantity=float(quantity),
                client_order_id=client_order_id,
            )
        elif side == "buy":
            submitted = submit_market_buy_notional(
                broker, symbol=symbol, notional=float(notional),
                client_order_id=client_order_id,
            )
        else:
            raise RuntimeError("Invalid Live order side.")
        order_id = str(getattr(submitted, "id", "") or "")
        if not order_id:
            raise RuntimeError("Alpaca returned no order ID; reconciliation required.")
        db[ORDER_LEDGER].update_one(
            {"_id": client_order_id},
            {"$set": {"status": "submitted", "broker_order_id": order_id}},
        )
        fill = wait_for_order(
            broker, order_id=order_id,
            timeout_seconds=LIVE_FILL_TIMEOUT_SECONDS,
            poll_interval_seconds=LIVE_FILL_POLL_SECONDS,
        )
        db[ORDER_LEDGER].update_one(
            {"_id": client_order_id},
            {"$set": {
                "status": fill.status, "filled_quantity": fill.filled_quantity,
                "filled_average_price": fill.filled_average_price,
                "filled_notional": fill.filled_notional,
                "filled_at": fill.filled_at,
            }},
        )
        if fill.status != "filled" or float(fill.filled_quantity) <= 0:
            raise RuntimeError("Live order did not fully fill; reconciliation required.")
        if side == "sell" and abs(float(fill.filled_quantity) - float(quantity)) > 1e-6:
            raise RuntimeError("Live sell quantity is incomplete; reconciliation required.")
        return fill
    except Exception as exc:
        db[ORDER_LEDGER].update_one(
            {"_id": client_order_id},
            {"$set": {"requires_reconciliation": True,
                      "error_type": type(exc).__name__,
                      "last_checked_at": datetime.now(timezone.utc)}},
        )
        raise


def execute_prepared_live_winner_session(db: Any, session: str) -> dict[str, Any]:
    """Execute once if explicitly enabled and broker state fully reconciled."""
    control = live_control_status(db)
    if control["control_mode"] not in {"active", "exit_only"}:
        return {"status": "blocked_by_live_mode", "order_submission_enabled": False}
    run = db[LIVE_RUNS].find_one({"execution_session": session})
    if not run or run.get("status") != "recalibrated_no_orders":
        return {"status": "no_recalibrated_live_plan", "order_submission_enabled": False}
    winner = get_trader_winner_summary(db)
    if (not isinstance(winner, dict)
        or str(winner.get("id") or "") != run["winner_strategy_id"]
        or str(winner.get("configuration_hash") or "") != run["winner_configuration_hash"]
        or int(winner.get("revision") or 0) != int(run["winner_revision"])):
        raise RuntimeError("Winner changed since Live recalibration; real orders blocked.")
    action = str(run.get("action") or "")
    if control["control_mode"] == "exit_only" and action not in {"hold", "stay_in_cash", "sell_to_cash"}:
        return {"status": "blocked_by_exit_only_mode", "order_submission_enabled": False}
    if action not in {"hold", "stay_in_cash", "buy", "sell_to_cash", "rotate"}:
        raise RuntimeError("Invalid Winner action.")
    if (not bool(run.get("winner_signal_prepared"))
        or not bool(run.get("post_close_signal_ready"))
        or not run.get("snapshot_sha256")
        or not run.get("target_asset")):
        raise RuntimeError("No verified post-close Live Winner signal; real orders blocked.")
    decision_date = str(run.get("decision_date") or "")[:10]
    expected_cutoff = xcals.get_calendar("XNYS").previous_session(
        pd.Timestamp(session)
    ).date().isoformat()
    if (decision_date != session
        or str(run.get("historical_cutoff") or "") != expected_cutoff
        or str(run.get("analysis_mode") or "") != "live_current_session_intraday"
        or not run.get("market_open_calibration_completed_at")):
        raise RuntimeError("LiveOpenCalibrationRequired: a current-session Winner signal is required before real orders.")

    broker = create_live_execution_client()
    account = broker.get_account()
    account_id = str(getattr(account, "id", "") or "")
    if account_id != run["live_account_id"]:
        raise RuntimeError("Live account mismatch; refusing to trade.")
    if (bool(getattr(account, "account_blocked", True))
        or bool(getattr(account, "trading_blocked", True))
        or bool(getattr(account, "trade_suspended_by_user", False))
        or str(getattr(account, "status", "")).lower() not in {"active", "accountstatus.active"}):
        raise RuntimeError("Live broker account is not permitted to trade.")
    clock = broker.get_clock()
    if not bool(getattr(clock, "is_open", False)):
        return {"status": "market_closed", "order_submission_enabled": False}
    current = pd.Timestamp(clock.timestamp)
    current = current.tz_localize("UTC") if current.tzinfo is None else current.tz_convert("UTC")
    if current.tz_convert(EASTERN).date().isoformat() != session:
        raise RuntimeError("Stale Live session.")
    expected_open_raw = run.get("expected_market_open")
    if expected_open_raw is None:
        raise RuntimeError("LiveExecutionWindowUnavailable: expected open missing; no real orders.")
    expected_open = pd.Timestamp(expected_open_raw)
    if expected_open.tzinfo is None:
        raise RuntimeError("LiveExecutionWindowUnavailable: expected open must be timezone-aware.")
    expected_open = expected_open.tz_convert("UTC")
    deadline, _ = live_execution_window_deadline(expected_open, market_clock=clock)
    if current >= deadline:
        db[LIVE_RUNS].update_one(
            {"_id": run["_id"], "status": "recalibrated_no_orders"},
            {"$set": {
                "status": "review_required",
                "phase": "execution_window_expired",
                "window_deadline_at": deadline.to_pydatetime(),
                "window_expired_at": datetime.now(timezone.utc),
                "order_submission_enabled": False,
            }},
        )
        return {"status": "execution_window_expired", "order_submission_enabled": False}
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest
    if broker.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=100)):
        raise RuntimeError("Existing Live broker orders require reconciliation.")

    state = verified_live_state(db, broker, session=session)
    if (state["revision"] != run.get("live_state_revision")
        or state["managed_symbol"] != run.get("current_asset")):
        raise RuntimeError("Live position changed since Winner model calibration.")
    target = str(run["target_asset"]).upper()
    if action in {"hold", "stay_in_cash"} and state["managed_symbol"] != target:
        raise RuntimeError("Winner HOLD action does not match broker position.")
    if action == "buy" and state["managed_symbol"] != "CASH":
        raise RuntimeError("Winner buy requires a flat Live account.")
    if action in {"sell_to_cash", "rotate"} and state["managed_symbol"] == "CASH":
        raise RuntimeError("Winner sell/rotate requires a managed Live position.")

    if action in {"buy", "rotate"}:
        asset = asset_snapshot(broker, target)
        if (not asset["tradable"] or not asset["fractionable"]
            or asset["status"] not in {"active"}):
            raise RuntimeError("Winner target cannot be traded fractionally in Alpaca Live.")
    # Atomically claim the full session before any broker submission.
    claimed = db[LIVE_RUNS].find_one_and_update(
        {"_id": run["_id"], "status": "recalibrated_no_orders",
         "winner_configuration_hash": run["winner_configuration_hash"]},
        {"$set": {"status": "executing", "phase": "checking_broker_before_orders",
                  "execution_started_at": datetime.now(timezone.utc)}},
        return_document=ReturnDocument.AFTER,
    )
    if claimed is None:
        raise RuntimeError("Live session already claimed or changed.")

    try:
        if action in {"sell_to_cash", "rotate"}:
            if live_control_status(db)["control_mode"] not in {"active", "exit_only"}:
                raise RuntimeError("Live mode changed before sell submission.")
            _assert_live_order_window(broker, session, deadline)
            fill = _execute_leg(
                db, broker, account=account_id, session=session,
                side="sell", symbol=state["managed_symbol"],
                quantity=state["managed_quantity"], notional=None,
                winner_hash=run["winner_configuration_hash"],
            )
            confirm_live_fill_state(
                db, previous=state, session=session, symbol="CASH",
                quantity=0.0, entry_session=None,
            )
            state = {**state, "managed_symbol": "CASH", "managed_quantity": 0.0,
                     "revision": state["revision"] + 1}
        if action in {"buy", "rotate"}:
            if live_control_status(db)["control_mode"] != "active":
                raise RuntimeError("Live mode no longer permits a buy.")
            _assert_live_order_window(broker, session, deadline)
            account = broker.get_account()
            notional = _full_live_cash_notional(account)
            fill = _execute_leg(
                db, broker, account=account_id, session=session,
                side="buy", symbol=target, quantity=None, notional=notional,
                winner_hash=run["winner_configuration_hash"],
            )
            confirm_live_fill_state(
                db, previous=state, session=session,
                symbol=target, quantity=float(fill.filled_quantity),
                entry_session=session,
            )
        db[LIVE_RUNS].update_one(
            {"_id": run["_id"], "status": "executing"},
            {"$set": {"status": "completed", "phase": "live_orders_reconciled",
                      "completed_at": datetime.now(timezone.utc)}},
        )
        return {"status": "completed", "action": action, "target_asset": target,
                "execution_session": session, "order_submission_enabled": True}
    except Exception as exc:
        db[LIVE_RUNS].update_one(
            {"_id": run["_id"], "status": "executing"},
            {"$set": {"status": "review_required",
                      "phase": "broker_reconciliation_required",
                      "failure_type": type(exc).__name__,
                      "order_submission_enabled": False}},
        )
        raise


# Compatibility export for existing callers and previously recorded sessions.
execute_recalibrated_live_winner_session = execute_prepared_live_winner_session
