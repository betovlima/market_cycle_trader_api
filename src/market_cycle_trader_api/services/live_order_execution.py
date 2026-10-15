"""Fail-closed Live order executor for independently recalibrated Winner sessions.

All real submission requires an explicit deployment-level opt-in plus active
Live mode. It cannot use a Paper account or Paper order/state collections.
"""
from __future__ import annotations

import hashlib
import math
import os
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

from ..infrastructure.trading.alpaca_live import create_live_execution_client
from ..infrastructure.trading.alpaca_paper import (
    asset_snapshot, submit_market_buy_notional, submit_market_sell,
    wait_for_order,
)
from ..infrastructure.persistence.mongo_repository import get_paper_trading_settings
from ..schemas.paper_trading import PaperTradingSettings
from .live_control import live_control_status
from .live_trader_state import verified_live_state, confirm_live_fill_state
from .live_winner_session import LIVE_RUNS
from .strategy_lab import get_trader_winner_summary

ORDER_LEDGER = "live_winner_broker_orders"
EASTERN = ZoneInfo("America/New_York")
_ORDER_OPT_IN = "I_ACKNOWLEDGE_LIVE_REAL_MONEY_ORDERS"


def order_submission_opted_in() -> bool:
    """The deployment must explicitly acknowledge use of REAL money."""
    return str(os.getenv("ALPACA_LIVE_ORDER_SUBMISSION_CONFIRM") or "") == _ORDER_OPT_IN


def _number_env(name: str, default: float, *, floor: float, ceiling: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError as exc:
        raise RuntimeError(f"Invalid {name}.") from exc
    if not math.isfinite(value) or not floor <= value <= ceiling:
        raise RuntimeError(f"{name} outside permitted limits.")
    return value


def _unique_order_id(account: str, session: str, side: str, asset: str) -> str:
    digest = hashlib.sha256(f"{account}|{session}|{side}|{asset}".encode("utf-8")).hexdigest()[:20]
    return f"mct-live-{side}-{digest}"[:48]


def _execute_leg(
    db: Any, broker: Any, *, account: str, session: str,
    side: str, symbol: str, quantity: float | None, notional: float | None,
    winner_hash: str, settings: PaperTradingSettings,
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
            timeout_seconds=settings.order_fill_timeout_seconds,
            poll_interval_seconds=settings.order_poll_interval_seconds,
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


def execute_recalibrated_live_winner_session(db: Any, session: str) -> dict[str, Any]:
    """Execute once if explicitly enabled and broker state fully reconciled."""
    if not order_submission_opted_in():
        return {"status": "order_submission_disabled", "order_submission_enabled": False}
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
    if not bool(run.get("winner_signal_prepared")) or not run.get("target_asset"):
        raise RuntimeError("The current Winner has no verified Live trading signal.")

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
    expected_open = pd.Timestamp(run["expected_market_open"])
    current = current.tz_localize("UTC") if current.tzinfo is None else current.tz_convert("UTC")
    expected_open = expected_open.tz_localize("UTC") if expected_open.tzinfo is None else expected_open.tz_convert("UTC")
    if current.tz_convert(EASTERN).date().isoformat() != session:
        raise RuntimeError("Stale Live session.")
    settings = PaperTradingSettings.model_validate(get_paper_trading_settings(db))
    earliest = expected_open + pd.Timedelta(seconds=settings.market_open_delay_seconds)
    latest = expected_open + pd.Timedelta(seconds=settings.market_execution_window_seconds)
    if current < earliest:
        return {"status": "waiting_for_open_delay", "order_submission_enabled": False}
    if current > latest:
        raise RuntimeError("Live execution window has expired.")
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

    max_buy = _number_env("ALPACA_LIVE_MAX_ORDER_USD", 25.0, floor=1, ceiling=100000)
    max_exposure = _number_env("ALPACA_LIVE_MAX_EQUITY_FRACTION", 0.95, floor=0.01, ceiling=1.0)
    max_daily_loss = _number_env("ALPACA_LIVE_MAX_DAILY_LOSS_FRACTION", 0.05, floor=0.001, ceiling=0.25)
    baseline = float(run.get("session_equity_baseline") or 0)
    equity = float(getattr(account, "equity", 0) or 0)
    if baseline <= 0 or equity <= 0:
        raise RuntimeError("Live account equity baseline is invalid.")
    if equity < baseline * (1 - max_daily_loss) and action in {"buy", "rotate"}:
        raise RuntimeError("Daily-loss threshold blocks new Live exposure; risk-reducing exits remain permitted.")
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
            fill = _execute_leg(
                db, broker, account=account_id, session=session,
                side="sell", symbol=state["managed_symbol"],
                quantity=state["managed_quantity"], notional=None,
                winner_hash=run["winner_configuration_hash"], settings=settings,
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
            account = broker.get_account()
            equity = float(getattr(account, "equity", 0) or 0)
            cash = float(getattr(account, "cash", 0) or 0)
            buying_power = float(getattr(account, "buying_power", 0) or 0)
            notional = min(max_buy, equity * max_exposure,
                           max(0, cash - float(settings.cash_reserve_dollars)),
                           buying_power)
            notional = math.floor(notional * 100) / 100
            if notional < 1:
                raise RuntimeError("Live balance is insufficient for a guarded buy.")
            fill = _execute_leg(
                db, broker, account=account_id, session=session,
                side="buy", symbol=target, quantity=None, notional=notional,
                winner_hash=run["winner_configuration_hash"], settings=settings,
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
