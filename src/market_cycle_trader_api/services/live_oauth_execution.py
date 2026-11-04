"""Per-account execution of the ONE canonical Winner daily target via OAuth.

Only the existing Live operating mode and original XNYS rules apply.
No new deployment flags, per-user Winner models or MCT cash balances.
This is deliberately NOT wired into the background scheduler or an HTTP
order endpoint until account-isolation regression tests are completed.
"""
from __future__ import annotations

from typing import Any
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd

from ..api.routers.alpaca_oauth import connected_trading_client
from ..infrastructure.trading.alpaca_live_orders import asset_snapshot
from .live_control import live_control_status
from .live_execution_window import live_execution_window_deadline
from .live_oauth_winner import account_action_for_shared_target
from .live_order_execution import (
    _assert_live_order_window,
    _execute_leg,
    _full_live_cash_notional,
)
from .live_shared_winner_signal import read_shared_winner_signal

EASTERN = ZoneInfo("America/New_York")


def _single_broker_position(broker: Any) -> tuple[str, float]:
    """Read the effective brokerage position, never an internal position cache."""
    positions = list(broker.get_all_positions())
    if not positions:
        return "CASH", 0.0
    if len(positions) != 1:
        raise RuntimeError("The existing single-asset Winner requires one Alpaca position.")
    position = positions[0]
    symbol = str(getattr(position, "symbol", "") or "").strip().upper()
    quantity = float(getattr(position, "qty", 0) or 0)
    side = str(getattr(getattr(position, "side", None), "value",
                       getattr(position, "side", "long"))).lower()
    if not symbol or quantity <= 0 or side not in {"long", "positionside.long"}:
        raise RuntimeError("The Alpaca position cannot be reconciled with the Winner.")
    return symbol, quantity


def execute_shared_winner_for_oauth_account(
    db: Any, identity: Any, *, session: str,
) -> dict[str, Any]:
    """Use the shared D-1 target and only this identity's Alpaca OAuth account.

    No new operational locks are introduced. Existing Live mode, XNYS
    session, broker and order-ledger reconciliation are reused.
    """
    mode = live_control_status(db)["control_mode"]
    if mode not in {"active", "exit_only"}:
        return {"status": "blocked_by_live_mode", "order_submission_enabled": False}

    broker = connected_trading_client(db, identity)  # Never API keys.
    account = broker.get_account()
    account_id = str(getattr(account, "id", "") or "")
    if not account_id:
        raise RuntimeError("The authorized Alpaca account ID is missing.")

    # Both the model identity and its final D-1 target belong to the one
    # canonical Winner, not to this account or its equity.
    shared = read_shared_winner_signal(db, session=session)
    if shared is None:
        return {"status": "shared_winner_signal_pending", "order_submission_enabled": False}

    if (bool(getattr(account, "account_blocked", True))
        or bool(getattr(account, "trading_blocked", True))
        or bool(getattr(account, "trade_suspended_by_user", False))
        or str(getattr(account, "status", "")).lower()
            not in {"active", "accountstatus.active"}):
        raise RuntimeError("The authorized Alpaca Live account cannot trade.")

    calendar = xcals.get_calendar("XNYS")
    session_date = pd.Timestamp(session)
    if not calendar.is_session(session_date):
        raise ValueError("The execution session is not an XNYS trading day.")
    expected_open = calendar.session_open(session_date)
    clock = broker.get_clock()
    deadline, _ = live_execution_window_deadline(
        expected_open, market_clock=clock,
    )
    _assert_live_order_window(broker, session, deadline)

    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest
    if broker.get_orders(filter=GetOrdersRequest(
        status=QueryOrderStatus.OPEN, limit=100,
    )):
        raise RuntimeError("Existing open Alpaca Live orders require reconciliation.")

    current_asset, current_quantity = _single_broker_position(broker)
    action = account_action_for_shared_target(
        target_asset=shared.target_asset, current_asset=current_asset,
    )
    if mode == "exit_only" and action not in {"hold", "stay_in_cash", "sell_to_cash"}:
        return {"status": "blocked_by_exit_only_mode", "order_submission_enabled": False}

    if action in {"hold", "stay_in_cash"}:
        return {
            "status": "completed", "action": action,
            "target_asset": shared.target_asset,
            "execution_session": session,
            "broker_account_suffix": account_id[-6:],
            "order_submission_enabled": False,
        }

    if action in {"buy", "rotate"}:
        asset = asset_snapshot(broker, shared.target_asset)
        if (not asset["tradable"] or not asset["fractionable"]
            or asset["status"] != "active"):
            raise RuntimeError("The shared Winner asset cannot trade fractionally.")

    order_sent = False
    if action in {"sell_to_cash", "rotate"}:
        if live_control_status(db)["control_mode"] not in {"active", "exit_only"}:
            raise RuntimeError("Live operating mode changed before sell.")
        refreshed_clock = broker.get_clock()
        refreshed_deadline, _ = live_execution_window_deadline(
            expected_open, market_clock=refreshed_clock,
        )
        _assert_live_order_window(broker, session, refreshed_deadline)
        _execute_leg(
            db, broker, account=account_id, session=session,
            side="sell", symbol=current_asset,
            quantity=current_quantity, notional=None,
            winner_hash=shared.winner_configuration_hash,
        )
        order_sent = True

    if action in {"buy", "rotate"}:
        if live_control_status(db)["control_mode"] != "active":
            raise RuntimeError("Live operating mode changed before buy.")
        refreshed_clock = broker.get_clock()
        refreshed_deadline, _ = live_execution_window_deadline(
            expected_open, market_clock=refreshed_clock,
        )
        _assert_live_order_window(broker, session, refreshed_deadline)
        if action == "rotate":
            # Alpaca must confirm that the prior holding was actually sold.
            remaining, _ = _single_broker_position(broker)
            if remaining != "CASH":
                raise RuntimeError("The Alpaca sell leg needs reconciliation.")
        notional = _full_live_cash_notional(broker.get_account())
        _execute_leg(
            db, broker, account=account_id, session=session,
            side="buy", symbol=shared.target_asset,
            quantity=None, notional=notional,
            winner_hash=shared.winner_configuration_hash,
        )
        order_sent = True

    return {
        "status": "completed",
        "action": action,
        "target_asset": shared.target_asset,
        "execution_session": session,
        "broker_account_suffix": account_id[-6:],
        "order_submission_enabled": order_sent,
    }
