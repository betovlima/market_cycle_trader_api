"""Inspect one shared Winner strategy and an individually authorized Alpaca account.

The shared Winner strategy is NOT reproduced per user. Account positions and
balances belong to Alpaca. This module neither calculates a new daily signal
nor submits broker orders; the legacy Live executor is unchanged.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from ..api.routers.alpaca_oauth import connected_trading_client
from .strategy_lab import get_trader_winner_summary

EASTERN = ZoneInfo("America/New_York")


def oauth_winner_readiness(db: Any, identity: Any) -> dict[str, Any]:
    """Read OAuth account identity and the global Winner, without any writes."""
    # Verify the trading grant before contacting Alpaca. If it has not been
    # issued, even a valid read-only OAuth token must NOT be treated as trade access.
    from ..api.routers.alpaca_oauth import CONNECTIONS, _key
    saved = db[CONNECTIONS].find_one({"_id": _key(identity.subject)})
    common = {
        "environment": "live",
        "execution_source": "oauth",
        "order_submission_enabled": False,
        "oauth_winner_execution_enabled": False,
        "trading_authorized": False,
        "account_id_suffix": (
            str(saved.get("account_id") or "")[-6:] if isinstance(saved, dict) else None
        ),
    }
    if not isinstance(saved, dict):
        return {**common, "status": "alpaca_not_connected"}
    if not (saved.get("subject") == identity.subject
            and saved.get("environment") == "live"
            and saved.get("trading_authorized") is True
            and saved.get("oauth_scope") == "trading"
            and saved.get("read_only") is False):
        return {**common, "status": "trading_consent_required"}

    broker = connected_trading_client(db, identity)
    account = broker.get_account()
    clock = broker.get_clock()
    account_id = str(getattr(account, "id", "") or "")
    stamp = getattr(clock, "timestamp", None)
    next_open = getattr(clock, "next_open", None)
    if not account_id or not isinstance(stamp, datetime) or stamp.tzinfo is None:
        raise RuntimeError("Alpaca OAuth Winner preflight received an invalid account/clock.")
    if bool(getattr(clock, "is_open", False)):
        session = stamp.astimezone(EASTERN).date().isoformat()
    elif isinstance(next_open, datetime) and next_open.tzinfo is not None:
        session = next_open.astimezone(EASTERN).date().isoformat()
    else:
        session = None

    # There is ONE Winner strategy. Do not infer a shared daily signal from
    # an account-specific legacy session, and do not build per-user winners.
    # The current D-1 calculation still depends on current_asset and
    # holding_sessions; extracting a truly shared signal is a separate change.
    winner = get_trader_winner_summary(db)
    winner_configured = bool(
        isinstance(winner, dict)
        and winner.get("id")
        and winner.get("configuration_hash")
        and winner.get("name")
    )
    return {
        **common,
        "status": "trading_authorized_execution_disabled",
        "trading_authorized": True,
        "account_id_suffix": account_id[-6:],
        "account_status": str(getattr(account, "status", "") or ""),
        "market_open": bool(getattr(clock, "is_open", False)),
        "execution_session": session,
        "winner_name": winner.get("name") if winner_configured else None,
        "winner_strategy_id": str(winner["id"]) if winner_configured else None,
        "shared_winner": True,
        "winner_configured": winner_configured,
        # No cross-account daily signal is persisted by this experimental path.
        "shared_signal_prepared": False,
        "winner_signal_prepared": False,
        "order_submission_enabled": False,
        "reason": (
            "One shared Winner is configured. A shared D-1 signal and individual "
            "Alpaca executions have not been connected to OAuth yet."
        ),
    }
