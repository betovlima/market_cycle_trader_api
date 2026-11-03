"""Read-only feasibility checks for an OAuth-authorized Winner account.

This is NOT a scheduler, optimizer, trading endpoint or broker order executor.
The original Live execution pathway remains keyed to the existing API keys.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from ..api.routers.alpaca_oauth import connected_trading_client
from .live_winner_session import LIVE_RUNS
from .strategy_lab import get_trader_winner_summary

EASTERN = ZoneInfo("America/New_York")


def oauth_winner_readiness(db: Any, identity: Any) -> dict[str, Any]:
    """Check the OAuth account and its existing Winner plan without any writes."""
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

    # Only account-qualified session IDs. Global 'execution_session' lookups
    # are insufficient for a multi-tenant Winner execution architecture.
    run = db[LIVE_RUNS].find_one({"_id": f"{account_id}:{session}"}) if session else None
    winner = get_trader_winner_summary(db)
    winner_matches = bool(
        isinstance(winner, dict) and isinstance(run, dict)
        and str(winner.get("id") or "") == str(run.get("winner_strategy_id") or "")
        and str(winner.get("configuration_hash") or "") ==
            str(run.get("winner_configuration_hash") or "")
        and int(winner.get("revision") or 0) == int(run.get("winner_revision") or 0)
    )
    signal_prepared = bool(
        winner_matches and run.get("status") == "recalibrated_no_orders"
        and run.get("winner_signal_prepared") is True
        and run.get("post_close_signal_ready") is True
        and run.get("snapshot_sha256")
        and run.get("target_asset")
    )
    return {
        **common,
        "status": "trading_authorized_execution_disabled",
        "trading_authorized": True,
        "account_id_suffix": account_id[-6:],
        "account_status": str(getattr(account, "status", "") or ""),
        "market_open": bool(getattr(clock, "is_open", False)),
        "execution_session": session,
        "winner_name": winner.get("name") if isinstance(winner, dict) else None,
        "winner_identity_matches": winner_matches,
        "winner_signal_prepared": signal_prepared,
        "order_submission_enabled": False,
        "reason": "User-scoped scheduler, per-account execution controls and order isolation are not enabled.",
    }
