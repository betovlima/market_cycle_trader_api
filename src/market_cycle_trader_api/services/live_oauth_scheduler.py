"""Fan out the one persisted Winner target to separately authorized Alpaca accounts.

No secondary model training or user cash ledgers. Existing Live control and
order idempotency remain the sole operating rules. Individual account errors
must not stop another user's execution or the legacy Winner account.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ..api.routers.alpaca_oauth import CONNECTIONS
from .live_control import live_control_status
from .live_oauth_execution import execute_shared_winner_for_oauth_account
from .live_shared_winner_signal import read_shared_winner_signal


def execute_oauth_accounts_for_shared_winner(db: Any, *, session: str) -> dict[str, Any]:
    if live_control_status(db)["control_mode"] not in {"active", "exit_only"}:
        return {"status": "live_stopped", "processed": 0}
    signal = read_shared_winner_signal(db, session=session)
    if signal is None:
        return {"status": "shared_signal_pending", "processed": 0}

    processed = 0
    completed = 0
    failed = 0
    # Only the users who explicitly granted trading scope. No access tokens
    # are loaded in the batch query; each trade independently resolves its
    # owner's token and verifies the actual Alpaca account.
    grants = db[CONNECTIONS].find(
        {"environment": "live", "trading_authorized": True,
         "oauth_scope": "trading", "read_only": False},
        {"subject": 1, "role": 1, "account_id": 1},
    )
    for grant in grants:
        if not isinstance(grant, dict) or not grant.get("subject"):
            continue
        if str(grant.get("account_id") or "") == signal.source_account_id:
            # The canonical account already runs with the legacy API keys.
            # Never execute it again via OAuth during this migration.
            continue
        identity = SimpleNamespace(
            subject=str(grant["subject"]),
            role=str(grant.get("role") or "trader"),
        )
        processed += 1
        try:
            result = execute_shared_winner_for_oauth_account(
                db, identity, session=session,
            )
            if result.get("status") == "completed":
                completed += 1
        except Exception:
            # Preserve the existing account's exception isolation. The order
            # ledger holds per-account reconciliation receipts, not money.
            failed += 1

    return {
        "status": "processed",
        "processed": processed,
        "completed": completed,
        "failed": failed,
    }
