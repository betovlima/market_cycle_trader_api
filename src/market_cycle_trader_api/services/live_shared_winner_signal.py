"""Canonical daily Winner target shared by all Alpaca OAuth accounts.

The existing Live Winner is the single signal producer. Reuse its immutable,
persisted D-1 target rather than recalculating models with each user's position.
The legacy account remains the canonical signal source, not a source of funds
or positions for other users. This module never submits broker orders.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import exchange_calendars as xcals
import pandas as pd

from .live_winner_session import LIVE_RUNS
from .strategy_lab import get_trader_winner_summary

_CANONICAL_STATUSES = frozenset({
    "recalibrated_no_orders", "executing", "completed",
})
_SUPPORTED_ACTIONS = frozenset({
    "hold", "stay_in_cash", "buy", "sell_to_cash", "rotate",
})


@dataclass(frozen=True)
class SharedWinnerSignal:
    execution_session: str
    decision_date: str
    target_asset: str
    winner_strategy_id: str
    winner_revision: int
    winner_configuration_hash: str
    snapshot_sha256: str
    source_account_id: str


def _canonical_account_id() -> str:
    """The account that ALREADY produces the Winner's live daily decision.

    Never take a source account ID from a client query parameter.
    """
    return str(
        os.getenv("ALPACA_LIVE_ACCOUNT_ID")
        or os.getenv("ALPACA_OAUTH_EXPECTED_ACCOUNT_ID")
        or ""
    ).strip()


def read_shared_winner_signal(
    db: Any, *, session: str,
) -> SharedWinnerSignal | None:
    """Get the one persisted target for this XNYS session, or None if pending.

    A new OAuth account never recalculates or overwrites a Winner decision.
    Historical decisions cannot be converted into the current day's target.
    """
    session_date = pd.Timestamp(session)
    calendar = xcals.get_calendar("XNYS")
    if session_date.tzinfo is not None or not calendar.is_session(session_date):
        raise ValueError("The requested Winner session must be an XNYS trading day.")
    source_account_id = _canonical_account_id()
    if not source_account_id:
        # Consent and account verification can still work while no canonical
        # signal source is configured in the local development environment.
        return None
    run = db[LIVE_RUNS].find_one({
        "_id": f"{source_account_id}:{session_date.date().isoformat()}",
    })
    if not run:
        return None
    # No interpretation of transient/incomplete, stale or failed sessions.
    if run.get("status") not in _CANONICAL_STATUSES:
        return None
    expected_day = calendar.previous_session(session_date).date().isoformat()
    if (run.get("live_account_id") != source_account_id
        or run.get("execution_session") != session_date.date().isoformat()
        or str(run.get("decision_date") or "")[:10] != expected_day
        or run.get("historical_cutoff") not in (None, expected_day)
        or run.get("analysis_mode") not in (None, "previous_completed_session_daily")
        or run.get("winner_signal_prepared") is not True
        or run.get("post_close_signal_ready") is not True
        or not run.get("snapshot_sha256")
        or run.get("action") not in _SUPPORTED_ACTIONS
        or not run.get("target_asset")):
        raise RuntimeError("The canonical Winner D-1 signal is inconsistent.")
    winner = get_trader_winner_summary(db)
    if (not isinstance(winner, dict)
        or str(winner.get("id") or "") != str(run.get("winner_strategy_id") or "")
        or str(winner.get("configuration_hash") or "") !=
            str(run.get("winner_configuration_hash") or "")
        or int(winner.get("revision") or 0) != int(run.get("winner_revision") or 0)):
        raise RuntimeError("The shared Winner changed after the D-1 decision.")
    target = str(run["target_asset"]).strip().upper()
    if not target or any(not (c.isalnum() or c in ".-") for c in target):
        raise RuntimeError("The persisted Winner target is invalid.")
    return SharedWinnerSignal(
        execution_session=session_date.date().isoformat(),
        decision_date=expected_day,
        target_asset=target,
        winner_strategy_id=str(run["winner_strategy_id"]),
        winner_revision=int(run["winner_revision"]),
        winner_configuration_hash=str(run["winner_configuration_hash"]),
        snapshot_sha256=str(run["snapshot_sha256"]),
        source_account_id=source_account_id,
    )
