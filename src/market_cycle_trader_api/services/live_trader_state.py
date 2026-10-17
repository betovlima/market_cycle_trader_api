"""Verified independent Live trading state, backed by broker positions.

Never infer ownership of a manually opened holding. New accounts must be flat
to initialize. Every mismatch stops trading instead of borrowing Paper state.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pymongo import ReturnDocument

from ..infrastructure.trading.alpaca_live_orders import calendar_session_dates

LIVE_STATE = "live_trader_strategy_state"
_QTY_EPS = 1e-6


def verified_live_state(db: Any, broker: Any, *, session: str) -> dict[str, Any]:
    account = broker.get_account()
    account_id = str(getattr(account, "id", "") or "")
    if not account_id:
        raise RuntimeError("Cannot verify Live account identity.")
    positions = list(broker.get_all_positions())
    if len(positions) > 1:
        raise RuntimeError("Live Winner supports only one managed asset; multiple positions require review.")
    if any(str(getattr(p, "side", "long")).lower() not in {"long", "positionside.long"} for p in positions):
        raise RuntimeError("Live short positions require manual review.")
    broker_symbol = str(getattr(positions[0], "symbol", "") or "").upper() if positions else "CASH"
    broker_quantity = float(getattr(positions[0], "qty", 0)) if positions else 0.0
    if positions and (not broker_symbol or broker_quantity <= 0):
        raise RuntimeError("Live position quantity or symbol is invalid.")
    state = db[LIVE_STATE].find_one({"_id": account_id})
    if state is None:
        if broker_symbol != "CASH":
            raise RuntimeError("Unmanaged Live position exists. A controlled adoption/reconciliation is required.")
        now = datetime.now(timezone.utc)
        db[LIVE_STATE].update_one(
            {"_id": account_id},
            {"$setOnInsert": {
                "market": "live", "managed_symbol": "CASH", "managed_quantity": 0.0,
                "entry_session": None, "revision": 0, "initialized_at": now,
                "last_execution_session": None,
            }}, upsert=True,
        )
        state = db[LIVE_STATE].find_one({"_id": account_id})
    if (str(state.get("managed_symbol") or "").upper() != broker_symbol
        or abs(float(state.get("managed_quantity") or 0) - broker_quantity) > _QTY_EPS):
        raise RuntimeError("Live broker positions differ from the independently recorded Trader state.")
    entry = str(state.get("entry_session") or "")
    if broker_symbol == "CASH":
        sessions_held = 0
    else:
        if not entry:
            raise RuntimeError("Live holding has no auditable entry session.")
        sessions = calendar_session_dates(
            broker, start_date=entry, end_date=session,
        )
        sessions_held = max(1, len([day for day in sessions if entry <= day <= session]))
    return {
        "account_id": account_id,
        "managed_symbol": broker_symbol,
        "managed_quantity": broker_quantity,
        "entry_session": entry or None,
        "holding_sessions": sessions_held,
        "revision": int(state.get("revision") or 0),
        "last_execution_session": state.get("last_execution_session"),
    }


def confirm_live_fill_state(db: Any, *, previous: dict[str, Any], session: str,
                            symbol: str, quantity: float,
                            entry_session: str | None) -> None:
    """CAS write after a confirmed filled broker leg; never guess a fill."""
    new_symbol = str(symbol).upper()
    if (new_symbol == "CASH" and quantity != 0) or (new_symbol != "CASH" and quantity <= 0):
        raise ValueError("Invalid Live fill state.")
    updated = db[LIVE_STATE].find_one_and_update(
        {"_id": previous["account_id"], "revision": previous["revision"],
         "managed_symbol": previous["managed_symbol"]},
        {"$set": {
            "managed_symbol": new_symbol, "managed_quantity": float(quantity),
            "entry_session": entry_session, "last_execution_session": session,
            "updated_at": datetime.now(timezone.utc),
        }, "$inc": {"revision": 1}},
        return_document=ReturnDocument.AFTER,
    )
    if updated is None:
        raise RuntimeError("Live state was modified concurrently; broker reconciliation is required.")
