"""Live portfolio from Live Alpaca only; no access to Paper transaction state."""
from __future__ import annotations

import io
import json
import math
import zipfile
from datetime import date, datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from typing import Any

from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest

from ..infrastructure.trading.alpaca_live import create_live_readonly_client
from .strategy_lab import get_trader_winner_summary
from .live_control import live_control_status
from .live_winner_session import LIVE_RUNS, latest_live_session
from .live_order_execution import ORDER_LEDGER
from .live_winner_candidates import latest_live_winner_decision
from .live_trader_state import LIVE_STATE

LIVE_PORTFOLIO_SNAPSHOTS_COLLECTION = "live_portfolio_snapshots"


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _label(value: Any) -> str:
    return str(getattr(value, "value", value) or "").lower()


def _timestamp(value: Any) -> str | None:
    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _position(p: Any) -> dict[str, Any]:
    quantity = _float(getattr(p, "qty", 0))
    entry = _float(getattr(p, "avg_entry_price", 0))
    value = _float(getattr(p, "market_value", 0))
    return {
        "symbol": str(getattr(p, "symbol", "") or ""),
        "quantity": quantity,
        "average_entry_price": entry,
        "current_price": _float(getattr(p, "current_price", 0)),
        "market_value": value,
        "cost_basis": quantity * entry,
        "unrealized_pnl": _float(getattr(p, "unrealized_pl", 0)),
        "unrealized_return": (
            _float(getattr(p, "unrealized_plpc", 0))
            if quantity * entry else None
        ),
    }


def _order(order: Any, *, execution_origin: str = "live_broker_unclassified") -> dict[str, Any]:
    quantity = _float(getattr(order, "qty", 0))
    filled = _float(getattr(order, "filled_qty", 0))
    avg = getattr(order, "filled_avg_price", None)
    return {
        "client_order_id": str(getattr(order, "client_order_id", "") or ""),
        "symbol": str(getattr(order, "symbol", "") or ""),
        "side": _label(getattr(order, "side", None)),
        "status": _label(getattr(order, "status", None)),
        "quantity": quantity,
        "filled_quantity": filled,
        "filled_average_price": _float(avg) if avg is not None else None,
        "filled_value": filled * _float(avg) if avg is not None else None,
        "submitted_at": _timestamp(getattr(order, "submitted_at", None)),
        "created_at": _timestamp(getattr(order, "created_at", None)),
        "filled_at": _timestamp(getattr(order, "filled_at", None)),
        "economic_fill": filled > 0,
        "execution_origin": execution_origin,
        "realized_pnl": None,  # Not derivable from a single fill.
        "decision_audit": None,  # Cannot infer a model decision from one broker fill.
    }


def _verified_order_origins(db: Any, account_id: str, broker_orders: list[Any]) -> dict[str, str]:
    """Attribute orders only with a persisted Live receipt or internal ledger.

    Never infer 'manual' solely from an unfamiliar Alpaca client_order_id.
    Historical orders without an exact receipt stay unclassified.
    """
    broker_ids = [str(getattr(o, "id", "") or "") for o in broker_orders]
    client_ids = [str(getattr(o, "client_order_id", "") or "") for o in broker_orders]
    broker_ids = [value for value in broker_ids if value]
    client_ids = [value for value in client_ids if value.startswith("mct-live-")]
    origin: dict[str, str] = {}
    if broker_ids:
        receipts = db[LIVE_RUNS].find(
            {"live_account_id": account_id, "status": "manual_reconciled",
             "manual_reconciliation.origin": "manual_alpaca_fills",
             "manual_reconciliation.order_ids": {"$in": broker_ids}},
            {"manual_reconciliation.order_ids": 1},
        )
        for receipt in receipts:
            for order_id in (receipt.get("manual_reconciliation") or {}).get("order_ids") or []:
                key = str(order_id)
                if key in broker_ids:
                    origin[key] = "manual_alpaca_fills"
    if client_ids:
        records = db[ORDER_LEDGER].find(
            {"broker_account_id": account_id, "_id": {"$in": client_ids}},
            {"_id": 1},
        )
        verified_client_ids = {str(record.get("_id") or "") for record in records}
        for order in broker_orders:
            client_id = str(getattr(order, "client_order_id", "") or "")
            broker_id = str(getattr(order, "id", "") or "")
            if broker_id and client_id in verified_client_ids:
                if broker_id in origin:
                    # Conflicting manual receipt and robot ledger: do not
                    # claim either provenance without human review.
                    origin[broker_id] = "live_broker_unclassified"
                else:
                    origin[broker_id] = "live_automated_verified"
    return origin


def _store_observation(
    db: Any, equity: float, cash: float, account_id: str, *, recorded_now: datetime
) -> list[dict[str, Any]]:
    collection = db[LIVE_PORTFOLIO_SNAPSHOTS_COLLECTION]
    now = recorded_now
    latest = collection.find_one({"account_id": account_id}, sort=[("recorded_at", -1)])
    recorded_at = latest.get("recorded_at") if latest else None
    if recorded_at is None or (
        now - (recorded_at.replace(tzinfo=timezone.utc) if recorded_at.tzinfo is None else recorded_at)
    ) >= timedelta(seconds=60):
        collection.insert_one({
            "recorded_at": now, "account_id": account_id, "portfolio_value": equity,
            "strategy_cash": cash, "market": "live",
        })
    documents = list(collection.find(
        {"account_id": account_id}, {"_id": 0, "recorded_at": 1, "portfolio_value": 1, "strategy_cash": 1},
    ).sort("recorded_at", -1).limit(500))
    documents.reverse()
    return [{
        "recorded_at": _timestamp(item["recorded_at"]),
        "portfolio_value": _float(item["portfolio_value"]),
        "strategy_cash": _float(item.get("strategy_cash")),
        # Drawdown is deliberately omitted: deposits/withdrawals distort equity series.
        "peak_portfolio_value": None,
    } for item in documents]



def _last_verified_live_execution(
    db: Any, *, account_id: str, positions: list[dict[str, Any]],
    recorded_now: datetime,
) -> dict[str, str | None]:
    """Read-only position-change metadata; never infer a trade from HOLD.

    Show the date only when the persisted Live state still matches the
    independently read broker position. A manual source also requires the
    reconciliation origin and session to match this last execution.
    """
    unavailable = {"session": None, "origin": None}
    state = db[LIVE_STATE].find_one(
        {"_id": account_id},
        {"managed_symbol": 1, "managed_quantity": 1,
         "last_execution_session": 1, "last_reconciliation_origin": 1,
         "last_reconciliation_session": 1},
    )
    if not isinstance(state, dict):
        return unavailable
    raw_session = state.get("last_execution_session")
    if not isinstance(raw_session, str):
        return unavailable
    try:
        session = date.fromisoformat(raw_session)
        if session.isoformat() != raw_session:
            return unavailable
        # The broker clock is the same trusted clock used for the snapshot.
        if session > recorded_now.astimezone(ZoneInfo("America/New_York")).date():
            return unavailable
    except (TypeError, ValueError):
        return unavailable

    if len(positions) > 1:
        return unavailable
    broker_symbol = str(positions[0]["symbol"] or "").upper() if positions else "CASH"
    broker_quantity = float(positions[0]["quantity"]) if positions else 0.0
    try:
        recorded_quantity = float(state.get("managed_quantity"))
    except (TypeError, ValueError):
        return unavailable
    if not math.isfinite(recorded_quantity) or not math.isfinite(broker_quantity):
        return unavailable
    if (str(state.get("managed_symbol") or "").upper() != broker_symbol
        or abs(recorded_quantity - broker_quantity) >
            max(1e-6, abs(broker_quantity) * 1e-6)):
        return unavailable

    origin = None
    if (state.get("last_reconciliation_session") == raw_session
        and state.get("last_reconciliation_origin") == "manual_alpaca_fills"):
        origin = "manual_alpaca_fills"
    # An automated origin is never guessed from a stale manual receipt.
    return {"session": raw_session, "origin": origin}


def live_portfolio_snapshot(db: Any) -> dict[str, Any]:
    client = create_live_readonly_client()
    account = client.get_account()
    account_id = str(getattr(account, 'id', '') or '')
    if not account_id:
        raise RuntimeError('Live broker account has no verifiable account id.')
    positions = [_position(p) for p in client.get_all_positions()]
    broker_orders = list(client.get_orders(filter=GetOrdersRequest(
        status=QueryOrderStatus.ALL, limit=100,
    )))
    # Match exact persisted manual reconciliation receipts or MCT intents.
    # If the evidence is absent, do not guess who submitted a broker order.
    order_origins = _verified_order_origins(db, account_id, broker_orders)
    orders = [
        _order(order, execution_origin=order_origins.get(
            str(getattr(order, "id", "") or ""), "live_broker_unclassified",
        ))
        for order in broker_orders
    ]
    cash = _float(getattr(account, "cash", 0))
    equity = _float(getattr(account, "equity", 0))
    buying_power_raw = getattr(account, "buying_power", None)
    non_marginable_raw = getattr(account, "non_marginable_buying_power", None)
    clock = client.get_clock()
    if getattr(clock, "timestamp", None) is None or clock.timestamp.tzinfo is None:
        raise RuntimeError("Live broker clock timestamp must include a timezone.")
    recorded_now = clock.timestamp.astimezone(timezone.utc)
    history = _store_observation(db, equity, cash, account_id, recorded_now=recorded_now)
    latest_decision = latest_live_winner_decision(db, account_id=account_id)
    last_execution = _last_verified_live_execution(
        db, account_id=account_id, positions=positions, recorded_now=recorded_now,
    )
    unrealized = sum(p["unrealized_pnl"] for p in positions)

    return {
        "status": "ready",
        "environment": "live",
        "recorded_at": recorded_now.isoformat(),
        "account_status": _label(getattr(account, "status", None)),
        "trading_blocked": bool(getattr(account, "trading_blocked", False)),
        "initial_capital": None,
        "strategy_cash": cash,
        "buying_power": _float(buying_power_raw) if buying_power_raw is not None else None,
        "non_marginable_buying_power": _float(non_marginable_raw) if non_marginable_raw is not None else None,
        "market_value": sum(p["market_value"] for p in positions),
        "portfolio_value": equity,
        "realized_pnl": None,
        "unrealized_pnl": unrealized,
        "total_pnl": None,
        "total_return": None,
        "position": positions[0] if len(positions) == 1 else None,
        "positions": positions,
        "last_decision_date": latest_decision.get("decision_date") if latest_decision else None,
        "last_execution_session": last_execution["session"],
        "last_execution_origin": last_execution["origin"],
        "market_clock": {
            "timestamp": _timestamp(getattr(clock, "timestamp", None)),
            "is_open": bool(getattr(clock, "is_open", False)),
            "next_open": _timestamp(getattr(clock, "next_open", None)),
            "next_close": _timestamp(getattr(clock, "next_close", None)),
        },
        "history": history,
        "recent_orders": orders,
        "audit": {
            "summary": {
                "distance_to_peak": None,
                "peak_portfolio_value": None,
                "maximum_drawdown": None,
            },
            "latest_decision": latest_decision,
            "position_excursion": None,
            "history": history,
            "pnl_by_asset": [{
                "symbol": item["symbol"], "realized_pnl": None,
                "unrealized_pnl": item["unrealized_pnl"],
            } for item in positions],
            "operations": orders,
        },
    }


def live_robot_status(db: Any) -> dict[str, Any]:
    """Independent Live Winner cycle and current real-order authorization."""
    from .live_observation_scheduler import live_observation_status
    observation = live_observation_status()
    run = latest_live_session(db)
    control = live_control_status(db)
    execution_active = control["control_mode"] in {"active", "exit_only"}
    return {
        **control,
        "mode": "live_operational" if execution_active else "live_inactive",
        "real_orders_armed": execution_active,
        "execution_enabled": execution_active,
        "order_submission_enabled": execution_active,
        "phase": run.get("phase") if run else control.get("phase"),
        "position_observer_alive": observation["scheduler_alive"],
        "scheduler_alive": observation["scheduler_alive"],
        "last_position_observation_at": observation["last_observed_at"],
        "observation_last_error": observation["last_error"],
        "next_execution_session": run.get("execution_session") if run else None,
        "active_run": run if run and run.get("status") not in {"completed", "cancelled"} else None,
        "latest_run": run,
        "trader_winner": get_trader_winner_summary(db),
    }


def build_live_portfolio_export(db: Any) -> bytes:
    snapshot = live_portfolio_snapshot(db)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "live_portfolio.json",
            json.dumps(snapshot, ensure_ascii=False, indent=2),
        )
        archive.writestr(
            "README.txt",
            "LIVE account only. Broker orders and equity observations; "
            "no Paper orders, inferred model decisions or deposit-adjusted P/L. "
            "No order submission performed.\n",
        )
    return output.getvalue()
