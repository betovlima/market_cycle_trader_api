"""Explicit admin reconciliation of a manually filled Live rotation.

Never submit, cancel or modify Alpaca orders. Fail closed on ambiguous fills,
concurrent changes, incomplete accounting or an active execution worker.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest
from pymongo import ReturnDocument

from ..infrastructure.trading.alpaca_live import create_live_readonly_client
from .live_control import live_control_status
from .live_trader_state import LIVE_STATE
from .live_winner_session import LIVE_RUNS
from .strategy_lab import get_trader_winner_summary

EASTERN = ZoneInfo("America/New_York")
LEDGER = "live_winner_broker_orders"
AUDIT = "live_manual_reconciliation_audit"
_TOLERANCE = 1e-6
_SAFE_MODES = frozenset({"paused", "stopped"})
_PLAN_STATUSES = frozenset({"recalibrated_no_orders", "review_required"})


class ManualReconciliationBlocked(RuntimeError):
    """The conditions for a reliable broker-backed adoption were not met."""


def _value(value: Any) -> str:
    return str(getattr(value, "value", value) or "").strip().lower().split(".")[-1]


def _quantity(value: Any) -> float:
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise ManualReconciliationBlocked("Quantidade de ordem invalida.") from exc
    if not math.isfinite(number) or number <= 0:
        raise ManualReconciliationBlocked("Quantidade de ordem nao positiva.")
    return number


def _session_of(value: Any) -> str:
    if not value:
        raise ManualReconciliationBlocked("Ordem sem horario confirmado de preenchimento.")
    try:
        date = datetime.fromisoformat(str(value).replace("Z", "+00:00")) if isinstance(value, str) else value
        if date.tzinfo is None:
            raise ValueError("naive")
        return date.astimezone(EASTERN).date().isoformat()
    except (TypeError, ValueError, AttributeError) as exc:
        raise ManualReconciliationBlocked("Horario de preenchimento da ordem e invalido.") from exc


def _broker_evidence(broker: Any, account_id: str, *, session: str,
                     source_symbol: str, target_symbol: str,
                     source_quantity: float) -> dict[str, Any]:
    account = broker.get_account()
    if str(getattr(account, "id", "") or "") != account_id:
        raise ManualReconciliationBlocked("A identidade da conta Alpaca mudou.")
    if bool(getattr(account, "account_blocked", False)) or bool(getattr(account, "trading_blocked", False)):
        raise ManualReconciliationBlocked("A conta Live possui restricao de negociacao.")
    positions = list(broker.get_all_positions())
    if len(positions) != 1:
        raise ManualReconciliationBlocked("A conta deve ter exatamente uma posicao Live.")
    position = positions[0]
    symbol = str(getattr(position, "symbol", "") or "").upper()
    if _value(getattr(position, "side", "long")) != "long":
        raise ManualReconciliationBlocked("Posicao vendida a descoberto nao pode ser adotada.")
    amount = _quantity(getattr(position, "qty", None))
    if symbol != target_symbol:
        raise ManualReconciliationBlocked("A posicao Alpaca nao corresponde ao alvo aprovado.")
    open_orders = broker.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=100))
    if open_orders:
        raise ManualReconciliationBlocked("Existem ordens abertas na Alpaca; reconciliar primeiro.")
    all_orders = list(broker.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.ALL, limit=100)))
    if len(all_orders) >= 100:
        raise ManualReconciliationBlocked("Historico Alpaca excede 100 ordens; revisao completa necessaria.")
    current_orders = []
    for order in all_orders:
        date = getattr(order, "filled_at", None)
        if not date:
            # A pending, rejected or canceled order for this session is
            # conservatively blocked when its submission date is known.
            submitted = getattr(order, "submitted_at", None)
            if submitted and _session_of(submitted) == session:
                raise ManualReconciliationBlocked("Ha ordem nao executada no dia. Revisao necessaria.")
            continue
        if _session_of(date) != session:
            continue
        if _value(getattr(order, "status", None)) != "filled":
            raise ManualReconciliationBlocked("Ordem do dia nao esta integralmente preenchida.")
        order_id = str(getattr(order, "id", "") or "")
        client_id = str(getattr(order, "client_order_id", "") or "")
        if not order_id or not client_id:
            raise ManualReconciliationBlocked("Ordem sem identidade auditavel na Alpaca.")
        if client_id.startswith("mct-live-"):
            raise ManualReconciliationBlocked("Ordem do executor MCT nao pode ser adotada como manual.")
        side = _value(getattr(order, "side", None))
        asset = str(getattr(order, "symbol", "") or "").upper()
        if (side, asset) not in {("sell", source_symbol), ("buy", target_symbol)}:
            raise ManualReconciliationBlocked("Ha outra negociacao Live no mesmo pregao.")
        completed = datetime.fromisoformat(str(date).replace("Z", "+00:00")) if isinstance(date, str) else date
        current_orders.append({
            "filled_at": completed.astimezone(timezone.utc).isoformat(),
            "id": order_id, "client_order_id": client_id,
            "side": side, "symbol": asset,
            "quantity": _quantity(getattr(order, "filled_qty", None)),
        })
    sales = [o for o in current_orders if o["side"] == "sell"]
    buys = [o for o in current_orders if o["side"] == "buy"]
    if not sales or not buys:
        raise ManualReconciliationBlocked("Faltam comprovantes da venda e da compra manual.")
    if len({o["id"] for o in current_orders}) != len(current_orders):
        raise ManualReconciliationBlocked("Ha identificadores de ordens repetidos.")
    sold = sum(o["quantity"] for o in sales)
    bought = sum(o["quantity"] for o in buys)
    if abs(sold - source_quantity) > _TOLERANCE:
        raise ManualReconciliationBlocked("Vendas manuais nao zeraram a posicao registrada.")
    if abs(bought - amount) > _TOLERANCE:
        raise ManualReconciliationBlocked("Compras manuais nao conferem com a posicao Alpaca.")
    if max(o["filled_at"] for o in sales) > min(o["filled_at"] for o in buys):
        raise ManualReconciliationBlocked("Compra antecedeu o encerramento das vendas manuais.")
    current_orders.sort(key=lambda order: order["id"])
    return {
        "symbol": symbol, "quantity": amount, "sold_quantity": sold,
        "bought_quantity": bought, "orders": current_orders,
    }


def _inspect(db: Any, *, execution_session: str | None = None) -> dict[str, Any]:
    broker = create_live_readonly_client()
    account_id = str(getattr(broker.get_account(), "id", "") or "")
    if not account_id:
        raise ManualReconciliationBlocked("Conta Alpaca Live indisponivel.")
    mode = live_control_status(db)["control_mode"]
    query = {"live_account_id": account_id}
    if execution_session is not None:
        query["execution_session"] = execution_session
    run = db[LIVE_RUNS].find_one(query, sort=[("execution_session", -1)])
    if not run:
        raise ManualReconciliationBlocked("Nenhuma sessao Winner Live da conta foi encontrada.")
    session = str(run.get("execution_session") or "")
    if run.get("status") == "manual_reconciled":
        state = db[LIVE_STATE].find_one({"_id": account_id})
        confirmation = run.get("manual_reconciliation") or {}
        positions = list(broker.get_all_positions())
        if (not state or len(positions) != 1
            or str(getattr(positions[0], "symbol", "") or "").upper() != confirmation.get("target_symbol")
            or str(state.get("managed_symbol") or "").upper() != confirmation.get("target_symbol")
            or abs(float(state.get("managed_quantity") or 0) - _quantity(getattr(positions[0], "qty", None))) > _TOLERANCE):
            raise ManualReconciliationBlocked("Sessao ja conciliada, mas a posicao atual divergiu. Revisao necessaria.")
        if broker.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=100)):
            raise ManualReconciliationBlocked("Existem ordens abertas na Alpaca; revisar sessao.")
        return {"status": "already_reconciled", "execution_session": session,
                "control_mode": mode, "message": "Sessao e posicao manual ja conciliadas."}
    if run.get("status") not in _PLAN_STATUSES:
        raise ManualReconciliationBlocked("Plano Live em estado inelegivel para conciliacao manual.")
    if run.get("action") != "rotate":
        raise ManualReconciliationBlocked("Somente rotacoes manuais ja aprovadas podem ser conciliadas.")
    source = str(run.get("current_asset") or "").upper()
    target = str(run.get("target_asset") or "").upper()
    if not source or not target or source == "CASH" or source == target:
        raise ManualReconciliationBlocked("Plano de rotacao incompleto.")
    if (str(run.get("decision_date") or "")[:10] != session
        or str(run.get("analysis_mode") or "") != "live_current_session_intraday"
        or not run.get("market_open_calibration_completed_at")):
        raise ManualReconciliationBlocked("O plano nao foi recalibrado para a sessao.")
    winner = get_trader_winner_summary(db)
    if (not isinstance(winner, dict)
        or str(winner.get("id")) != str(run.get("winner_strategy_id"))
        or str(winner.get("configuration_hash")) != str(run.get("winner_configuration_hash"))
        or int(winner.get("revision") or 0) != int(run.get("winner_revision") or 0)):
        raise ManualReconciliationBlocked("A Winner ativa difere da Winner do plano.")
    state = db[LIVE_STATE].find_one({"_id": account_id})
    if not state:
        raise ManualReconciliationBlocked("Estado interno nao inicializado; revisao manual necessaria.")
    registered = str(state.get("managed_symbol") or "").upper()
    registered_quantity = float(state.get("managed_quantity") or 0)
    planned_quantity = float(run.get("current_quantity") or 0)
    if registered != source or abs(registered_quantity - planned_quantity) > _TOLERANCE:
        raise ManualReconciliationBlocked("Posicao interna nao corresponde a origem do plano; revisao necessaria.")
    if db[LEDGER].count_documents({"broker_account_id": account_id, "execution_session": session}):
        raise ManualReconciliationBlocked("Existem intencoes do executor MCT na sessao. Revisar ledger.")
    evidence = _broker_evidence(
        broker, account_id, session=session, source_symbol=source,
        target_symbol=target, source_quantity=registered_quantity,
    )
    fields = {
        "account": account_id, "session": session,
        "control_mode": mode, "run_id": str(run["_id"]),
        "run_status": run["status"], "winner_hash": run["winner_configuration_hash"],
        "revision": int(state.get("revision") or 0),
        "source": source, "source_quantity": registered_quantity,
        "target": target, "evidence": evidence,
    }
    token = hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "status": "ready" if mode in _SAFE_MODES else "pause_required",
        "execution_session": session, "control_mode": mode,
        "source": source, "target": target,
        "registered_quantity": registered_quantity,
        "broker_quantity": evidence["quantity"],
        "sell_order_count": sum(1 for o in evidence["orders"] if o["side"] == "sell"),
        "buy_order_count": sum(1 for o in evidence["orders"] if o["side"] == "buy"),
        "order_ids": [o["id"] for o in evidence["orders"]],
        "requires_update": True, "confirmation_token": token,
        "_fields": fields, "_state": state, "_run": run,
        "_evidence": evidence,
    }


def manual_reconciliation_preview(db: Any, *, execution_session: str | None = None) -> dict[str, Any]:
    """Read-only broker inspection: no updates and no order submissions."""
    try:
        info = _inspect(db, execution_session=execution_session)
        return {key: value for key, value in info.items() if not key.startswith("_")}
    except ManualReconciliationBlocked as exc:
        return {"status": "blocked", "message": str(exc), "execution_session": execution_session}


def reconcile_manual_live_rotation(db: Any, *, execution_session: str,
                                   confirmation_token: str, actor: str) -> dict[str, Any]:
    """CAS handoff of a validated manual rotation; never touches broker orders."""
    info = _inspect(db, execution_session=execution_session)
    if info["status"] == "already_reconciled":
        return info
    if info["status"] != "ready":
        raise ManualReconciliationBlocked("Pause ou pare o Trader Live antes de reconciliar.")
    if info["confirmation_token"] != confirmation_token:
        raise ManualReconciliationBlocked("A evidencia mudou; revise novamente antes de confirmar.")
    state, run = info["_state"], info["_run"]
    session = info["execution_session"]
    account_id = str(run["live_account_id"])
    claimed = db[LIVE_RUNS].find_one_and_update(
        {"_id": run["_id"], "live_account_id": account_id,
         "status": run["status"], "action": "rotate",
         "winner_configuration_hash": run["winner_configuration_hash"],
         "current_asset": info["source"], "target_asset": info["target"]},
        {"$set": {"status": "manual_reconciliation_in_progress",
                  "phase": "manual_broker_reconciliation",
                  "order_submission_enabled": False,
                  "manual_reconciliation_started_at": datetime.now(timezone.utc)}},
        return_document=ReturnDocument.AFTER,
    )
    if claimed is None:
        raise ManualReconciliationBlocked("Sessao alterada durante a conciliacao.")
    try:
        if live_control_status(db)["control_mode"] not in _SAFE_MODES:
            raise ManualReconciliationBlocked("Trader Live reativado durante a conciliacao.")
        fresh_broker = create_live_readonly_client()
        fresh_evidence = _broker_evidence(
            fresh_broker, account_id, session=session,
            source_symbol=info["source"], target_symbol=info["target"],
            source_quantity=info["registered_quantity"],
        )
        if fresh_evidence != info["_evidence"]:
            raise ManualReconciliationBlocked("Posicao ou ordens Alpaca alteradas durante a conciliacao.")
        if db[LEDGER].count_documents({"broker_account_id": account_id, "execution_session": session}):
            raise ManualReconciliationBlocked("Nova intencao MCT detectada durante conciliacao.")
        changed = db[LIVE_STATE].find_one_and_update(
            {"_id": account_id, "revision": int(state.get("revision") or 0),
             "managed_symbol": info["source"],
             "managed_quantity": state["managed_quantity"]},
            {"$set": {"managed_symbol": info["target"],
                      "managed_quantity": float(fresh_evidence["quantity"]),
                      "entry_session": session, "last_execution_session": session,
                      "updated_at": datetime.now(timezone.utc),
                      "last_reconciliation_origin": "manual_alpaca_fills",
                      "last_reconciliation_session": session},
             "$inc": {"revision": 1}},
            return_document=ReturnDocument.AFTER,
        )
        if changed is None:
            raise ManualReconciliationBlocked("Estado Live mudou simultaneamente; revisao necessaria.")
        audit = {
            "broker_account_id": account_id, "execution_session": session,
            "origin": "manual_alpaca_fills", "actor": actor,
            "winner_configuration_hash": run["winner_configuration_hash"],
            "source_symbol": info["source"], "target_symbol": info["target"],
            "source_quantity": info["registered_quantity"],
            "target_quantity": fresh_evidence["quantity"],
            "order_ids": [o["id"] for o in fresh_evidence["orders"]],
            "created_at": datetime.now(timezone.utc),
        }
        completion = db[LIVE_RUNS].update_one(
            {"_id": run["_id"], "status": "manual_reconciliation_in_progress"},
            {"$set": {"status": "manual_reconciled", "phase": "manual_alpaca_fills_reconciled",
                      "order_submission_enabled": False, "execution_origin": "manual_alpaca_fills",
                      "manual_reconciliation": audit, "completed_at": datetime.now(timezone.utc)}},
        )
        if completion.matched_count != 1:
            raise ManualReconciliationBlocked("Falha ao arquivar o plano antigo; requer revisao.")
        # An immutable receipt in the run above is the authoritative audit;
        # a separate collection improves discoverability without blocking.
        try:
            db[AUDIT].insert_one(audit)
        except Exception:
            pass
        return {"status": "reconciled", "execution_session": session,
                "source": info["source"], "target": info["target"],
                "broker_quantity": fresh_evidence["quantity"],
                "order_count": len(fresh_evidence["orders"]),
                "order_submission_enabled": False,
                "message": "Rotacao manual conciliada sem ordens Alpaca."}
    except Exception:
        # Never make the old plan eligible again after acquiring the claim.
        db[LIVE_RUNS].update_one(
            {"_id": run["_id"], "status": "manual_reconciliation_in_progress"},
            {"$set": {"status": "review_required",
                      "phase": "manual_reconciliation_incomplete",
                      "order_submission_enabled": False}},
        )
        raise
