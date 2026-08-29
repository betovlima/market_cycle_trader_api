from __future__ import annotations

import csv
import io
import json
import zipfile
from datetime import date, datetime, timezone
from typing import Any

from ..core.config import API_VERSION
from ..infrastructure.persistence.mongo_repository import (
    PAPER_PORTFOLIO_SNAPSHOTS_COLLECTION,
    PAPER_TRADE_ORDERS_COLLECTION,
    PAPER_TRADE_PLANS_COLLECTION,
    bson_value,
)
from .paper_portfolio import _public_decision_audit, _public_order, paper_portfolio_snapshot


def _json_safe(value: Any) -> Any:
    value = bson_value(value)
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat() if value.tzinfo else value.replace(tzinfo=timezone.utc).isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _csv_text(rows: list[dict[str, Any]], fieldnames: list[str]) -> str:
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    for row in rows:
        normalized = {}
        for key in fieldnames:
            value = _json_safe(row.get(key))
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
            normalized[key] = value
        writer.writerow(normalized)
    return buffer.getvalue()


def _all_candidates(plan: dict[str, Any]) -> list[dict[str, Any]]:
    utilities = plan.get('utilities') if isinstance(plan.get('utilities'), dict) else {}
    cash_edges = plan.get('cash_edges') if isinstance(plan.get('cash_edges'), dict) else {}
    current_asset = str(plan.get('current_asset') or '')
    target_asset = str(plan.get('target_asset') or '')
    raw_best_asset = str(plan.get('raw_best_asset') or '')
    rows: list[dict[str, Any]] = []
    for symbol, raw_utility in utilities.items():
        if str(symbol) == 'CASH':
            continue
        try:
            utility = float(raw_utility)
        except (TypeError, ValueError):
            continue
        cash_edge = None
        try:
            if cash_edges.get(symbol) is not None:
                cash_edge = float(cash_edges.get(symbol))
        except (TypeError, ValueError):
            cash_edge = None
        rows.append({
            'plan_id': str(plan.get('plan_id') or ''),
            'decision_date': plan.get('decision_date'),
            'execution_session': plan.get('execution_session'),
            'strategy_name': plan.get('winner_strategy_name'),
            'strategy_revision': plan.get('winner_strategy_revision'),
            'symbol': str(symbol),
            'utility': utility,
            'cash_edge': cash_edge,
            'is_current': str(symbol) == current_asset,
            'is_target': str(symbol) == target_asset,
            'is_raw_best': str(symbol) == raw_best_asset,
        })
    rows.sort(key=lambda item: (str(item.get('plan_id') or ''), -float(item.get('utility') or 0.0), str(item.get('symbol') or '')))
    for index, row in enumerate(rows):
        row['utility_rank'] = index + 1
    return rows


def build_paper_portfolio_export(db: Any) -> bytes:
    current = paper_portfolio_snapshot(db)
    orders = list(db[PAPER_TRADE_ORDERS_COLLECTION].find({}).sort('created_at', 1))
    plans = list(db[PAPER_TRADE_PLANS_COLLECTION].find({}).sort('created_at', 1))
    plan_map = {str(plan.get('plan_id')): plan for plan in plans if plan.get('plan_id')}
    history = list(db[PAPER_PORTFOLIO_SNAPSHOTS_COLLECTION].find({}, {'_id': 0}).sort('recorded_at', 1))

    transaction_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    structured_orders: list[dict[str, Any]] = []

    for plan in plans:
        audit = _public_decision_audit(plan)
        if audit:
            decision_rows.append({
                'plan_id': plan.get('plan_id'),
                'status': plan.get('status'),
                'decision_date': audit.get('decision_date'),
                'execution_session': audit.get('execution_session'),
                'strategy_id': audit.get('winner_strategy_id'),
                'strategy_name': audit.get('winner_strategy_name'),
                'strategy_revision': audit.get('winner_strategy_revision'),
                'configuration_hash': audit.get('winner_configuration_hash'),
                'current_asset': audit.get('current_asset'),
                'target_asset': audit.get('target_asset'),
                'raw_best_asset': audit.get('raw_best_asset'),
                'action': audit.get('action'),
                'selection_reason': audit.get('selection_reason'),
                'selected_utility': audit.get('selected_utility'),
                'current_utility': audit.get('current_utility'),
                'target_utility': audit.get('target_utility'),
                'target_vs_current_utility': audit.get('target_vs_current_utility'),
                'effective_switch_margin': audit.get('effective_switch_margin'),
                'calibrated_candidate_margin': audit.get('calibrated_candidate_margin'),
                'calibration_score': audit.get('calibration_score'),
                'training_end': audit.get('training_end'),
                'calibration_start': audit.get('calibration_start'),
                'calibration_end': audit.get('calibration_end'),
                'final_fit_end': audit.get('final_fit_end'),
                'stateful_intervention': audit.get('stateful_intervention'),
                'stateful_control_target_asset': audit.get('stateful_control_target_asset'),
                'stateful_risk_score': audit.get('stateful_risk_score'),
                'stateful_risk_threshold': audit.get('stateful_risk_threshold'),
                'stateful_confidence_margin': audit.get('stateful_confidence_margin'),
                'stateful_confidence_threshold': audit.get('stateful_confidence_threshold'),
                'decision_origin': audit.get('decision_origin'),
                'execution_origin': audit.get('execution_origin'),
                'execution_trigger': audit.get('execution_trigger'),
                'plan_source': audit.get('plan_source'),
                'manual_current_session_recovery': audit.get('manual_current_session_recovery'),
                'manual_execution_requested_at': audit.get('manual_execution_requested_at'),
            })
        candidate_rows.extend(_all_candidates(plan))

    for order in orders:
        plan_id = str(order.get('plan_id') or '')
        audit = _public_decision_audit(plan_map.get(plan_id)) if plan_id else None
        public_order = _public_order(order)
        structured_orders.append({**public_order, 'decision_audit': audit})
        transaction_rows.append({
            'created_at': order.get('created_at'),
            'submitted_at': order.get('submitted_at'),
            'filled_at': order.get('filled_at'),
            'symbol': order.get('symbol'),
            'side': order.get('side'),
            'status': order.get('status'),
            'quantity': order.get('quantity'),
            'filled_quantity': order.get('filled_quantity'),
            'filled_average_price': order.get('filled_average_price'),
            'notional': order.get('notional'),
            'client_order_id': order.get('client_order_id'),
            'plan_id': order.get('plan_id'),
            'decision_available': bool(audit),
            'strategy_name': audit.get('winner_strategy_name') if audit else None,
            'strategy_revision': audit.get('winner_strategy_revision') if audit else None,
            'decision_date': audit.get('decision_date') if audit else None,
            'execution_session': audit.get('execution_session') if audit else None,
            'current_asset': audit.get('current_asset') if audit else None,
            'target_asset': audit.get('target_asset') if audit else None,
            'raw_best_asset': audit.get('raw_best_asset') if audit else None,
            'selection_reason': audit.get('selection_reason') if audit else None,
            'current_utility': audit.get('current_utility') if audit else None,
            'target_utility': audit.get('target_utility') if audit else None,
            'target_vs_current_utility': audit.get('target_vs_current_utility') if audit else None,
            'effective_switch_margin': audit.get('effective_switch_margin') if audit else None,
            'calibrated_candidate_margin': audit.get('calibrated_candidate_margin') if audit else None,
            'calibration_score': audit.get('calibration_score') if audit else None,
            'stateful_intervention': audit.get('stateful_intervention') if audit else None,
            'decision_origin': audit.get('decision_origin') if audit else None,
            'execution_origin': audit.get('execution_origin') if audit else None,
            'execution_trigger': audit.get('execution_trigger') if audit else None,
            'plan_source': audit.get('plan_source') if audit else None,
            'manual_current_session_recovery': audit.get('manual_current_session_recovery') if audit else None,
        })

    current_summary = {key: current.get(key) for key in (
        'status', 'recorded_at', 'initial_capital', 'strategy_cash', 'market_value',
        'portfolio_value', 'realized_pnl', 'unrealized_pnl', 'total_pnl', 'total_return',
        'position', 'last_decision_date', 'last_execution_session', 'market_clock',
    )}
    manifest = {
        'schema_version': 1,
        'package_kind': 'market_cycle_trader_paper_transaction_audit',
        'api_version': API_VERSION,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'order_count': len(orders),
        'decision_plan_count': len(decision_rows),
        'candidate_row_count': len(candidate_rows),
        'portfolio_snapshot_count': len(history),
        'decision_linked_order_count': sum(1 for row in transaction_rows if row['decision_available']),
    }
    package = {
        'manifest': manifest,
        'current_portfolio': _json_safe(current_summary),
        'orders': _json_safe(structured_orders),
        'decisions': _json_safe(decision_rows),
        'decision_candidates': _json_safe(candidate_rows),
        'portfolio_history': _json_safe(history),
    }

    transaction_fields = [
        'created_at', 'submitted_at', 'filled_at', 'symbol', 'side', 'status', 'quantity',
        'filled_quantity', 'filled_average_price', 'notional', 'client_order_id', 'plan_id',
        'decision_available', 'strategy_name', 'strategy_revision', 'decision_date',
        'execution_session', 'current_asset', 'target_asset', 'raw_best_asset',
        'selection_reason', 'current_utility', 'target_utility', 'target_vs_current_utility',
        'effective_switch_margin', 'calibrated_candidate_margin', 'calibration_score',
        'stateful_intervention', 'decision_origin', 'execution_origin', 'execution_trigger',
        'plan_source', 'manual_current_session_recovery',
    ]
    decision_fields = [
        'plan_id', 'status', 'decision_date', 'execution_session', 'strategy_id', 'strategy_name',
        'strategy_revision', 'configuration_hash', 'current_asset', 'target_asset', 'raw_best_asset',
        'action', 'selection_reason', 'selected_utility', 'current_utility', 'target_utility',
        'target_vs_current_utility', 'effective_switch_margin', 'calibrated_candidate_margin',
        'calibration_score', 'training_end', 'calibration_start', 'calibration_end', 'final_fit_end',
        'stateful_intervention', 'stateful_control_target_asset', 'stateful_risk_score',
        'stateful_risk_threshold', 'stateful_confidence_margin', 'stateful_confidence_threshold',
        'decision_origin', 'execution_origin', 'execution_trigger', 'plan_source',
        'manual_current_session_recovery', 'manual_execution_requested_at',
    ]
    candidate_fields = [
        'plan_id', 'decision_date', 'execution_session', 'strategy_name', 'strategy_revision',
        'utility_rank', 'symbol', 'utility', 'cash_edge', 'is_current', 'is_target', 'is_raw_best',
    ]
    history_fields = [
        'recorded_at', 'portfolio_value', 'strategy_cash', 'market_value', 'total_pnl',
        'total_return', 'managed_symbol',
    ]

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('manifest.json', json.dumps(_json_safe(manifest), ensure_ascii=False, indent=2))
        archive.writestr('paper_transaction_audit.json', json.dumps(package, ensure_ascii=False, indent=2))
        archive.writestr('paper_transactions.csv', _csv_text(transaction_rows, transaction_fields))
        archive.writestr('paper_decisions.csv', _csv_text(decision_rows, decision_fields))
        archive.writestr('paper_decision_candidates.csv', _csv_text(candidate_rows, candidate_fields))
        archive.writestr('portfolio_history.csv', _csv_text(history, history_fields))
    return buffer.getvalue()
