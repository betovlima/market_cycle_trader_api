"""Live-only immutable candidate audit derived from the exact Winner decision.

The Paper audit ranking is a pure formatting calculation and never reads Paper
collections, positions or orders. All persisted decision data comes from Live.
"""
from __future__ import annotations

import math
from typing import Any

from .paper_portfolio_audit import decision_candidates
from .live_winner_session import LIVE_RUNS


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def winner_candidate_fields(decision: Any) -> dict[str, Any]:
    utilities = {
        str(symbol).upper(): number
        for symbol, value in dict(decision.utilities).items()
        if (number := _finite(value)) is not None
    }
    if not utilities:
        raise RuntimeError("Winner returned no finite candidate utilities.")
    raw_best = str(decision.raw_best_asset or "").upper()
    target = str(decision.target_asset or "").upper()
    current = str(decision.current_asset or "CASH").upper()
    if raw_best not in utilities or target not in utilities or current not in utilities:
        raise RuntimeError("Winner utility vector omits a decision asset.")
    fields = {
        "utilities": utilities,
        "cash_edges": {
            str(symbol).upper(): number
            for symbol, value in dict(decision.cash_edges or {}).items()
            if (number := _finite(value)) is not None
        },
        "raw_best_asset": raw_best,
        "selected_utility": _finite(decision.selected_utility),
        "effective_switch_margin": _finite(decision.effective_switch_margin),
        "calibrated_candidate_margin": _finite(decision.calibrated_candidate_margin),
        "calibration_score": _finite(decision.calibration_score),
    }
    fields["candidate_count"] = len(decision_candidates({
        **fields, "current_asset": current, "target_asset": target,
    }))
    return fields


def latest_live_winner_decision(db: Any, *, account_id: str) -> dict[str, Any] | None:
    run = db[LIVE_RUNS].find_one(
        {"live_account_id": account_id, "winner_signal_prepared": True,
         "utilities": {"$type": "object"}},
        sort=[("execution_session", -1)],
    )
    if not run:
        return None
    document = {
        key: run.get(key)
        for key in (
            "execution_session", "decision_date", "current_asset", "target_asset",
            "raw_best_asset", "action", "utilities", "cash_edges",
            "effective_switch_margin", "selected_utility",
            "calibrated_candidate_margin", "calibration_score",
            "winner_configuration_hash", "winner_strategy_id", "winner_name",
            "status", "phase", "candidate_count",
        )
    }
    status = str(run.get("status") or "")
    action = str(run.get("action") or "")
    if status == "manual_reconciled" and (run.get("manual_reconciliation") or {}).get("origin") == "manual_alpaca_fills":
        execution_origin = "manual_alpaca_fills"
    elif status == "completed" and action in {"hold", "stay_in_cash"}:
        # A completed HOLD/CASH session means broker reconciliation succeeded,
        # NOT that the executor submitted an order.
        execution_origin = "live_no_orders"
    elif status == "completed":
        execution_origin = "live_broker"
    else:
        execution_origin = "live_not_executed"
    return {
        **{key: value for key, value in document.items() if value is not None},
        "plan_id": str(run.get("_id") or ""),
        "decision_origin": "live_winner_model",
        "execution_origin": execution_origin,
        "candidates": decision_candidates(document),
    }
