from __future__ import annotations

import math
from collections import Counter
from typing import Any

import pandas as pd

from .config import ANALYSIS_VERSION, HORIZONS, PRIMARY_HORIZON, SCHEMA_VERSION


def _finite(value: Any) -> float | None:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return None
    return n if math.isfinite(n) else None


def _stamp(value: Any) -> pd.Timestamp | None:
    try:
        s = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    return s.tz_localize("UTC") if s.tzinfo is None else s.tz_convert("UTC")


def _series_by_symbol(market_rows: list[dict[str, Any]]) -> dict[str, pd.Series]:
    frame = pd.DataFrame(market_rows)
    if frame.empty:
        return {}
    frame["execution_date"] = pd.to_datetime(frame["execution_date"], utc=True, errors="coerce")
    frame["execution_open"] = pd.to_numeric(frame["execution_open"], errors="coerce")
    frame = frame.dropna(subset=["execution_date", "execution_open", "symbol"])
    result: dict[str, pd.Series] = {}
    for symbol, group in frame.groupby("symbol"):
        series = group.drop_duplicates("execution_date").sort_values("execution_date").set_index("execution_date")["execution_open"]
        result[str(symbol).upper()] = series
    return result


def _horizon_return(series: pd.Series | None, execution_at: pd.Timestamp, horizon: int) -> float | None:
    if series is None or execution_at not in series.index:
        return None
    loc = series.index.get_loc(execution_at)
    if not isinstance(loc, int) or loc + horizon >= len(series):
        return None
    start = _finite(series.iloc[loc])
    end = _finite(series.iloc[loc + horizon])
    if start in {None, 0.0} or end is None:
        return None
    return float(end / start - 1.0)


def _best_action(returns: dict[str, float | None]) -> tuple[str | None, float | None]:
    valid = {key: value for key, value in returns.items() if value is not None}
    if not valid:
        return None, None
    best = max(valid, key=lambda key: valid[key])
    rotate = returns.get("ROTATE")
    edge = None if rotate is None else float(valid[best] - rotate)
    return best, edge


def _summary_for(rows: list[dict[str, Any]], horizon: int) -> dict[str, Any]:
    if not rows:
        return {"count": 0, "best_action_counts": {"ROTATE": 0, "HOLD": 0, "CASH": 0}}
    key = f"best_action_{horizon}d"
    counts = Counter(str(row.get(key)) for row in rows if row.get(key))
    def avg(action: str) -> float | None:
        values = [_finite(row.get(f"{action.lower()}_return_{horizon}d")) for row in rows]
        values = [value for value in values if value is not None]
        return float(sum(values) / len(values)) if values else None
    edges = [_finite(row.get(f"best_edge_vs_rotate_{horizon}d")) for row in rows]
    edges = [value for value in edges if value is not None]
    return {
        "count": len(rows),
        "best_action_counts": {action: int(counts.get(action, 0)) for action in ("ROTATE", "HOLD", "CASH")},
        "average_returns": {action: avg(action) for action in ("ROTATE", "HOLD", "CASH")},
        "average_oracle_edge_vs_rotate": float(sum(edges) / len(edges)) if edges else None,
    }


def build_analysis(*, risk: dict[str, Any], market_rows: list[dict[str, Any]], run_id: str, processing_id: str, period_start: str, period_end: str) -> dict[str, Any]:
    oos = risk.get("oos") if isinstance(risk.get("oos"), dict) else {}
    scored = [dict(row) for row in (oos.get("scored_transitions") or []) if isinstance(row, dict)]
    alerts = [row for row in scored if bool(row.get("high_risk"))]
    if not alerts:
        raise ValueError("Risk-Aware Alternative Action requires OOS high-risk transition alerts.")
    series = _series_by_symbol(market_rows)
    if not series:
        raise ValueError("Risk-Aware Alternative Action requires market replay prices.")

    rows: list[dict[str, Any]] = []
    for source in alerts:
        execution = _stamp(source.get("execution_at"))
        if execution is None:
            continue
        incumbent = str(source.get("from_asset") or "CASH").upper()
        challenger = str(source.get("to_asset") or "CASH").upper()
        row = {
            "transition_key": source.get("transition_key"),
            "decision_at": source.get("decision_at"),
            "execution_at": source.get("execution_at"),
            "year": source.get("year"),
            "from_asset": incumbent,
            "to_asset": challenger,
            "risk_score": source.get("risk_score"),
            "risk_threshold": source.get("risk_threshold"),
            "risk_margin": (_finite(source.get("risk_score")) or 0.0) - (_finite(source.get("risk_threshold")) or 0.0),
            "rotation_value_added": source.get("rotation_value_added"),
            "severe": bool(source.get("severe")),
            "realized_rotation_harmful": (_finite(source.get("rotation_value_added")) or 0.0) < 0.0,
        }
        for horizon in HORIZONS:
            rotate = 0.0 if challenger == "CASH" else _horizon_return(series.get(challenger), execution, horizon)
            hold = 0.0 if incumbent == "CASH" else _horizon_return(series.get(incumbent), execution, horizon)
            cash = 0.0
            returns = {"ROTATE": rotate, "HOLD": hold, "CASH": cash}
            for action, value in returns.items():
                row[f"{action.lower()}_return_{horizon}d"] = value
            best, edge = _best_action(returns)
            row[f"best_action_{horizon}d"] = best
            row[f"best_edge_vs_rotate_{horizon}d"] = edge
        actions = [row.get(f"best_action_{h}d") for h in (3, 5, 10) if row.get(f"best_action_{h}d")]
        row["stable_best_action"] = actions[0] if actions and len(set(actions)) == 1 else "MIXED"
        rows.append(row)

    harmful = [row for row in rows if row["realized_rotation_harmful"]]
    severe = [row for row in rows if row["severe"]]
    yearly = []
    for year in sorted({int(row["year"]) for row in rows if row.get("year") is not None}):
        year_rows = [row for row in rows if int(row.get("year") or 0) == year]
        yearly.append({"test_year": year, **_summary_for(year_rows, PRIMARY_HORIZON)})

    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "status": "completed",
        "run_id": str(run_id),
        "processing_id": str(processing_id),
        "period_start": str(period_start),
        "period_end": str(period_end),
        "protocol": {
            "source": "persisted_oos_risk_alerts",
            "candidate_actions": ["ROTATE", "HOLD", "CASH"],
            "horizons_sessions": list(HORIZONS),
            "primary_horizon_sessions": PRIMARY_HORIZON,
            "cash_return_assumption": 0.0,
            "market_return_basis": "execution_open_to_future_execution_open",
            "transaction_costs": "not_applied_in_first_counterfactual_diagnostic",
            "future_information_in_detector": False,
            "future_information_in_counterfactual_label": True,
            "strategy_decisions_changed": False,
            "research_only": True,
        },
        "readiness": {
            "status": "counterfactual_diagnostic_only",
            "policy_ready": False,
            "reason": "The best action is observed post-hoc. A separate walk-forward action selector is required before any policy change.",
        },
        "summary": {
            "alerts": len(rows),
            "harmful_alerts": len(harmful),
            "severe_alerts": len(severe),
            "primary_horizon": _summary_for(rows, PRIMARY_HORIZON),
            "harmful_primary_horizon": _summary_for(harmful, PRIMARY_HORIZON),
            "severe_primary_horizon": _summary_for(severe, PRIMARY_HORIZON),
        },
        "horizons": [{"horizon": horizon, **_summary_for(rows, horizon)} for horizon in HORIZONS],
        "yearly_oos": yearly,
        "alerts": rows,
    }
