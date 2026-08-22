from __future__ import annotations

from typing import Any

from .utils import as_float


def rank_value(row: dict[str, Any]) -> float:
    for field in ("risk_adjusted_entry_score", "opportunity_gate_score", "entry_rank_score"):
        value = as_float(row.get(field))
        if value is not None:
            return value
    return -1e9


def tail_risk_proxy(row: dict[str, Any]) -> float:
    safety = as_float(row.get("all_horizon_risk_safety"), 0.5) or 0.5
    return max(0.0, min(1.0, 1.0 - safety))


def objective_breakdown(
    row: dict[str, Any],
    *,
    current_symbol: str,
    anchor_symbol: str,
    configuration: dict[str, Any],
) -> dict[str, float]:
    opportunity = rank_value(row)
    short_signal = as_float(row.get("short_profit_consensus"), 0.5) or 0.5
    long_signal = as_float(row.get("long_profit_confirmation"), 0.5) or 0.5
    agreement = as_float(row.get("cross_horizon_agreement"), 0.5) or 0.5
    temporal_signal = (short_signal + long_signal + agreement) / 3.0
    symbol = str(row.get("symbol") or "").upper()
    persistence = (as_float(row.get("incumbent_persistence_score"), 0.0) or 0.0) if symbol == current_symbol else 0.0
    predicted_drawdown = max(0.0, as_float(row.get("predicted_drawdown"), 0.0) or 0.0)
    switched = 0.0 if symbol == current_symbol else 1.0
    anchored = 1.0 if symbol == anchor_symbol else 0.0
    components = {
        "opportunity": float(configuration["opportunity_weight"]) * opportunity,
        "temporal": float(configuration["temporal_weight"]) * temporal_signal,
        "persistence": float(configuration["persistence_weight"]) * persistence,
        "reference_anchor": float(configuration["reference_anchor_weight"]) * anchored,
        "risk": -float(configuration["risk_penalty"]) * predicted_drawdown,
        "tail_risk": -float(configuration["tail_risk_penalty"]) * tail_risk_proxy(row),
        "turnover": -float(configuration["turnover_penalty"]) * switched,
        "switch": -float(configuration["switch_penalty"]) * switched,
    }
    components["objective"] = float(sum(components.values()))
    return components
