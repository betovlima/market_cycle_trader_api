from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from .capital_rotation import ROTATION_FEATURES, _risk_off_enabled
from .selective_opportunity import SelectiveOpportunityGate, evaluate_opportunity, opportunity_cash_gate_enabled, selective_opportunity_enabled


def live_model_utilities(
    models: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    timestamp: pd.Timestamp,
) -> np.ndarray:
    






    values = [0.0]
    for symbol in symbols:
        model = models.get(symbol)
        frame = frames[symbol]
        if model is None or timestamp not in frame.index:
            values.append(float("-inf"))
            continue
        row = frame.loc[[timestamp], ROTATION_FEATURES]
        if row.empty or row.isna().any(axis=None):
            values.append(float("-inf"))
            continue
        values.append(float(model.predict(row)[0]))
    return np.asarray(values, dtype=np.float64)


def build_live_rotation_policy(
    models: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    config: Any,
    switch_margin: float,
    *,
    cash_edge_models: dict[str, Any] | None = None,
    opportunity_gate: SelectiveOpportunityGate | None = None,
) -> Callable[[pd.Timestamp, int, int], tuple[int, float]]:
    

    risk_off = _risk_off_enabled(config)
    selective = selective_opportunity_enabled(config)
    opportunity_cash_gate = opportunity_cash_gate_enabled(config)
    if risk_off and cash_edge_models is None:
        raise ValueError("Explicit risk-off mode requires live cash-edge models.")
    if selective and opportunity_gate is None:
        raise ValueError("Selective Opportunity mode requires a calibrated live opportunity gate.")

    def policy(timestamp: pd.Timestamp, current_position: int, holding_days: int) -> tuple[int, float]:
        utilities = live_model_utilities(models, frames, symbols, timestamp)
        if not np.isfinite(utilities[1:]).any():
            return (0, 0.0)

        if selective and opportunity_gate is not None:
            opportunity = evaluate_opportunity(
                opportunity_gate,
                utilities,
                frames,
                symbols,
                timestamp,
                current_position=current_position if opportunity_cash_gate else None,
            )
            if opportunity is None or not bool(opportunity.accepted):
                return (0, 0.0)

        if not risk_off:
            best = int(np.nanargmax(utilities))
            best_value = float(utilities[best])
            current_value = float(utilities[current_position])
            minimum = float(config.rotation_cash_threshold)
            required = max(float(config.rotation_switch_margin), float(switch_margin))

            if (
                current_position > 0
                and np.isfinite(current_value)
                and holding_days < int(config.rotation_min_holding_days)
            ):
                return (current_position, current_value)
            if best == 0 or best_value <= minimum:
                return (0, 0.0)
            if current_position == 0:
                if best_value >= minimum + float(config.rotation_min_expected_edge):
                    return (best, best_value)
                return (0, 0.0)
            if best == current_position:
                return (current_position, current_value)
            if best_value >= current_value + required:
                return (best, best_value)
            return (current_position, current_value)

        cash_edges = live_model_utilities(cash_edge_models or {}, frames, symbols, timestamp)
        ranked_positions = sorted(
            (
                position
                for position in range(1, len(utilities))
                if np.isfinite(utilities[position])
            ),
            key=lambda position: (-float(utilities[position]), symbols[position - 1]),
        )
        if not ranked_positions:
            return (0, 0.0)

        minimum = float(config.rotation_cash_threshold)
        entry_threshold = minimum + float(config.rotation_min_expected_edge)
        required = max(float(config.rotation_switch_margin), float(switch_margin))
        current_value = float(utilities[current_position])
        current_cash_edge = float(cash_edges[current_position]) if current_position < len(cash_edges) else float("-inf")

        entry_candidates = [
            position
            for position in ranked_positions
            if np.isfinite(cash_edges[position])
            and float(cash_edges[position]) >= entry_threshold
        ]

        if current_position == 0:
            if not entry_candidates:
                return (0, 0.0)
            target = entry_candidates[0]
            return (target, float(utilities[target]))

        current_is_investable = np.isfinite(current_cash_edge) and current_cash_edge > minimum
        if not current_is_investable:
            if entry_candidates:
                target = entry_candidates[0]
                return (target, float(utilities[target]))
            return (0, 0.0)

        if holding_days < int(config.rotation_min_holding_days):
            return (current_position, current_value)

        if not entry_candidates:
            return (current_position, current_value)
        target = entry_candidates[0]
        if target == current_position:
            return (current_position, current_value)
        if float(utilities[target]) >= current_value + required:
            return (target, float(utilities[target]))
        return (current_position, current_value)

    return policy
