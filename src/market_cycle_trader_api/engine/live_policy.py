from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from .capital_rotation import ROTATION_FEATURES


def live_model_utilities(
    models: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    timestamp: pd.Timestamp,
) -> np.ndarray:
    """Predict utilities after the latest completed close for next-open execution.

    Unlike the backtest utility helper, live inference must not require the next
    session's OHLC row: that session is exactly where the prepared order will be
    executed and is unknown at decision time.
    """

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
) -> Callable[[pd.Timestamp, int, int], tuple[int, float]]:
    """Mirror the validated rotation guards using live-safe model utilities."""

    def policy(timestamp: pd.Timestamp, current_position: int, holding_days: int) -> tuple[int, float]:
        utilities = live_model_utilities(models, frames, symbols, timestamp)
        if not np.isfinite(utilities[1:]).any():
            return (0, 0.0)
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

    return policy
