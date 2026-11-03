"""Read-only native TreeSHAP audit for already fitted Live LightGBM models.

Attributions are calculated after the original utility prediction. They never
replace model scores, calibrate a policy, or change trading decisions.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def explain_live_utilities(
    models: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    timestamp: pd.Timestamp,
    utilities: np.ndarray,
    feature_names: list[str] | tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    """Reconstruct each finite asset prediction from native LightGBM contributions.

    Native `pred_contrib=True` supplies one value per input feature plus the
    model's baseline. Invalid/missing explanations are discarded individually
    so audit collection cannot change a trading decision.
    """
    names = list(feature_names)
    explanations: dict[str, dict[str, Any]] = {}
    for index, symbol in enumerate(symbols, start=1):
        try:
            if index >= len(utilities) or not np.isfinite(utilities[index]):
                continue
            model = models.get(symbol)
            frame = frames.get(symbol)
            if model is None or frame is None or timestamp not in frame.index:
                continue
            row = frame.loc[[timestamp], names]
            if row.shape != (1, len(names)) or row.isna().any(axis=None):
                continue
            raw = model.predict(row, pred_contrib=True)
            contributions = np.asarray(raw, dtype=np.float64).reshape(-1)
            values = row.iloc[0].to_numpy(dtype=np.float64)
            if contributions.size != len(names) + 1:
                continue
            if not np.isfinite(contributions).all() or not np.isfinite(values).all():
                continue
            baseline = float(contributions[-1])
            observed = float(utilities[index])
            reconstructed = float(np.sum(contributions))
            error = abs(reconstructed - observed)
            if error > 1e-6 * max(1.0, abs(observed)):
                continue
            explanations[symbol] = {
                "method": "lightgbm_tree_shap",
                "base_value": baseline,
                "prediction": observed,
                "reconstruction_error": error,
                "feature_values": {
                    name: float(value) for name, value in zip(names, values)
                },
                "contributions": {
                    name: float(value)
                    for name, value in zip(names, contributions[:-1])
                },
            }
        except Exception:
            # Explanations are optional. Trading predictions are authoritative.
            continue
    return explanations
