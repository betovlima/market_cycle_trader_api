from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from ...engine.capital_rotation import _build_walk_forward_folds, prepare_rotation_panel
from .errors import TemporalModelTuningCancelled
from ...engine.temporal_intelligence import (
    _align_test_targets,
    _future_target_matrices,
    _open_price_matrix,
    _pooled_features,
)

_TARGET_NAMES = (
    "profit_before_loss",
    "bottom",
    "top",
    "trend_persistence",
    "trend_direction",
    "drawdown",
)


def _split_payload(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    dates: pd.DatetimeIndex,
) -> dict[str, Any]:
    x, metadata = _pooled_features(frames, symbols, dates)
    return {"x": x, "metadata": metadata}


def _target_payload(metadata: pd.DataFrame, targets: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        name: _align_test_targets(metadata, targets[name])
        for name in _TARGET_NAMES
    }


def prepare_training_context(
    bars_by_symbol: dict[str, pd.DataFrame],
    config: Any,
    *,
    progress_callback: Callable[[float, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    def cancelled() -> None:
        if cancel_check is not None and bool(cancel_check()):
            raise TemporalModelTuningCancelled("Temporal Model Tuning cancelled by user.")

    cancelled()
    if progress_callback:
        progress_callback(10.0, "Building temporal feature panel")
    frames, common_dates = prepare_rotation_panel(bars_by_symbol, config)
    cancelled()
    symbols = sorted(frames)
    horizons = sorted({int(value) for value in config.rotation_target_horizons})
    folds = _build_walk_forward_folds(common_dates, config)
    if progress_callback:
        progress_callback(12.0, "Building temporal target matrices")
    targets_by_horizon = _future_target_matrices(frames, common_dates, symbols, horizons)
    open_prices = _open_price_matrix(frames, common_dates, symbols)
    cancelled()

    fold_contexts: dict[int, dict[str, Any]] = {}
    for fold_position, fold in enumerate(folds, start=1):
        train_dates = common_dates[: int(fold["train_end_index"])]
        calibration_dates = common_dates[int(fold["calibration_start_index"]): int(fold["calibration_end_index"])]
        final_fit_dates = common_dates[: int(fold["final_fit_end_index"])]
        test_dates = common_dates[int(fold["test_start_index"]): int(fold["test_end_index"])]
        splits = {
            "train": _split_payload(frames, symbols, train_dates),
            "calibration": _split_payload(frames, symbols, calibration_dates),
            "final_fit": _split_payload(frames, symbols, final_fit_dates),
            "test": _split_payload(frames, symbols, test_dates),
        }
        horizon_targets: dict[int, dict[str, dict[str, np.ndarray]]] = {}
        for horizon in horizons:
            targets = targets_by_horizon[horizon]
            horizon_targets[horizon] = {
                split_name: _target_payload(split_payload["metadata"], targets)
                for split_name, split_payload in splits.items()
            }
        fold_contexts[int(fold["fold_id"])] = {
            "splits": splits,
            "targets": horizon_targets,
        }
        cancelled()
        if progress_callback:
            progress_callback(
                13.0 + 5.0 * (fold_position / max(1, len(folds))),
                f"Preparing reusable training splits {fold_position}/{len(folds)}",
            )

    return {
        "frames": frames,
        "common_dates": common_dates,
        "symbols": symbols,
        "horizons": horizons,
        "folds": folds,
        "targets_by_horizon": targets_by_horizon,
        "open_prices": open_prices,
        "fold_contexts": fold_contexts,
    }
