from __future__ import annotations

from typing import Any

from .relative_model import fit_fold_horizon


def calibrate_fold_horizon(
    training: dict[str, Any],
    config: Any,
    *,
    fold: dict[str, Any],
    horizon: int,
    settings: dict[str, Any],
    round_trip_cost_rate: float,
) -> dict[str, Any]:
    return fit_fold_horizon(
        training,
        config,
        fold=fold,
        horizon=horizon,
        settings=settings,
        round_trip_cost_rate=round_trip_cost_rate,
    )
