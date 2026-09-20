from __future__ import annotations

import math

import pandas as pd

from market_cycle_trader_api.engine.research_challengers import (
    _lightgbm_temporal_fit_split,
    _regression_error_diagnostics,
)
from market_cycle_trader_api.services.model_tuning import _SEARCH_SPACE


def test_temporal_early_stopping_uses_chronological_tail() -> None:
    frame = pd.DataFrame(
        {"value": range(700)},
        index=pd.date_range("2020-01-01", periods=700, freq="D", tz="UTC"),
    )
    train, validation = _lightgbm_temporal_fit_split(
        frame,
        {
            "early_stopping_enabled": True,
            "early_stopping_validation_fraction": 0.15,
            "early_stopping_min_validation_sessions": 40,
            "early_stopping_max_validation_sessions": 126,
        },
        minimum_rows=700,
    )

    assert len(train) == 595
    assert len(validation) == 105
    assert train.index.max() < validation.index.min()
    assert validation.index.equals(frame.index[-105:])


def test_temporal_early_stopping_can_be_disabled() -> None:
    frame = pd.DataFrame(
        {"value": range(700)},
        index=pd.date_range("2020-01-01", periods=700, freq="D", tz="UTC"),
    )
    train, validation = _lightgbm_temporal_fit_split(
        frame,
        {"early_stopping_enabled": False},
        minimum_rows=700,
    )

    assert len(train) == len(frame)
    assert validation.empty


def test_regression_diagnostics_report_mae_and_rmse() -> None:
    diagnostics = _regression_error_diagnostics(
        actual=[0.0, 1.0, 2.0],
        predicted=[0.0, 2.0, 1.0],
    )

    assert diagnostics["rows"] == 3
    assert math.isclose(float(diagnostics["mae"]), 2.0 / 3.0)
    assert math.isclose(float(diagnostics["rmse"]), math.sqrt(2.0 / 3.0))


def test_unified_caro_contains_variance_control_dimensions() -> None:
    names = {str(item["name"]) for item in _SEARCH_SPACE}

    assert "min_child_weight" in names
    assert "subsample" in names
    assert "subsample_freq" in names
