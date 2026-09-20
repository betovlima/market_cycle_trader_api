from __future__ import annotations

import math

from market_cycle_trader_api.engine.research_challengers import (
    _regression_error_diagnostics,
)
from market_cycle_trader_api.services.model_tuning import _SEARCH_SPACE
from market_cycle_trader_api.services.model_tuning_space import (
    sample_value,
    unit_value_for_setting,
)


def test_regression_diagnostics_report_mae_and_rmse() -> None:
    diagnostics = _regression_error_diagnostics(
        actual=[0.0, 1.0, 2.0],
        predicted=[0.0, 2.0, 1.0],
    )

    assert diagnostics["rows"] == 3
    assert math.isclose(float(diagnostics["mae"]), 2.0 / 3.0)
    assert math.isclose(float(diagnostics["rmse"]), math.sqrt(2.0 / 3.0))


def test_unified_caro_contains_variance_control_dimensions() -> None:
    specs = {str(item["name"]): item for item in _SEARCH_SPACE}

    assert "min_child_weight" in specs
    assert "subsample" in specs
    assert "subsample_freq" in specs
    assert specs["min_child_weight"]["scale"] == "log"
    assert float(specs["min_child_weight"]["min"]) <= 5.0
    assert float(specs["min_child_weight"]["max"]) >= 5.0
    assert int(specs["subsample_freq"]["min"]) == 0


def test_log_scaled_child_weight_round_trip_contains_control() -> None:
    spec = next(
        item for item in _SEARCH_SPACE
        if item["name"] == "min_child_weight"
    )
    unit = unit_value_for_setting(spec, 5.0)
    reconstructed = sample_value(spec, unit)

    assert 0.0 < unit < 1.0
    assert math.isclose(float(reconstructed), 5.0, rel_tol=1e-5, abs_tol=1e-5)


def test_log_scaled_midpoint_is_geometric_not_arithmetic() -> None:
    spec = {
        "name": "example",
        "type": "number",
        "min": 0.001,
        "max": 10.0,
        "precision": 8,
        "scale": "log",
    }
    midpoint = sample_value(spec, 0.5)

    assert math.isclose(float(midpoint), 0.1, rel_tol=1e-6)
