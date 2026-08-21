from __future__ import annotations

from copy import deepcopy
from typing import Any

from ...infrastructure.persistence.mongo_repository import bson_value
from ..temporal_policy_replay import _finite

TRAJECTORY_SIGNALS: tuple[dict[str, str], ...] = (
    {"name": "hold_score", "label": "Hold score", "direction": "decrease"},
    {
        "name": "profit_before_loss_probability_h20",
        "label": "Profit-before-loss 20d",
        "direction": "decrease",
    },
    {
        "name": "trend_persistence_probability_h20",
        "label": "Trend persistence 20d",
        "direction": "decrease",
    },
    {
        "name": "predicted_drawdown_h10",
        "label": "Predicted drawdown 10d",
        "direction": "increase",
    },
)

TEMPORAL_POLICY_SEARCH_SPACE: tuple[dict[str, Any], ...] = (
    {
        "name": "timing_base_weak_threshold",
        "type": "number",
        "min": 0.35,
        "max": 0.65,
        "precision": 4,
        "role": "winner_anchor_weakness",
    },
    {
        "name": "timing_challenger_minimum",
        "type": "number",
        "min": 0.45,
        "max": 0.80,
        "precision": 4,
        "role": "challenger_confirmation",
    },
    {
        "name": "timing_minimum_advantage",
        "type": "number",
        "min": 0.05,
        "max": 0.35,
        "precision": 4,
        "role": "short_horizon_advantage",
    },
    {
        "name": "timing_maximum_advantage",
        "type": "number",
        "min": 0.45,
        "max": 1.00,
        "precision": 4,
        "role": "short_horizon_overconfidence_ceiling",
    },
    {
        "name": "trajectory_lookback_sessions",
        "type": "integer",
        "min": 1,
        "max": 5,
        "role": "incumbent_trajectory_window",
    },
    {
        "name": "trajectory_deterioration_quantile",
        "type": "number",
        "min": 0.60,
        "max": 0.95,
        "precision": 4,
        "role": "fold_relative_deterioration_threshold",
    },
    {
        "name": "trajectory_min_signals",
        "type": "integer",
        "min": 1,
        "max": 4,
        "role": "trajectory_agreement",
    },
    {
        "name": "late_exit_min_challenger_advantage",
        "type": "number",
        "min": 0.00,
        "max": 0.35,
        "precision": 4,
        "role": "late_exit_challenger_quality",
    },
    {
        "name": "late_exit_cash_guard",
        "type": "integer",
        "min": 0,
        "max": 1,
        "role": "temporary_cash_response",
    },
)


def base_settings(run: dict[str, Any]) -> dict[str, Any]:
    result = run.get("result") if isinstance(run.get("result"), dict) else {}
    multi = result.get("multi_horizon_metrics") if isinstance(result.get("multi_horizon_metrics"), dict) else {}
    capital = multi.get("shadow_capital") if isinstance(multi.get("shadow_capital"), dict) else {}
    defaults = {
        "timing_base_weak_threshold": 0.50,
        "timing_challenger_minimum": 0.60,
        "timing_minimum_advantage": 0.25,
        "timing_maximum_advantage": 1.00,
    }
    values: dict[str, Any] = {
        name: float(capital.get(name)) if _finite(capital.get(name)) is not None else default
        for name, default in defaults.items()
    }
    values.update({
        "trajectory_lookback_sessions": 2,
        "trajectory_deterioration_quantile": 0.80,
        "trajectory_min_signals": 2,
        "late_exit_min_challenger_advantage": 0.10,
        "late_exit_cash_guard": 0,
    })
    return normalize_settings(values)


def normalize_settings(settings: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(settings)
    for spec in TEMPORAL_POLICY_SEARCH_SPACE:
        name = str(spec["name"])
        value = normalized.get(name)
        if spec["type"] == "integer":
            parsed = int(round(float(value if value is not None else spec["min"])))
            normalized[name] = max(int(spec["min"]), min(parsed, int(spec["max"])))
        else:
            parsed = float(value if value is not None else spec["min"])
            parsed = max(float(spec["min"]), min(parsed, float(spec["max"])))
            normalized[name] = round(parsed, int(spec.get("precision") or 8))
    return normalized


def search_space_payload(run: dict[str, Any], *, fold_ids: list[int]) -> dict[str, Any]:
    return bson_value({
        "strategy_mode": "WINNER_ANCHORED_TEMPORAL_POLICY_SEARCH",
        "objective": "compound_capital_with_temporal_robustness",
        "dimensions": [dict(item) for item in TEMPORAL_POLICY_SEARCH_SPACE],
        "base_settings": base_settings(run),
        "trajectory_signals": [dict(item) for item in TRAJECTORY_SIGNALS],
        "available_outer_folds": [int(item) for item in fold_ids],
        "protocol": {
            "sampling": "latin_hypercube",
            "adaptive_refinement": "CARO_gaussian_process",
            "validation": "nested_leave_one_temporal_fold_out",
            "comparison": "stitched_outer_fold_controlled_comparison",
            "selection_rule": "outer_fold_robustness_before_full_period_descriptive_metrics",
            "future_data_in_decision_features": False,
            "frozen_temporal_observations": True,
            "winner_anchor_unchanged": True,
        },
    })
