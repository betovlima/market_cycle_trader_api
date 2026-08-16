from __future__ import annotations

from pathlib import Path

import pytest

from market_cycle_trader_api.services.temporal_policy_replay import _replay_rows


def test_temporal_reproducibility_request_normalizes_xgb_threads_without_mutating_strategy() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "market_cycle_trader_api"
    source = (root / "services" / "temporal_intelligence.py").read_text(encoding="utf-8")
    assert '"deterministic_execution": True' in source
    assert source.count('"xgb_n_jobs": 1') >= 2
    assert '"numeric_thread_limit": 1' in source


def test_temporal_strategy_has_dedicated_model_tuning_selection_contract() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "market_cycle_trader_api"
    strategy_service = (root / "services" / "strategy_lab.py").read_text(encoding="utf-8")
    tuning_service = (root / "services" / "model_tuning.py").read_text(encoding="utf-8")
    router = (root / "api" / "routers" / "strategy_lab.py").read_text(encoding="utf-8")
    assert "model_tuning_strategy_id" in strategy_service
    assert "select_model_tuning_strategy" in strategy_service
    assert '"/{strategy_id}/select-for-model-tuning"' in router
    assert "TEMPORAL_POLICY_TUNING_SCOPE" in tuning_service
    assert "frozen_temporal_replay" in tuning_service
    assert "evaluate_temporal_policy_candidate" in tuning_service


def test_temporal_policy_replay_changes_only_top1_top2_timing() -> None:
    observations = {
        "2026-01-02T00:00:00+00:00": {
            "fold_id": 1,
            "rows_by_symbol": {
                "A": {"short_profit_consensus": 0.40, "open_to_open_return": 0.00},
                "B": {"short_profit_consensus": 0.80, "open_to_open_return": 0.10},
            },
        },
        "2026-01-05T00:00:00+00:00": {
            "fold_id": 1,
            "rows_by_symbol": {
                "A": {"short_profit_consensus": 0.70, "open_to_open_return": 0.02},
                "B": {"short_profit_consensus": 0.50, "open_to_open_return": -0.05},
            },
        },
    }
    winner_rows = [
        {"decision_date": "2026-01-02T00:00:00+00:00", "selected_asset": "A", "top_1_asset": "A", "top_2_asset": "B"},
        {"decision_date": "2026-01-05T00:00:00+00:00", "selected_asset": "A", "top_1_asset": "A", "top_2_asset": "B"},
    ]
    metrics, _ = _replay_rows(
        observations,
        winner_rows,
        initial_capital=100.0,
        one_side_cost=0.0,
        settings={
            "timing_base_weak_threshold": 0.50,
            "timing_challenger_minimum": 0.60,
            "timing_minimum_advantage": 0.25,
        },
        winner_fold_returns={1: 0.02},
    )
    assert metrics["timing_override_count"] == 1
    assert metrics["ending_capital"] == pytest.approx(112.2)
    assert metrics["capital_rotations"] == 2
    assert metrics["eligible"] is True
