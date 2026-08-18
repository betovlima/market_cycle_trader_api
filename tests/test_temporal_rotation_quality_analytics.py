from __future__ import annotations

import pytest

from market_cycle_trader_api.services.temporal_rotation_quality_analytics import (
    _combined_equity,
    _rotations_from_replay_rows,
    _stitch_equity_rows,
)


def test_stitch_equity_rows_compounds_fold_reset_curves() -> None:
    rows = [
        {"fold_id": 1, "decision_timestamp": "2024-01-02T00:00:00+00:00", "strategy_equity": 11_000.0},
        {"fold_id": 1, "decision_timestamp": "2024-01-03T00:00:00+00:00", "strategy_equity": 12_000.0},
        {"fold_id": 2, "decision_timestamp": "2025-01-02T00:00:00+00:00", "strategy_equity": 10_500.0},
    ]
    stitched = _stitch_equity_rows(rows)
    assert [row["value"] for row in stitched] == pytest.approx([11_000.0, 12_000.0, 12_600.0])


def test_combined_equity_aligns_candidate_and_control_by_fold_and_timestamp() -> None:
    candidate = [
        {"fold_id": 1, "decision_timestamp": "2024-01-02T00:00:00+00:00", "strategy_equity": 11_000.0},
        {"fold_id": 2, "decision_timestamp": "2025-01-02T00:00:00+00:00", "strategy_equity": 11_000.0},
    ]
    control = [
        {"fold_id": 1, "decision_timestamp": "2024-01-02T00:00:00+00:00", "strategy_equity": 10_500.0},
        {"fold_id": 2, "decision_timestamp": "2025-01-02T00:00:00+00:00", "strategy_equity": 10_000.0},
    ]
    combined = _combined_equity(candidate, control)
    assert combined[0]["simulation_equity"] == pytest.approx(11_000.0)
    assert combined[0]["reference_equity"] == pytest.approx(10_500.0)
    assert combined[1]["simulation_equity"] == pytest.approx(12_100.0)
    assert combined[1]["reference_equity"] == pytest.approx(10_500.0)


def test_rotations_preserve_strong_challenger_metadata() -> None:
    rows = [
        {
            "fold_id": 1,
            "decision_timestamp": "2026-01-05T00:00:00+00:00",
            "simulated_current_symbol": "AAPL",
            "chosen_target_symbol": "NVDA",
            "rotation_blocked": False,
            "strong_challenger_override": True,
            "challenger_quality_floor": 0.65,
            "challenger_entry_rank_score": 0.68,
        }
    ]
    rotations = _rotations_from_replay_rows(rows)
    assert len(rotations) == 1
    assert rotations[0]["from_asset"] == "AAPL"
    assert rotations[0]["to_asset"] == "NVDA"
    assert rotations[0]["strong_challenger_override"] is True
    assert rotations[0]["challenger_quality_floor"] == pytest.approx(0.65)
