from pathlib import Path

import pytest
from pydantic import ValidationError

from market_cycle_trader_api.schemas.model_tuning import ModelTuningStartRequest
from market_cycle_trader_api.services.model_tuning import tuning_catalog
from market_cycle_trader_api.services.temporal_policy_replay import replay_temporal_policy_details
from market_cycle_trader_api.services.temporal_policy_tuning import _replay_rows


def _minimal_replay_rows():
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
    winner = [
        {"decision_date": "2026-01-02T00:00:00+00:00", "selected_asset": "A", "top_1_asset": "A", "top_2_asset": "B"},
        {"decision_date": "2026-01-05T00:00:00+00:00", "selected_asset": "A", "top_1_asset": "A", "top_2_asset": "B"},
    ]
    settings = {
        "timing_base_weak_threshold": 0.5,
        "timing_challenger_minimum": 0.6,
        "timing_minimum_advantage": 0.25,
    }
    return observations, winner, settings


def test_research_budget_is_not_scientifically_capped_at_sixty() -> None:
    request = ModelTuningStartRequest(candidate_count=500)
    assert request.candidate_count == 500
    with pytest.raises(ValidationError):
        ModelTuningStartRequest(candidate_count=2001)


def test_catalog_exposes_continuation_and_adaptive_stopping_contract() -> None:
    catalog = tuning_catalog()
    assert catalog["research_budget_unbounded_across_continuations"] is True
    assert catalog["continue_research_available"] is True
    assert catalog["research_budget_technical_segment_max"] == 2000
    probability = catalog["probability"]
    assert probability["default_adaptive_stopping_enabled"] is True
    assert probability["default_no_improvement_trial_limit"] >= 10
    assert probability["default_minimum_meaningful_improvement"] >= 0


def test_detailed_temporal_policy_replay_preserves_economic_result() -> None:
    observations, winner, settings = _minimal_replay_rows()
    compact = _replay_rows(
        observations,
        winner,
        initial_capital=10_000.0,
        one_side_cost=0.0,
        settings=settings,
        winner_fold_returns={1: 0.02},
    )
    detailed = replay_temporal_policy_details(
        observations,
        winner,
        initial_capital=10_000.0,
        one_side_cost=0.0,
        settings=settings,
        winner_fold_returns={1: 0.02},
    )
    assert detailed["metrics"]["ending_capital"] == pytest.approx(compact[0]["ending_capital"])
    assert detailed["metrics"]["strategy_return"] == pytest.approx(compact[0]["strategy_return"])
    assert len(detailed["equity"]) == 2


def test_champion_lifecycle_and_processing_routes_exist_without_bypassing_trader_boundary() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "market_cycle_trader_api"
    tuning_router = (root / "api" / "routers" / "model_tuning.py").read_text(encoding="utf-8")
    analytics_router = (root / "api" / "routers" / "analytics.py").read_text(encoding="utf-8")
    validation = (root / "services" / "model_tuning_validation.py").read_text(encoding="utf-8")
    strategy_lab = (root / "services" / "strategy_lab.py").read_text(encoding="utf-8")

    assert '"/{run_id}/candidates/{candidate_id}/validate-champion"' in tuning_router
    assert '"/processings"' in analytics_router
    assert '"/processings/{processing_id}"' in analytics_router
    assert '"/processings/{processing_id}/rotation-period"' in analytics_router
    assert '"trader_winner_eligible": False' in validation
    assert "TEMPORAL live execution is not installed" in validation
    assert "_assert_standard_strategy_action(profile, \"Candidate promotion\")" in strategy_lab


def test_frontend_exposes_continue_validate_and_dashboard_processing_flow() -> None:
    front = Path(__file__).resolve().parents[2] / "market_cycle_trader" / "src"
    # This test is optional when the API is tested as a standalone artifact.
    if not front.exists():
        pytest.skip("Frontend source is not adjacent to the API worktree.")
    panel = (front / "features" / "ModelTuningPanel.jsx").read_text(encoding="utf-8")
    dashboard = (front / "features" / "dashboard" / "components" / "DashboardBacktestAnalyticsSection.jsx").read_text(encoding="utf-8")
    assert "Continue Research" in panel
    assert "validate-champion" in panel
    assert "mct:open-dashboard-processing" in panel
    assert "/analytics/processings" in dashboard
