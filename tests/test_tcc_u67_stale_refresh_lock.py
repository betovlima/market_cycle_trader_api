from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

from market_cycle_trader_api.infrastructure.persistence.mongo_repository import (
    PAPER_MARKET_RUNS_COLLECTION,
    PAPER_TRADE_PLANS_COLLECTION,
    STRATEGY_CONTROL_COLLECTION,
    utc_now,
)
from market_cycle_trader_api.services.tcc_u67_operational_strategy import (
    recover_stale_live_market_refresh_lock,
)


def _db_with_lock(*, started_at, active_run=None, executing_plan=None):
    control = MagicMock()
    control.find_one.return_value = {
        "_id": "default",
        "live_market_refresh_in_progress": True,
        "live_market_refresh_started_at": started_at,
        "live_market_refresh_source": "premarket_plan_refresh",
    }
    control.update_one.return_value = SimpleNamespace(modified_count=1)

    runs = MagicMock()
    runs.find_one.return_value = active_run

    plans = MagicMock()
    plans.find_one.return_value = executing_plan

    db = {
        STRATEGY_CONTROL_COLLECTION: control,
        PAPER_MARKET_RUNS_COLLECTION: runs,
        PAPER_TRADE_PLANS_COLLECTION: plans,
    }
    return db, control


def test_stale_refresh_lock_is_released_without_touching_paper_state() -> None:
    db, control = _db_with_lock(
        started_at=utc_now() - timedelta(hours=7),
    )

    result = recover_stale_live_market_refresh_lock(db)

    assert result["released"] is True
    assert result["code"] == "stale_lock_released"
    assert result["broker_interaction_performed"] is False
    assert result["paper_state_changed"] is False
    assert result["winner_changed"] is False
    control.update_one.assert_called_once()


def test_recent_refresh_lock_is_not_released() -> None:
    db, control = _db_with_lock(
        started_at=utc_now() - timedelta(hours=1),
    )

    result = recover_stale_live_market_refresh_lock(db)

    assert result["released"] is False
    assert result["code"] == "refresh_not_stale"
    control.update_one.assert_not_called()


def test_stale_lock_is_not_released_during_active_calibration() -> None:
    db, control = _db_with_lock(
        started_at=utc_now() - timedelta(hours=7),
        active_run={
            "run_id": "run-1",
            "status": "preparing",
            "phase": "refreshing_market_data_and_preparing_post_close_plan",
        },
    )

    result = recover_stale_live_market_refresh_lock(db)

    assert result["released"] is False
    assert result["code"] == "active_pipeline_detected"
    control.update_one.assert_not_called()


def test_stale_lock_is_not_released_during_order_execution() -> None:
    db, control = _db_with_lock(
        started_at=utc_now() - timedelta(hours=7),
        executing_plan={"plan_id": "plan-1", "status": "executing"},
    )

    result = recover_stale_live_market_refresh_lock(db)

    assert result["released"] is False
    assert result["code"] == "active_pipeline_detected"
    control.update_one.assert_not_called()
