from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

from market_cycle_trader_api.services import paper_market_scheduler as scheduler


def _open_context(*, run=None, plan=None):
    return {
        "readiness": {},
        "clock": {
            "is_open": True,
            "timestamp": pd.Timestamp("2026-10-07T17:00:00Z"),
            "next_open": pd.Timestamp("2026-10-08T13:30:00Z"),
            "next_close": pd.Timestamp("2026-10-07T20:00:00Z"),
        },
        "timestamp": pd.Timestamp("2026-10-07T17:00:00Z"),
        "current_session": "2026-10-07",
        "run": run,
        "plan": plan,
        "contingency_plan": None,
        "execution_plan": plan,
    }


def test_manual_reanalysis_does_not_require_scheduled_run() -> None:
    context = _open_context(run=None, plan=None)
    with (
        patch.object(
            scheduler,
            "_manual_recovery_context",
            return_value=context,
        ),
        patch.object(
            scheduler,
            "_infer_recoverable_contingency",
            return_value=(None, False),
        ),
    ):
        status = scheduler.paper_market_manual_recovery_status({})

    assert status["available"] is True
    assert status["can_prepare"] is True
    assert status["scheduled_run_required"] is False
    assert status["prepare_reason"] is None
    assert status["can_execute"] is False


def test_manual_reanalysis_without_run_uses_current_session_open() -> None:
    context = _open_context(run=None, plan=None)
    plans = MagicMock()
    db = {scheduler.PAPER_MARKET_RUNS_COLLECTION: MagicMock()}

    prepared_plan = {
        "plan_id": "plan-current-session",
        "current_asset": "XSD",
        "target_asset": "XSD",
        "action": "hold",
    }

    with (
        patch.object(
            scheduler,
            "_manual_recovery_context",
            return_value=context,
        ),
        patch.object(
            scheduler,
            "prepare_next_paper_plan",
            return_value=prepared_plan,
        ) as prepare,
        patch.object(
            scheduler,
            "_automation_document",
            return_value={"control_mode": "active"},
        ),
        patch.object(scheduler, "_record_admin_operation"),
        patch.object(
            scheduler,
            "paper_market_manual_recovery_status",
            return_value={"can_prepare": True, "can_execute": True},
        ),
    ):
        result = scheduler.prepare_manual_current_session_plan(
            db,
            actor_email="admin@example.com",
        )

    assert result["status"] == "prepared"
    args = prepare.call_args.kwargs
    assert args["allow_open_market"] is True
    assert args["execution_session_override"] == "2026-10-07"
    assert pd.Timestamp(args["expected_market_open_override"]) == pd.Timestamp(
        "2026-10-07T13:30:00Z"
    )
    assert args["refresh_source"] == "manual_current_session_recovery"
    db[scheduler.PAPER_MARKET_RUNS_COLLECTION].update_one.assert_not_called()


def test_manual_reanalysis_does_not_replace_executed_plan() -> None:
    context = _open_context(
        run=None,
        plan={
            "plan_id": "already-executed",
            "status": "executed",
        },
    )
    with patch.object(
        scheduler,
        "_manual_recovery_context",
        return_value=context,
    ):
        try:
            scheduler.prepare_manual_current_session_plan({})
        except RuntimeError as exc:
            assert "status=executed" in str(exc)
        else:
            raise AssertionError("Expected an executed plan to remain immutable.")
