from __future__ import annotations

from types import SimpleNamespace
from datetime import datetime, timezone
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


def _manual_run(*, status="manual_pending", plan_id="manual-7"):
    return {
        "run_id": "paper-market-2026-10-07-test",
        "active_key": scheduler.ACTIVE_KEY,
        "execution_session": "2026-10-07",
        "status": status,
        "phase": "awaiting_manual_paper_execution",
        "plan_id": plan_id,
        "manual_recovery_plan_id": plan_id,
        "expected_market_open": pd.Timestamp("2026-10-07T13:30:00Z"),
    }


def _man_manual_plan():
    return {
        "plan_id": "manual-7",
        "status": "prepared",
        "execution_session": "2026-10-07",
        "decision_date": "2026-10-07",
        "current_asset": "CASH",
        "target_asset": "MAN",
        "action": "buy",
        "plan_source": "manual_current_session_recovery",
    }


def _manual_db():
    runs = MagicMock()
    runs.find_one_and_update.return_value = {"run_id": "paper-market-2026-10-07-test"}
    runs.update_one.return_value.matched_count = 1
    plans = MagicMock()
    return {
        scheduler.PAPER_MARKET_RUNS_COLLECTION: runs,
        scheduler.PAPER_TRADE_PLANS_COLLECTION: plans,
    }, runs, plans


def test_manual_reanalysis_claims_run_before_training_and_links_new_plan():
    run = _manual_run(status="prepared", plan_id="scheduled-old")
    run["phase"] = "waiting_for_next_market_open"
    context = _open_context(run=run, plan={"plan_id": "scheduled-old", "status": "prepared"})
    prepared_plan = _man_manual_plan()
    db, runs, _ = _manual_db()
    with (
        patch.object(scheduler, "_manual_recovery_context", return_value=context),
        patch.object(scheduler, "prepare_next_paper_plan", return_value=prepared_plan) as prepare,
        patch.object(scheduler, "_automation_document", return_value={"control_mode": "active"}),
        patch.object(scheduler, "_record_admin_operation"),
        patch.object(scheduler, "paper_market_manual_recovery_status", return_value={"can_execute": True}),
    ):
        result = scheduler.prepare_manual_current_session_plan(db)

    assert result["status"] == "prepared"
    assert prepare.call_count == 1
    claim = runs.find_one_and_update.call_args
    assert claim.args[0]["active_key"] == scheduler.ACTIVE_KEY
    assert claim.args[1]["$set"]["status"] == "manual_preparing"
    published = runs.update_one.call_args.args[1]["$set"]
    assert published["status"] == "manual_pending"
    assert published["plan_id"] == "manual-7"
    assert published["manual_recovery_plan_id"] == "manual-7"
    assert published["action"] == "buy"
    assert published["target_asset"] == "MAN"


def test_manual_execution_finalizes_active_run_without_duplicate_order():
    run = _manual_run()
    plan = _man_manual_plan()
    context = _open_context(run=run, plan=plan)
    db, runs, _ = _manual_db()
    fill_result = {"action": "buy", "target_asset": "MAN", "plan_id": "manual-7"}
    with (
        patch.object(scheduler, "_manual_recovery_context", return_value=context),
        patch.object(scheduler, "_infer_recoverable_contingency", return_value=(None, False)),
        patch.object(scheduler, "_automation_document", return_value={"control_mode": "active"}),
        patch.object(scheduler, "_record_admin_operation"),
        patch.object(scheduler, "execute_prepared_paper_plan", return_value=fill_result) as execute,
        patch.object(scheduler, "paper_market_manual_recovery_status", return_value={}),
    ):
        result = scheduler.execute_manual_current_session_plan(db, plan_id="manual-7")

    assert result["status"] == "executed"
    execute.assert_called_once_with(db, plan_id="manual-7")
    assert runs.find_one_and_update.call_args.args[1]["$set"]["status"] == "manual_executing"
    finishing = runs.update_one.call_args.args[1]
    assert finishing["$set"]["status"] == "completed"
    assert finishing["$set"]["execution_origin"] == "manual_recovery"
    assert finishing["$unset"]["active_key"] == ""
    assert finishing["$set"]["execution_result"]["target_asset"] == "MAN"


def test_manual_execution_rejects_old_or_unapproved_plan_without_orders():
    run = _manual_run()
    db, runs, _ = _manual_db()
    scenarios = [
        {**_man_manual_plan(), "plan_id": "scheduled-old"},
        {**_man_manual_plan(), "plan_source": "premarket_plan_refresh"},
        {**_man_manual_plan(), "execution_session": "2026-10-06"},
    ]
    for bad_plan in scenarios:
        context = _open_context(run=run, plan=bad_plan)
        with (
            patch.object(scheduler, "_manual_recovery_context", return_value=context),
            patch.object(scheduler, "_infer_recoverable_contingency", return_value=(None, False)),
            patch.object(scheduler, "_automation_document", return_value={"control_mode": "active"}),
            patch.object(scheduler, "execute_prepared_paper_plan") as execute,
        ):
            try:
                scheduler.execute_manual_current_session_plan(db, plan_id=bad_plan["plan_id"])
            except RuntimeError:
                pass
            else:
                raise AssertionError("Old/unapproved Paper plan must be rejected.")
            execute.assert_not_called()
    runs.find_one_and_update.assert_not_called()


def test_manual_execution_failure_marks_uncertain_order_state_for_review():
    run = _manual_run()
    plan = _man_manual_plan()
    db, runs, plans = _manual_db()
    plans.find_one.return_value = {"plan_id": "manual-7", "status": "executing"}
    with (
        patch.object(scheduler, "_manual_recovery_context", return_value=_open_context(run=run, plan=plan)),
        patch.object(scheduler, "_infer_recoverable_contingency", return_value=(None, False)),
        patch.object(scheduler, "_automation_document", return_value={"control_mode": "active"}),
        patch.object(scheduler, "_record_admin_operation"),
        patch.object(scheduler, "execute_prepared_paper_plan", side_effect=RuntimeError("broker timeout")),
    ):
        try:
            scheduler.execute_manual_current_session_plan(db, plan_id="manual-7")
        except RuntimeError as exc:
            assert "broker timeout" in str(exc)
        else:
            raise AssertionError("Uncertain manual execution must fail.")
    assert runs.update_one.call_args.args[1]["$set"]["status"] == "review_required"


def test_manual_plan_prevents_scheduler_from_reexecuting_stale_plan():
    run = _manual_run(status="manual_pending")
    db, _, _ = _manual_db()
    with (
        patch.object(scheduler, "_automation_document", return_value={"control_mode": "active"}),
        patch.object(scheduler, "_ensure_continuous_run", return_value=run),
        patch.object(scheduler, "utc_now", return_value=datetime(2026, 10, 7, 17, 0, tzinfo=timezone.utc)),
        patch.object(scheduler, "refresh_trader_live_market_data") as refresh,
        patch.object(scheduler, "_execute_run") as execute,
    ):
        scheduler._advance_one(db, "worker-test")
    refresh.assert_not_called()
    execute.assert_not_called()


def test_abandoned_manual_plan_expires_and_releases_next_session():
    run = _manual_run(status="manual_pending")
    db, runs, plans = _manual_db()
    with (
        patch.object(scheduler, "_automation_document", return_value={"control_mode": "active"}),
        patch.object(scheduler, "_ensure_continuous_run", return_value=run),
        patch.object(scheduler, "utc_now", return_value=datetime(2026, 10, 8, 19, 0, tzinfo=timezone.utc)),
        patch.object(scheduler, "refresh_trader_live_market_data") as refresh,
    ):
        scheduler._advance_one(db, "worker-test")
    refresh.assert_not_called()
    assert plans.update_one.call_args.args[0]["plan_id"] == "manual-7"
    assert plans.update_one.call_args.args[1]["$set"]["status"] == "cancelled"
    assert runs.update_one.call_args.args[1]["$set"]["status"] == "failed"
    assert runs.update_one.call_args.args[1]["$unset"]["active_key"] == ""


def test_manual_prepare_failure_does_not_release_stale_automatic_plan():
    run = _manual_run(status="prepared", plan_id="scheduled-old")
    run["phase"] = "waiting_for_next_market_open"
    db, runs, _ = _manual_db()
    with (
        patch.object(scheduler, "_manual_recovery_context", return_value=_open_context(run=run, plan=None)),
        patch.object(scheduler, "prepare_next_paper_plan", side_effect=RuntimeError("calibration failed")),
    ):
        try:
            scheduler.prepare_manual_current_session_plan(db)
        except RuntimeError as exc:
            assert "calibration failed" in str(exc)
        else:
            raise AssertionError("Manual calibration failure must be reported.")
    assert runs.update_one.call_args.args[1]["$set"]["status"] == "manual_pending"
    assert runs.update_one.call_args.args[1]["$set"]["phase"] == "manual_reanalysis_failed"


def test_manual_recovery_status_prefers_approved_plan_over_unrelated_contingency():
    context = _open_context(run=_manual_run(), plan=_man_manual_plan())
    with (
        patch.object(scheduler, "_manual_recovery_context", return_value=context),
        patch.object(scheduler, "_infer_recoverable_contingency", return_value=(None, False)),
    ):
        result = scheduler.paper_market_manual_recovery_status({})
    assert result["can_execute"] is True
    assert result["plan_id"] == "manual-7"


