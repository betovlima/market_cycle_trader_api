from __future__ import annotations

from types import SimpleNamespace
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd

from market_cycle_trader_api.services import paper_market_scheduler as scheduler
from market_cycle_trader_api.api.routers import admin_trader


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
    db = {
        scheduler.PAPER_MARKET_RUNS_COLLECTION: MagicMock(),
        scheduler.PAPER_MARKET_AUTOMATION_COLLECTION: MagicMock(),
    }

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
        scheduler.PAPER_MARKET_AUTOMATION_COLLECTION: MagicMock(),
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
    updates = [
        call.args[1] for call in
        db[scheduler.PAPER_MARKET_AUTOMATION_COLLECTION].update_one.call_args_list
    ]
    assert [row["$set"]["manual_execution"]["status"] for row in updates] == [
        "running", "completed",
    ]
    assert [row["$push"]["manual_operation_log"]["$each"][0]["operation"]
            for row in updates] == ["execution", "execution"]


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
    assert "manual_recovery_plan_id" in runs.update_one.call_args.args[1]["$unset"]


def test_manual_recovery_status_prefers_approved_plan_over_unrelated_contingency():
    context = _open_context(run=_manual_run(), plan=_man_manual_plan())
    with (
        patch.object(scheduler, "_manual_recovery_context", return_value=context),
        patch.object(scheduler, "_infer_recoverable_contingency", return_value=(None, False)),
    ):
        result = scheduler.paper_market_manual_recovery_status({})
    assert result["can_execute"] is True
    assert result["plan_id"] == "manual-7"



def test_restart_recovers_already_executed_manual_plan_without_new_order():
    run = _manual_run(status="manual_executing")
    db, runs, plans = _manual_db()
    plans.find_one.return_value = {
        "plan_id": "manual-7", "status": "executed",
        "execution_origin": "manual_recovery",
        "final_state": {"managed_symbol": "MAN", "holding_sessions": 1},
    }
    with (
        patch.object(scheduler, "_automation_document", return_value={"control_mode": "active"}),
        patch.object(scheduler, "_ensure_continuous_run", return_value=run),
        patch.object(scheduler, "utc_now", return_value=datetime(2026, 10, 7, 18, 0, tzinfo=timezone.utc)),
        patch.object(scheduler, "refresh_trader_live_market_data") as refresh,
        patch.object(scheduler, "execute_prepared_paper_plan") as execute,
    ):
        scheduler._advance_one(db, "worker-test")
    refresh.assert_not_called()
    execute.assert_not_called()
    assert runs.update_one.call_args.args[1]["$set"]["status"] == "completed"
    assert runs.update_one.call_args.args[1]["$unset"]["active_key"] == ""


def test_manual_reanalysis_persists_real_progress_even_without_today_run():
    context = _open_context(run=None, plan=None)
    prepared_plan = _man_manual_plan()
    progress_messages = []

    def simulate_preparation(db, *, progress_callback=None, **kwargs):
        assert progress_callback is not None
        for step in [
            ("download", 1, 77, "NVDA"),
            ("intraday_iex", 52, 75, "MAN"),
            ("intraday_sip", 75, 75, "AMS"),
            ("training", 1, 1, "LightGBM"),
            ("saving_plan", 1, 1, "manual-7"),
        ]:
            progress_callback(*step)
        return prepared_plan

    db, _, _ = _manual_db()
    automated = db[scheduler.PAPER_MARKET_AUTOMATION_COLLECTION]
    def record_progress(filter_doc, update_doc, **kwargs):
        progress_messages.append(update_doc["$set"]["manual_reanalysis"])

    automated.update_one.side_effect = record_progress
    with (
        patch.object(scheduler, "_manual_recovery_context", return_value=context),
        patch.object(scheduler, "prepare_next_paper_plan", side_effect=simulate_preparation),
        patch.object(scheduler, "_automation_document", return_value={"control_mode": "active"}),
        patch.object(scheduler, "_record_admin_operation"),
        patch.object(scheduler, "paper_market_manual_recovery_status", return_value={}),
    ):
        result = scheduler.prepare_manual_current_session_plan(db)
    assert result["status"] == "prepared"
    assert progress_messages[0]["status"] == "running"
    assert any(row["phase"] == "intraday_sip" and row["completed"] == 75
               for row in progress_messages)
    assert progress_messages[-1]["status"] == "prepared"
    assert progress_messages[-1]["plan_id"] == "manual-7"


def test_failed_manual_reanalysis_publishes_actual_data_coverage_error():
    db, _, _ = _manual_db()
    history = []
    automated = db[scheduler.PAPER_MARKET_AUTOMATION_COLLECTION]
    automated.update_one.side_effect = lambda selector, update, **kw: history.append(
        update["$set"]["manual_reanalysis"]
    )
    with (
        patch.object(scheduler, "_manual_recovery_context",
                     return_value=_open_context(run=None, plan=None)),
        patch.object(scheduler, "prepare_next_paper_plan",
                     side_effect=RuntimeError(
                         "CurrentSessionIntradayCoverageUnavailable: AMS IEX missing; SIP denied")),
    ):
        try:
            scheduler.prepare_manual_current_session_plan(db)
        except RuntimeError as exc:
            assert "AMS" in str(exc)
        else:
            raise AssertionError("Missing intraday bars must fail.")
    assert history[-1]["status"] == "failed"
    assert "AMS" in history[-1]["error"]


def test_manual_analysis_log_records_milestones_without_raw_errors():
    db, _, _ = _manual_db()
    automated = db[scheduler.PAPER_MARKET_AUTOMATION_COLLECTION]

    scheduler._record_manual_reanalysis_progress(
        db, attempt_id="audit-run-1", status="running",
        phase="intraday_iex", completed=10, total=75,
        symbol="AMS", log_event=True,
    )
    update = automated.update_one.call_args.args[1]
    assert update["$set"]["manual_reanalysis"]["total"] == 75
    entry = update["$push"]["manual_operation_log"]["$each"][0]
    assert update["$push"]["manual_operation_log"]["$slice"] == -90
    assert entry["operation"] == "analysis"
    assert entry["phase"] == "intraday_iex"
    assert entry["symbol"] == "AMS"
    assert entry["completed"] == 10
    assert entry["total"] == 75

    scheduler._record_manual_reanalysis_progress(
        db, attempt_id="audit-run-1", status="failed",
        phase="failed", error="broker private key: xxx",
        log_event=True,
    )
    update = automated.update_one.call_args.args[1]
    assert "broker private key" in update["$set"]["manual_reanalysis"]["error"]
    assert "error" not in update["$push"]["manual_operation_log"]["$each"][0]


def test_manual_execution_log_records_broker_reconciliation_without_private_details():
    db, _, _ = _manual_db()
    automated = db[scheduler.PAPER_MARKET_AUTOMATION_COLLECTION]
    scheduler._record_manual_execution_progress(
        db, attempt_id="execution-a", plan_id="plan-12",
        target_asset="MAN", status="review_required",
        phase="failed", error="broker secret=never-public",
    )
    update = automated.update_one.call_args.args[1]
    assert update["$set"]["manual_execution"]["status"] == "review_required"
    assert update["$push"]["manual_operation_log"]["$slice"] == -90
    event = update["$push"]["manual_operation_log"]["$each"][0]
    assert event["operation"] == "execution"
    assert event["symbol"] == "MAN"
    assert "error" not in event
    assert "secret" not in str(event)


def test_robot_public_status_contains_bounded_log_but_redacts_broker_error():
    db, _, _ = _manual_db()
    db[scheduler.STRATEGY_CONTROL_COLLECTION] = MagicMock()
    db[scheduler.STRATEGY_CONTROL_COLLECTION].find_one.return_value = {}
    runs = db[scheduler.PAPER_MARKET_RUNS_COLLECTION]
    runs.find_one.return_value = None
    controller = {
        "enabled": True, "status": "active", "control_mode": "active",
        "manual_reanalysis": {
            "status": "failed", "phase": "failed",
            "error": "CurrentSessionIntradayDataMissing: AMS has no IEX regular-session bars",
        },
        "manual_execution": {
            "status": "review_required", "phase": "failed",
            "error": "private credential=xxx",
        },
        "manual_operation_log": [
            {"at": pd.Timestamp("2026-10-14T17:01:00Z"),
             "operation": "analysis", "status": "failed",
             "phase": "failed", "symbol": "AMS"},
        ],
    }
    with (
        patch.object(scheduler, "_automation_document", return_value=controller),
        patch.object(scheduler, "get_trader_winner_summary", return_value=None),
        patch.object(scheduler._SCHEDULER, "snapshot", return_value={}),
    ):
        response = scheduler.paper_market_robot_status(db, public=True)
    assert response["manual_reanalysis"]["error"].startswith("AMS:")
    assert "credential" not in response["manual_execution"]["error"]
    assert response["manual_operation_log"][0]["operation"] == "analysis"


def test_close_old_paper_plans_uses_single_mongo_update_after_close() -> None:
    plans = MagicMock()
    plans.update_many.return_value.modified_count = 2
    db = {admin_trader.PAPER_TRADE_PLANS_COLLECTION: plans}
    # 21:16 UTC is 17:16 New York in October: regular session is closed.
    with (
        patch.object(admin_trader, "database", return_value=db),
        patch.object(
            admin_trader, "utc_now",
            return_value=datetime(2026, 10, 14, 21, 16, tzinfo=timezone.utc),
        ),
    ):
        result = admin_trader.close_old_paper_plans(SimpleNamespace(email="admin@example.com"))
    assert result == {
        "status": "completed", "closed_count": 2,
        "closed_through_session": "2026-10-14",
    }
    plans.update_many.assert_called_once()
    filters, update = plans.update_many.call_args.args
    assert filters == {
        "status": "prepared", "execution_session": {"$lte": "2026-10-14"},
    }
    assert update["$set"]["status"] == "cancelled"
    assert update["$set"]["cancel_reason"]


def test_close_old_paper_plans_preserves_current_session_before_close() -> None:
    plans = MagicMock()
    plans.update_many.return_value.modified_count = 0
    with (
        patch.object(
            admin_trader, "database",
            return_value={admin_trader.PAPER_TRADE_PLANS_COLLECTION: plans},
        ),
        patch.object(
            admin_trader, "utc_now",
            return_value=datetime(2026, 10, 14, 18, 16, tzinfo=timezone.utc),
        ),
    ):
        result = admin_trader.close_old_paper_plans(SimpleNamespace(email="admin@example.com"))
    assert result["closed_through_session"] == "2026-10-13"
    assert result["closed_count"] == 0
    assert plans.update_many.call_args.args[0]["execution_session"] == {
        "$lte": "2026-10-13",
    }
