"""Independent, read-only Winner signal calculation for a CASH-only Live account.

Reuses the exact operational Winner model functions; never reads Paper positions,
never creates Paper plans and never submits broker orders. Unsupported account
states fail closed instead of silently borrowing Paper state.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from time import monotonic
from typing import Any

import exchange_calendars as xcals
import pandas as pd
from pymongo import ReturnDocument

from ..core.config import TCC_U67_CONTROL_OPERATIONAL_MODE, TCC_U77_FIXED_CHALLENGER_MODE
from ..engine.live_model_signal import build_live_model_decision
from ..engine.market_data import latest_safe_completed_xnys_session
from ..engine.operational_market_snapshot import (
    DATA_DIRECTORY as TCC_U67_DATA_DIRECTORY,
    download_operational_market_snapshot,
)
from ..engine.tcc_u67_operational_runtime import assert_tcc_u67_model_snapshot, assert_tcc_u67_operational_contract
from ..engine.tcc_u77_static_challenger import U77_REQUESTED_ASSETS, assert_tcc_u77_contract
from ..infrastructure.trading.alpaca_live import create_live_readonly_client
from .model_research import apply_execution_profile
from .system_settings import apply_training_runtime_settings, get_system_settings
from .strategy_lab import get_trader_winner_context, get_trader_winner_model_snapshot
from .live_winner_session import LIVE_RUNS
from .live_winner_events import append_live_event
from .live_winner_candidates import winner_candidate_fields
from .live_trader_state import verified_live_state

_ALLOWED = frozenset({TCC_U67_CONTROL_OPERATIONAL_MODE, TCC_U77_FIXED_CHALLENGER_MODE})

def _utc_datetime(value: Any) -> datetime:
    stamp = value
    if stamp.tzinfo is None:
        # PyMongo decodes UTC timestamps as naive unless tz_aware was enabled.
        return stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc)



def prepare_live_winner_signal(db: Any, session: str) -> dict[str, Any]:
    """Prepare a non-executable Winner signal for a flat Live account.

    This entry point is not a trading or rebalancing function. Any existing
    position or pending order requires a separate reconciled Live executor.
    """
    run = db[LIVE_RUNS].find_one({"execution_session": session})
    if not run:
        raise RuntimeError("No Live session is armed.")
    calendar = xcals.get_calendar("XNYS")
    if not calendar.is_session(pd.Timestamp(session)):
        raise RuntimeError("Live execution session is not an XNYS trading day.")
    required_day = calendar.previous_session(pd.Timestamp(session)).date().isoformat()
    # A completed plan already stamped with its correct decision date needs
    # no fresh clock lookup. Invalid states must remain non-executable.
    if run.get("status") == "signal_prepared":
        return {"status": "signal_prepared", "execution_session": session,
                "target_asset": run.get("target_asset"), "order_submission_enabled": False}
    if (run.get("status") == "recalibrated_no_orders"
        and str(run.get("decision_date") or "")[:10] == required_day):
        return {"status": "recalibrated_no_orders", "execution_session": session,
                "target_asset": run.get("target_asset"), "order_submission_enabled": False}
    if run.get("status") not in {
        "waiting_for_winner_signal", "preparing_signal", "recalibrated_no_orders"
    }:
        raise RuntimeError("Live session is not eligible for Winner preparation.")
    # Use the actual Alpaca Live broker timestamp, not Railway's clock or timezone.
    # No fixed 20-minute post-close delay: actual daily data availability and
    # the computed decision_date determine when the signal is usable.
    broker = create_live_readonly_client()
    clock = broker.get_clock()
    broker_time = _utc_datetime(clock.timestamp)
    available_day = latest_safe_completed_xnys_session(
        now=broker_time, data_delay_minutes=0
    ).date().isoformat()
    if run.get("status") == "preparing_signal":
        # A long-running model is not an abandoned worker. Do not reclaim a
        # persisted claim based on an arbitrary number of minutes.
        return {"status": "post_close_training_in_progress",
                "execution_session": session, "order_submission_enabled": False}
    # Never freeze Friday history as Tuesday's decision. Wait for Monday's
    # completed daily bar and replace any unexecuted stale stored signal.
    if run.get("status") == "signal_prepared":
        return {"status": "signal_prepared", "execution_session": session,
                "target_asset": run.get("target_asset"), "order_submission_enabled": False}
    if run.get("status") == "recalibrated_no_orders":
        if str(run.get("decision_date") or "")[:10] == required_day:
            return {"status": "recalibrated_no_orders", "execution_session": session,
                    "target_asset": run.get("target_asset"), "order_submission_enabled": False}
        if available_day != required_day:
            return {"status": "waiting_for_completed_daily_bar",
                    "execution_session": session, "required_day": required_day,
                    "available_day": available_day, "order_submission_enabled": False}
        refreshed = db[LIVE_RUNS].find_one_and_update(
            {"_id": run["_id"], "status": run["status"],
             "winner_configuration_hash": run["winner_configuration_hash"]},
            {"$set": {
                "status": "waiting_for_winner_signal",
                "phase": "replacing_stale_post_close_signal",
                "superseded_decision_date": run.get("decision_date"),
                "superseded_target_asset": run.get("target_asset"),
                "superseded_snapshot_sha256": run.get("snapshot_sha256"),
                "preparation_attempts": 0, "winner_signal_prepared": False,
                "post_close_signal_ready": False, "order_submission_enabled": False,
                "decision_date": None, "action": None, "target_asset": None,
                "snapshot_sha256": None,
            }, "$unset": {"preparation_retry_not_before": ""}},
            return_document=ReturnDocument.AFTER,
        )
        if not refreshed:
            return {"status": "stale_signal_replacement_claimed_elsewhere",
                    "order_submission_enabled": False}
        run = refreshed
    if run.get("status") != "waiting_for_winner_signal":
        raise RuntimeError("Live session is not eligible for Winner preparation.")
    attempts = int(run.get("preparation_attempts") or 0)
    if available_day != required_day:
        return {"status": "waiting_for_completed_daily_bar",
                "execution_session": session, "required_day": required_day,
                "available_day": available_day, "order_submission_enabled": False}
    if not bool(get_system_settings(db)["training"]["enabled"]):
        raise RuntimeError("Winner training is disabled in system settings.")

    config, profile = get_trader_winner_context(db)
    model = get_trader_winner_model_snapshot(db)
    if (str(profile["id"]) != run["winner_strategy_id"] or
        int(profile["revision"]) != run["winner_revision"] or
        str(profile["configuration_hash"]) != run["winner_configuration_hash"]):
        raise RuntimeError("Winner identity changed since Live session was armed.")
    strategy = apply_training_runtime_settings(db, config)
    strategy = apply_execution_profile(
        strategy, model["family"], model.get("settings_snapshot") or {}
    )
    strategy = strategy.model_copy(update={
        "research_model_family": model["family"],
        "research_model_settings": model.get("settings_snapshot") or {},
    })
    if str(strategy.strategy_mode) not in _ALLOWED or model["family"] != "lightgbm_utility":
        raise RuntimeError("Live Winner is not supported by this locked execution profile.")
    if str(strategy.strategy_mode) == TCC_U77_FIXED_CHALLENGER_MODE:
        assert_tcc_u77_contract(strategy)
    else:
        assert_tcc_u67_operational_contract(strategy)
    assert_tcc_u67_model_snapshot(model)

    account = broker.get_account()
    if str(getattr(account, "id", "") or "") != run["live_account_id"]:
        raise RuntimeError("Live account identity changed.")
    clock = broker.get_clock()
    if bool(getattr(clock, "is_open", True)):
        raise RuntimeError("Winner preparation requires closed regular market.")
    state = verified_live_state(db, broker, session=session)
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    if broker.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=100)):
        raise RuntimeError("Outstanding Live orders require reconciliation.")

    now = broker_time
    claimed = db[LIVE_RUNS].find_one_and_update(
        {"_id": f'{run["live_account_id"]}:{session}',
         "status": "waiting_for_winner_signal",
        },
        {"$set": {"status": "preparing_signal", "phase": "post_close_winner_training",
                  "preparation_started_at": now, "order_submission_enabled": False,
                  "winner_signal_prepared": False, "post_close_signal_ready": False,
                  "decision_date": None, "target_asset": None, "action": None,
                  "snapshot_sha256": None},
         "$inc": {"preparation_attempts": 1}},
        return_document=ReturnDocument.AFTER,
    )
    if not claimed:
        raise RuntimeError("Live preparation was already claimed.")

    started = monotonic()
    stage = "initial_event"
    try:
        append_live_event(db, account_id=run["live_account_id"], session=session,
                          stage="signal_preparation_started", message="Preparacao do sinal Winner iniciada.")
        stage = "download"
        cutoff = latest_safe_completed_xnys_session(
            now=_utc_datetime(broker.get_clock().timestamp), data_delay_minutes=0
        ).date().isoformat()
        if cutoff != required_day:
            raise RuntimeError("LiveDailyCutoffChanged: safe prior-session date changed.")
        u77 = str(strategy.strategy_mode) == TCC_U77_FIXED_CHALLENGER_MODE
        snapshot = download_operational_market_snapshot(
            job_id=f"live-{session}-{now.strftime('%Y%m%d%H%M%S%f')}",
            config=strategy.model_copy(update={"analysis_end_date": cutoff}),
            requested_assets=(U77_REQUESTED_ASSETS if u77 else tuple(strategy.assets)),
            data_directory=TCC_U67_DATA_DIRECTORY.parent.parent / "live_winner" / "snapshots",
        )
        if not snapshot.frames or any(frame.empty for frame in snapshot.frames.values()):
            raise RuntimeError("Incomplete Alpaca snapshot for Live Winner.")
        if not isinstance(snapshot.manifest.get("snapshot_sha256"), str) or not snapshot.manifest["snapshot_sha256"]:
            raise RuntimeError("Missing immutable post-close Alpaca snapshot SHA-256.")
        append_live_event(db, account_id=run["live_account_id"], session=session,
                          stage="signal_snapshot_downloaded", message="Snapshot do sinal preparado.",
                          details={"assets": len(snapshot.frames), "duration_seconds": round(monotonic() - started, 2)})
        stage = "model_training_and_decision"
        append_live_event(db, account_id=run["live_account_id"], session=session,
                          stage="signal_training_started", message="LightGBM: treinamento do sinal iniciado.")
        decision = build_live_model_decision(
            snapshot.frames, strategy, model_family=model["family"],
            current_asset=state["managed_symbol"], holding_sessions=state["holding_sessions"],
        )
        append_live_event(db, account_id=run["live_account_id"], session=session,
                          stage="signal_decision_computed", message="Sinal Winner calculado.",
                          details={"duration_seconds": round(monotonic() - started, 2)})
        stage = "decision_validation"
        decision_date = str(decision.decision_date)[:10]
        if decision_date != cutoff or decision_date != required_day:
            raise RuntimeError(
                f"Winner cutoff mismatch: decision={decision_date}, expected={cutoff}, next={session}"
            )
        target = str(decision.target_asset).upper()
        if not target or (target != "CASH" and target not in strategy.assets):
            raise RuntimeError("Winner signal target is not a permitted asset.")
        candidate_fields = winner_candidate_fields(decision)
        current = state["managed_symbol"]
        action = ("stay_in_cash" if target == current == "CASH" else
                  "hold" if target == current else
                  "buy" if current == "CASH" else
                  "sell_to_cash" if target == "CASH" else "rotate")
        stage = "decision_persistence"
        # If preparation spilled into the open session, the next-session plan
        # is no longer an after-close decision. Fail without releasing orders.
        completed_clock = broker.get_clock()
        if bool(getattr(completed_clock, "is_open", False)):
            raise RuntimeError("PostCloseSignalMissedWindow: market opened during preparation.")
        saved = db[LIVE_RUNS].find_one_and_update(
            {"_id": claimed["_id"], "status": "preparing_signal",
             "winner_configuration_hash": run["winner_configuration_hash"]},
            {"$set": {"status": "recalibrated_no_orders",
                      "phase": "post_close_signal_ready_for_execution",
                      "post_close_signal_ready": True,
                      "decision_date": decision_date,
                      "target_asset": target,
                      **candidate_fields,
                      "action": action,
                      "current_asset": current,
                      "current_quantity": state["managed_quantity"],
                      "live_state_revision": state["revision"],
                      "session_equity_baseline": float(getattr(account, "equity", 0) or 0),
                      "winner_signal_prepared": True,
                      "snapshot_sha256": snapshot.manifest.get("snapshot_sha256"),
                      "signal_prepared_at": datetime.now(timezone.utc),
                      "order_submission_enabled": False},
             "$unset": {"preparation_retry_not_before": "",
                        "last_preparation_error_type": ""}},
            return_document=ReturnDocument.AFTER,
        )
        if not saved:
            raise RuntimeError("Live Winner plan changed during preparation.")
        append_live_event(db, account_id=run["live_account_id"], session=session,
                          stage="signal_prepared", message="Sinal Winner registrado.",
                          details={"target_asset": target, "action": action,
                                   "duration_seconds": round(monotonic() - started, 2)})
        return {"status": "recalibrated_no_orders", "execution_session": session,
                "decision_date": decision_date, "target_asset": target,
                "action": saved["action"], "order_submission_enabled": False}
    except Exception as exc:
        # Persist the recoverable state before emitting secondary audit events.
        # A logging failure must not strand a claimed job in preparing_signal.
        updates: dict[str, Any] = {
            "status": "waiting_for_winner_signal",
            "phase": "post_close_signal_retry_wait",
            "failure_type": type(exc).__name__,
            "last_preparation_error_type": type(exc).__name__,
            "post_close_signal_ready": False,
            "winner_signal_prepared": False,
            "decision_date": None, "target_asset": None, "action": None,
            "snapshot_sha256": None,
            "order_submission_enabled": False,
        }
        db[LIVE_RUNS].update_one(
            {"_id": claimed["_id"], "status": "preparing_signal"},
            {"$set": updates, "$unset": {"preparation_retry_not_before": ""}},
        )
        try:
            append_live_event(db, account_id=run["live_account_id"], session=session,
                              stage="signal_preparation_failed", level="error",
                              message="Falha no sinal Winner durante: " + stage,
                              details={"error_type": type(exc).__name__,
                                       "duration_seconds": round(monotonic() - started, 2)})
        except Exception:
            # Recovery state is authoritative; preserve the original failure.
            pass
        raise


# Compatibility export for callers predating the independent Live pipeline.
prepare_live_cash_winner_signal = prepare_live_winner_signal
