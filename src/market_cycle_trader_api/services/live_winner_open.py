"""Independent Live market-open Winner recalibration before real orders.

Never import or invoke Paper services, Paper ledgers or Paper scheduling.
The completed prior-session history is augmented with fresh Live-only bars.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from time import monotonic
from typing import Any
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd
from pymongo import ReturnDocument

from ..core.config import TCC_U67_CONTROL_OPERATIONAL_MODE, TCC_U77_FIXED_CHALLENGER_MODE
from ..engine.market_data import latest_safe_completed_xnys_session
from ..engine.live_model_signal import build_live_model_decision
from ..engine.operational_market_snapshot import (
    DATA_DIRECTORY as MARKET_DATA_DIRECTORY,
    download_operational_market_snapshot,
)
from ..engine.tcc_u67_operational_runtime import (
    assert_tcc_u67_model_snapshot, assert_tcc_u67_operational_contract,
)
from ..engine.tcc_u77_static_challenger import U77_REQUESTED_ASSETS, assert_tcc_u77_contract
from ..infrastructure.trading.alpaca_live import create_live_readonly_client, live_credentials
from .live_control import live_control_status
from .live_execution_window import live_execution_window_deadline
from .live_market_bars import load_live_intraday_session
from .live_winner_candidates import winner_candidate_fields
from .live_winner_events import append_live_event
from .live_winner_session import LIVE_RUNS
from .live_trader_state import verified_live_state
from .model_research import apply_execution_profile
from .strategy_lab import get_trader_winner_context, get_trader_winner_model_snapshot
from .system_settings import apply_training_runtime_settings, get_system_settings

EASTERN = ZoneInfo("America/New_York")
_OPEN_MIN_BARS_SECONDS = 180
_OPEN_RETRY_SECONDS = 90
_OPEN_MAX_ATTEMPTS = 12
_ELIGIBLE_STATUSES = ("waiting_for_winner_signal", "recalibrated_no_orders", "waiting_for_open_recalibration")


def _utc(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def refresh_live_open_winner_signal(db: Any, session: str) -> dict[str, Any]:
    """Refresh the same Live session from its last closed bar and today's minutes.

    Each session can become executable only after this completed operation.
    """
    if live_control_status(db)["control_mode"] not in {"active", "exit_only"}:
        return {"status": "blocked_by_live_mode", "order_submission_enabled": False}
    run = db[LIVE_RUNS].find_one({"execution_session": session})
    if not run:
        return {"status": "no_live_session", "order_submission_enabled": False}
    if run.get("status") in {"completed", "executing", "analysis_only_completed"}:
        return {"status": str(run["status"]), "order_submission_enabled": False}
    if run.get("status") == "open_recalibration":
        started_at = run.get("open_calibration_started_at")
        if started_at is None or _utc(datetime.now(timezone.utc)) - _utc(started_at) <= pd.Timedelta(minutes=25):
            return {"status": "live_open_recalibration_in_progress", "order_submission_enabled": False}
        # Recover a worker interrupted after claiming the model task. Open
        # calibration never submits orders; still verify the independent
        # broker ledger before releasing the stale claim.
        pending_intent = db["live_winner_broker_orders"].find_one({
            "broker_account_id": run["live_account_id"],
            "execution_session": session,
        })
        if pending_intent:
            db[LIVE_RUNS].update_one(
                {"_id": run["_id"], "status": "open_recalibration"},
                {"$set": {"status": "review_required",
                          "phase": "live_open_broker_reconciliation_required",
                          "order_submission_enabled": False}},
            )
            return {"status": "broker_reconciliation_required", "order_submission_enabled": False}
        recovered = db[LIVE_RUNS].find_one_and_update(
            {"_id": run["_id"], "status": "open_recalibration",
             "open_calibration_started_at": started_at},
            {"$set": {"status": "waiting_for_open_recalibration",
                      "phase": "recovered_abandoned_live_calibration",
                      "winner_signal_prepared": False,
                      "post_close_signal_ready": False,
                      "order_submission_enabled": False}},
            return_document=ReturnDocument.AFTER,
        )
        return {"status": "recovered_abandoned_live_calibration"
                if recovered else "live_open_recalibration_in_progress",
                "order_submission_enabled": False}
    calendar = xcals.get_calendar("XNYS")
    requested = pd.Timestamp(session)
    if not calendar.is_session(requested):
        raise RuntimeError("Live execution date is not a regular XNYS trading session.")
    expected_cutoff = calendar.previous_session(requested).date().isoformat()
    broker = create_live_readonly_client()
    account = broker.get_account()
    if str(getattr(account, "id", "") or "") != run.get("live_account_id"):
        raise RuntimeError("Live account identity mismatch before open recalibration.")
    clock = broker.get_clock()
    timestamp = getattr(clock, "timestamp", None)
    if timestamp is None or not bool(getattr(clock, "is_open", False)):
        return {"status": "waiting_for_live_open", "order_submission_enabled": False}
    stamp = _utc(timestamp)
    if stamp.tz_convert(EASTERN).date().isoformat() != session:
        raise RuntimeError("Live open-session clock does not match the execution session.")
    opened = _utc(run["expected_market_open"])
    deadline, _ = live_execution_window_deadline(opened)
    if stamp >= deadline:
        db[LIVE_RUNS].update_one(
            {"_id": run["_id"], "status": {"$in": _ELIGIBLE_STATUSES}},
            {"$set": {"status": "review_required",
                      "phase": "live_open_window_expired",
                      "order_submission_enabled": False,
                      "updated_at": datetime.now(timezone.utc)}},
        )
        return {"status": "execution_window_expired", "order_submission_enabled": False}
    if stamp < opened + pd.Timedelta(seconds=_OPEN_MIN_BARS_SECONDS):
        return {"status": "waiting_for_completed_live_minutes", "order_submission_enabled": False}
    if latest_safe_completed_xnys_session().date().isoformat() != expected_cutoff:
        return {"status": "waiting_for_correct_prior_daily_bar", "order_submission_enabled": False}
    if (run.get("status") == "recalibrated_no_orders"
        and run.get("market_open_calibration_completed_at")
        and str(run.get("decision_date") or "")[:10] == session
        and str(run.get("historical_cutoff") or "") == expected_cutoff
        and run.get("analysis_mode") == "live_current_session_intraday"):
        return {"status": "recalibrated_no_orders", "order_submission_enabled": False}
    if run.get("status") not in _ELIGIBLE_STATUSES:
        return {"status": "not_eligible_for_live_open_refresh", "order_submission_enabled": False}
    retry_at = run.get("open_retry_not_before")
    if retry_at is not None and stamp < _utc(retry_at):
        return {"status": "waiting_for_open_retry", "order_submission_enabled": False}
    if int(run.get("open_preparation_attempts") or 0) >= _OPEN_MAX_ATTEMPTS:
        db[LIVE_RUNS].update_one(
            {"_id": run["_id"], "status": {"$in": _ELIGIBLE_STATUSES}},
            {"$set": {"status": "review_required", "phase": "open_recalibration_attempts_exhausted",
                      "order_submission_enabled": False}},
        )
        return {"status": "open_recalibration_attempts_exhausted", "order_submission_enabled": False}
    if not bool(get_system_settings(db)["training"]["enabled"]):
        return {"status": "live_training_disabled", "order_submission_enabled": False}
    config, winner = get_trader_winner_context(db)
    model = get_trader_winner_model_snapshot(db)
    if (str(winner["id"]) != run["winner_strategy_id"]
        or int(winner["revision"]) != int(run["winner_revision"])
        or str(winner["configuration_hash"]) != run["winner_configuration_hash"]):
        raise RuntimeError("Winner identity changed; the Live order plan cannot be refreshed.")
    strategy = apply_training_runtime_settings(db, config)
    strategy = apply_execution_profile(strategy, model["family"], model.get("settings_snapshot") or {})
    strategy = strategy.model_copy(update={
        "research_model_family": model["family"],
        "research_model_settings": model.get("settings_snapshot") or {},
    })
    mode = str(strategy.strategy_mode)
    if model["family"] != "lightgbm_utility" or mode not in {
        TCC_U67_CONTROL_OPERATIONAL_MODE, TCC_U77_FIXED_CHALLENGER_MODE,
    }:
        raise RuntimeError("Live Winner model is not the protected U67/U77 LightGBM.")
    if mode == TCC_U77_FIXED_CHALLENGER_MODE:
        assert_tcc_u77_contract(strategy)
    else:
        assert_tcc_u67_operational_contract(strategy)
    assert_tcc_u67_model_snapshot(model)
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    if broker.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=100)):
        raise RuntimeError("Live broker has open orders; independent reconciliation required.")
    state = verified_live_state(db, broker, session=session)

    started_at = datetime.now(timezone.utc)
    claimed = db[LIVE_RUNS].find_one_and_update(
        {"_id": run["_id"], "status": {"$in": _ELIGIBLE_STATUSES},
         "winner_configuration_hash": run["winner_configuration_hash"],
         "market_open_calibration_completed_at": {"$exists": False},
         "$or": [{"open_preparation_attempts": {"$exists": False}},
                 {"open_preparation_attempts": {"$lt": _OPEN_MAX_ATTEMPTS}}]},
        {"$set": {
            "status": "open_recalibration", "phase": "refreshing_live_intraday_market_data",
            "winner_signal_prepared": False, "post_close_signal_ready": False,
            "order_submission_enabled": False, "open_calibration_started_at": started_at,
            "previous_post_close_decision_date": run.get("decision_date"),
            "previous_post_close_target_asset": run.get("target_asset"),
            "previous_post_close_snapshot_sha256": run.get("snapshot_sha256"),
        }, "$inc": {"open_preparation_attempts": 1}},
        return_document=ReturnDocument.AFTER,
    )
    if not claimed:
        return {"status": "open_recalibration_claimed_elsewhere", "order_submission_enabled": False}
    started = monotonic()
    stage = "daily_history"
    try:
        snapshot = download_operational_market_snapshot(
            job_id=f"live-open-{session}-{started_at.strftime('%Y%m%d%H%M%S%f')}",
            config=strategy.model_copy(update={"analysis_end_date": expected_cutoff}),
            requested_assets=(U77_REQUESTED_ASSETS if mode == TCC_U77_FIXED_CHALLENGER_MODE
                              else tuple(strategy.assets)),
            data_directory=MARKET_DATA_DIRECTORY.parent.parent / "live_winner" / "snapshots",
            credentials_override=dict(zip(("api_key_id", "secret_key"), live_credentials())),
        )
        if not snapshot.frames or any(frame.empty for frame in snapshot.frames.values()):
            raise RuntimeError("Live daily history snapshot is incomplete.")
        if not snapshot.manifest.get("snapshot_sha256"):
            raise RuntimeError("Live daily history snapshot has no content hash.")
        stage = "current_session_bars"
        # Downloading 77 daily histories can take several minutes. Obtain a
        # *fresh* broker time for the current-session as-of cutoff.
        intraday_clock = broker.get_clock()
        if not bool(getattr(intraday_clock, "is_open", False)):
            raise RuntimeError("LiveIntradayCoverageNotReady: session closed during daily download.")
        merged, intraday = load_live_intraday_session(
            snapshot.frames, strategy, session=session, now=intraday_clock.timestamp,
        )
        stage = "lightgbm_decision"
        decision = build_live_model_decision(
            merged, strategy, model_family=model["family"],
            current_asset=state["managed_symbol"], holding_sessions=state["holding_sessions"],
        )
        date = str(decision.decision_date)[:10]
        if date != session:
            raise RuntimeError(
                f"Live intraday decision date mismatch: actual={date}, expected={session}."
            )
        target = str(decision.target_asset).upper()
        if not target or (target != "CASH" and target not in strategy.assets):
            raise RuntimeError("Live Winner intraday target not in the locked universe.")
        current = state["managed_symbol"]
        action = ("stay_in_cash" if target == current == "CASH" else
                  "hold" if target == current else
                  "buy" if current == "CASH" else
                  "sell_to_cash" if target == "CASH" else "rotate")
        # Market-open calibration may take minutes: never record an executable
        # result for another session or after the broker execution deadline.
        finished_clock = broker.get_clock()
        finished_time = _utc(finished_clock.timestamp)
        if (not getattr(finished_clock, "is_open", False)
            or finished_time.tz_convert(EASTERN).date().isoformat() != session
            or finished_time >= deadline):
            raise RuntimeError("LiveOpenRecalibrationMissedWindow: no orders allowed.")
        saved = db[LIVE_RUNS].find_one_and_update(
            {"_id": claimed["_id"], "status": "open_recalibration",
             "winner_configuration_hash": run["winner_configuration_hash"]},
            {"$set": {
                "status": "recalibrated_no_orders", "phase": "live_intraday_signal_ready_for_execution",
                "decision_date": session, "historical_cutoff": expected_cutoff,
                "analysis_mode": "live_current_session_intraday",
                "market_open_calibration_completed_at": datetime.now(timezone.utc),
                "target_asset": target, "action": action, "current_asset": current,
                "current_quantity": state["managed_quantity"],
                "live_state_revision": state["revision"],
                "session_equity_baseline": float(getattr(account, "equity", 0) or 0),
                "winner_signal_prepared": True, "post_close_signal_ready": True,
                "snapshot_sha256": snapshot.manifest["snapshot_sha256"],
                "order_submission_enabled": False,
                **intraday, **winner_candidate_fields(decision),
            }, "$unset": {"open_retry_not_before": "", "last_open_calibration_error": ""}},
            return_document=ReturnDocument.AFTER,
        )
        if not saved:
            raise RuntimeError("Live session changed while open recalibration was running.")
        append_live_event(
            db, account_id=run["live_account_id"], session=session,
            stage="live_open_signal_prepared",
            message="Live Winner reanalisada com o ultimo pregão completo e barras da sessao atual.",
            details={"decision_date": session, "historical_cutoff": expected_cutoff,
                     "target_asset": target, "action": action,
                     "duration_seconds": round(monotonic() - started, 2)},
        )
        return {"status": "recalibrated_no_orders", "execution_session": session,
                "decision_date": session, "target_asset": target,
                "action": action, "order_submission_enabled": False}
    except Exception as exc:
        attempts = int(claimed.get("open_preparation_attempts") or
                       int(run.get("open_preparation_attempts") or 0) + 1)
        retry = attempts < _OPEN_MAX_ATTEMPTS and not str(exc).startswith(
            "LiveOpenRecalibrationMissedWindow:"
        )
        db[LIVE_RUNS].update_one(
            {"_id": claimed["_id"], "status": "open_recalibration"},
            {"$set": {
                "status": "waiting_for_open_recalibration" if retry else "review_required",
                "phase": "waiting_for_open_data_retry" if retry else "open_recalibration_failed",
                "order_submission_enabled": False, "winner_signal_prepared": False,
                "post_close_signal_ready": False, "last_open_calibration_error": str(exc),
                **({"open_retry_not_before": datetime.now(timezone.utc) +
                     timedelta(seconds=_OPEN_RETRY_SECONDS)} if retry else {}),
            }},
        )
        append_live_event(
            db, account_id=run["live_account_id"], session=session,
            stage="live_open_signal_failed", level="error",
            message="Live reanalise de abertura falhou; aguarda recuperacao automatica.",
            details={"error_type": type(exc).__name__, "stage": stage,
                     "retry_allowed": retry, "duration_seconds": round(monotonic() - started, 2)},
        )
        return {"status": "waiting_for_open_retry" if retry else "review_required",
                "error_type": type(exc).__name__, "order_submission_enabled": False}
