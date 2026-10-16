"""Open-session Winner recalibration for a flat Live account, without order execution."""
from __future__ import annotations

from datetime import datetime, timezone
from time import monotonic
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from pymongo import ReturnDocument

from ..core.config import TCC_U67_CONTROL_OPERATIONAL_MODE, TCC_U77_FIXED_CHALLENGER_MODE
from ..engine.live_model_signal import build_live_model_decision
from ..engine.market_data import latest_safe_completed_xnys_session
from ..engine.tcc_u67_operational_market_data import (
    DATA_DIRECTORY as TCC_U67_DATA_DIRECTORY,
    append_u67_current_session_intraday,
    download_u67_operational_snapshot,
)
from ..engine.tcc_u67_operational_runtime import assert_tcc_u67_model_snapshot, assert_tcc_u67_operational_contract
from ..engine.tcc_u77_static_challenger import U77_REQUESTED_ASSETS, assert_tcc_u77_contract
from ..infrastructure.trading.alpaca_live import create_live_readonly_client
from ..infrastructure.persistence.mongo_repository import get_paper_trading_settings
from ..schemas.paper_trading import PaperTradingSettings
from .live_winner_session import LIVE_RUNS
from .live_execution_window import live_execution_window_deadline
from .live_winner_events import append_live_event
from .live_winner_candidates import winner_candidate_fields
from .live_trader_state import verified_live_state
from .model_research import apply_execution_profile
from .system_settings import apply_training_runtime_settings
from .strategy_lab import get_trader_winner_context, get_trader_winner_model_snapshot

EASTERN = ZoneInfo("America/New_York")


def recalibrate_live_cash_winner_signal(db: Any, session: str) -> dict[str, Any]:
    """Recompute the same Winner model with current-session bars.

    Requires zero Live holdings and a previously completed post-close signal.
    Every exception leaves real-order execution disabled.
    """
    run = db[LIVE_RUNS].find_one({"execution_session": session})
    if not run:
        raise RuntimeError("No armed Live session.")
    if run["status"] in {"recalibrated_no_orders", "analysis_only_completed"}:
        return {"status": run["status"], "target_asset": run.get("target_asset"),
                "order_submission_enabled": False}
    analysis_only = run["status"] == "analysis_only_requested"
    if run["status"] not in {"signal_prepared", "analysis_only_requested"}:
        raise RuntimeError("Live signal is not ready for market-open recalibration.")
    broker = create_live_readonly_client()
    account = broker.get_account()
    if str(getattr(account, "id", "") or "") != run["live_account_id"]:
        raise RuntimeError("Live account ID changed.")
    clock = broker.get_clock()
    if not bool(getattr(clock, "is_open", False)):
        return {"status": "waiting_for_market_open", "order_submission_enabled": False}
    now = pd.Timestamp(clock.timestamp)
    if now.tzinfo is None:
        raise RuntimeError("Broker clock has no timezone.")
    now = now.tz_convert("UTC")
    expected_open = pd.Timestamp(run["expected_market_open"])
    expected_open = expected_open.tz_localize("UTC") if expected_open.tzinfo is None else expected_open.tz_convert("UTC")
    if now.tz_convert(EASTERN).date().isoformat() != session:
        raise RuntimeError("Live session is stale.")
    settings = PaperTradingSettings.model_validate(get_paper_trading_settings(db))
    latest, live_window_seconds = live_execution_window_deadline(expected_open, settings)
    if now > latest and not analysis_only:
        db[LIVE_RUNS].update_one({"_id": run["_id"], "status": "signal_prepared"},
                                {"$set": {"status": "review_required", "phase": "execution_window_expired",
                                          "window_deadline_at": latest.to_pydatetime(),
                                          "window_expired_at": now.to_pydatetime(),
                                          "window_seconds": live_window_seconds,
                                          "order_submission_enabled": False}})
        raise RuntimeError("Live open recalibration window expired.")
    state = verified_live_state(db, broker, session=session)
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    if broker.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=100)):
        raise RuntimeError("Live broker has outstanding orders.")

    profile_configuration, profile = get_trader_winner_context(db)
    model = get_trader_winner_model_snapshot(db)
    if (str(profile["id"]) != run["winner_strategy_id"]
        or int(profile["revision"]) != run["winner_revision"]
        or str(profile["configuration_hash"]) != run["winner_configuration_hash"]):
        raise RuntimeError("Winner identity changed since Live session was armed.")
    strategy = apply_training_runtime_settings(db, profile_configuration)
    strategy = apply_execution_profile(
        strategy, model["family"], model.get("settings_snapshot") or {}
    )
    strategy = strategy.model_copy(update={
        "research_model_family": model["family"],
        "research_model_settings": model.get("settings_snapshot") or {},
    })
    if str(strategy.strategy_mode) not in {TCC_U67_CONTROL_OPERATIONAL_MODE, TCC_U77_FIXED_CHALLENGER_MODE}:
        raise RuntimeError("Unsupported operational Winner strategy.")
    if model["family"] != "lightgbm_utility":
        raise RuntimeError("Unsupported Live Winner model.")
    u77 = str(strategy.strategy_mode) == TCC_U77_FIXED_CHALLENGER_MODE
    if u77:
        assert_tcc_u77_contract(strategy)
    else:
        assert_tcc_u67_operational_contract(strategy)
    assert_tcc_u67_model_snapshot(model)
    recalibrating_status = "analysis_only_recalibrating" if analysis_only else "recalibrating"
    recalibration_started_at = datetime.now(timezone.utc)
    claimed = db[LIVE_RUNS].find_one_and_update(
        {"_id": run["_id"], "status": run["status"]},
        {"$set": {"status": recalibrating_status,
                  "phase": "late_intraday_analysis" if analysis_only else "market_open_recalibration",
                  "calibration_started_at": recalibration_started_at,
                  "window_seconds": live_window_seconds,
                  "window_deadline_at": latest.to_pydatetime(),
                  "order_submission_enabled": False}},
        return_document=ReturnDocument.AFTER,
    )
    if claimed is None:
        raise RuntimeError("Live open recalibration already claimed.")
    started = monotonic()
    stage = "download"
    append_live_event(db, account_id=run["live_account_id"], session=session,
                      stage="recalibration_started", message="Recalibracao intraday iniciada.",
                      details={"cutoff": session})
    try:
        cutoff = latest_safe_completed_xnys_session().date().isoformat()
        snapshot = download_u67_operational_snapshot(
            job_id=f"live-open-{session}-{datetime.now(timezone.utc).strftime('%H%M%S%f')}",
            config=strategy.model_copy(update={"analysis_end_date": cutoff}),
            requested_assets=(U77_REQUESTED_ASSETS if u77 else tuple(strategy.assets)),
            data_directory=TCC_U67_DATA_DIRECTORY.parent.parent / "live_winner" / "snapshots",
        )
        if not snapshot.frames or any(frame.empty for frame in snapshot.frames.values()):
            raise RuntimeError("Incomplete Live Winner snapshot.")
        append_live_event(db, account_id=run["live_account_id"], session=session,
                          stage="snapshot_downloaded", message="Snapshot Alpaca concluido.",
                          details={"assets": len(snapshot.frames), "duration_seconds": round(monotonic() - started, 2)})
        stage = "intraday_bars"
        augmented, metadata = append_u67_current_session_intraday(
            snapshot.frames, strategy, session=session, now=clock.timestamp,
        )
        append_live_event(db, account_id=run["live_account_id"], session=session,
                          stage="intraday_bars_ready", message="Dados da sessao atual preparados.",
                          details={"duration_seconds": round(monotonic() - started, 2)})
        stage = "model_training_and_decision"
        append_live_event(db, account_id=run["live_account_id"], session=session,
                          stage="model_training_started", message="LightGBM: treinamento e decisao iniciados.")
        decision = build_live_model_decision(
            augmented, strategy, model_family=model["family"],
            current_asset=state["managed_symbol"], holding_sessions=state["holding_sessions"],
        )
        append_live_event(db, account_id=run["live_account_id"], session=session,
                          stage="model_decision_computed", message="LightGBM: decisao calculada.",
                          details={"duration_seconds": round(monotonic() - started, 2)})
        stage = "decision_validation"
        signal_date = pd.Timestamp(decision.decision_date).date().isoformat()
        if signal_date != session:
            raise RuntimeError("Winner recalibration decision date is not the current session.")
        target = str(decision.target_asset).upper()
        if not target or (target != "CASH" and target not in strategy.assets):
            raise RuntimeError("Winner selected an asset outside the locked universe.")
        candidate_fields = winner_candidate_fields(decision)
        current = state["managed_symbol"]
        action = ("stay_in_cash" if target == current == "CASH" else
                  "hold" if target == current else
                  "buy" if current == "CASH" else
                  "sell_to_cash" if target == "CASH" else "rotate")
        stage = "decision_persistence"
        updated = db[LIVE_RUNS].find_one_and_update(
            {"_id": claimed["_id"], "status": recalibrating_status,
             "winner_configuration_hash": run["winner_configuration_hash"]},
            {"$set": {
                "status": "analysis_only_completed" if analysis_only else "recalibrated_no_orders",
                "phase": "missed_window_analysis_only_no_orders" if analysis_only else "ready_for_execution_review",
                "decision_date": signal_date,
                "target_asset": target,
                      **candidate_fields,
                "action": action,
                "current_asset": current,
                "current_quantity": state["managed_quantity"],
                "live_state_revision": state["revision"],
                "market_open_calibration_completed_at": datetime.now(timezone.utc),
                "calibration_duration_seconds": (datetime.now(timezone.utc) - recalibration_started_at).total_seconds(),
                "current_session_feed": metadata.get("live_feed"),
                "current_session_market_data_cutoff_utc": metadata.get("intraday_market_data_cutoff_utc"),
                "winner_signal_prepared": True,
                "order_submission_enabled": False,
            }, "$unset": {"last_calibration_error_type": ""}},
            return_document=ReturnDocument.AFTER,
        )
        if updated is None:
            raise RuntimeError("Live recalibration state changed during preparation.")
        append_live_event(db, account_id=run["live_account_id"], session=session,
                          stage="analysis_only_completed" if analysis_only else "recalibration_completed",
                          message="Analise concluida, sem envio de ordens." if analysis_only else "Decisao pronta para revisao.",
                          details={"target_asset": target, "action": action,
                                   "duration_seconds": round(monotonic() - started, 2)})
        return {"status": updated["status"], "execution_session": session,
                "target_asset": target, "action": updated["action"],
                "order_submission_enabled": False}
    except Exception as exc:
        append_live_event(db, account_id=run["live_account_id"], session=session,
                          stage="recalibration_failed", level="error",
                          message="Falha na recalibracao durante a etapa: " + stage,
                          details={"error_type": type(exc).__name__,
                                   "duration_seconds": round(monotonic() - started, 2)})
        db[LIVE_RUNS].update_one(
            {"_id": claimed["_id"], "status": recalibrating_status},
            {"$set": {"status": "review_required" if analysis_only else "signal_prepared",
                      "phase": "late_analysis_failed" if analysis_only else "market_open_recalibration_retry",
                      "last_calibration_error_type": type(exc).__name__,
                      "order_submission_enabled": False}},
        )
        raise
