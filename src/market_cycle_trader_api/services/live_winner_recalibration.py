"""Open-session independent Winner recalibration for the Alpaca Live account."""
from __future__ import annotations

from datetime import datetime, timezone
import re
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
    SIP_HISTORICAL_DELAY_MINUTES,
    download_u67_operational_snapshot,
)
from ..engine.tcc_u67_operational_runtime import assert_tcc_u67_model_snapshot, assert_tcc_u67_operational_contract
from ..engine.tcc_u77_static_challenger import U77_REQUESTED_ASSETS, assert_tcc_u77_contract
from ..infrastructure.trading.alpaca_live import create_live_readonly_client
from .live_winner_session import LIVE_RUNS
from .live_execution_window import live_execution_window_deadline
from .live_winner_events import append_live_event
from .live_winner_candidates import winner_candidate_fields
from .live_trader_state import verified_live_state
from .model_research import apply_execution_profile
from .system_settings import apply_training_runtime_settings
from .strategy_lab import get_trader_winner_context, get_trader_winner_model_snapshot

EASTERN = ZoneInfo("America/New_York")

# The release must not reveal raw Alpaca errors, URLs or credentials in events.
_AUDITABLE_INTRADAY_CODES = frozenset({
    "CurrentSessionIntradayDataMissing",
    "CurrentSessionIntradayCoverageUnavailable",
    "CurrentSessionIntradayDelayedSIPNotReady",
    "CurrentSessionIntradayNotStarted",
    "CurrentSessionIntradayDateMismatch",
    "LiveExecutionWindowExpired",
    "LiveIntradayMarketClockUnavailable",
})


def _utc_time(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise RuntimeError("LiveIntradayMarketClockUnavailable: broker timestamp has no timezone.")
    return stamp.tz_convert("UTC")


def _intraday_error_fields(exc: Exception) -> dict[str, str]:
    """Expose only explicitly recognized fields, never provider response text."""
    message = str(exc)
    prefix = message.partition(":")[0]
    code = prefix if prefix in _AUDITABLE_INTRADAY_CODES else type(exc).__name__
    result = {"error_code": code}
    match = re.search(
        r"\b([A-Z][A-Z0-9.]{0,9}) has no (IEX|SIP) regular-session bars\b",
        message,
    )
    if match:
        result["symbol"] = match.group(1)
        result["feed"] = match.group(2).lower()
    return result


def _expire_live_window(db: Any, run: dict[str, Any], *, deadline: pd.Timestamp) -> None:
    """Fail closed for stale model/market sessions; no broker submission."""
    db[LIVE_RUNS].update_one(
        {"_id": run["_id"], "status": run["status"]},
        {"$set": {
            "status": "review_required",
            "phase": "execution_window_expired",
            "window_deadline_at": deadline.to_pydatetime(),
            "window_expired_at": datetime.now(timezone.utc),
            "order_submission_enabled": False,
        }},
    )
    append_live_event(
        db, account_id=run["live_account_id"], session=run["execution_session"],
        stage="execution_window_expired", level="warning",
        message="Janela Live encerrada; nenhuma ordem autorizada.",
        details={"error_code": "LiveExecutionWindowExpired"},
    )


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

    # This Live-only 30-minute default never inherits a shorter Paper window.
    deadline, _ = live_execution_window_deadline(expected_open)
    if not analysis_only and now >= deadline:
        _expire_live_window(db, run, deadline=deadline)
        return {"status": "execution_window_expired", "order_submission_enabled": False}
    retry_after_raw = run.get("retry_not_before")
    if not analysis_only and retry_after_raw:
        # PyMongo usually decodes UTC datetimes without tzinfo.
        retry_at = pd.Timestamp(retry_after_raw)
        retry_at = retry_at.tz_localize("UTC") if retry_at.tzinfo is None else retry_at.tz_convert("UTC")
        if now < retry_at:
            return {"status": "waiting_for_intraday_data", "order_submission_enabled": False,
                    "retry_after": retry_at.isoformat()}
    # At 09:30:XX no complete one-minute bar exists. Retry next scheduler tick
    # without downloading all historical assets or entering the order path.
    if now.floor("min") <= expected_open:
        return {"status": "waiting_for_intraday_data", "order_submission_enabled": False,
                "retry_after": (expected_open + pd.Timedelta(minutes=1)).isoformat()}
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
        # A historical refresh can take minutes: do NOT reuse the clock that
        # was sampled before the snapshot. All 75 assets share one fresh cutoff.
        stage = "intraday_bars"
        refreshed_clock = broker.get_clock()
        if not bool(getattr(refreshed_clock, "is_open", False)):
            raise RuntimeError("LiveIntradayMarketClockUnavailable: market closed during refresh.")
        refreshed_now = _utc_time(refreshed_clock.timestamp)
        if refreshed_now.tz_convert(EASTERN).date().isoformat() != session:
            raise RuntimeError("LiveIntradayMarketClockUnavailable: market session changed.")
        if not analysis_only and refreshed_now >= deadline:
            raise RuntimeError("LiveExecutionWindowExpired: historical refresh exceeded order window.")
        intraday_cutoff = refreshed_now.floor("min")
        if intraday_cutoff <= expected_open:
            raise RuntimeError("CurrentSessionIntradayNotStarted: no complete regular-session minute.")
        augmented, metadata = append_u67_current_session_intraday(
            snapshot.frames, strategy, session=session, now=intraday_cutoff,
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
        if not analysis_only:
            completed_clock = broker.get_clock()
            completed_at = _utc_time(completed_clock.timestamp)
            if (not bool(getattr(completed_clock, "is_open", False))
                or completed_at >= deadline
                or completed_at.tz_convert(EASTERN).date().isoformat() != session):
                raise RuntimeError("LiveExecutionWindowExpired: model decision completed outside order window.")
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
                "window_deadline_at": deadline.to_pydatetime(),
                "market_open_calibration_completed_at": datetime.now(timezone.utc),
                "calibration_duration_seconds": (datetime.now(timezone.utc) - recalibration_started_at).total_seconds(),
                "current_session_feed": metadata.get("live_feed"),
                "current_session_market_data_cutoff_utc": metadata.get("intraday_market_data_cutoff_utc"),
                "winner_signal_prepared": True,
                "order_submission_enabled": False,
            }, "$unset": {"last_calibration_error_type": "",
                          "last_calibration_error_code": "",
                          "last_intraday_error_symbol": "",
                          "last_intraday_error_feed": "",
                          "retry_not_before": ""}},
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
        diagnostic = _intraday_error_fields(exc)
        expired = diagnostic["error_code"] == "LiveExecutionWindowExpired"
        details: dict[str, Any] = {
            "error_type": type(exc).__name__,
            **diagnostic,
            "duration_seconds": round(monotonic() - started, 2),
        }
        if stage == "intraday_bars" and "intraday_cutoff" in locals():
            details["cutoff"] = intraday_cutoff.isoformat()
        retry_at = None
        if (not analysis_only and not expired
            and diagnostic["error_code"] == "CurrentSessionIntradayDelayedSIPNotReady"):
            # SIP may only be queried after the uniform 20-minute delay.
            retry_at = expected_open + pd.Timedelta(minutes=SIP_HISTORICAL_DELAY_MINUTES + 1)
            details["retry_after"] = retry_at.isoformat()
        append_live_event(db, account_id=run["live_account_id"], session=session,
                          stage="recalibration_failed", level="error",
                          message="Falha na recalibracao durante a etapa: " + stage,
                          details=details)
        changes: dict[str, Any] = {
            "status": "review_required" if analysis_only or expired else "signal_prepared",
            "phase": ("execution_window_expired" if expired else
                      "late_analysis_failed" if analysis_only else
                      "market_open_recalibration_retry"),
            "last_calibration_error_type": type(exc).__name__,
            "last_calibration_error_code": diagnostic["error_code"],
            "last_intraday_error_symbol": diagnostic.get("symbol"),
            "last_intraday_error_feed": diagnostic.get("feed"),
            "order_submission_enabled": False,
        }
        if expired:
            changes["window_deadline_at"] = deadline.to_pydatetime()
            changes["window_expired_at"] = datetime.now(timezone.utc)
        if retry_at is not None:
            changes["retry_not_before"] = retry_at.to_pydatetime()
        update: dict[str, Any] = {"$set": changes}
        if retry_at is None:
            update["$unset"] = {"retry_not_before": ""}
        db[LIVE_RUNS].update_one(
            {"_id": claimed["_id"], "status": recalibrating_status},
            update,
        )
        raise
