"""Manual current-session Winner analysis for the canonical Alpaca Live account.

Read-only calculation, independent of the automatic D-1 run and Paper state.
A separate, explicit endpoint is required to execute a verified manual result.
Heavy work runs as a FastAPI BackgroundTask and reports through MongoDB.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import exchange_calendars as xcals
import pandas as pd
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

from ..core.config import TCC_U67_CONTROL_OPERATIONAL_MODE, TCC_U77_FIXED_CHALLENGER_MODE
from ..core.runtime import database
from ..engine.live_model_signal import build_live_model_decision
from ..engine.market_data import latest_safe_completed_xnys_session
from ..engine.operational_market_snapshot import (
    DATA_DIRECTORY as TCC_U67_DATA_DIRECTORY,
    append_current_session_intraday,
    download_operational_market_snapshot,
)
from ..engine.tcc_u67_operational_runtime import assert_tcc_u67_operational_contract, assert_tcc_u67_model_snapshot
from ..engine.tcc_u77_static_challenger import U77_REQUESTED_ASSETS, assert_tcc_u77_contract
from ..infrastructure.trading.alpaca_live import create_live_readonly_client
from .live_observation_scheduler import _regular_xnys_session_open
from .live_trader_state import verified_live_state
from .live_winner_session import LIVE_RUNS
from .live_winner_candidates import winner_candidate_fields
from .model_research import apply_execution_profile
from .strategy_lab import get_trader_winner_context, get_trader_winner_model_snapshot
from .system_settings import apply_training_runtime_settings

MANUAL_ANALYSES = "live_winner_manual_intraday_analyses"
_ACTIVE = {"running", "executing"}
_EASTERN = "America/New_York"


def _verified_current_session(broker: Any) -> tuple[str, str, datetime]:
    account = broker.get_account()
    account_id = str(getattr(account, "id", "") or "")
    clock = broker.get_clock()
    raw = getattr(clock, "timestamp", None)
    if not account_id or not isinstance(raw, datetime) or raw.tzinfo is None:
        raise RuntimeError("Cannot verify Live broker account and timestamp.")
    session = raw.astimezone(pd.Timestamp.now(tz=_EASTERN).tzinfo).date().isoformat()
    # Both the real broker clock and exchange calendar must agree.
    if not _regular_xnys_session_open(
        session=session, broker_timestamp=raw,
        broker_is_open=bool(getattr(clock, "is_open", False)),
        broker_next_close=getattr(clock, "next_close", None),
    ):
        raise RuntimeError("Live current-session analysis requires the regular market to be open.")
    return account_id, session, raw


def _safe_public(doc: dict[str, Any] | None, *,
                 account_id: str, session: str, market_open: bool) -> dict[str, Any]:
    doc = doc or {}
    status = str(doc.get("status") or "idle")
    # Avoid exposing raw provider error strings, credentials, private account IDs.
    visible = {
        "status": status, "session": session,
        "analysis_id": doc.get("analysis_id"),
        "started_at": doc["started_at"].isoformat() if isinstance(doc.get("started_at"), datetime) else None,
        "completed_at": doc["completed_at"].isoformat() if isinstance(doc.get("completed_at"), datetime) else None,
        "current_asset": doc.get("current_asset"),
        "target_asset": doc.get("target_asset"),
        "raw_best_asset": doc.get("raw_best_asset"),
        "action": doc.get("action"),
        "historical_cutoff": doc.get("historical_cutoff"),
        "analysis_timestamp_utc": doc.get("analysis_timestamp_utc"),
        "intraday_market_data_cutoff_utc": doc.get("intraday_market_data_cutoff_utc"),
        "intraday_data_delay_minutes": doc.get("intraday_data_delay_minutes"),
        "live_feed": doc.get("live_feed"),
        "selected_utility": doc.get("selected_utility"),
        "snapshot_sha256": doc.get("snapshot_sha256"),
        "original_signal_target": doc.get("original_signal_target"),
        "original_signal_status": doc.get("original_signal_status"),
        "failure_type": doc.get("failure_type"),
        "phase": doc.get("phase"),
        "can_analyze": bool(market_open and status not in _ACTIVE),
        "order_submission_enabled": False,
        "method": "completed_daily_plus_current_intraday",
    }
    return visible


def manual_intraday_status(db: Any) -> dict[str, Any]:
    broker = create_live_readonly_client()
    account = broker.get_account()
    account_id = str(getattr(account, "id", "") or "")
    clock = broker.get_clock()
    timestamp = getattr(clock, "timestamp", None)
    if not account_id or not isinstance(timestamp, datetime) or timestamp.tzinfo is None:
        raise RuntimeError("Cannot verify Alpaca Live account and clock.")
    session = timestamp.astimezone(pd.Timestamp.now(tz=_EASTERN).tzinfo).date().isoformat()
    available = _regular_xnys_session_open(
        session=session, broker_timestamp=timestamp,
        broker_is_open=bool(getattr(clock, "is_open", False)),
        broker_next_close=getattr(clock, "next_close", None),
    )
    doc = db[MANUAL_ANALYSES].find_one({"_id": f"{account_id}:{session}"})
    # The operator may retry a job orphaned by a worker restart, but only
    # after a generous time allowance. No stale "running" job can authorize
    # execution of real orders.
    if doc and doc.get("status") == "running":
        started = doc.get("started_at")
        if isinstance(started, datetime):
            started = started.replace(tzinfo=timezone.utc) if started.tzinfo is None else started
            if datetime.now(timezone.utc) - started > timedelta(hours=2):
                db[MANUAL_ANALYSES].update_one(
                    {"_id": doc["_id"], "analysis_id": doc.get("analysis_id"), "status": "running"},
                    {"$set": {"status": "failed", "failure_type": "InterruptedWorker",
                              "phase": "stale_worker", "completed_at": datetime.now(timezone.utc)}},
                )
                doc = db[MANUAL_ANALYSES].find_one({"_id": f"{account_id}:{session}"})
    return _safe_public(doc, account_id=account_id, session=session, market_open=available)


def request_manual_intraday_analysis(db: Any, *, actor: str) -> dict[str, Any]:
    """Atomic claim of one read-only analysis; returns task identity to enqueue."""
    if not actor:
        raise RuntimeError("Administrator identity is required.")
    broker = create_live_readonly_client()
    account_id, session, _ = _verified_current_session(broker)
    doc_id = f"{account_id}:{session}"
    existing = db[MANUAL_ANALYSES].find_one({"_id": doc_id})
    if existing and existing.get("status") in _ACTIVE:
        return {**_safe_public(existing, account_id=account_id, session=session, market_open=True),
                "queued_new": False}
    analysis_id = uuid4().hex
    now = datetime.now(timezone.utc)
    try:
        claimed = db[MANUAL_ANALYSES].find_one_and_update(
            {"_id": doc_id, "status": {"$nin": list(_ACTIVE)}},
            {"$set": {
                "market": "live", "session": session, "account_id": account_id,
                "status": "running", "phase": "validating", "analysis_id": analysis_id,
                "started_at": now, "completed_at": None, "actor": actor,
                "order_submission_enabled": False,
                "failure_type": None, "target_asset": None,
                "action": None, "snapshot_sha256": None,
            }},
            upsert=True, return_document=ReturnDocument.AFTER,
        )
    except DuplicateKeyError:
        existing = db[MANUAL_ANALYSES].find_one({"_id": doc_id})
        return {**_safe_public(existing, account_id=account_id, session=session, market_open=True),
                "queued_new": False}
    if not claimed or claimed.get("analysis_id") != analysis_id:
        raise RuntimeError("Cannot claim isolated Live manual analysis.")
    return {**_safe_public(claimed, account_id=account_id, session=session, market_open=True),
            "queued_new": True}


def run_manual_intraday_analysis(account_id: str, session: str, analysis_id: str) -> None:
    """Background worker: never invokes TradingClient order-write methods."""
    db = database()
    selector = {"_id": f"{account_id}:{session}",
                "analysis_id": analysis_id, "status": "running"}
    def phase(name: str) -> None:
        db[MANUAL_ANALYSES].update_one(selector, {"$set": {"phase": name}})
    try:
        broker = create_live_readonly_client()
        verified_account, current_session, broker_timestamp = _verified_current_session(broker)
        if verified_account != account_id or current_session != session:
            raise RuntimeError("Live account/session identity changed during analysis.")
        state = verified_live_state(db, broker, session=session)
        if state["last_execution_session"] == session:
            # Analysis is still allowed after an executed session. It never
            # makes a second execution eligible; this is advisory only.
            pass
        config, profile = get_trader_winner_context(db)
        model = get_trader_winner_model_snapshot(db)
        strategy = apply_training_runtime_settings(db, config)
        strategy = apply_execution_profile(
            strategy, model["family"], model.get("settings_snapshot") or {},
        )
        strategy = strategy.model_copy(update={
            "research_model_family": model["family"],
            "research_model_settings": model.get("settings_snapshot") or {},
        })
        if (str(strategy.strategy_mode) not in {
                TCC_U67_CONTROL_OPERATIONAL_MODE, TCC_U77_FIXED_CHALLENGER_MODE}
                or model["family"] != "lightgbm_utility"):
            raise RuntimeError("Unsupported locked Live Winner model.")
        u77 = str(strategy.strategy_mode) == TCC_U77_FIXED_CHALLENGER_MODE
        if u77:
            assert_tcc_u77_contract(strategy)
        else:
            assert_tcc_u67_operational_contract(strategy)
        assert_tcc_u67_model_snapshot(model)
        calendar = xcals.get_calendar("XNYS")
        previous = calendar.previous_session(pd.Timestamp(session)).date().isoformat()
        safe = latest_safe_completed_xnys_session(now=broker_timestamp, data_delay_minutes=0).date().isoformat()
        if safe != previous:
            raise RuntimeError("Latest completed XNYS candle does not match the required previous session.")

        phase("downloading_daily_history")
        snapshot = download_operational_market_snapshot(
            job_id=f"live-manual-{session}-{analysis_id}",
            config=strategy.model_copy(update={"analysis_end_date": previous}),
            requested_assets=U77_REQUESTED_ASSETS if u77 else tuple(strategy.assets),
            data_directory=TCC_U67_DATA_DIRECTORY.parent.parent / "live_manual_intraday" / "snapshots",
        )
        if (not snapshot.frames or any(frame.empty for frame in snapshot.frames.values())
                or not snapshot.manifest.get("snapshot_sha256")):
            raise RuntimeError("Full completed-daily snapshot is unavailable.")

        phase("collecting_current_session")
        frames, market = append_current_session_intraday(
            snapshot.frames, strategy, session=session, now=broker_timestamp,
        )
        if (market.get("analysis_mode") != "current_session_intraday"
            or market.get("live_session") != session
            or len(frames) != len(snapshot.frames)):
            raise RuntimeError("Current-session market-data coverage is incomplete.")
        phase("model_calibration")
        decision = build_live_model_decision(
            frames, strategy, model_family=model["family"],
            current_asset=state["managed_symbol"],
            holding_sessions=state["holding_sessions"],
        )
        if str(decision.decision_date)[:10] != session:
            raise RuntimeError("Intraday Winner decision_date does not match the current session.")
        target = str(decision.target_asset).upper()
        if not target or (target != "CASH" and target not in strategy.assets):
            raise RuntimeError("Live Winner selected a non-eligible target.")
        current = state["managed_symbol"]
        action = ("stay_in_cash" if current == target == "CASH" else
                  "hold" if current == target else
                  "buy" if current == "CASH" else
                  "sell_to_cash" if target == "CASH" else "rotate")

        run = db[LIVE_RUNS].find_one({"_id": f"{account_id}:{session}"}) or {}
        # Decision remains separate from the automatic daily session.
        saved = db[MANUAL_ANALYSES].update_one(selector, {"$set": {
            "status": "prepared", "phase": "finished", "completed_at": datetime.now(timezone.utc),
            "winner_strategy_id": str(profile["id"]),
            "winner_revision": int(profile["revision"]),
            "winner_configuration_hash": str(profile["configuration_hash"]),
            "winner_model_family": model["family"],
            "session": session, "current_asset": current, "target_asset": target,
            "raw_best_asset": str(decision.raw_best_asset),
            "action": action,
            "live_state_revision": state["revision"],
            "current_quantity": state["managed_quantity"],
            "selected_utility": float(decision.selected_utility),
            "decision_date": session, "historical_cutoff": previous,
            "snapshot_sha256": snapshot.manifest["snapshot_sha256"],
            "analysis_mode": "current_session_intraday",
            "analysis_timestamp_utc": market["analysis_timestamp_utc"],
            "intraday_market_data_cutoff_utc": market["intraday_market_data_cutoff_utc"],
            "intraday_data_delay_minutes": int(market.get("intraday_data_delay_minutes") or 0),
            "live_feed": market.get("live_feed"),
            "original_signal_target": run.get("target_asset"),
            "original_signal_status": run.get("status"),
            "candidate_count": len(getattr(decision, "utilities", {}) or {}),
            "order_submission_enabled": False,
        }})
        if saved.matched_count != 1:
            raise RuntimeError("Manual analysis claim changed before saving result.")
    except Exception as exc:
        # Do not expose raw broker errors in HTTP or UI.
        db[MANUAL_ANALYSES].update_one(selector, {"$set": {
            "status": "failed", "phase": "failed", "failure_type": type(exc).__name__,
            "completed_at": datetime.now(timezone.utc),
            "order_submission_enabled": False,
        }})
