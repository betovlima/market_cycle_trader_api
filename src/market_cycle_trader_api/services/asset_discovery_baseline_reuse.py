from __future__ import annotations

import statistics
import threading
from typing import Any

import pandas as pd

from ..infrastructure.persistence.mongo_repository import COMPARISONS_COLLECTION, JOBS_COLLECTION


_INSTALLED = False
_STATE_LOCK = threading.RLock()
_ACTIVE_BASELINE: dict[str, Any] | None = None
_VALIDATION_THREAD_NAME = "asset-discovery-full-strategy-validation"


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def _median(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [_finite(row.get(key)) for row in rows]
    clean = [value for value in values if value is not None]
    return float(statistics.median(clean)) if clean else None


def _worst_fold_return(rows: list[dict[str, Any]]) -> float | None:
    values: list[float] = []
    for row in rows:
        folds = row.get("walk_forward_folds") if isinstance(row.get("walk_forward_folds"), list) else []
        for fold in folds:
            if not isinstance(fold, dict):
                continue
            value = _finite(fold.get("strategy_return"))
            if value is not None:
                values.append(value)
    return min(values) if values else None


def _normalized_assets(values: Any) -> list[str]:
    return [str(value or "").strip().upper() for value in list(values or []) if str(value or "").strip()]


def _iso_date(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp):
            return ""
        return stamp.date().isoformat()
    except Exception:
        return str(value).strip()


def _eligible_baseline(service: Any, db: Any, document: dict[str, Any]) -> dict[str, Any] | None:
    if str(document.get("discovery_mode") or "").strip().lower() != "predictive_only":
        return None

    try:
        source_raw, source_config = service._current_research_source(db)
    except Exception:
        return None

    source_id = str(source_raw.get("_id") or "").strip()
    source_revision = int(source_raw.get("revision") or 1)
    source_hash = str(source_raw.get("configuration_hash") or "").strip()
    last_backtest_id = str(source_raw.get("last_backtest_id") or "").strip()
    if not source_id or not source_hash or not last_backtest_id:
        return None
    if str(source_raw.get("last_backtest_status") or "").strip().lower() != "completed":
        return None
    if int(source_raw.get("last_backtest_revision") or 0) != source_revision:
        return None

    job = db[JOBS_COLLECTION].find_one({"id": last_backtest_id}) or {}
    if str(job.get("status") or "").strip().lower() != "completed":
        return None
    if str(job.get("strategy_profile_id") or "").strip() != source_id:
        return None
    if int(job.get("strategy_profile_revision") or 0) != source_revision:
        return None
    if str(job.get("strategy_configuration_hash") or "").strip() != source_hash:
        return None

    try:
        model_snapshot = service.get_strategy_model_snapshot(db, source_id)
    except Exception:
        return None
    current_model_hash = str(model_snapshot.get("settings_hash") or "").strip()
    if current_model_hash and str(job.get("research_model_settings_hash") or "").strip() != current_model_hash:
        return None

    request = job.get("request") if isinstance(job.get("request"), dict) else {}
    job_assets = _normalized_assets(request.get("assets"))
    source_assets = _normalized_assets(source_config.assets)
    if job_assets != source_assets:
        return None

    baseline = document.get("baseline") if isinstance(document.get("baseline"), dict) else {}
    snapshot_end = _iso_date(baseline.get("market_snapshot_end"))
    if not snapshot_end:
        snapshot_end = _iso_date((document.get("discovery_selection_model") or {}).get("snapshot_end"))
    if not snapshot_end:
        return None
    if _iso_date(request.get("analysis_end_date")) != snapshot_end:
        return None
    if _iso_date(request.get("analysis_start_date")) != _iso_date(source_config.start_date):
        return None

    comparison = db[COMPARISONS_COLLECTION].find_one({"job_id": last_backtest_id}) or {}
    raw_rows = comparison.get("results") if isinstance(comparison.get("results"), list) else []
    model_family = str(job.get("research_model_family") or "").strip()
    rows = [
        row for row in raw_rows
        if isinstance(row, dict)
        and bool(row.get("portfolio_rotation"))
        and (not model_family or str(row.get("model_family") or row.get("backend") or "").strip() == model_family)
    ]
    if not rows:
        rows = [row for row in raw_rows if isinstance(row, dict) and bool(row.get("portfolio_rotation"))]
    ending_capital = _median(rows, "strategy_ending_capital")
    if ending_capital is None:
        return None

    metrics = {
        "ending_capital": ending_capital,
        "cagr": _median(rows, "strategy_cagr"),
        "sharpe": _median(rows, "strategy_sharpe"),
        "maximum_drawdown": _median(rows, "strategy_maximum_drawdown"),
        "market_exposure": _median(rows, "market_exposure"),
        "cash_days": _median(rows, "cash_days"),
        "switches": _median(rows, "capital_rotations"),
        "worst_fold_return": _worst_fold_return(rows),
        "negative_months": None,
        "severe_negative_months": None,
        "severe_month_threshold": float(getattr(service, "DEFAULT_SEVERE_MONTH_THRESHOLD", -0.05)),
        "repetition_count": len(rows),
        "decision_session_count": None,
        "decision_session_start": None,
        "decision_session_end": None,
        "reused_certified_backtest": True,
        "reused_backtest_id": last_backtest_id,
    }
    return {
        "db": db,
        "run_id": str(document.get("run_id") or ""),
        "source_strategy_id": source_id,
        "source_strategy_revision": source_revision,
        "source_strategy_hash": source_hash,
        "snapshot_end": snapshot_end,
        "backtest_id": last_backtest_id,
        "metrics": metrics,
        "consumed": False,
    }


def install_asset_discovery_baseline_reuse() -> None:
    global _INSTALLED, _ACTIVE_BASELINE
    if _INSTALLED:
        return

    from . import asset_discovery as service

    original_start = service.start_full_strategy_validation
    original_run_replay = service._run_rotation_replay
    original_context = service._research_context_compatibility
    original_worker = service._run_full_strategy_validation_worker

    if getattr(original_start, "_asset_discovery_baseline_reuse", False):
        _INSTALLED = True
        return

    def start_with_baseline_reuse(db: Any, *, run_id: str | None, symbols: list[str]) -> dict[str, Any]:
        global _ACTIVE_BASELINE
        document = service._campaign(db) or {}
        reusable = _eligible_baseline(service, db, document)
        with _STATE_LOCK:
            _ACTIVE_BASELINE = reusable
        try:
            return original_start(db, run_id=run_id, symbols=symbols)
        except Exception:
            with _STATE_LOCK:
                _ACTIVE_BASELINE = None
            raise

    def run_replay_with_baseline_reuse(*args: Any, **kwargs: Any) -> Any:
        global _ACTIVE_BASELINE
        if threading.current_thread().name == _VALIDATION_THREAD_NAME:
            request = args[1] if len(args) > 1 else None
            candidate_assets = _normalized_assets(getattr(request, "research_candidate_assets", []))
            with _STATE_LOCK:
                state = _ACTIVE_BASELINE
                if state is not None and not state.get("consumed") and not candidate_assets:
                    state["consumed"] = True
                    metrics = dict(state.get("metrics") or {})
                    try:
                        state["db"][service.COLLECTION].update_one(
                            {"_id": service.CURRENT_ID, "run_id": state.get("run_id")},
                            {"$set": {
                                "full_strategy_validation.current_stage": "Certified Strategy backtest reused as baseline",
                                "full_strategy_validation.progress_percent": 50.0,
                                "updated_at": service.utc_now(),
                            }},
                        )
                    except Exception:
                        pass
                    return metrics, pd.DatetimeIndex([])
        return original_run_replay(*args, **kwargs)

    def context_with_reused_baseline(baseline_sessions: Any, candidate_sessions: Any) -> dict[str, Any]:
        result = dict(original_context(baseline_sessions, candidate_sessions))
        if threading.current_thread().name != _VALIDATION_THREAD_NAME:
            return result
        with _STATE_LOCK:
            state = dict(_ACTIVE_BASELINE or {})
        if not state.get("consumed") or len(pd.DatetimeIndex(baseline_sessions)):
            return result
        candidate = pd.DatetimeIndex(candidate_sessions).tz_localize(None).normalize().unique().sort_values()
        result.update({
            "research_context_compatible": True,
            "research_context_baseline_sessions": int(len(candidate)),
            "research_context_candidate_sessions": int(len(candidate)),
            "research_context_missing_sessions": 0,
            "research_context_first_missing_session": None,
            "research_context_last_missing_session": None,
            "baseline_source": "certified_strategy_backtest",
            "baseline_backtest_id": state.get("backtest_id"),
        })
        return result

    def worker_with_cleanup(db: Any, run_id: str, validation_id: str, worker_id: str) -> None:
        global _ACTIVE_BASELINE
        try:
            original_worker(db, run_id, validation_id, worker_id)
        finally:
            with _STATE_LOCK:
                _ACTIVE_BASELINE = None

    setattr(start_with_baseline_reuse, "_asset_discovery_baseline_reuse", True)
    service.start_full_strategy_validation = start_with_baseline_reuse
    service._run_rotation_replay = run_replay_with_baseline_reuse
    service._research_context_compatibility = context_with_reused_baseline
    service._run_full_strategy_validation_worker = worker_with_cleanup
    _INSTALLED = True
