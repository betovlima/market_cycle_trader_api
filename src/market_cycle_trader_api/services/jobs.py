from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import threading
import sys
from typing import Any

from fastapi import HTTPException

from ..core.config import ENGINE_MODULE, ENGINE_PATH, SOURCE_ROOT
from ..core.environment import build_subprocess_environment, load_project_environment
from ..core.runtime import database
from ..infrastructure.persistence.mongo_repository import COMPARISONS_COLLECTION, JOBS_COLLECTION, RUNS_COLLECTION, utc_now
from .serialization import iso_value
from .strategy_lab import mark_strategy_backtest

logger = logging.getLogger("uvicorn.error")


WINNER_ENGINE_COMPATIBILITY = "api-v1.13.16"
_NUMERIC_THREAD_ENVIRONMENT_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

_ACTIVE_JOB_PROCESSES: dict[str, subprocess.Popen] = {}
_ACTIVE_JOB_PROCESSES_LOCK = threading.Lock()


def _register_active_process(job_id: str, process: subprocess.Popen) -> None:
    with _ACTIVE_JOB_PROCESSES_LOCK:
        _ACTIVE_JOB_PROCESSES[job_id] = process


def _unregister_active_process(job_id: str, process: subprocess.Popen) -> None:
    with _ACTIVE_JOB_PROCESSES_LOCK:
        if _ACTIVE_JOB_PROCESSES.get(job_id) is process:
            _ACTIVE_JOB_PROCESSES.pop(job_id, None)


def _terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        process.terminate()
    except (ProcessLookupError, OSError):
        return

    def force_kill_if_needed() -> None:
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except (ProcessLookupError, OSError):
                pass

    threading.Thread(target=force_kill_if_needed, daemon=True).start()


def request_job_cancel(job_id: str, *, reason: str = "Cancellation requested.") -> bool:
    






    db = database()
    job = db[JOBS_COLLECTION].find_one({"id": job_id}) or {}
    status = str(job.get("status") or "").strip().lower()
    if status not in {"queued", "running"}:
        return False
    db[JOBS_COLLECTION].update_one(
        {"id": job_id},
        {"$set": {
            "cancel_requested": True,
            "cancel_reason": str(reason)[:300],
            "updated_at": utc_now(),
        }},
    )
    with _ACTIVE_JOB_PROCESSES_LOCK:
        process = _ACTIVE_JOB_PROCESSES.get(job_id)
    if process is not None:
        _terminate_process(process)
    return True


def numeric_thread_environment(request_payload: dict[str, Any]) -> dict[str, str]:
    






    if not bool(request_payload.get("deterministic_execution")):
        return {}
    numeric_threads = max(1, int(request_payload.get("numeric_thread_limit") or 1))
    return {key: str(numeric_threads) for key in _NUMERIC_THREAD_ENVIRONMENT_KEYS}


PUBLIC_JOB_FIELDS = frozenset({
    "id",
    "status",
    "stage",
    "progress",
    "created_at",
    "updated_at",
    "started_at",
    "finished_at",
    "completed_runs",
    "total_runs",
    "return_code",
    "strategy_profile_id",
    "strategy_profile_name",
    "strategy_profile_revision",
    "strategy_configuration_hash",
    "progress_detail",
})

_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)\b(ALPACA_API_KEY_ID|ALPACA_SECRET_KEY|MONGO_URL)\b\s*[:=]\s*([^\s,;]+)"
)
_MONGODB_CREDENTIAL_PATTERN = re.compile(
    r"(mongodb(?:\+srv)?://)([^@\s]+)@",
    flags=re.IGNORECASE,
)
_SAFE_PROGRESS_PATTERNS = (
    re.compile(r"^Prepared \d+ assets and \d+ folds — XGBoost=(?:CPU|CUDA(?: — .+)?)$"),
    re.compile(r"^Prepared \d+ assets and \d+ folds — LightGBM=CPU$"),
    re.compile(r"^Prepared \d+ assets and \d+ folds — IQN=(?:CPU|CUDA(?: — .+)?)$"),
    re.compile(r"^Run \d+/\d+ — fold \d+/\d+ — (?:calibration training|final training) \d+/\d+$"),
    re.compile(r"^Run \d+/\d+ — fold \d+/\d+ — evaluating rotation policy candidates$"),
    re.compile(r"^Run \d+/\d+ — fold \d+/\d+ completed$"),
    re.compile(r"^Run \d+/\d+ — simulating out-of-sample portfolio$"),
    re.compile(r"^XGBoost Utility run \d+/\d+ completed$"),
    re.compile(r"^LightGBM Utility run \d+/\d+ completed$"),
    re.compile(r"^IQN run \d+/\d+ completed$"),
    re.compile(r"^Run \d+/\d+ — fold \d+/\d+ — IQN training \d+%$"),
)
_PROGRESS_DETAIL_FIELDS = frozenset({
    "run_index",
    "run_count",
    "fold_index",
    "fold_count",
    "phase",
    "trained_models",
    "total_models",
    "device",
})
_PROGRESS_PHASES = frozenset({
    "Preparing run",
    "Calibration training",
    "Policy calibration",
    "Final training",
    "Fold completed",
    "Out-of-sample simulation",
    "Run completed",
    "IQN training",
})
_PROGRESS_DEVICES = frozenset({"CPU", "CUDA"})


def _redact_sensitive_text(raw_line: Any) -> str:
    line = str(raw_line).strip()
    line = _SECRET_ASSIGNMENT_PATTERN.sub(r"\1=***", line)
    return _MONGODB_CREDENTIAL_PATTERN.sub(r"\1***@", line)


def _safe_progress_line(line: str) -> bool:
    return any(pattern.fullmatch(line) is not None for pattern in _SAFE_PROGRESS_PATTERNS)


def _clean_progress_detail(raw_detail: Any) -> dict[str, Any]:
    if not isinstance(raw_detail, dict):
        return {}

    clean: dict[str, Any] = {}
    for key in (
        "run_index",
        "run_count",
        "fold_index",
        "fold_count",
        "trained_models",
        "total_models",
    ):
        try:
            clean[key] = max(0, min(10_000, int(raw_detail.get(key) or 0)))
        except (TypeError, ValueError):
            clean[key] = 0

    phase = str(raw_detail.get("phase") or "").strip()
    if phase in _PROGRESS_PHASES:
        clean["phase"] = phase

    device = str(raw_detail.get("device") or "").strip().upper()
    if device in _PROGRESS_DEVICES:
        clean["device"] = device

    return clean


def _public_stage(raw_stage: Any) -> str:
    stage = str(raw_stage or "").strip()
    lowered = stage.lower()
    if not lowered:
        return "Queued"
    if _safe_progress_line(stage):
        return "Running analysis"
    if "interrupt" in lowered or "cancel" in lowered:
        return "Interrupted"
    if "fail" in lowered or "error" in lowered:
        return "Backtest failed"
    if "complete" in lowered:
        return "Completed"
    if "final" in lowered or "report" in lowered:
        return "Finalizing results"
    if "load" in lowered or "market data" in lowered:
        return "Loading market data"
    if "prepare" in lowered or "build" in lowered or "align" in lowered:
        return "Preparing analysis"
    if "run" in lowered or "train" in lowered or "fold" in lowered:
        return "Running analysis"
    if "start" in lowered:
        return "Starting backtest"
    if "queue" in lowered:
        return "Queued"
    return "Running analysis"


def _public_log_line(raw_line: Any) -> str | None:
    line = str(raw_line).strip()
    if not line:
        return None

    line = _redact_sensitive_text(line)
    lowered = line.lower()

    if _safe_progress_line(line):
        return line

    if lowered == "backtest queued.":
        return "Backtest queued."
    if lowered.startswith("error"):
        return "Backtest failed. Check the protected server logs."
    if lowered.startswith("loading "):
        return "Loading market data."
    if lowered.startswith("preparing ") or lowered.startswith("building "):
        return "Preparing analysis."
    if lowered.startswith("running ") or lowered.startswith("training "):
        return "Running analysis."
    if lowered.startswith("finalizing "):
        return "Finalizing results."
    if lowered.startswith("portfolio/"):
        return "Analysis run completed."
    if "local simulation only" in lowered:
        return "Simulation environment ready."
    return None


def public_job(document: dict[str, Any] | None) -> dict[str, Any] | None:
    if document is None:
        return None

    payload = {
        key: iso_value(value)
        for key, value in document.items()
        if key in PUBLIC_JOB_FIELDS
    }
    status = str(document.get("status") or "").strip().lower()
    if status == "failed":
        payload["stage"] = "Backtest failed"
    elif status in {"interrupted", "cancelled"}:
        payload["stage"] = "Interrupted"
    elif status == "completed":
        payload["stage"] = "Completed"
    else:
        payload["stage"] = _public_stage(document.get("stage"))

    payload["progress_detail"] = {
        key: iso_value(value)
        for key, value in _clean_progress_detail(document.get("progress_detail")).items()
    }

    raw_logs = document.get("logs")
    if isinstance(raw_logs, list):
        public_logs = [
            line
            for raw_line in raw_logs
            if (line := _public_log_line(raw_line)) is not None
        ]
        deduplicated: list[str] = []
        for line in public_logs:
            if not deduplicated or deduplicated[-1] != line:
                deduplicated.append(line)
        payload["logs"] = deduplicated[-120:]
    else:
        payload["logs"] = []

    return payload


def require_job(job_id: str) -> dict[str, Any]:
    job = database()[JOBS_COLLECTION].find_one({"id": job_id})
    if job is None:
        raise HTTPException(status_code=404, detail="Backtest job not found.")
    return job


def append_log(job_id: str, raw_line: str) -> None:
    line = raw_line.rstrip()
    if not line:
        return
    db = database()
    job = db[JOBS_COLLECTION].find_one({"id": job_id}, {"completed_runs": 1, "total_runs": 1, "live_trade_count": 1}) or {}
    stripped = line.strip()
    if stripped.startswith("JOB_DETAIL|"):
        try:
            detail = json.loads(stripped.removeprefix("JOB_DETAIL|"))
        except json.JSONDecodeError:
            return
        if not isinstance(detail, dict):
            return
        clean = {
            key: iso_value(value)
            for key, value in _clean_progress_detail(detail).items()
        }
        db[JOBS_COLLECTION].update_one(
            {"id": job_id},
            {"$set": {"progress_detail": clean, "updated_at": utc_now()}},
        )
        return
    if stripped.startswith("XGB_TECH|") or stripped.startswith("RESEARCH_TECH|"):
        return
    if stripped.startswith("JOB_TRADE|"):
        try:
            trade = json.loads(stripped.removeprefix("JOB_TRADE|"))
        except json.JSONDecodeError:
            return
        if isinstance(trade, dict):
            allowed = {"backend", "model", "timestamp", "asset", "action", "reason", "execution_price", "quantity", "total_fee", "realized_pnl", "position_return", "cash_after_trade", "walk_forward_fold", "model_family", "random_seed", "repetition_index"}
            clean = {key: iso_value(value) for key, value in trade.items() if key in allowed}
            clean["received_at"] = utc_now()
            db[JOBS_COLLECTION].update_one({"id": job_id}, {"$set": {"updated_at": utc_now()}, "$inc": {"live_trade_count": 1}, "$push": {"live_trades": {"$each": [clean], "$slice": -300}}})
        return

    changes: dict[str, Any] = {"updated_at": utc_now()}
    log_line = line
    if stripped.startswith("JOB_PROGRESS|"):
        parts = stripped.split("|", 3)
        if len(parts) == 4:
            try:
                percent = max(0.0, min(99.9, float(parts[1])))
            except (TypeError, ValueError):
                percent = float(job.get("progress", 0) or 0)
            try:
                completed = max(0, int(parts[2]))
            except (TypeError, ValueError):
                completed = int(job.get("completed_runs", 0) or 0)
            total = int(job.get("total_runs", 0) or 0)
            changes["progress"] = percent
            changes["completed_runs"] = min(total, completed) if total else completed
            changes["stage"] = parts[3].strip() or "Running backtest"
            log_line = changes["stage"]
    elif stripped.startswith("Loading "):
        changes["stage"] = stripped.removesuffix("...")
    elif stripped.startswith("Running "):
        changes["stage"] = stripped.removesuffix("...")
    elif stripped.startswith("ERROR"):
        changes["stage"] = "Backtest error"

    db[JOBS_COLLECTION].update_one({"id": job_id}, {"$set": changes, "$push": {"logs": {"$each": [log_line], "$slice": -400}}})


def _write_child_line_to_console(job_id: str, raw_line: str) -> None:
    line = _redact_sensitive_text(raw_line)
    if not line or line.startswith("JOB_TRADE|") or line.startswith("JOB_DETAIL|"):
        return
    if line.startswith("JOB_PROGRESS|"):
        parts = line.split("|", 3)
        if len(parts) == 4:
            logger.info(
                "Backtest %s | progress=%s%% | %s",
                job_id,
                parts[1],
                parts[3],
            )
        return
    if line.startswith("XGB_TECH|"):
        logger.info("Backtest %s | XGBoost | %s", job_id, line.removeprefix("XGB_TECH|"))
        return
    if line.startswith("RESEARCH_TECH|"):
        logger.info("Backtest %s | Model Research | %s", job_id, line.removeprefix("RESEARCH_TECH|"))
        return
    if line.startswith("ERROR") or line.startswith("Traceback"):
        logger.error("Backtest %s | %s", job_id, line)
        return
    logger.info("Backtest %s | %s", job_id, line)


def run_job(job_id: str) -> None:
    
    
    load_project_environment()
    db = database()
    job_document = db[JOBS_COLLECTION].find_one({"id": job_id}) or {}
    if bool(job_document.get("cancel_requested")):
        db[JOBS_COLLECTION].update_one(
            {"id": job_id},
            {"$set": {
                "status": "cancelled",
                "stage": "Interrupted",
                "finished_at": utc_now(),
                "updated_at": utc_now(),
                "return_code": None,
            }, "$unset": {"process_id": ""}},
        )
        return
    strategy_profile_id = str(job_document.get("strategy_profile_id") or "") or None
    strategy_profile_revision = int(job_document.get("strategy_profile_revision") or 0) or None
    timeout_seconds = max(300, int(job_document.get("training_timeout_seconds") or 21_600))
    request_payload = job_document.get("request") if isinstance(job_document.get("request"), dict) else {}
    research_model_family = str(
        job_document.get("research_model_family")
        or request_payload.get("research_model_family")
        or "xgboost_utility"
    )
    research_model_settings = (
        request_payload.get("research_model_settings")
        if isinstance(request_payload.get("research_model_settings"), dict)
        else {}
    )
    certifies_strategy = research_model_family in {"xgboost_utility", "lightgbm_utility"}
    certifies_strategy = certifies_strategy and bool(job_document.get("certifies_strategy", True))
    db[JOBS_COLLECTION].update_one({"id": job_id}, {"$set": {"status": "running", "stage": "Starting backtest", "started_at": utc_now(), "updated_at": utc_now(), "progress": 0, "progress_detail": {}}})
    python_path = str(SOURCE_ROOT)
    existing_python_path = os.environ.get("PYTHONPATH", "")
    if existing_python_path:
        python_path = python_path + os.pathsep + existing_python_path
    command = [sys.executable, "-u", "-m", ENGINE_MODULE, "--job-id", job_id]
    numeric_environment = numeric_thread_environment(request_payload)
    runtime_thread_limit = int(job_document.get("runtime_thread_limit") or 0)
    if runtime_thread_limit > 0:
        numeric_environment = {key: str(runtime_thread_limit) for key in _NUMERIC_THREAD_ENVIRONMENT_KEYS}
        numeric_environment["MCT_MODEL_THREADS_OVERRIDE"] = str(runtime_thread_limit)
    child_environment = build_subprocess_environment({
        "PYTHONPATH": python_path,
        **numeric_environment,
    })
    engine_identity = {
        "engine_module": ENGINE_MODULE,
        "engine_path": str(ENGINE_PATH),
        "python_executable": sys.executable,
        "winner_engine_compatibility": WINNER_ENGINE_COMPATIBILITY,
        "numeric_thread_environment_applied": bool(numeric_environment),
    }
    db[JOBS_COLLECTION].update_one(
        {"id": job_id},
        {
            "$set": {
                **engine_identity,
                "updated_at": utc_now(),
            },
            "$push": {
                "logs": {
                    "$each": [f"Backtest engine: {ENGINE_MODULE}"],
                    "$slice": -400,
                }
            },
        },
    )
    try:
        process = subprocess.Popen(
            command,
            cwd=str(SOURCE_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=child_environment,
        )
        _register_active_process(job_id, process)
        db[JOBS_COLLECTION].update_one(
            {"id": job_id},
            {"$set": {"process_id": process.pid, "updated_at": utc_now()}},
        )
        timed_out = threading.Event()
        cancel_watch_stop = threading.Event()

        def terminate_for_timeout() -> None:
            if process.poll() is None:
                timed_out.set()
                try:
                    process.kill()
                except ProcessLookupError:
                    pass

        def watch_for_cancellation() -> None:
            while not cancel_watch_stop.wait(0.25):
                if process.poll() is not None:
                    return
                refreshed = db[JOBS_COLLECTION].find_one({"id": job_id}, {"cancel_requested": 1}) or {}
                if bool(refreshed.get("cancel_requested")):
                    _terminate_process(process)
                    return

        timeout_timer = threading.Timer(timeout_seconds, terminate_for_timeout)
        timeout_timer.daemon = True
        timeout_timer.start()
        cancel_watcher = threading.Thread(target=watch_for_cancellation, daemon=True)
        cancel_watcher.start()
        try:
            assert process.stdout is not None
            for line in process.stdout:
                _write_child_line_to_console(job_id, line)
                append_log(job_id, line)
            return_code = process.wait()
        finally:
            cancel_watch_stop.set()
            timeout_timer.cancel()
            _unregister_active_process(job_id, process)

        refreshed_job = db[JOBS_COLLECTION].find_one({"id": job_id}, {"cancel_requested": 1}) or {}
        if bool(refreshed_job.get("cancel_requested")):
            db[JOBS_COLLECTION].update_one(
                {"id": job_id},
                {
                    "$set": {
                        "status": "cancelled",
                        "stage": "Interrupted",
                        "finished_at": utc_now(),
                        "updated_at": utc_now(),
                        "return_code": return_code,
                    },
                    "$unset": {"process_id": ""},
                },
            )
            return

        if timed_out.is_set():
            append_log(job_id, "ERROR: Training exceeded the configured time limit.")
            db[JOBS_COLLECTION].update_one(
                {"id": job_id},
                {
                    "$set": {
                        "status": "failed",
                        "stage": "Backtest failed",
                        "finished_at": utc_now(),
                        "updated_at": utc_now(),
                        "return_code": return_code,
                        "timed_out": True,
                    },
                    "$unset": {"process_id": ""},
                },
            )
            if certifies_strategy:
                mark_strategy_backtest(db, strategy_id=strategy_profile_id, strategy_revision=strategy_profile_revision, job_id=job_id, status="failed", research_model_family=research_model_family, research_model_settings=research_model_settings)
            return
        run_count = db[RUNS_COLLECTION].count_documents({"job_id": job_id})
        comparison_exists = db[COMPARISONS_COLLECTION].find_one({"job_id": job_id}, {"_id": 1}) is not None
        if return_code == 0 and comparison_exists and run_count > 0:
            db[JOBS_COLLECTION].update_one({"id": job_id}, {"$set": {"status": "completed", "stage": "Completed", "progress": 100, "completed_runs": run_count, "finished_at": utc_now(), "updated_at": utc_now(), "return_code": return_code}, "$unset": {"process_id": ""}})
            if certifies_strategy:
                mark_strategy_backtest(db, strategy_id=strategy_profile_id, strategy_revision=strategy_profile_revision, job_id=job_id, status="completed", research_model_family=research_model_family, research_model_settings=research_model_settings)
            return
        stored = db[COMPARISONS_COLLECTION].find_one({"job_id": job_id}, {"_id": 0, "failures": 1}) or {}
        for failure in stored.get("failures", []):
            append_log(job_id, f"ERROR {failure.get('symbol', 'unknown')}/{failure.get('backend', 'unknown')}: {failure.get('error', 'Unknown error')}")
        if return_code != 0 and not stored.get("failures"):
            append_log(job_id, f"ERROR: Backtest engine exited with code {return_code}.")
        db[JOBS_COLLECTION].update_one({"id": job_id}, {"$set": {"status": "failed", "stage": "Backtest failed", "finished_at": utc_now(), "updated_at": utc_now(), "return_code": return_code}, "$unset": {"process_id": ""}})
        if certifies_strategy:
            mark_strategy_backtest(db, strategy_id=strategy_profile_id, strategy_revision=strategy_profile_revision, job_id=job_id, status="failed", research_model_family=research_model_family, research_model_settings=research_model_settings)
    except Exception as exc:
        logger.exception("Backtest %s failed in the API worker", job_id)
        append_log(job_id, f"ERROR: {exc}")
        db[JOBS_COLLECTION].update_one({"id": job_id}, {"$set": {"status": "failed", "stage": "Backtest failed", "finished_at": utc_now(), "updated_at": utc_now(), "error": str(exc)}, "$unset": {"process_id": ""}})
        if certifies_strategy:
            mark_strategy_backtest(db, strategy_id=strategy_profile_id, strategy_revision=strategy_profile_revision, job_id=job_id, status="failed", research_model_family=research_model_family, research_model_settings=research_model_settings)
