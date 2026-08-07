from __future__ import annotations

from datetime import datetime, timezone
from statistics import fmean, median
from typing import Any, Iterable

from fastapi import HTTPException

from ..infrastructure.persistence.mongo_repository import (
    COMPARISONS_COLLECTION,
    JOBS_COLLECTION,
    PREDICTIONS_COLLECTION,
    RUNS_COLLECTION,
    STRATEGY_CONTROL_COLLECTION,
    STRATEGY_PROFILES_COLLECTION,
)
from .serialization import downsample_documents, iso_value


_PUBLIC_JOB_PROJECTION = {
    "_id": 0,
    "id": 1,
    "status": 1,
    "stage": 1,
    "progress": 1,
    "created_at": 1,
    "updated_at": 1,
    "started_at": 1,
    "finished_at": 1,
    "strategy_profile_name": 1,
}

_METRIC_KEYS = {
    "starting_capital": "initial_capital",
    "ending_capital": "strategy_ending_capital",
    "simulation_return": "strategy_return",
    "reference_ending_capital": "buy_hold_ending_capital",
    "reference_return": "buy_hold_return",
    "cagr": "strategy_cagr",
    "reference_cagr": "buy_hold_cagr",
    "sharpe": "strategy_sharpe",
    "reference_sharpe": "buy_hold_sharpe",
    "maximum_drawdown": "strategy_maximum_drawdown",
    "reference_maximum_drawdown": "buy_hold_maximum_drawdown",
    "market_exposure": "market_exposure",
    "session_win_rate": "session_win_rate",
    "position_changes": "capital_rotations",
    "average_holding_days": "average_holding_days",
}


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def _public_metric_rows(comparison: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows = comparison.get("results", []) if comparison else []
    if not isinstance(rows, list):
        return []
    portfolio_rows = [row for row in rows if isinstance(row, dict) and row.get("portfolio_rotation")]
    return portfolio_rows or [row for row in rows if isinstance(row, dict)]


def _median_value(rows: Iterable[dict[str, Any]], source_key: str) -> float | None:
    values = [number for row in rows if (number := _as_float(row.get(source_key))) is not None]
    return float(median(values)) if values else None


def _public_metrics(comparison: dict[str, Any] | None) -> dict[str, float | None] | None:
    rows = _public_metric_rows(comparison)
    if not rows:
        return None
    payload = {
        public_key: _median_value(rows, source_key)
        for public_key, source_key in _METRIC_KEYS.items()
    }
    return payload if any(value is not None for value in payload.values()) else None


def _as_utc(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _duration_seconds(job: dict[str, Any]) -> int | None:
    start = _as_utc(job.get("started_at")) or _as_utc(job.get("created_at"))
    end = _as_utc(job.get("finished_at")) or _as_utc(job.get("updated_at"))
    if start is None or end is None or end < start:
        return None
    return int((end - start).total_seconds())




def _public_stage(job: dict[str, Any]) -> str:
    status = str(job.get("status") or "").lower()
    if status == "completed":
        return "Completed"
    if status == "failed":
        return "Backtest failed"
    if status == "interrupted":
        return "Interrupted"
    if status == "queued":
        return "Queued"
    if status == "running":
        return "Running analysis"
    return "Pending"

def _public_job_summary(
    job: dict[str, Any],
    metrics: dict[str, float | None] | None,
) -> dict[str, Any]:
    return {
        "id": str(job.get("id") or ""),
        "status": str(job.get("status") or "unknown"),
        "stage": _public_stage(job),
        "progress": _as_float(job.get("progress")) or 0.0,
        "created_at": iso_value(job.get("created_at")),
        "started_at": iso_value(job.get("started_at")),
        "finished_at": iso_value(job.get("finished_at")),
        "duration_seconds": _duration_seconds(job),
        "strategy_profile_name": str(job.get("strategy_profile_name") or "") or None,
        "metrics": metrics,
    }


def _comparison_map(db: Any, job_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not job_ids:
        return {}
    documents = db[COMPARISONS_COLLECTION].find(
        {"job_id": {"$in": job_ids}},
        {"_id": 0, "job_id": 1, "results": 1},
    )
    return {
        str(document.get("job_id")): document
        for document in documents
        if document.get("job_id")
    }


def _selected_backtest_strategy_name(db: Any) -> str | None:
    """Return only the public display name of the strategy selected for backtests."""
    try:
        control = db[STRATEGY_CONTROL_COLLECTION].find_one(
            {"_id": "default"},
            {"_id": 0, "research_strategy_id": 1},
        )
        strategy_id = str((control or {}).get("research_strategy_id") or "")
        if not strategy_id:
            return None
        profile = db[STRATEGY_PROFILES_COLLECTION].find_one(
            {"_id": strategy_id},
            {"_id": 0, "name": 1},
        )
        name = str((profile or {}).get("name") or "").strip()
        return name or None
    except (KeyError, TypeError, AttributeError):
        return None


def dashboard_summary(db: Any, *, limit: int = 10) -> dict[str, Any]:
    safe_limit = max(1, min(50, int(limit)))
    total_backtests = int(db[JOBS_COLLECTION].count_documents({}))
    completed_backtests = int(db[JOBS_COLLECTION].count_documents({"status": "completed"}))
    failed_backtests = int(db[JOBS_COLLECTION].count_documents({"status": "failed"}))
    interrupted_backtests = int(db[JOBS_COLLECTION].count_documents({"status": "interrupted"}))

    recent_jobs = list(
        db[JOBS_COLLECTION]
        .find({}, _PUBLIC_JOB_PROJECTION)
        .sort("created_at", -1)
        .limit(safe_limit)
    )
    completed_jobs = list(
        db[JOBS_COLLECTION]
        .find({"status": "completed"}, _PUBLIC_JOB_PROJECTION)
        .sort("created_at", -1)
    )

    all_job_ids = list(
        dict.fromkeys(
            str(job.get("id"))
            for job in [*recent_jobs, *completed_jobs]
            if job.get("id")
        )
    )
    comparisons = _comparison_map(db, all_job_ids)
    metrics_by_job = {
        job_id: _public_metrics(comparison)
        for job_id, comparison in comparisons.items()
    }

    completed_with_metrics = [
        (job, metrics_by_job.get(str(job.get("id"))))
        for job in completed_jobs
        if metrics_by_job.get(str(job.get("id"))) is not None
    ]
    valid_returns = [
        (job, metrics)
        for job, metrics in completed_with_metrics
        if metrics and metrics.get("simulation_return") is not None
    ]
    best_pair = max(
        valid_returns,
        key=lambda item: float(item[1]["simulation_return"]),
        default=None,
    )
    sharpe_values = [
        float(metrics["sharpe"])
        for _, metrics in completed_with_metrics
        if metrics and metrics.get("sharpe") is not None
    ]
    profitable_count = sum(
        1
        for _, metrics in valid_returns
        if float(metrics["simulation_return"]) > 0
    )

    recent_backtests = [
        _public_job_summary(
            job,
            metrics_by_job.get(str(job.get("id"))),
        )
        for job in recent_jobs
    ]
    last_backtest = recent_backtests[0] if recent_backtests else None

    best_performance = None
    if best_pair is not None:
        best_performance = _public_job_summary(best_pair[0], best_pair[1])

    return {
        "total_backtests": total_backtests,
        "completed_backtests": completed_backtests,
        "failed_backtests": failed_backtests,
        "interrupted_backtests": interrupted_backtests,
        "average_sharpe": float(fmean(sharpe_values)) if sharpe_values else None,
        "profitable_backtest_rate": (
            profitable_count / len(valid_returns)
            if valid_returns
            else None
        ),
        "best_performance": best_performance,
        "last_backtest": last_backtest,
        "recent_backtests": recent_backtests,
        "selected_backtest_strategy_name": _selected_backtest_strategy_name(db),
    }


def _selected_internal_row(comparison: dict[str, Any] | None) -> dict[str, Any] | None:
    rows = _public_metric_rows(comparison)
    if not rows:
        return None
    returns = [
        number
        for row in rows
        if (number := _as_float(row.get("strategy_return"))) is not None
    ]
    if not returns:
        return rows[0]
    target = float(median(returns))
    return min(
        rows,
        key=lambda row: abs((_as_float(row.get("strategy_return")) or target) - target),
    )


def _public_series(db: Any, job_id: str, comparison: dict[str, Any] | None) -> list[dict[str, Any]]:
    selected = _selected_internal_row(comparison)
    if selected is None:
        return []

    backend = str(selected.get("backend") or "").strip()
    run_filter: dict[str, Any] = {"job_id": job_id}
    if backend:
        run_filter["backend"] = backend

    run = db[RUNS_COLLECTION].find_one(
        {**run_filter, "symbol": "PORTFOLIO"},
        {"_id": 0, "job_id": 1, "symbol": 1, "backend": 1},
    )
    if run is None:
        run = db[RUNS_COLLECTION].find_one(
            run_filter,
            {"_id": 0, "job_id": 1, "symbol": 1, "backend": 1},
        )
    if run is None:
        return []

    prediction_filter = {
        "job_id": job_id,
        "symbol": run.get("symbol"),
        "backend": run.get("backend"),
    }
    rows = list(
        db[PREDICTIONS_COLLECTION]
        .find(
            prediction_filter,
            {
                "_id": 0,
                "timestamp": 1,
                "strategy_equity": 1,
                "buy_hold_equity": 1,
            },
        )
        .sort("timestamp", 1)
    )
    return [
        {
            "timestamp": iso_value(row.get("timestamp")),
            "simulation_equity": _as_float(row.get("strategy_equity")),
            "reference_equity": _as_float(row.get("buy_hold_equity")),
        }
        for row in downsample_documents(rows)
    ]


def dashboard_job_detail(db: Any, job_id: str) -> dict[str, Any]:
    job = db[JOBS_COLLECTION].find_one({"id": job_id}, _PUBLIC_JOB_PROJECTION)
    if job is None:
        raise HTTPException(status_code=404, detail="Backtest job not found.")

    comparison = db[COMPARISONS_COLLECTION].find_one(
        {"job_id": job_id},
        {"_id": 0, "job_id": 1, "results": 1},
    )
    metrics = _public_metrics(comparison)
    return {
        **_public_job_summary(job, metrics),
        "series": _public_series(db, job_id, comparison),
    }
