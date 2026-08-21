from __future__ import annotations

from datetime import datetime, timezone
from statistics import fmean, median
from typing import Any, Iterable

from fastapi import HTTPException

from ..infrastructure.persistence.mongo_repository import (
    COMPARISONS_COLLECTION,
    JOBS_COLLECTION,
    MODEL_TUNING_RUNS_COLLECTION,
    PAPER_TRADE_PLANS_COLLECTION,
    PREDICTIONS_COLLECTION,
    RUNS_COLLECTION,
    STRATEGY_CONTROL_COLLECTION,
    STRATEGY_PROFILES_COLLECTION,
    bson_value,
)
from .model_tuning import public_model_tuning_run
from .serialization import downsample_documents, iso_value
from .strategy_lab import get_strategy


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
    "average_cash_weight": "average_cash_weight",
    "average_assets_held": "average_assets_held",
    "maximum_assets_held": "maximum_assets_held",
    "allocation_rebalances": "allocation_rebalances",
    "average_primary_weight": "average_primary_weight",
    "average_primary_share_of_risk": "average_primary_share_of_risk",
    "average_secondary_weight": "average_secondary_weight",
    "opportunity_cash_gate_enabled": "opportunity_cash_gate_enabled",
    "cash_days": "cash_days",
    "opportunity_gate_decisions": "opportunity_gate_decisions",
    "opportunity_gate_accepted": "opportunity_gate_accepted",
    "opportunity_gate_rejected": "opportunity_gate_rejected",
    "opportunity_gate_acceptance_rate": "opportunity_gate_acceptance_rate",
    "opportunity_entry_threshold_mean": "opportunity_entry_threshold_mean",
    "opportunity_exit_threshold_mean": "opportunity_exit_threshold_mean",
    "opportunity_gate_adaptive_refreshes": "opportunity_gate_adaptive_refreshes",
    "opportunity_gate_regularized_sessions": "opportunity_gate_regularized_sessions",
    "opportunity_target_horizon_sessions": "opportunity_target_horizon_sessions",
    "cash_gate_changed_base_action_sessions": "cash_gate_changed_base_action_sessions",
    "cash_gate_entries": "cash_gate_entries",
    "cash_gate_exits": "cash_gate_exits",
    "cash_gate_counterfactual_negative_sessions": "cash_gate_counterfactual_negative_sessions",
    "cash_gate_counterfactual_positive_sessions": "cash_gate_counterfactual_positive_sessions",
    "cash_gate_avoided_loss_return_sum": "cash_gate_avoided_loss_return_sum",
    "cash_gate_missed_gain_return_sum": "cash_gate_missed_gain_return_sum",
    "cash_gate_net_avoided_return_sum": "cash_gate_net_avoided_return_sum",
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
    public_job_filter = {"internal_job": {"$ne": True}}
    total_backtests = int(db[JOBS_COLLECTION].count_documents(public_job_filter))
    completed_backtests = int(db[JOBS_COLLECTION].count_documents({**public_job_filter, "status": "completed"}))
    failed_backtests = int(db[JOBS_COLLECTION].count_documents({**public_job_filter, "status": "failed"}))
    interrupted_backtests = int(db[JOBS_COLLECTION].count_documents({**public_job_filter, "status": "interrupted"}))

    recent_jobs = list(
        db[JOBS_COLLECTION]
        .find(public_job_filter, _PUBLIC_JOB_PROJECTION)
        .sort("created_at", -1)
        .limit(safe_limit)
    )
    completed_jobs = list(
        db[JOBS_COLLECTION]
        .find({**public_job_filter, "status": "completed"}, _PUBLIC_JOB_PROJECTION)
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
        "selected_strategy_research_name": _selected_backtest_strategy_name(db),
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


_STRATEGY_DECISION_FIELDS = (
    "decision_date",
    "selected_asset",
    "previous_asset",
    "trade_action",
    "decision_score",
    "strategy_risk_off_enabled",
    "strategy_selective_opportunity_enabled",
    "strategy_absolute_utility_cash_gate_enabled",
    "absolute_utility_best_score",
    "absolute_utility_entry_threshold",
    "absolute_utility_exit_threshold",
    "absolute_utility_active_threshold",
    "absolute_utility_accepted",
    "opportunity_probability",
    "opportunity_confidence",
    "opportunity_threshold",
    "opportunity_accepted",
    "current_asset",
    "current_score",
    "current_cash_edge",
    "holding_days_at_decision",
    "raw_best_asset",
    "raw_best_score",
    "best_asset",
    "best_score",
    "best_cash_edge",
    "second_asset",
    "second_score",
    "second_cash_edge",
    "best_vs_second_gap",
    "best_vs_current_gap",
    "best_vs_cash_gap",
    "cash_score",
    "cash_exit_threshold",
    "cash_entry_threshold",
    "rotation_cash_threshold",
    "rotation_min_expected_edge",
    "base_switch_margin",
    "calibrated_switch_margin",
    "effective_switch_margin",
    "final_action_asset",
    "final_action_score",
    "final_action_cash_edge",
    "decision_reason",
    "decision_is_rotation",
    "decision_is_entry",
    "decision_is_exit_to_cash",
    "min_hold_guard_applied",
    "switch_margin_guard_applied",
    "cash_threshold_guard_applied",
    "minimum_expected_edge_guard_applied",
    "top_1_asset",
    "top_1_score",
    "top_1_cash_edge",
    "top_2_asset",
    "top_2_score",
    "top_2_cash_edge",
    "top_3_asset",
    "top_3_score",
    "top_3_cash_edge",
    "current_asset_rank",
    "universe_score_mean",
    "universe_score_std",
    "current_score_zscore",
    "best_score_zscore",
    "best_vs_second_zscore",
    "positive_score_count",
    "finite_score_count",
)


def _strategy_profile_detail(db: Any, strategy_id: str | None) -> dict[str, Any] | None:
    if not strategy_id:
        return None
    try:
        return get_strategy(db, str(strategy_id))
    except Exception:
        return None


def _strategy_control_ids(db: Any) -> tuple[str | None, str | None]:
    control = db[STRATEGY_CONTROL_COLLECTION].find_one(
        {"_id": "default"},
        {"_id": 0, "research_strategy_id": 1, "trader_winner_strategy_id": 1},
    ) or {}
    research_id = str(control.get("research_strategy_id") or "").strip() or None
    winner_id = str(control.get("trader_winner_strategy_id") or "").strip() or None
    return research_id, winner_id


def _latest_strategy_forecast(db: Any, winner_strategy: dict[str, Any] | None) -> dict[str, Any] | None:
    query: dict[str, Any] = {"status": {"$in": ["prepared", "executing", "executed"]}}
    winner_id = str((winner_strategy or {}).get("id") or "").strip()
    if winner_id:
        query["winner_strategy_id"] = winner_id
    plan = db[PAPER_TRADE_PLANS_COLLECTION].find_one(
        query,
        sort=[("created_at", -1)],
    )
    if plan is None:
        return None

    utilities = plan.get("utilities") if isinstance(plan.get("utilities"), dict) else {}
    cash_edges = plan.get("cash_edges") if isinstance(plan.get("cash_edges"), dict) else {}
    assets = list(dict.fromkeys([
        *[str(item).upper() for item in plan.get("winner_assets") or []],
        *[str(item).upper() for item in utilities if str(item).upper() != "CASH"],
        *[str(item).upper() for item in cash_edges if str(item).upper() != "CASH"],
    ]))
    ranked = sorted(
        assets,
        key=lambda asset: -(_as_float(utilities.get(asset)) if _as_float(utilities.get(asset)) is not None else float("-inf")),
    )
    asset_rows = []
    for rank, asset in enumerate(ranked, start=1):
        asset_rows.append({
            "asset": asset,
            "rank": rank,
            "ranking_utility": _as_float(utilities.get(asset)),
            "cash_edge": _as_float(cash_edges.get(asset)),
            "is_raw_best": asset == str(plan.get("raw_best_asset") or "").upper(),
            "is_target": asset == str(plan.get("target_asset") or "").upper(),
            "is_current": asset == str(plan.get("current_asset") or "").upper(),
        })

    winner_config = (winner_strategy or {}).get("configuration") if isinstance(winner_strategy, dict) else {}
    cash_exit_threshold = _as_float((winner_config or {}).get("rotation_cash_threshold"))
    minimum_edge = _as_float((winner_config or {}).get("rotation_min_expected_edge"))
    cash_entry_threshold = (
        cash_exit_threshold + minimum_edge
        if cash_exit_threshold is not None and minimum_edge is not None
        else None
    )
    return bson_value({
        "source": "paper_next_open_plan",
        "plan_id": plan.get("plan_id"),
        "status": plan.get("status"),
        "strategy_id": plan.get("winner_strategy_id"),
        "strategy_name": plan.get("winner_strategy_name"),
        "strategy_revision": plan.get("winner_strategy_revision"),
        "model_family": plan.get("winner_model_family"),
        "decision_date": plan.get("decision_date"),
        "expected_market_open": plan.get("expected_market_open"),
        "execution_session": plan.get("execution_session"),
        "current_asset": plan.get("current_asset"),
        "target_asset": plan.get("target_asset"),
        "raw_best_asset": plan.get("raw_best_asset"),
        "action": plan.get("action"),
        "selected_utility": _as_float(plan.get("selected_utility")),
        "utilities": {str(key): _as_float(value) for key, value in utilities.items()},
        "cash_edges": {str(key): _as_float(value) for key, value in cash_edges.items()},
        "opportunity_probability": _as_float(plan.get("opportunity_probability")),
        "opportunity_confidence": _as_float(plan.get("opportunity_confidence")),
        "opportunity_threshold": _as_float(plan.get("opportunity_threshold")),
        "opportunity_accepted": plan.get("opportunity_accepted"),
        "asset_forecast": asset_rows,
        "cash_exit_threshold": cash_exit_threshold,
        "cash_entry_threshold": cash_entry_threshold,
        "minimum_expected_edge": minimum_edge,
        "effective_switch_margin": _as_float(plan.get("effective_switch_margin")),
        "calibrated_candidate_margin": _as_float(plan.get("calibrated_candidate_margin")),
        "calibration_score": _as_float(plan.get("calibration_score")),
        "random_state": plan.get("random_state"),
        "training_end": plan.get("training_end"),
        "calibration_start": plan.get("calibration_start"),
        "calibration_end": plan.get("calibration_end"),
        "final_fit_end": plan.get("final_fit_end"),
        "created_at": plan.get("created_at"),
    })


def _strategy_decision_history(db: Any, job_id: str | None) -> dict[str, Any] | None:
    if not job_id:
        return None
    job = db[JOBS_COLLECTION].find_one(
        {"id": str(job_id)},
        {"_id": 0, "id": 1, "status": 1, "strategy_profile_name": 1, "strategy_profile_id": 1},
    )
    if job is None:
        return None
    comparison = db[COMPARISONS_COLLECTION].find_one(
        {"job_id": str(job_id)},
        {"_id": 0, "job_id": 1, "results": 1},
    )
    selected = _selected_internal_row(comparison)
    if selected is None:
        return {"job_id": str(job_id), "rows": [], "metrics": None}
    backend = str(selected.get("backend") or "").strip()
    run_filter: dict[str, Any] = {"job_id": str(job_id)}
    if backend:
        run_filter["backend"] = backend
    run = db[RUNS_COLLECTION].find_one(
        {**run_filter, "symbol": "PORTFOLIO"},
        {"_id": 0, "symbol": 1, "backend": 1},
    )
    if run is None:
        run = db[RUNS_COLLECTION].find_one(
            run_filter,
            {"_id": 0, "symbol": 1, "backend": 1},
        )
    if run is None:
        return {"job_id": str(job_id), "rows": [], "metrics": bson_value(selected)}

    projection = {
        "_id": 0,
        "timestamp": 1,
        "strategy_equity": 1,
        "buy_hold_equity": 1,
    }
    projection.update({field: 1 for field in _STRATEGY_DECISION_FIELDS})
    rows = list(
        db[PREDICTIONS_COLLECTION]
        .find(
            {
                "job_id": str(job_id),
                "symbol": run.get("symbol"),
                "backend": run.get("backend"),
            },
            projection,
        )
        .sort("timestamp", 1)
    )
    result_rows: list[dict[str, Any]] = []
    for row in downsample_documents(rows, maximum_points=800):
        item: dict[str, Any] = {
            "timestamp": iso_value(row.get("timestamp")),
            "simulation_equity": _as_float(row.get("strategy_equity")),
            "reference_equity": _as_float(row.get("buy_hold_equity")),
        }
        for field in _STRATEGY_DECISION_FIELDS:
            value = row.get(field)
            if field == "decision_date":
                item[field] = iso_value(value)
            else:
                item[field] = bson_value(value)
        result_rows.append(item)
    return {
        "job_id": str(job_id),
        "status": job.get("status"),
        "strategy_profile_id": job.get("strategy_profile_id"),
        "strategy_profile_name": job.get("strategy_profile_name"),
        "metrics": bson_value(selected),
        "rows": result_rows,
    }


def _latest_tuning_for_strategy(db: Any, strategy_id: str | None) -> dict[str, Any] | None:
    if not strategy_id:
        return None
    document = db[MODEL_TUNING_RUNS_COLLECTION].find_one(
        {"strategy_profile_id": str(strategy_id)},
        sort=[("created_at", -1)],
    )
    return public_model_tuning_run(db, document)


def dashboard_strategy_intelligence(
    db: Any,
    *,
    job_id: str | None = None,
) -> dict[str, Any]:
    




    research_id, winner_id = _strategy_control_ids(db)
    research_strategy = _strategy_profile_detail(db, research_id)
    winner_strategy = _strategy_profile_detail(db, winner_id)
    control = db[STRATEGY_CONTROL_COLLECTION].find_one(
        {"_id": "default"},
        {"_id": 0, "candidate_strategy_id": 1, "promoted_candidate_strategy_id": 1},
    ) or {}
    tuning_strategy_id = (
        str(control.get("candidate_strategy_id") or "").strip()
        or str(control.get("promoted_candidate_strategy_id") or "").strip()
        or research_id
    )
    if job_id is None and isinstance(research_strategy, dict):
        job_id = str(research_strategy.get("last_backtest_id") or "").strip() or None
    return {
        "research_strategy": research_strategy,
        "winner_strategy": winner_strategy,
        "forecast": _latest_strategy_forecast(db, winner_strategy),
        "decision_history": _strategy_decision_history(db, job_id),
        "tuning": _latest_tuning_for_strategy(db, tuning_strategy_id),
    }


def dashboard_tuning_candidate_detail(
    db: Any,
    run_id: str,
    candidate_id: int,
) -> dict[str, Any]:
    document = db[MODEL_TUNING_RUNS_COLLECTION].find_one({"id": str(run_id)})
    if document is None:
        raise HTTPException(status_code=404, detail="Model tuning campaign not found.")
    candidate = next(
        (
            item for item in document.get("candidates") or []
            if int(item.get("candidate_id") if item.get("candidate_id") is not None else -1) == int(candidate_id)
        ),
        None,
    )
    if candidate is None:
        raise HTTPException(status_code=404, detail="Model tuning candidate not found.")
    return bson_value({
        "run_id": str(run_id),
        "candidate_id": int(candidate_id),
        "kind": candidate.get("kind"),
        "is_control": bool(candidate.get("is_control")),
        "status": candidate.get("status"),
        "rank": candidate.get("rank"),
        "settings": candidate.get("settings") or {},
        "settings_hash": candidate.get("settings_hash"),
        "metrics": candidate.get("metrics") or None,
        "proposal": candidate.get("proposal") or None,
        "champion_gate": candidate.get("champion_gate") or None,
        "champion_gate_passed": candidate.get("champion_gate_passed"),
        "job_id": candidate.get("job_id"),
        "equity_preview": candidate.get("equity_preview") or [],
        "started_at": candidate.get("started_at"),
        "finished_at": candidate.get("finished_at"),
    })
