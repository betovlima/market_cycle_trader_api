from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from research_asset_signature_pipeline_ranking import canonical_hash

JOBS_COLLECTION = "backtest_jobs"


def exact_baseline_job(db: Any, strategy: dict[str, Any], snapshot_end: pd.Timestamp, job_id: str | None) -> dict[str, Any]:
    if job_id:
        job = db[JOBS_COLLECTION].find_one({"id": job_id, "status": "completed"})
    else:
        query = {
            "status": "completed",
            "strategy_profile_id": str(strategy.get("_id") or ""),
            "strategy_profile_revision": int(strategy.get("revision") or 0),
            "strategy_configuration_hash": strategy.get("configuration_hash"),
            "request.analysis_end_date": snapshot_end.date().isoformat(),
            "internal_job": {"$ne": True},
        }
        job = db[JOBS_COLLECTION].find_one(query, sort=[("finished_at", -1), ("created_at", -1)])
    if not job:
        raise RuntimeError("No exact completed baseline Backtest matches Strategy revision/hash/snapshot.")
    request = job.get("request") if isinstance(job.get("request"), dict) else {}
    checks = {
        "strategy_profile_id": str(job.get("strategy_profile_id") or "") == str(strategy.get("_id") or ""),
        "strategy_profile_revision": int(job.get("strategy_profile_revision") or 0) == int(strategy.get("revision") or 0),
        "strategy_configuration_hash": str(job.get("strategy_configuration_hash") or "") == str(strategy.get("configuration_hash") or ""),
        "analysis_end_date": str(request.get("analysis_end_date") or "") == snapshot_end.date().isoformat(),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError("Baseline Backtest mismatch: " + ", ".join(failed))
    return job


def experiment_job(
    baseline_job: dict[str, Any],
    strategy: dict[str, Any],
    baseline_assets: list[str],
    selected: list[str],
    snapshot_end: pd.Timestamp,
    frozen: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    request = deepcopy(baseline_job.get("request") or {})
    if not request:
        raise RuntimeError("Baseline job has no immutable request payload.")
    request["assets"] = list(dict.fromkeys([*baseline_assets, *selected]))
    request["analysis_end_date"] = snapshot_end.date().isoformat()
    request["research_reference_assets"] = list(baseline_assets)
    request["research_candidate_assets"] = list(selected)
    request["research_market_data_mode"] = "database_only"
    anchors = [item for item in request.get("calendar_anchor_assets") or [] if item in set(baseline_assets)]
    request["calendar_anchor_assets"] = anchors if len(anchors) >= 2 else list(baseline_assets)

    now = datetime.now(timezone.utc)
    job = deepcopy(baseline_job)
    job.pop("_id", None)
    for key in ("process_id", "return_code", "timed_out", "error", "cancel_requested", "cancel_reason"):
        job.pop(key, None)
    job.update({
        "id": now.strftime("%Y%m%dT%H%M%S") + "-signature-" + uuid4().hex[:8],
        "status": "queued",
        "stage": "Queued",
        "progress": 0,
        "completed_runs": 0,
        "created_at": now,
        "updated_at": now,
        "started_at": None,
        "finished_at": None,
        "request": request,
        "strategy_profile_name": f"{strategy.get('name') or 'Strategy'} + frozen signature selection",
        "strategy_configuration_hash": canonical_hash(request),
        "source_strategy_configuration_hash": strategy.get("configuration_hash"),
        "research_reference_assets": list(baseline_assets),
        "research_candidate_assets": list(selected),
        "certifies_strategy": False,
        "internal_job": True,
        "tuning_summary_only": False,
        "tuning_run_id": None,
        "tuning_candidate_id": None,
        "logs": ["Research validation Backtest queued after ranking freeze."],
        "progress_detail": {},
        "ranking_snapshot_sha256": frozen["decision_snapshot_sha256"],
        "ranking_sha256": frozen["ranking_sha256"],
        "selected_symbols_sha256": frozen["selected_symbols_sha256"],
        "experiment_kind": "post_ranking_full_backtest_validation",
    })
    return job, request


def _find_numeric(value: Any, aliases: tuple[str, ...]) -> float | None:
    if isinstance(value, dict):
        for alias in aliases:
            if alias in value:
                try:
                    number = float(value[alias])
                    if np.isfinite(number):
                        return number
                except (TypeError, ValueError):
                    pass
        for nested in value.values():
            found = _find_numeric(nested, aliases)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _find_numeric(nested, aliases)
            if found is not None:
                return found
    return None


def normalized_metrics(results: dict[str, Any]) -> dict[str, Any]:
    aliases = {
        "ending_capital": ("ending_capital", "final_capital", "ending_equity"),
        "cagr": ("cagr", "compound_annual_growth_rate"),
        "sharpe": ("sharpe", "sharpe_ratio"),
        "maximum_drawdown": ("maximum_drawdown", "max_drawdown", "max_dd"),
        "worst_fold_return": ("worst_fold_return", "worst_fold"),
        "switches": ("switches", "switch_count", "rotation_count"),
        "cash_days": ("cash_days",),
        "market_exposure": ("market_exposure", "exposure"),
    }
    return {metric: _find_numeric(results, names) for metric, names in aliases.items()}


def compare_results(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    base = normalized_metrics(baseline)
    new = normalized_metrics(candidate)
    deltas: dict[str, Any] = {}
    for metric, old in base.items():
        current = new.get(metric)
        if old is None or current is None:
            delta = rate = None
        else:
            delta = float(current - old)
            rate = float(delta / abs(old)) if abs(old) > 1e-12 else None
        deltas[metric] = {"baseline": old, "candidate": current, "delta": delta, "delta_rate": rate}
    return {
        "baseline_metrics": base,
        "candidate_metrics": new,
        "deltas": deltas,
        "interpretation": "The Backtest validates the frozen ranking and never changes it.",
    }
