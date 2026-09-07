from __future__ import annotations

import statistics
from typing import Any

import pandas as pd

from ..infrastructure.persistence.mongo_repository import COMPARISONS_COLLECTION, JOBS_COLLECTION


_INSTALLED = False
_MAX_BASELINE_RELATIVE_DRIFT = 0.005


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


def _normalized_assets(values: Any) -> list[str]:
    return [
        str(value or "").strip().upper()
        for value in list(values or [])
        if str(value or "").strip()
    ]


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


def _source_id(strategy: dict[str, Any]) -> str:
    return str(strategy.get("_id") or strategy.get("id") or "").strip()


def _certified_job(service: Any, db: Any, strategy: dict[str, Any], evaluation_end: str) -> tuple[dict[str, Any], dict[str, Any]]:
    source_id = _source_id(strategy)
    source_revision = int(strategy.get("revision") or 1)
    source_hash = str(strategy.get("configuration_hash") or "").strip()
    last_backtest_id = str(strategy.get("last_backtest_id") or "").strip()

    if not source_id or not source_hash or not last_backtest_id:
        raise RuntimeError(
            "Asset Discovery requires an exact completed certified backtest for the current Strategy revision before research can start."
        )
    if str(strategy.get("last_backtest_status") or "").strip().lower() != "completed":
        raise RuntimeError("Asset Discovery certified Strategy backtest is not completed.")
    if int(strategy.get("last_backtest_revision") or 0) != source_revision:
        raise RuntimeError("Asset Discovery certified Strategy backtest revision does not match the current Strategy revision.")

    job = db[JOBS_COLLECTION].find_one({"id": last_backtest_id}) or {}
    if str(job.get("status") or "").strip().lower() != "completed":
        raise RuntimeError("Asset Discovery could not load the completed certified Strategy backtest.")
    if str(job.get("strategy_profile_id") or "").strip() != source_id:
        raise RuntimeError("Asset Discovery certified backtest belongs to another Strategy.")
    if int(job.get("strategy_profile_revision") or 0) != source_revision:
        raise RuntimeError("Asset Discovery certified backtest revision is stale.")
    if str(job.get("strategy_configuration_hash") or "").strip() != source_hash:
        raise RuntimeError("Asset Discovery certified backtest configuration hash is stale.")

    request = job.get("request") if isinstance(job.get("request"), dict) else {}
    if not request:
        raise RuntimeError("Asset Discovery certified backtest request snapshot is unavailable.")
    if _iso_date(request.get("analysis_end_date")) != _iso_date(evaluation_end):
        raise RuntimeError(
            "Asset Discovery snapshot end differs from the certified Strategy backtest. Run a new certified Strategy backtest for the current market snapshot first."
        )

    try:
        model_snapshot = service.get_strategy_model_snapshot(db, source_id)
    except Exception as exc:
        raise RuntimeError("Asset Discovery could not resolve the current Strategy model snapshot.") from exc

    current_model_hash = str(model_snapshot.get("settings_hash") or "").strip()
    certified_model_hash = str(job.get("research_model_settings_hash") or "").strip()
    if current_model_hash and certified_model_hash and current_model_hash != certified_model_hash:
        raise RuntimeError(
            "Asset Discovery current model settings differ from the certified Strategy backtest. Run a new certified Strategy backtest first."
        )

    return job, request


def _certified_ending_capital(db: Any, job: dict[str, Any]) -> float:
    backtest_id = str(job.get("id") or "").strip()
    comparison = db[COMPARISONS_COLLECTION].find_one({"job_id": backtest_id}) or {}
    raw_rows = comparison.get("results") if isinstance(comparison.get("results"), list) else []
    model_family = str(job.get("research_model_family") or "").strip()
    rows = [
        row
        for row in raw_rows
        if isinstance(row, dict)
        and bool(row.get("portfolio_rotation"))
        and (
            not model_family
            or str(row.get("model_family") or row.get("backend") or "").strip() == model_family
        )
    ]
    if not rows:
        rows = [row for row in raw_rows if isinstance(row, dict) and bool(row.get("portfolio_rotation"))]
    ending_capital = _median(rows, "strategy_ending_capital")
    if ending_capital is None or ending_capital <= 0.0:
        raise RuntimeError("Asset Discovery certified Strategy ending capital is unavailable.")
    return ending_capital


def install_asset_discovery_certified_baseline() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from . import asset_discovery as service

    original_request = service._marginal_execution_request
    original_replay = service._run_marginal_capital_replay

    if getattr(original_request, "_asset_discovery_certified_baseline", False):
        _INSTALLED = True
        return

    def certified_execution_request(
        db: Any,
        base_config: Any,
        strategy: dict[str, Any],
        winner_config: Any,
        end_session: str,
        *,
        assets: list[str],
        reference_assets: list[str],
        candidate_assets: list[str],
        analysis_start_date: str | None = None,
        analysis_end_date: str | None = None,
    ) -> Any:
        request = original_request(
            db,
            base_config,
            strategy,
            winner_config,
            end_session,
            assets=assets,
            reference_assets=reference_assets,
            candidate_assets=candidate_assets,
            analysis_start_date=analysis_start_date,
            analysis_end_date=analysis_end_date,
        )
        job, certified = _certified_job(service, db, strategy, analysis_end_date or end_session)

        certified_assets = _normalized_assets(certified.get("assets"))
        baseline_assets = _normalized_assets(reference_assets)
        if certified_assets != baseline_assets:
            raise RuntimeError(
                "Asset Discovery baseline assets do not match the exact asset universe of the certified Strategy backtest."
            )

        certified_anchors = _normalized_assets(certified.get("calendar_anchor_assets"))
        certified_reference = _normalized_assets(certified.get("research_reference_assets"))
        certified_candidates = _normalized_assets(certified.get("research_candidate_assets"))
        new_candidates = _normalized_assets(candidate_assets)

        if not certified_anchors:
            certified_anchors = list(certified_reference or certified_assets)
        if not certified_reference:
            certified_reference = list(certified_assets)

        combined_candidates = list(dict.fromkeys([*certified_candidates, *new_candidates]))
        certified_family = str(certified.get("research_model_family") or "").strip()
        certified_settings = (
            dict(certified.get("research_model_settings") or {})
            if isinstance(certified.get("research_model_settings"), dict)
            else None
        )

        updates: dict[str, Any] = {
            "calendar_anchor_assets": certified_anchors,
            "research_reference_assets": certified_reference,
            "research_candidate_assets": combined_candidates,
        }
        if certified_family:
            updates["research_model_family"] = certified_family
        if certified_settings is not None:
            updates["research_model_settings"] = certified_settings

        return request.model_copy(update=updates)

    def replay_with_certified_baseline(db: Any, run_id: str, *args: Any, **kwargs: Any) -> Any:
        strategy = kwargs.get("strategy")
        end_session = str(kwargs.get("end_session") or "").strip()
        if not isinstance(strategy, dict) or not end_session:
            return original_replay(db, run_id, *args, **kwargs)

        job, _ = _certified_job(service, db, strategy, end_session)
        certified_capital = _certified_ending_capital(db, job)

        retained, summary = original_replay(db, run_id, *args, **kwargs)
        replay_summary = dict(summary or {})
        baseline = replay_summary.get("baseline") if isinstance(replay_summary.get("baseline"), dict) else {}
        replay_capital = _finite(baseline.get("ending_capital"))
        if replay_capital is None or replay_capital <= 0.0:
            raise RuntimeError("Asset Discovery baseline replay produced no valid ending capital.")

        relative_drift = abs(replay_capital / certified_capital - 1.0)
        parity = {
            "status": "passed" if relative_drift <= _MAX_BASELINE_RELATIVE_DRIFT else "failed",
            "certified_backtest_id": str(job.get("id") or ""),
            "certified_ending_capital": certified_capital,
            "replayed_ending_capital": replay_capital,
            "relative_drift": relative_drift,
            "maximum_relative_drift": _MAX_BASELINE_RELATIVE_DRIFT,
        }
        replay_summary["certified_baseline"] = parity

        db[service.COLLECTION].update_one(
            {"_id": service.CURRENT_ID, "run_id": run_id},
            {"$set": {
                "certified_baseline": service.bson_value(parity),
                "updated_at": service.utc_now(),
            }},
        )

        if relative_drift > _MAX_BASELINE_RELATIVE_DRIFT:
            raise RuntimeError(
                "Asset Discovery baseline parity check failed: the fresh Strategy replay does not reproduce the certified Strategy capital. "
                f"Certified={certified_capital:.6f}, replay={replay_capital:.6f}, drift={relative_drift:.4%}."
            )
        return retained, replay_summary

    setattr(certified_execution_request, "_asset_discovery_certified_baseline", True)
    setattr(replay_with_certified_baseline, "_asset_discovery_certified_baseline", True)
    service._marginal_execution_request = certified_execution_request
    service._run_marginal_capital_replay = replay_with_certified_baseline
    _INSTALLED = True
