from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from ...infrastructure.persistence.mongo_repository import (
    TEMPORAL_INTELLIGENCE_RUNS_COLLECTION,
    TEMPORAL_POLICY_SEARCH_COLLECTION,
    bson_value,
)
from ..temporal_policy_tuning import _load_artifact_rows, _load_observations
from .caro import run_caro_refinement
from .controlled_comparison import controlled_comparison
from .decision_log import log_entry
from .policy import (
    available_folds,
    build_trajectory_thresholds,
    filter_observations,
    filter_winner_rows,
    replay_search_policy,
)
from .sampling import evaluate_latin_hypercube, settings_hash
from .search_space import base_settings, search_space_payload
from .validation import nested_validation


class TemporalPolicySearchError(RuntimeError):
    pass


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _public(document: dict[str, Any] | None) -> dict[str, Any] | None:
    if document is None:
        return None
    payload = {key: value for key, value in document.items() if key != "_id"}
    return bson_value(payload)


def _completed_run(db: Any, run_id: str) -> dict[str, Any]:
    run = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find_one({"id": str(run_id)})
    if run is None:
        raise TemporalPolicySearchError("Temporal Intelligence run not found.")
    if str(run.get("status") or "").lower() != "completed":
        raise TemporalPolicySearchError("Temporal Policy Search requires a completed Temporal Intelligence run.")
    return run


def _campaign(db: Any, run_id: str, search_id: str) -> dict[str, Any]:
    document = db[TEMPORAL_POLICY_SEARCH_COLLECTION].find_one({"id": str(search_id), "run_id": str(run_id)})
    if document is None:
        raise TemporalPolicySearchError("Temporal Policy Search campaign not found.")
    return document


def _save(db: Any, document: dict[str, Any]) -> dict[str, Any]:
    document["updated_at"] = _utc_now()
    db[TEMPORAL_POLICY_SEARCH_COLLECTION].replace_one(
        {"id": str(document["id"])},
        bson_value(document),
        upsert=True,
    )
    return _public(document) or {}


def _runtime_inputs(db: Any, run: dict[str, Any], *, start_month: str, end_month: str) -> dict[str, Any]:
    run_id = str(run["id"])
    observations = filter_observations(
        _load_observations(db, run_id),
        start_month=start_month,
        end_month=end_month,
    )
    winner_rows = _load_artifact_rows(db, run_id, "winner_reference_daily")
    winner_rows = filter_winner_rows(winner_rows, observations)
    if not observations or not winner_rows:
        raise TemporalPolicySearchError("Frozen Temporal observations or Winner replay are unavailable for the selected period.")
    folds = available_folds(observations)
    if len(folds) < 2:
        raise TemporalPolicySearchError(
            "Nested temporal validation requires at least two frozen folds in the selected study period."
        )
    request = run.get("request") if isinstance(run.get("request"), dict) else {}
    initial_capital = float(request.get("initial_capital") or 10_000.0)
    one_side_cost = max(0.0, float(request.get("slippage_bps") or 0.0) / 10_000.0) + max(
        0.0,
        float(request.get("commission_rate") or 0.0),
    )
    result = run.get("result") if isinstance(run.get("result"), dict) else {}
    fold_rows = result.get("multi_horizon_fold_metrics") if isinstance(result.get("multi_horizon_fold_metrics"), list) else []
    winner_fold_returns = {
        int(item.get("fold_id")): float(item.get("winner_reference_return") or 0.0)
        for item in fold_rows
        if isinstance(item, dict) and item.get("fold_id") is not None
    }
    return {
        "observations": observations,
        "winner_rows": winner_rows,
        "fold_ids": folds,
        "initial_capital": initial_capital,
        "one_side_cost": one_side_cost,
        "winner_fold_returns": winner_fold_returns,
    }


def _winner_reference(run: dict[str, Any]) -> dict[str, Any]:
    result = run.get("result") if isinstance(run.get("result"), dict) else {}
    direct = result.get("winner_reference") if isinstance(result.get("winner_reference"), dict) else {}
    multi = result.get("multi_horizon_metrics") if isinstance(result.get("multi_horizon_metrics"), dict) else {}
    shadow = multi.get("shadow_capital") if isinstance(multi.get("shadow_capital"), dict) else {}
    return {
        "ending_capital": direct.get("ending_capital") or shadow.get("winner_ending_capital"),
        "cagr": direct.get("cagr") or shadow.get("winner_cagr"),
        "sharpe": direct.get("sharpe") or shadow.get("winner_sharpe"),
        "maximum_drawdown": direct.get("maximum_drawdown") or direct.get("max_drawdown") or shadow.get("winner_max_drawdown"),
        "capital_rotations": direct.get("switch_count") or shadow.get("winner_switch_count"),
    }


def create_temporal_policy_search(
    db: Any,
    run_id: str,
    *,
    start_month: str,
    end_month: str,
    processing_id: str | None,
    lhs_trials: int,
    caro_trials: int,
    seed: int,
) -> dict[str, Any]:
    if end_month < start_month:
        raise TemporalPolicySearchError("end_month must be greater than or equal to start_month.")
    if lhs_trials < 4:
        raise TemporalPolicySearchError("Latin Hypercube requires at least four trials.")
    if caro_trials < 1:
        raise TemporalPolicySearchError("CARO requires at least one adaptive trial.")
    run = _completed_run(db, run_id)
    runtime = _runtime_inputs(db, run, start_month=start_month, end_month=end_month)
    now = _utc_now()
    search_id = f"{now.strftime('%Y%m%dT%H%M%S')}-policy-search-{uuid4().hex[:8]}"
    document: dict[str, Any] = {
        "schema_version": 2,
        "id": search_id,
        "run_id": str(run_id),
        "processing_id": str(processing_id or "") or None,
        "status": "prepared",
        "created_at": now,
        "updated_at": now,
        "period_start": str(start_month),
        "period_end": str(end_month),
        "seed": int(seed),
        "budgets": {"latin_hypercube_trials": int(lhs_trials), "caro_trials_per_outer_fold": int(caro_trials)},
        "search_space": search_space_payload(run, fold_ids=runtime["fold_ids"]),
        "sampling": None,
        "caro": None,
        "validation": None,
        "controlled_comparison": None,
        "final_candidate": None,
        "decision_log": [
            log_entry(
                "search_space",
                "prepared",
                "Temporal policy search space prepared on frozen Winner-Anchored observations.",
                {"outer_folds": runtime["fold_ids"], "lhs_trials": lhs_trials, "caro_trials": caro_trials},
            )
        ],
    }
    return _save(db, document)


def get_temporal_policy_search(db: Any, run_id: str, search_id: str) -> dict[str, Any]:
    return _public(_campaign(db, run_id, search_id)) or {}


def get_latest_temporal_policy_search(
    db: Any,
    run_id: str,
    *,
    start_month: str | None = None,
    end_month: str | None = None,
) -> dict[str, Any] | None:
    query: dict[str, Any] = {"run_id": str(run_id)}
    if start_month:
        query["period_start"] = str(start_month)
    if end_month:
        query["period_end"] = str(end_month)
    document = db[TEMPORAL_POLICY_SEARCH_COLLECTION].find_one(query, sort=[("created_at", -1)])
    return _public(document)


def run_temporal_policy_sampling(db: Any, run_id: str, search_id: str) -> dict[str, Any]:
    document = _campaign(db, run_id, search_id)
    run = _completed_run(db, run_id)
    runtime = _runtime_inputs(db, run, start_month=document["period_start"], end_month=document["period_end"])
    base = dict((document.get("search_space") or {}).get("base_settings") or base_settings(run))
    outer_folds: list[dict[str, Any]] = []
    for outer_fold_id in runtime["fold_ids"]:
        inner_fold_ids = [item for item in runtime["fold_ids"] if item != outer_fold_id]
        inner_observations = filter_observations(runtime["observations"], fold_ids=set(inner_fold_ids))
        inner_winner_rows = filter_winner_rows(runtime["winner_rows"], inner_observations)

        def evaluate(settings: dict[str, Any]) -> dict[str, Any]:
            context = build_trajectory_thresholds(inner_observations, settings)
            return replay_search_policy(
                inner_observations,
                inner_winner_rows,
                initial_capital=runtime["initial_capital"],
                one_side_cost=runtime["one_side_cost"],
                settings=settings,
                winner_fold_returns=runtime["winner_fold_returns"],
                trajectory_context=context,
            )

        result = evaluate_latin_hypercube(
            base,
            candidate_count=int((document.get("budgets") or {}).get("latin_hypercube_trials") or 24),
            seed=int(document.get("seed") or 42) + int(outer_fold_id) * 1009,
            evaluate=evaluate,
        )
        outer_folds.append({
            "outer_fold_id": int(outer_fold_id),
            "inner_fold_ids": inner_fold_ids,
            **result,
        })
    document["sampling"] = {
        "status": "completed",
        "method": "latin_hypercube",
        "outer_folds": outer_folds,
        "completed_at": _utc_now(),
    }
    document["caro"] = None
    document["validation"] = None
    document["controlled_comparison"] = None
    document["final_candidate"] = None
    document["status"] = "sampling_completed"
    document.setdefault("decision_log", []).append(log_entry(
        "sampling",
        "completed",
        "Latin Hypercube exploration completed independently inside every outer-fold research partition.",
        {"outer_fold_count": len(outer_folds), "trials_per_partition": int((document.get("budgets") or {}).get("latin_hypercube_trials") or 24)},
    ))
    return _save(db, document)


def run_temporal_policy_caro(db: Any, run_id: str, search_id: str) -> dict[str, Any]:
    document = _campaign(db, run_id, search_id)
    if not isinstance(document.get("sampling"), dict) or document["sampling"].get("status") != "completed":
        raise TemporalPolicySearchError("Run Latin Hypercube sampling before CARO refinement.")
    run = _completed_run(db, run_id)
    runtime = _runtime_inputs(db, run, start_month=document["period_start"], end_month=document["period_end"])
    base = dict((document.get("search_space") or {}).get("base_settings") or base_settings(run))
    sampling_by_outer = {int(item["outer_fold_id"]): item for item in document["sampling"].get("outer_folds") or []}
    outer_folds: list[dict[str, Any]] = []
    for outer_fold_id in runtime["fold_ids"]:
        sampling_result = sampling_by_outer.get(int(outer_fold_id))
        if not sampling_result:
            raise TemporalPolicySearchError(f"Latin Hypercube result is missing for outer fold {outer_fold_id}.")
        inner_fold_ids = [item for item in runtime["fold_ids"] if item != outer_fold_id]
        inner_observations = filter_observations(runtime["observations"], fold_ids=set(inner_fold_ids))
        inner_winner_rows = filter_winner_rows(runtime["winner_rows"], inner_observations)

        def evaluate(settings: dict[str, Any]) -> dict[str, Any]:
            context = build_trajectory_thresholds(inner_observations, settings)
            return replay_search_policy(
                inner_observations,
                inner_winner_rows,
                initial_capital=runtime["initial_capital"],
                one_side_cost=runtime["one_side_cost"],
                settings=settings,
                winner_fold_returns=runtime["winner_fold_returns"],
                trajectory_context=context,
            )

        result = run_caro_refinement(
            sampling_result,
            base_settings=base,
            trial_count=int((document.get("budgets") or {}).get("caro_trials_per_outer_fold") or 12),
            seed=int(document.get("seed") or 42) + int(outer_fold_id) * 2029,
            evaluate=evaluate,
        )
        outer_folds.append({
            "outer_fold_id": int(outer_fold_id),
            "inner_fold_ids": inner_fold_ids,
            **result,
        })
    document["caro"] = {
        "status": "completed",
        "method": "CARO_gaussian_process",
        "outer_folds": outer_folds,
        "completed_at": _utc_now(),
    }
    document["validation"] = None
    document["controlled_comparison"] = None
    document["final_candidate"] = None
    document["status"] = "caro_completed"
    document.setdefault("decision_log", []).append(log_entry(
        "caro",
        "completed",
        "CARO adaptive refinement completed without using the held-out outer fold outcomes.",
        {"outer_fold_count": len(outer_folds), "trials_per_partition": int((document.get("budgets") or {}).get("caro_trials_per_outer_fold") or 12)},
    ))
    return _save(db, document)


def run_temporal_policy_validation(db: Any, run_id: str, search_id: str) -> dict[str, Any]:
    document = _campaign(db, run_id, search_id)
    if not isinstance(document.get("caro"), dict) or document["caro"].get("status") != "completed":
        raise TemporalPolicySearchError("Run CARO refinement before nested validation.")
    run = _completed_run(db, run_id)
    runtime = _runtime_inputs(db, run, start_month=document["period_start"], end_month=document["period_end"])
    base = dict((document.get("search_space") or {}).get("base_settings") or base_settings(run))
    caro_by_outer = {int(item["outer_fold_id"]): item for item in document["caro"].get("outer_folds") or []}
    outer_payloads: list[dict[str, Any]] = []
    for outer_fold_id in runtime["fold_ids"]:
        caro_result = caro_by_outer.get(int(outer_fold_id))
        if not caro_result:
            raise TemporalPolicySearchError(f"CARO result is missing for outer fold {outer_fold_id}.")
        champion = caro_result.get("champion") if isinstance(caro_result.get("champion"), dict) else None
        if not champion:
            raise TemporalPolicySearchError(f"CARO did not produce a champion for outer fold {outer_fold_id}.")
        outer_payloads.append({
            "outer_fold_id": int(outer_fold_id),
            "inner_fold_ids": [item for item in runtime["fold_ids"] if item != outer_fold_id],
            "champion": champion,
        })

    def evaluate_outer(outer_fold_id: int, settings: dict[str, Any]) -> dict[str, Any]:
        inner_ids = {item for item in runtime["fold_ids"] if item != outer_fold_id}
        inner_observations = filter_observations(runtime["observations"], fold_ids=inner_ids)
        outer_observations = filter_observations(runtime["observations"], fold_ids={outer_fold_id})
        outer_winner_rows = filter_winner_rows(runtime["winner_rows"], outer_observations)
        context = build_trajectory_thresholds(inner_observations, settings)
        return replay_search_policy(
            outer_observations,
            outer_winner_rows,
            initial_capital=runtime["initial_capital"],
            one_side_cost=runtime["one_side_cost"],
            settings=settings,
            winner_fold_returns=runtime["winner_fold_returns"],
            trajectory_context=context,
        )

    def evaluate_baseline_outer(outer_fold_id: int) -> dict[str, Any]:
        inner_ids = {item for item in runtime["fold_ids"] if item != outer_fold_id}
        inner_observations = filter_observations(runtime["observations"], fold_ids=inner_ids)
        outer_observations = filter_observations(runtime["observations"], fold_ids={outer_fold_id})
        outer_winner_rows = filter_winner_rows(runtime["winner_rows"], outer_observations)
        context = build_trajectory_thresholds(inner_observations, base)
        return replay_search_policy(
            outer_observations,
            outer_winner_rows,
            initial_capital=runtime["initial_capital"],
            one_side_cost=runtime["one_side_cost"],
            settings=base,
            winner_fold_returns=runtime["winner_fold_returns"],
            trajectory_context=context,
        )

    validation = nested_validation(
        outer_payloads,
        initial_capital=runtime["initial_capital"],
        winner_fold_returns=runtime["winner_fold_returns"],
        evaluate_outer=evaluate_outer,
        evaluate_baseline_outer=evaluate_baseline_outer,
    )
    document["validation"] = {"status": "completed", "completed_at": _utc_now(), **validation}
    document["controlled_comparison"] = None
    document["final_candidate"] = None
    document["status"] = "validation_completed"
    document.setdefault("decision_log", []).append(log_entry(
        "validation",
        "supported" if validation.get("supported") else "not_supported",
        "Nested temporal validation completed on outer folds that were not used to select their candidate settings.",
        {"outer_fold_count": validation.get("outer_fold_count"), "criteria": validation.get("criteria")},
    ))
    return _save(db, document)


def _full_period_refit(
    document: dict[str, Any],
    runtime: dict[str, Any],
    base: dict[str, Any],
) -> dict[str, Any]:
    def evaluate(settings: dict[str, Any]) -> dict[str, Any]:
        context = build_trajectory_thresholds(runtime["observations"], settings)
        return replay_search_policy(
            runtime["observations"],
            runtime["winner_rows"],
            initial_capital=runtime["initial_capital"],
            one_side_cost=runtime["one_side_cost"],
            settings=settings,
            winner_fold_returns=runtime["winner_fold_returns"],
            trajectory_context=context,
        )

    seed = int(document.get("seed") or 42)
    lhs_trials = int((document.get("budgets") or {}).get("latin_hypercube_trials") or 24)
    caro_trials = int((document.get("budgets") or {}).get("caro_trials_per_outer_fold") or 12)
    sampling = evaluate_latin_hypercube(
        base,
        candidate_count=lhs_trials,
        seed=seed + 104729,
        evaluate=evaluate,
    )
    caro = run_caro_refinement(
        sampling,
        base_settings=base,
        trial_count=caro_trials,
        seed=seed + 130363,
        evaluate=evaluate,
    )
    champion = deepcopy(caro.get("champion") or sampling.get("champion") or {})
    settings = dict(champion.get("settings") or base)
    fingerprint = str(champion.get("settings_hash") or settings_hash(settings))
    return {
        "id": f"final-{fingerprint[:16]}",
        "status": "ready",
        "created_at": _utc_now(),
        "selection_method": "post_validation_full_frozen_refit",
        "evaluation_role": "post_validation_descriptive_replay",
        "acceptance_source": "nested_outer_fold_validation",
        "kind": str(champion.get("kind") or "policy_search"),
        "is_control": bool(champion.get("is_control")),
        "source_candidate_id": champion.get("candidate_id"),
        "settings": settings,
        "settings_hash": fingerprint,
        "metrics": deepcopy(champion.get("metrics") or {}),
        "search": {
            "latin_hypercube_trials": lhs_trials,
            "caro_trials": caro_trials,
            "seed": seed,
            "latin_hypercube_evaluated": int(sampling.get("evaluated_count") or 0),
            "caro_completed": int(caro.get("completed_count") or 0),
        },
    }


def _replay_analytics(
    replay: dict[str, Any],
    baseline_replay: dict[str, Any],
    *,
    initial_capital: float,
    one_side_cost: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    intervals = list(replay.get("intervals") or [])
    baseline_intervals = {str(item.get("decision_date")): item for item in baseline_replay.get("intervals") or []}
    equity: list[dict[str, Any]] = []
    rotations: list[dict[str, Any]] = []
    contexts: list[dict[str, Any]] = []
    monthly_details: dict[str, Any] = {}
    capital = float(initial_capital)
    baseline_capital = float(initial_capital)
    peak = capital
    entry_equity: float | None = None
    entry_at: str | None = None

    for interval in intervals:
        decision_date = str(interval.get("decision_date") or "")
        execution_date = str(interval.get("execution_date") or decision_date)
        next_execution_date = str(interval.get("next_execution_date") or execution_date)
        from_asset = str(interval.get("from_asset") or "CASH")
        selected_asset = str(interval.get("selected_asset") or "CASH")
        capital_before = capital
        transition = from_asset != selected_asset
        cost_sides = int(interval.get("cost_sides") or 0)

        if transition:
            realized_pnl = None
            holding_days = None
            if from_asset != "CASH" and entry_equity is not None:
                realized_pnl = float(capital_before - entry_equity)
                if entry_at:
                    try:
                        holding_days = max(0, (datetime.fromisoformat(execution_date.replace("Z", "+00:00")) - datetime.fromisoformat(entry_at.replace("Z", "+00:00"))).days)
                    except ValueError:
                        holding_days = None
            cost_factor = max(1e-9, 1.0 - float(cost_sides) * float(one_side_cost))
            movement = {
                "executed_at": execution_date,
                "from_asset": from_asset,
                "to_asset": selected_asset,
                "holding_days": holding_days,
                "realized_pnl": realized_pnl,
                "transaction_fees": float(capital_before * (1.0 - cost_factor)),
                "sell_reason": "POLICY_SEARCH_ROTATION" if from_asset != "CASH" else None,
                "buy_reason": "POLICY_SEARCH_ROTATION" if selected_asset != "CASH" else None,
            }
            rotations.append(movement)
            month = execution_date[:7]
            monthly_details.setdefault(month, {"month": month, "movements": []})["movements"].append(deepcopy(movement))
            if selected_asset != "CASH":
                entry_equity = float(capital_before * cost_factor)
                entry_at = execution_date
            else:
                entry_equity = None
                entry_at = None
        elif selected_asset != "CASH" and entry_equity is None:
            entry_equity = capital_before
            entry_at = execution_date

        capital *= max(1e-9, 1.0 + float(interval.get("net_return") or 0.0))
        baseline_interval = baseline_intervals.get(decision_date)
        if baseline_interval is not None:
            baseline_capital *= max(1e-9, 1.0 + float(baseline_interval.get("net_return") or 0.0))
        peak = max(peak, capital)
        equity.append({
            "timestamp": next_execution_date,
            "simulation_equity": float(capital),
            "reference_equity": float(baseline_capital),
            "drawdown": float(capital / peak - 1.0),
        })
        action = "HOLD" if from_asset == selected_asset else ("CASH" if selected_asset == "CASH" else "ROTATE")
        reason = "late_exit_cash_guard" if interval.get("late_exit_cash_guard") else ("timing_override" if interval.get("timing_override") else "winner_anchor")
        contexts.append({
            "fold_id": int(interval.get("fold_id") or 0),
            "decision_at": decision_date,
            "execution_at": execution_date,
            "next_execution_at": next_execution_date,
            "current_symbol": from_asset,
            "target_symbol": selected_asset,
            "action": action,
            "reason": reason,
            "outcome": {
                "gross_interval_return": float(interval.get("gross_return") or 0.0),
                "net_interval_return": float(interval.get("net_return") or 0.0),
                "cost_sides": cost_sides,
                "strategy_equity": float(capital),
                "strategy_drawdown": float(capital / peak - 1.0),
            },
            "policy_search": {
                "proposed_asset": interval.get("proposed_asset"),
                "base_asset": interval.get("base_asset"),
                "top_1_asset": interval.get("top_1_asset"),
                "top_2_asset": interval.get("top_2_asset"),
                "timing_override": bool(interval.get("timing_override")),
                "late_exit_risk": bool(interval.get("late_exit_risk")),
                "late_exit_cash_guard": bool(interval.get("late_exit_cash_guard")),
                "challenger_advantage": interval.get("challenger_advantage"),
                "trajectory_state": deepcopy(interval.get("trajectory_state")),
            },
        })

    analytics = {
        "equity": equity,
        "rotations": rotations,
        "metrics": deepcopy(replay.get("metrics") or {}),
        "reference_metrics": deepcopy(baseline_replay.get("metrics") or {}),
    }
    decision_context = {"items": contexts, "count": len(contexts)}
    return analytics, decision_context, monthly_details


def run_temporal_policy_comparison(db: Any, run_id: str, search_id: str) -> dict[str, Any]:
    document = _campaign(db, run_id, search_id)
    validation = document.get("validation") if isinstance(document.get("validation"), dict) else None
    if not validation or validation.get("status") != "completed":
        raise TemporalPolicySearchError("Run nested temporal validation before controlled comparison.")
    run = _completed_run(db, run_id)
    comparison = controlled_comparison(validation, _winner_reference(run))
    document["schema_version"] = max(2, int(document.get("schema_version") or 1))
    document["controlled_comparison"] = {
        **comparison,
        "status": "completed",
        "completed_at": _utc_now(),
    }
    document["final_candidate"] = None
    if comparison.get("outcome") == "supported":
        runtime = _runtime_inputs(db, run, start_month=document["period_start"], end_month=document["period_end"])
        base = dict((document.get("search_space") or {}).get("base_settings") or base_settings(run))
        document["final_candidate"] = _full_period_refit(document, runtime, base)
    document["status"] = "completed"
    document.setdefault("decision_log", []).append(log_entry(
        "controlled_comparison",
        str(comparison.get("outcome") or "completed"),
        "Controlled comparison finalized using stitched outer-fold results.",
        {
            "decision": comparison.get("decision"),
            "rejection_reasons": comparison.get("rejection_reasons"),
            "final_candidate_id": (document.get("final_candidate") or {}).get("id"),
        },
    ))
    return _save(db, document)


def run_temporal_policy_study(db: Any, run_id: str, search_id: str) -> dict[str, Any]:
    document = _campaign(db, run_id, search_id)
    comparison = document.get("controlled_comparison") if isinstance(document.get("controlled_comparison"), dict) else {}
    final_candidate = document.get("final_candidate") if isinstance(document.get("final_candidate"), dict) else None
    if comparison.get("status") != "completed" or comparison.get("outcome") != "supported" or not final_candidate:
        raise TemporalPolicySearchError("Controlled comparison must support a final policy candidate before Run Study.")
    run = _completed_run(db, run_id)
    runtime = _runtime_inputs(db, run, start_month=document["period_start"], end_month=document["period_end"])
    settings = dict(final_candidate.get("settings") or {})
    context = build_trajectory_thresholds(runtime["observations"], settings)
    replay = replay_search_policy(
        runtime["observations"],
        runtime["winner_rows"],
        initial_capital=runtime["initial_capital"],
        one_side_cost=runtime["one_side_cost"],
        settings=settings,
        winner_fold_returns=runtime["winner_fold_returns"],
        trajectory_context=context,
    )
    base = dict((document.get("search_space") or {}).get("base_settings") or base_settings(run))
    base_context = build_trajectory_thresholds(runtime["observations"], base)
    baseline_replay = replay_search_policy(
        runtime["observations"],
        runtime["winner_rows"],
        initial_capital=runtime["initial_capital"],
        one_side_cost=runtime["one_side_cost"],
        settings=base,
        winner_fold_returns=runtime["winner_fold_returns"],
        trajectory_context=base_context,
    )
    analytics, decision_context, monthly_details = _replay_analytics(
        replay,
        baseline_replay,
        initial_capital=runtime["initial_capital"],
        one_side_cost=runtime["one_side_cost"],
    )
    return bson_value({
        "schema_version": 1,
        "source_kind": "policy_search_candidate",
        "executed_at": _utc_now(),
        "start_month": document["period_start"],
        "end_month": document["period_end"],
        "run_id": str(run_id),
        "processing_id": document.get("processing_id"),
        "policy_search_id": str(search_id),
        "policy_candidate": deepcopy(final_candidate),
        "parameters": {
            "temporal_run_id": str(run_id),
            "source_processing_id": document.get("processing_id"),
            "research_snapshot_cutoff": run.get("research_snapshot_cutoff") or run.get("analysis_end_date"),
            "period_start": document["period_start"],
            "period_end": document["period_end"],
            **settings,
            "one_side_cost_rate": runtime["one_side_cost"],
        },
        "analytics": analytics,
        "decision_context": decision_context,
        "monthly_details": monthly_details,
        "policy_search": _public(document),
    })
