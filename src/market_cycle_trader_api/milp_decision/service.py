from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from ..infrastructure.persistence.mongo_repository import TEMPORAL_INTELLIGENCE_RUNS_COLLECTION, bson_value, utc_now
from .config import COST_STRESS_BPS, DEFAULT_CONFIGURATION
from .errors import MilpDecisionError
from .inputs import artifact_rows, observation_rows
from .metrics import fold_metrics, metrics, monthly_decision_map
from .parity import compare as compare_control_parity
from .parity import reference_analytics, reference_path
from .persistence import latest_raw, public_document, save
from .replay import build_decisions
from .utils import as_float


def _stop_requested(db: Any, run_id: str) -> bool:
    state = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find_one(
        {"id": str(run_id)}, {"_id": 0, "strategy_research_pipeline.status": 1}
    ) or {}
    pipeline = state.get("strategy_research_pipeline") if isinstance(state.get("strategy_research_pipeline"), dict) else {}
    return str(pipeline.get("status") or "").lower() in {"stop_requested", "stopped"}


def run(db: Any, run_id: str, *, processing_id: str, start_month: str, end_month: str) -> dict[str, Any]:
    temporal_run = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find_one({"id": str(run_id)}, {"_id": 0})
    if temporal_run is None:
        raise MilpDecisionError("Temporal Intelligence run not found.")
    if str(temporal_run.get("status") or "").lower() != "completed":
        raise MilpDecisionError("Decision Optimization requires a completed Temporal Intelligence run.")
    if end_month < start_month:
        raise MilpDecisionError("end_month must be greater than or equal to start_month.")

    configuration = dict(DEFAULT_CONFIGURATION)
    observations = observation_rows(db, run_id)
    diagnostics, economics = artifact_rows(db, run_id)
    result = temporal_run.get("result") if isinstance(temporal_run.get("result"), dict) else {}
    multi = result.get("multi_horizon_metrics") if isinstance(result.get("multi_horizon_metrics"), dict) else {}
    capital = multi.get("shadow_capital") if isinstance(multi.get("shadow_capital"), dict) else {}
    initial_capital = as_float(capital.get("initial_capital"), 10000.0) or 10000.0
    base_cost_rate = as_float(capital.get("one_side_cost_rate"), 0.0) or 0.0

    source_analytics = reference_analytics(db, processing_id)
    source_metrics = source_analytics.get("metrics") if isinstance(source_analytics.get("metrics"), dict) else {}
    control_path = reference_path(source_analytics)
    count_cash_transitions = bool(control_path.get("count_cash_transitions_as_rotations"))
    source_initial = as_float(source_metrics.get("initial_capital"))
    if source_initial is not None:
        initial_capital = source_initial

    control_decisions, control_stats = build_decisions(
        diagnostics=diagnostics,
        observations=observations,
        economics=economics,
        reference_path=control_path,
        configuration=configuration,
        start_month=start_month,
        end_month=end_month,
        base_cost_rate=base_cost_rate,
        should_stop=lambda: _stop_requested(db, run_id),
        force_control=True,
    )
    control_metrics = metrics(
        control_decisions, initial_capital, base_cost_rate,
        count_cash_transitions_as_rotations=count_cash_transitions,
    )
    control_parity = compare_control_parity(source_analytics, control_metrics)
    if str(control_parity.get("status") or "") != "passed":
        raise MilpDecisionError(
            "MILP Control replay failed exact parity before Decision Optimization. "
            + json.dumps({
                "checks": control_parity.get("checks") or {},
                "reference": control_parity.get("reference") or {},
                "replay": control_parity.get("replay") or {},
            }, sort_keys=True)
        )

    decisions, replay_stats = build_decisions(
        diagnostics=diagnostics,
        observations=observations,
        economics=economics,
        reference_path=control_path,
        configuration=configuration,
        start_month=start_month,
        end_month=end_month,
        base_cost_rate=base_cost_rate,
        should_stop=lambda: _stop_requested(db, run_id),
    )

    result_metrics = metrics(
        decisions, initial_capital, base_cost_rate,
        count_cash_transitions_as_rotations=count_cash_transitions,
    )
    folds = fold_metrics(
        decisions, initial_capital, base_cost_rate,
        count_cash_transitions_as_rotations=count_cash_transitions,
    )
    cost_stress = []
    for bps in COST_STRESS_BPS:
        stressed = metrics(
            decisions, initial_capital, bps / 10000.0,
            count_cash_transitions_as_rotations=count_cash_transitions,
        )
        cost_stress.append({
            "one_side_cost_bps": bps,
            "ending_capital": stressed["ending_capital"],
            "total_return": stressed["total_return"],
            "cagr": stressed["cagr"],
            "sharpe": stressed["sharpe"],
            "maximum_drawdown": stressed["maximum_drawdown"],
        })

    optimization_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-milp-" + uuid.uuid4().hex[:8]
    now = utc_now()
    document = bson_value({
        "id": optimization_id,
        "status": "completed",
        "schema_version": 3,
        "run_id": str(run_id),
        "processing_id": str(processing_id),
        "period_start": str(start_month),
        "period_end": str(end_month),
        "strategy_profile_id": temporal_run.get("strategy_profile_id"),
        "strategy_profile_revision": temporal_run.get("strategy_profile_revision"),
        "strategy_configuration_hash": temporal_run.get("strategy_configuration_hash"),
        "protocol": {
            "decision_horizon_sessions": 1,
            "decision_space": "one_binary_position_across_ranked_assets_and_cash",
            "objective_information": "decision_time_causal_features_only",
            "realized_returns_usage": "post_hoc_replay_folds_cost_stress_and_attribution_only",
            "economic_replay": "exact_control_anchored_relative_overlay",
            "control_parity_required": True,
            "control_path_source": "selected_strategy_processing_analytics",
            "economic_overlay": "exact_reference_factor_with_residual_alternative_return",
            "rotation_semantics": (
                "all_position_changes" if count_cash_transitions else "invested_asset_to_invested_asset"
            ),
            "stateful_candidate_consumed_as_input": False,
            "parallel_candidate_comparison": True,
            "configuration_origin": "fixed_non_tuned_research_baseline",
            "promotion_policy": "research_only_until_live_runtime_parity",
        },
        "control_parity": control_parity,
        "control_replay": {
            "metrics": {key: value for key, value in control_metrics.items() if key != "equity"},
            "solver": {
                "decisions": len(control_decisions),
                "forced_control_decisions": control_stats.get("forced_control_decisions"),
            },
        },
        "solver": {
            "model": "binary_one_hot_milp",
            "algorithm": "deterministic_branch_and_bound",
            "decisions_evaluated": len(decisions),
            "decisions_solved": replay_stats["decisions_solved"],
            "forced_control_decisions": replay_stats["forced_control_decisions"],
            "nodes_explored": replay_stats["nodes_explored"],
            "nodes_pruned": replay_stats["nodes_pruned"],
            "average_solve_ms": replay_stats["average_solve_ms"],
        },
        "configuration": configuration,
        "metrics": {key: value for key, value in result_metrics.items() if key != "equity"},
        "analytics": _analytics_snapshot(
            result_metrics, decisions,
            rotation_semantics=(
                "all_position_changes" if count_cash_transitions else "invested_asset_to_invested_asset"
            ),
        ),
        "folds": folds,
        "cost_stress": cost_stress,
        "decision_map": monthly_decision_map(decisions),
        "attribution": {key: replay_stats[key] for key in ("same_decision", "different_decision", "milp_better", "control_better", "neutral")},
        "decision_samples": sorted(
            [item for item in decisions if item.get("target_symbol") != item.get("control_target_symbol")],
            key=lambda item: abs(float(item.get("decision_value_added_vs_control") or 0.0)),
            reverse=True,
        )[:24],
        "decisions": decisions,
        "created_at": now,
        "updated_at": now,
    })
    return save(db, document)


def _analytics_snapshot(
    result_metrics: dict[str, Any],
    decisions: list[dict[str, Any]],
    *,
    rotation_semantics: str,
) -> dict[str, Any]:
    return {
        "protocol": {"rotation_semantics": str(rotation_semantics)},
        "metrics": {key: value for key, value in result_metrics.items() if key != "equity"},
        "equity": result_metrics["equity"],
        "rotations": [
            {
                "sequence": index + 1,
                "executed_at": item.get("execution_at"),
                "from_asset": item.get("current_symbol"),
                "to_asset": item.get("target_symbol"),
                "holding_days": item.get("holding_days_before"),
                "transaction_fees": 0.0,
                "buy_reason": "milp_decision_optimization",
                "sell_reason": "milp_decision_optimization",
            }
            for index, item in enumerate(decisions)
            if item.get("current_symbol") != item.get("target_symbol")
        ],
    }


def latest(db: Any, run_id: str, *, processing_id: str, start_month: str, end_month: str) -> dict[str, Any] | None:
    return public_document(latest_raw(db, run_id, processing_id=processing_id, start_month=start_month, end_month=end_month))
