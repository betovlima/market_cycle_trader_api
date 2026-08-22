from __future__ import annotations

import statistics
from typing import Any, Callable

from .config import ROBUST_SELECTION_LIMITS, SEARCH_CANDIDATE_COUNT, SELECTION_COST_BPS
from .metrics import fold_metrics, metrics
from .replay import build_decisions
from .search_space import configurations
from .utils import as_float


def _fold_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("fold_id")): row.get("metrics") or {} for row in rows}


def _delta(candidate: Any, control: Any) -> float:
    candidate_value = as_float(candidate)
    control_value = as_float(control)
    if candidate_value is None or control_value in {None, 0.0}:
        return 0.0
    return candidate_value / control_value - 1.0


def _difference(candidate: Any, control: Any) -> float:
    return float(as_float(candidate, 0.0) or 0.0) - float(as_float(control, 0.0) or 0.0)


def _summary(
    *,
    candidate_id: str,
    configuration: dict[str, Any],
    decisions: list[dict[str, Any]],
    replay_stats: dict[str, Any],
    initial_capital: float,
    base_cost_rate: float,
    control_folds: dict[str, dict[str, Any]],
    control_cost_folds: dict[str, dict[str, Any]],
    calibration_fold_ids: list[str],
) -> dict[str, Any]:
    candidate_folds = _fold_map(fold_metrics(decisions, initial_capital, base_cost_rate))
    candidate_cost_folds = _fold_map(fold_metrics(decisions, initial_capital, SELECTION_COST_BPS / 10000.0))
    capital_deltas: list[float] = []
    cost_deltas: list[float] = []
    sharpe_deltas: list[float] = []
    drawdown_deltas: list[float] = []
    candidate_switches = control_switches = 0.0
    changed_calibration = 0
    calibration_decisions = 0

    for fold_id in calibration_fold_ids:
        candidate = candidate_folds.get(fold_id) or {}
        control = control_folds.get(fold_id) or {}
        candidate_cost = candidate_cost_folds.get(fold_id) or {}
        control_cost = control_cost_folds.get(fold_id) or {}
        capital_deltas.append(_delta(candidate.get("ending_capital"), control.get("ending_capital")))
        cost_deltas.append(_delta(candidate_cost.get("ending_capital"), control_cost.get("ending_capital")))
        sharpe_deltas.append(_difference(candidate.get("sharpe"), control.get("sharpe")))
        drawdown_deltas.append(_difference(candidate.get("maximum_drawdown"), control.get("maximum_drawdown")))
        candidate_switches += float(as_float(candidate.get("capital_rotations"), 0.0) or 0.0)
        control_switches += float(as_float(control.get("capital_rotations"), 0.0) or 0.0)

    calibration_set = set(calibration_fold_ids)
    for item in decisions:
        if str(item.get("fold_id")) in calibration_set:
            calibration_decisions += 1
            if item.get("target_symbol") != item.get("control_target_symbol"):
                changed_calibration += 1
    decision_change_rate = changed_calibration / calibration_decisions if calibration_decisions > 0 else 0.0

    mean_capital = statistics.mean(capital_deltas) if capital_deltas else 0.0
    worst_capital = min(capital_deltas) if capital_deltas else 0.0
    mean_cost = statistics.mean(cost_deltas) if cost_deltas else 0.0
    worst_cost = min(cost_deltas) if cost_deltas else 0.0
    mean_sharpe = statistics.mean(sharpe_deltas) if sharpe_deltas else 0.0
    worst_drawdown = min(drawdown_deltas) if drawdown_deltas else 0.0
    switch_inflation = candidate_switches / control_switches - 1.0 if control_switches > 0 else 0.0
    limits = ROBUST_SELECTION_LIMITS
    checks = {
        "changed_decisions": changed_calibration > 0,
        "mean_capital": mean_capital > float(limits["minimum_mean_capital_delta"]),
        "worst_fold_capital": worst_capital >= float(limits["minimum_worst_fold_capital_delta"]),
        "mean_cost_stress": mean_cost > float(limits["minimum_mean_cost_stress_delta"]),
        "worst_cost_stress": worst_cost >= float(limits["minimum_worst_cost_stress_delta"]),
        "drawdown": worst_drawdown >= -float(limits["maximum_drawdown_deterioration"]),
        "switches": switch_inflation <= float(limits["maximum_switch_inflation"]),
        "decision_change_rate": decision_change_rate <= float(limits["maximum_decision_change_rate"]),
    }
    robust_score = 100.0 * (
        0.30 * mean_capital
        + 0.25 * worst_capital
        + 0.20 * mean_cost
        + 0.10 * worst_cost
        + 0.10 * mean_sharpe
        + 0.05 * worst_drawdown
        - 0.10 * max(0.0, switch_inflation)
        - 0.20 * decision_change_rate
    )
    return {
        "candidate_id": candidate_id,
        "configuration": configuration,
        "passed": all(checks.values()),
        "checks": checks,
        "robust_score": robust_score,
        "calibration": {
            "mean_capital_delta": mean_capital,
            "worst_fold_capital_delta": worst_capital,
            "mean_cost_stress_delta": mean_cost,
            "worst_cost_stress_delta": worst_cost,
            "mean_sharpe_delta": mean_sharpe,
            "worst_drawdown_delta": worst_drawdown,
            "switch_inflation": switch_inflation,
            "different_decisions": changed_calibration,
            "decision_change_rate": decision_change_rate,
        },
        "solver": {
            "decisions_solved": replay_stats.get("decisions_solved"),
            "nodes_explored": replay_stats.get("nodes_explored"),
        },
    }


def optimize(
    *,
    diagnostics: list[dict[str, Any]],
    observations: dict[str, list[dict[str, Any]]],
    economics: dict[str, dict[str, Any]],
    control_decisions: list[dict[str, Any]],
    initial_capital: float,
    base_cost_rate: float,
    start_month: str,
    end_month: str,
    should_stop: Callable[[], bool],
) -> dict[str, Any]:
    control_folds_rows = fold_metrics(control_decisions, initial_capital, base_cost_rate)
    fold_ids = [str(row.get("fold_id")) for row in control_folds_rows]
    if len(fold_ids) < 2:
        return {
            "selection_status": "insufficient_folds",
            "candidate_count": 0,
            "passed_candidate_count": 0,
            "calibration_fold_ids": fold_ids,
            "validation_fold_id": None,
            "selection_used_validation_fold": False,
        }
    calibration_fold_ids = fold_ids[:-1]
    validation_fold_id = fold_ids[-1]
    control_folds = _fold_map(control_folds_rows)
    control_cost_folds = _fold_map(fold_metrics(control_decisions, initial_capital, SELECTION_COST_BPS / 10000.0))
    summaries: list[dict[str, Any]] = []
    best_summary: dict[str, Any] | None = None
    best_decisions: list[dict[str, Any]] | None = None
    best_stats: dict[str, Any] | None = None

    for index, candidate in enumerate(configurations(SEARCH_CANDIDATE_COUNT)):
        if index % 8 == 0 and should_stop():
            raise RuntimeError("MILP policy optimization stopped by user.")
        decisions, stats = build_decisions(
            diagnostics=diagnostics,
            observations=observations,
            economics=economics,
            configuration=dict(candidate["configuration"]),
            start_month=start_month,
            end_month=end_month,
            base_cost_rate=base_cost_rate,
            should_stop=should_stop,
        )
        summary = _summary(
            candidate_id=str(candidate["candidate_id"]),
            configuration=dict(candidate["configuration"]),
            decisions=decisions,
            replay_stats=stats,
            initial_capital=initial_capital,
            base_cost_rate=base_cost_rate,
            control_folds=control_folds,
            control_cost_folds=control_cost_folds,
            calibration_fold_ids=calibration_fold_ids,
        )
        summaries.append(summary)
        if not summary["passed"]:
            continue
        ordering = (
            float(summary["robust_score"]),
            float(summary["calibration"]["worst_fold_capital_delta"]),
            float(summary["calibration"]["mean_cost_stress_delta"]),
            -float(summary["calibration"]["switch_inflation"]),
            str(summary["candidate_id"]),
        )
        previous = None if best_summary is None else (
            float(best_summary["robust_score"]),
            float(best_summary["calibration"]["worst_fold_capital_delta"]),
            float(best_summary["calibration"]["mean_cost_stress_delta"]),
            -float(best_summary["calibration"]["switch_inflation"]),
            str(best_summary["candidate_id"]),
        )
        if previous is None or ordering > previous:
            best_summary = summary
            best_decisions = decisions
            best_stats = stats

    ranked = sorted(
        summaries,
        key=lambda row: (bool(row["passed"]), float(row["robust_score"]), str(row["candidate_id"])),
        reverse=True,
    )
    if best_summary is None or best_decisions is None or best_stats is None:
        return {
            "selection_status": "no_robust_candidate",
            "candidate_count": len(summaries),
            "passed_candidate_count": 0,
            "calibration_fold_ids": calibration_fold_ids,
            "validation_fold_id": validation_fold_id,
            "selection_used_validation_fold": False,
            "search_candidates": summaries,
            "top_candidates": ranked[:12],
        }

    validation_control, _ = build_decisions(
        diagnostics=diagnostics,
        observations=observations,
        economics=economics,
        configuration=dict(best_summary["configuration"]),
        start_month=start_month,
        end_month=end_month,
        base_cost_rate=base_cost_rate,
        should_stop=should_stop,
        force_control=True,
        allowed_fold_ids={validation_fold_id},
    )
    validation_candidate, validation_stats = build_decisions(
        diagnostics=diagnostics,
        observations=observations,
        economics=economics,
        configuration=dict(best_summary["configuration"]),
        start_month=start_month,
        end_month=end_month,
        base_cost_rate=base_cost_rate,
        should_stop=should_stop,
        allowed_fold_ids={validation_fold_id},
    )
    validation_control_metrics = (fold_metrics(validation_control, initial_capital, base_cost_rate)[0].get("metrics") or {})
    validation_candidate_metrics = (fold_metrics(validation_candidate, initial_capital, base_cost_rate)[0].get("metrics") or {})
    validation_control_cost = (fold_metrics(validation_control, initial_capital, SELECTION_COST_BPS / 10000.0)[0].get("metrics") or {})
    validation_candidate_cost = (fold_metrics(validation_candidate, initial_capital, SELECTION_COST_BPS / 10000.0)[0].get("metrics") or {})
    validation = {
        "fold_id": validation_fold_id,
        "used_for_selection": False,
        "capital_delta": _delta(validation_candidate_metrics.get("ending_capital"), validation_control_metrics.get("ending_capital")),
        "cost_stress_delta": _delta(validation_candidate_cost.get("ending_capital"), validation_control_cost.get("ending_capital")),
        "sharpe_delta": _difference(validation_candidate_metrics.get("sharpe"), validation_control_metrics.get("sharpe")),
        "drawdown_delta": _difference(validation_candidate_metrics.get("maximum_drawdown"), validation_control_metrics.get("maximum_drawdown")),
        "switch_delta": int(validation_candidate_metrics.get("capital_rotations") or 0) - int(validation_control_metrics.get("capital_rotations") or 0),
        "different_decisions": validation_stats.get("different_decision"),
        "control_metrics": {key: value for key, value in validation_control_metrics.items() if key != "equity"},
        "candidate_metrics": {key: value for key, value in validation_candidate_metrics.items() if key != "equity"},
    }
    validation["status"] = "passed" if (
        validation["capital_delta"] >= 0.0
        and validation["cost_stress_delta"] >= 0.0
        and validation["drawdown_delta"] >= -float(ROBUST_SELECTION_LIMITS["maximum_drawdown_deterioration"])
    ) else "failed"
    report = {
        "selection_status": "selected",
        "method": "deterministic_low_discrepancy_robust_search",
        "candidate_count": len(summaries),
        "passed_candidate_count": sum(1 for row in summaries if row["passed"]),
        "calibration_fold_ids": calibration_fold_ids,
        "validation_fold_id": validation_fold_id,
        "selection_used_validation_fold": False,
        "selection_cost_bps": SELECTION_COST_BPS,
        "selection_limits": dict(ROBUST_SELECTION_LIMITS),
        "selected_candidate_id": best_summary["candidate_id"],
        "selected_score": best_summary["robust_score"],
        "selected_configuration": dict(best_summary["configuration"]),
        "selected_calibration": dict(best_summary["calibration"]),
        "validation": validation,
        "search_candidates": summaries,
        "top_candidates": ranked[:12],
    }
    report["decisions"] = best_decisions
    report["replay_stats"] = best_stats
    return report
