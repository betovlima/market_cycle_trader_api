from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

from .metrics import equity_metrics, finite


def _stamp(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _asset(value: Any) -> str:
    text = str(value or "CASH").strip().upper()
    return text or "CASH"


def _cost_sides(previous: str, target: str) -> int:
    if previous == target:
        return 0
    if previous == "CASH" or target == "CASH":
        return 1
    return 2


def _fold_metrics(equity_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in equity_rows:
        grouped[int(row.get("fold_id") or 0)].append(row)
    output: list[dict[str, Any]] = []
    for fold_id, rows in sorted(grouped.items()):
        values = [float(row["equity"]) for row in rows if finite(row.get("equity")) is not None]
        if not values:
            continue
        initial = float(values[0])
        metrics = equity_metrics(values, initial)
        output.append({
            "fold_id": fold_id,
            "test_start": rows[0].get("decision_timestamp"),
            "test_end": rows[-1].get("decision_timestamp"),
            **metrics,
        })
    return output


def run_replay(
    *,
    observations: list[dict[str, Any]],
    winner_daily: list[dict[str, Any]],
    reference_analytics: dict[str, Any],
    thresholds: dict[tuple[int, int], float],
    entry_horizons: list[int],
    one_side_cost: float,
    start_month: str,
    end_month: str,
) -> dict[str, Any]:
    equity_source = [dict(row) for row in (reference_analytics.get("equity") or []) if isinstance(row, dict)]
    equity_source.sort(key=lambda row: _stamp(row.get("timestamp")) or pd.Timestamp.max.tz_localize("UTC"))
    winner_by_execution = {_stamp(row.get("timestamp")): dict(row) for row in winner_daily if isinstance(row, dict) and _stamp(row.get("timestamp")) is not None}
    obs_by_decision: dict[pd.Timestamp, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in observations:
        decision = _stamp(row.get("timestamp"))
        symbol = _asset(row.get("symbol"))
        if decision is not None and symbol != "CASH":
            obs_by_decision[decision][symbol] = dict(row)

    aligned: list[tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]] = []
    for session in equity_source:
        execution = _stamp(session.get("timestamp"))
        winner = winner_by_execution.get(execution)
        if not winner:
            continue
        decision = _stamp(winner.get("decision_date"))
        if decision is None:
            continue
        month = decision.strftime("%Y-%m")
        if month < start_month or month > end_month:
            continue
        aligned.append((session, winner, obs_by_decision.get(decision, {})))
    if len(aligned) < 2:
        raise ValueError("ROC Decision Policy replay does not have enough aligned reference sessions in the selected period.")

    source_metrics = reference_analytics.get("metrics") if isinstance(reference_analytics.get("metrics"), dict) else {}
    first_reference = finite(aligned[0][0].get("simulation_equity"))
    if first_reference is None:
        raise ValueError("ROC Decision Policy replay could not resolve the initial reference equity.")
    original_initial = finite(source_metrics.get("initial_capital"))
    global_first_stamp = _stamp(equity_source[0].get("timestamp")) if equity_source else None
    aligned_first_stamp = _stamp(aligned[0][0].get("timestamp"))
    uses_full_oos_start = global_first_stamp is not None and aligned_first_stamp == global_first_stamp
    initial_capital = float(original_initial if uses_full_oos_start and original_initial is not None else first_reference)
    candidate_capital = float(first_reference)
    current_symbol = _asset(
        aligned[0][1].get("strategy_research_control_previous_asset")
        or aligned[0][1].get("previous_asset")
    )
    equity_values: list[float] = []
    equity_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    override_count = 0
    switch_count = 0
    exposure_days = 0
    cash_days = 0

    def roc_margin(rows: dict[str, dict[str, Any]], symbol: str, fold_id: int) -> tuple[float | None, list[dict[str, Any]]]:
        if symbol == "CASH" or symbol not in rows:
            return None, []
        margins: list[float] = []
        detail: list[dict[str, Any]] = []
        row = rows[symbol]
        for horizon in entry_horizons:
            threshold = thresholds.get((fold_id, int(horizon)))
            probability = finite(row.get(f"profit_before_loss_probability_h{int(horizon)}"))
            if threshold is None or probability is None:
                continue
            margin = float(probability) - float(threshold)
            margins.append(margin)
            detail.append({"horizon": int(horizon), "probability": probability, "threshold": float(threshold), "margin": margin})
        return (sum(margins) / len(margins) if margins else None), detail

    for index, (session, winner, rows) in enumerate(aligned):
        decision = _stamp(winner.get("decision_date"))
        fold_id = int(winner.get("walk_forward_fold") or winner.get("decision_fold_id") or winner.get("fold_id") or 0)
        control_previous = _asset(
            winner.get("strategy_research_control_previous_asset")
            or winner.get("previous_asset")
            or (aligned[index - 1][1].get("strategy_research_control_asset") if index > 0 else current_symbol)
        )
        base = _asset(winner.get("strategy_research_control_asset") or winner.get("selected_asset") or session.get("selected_asset"))
        top1 = _asset(winner.get("top_1_asset") or winner.get("raw_best_asset") or winner.get("best_asset"))
        challenger = _asset(winner.get("top_2_asset") or winner.get("second_asset"))
        base_margin, base_detail = roc_margin(rows, base, fold_id)
        challenger_margin, challenger_detail = roc_margin(rows, challenger, fold_id)
        override = bool(
            base != "CASH"
            and base == top1
            and challenger not in {"CASH", base}
            and base_margin is not None
            and challenger_margin is not None
            and base_margin < 0.0
            and challenger_margin >= 0.0
        )
        target = challenger if override else base
        if target != base and finite((rows.get(target) or {}).get("open_to_open_return")) is None:
            target = base
            override = False
        if override:
            override_count += 1
        if current_symbol != target:
            switch_count += 1
        if target == "CASH":
            cash_days += 1
        else:
            exposure_days += 1
        equity_values.append(float(candidate_capital))
        equity_rows.append({
            "fold_id": fold_id,
            "decision_timestamp": decision,
            "equity": float(candidate_capital),
            "initial_equity": float(equity_values[0]),
        })
        diagnostics.append({
            "fold_id": fold_id,
            "decision_timestamp": decision,
            "base_asset": base,
            "challenger_asset": challenger,
            "target_asset": target,
            "roc_override": override,
            "base_roc_margin": base_margin,
            "challenger_roc_margin": challenger_margin,
            "base_horizons": base_detail,
            "challenger_horizons": challenger_detail,
        })
        if index < len(aligned) - 1:
            current_reference = finite(session.get("simulation_equity"))
            next_reference = finite(aligned[index + 1][0].get("simulation_equity"))
            if current_reference in {None, 0.0} or next_reference is None:
                raise ValueError("ROC Decision Policy replay encountered an invalid reference equity interval.")
            baseline_factor = float(next_reference / current_reference)
            control_row = rows.get(base) or {}
            candidate_row = rows.get(target) or {}
            control_return = finite(control_row.get("open_to_open_return")) if base != "CASH" else 0.0
            candidate_return = finite(candidate_row.get("open_to_open_return")) if target != "CASH" else 0.0
            control_sides = _cost_sides(control_previous, base)
            candidate_sides = _cost_sides(current_symbol, target)
            if target == base and current_symbol == control_previous:
                candidate_factor = baseline_factor
            else:
                expected_control = max(1e-9, 1.0 - control_sides * float(one_side_cost))
                if control_return is not None:
                    expected_control *= max(1e-9, 1.0 + float(control_return))
                residual = baseline_factor / expected_control if expected_control > 0 else 1.0
                candidate_factor = residual * max(1e-9, 1.0 - candidate_sides * float(one_side_cost)) * max(1e-9, 1.0 + float(candidate_return or 0.0))
            candidate_capital *= max(1e-9, float(candidate_factor))
        current_symbol = target

    metrics = equity_metrics(equity_values, initial_capital)
    metrics.update({
        "switch_count": int(switch_count),
        "timing_override_count": int(override_count),
        "exposure": exposure_days / max(1, len(equity_values)),
        "cash_days": int(cash_days),
        "decision_policy": "roc_calibrated_winner_anchored_timing",
    })
    return {
        "metrics": metrics,
        "folds": _fold_metrics(equity_rows),
        "diagnostics": diagnostics,
        "equity": equity_rows,
    }
