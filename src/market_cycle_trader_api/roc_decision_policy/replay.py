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


def _rotation_increment(previous: str, target: str, *, count_cash_transitions: bool) -> int:
    if previous == target:
        return 0
    if not count_cash_transitions and (previous == "CASH" or target == "CASH"):
        return 0
    return 1


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


def _scoped_temporal_rows(
    temporal_curve: list[dict[str, Any]],
    *,
    start_month: str,
    end_month: str,
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in temporal_curve if isinstance(row, dict)]
    rows.sort(key=lambda row: _stamp(row.get("decision_timestamp")) or pd.Timestamp.max.tz_localize("UTC"))
    scoped: list[dict[str, Any]] = []
    for row in rows:
        decision = _stamp(row.get("decision_timestamp"))
        if decision is None:
            continue
        month = decision.strftime("%Y-%m")
        if start_month <= month <= end_month:
            scoped.append(row)
    return scoped


def run_replay(
    *,
    observations: list[dict[str, Any]],
    winner_daily: list[dict[str, Any]],
    temporal_curve: list[dict[str, Any]],
    relative_scores: dict[pd.Timestamp, dict[str, Any]],
    one_side_cost: float,
    initial_capital: float,
    count_cash_transitions_as_rotations: bool,
    start_month: str,
    end_month: str,
    enable_roc: bool = True,
) -> dict[str, Any]:
    temporal_rows = _scoped_temporal_rows(temporal_curve, start_month=start_month, end_month=end_month)
    if len(temporal_rows) < 2:
        raise ValueError("ROC Decision Policy replay does not have enough frozen Temporal sessions in the selected period.")

    winner_by_execution = {
        _stamp(row.get("timestamp")): dict(row)
        for row in winner_daily
        if isinstance(row, dict) and _stamp(row.get("timestamp")) is not None
    }
    obs_by_decision: dict[pd.Timestamp, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in observations:
        decision = _stamp(row.get("timestamp"))
        symbol = _asset(row.get("symbol"))
        if decision is not None and symbol != "CASH":
            obs_by_decision[decision][symbol] = dict(row)

    first_equity = finite(temporal_rows[0].get("strategy_equity"))
    if first_equity is None or initial_capital <= 0:
        raise ValueError("ROC Decision Policy replay could not resolve the Temporal starting equity.")

    candidate_capital = float(first_equity)
    current_symbol = _asset(temporal_rows[0].get("current_symbol"))
    equity_values: list[float] = []
    equity_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    roc_override_count = 0
    roc_candidate_count = 0
    roc_signal_qualified_count = 0
    roc_abstention_count = 0
    switch_count = 0
    exposure_days = 0
    cash_days = 0
    temporal_override_count = 0

    for index, temporal in enumerate(temporal_rows):
        decision = _stamp(temporal.get("decision_timestamp"))
        execution = _stamp(temporal.get("execution_date"))
        winner = winner_by_execution.get(execution) or {}
        rows = obs_by_decision.get(decision, {}) if decision is not None else {}
        fold_id = int(temporal.get("fold_id") or winner.get("walk_forward_fold") or winner.get("fold_id") or 0)

        temporal_previous = _asset(temporal.get("current_symbol"))
        temporal_target = _asset(temporal.get("target_symbol"))
        temporal_override = bool(temporal.get("temporal_timing_override"))
        temporal_override_count += int(temporal_override)

        winner_anchor = _asset(temporal.get("winner_anchor_symbol"))
        top1 = _asset(winner.get("top_1_asset") or winner.get("raw_best_asset") or winner.get("best_asset"))
        challenger = _asset(temporal.get("winner_top2_symbol") or winner.get("top_2_asset") or winner.get("second_asset"))
        score = relative_scores.get(decision) if decision is not None else None
        relative_margin = finite((score or {}).get("aggregate_margin"))
        signal_qualified = bool((score or {}).get("signal_qualified"))
        if enable_roc and score is not None:
            roc_candidate_count += 1
            if signal_qualified:
                roc_signal_qualified_count += 1
            else:
                roc_abstention_count += 1

        override = bool(
            enable_roc
            and signal_qualified
            and not temporal_override
            and temporal_target != "CASH"
            and temporal_target == winner_anchor
            and temporal_target == top1
            and challenger not in {"CASH", temporal_target}
            and score is not None
            and _asset(score.get("control_asset")) == temporal_target
            and _asset(score.get("challenger_asset")) == challenger
            and relative_margin is not None
            and relative_margin >= 0.0
        )
        target = challenger if override else temporal_target

        if override and finite((rows.get(target) or {}).get("open_to_open_return")) is None:
            target = temporal_target
            override = False

        if override:
            roc_override_count += 1
        switch_count += _rotation_increment(
            current_symbol,
            target,
            count_cash_transitions=count_cash_transitions_as_rotations,
        )
        if target == "CASH":
            cash_days += 1
        else:
            exposure_days += 1

        equity_values.append(float(candidate_capital))
        equity_rows.append({
            "fold_id": fold_id,
            "decision_timestamp": decision,
            "execution_date": execution,
            "temporal_current_symbol": temporal_previous,
            "temporal_target_symbol": temporal_target,
            "current_symbol": current_symbol,
            "target_symbol": target,
            "temporal_timing_override": temporal_override,
            "roc_override": override,
            "equity": float(candidate_capital),
        })
        diagnostics.append({
            "fold_id": fold_id,
            "decision_timestamp": decision,
            "execution_date": execution,
            "temporal_current_asset": temporal_previous,
            "temporal_target_asset": temporal_target,
            "winner_anchor_asset": winner_anchor,
            "top_1_asset": top1,
            "challenger_asset": challenger,
            "target_asset": target,
            "temporal_timing_override": temporal_override,
            "roc_override": override,
            "roc_signal_qualified": signal_qualified,
            "roc_abstention": bool(enable_roc and score is not None and not signal_qualified),
            "qualified_horizons": (score or {}).get("qualified_horizons") or [],
            "abstained_horizons": (score or {}).get("abstained_horizons") or [],
            "relative_probability": (score or {}).get("aggregate_probability"),
            "relative_threshold": (score or {}).get("aggregate_threshold"),
            "relative_margin": relative_margin,
            "relative_horizons": (score or {}).get("horizons") or [],
        })

        if index < len(temporal_rows) - 1:
            current_temporal_equity = finite(temporal.get("strategy_equity"))
            next_temporal_equity = finite(temporal_rows[index + 1].get("strategy_equity"))
            if current_temporal_equity in {None, 0.0} or next_temporal_equity is None:
                raise ValueError("ROC Decision Policy replay encountered an invalid Temporal equity interval.")
            temporal_factor = float(next_temporal_equity / current_temporal_equity)

            if target == temporal_target and current_symbol == temporal_previous:
                candidate_factor = temporal_factor
            else:
                control_row = rows.get(temporal_target) or {}
                candidate_row = rows.get(target) or {}
                control_return = finite(control_row.get("open_to_open_return")) if temporal_target != "CASH" else 0.0
                candidate_return = finite(candidate_row.get("open_to_open_return")) if target != "CASH" else 0.0
                control_sides = _cost_sides(temporal_previous, temporal_target)
                candidate_sides = _cost_sides(current_symbol, target)

                if target != "CASH" and candidate_return is None:
                    target = temporal_target
                    override = False
                    candidate_row = rows.get(target) or {}
                    candidate_return = finite(candidate_row.get("open_to_open_return")) if target != "CASH" else 0.0
                    candidate_sides = _cost_sides(current_symbol, target)

                expected_control = max(1e-9, 1.0 - control_sides * float(one_side_cost))
                if control_return is not None:
                    expected_control *= max(1e-9, 1.0 + float(control_return))
                residual = temporal_factor / expected_control if expected_control > 0 else 1.0
                candidate_factor = (
                    residual
                    * max(1e-9, 1.0 - candidate_sides * float(one_side_cost))
                    * max(1e-9, 1.0 + float(candidate_return or 0.0))
                )
            candidate_capital *= max(1e-9, float(candidate_factor))
        current_symbol = target

    metrics = equity_metrics(equity_values, float(initial_capital))
    metrics.update({
        "switch_count": int(switch_count),
        "roc_override_count": int(roc_override_count),
        "roc_candidate_count": int(roc_candidate_count),
        "roc_signal_qualified_count": int(roc_signal_qualified_count),
        "roc_abstention_count": int(roc_abstention_count),
        "temporal_timing_override_count": int(temporal_override_count),
        "exposure": exposure_days / max(1, len(equity_values)),
        "cash_days": int(cash_days),
        "decision_policy": "temporal_control_plus_qualified_relative_roc_rotation" if enable_roc else "temporal_control_parity_replay",
    })
    return {
        "metrics": metrics,
        "folds": _fold_metrics(equity_rows),
        "diagnostics": diagnostics,
        "equity": equity_rows,
    }
