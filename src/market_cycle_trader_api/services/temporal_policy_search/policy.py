from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from ..temporal_policy_replay import _finite, _timestamp_key
from .search_space import TRAJECTORY_SIGNALS, normalize_settings


def filter_observations(
    observations: dict[str, dict[str, Any]],
    *,
    fold_ids: set[int] | None = None,
    start_month: str | None = None,
    end_month: str | None = None,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for key, payload in observations.items():
        fold_id = int(payload.get("fold_id") or 0)
        if fold_ids is not None and fold_id not in fold_ids:
            continue
        month = str(key)[:7]
        if start_month and month < start_month:
            continue
        if end_month and month > end_month:
            continue
        result[key] = payload
    return result


def filter_winner_rows(winner_rows: list[dict[str, Any]], observations: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    keys = set(observations)
    result: list[dict[str, Any]] = []
    for row in winner_rows:
        if not isinstance(row, dict):
            continue
        key = _timestamp_key(row.get("decision_date"))
        if key in keys:
            result.append(row)
    return result


def available_folds(observations: dict[str, dict[str, Any]]) -> list[int]:
    return sorted({int(item.get("fold_id") or 0) for item in observations.values() if int(item.get("fold_id") or 0) > 0})


def _deterioration(current: dict[str, Any], previous: dict[str, Any], signal: dict[str, str]) -> float | None:
    current_value = _finite(current.get(signal["name"]))
    previous_value = _finite(previous.get(signal["name"]))
    if current_value is None or previous_value is None:
        return None
    if signal["direction"] == "increase":
        return float(current_value - previous_value)
    return float(previous_value - current_value)


def build_trajectory_thresholds(
    observations: dict[str, dict[str, Any]],
    settings: dict[str, Any],
) -> dict[str, Any]:
    normalized = normalize_settings(settings)
    lookback = int(normalized["trajectory_lookback_sessions"])
    quantile = float(normalized["trajectory_deterioration_quantile"])
    history: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    samples: dict[str, list[float]] = {item["name"]: [] for item in TRAJECTORY_SIGNALS}

    for key in sorted(observations, key=lambda item: pd.Timestamp(item)):
        payload = observations[key]
        fold_id = int(payload.get("fold_id") or 0)
        rows_by_symbol = payload.get("rows_by_symbol") or {}
        for symbol, row in rows_by_symbol.items():
            symbol_history = history[(fold_id, str(symbol))]
            if len(symbol_history) >= lookback:
                previous = symbol_history[-lookback]
                for signal in TRAJECTORY_SIGNALS:
                    value = _deterioration(row, previous, signal)
                    if value is not None and math.isfinite(value):
                        samples[signal["name"]].append(float(value))
            symbol_history.append(row)
            if len(symbol_history) > 8:
                del symbol_history[:-8]

    thresholds: dict[str, float | None] = {}
    counts: dict[str, int] = {}
    for signal in TRAJECTORY_SIGNALS:
        values = samples[signal["name"]]
        counts[signal["name"]] = len(values)
        if not values:
            thresholds[signal["name"]] = None
            continue
        thresholds[signal["name"]] = max(0.0, float(np.quantile(np.asarray(values, dtype=float), quantile)))
    return {
        "lookback_sessions": lookback,
        "quantile": quantile,
        "thresholds": thresholds,
        "sample_counts": counts,
    }


def _trajectory_state(
    current_row: dict[str, Any] | None,
    previous_row: dict[str, Any] | None,
    trajectory_context: dict[str, Any],
) -> dict[str, Any]:
    if not current_row or not previous_row:
        return {"available": 0, "deteriorating": 0, "signals": {}}
    thresholds = trajectory_context.get("thresholds") if isinstance(trajectory_context.get("thresholds"), dict) else {}
    signals: dict[str, Any] = {}
    deteriorating = 0
    available = 0
    for signal in TRAJECTORY_SIGNALS:
        value = _deterioration(current_row, previous_row, signal)
        threshold = _finite(thresholds.get(signal["name"]))
        triggered = None
        if value is not None and threshold is not None:
            available += 1
            triggered = bool(value >= threshold)
            deteriorating += int(triggered)
        signals[signal["name"]] = {
            "delta": value,
            "threshold": threshold,
            "deteriorating": triggered,
        }
    return {"available": available, "deteriorating": deteriorating, "signals": signals}


def _search_utility(metrics: dict[str, Any]) -> float:
    cagr = float(metrics.get("cagr") or 0.0)
    sharpe = float(metrics.get("sharpe") or 0.0)
    drawdown = float(metrics.get("maximum_drawdown") or 0.0)
    worst_fold = float(metrics.get("worst_fold_return") or 0.0)
    turnover = float(metrics.get("turnover_ratio") or 0.0)
    return float(cagr + 0.20 * sharpe + 0.40 * drawdown + 0.30 * worst_fold - 0.05 * turnover)


def metrics_from_interval_returns(
    interval_rows: list[dict[str, Any]],
    *,
    initial_capital: float,
    winner_fold_returns: dict[int, float] | None = None,
) -> dict[str, Any]:
    if not interval_rows:
        raise ValueError("Temporal policy replay produced no usable decision intervals.")
    values = np.asarray([float(item["net_return"]) for item in interval_rows], dtype=float)
    fold_ids = [int(item.get("fold_id") or 0) for item in interval_rows]
    equity = float(initial_capital) * np.cumprod(1.0 + values)
    peaks = np.maximum.accumulate(equity)
    drawdowns = equity / peaks - 1.0
    years = max(len(values) / 252.0, 1.0 / 252.0)
    ending_capital = float(equity[-1])
    cagr = (ending_capital / float(initial_capital)) ** (1.0 / years) - 1.0 if ending_capital > 0 else -1.0
    volatility = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    sharpe = float(np.mean(values) / volatility * math.sqrt(252.0)) if volatility > 1e-12 else 0.0

    folds: list[dict[str, Any]] = []
    for fold_id in sorted(set(fold_ids)):
        indices = [index for index, value in enumerate(fold_ids) if value == fold_id]
        fold_values = values[indices]
        fold_equity = float(initial_capital) * np.cumprod(1.0 + fold_values)
        fold_peaks = np.maximum.accumulate(fold_equity)
        fold_drawdown = fold_equity / fold_peaks - 1.0
        folds.append({
            "fold_id": int(fold_id),
            "strategy_return": float(fold_equity[-1] / float(initial_capital) - 1.0),
            "maximum_drawdown": float(np.min(fold_drawdown)),
            "benchmark_return": float((winner_fold_returns or {}).get(fold_id, 0.0)),
        })
    fold_returns = [float(item["strategy_return"]) for item in folds]
    switch_count = sum(int(item.get("cost_sides") or 0) > 0 for item in interval_rows)
    exposure_days = sum(str(item.get("selected_asset") or "CASH") != "CASH" for item in interval_rows)
    metrics = {
        "initial_capital": float(initial_capital),
        "ending_capital": ending_capital,
        "strategy_return": ending_capital / float(initial_capital) - 1.0,
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "maximum_drawdown": float(np.min(drawdowns)),
        "turnover_ratio": float(switch_count / max(1, len(values))),
        "capital_rotations": int(switch_count),
        "average_holding_days": float(len(values) / max(1, switch_count)),
        "market_exposure": float(exposure_days / max(1, len(values))),
        "cash_days": int(len(values) - exposure_days),
        "folds": folds,
        "worst_fold_return": min(fold_returns) if fold_returns else None,
        "eligible": bool(fold_returns) and all(value > 0 for value in fold_returns),
    }
    metrics["search_utility"] = _search_utility(metrics)
    return metrics


def replay_search_policy(
    observations: dict[str, dict[str, Any]],
    winner_rows: list[dict[str, Any]],
    *,
    initial_capital: float,
    one_side_cost: float,
    settings: dict[str, Any],
    winner_fold_returns: dict[int, float],
    trajectory_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = normalize_settings(settings)
    context = trajectory_context or build_trajectory_thresholds(observations, normalized)
    winner_by_decision = {
        key: row
        for row in winner_rows
        if isinstance(row, dict) and (key := _timestamp_key(row.get("decision_date")))
    }
    weak_threshold = float(normalized["timing_base_weak_threshold"])
    challenger_minimum = float(normalized["timing_challenger_minimum"])
    minimum_advantage = float(normalized["timing_minimum_advantage"])
    maximum_advantage = float(normalized.get("timing_maximum_advantage", 1.0))
    lookback = int(normalized["trajectory_lookback_sessions"])
    minimum_signals = int(normalized["trajectory_min_signals"])
    late_exit_advantage = float(normalized["late_exit_min_challenger_advantage"])
    cash_guard_enabled = bool(int(normalized["late_exit_cash_guard"]))

    histories: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    current_fold: int | None = None
    current_symbol: str | None = None
    intervals: list[dict[str, Any]] = []
    timing_override_count = 0
    late_exit_risk_count = 0
    late_exit_cash_guard_count = 0

    ordered_keys = sorted(observations, key=lambda item: pd.Timestamp(item))
    for key in ordered_keys:
        payload = observations[key]
        fold_id = int(payload.get("fold_id") or 0)
        rows_by_symbol = payload.get("rows_by_symbol") or {}
        winner_row = winner_by_decision.get(key)
        if fold_id <= 0 or winner_row is None or not rows_by_symbol:
            continue
        if not any(_finite(row.get("open_to_open_return")) is not None for row in rows_by_symbol.values()):
            continue

        if current_fold is not None and fold_id != current_fold:
            if current_symbol is not None and intervals:
                last = intervals[-1]
                last["net_return"] = max(1e-9, (1.0 + float(last["net_return"])) * max(1e-9, 1.0 - one_side_cost)) - 1.0
                last["fold_close_cost_applied"] = True
            current_symbol = None
        current_fold = fold_id

        top1_value = winner_row.get("top_1_asset") or winner_row.get("raw_best_asset") or winner_row.get("best_asset")
        top2_value = winner_row.get("top_2_asset") or winner_row.get("second_asset")
        base_value = winner_row.get("selected_asset") or winner_row.get("final_action_asset") or top1_value
        base_symbol = str(base_value) if base_value not in {None, "", "CASH"} else None
        top1_symbol = str(top1_value) if top1_value not in {None, "", "CASH"} else None
        challenger_symbol = str(top2_value) if top2_value not in {None, "", "CASH"} else None

        base_row = rows_by_symbol.get(base_symbol) if base_symbol else None
        challenger_row = rows_by_symbol.get(challenger_symbol) if challenger_symbol else None
        base_short = _finite((base_row or {}).get("short_profit_consensus"))
        challenger_short = _finite((challenger_row or {}).get("short_profit_consensus"))
        timing_override = False
        if base_short is not None and challenger_short is not None:
            timing_override = bool(
                base_symbol == top1_symbol
                and challenger_symbol != base_symbol
                and base_short < weak_threshold
                and challenger_short >= challenger_minimum
                and (challenger_short - base_short) >= minimum_advantage
                and (challenger_short - base_short) <= maximum_advantage
            )
        proposed_symbol = challenger_symbol if timing_override else base_symbol
        target_symbol = proposed_symbol

        trajectory_state: dict[str, Any] | None = None
        late_exit_risk = False
        cash_guard = False
        challenger_advantage: float | None = None
        if current_symbol is not None and proposed_symbol is not None and current_symbol != proposed_symbol:
            current_row = rows_by_symbol.get(current_symbol)
            history = histories.get((fold_id, current_symbol), [])
            previous_row = history[-lookback] if len(history) >= lookback else None
            trajectory_state = _trajectory_state(current_row, previous_row, context)
            late_exit_risk = bool(
                int(trajectory_state.get("available") or 0) >= minimum_signals
                and int(trajectory_state.get("deteriorating") or 0) >= minimum_signals
            )
            if late_exit_risk:
                late_exit_risk_count += 1
                current_short = _finite((current_row or {}).get("short_profit_consensus"))
                target_short = _finite((rows_by_symbol.get(proposed_symbol) or {}).get("short_profit_consensus"))
                if current_short is not None and target_short is not None:
                    challenger_advantage = float(target_short - current_short)
                if cash_guard_enabled and challenger_advantage is not None and challenger_advantage < late_exit_advantage:
                    target_symbol = None
                    timing_override = False
                    cash_guard = True
                    late_exit_cash_guard_count += 1

        def interval_return(symbol: str | None) -> float | None:
            if symbol is None:
                return 0.0
            return _finite((rows_by_symbol.get(symbol) or {}).get("open_to_open_return"))

        target_return = interval_return(target_symbol)
        if target_symbol is not None and target_return is None:
            base_return = interval_return(base_symbol)
            if base_symbol is not None and base_return is not None:
                target_symbol = base_symbol
                target_return = base_return
                timing_override = False
                cash_guard = False
            else:
                target_symbol = None
                target_return = 0.0
                timing_override = False
                cash_guard = False

        previous_symbol = current_symbol
        if current_symbol is None and target_symbol is not None:
            cost_sides = 1
        elif current_symbol is not None and target_symbol is None:
            cost_sides = 1
        elif current_symbol is not None and target_symbol is not None and current_symbol != target_symbol:
            cost_sides = 2
        else:
            cost_sides = 0

        if timing_override:
            timing_override_count += 1
        gross_return = float(target_return or 0.0)
        net_return = max(1e-9, 1.0 - float(cost_sides) * one_side_cost) * max(1e-9, 1.0 + gross_return) - 1.0
        sample_row = rows_by_symbol.get(target_symbol) if target_symbol else (base_row or challenger_row or next(iter(rows_by_symbol.values()), {}))
        intervals.append({
            "decision_date": key,
            "execution_date": _timestamp_key((sample_row or {}).get("execution_date")) or key,
            "next_execution_date": _timestamp_key((sample_row or {}).get("next_execution_date")) or key,
            "fold_id": fold_id,
            "from_asset": previous_symbol or "CASH",
            "selected_asset": target_symbol or "CASH",
            "proposed_asset": proposed_symbol or "CASH",
            "base_asset": base_symbol or "CASH",
            "top_1_asset": top1_symbol or "CASH",
            "top_2_asset": challenger_symbol or "CASH",
            "timing_override": bool(timing_override),
            "late_exit_risk": bool(late_exit_risk),
            "late_exit_cash_guard": bool(cash_guard),
            "challenger_advantage": challenger_advantage,
            "trajectory_state": trajectory_state,
            "gross_return": gross_return,
            "net_return": float(net_return),
            "cost_sides": int(cost_sides),
            "fold_close_cost_applied": False,
        })
        current_symbol = target_symbol

        for symbol, row in rows_by_symbol.items():
            history = histories[(fold_id, str(symbol))]
            history.append(row)
            if len(history) > 8:
                del history[:-8]

    if current_symbol is not None and intervals:
        last = intervals[-1]
        last["net_return"] = max(1e-9, (1.0 + float(last["net_return"])) * max(1e-9, 1.0 - one_side_cost)) - 1.0
        last["fold_close_cost_applied"] = True
    metrics = metrics_from_interval_returns(
        intervals,
        initial_capital=initial_capital,
        winner_fold_returns=winner_fold_returns,
    )
    metrics.update({
        "timing_override_count": int(timing_override_count),
        "late_exit_risk_count": int(late_exit_risk_count),
        "late_exit_cash_guard_count": int(late_exit_cash_guard_count),
    })
    return {
        "settings": normalized,
        "trajectory_context": context,
        "metrics": metrics,
        "intervals": intervals,
    }
