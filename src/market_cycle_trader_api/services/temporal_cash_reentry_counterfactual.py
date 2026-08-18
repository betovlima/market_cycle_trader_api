from __future__ import annotations

from collections import defaultdict
import math
from typing import Any

import numpy as np
import pandas as pd

from .temporal_policy_replay import _finite, _timestamp_key


ABSOLUTE_OPPORTUNITY_WEIGHTS: dict[str, float] = {
    "short_profit_consensus": 0.45,
    "all_horizon_risk_safety": 0.20,
    "short_horizon_agreement": 0.12,
    "long_profit_confirmation": 0.10,
    "horizon_agreement": 0.08,
    "long_trend_support": 0.05,
}

DEFAULT_TIMING_SETTINGS: dict[str, float] = {
    "timing_base_weak_threshold": 0.50,
    "timing_challenger_minimum": 0.60,
    "timing_minimum_advantage": 0.25,
}


def absolute_opportunity_score(row: dict[str, Any] | None) -> float | None:
    """Absolute, causal opportunity quality independent of cross-sectional rank.

    The score intentionally excludes relative fields such as rank percentile,
    top-gap and separation strength. Missing components are re-normalized over
    the available causal prediction fields rather than filled from realized data.
    """
    if not isinstance(row, dict):
        return None
    weighted = 0.0
    weight_sum = 0.0
    for field, weight in ABSOLUTE_OPPORTUNITY_WEIGHTS.items():
        value = _finite(row.get(field))
        if value is None:
            continue
        weighted += float(weight) * min(1.0, max(0.0, float(value)))
        weight_sum += float(weight)
    if weight_sum <= 0.0:
        return None
    return min(1.0, max(0.0, weighted / weight_sum))


def _winner_lookup(winner_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        key: row
        for row in winner_rows
        if isinstance(row, dict) and (key := _timestamp_key(row.get("decision_date")))
    }


def _baseline_target(
    rows_by_symbol: dict[str, dict[str, Any]],
    winner_row: dict[str, Any],
    timing_settings: dict[str, Any],
) -> tuple[str | None, dict[str, Any]]:
    weak_threshold = float(timing_settings["timing_base_weak_threshold"])
    challenger_minimum = float(timing_settings["timing_challenger_minimum"])
    minimum_advantage = float(timing_settings["timing_minimum_advantage"])

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
    override = False
    if base_short is not None and challenger_short is not None:
        override = bool(
            base_symbol == top1_symbol
            and challenger_symbol != base_symbol
            and base_short < weak_threshold
            and challenger_short >= challenger_minimum
            and (challenger_short - base_short) >= minimum_advantage
        )
    target_symbol = challenger_symbol if override else base_symbol
    return target_symbol, {
        "base_asset": base_symbol or "CASH",
        "top_1_asset": top1_symbol or "CASH",
        "top_2_asset": challenger_symbol or "CASH",
        "timing_override": bool(override),
        "base_short_profit": base_short,
        "challenger_short_profit": challenger_short,
    }


def _fold_metrics(
    values: np.ndarray,
    fold_ids: list[int],
    initial_capital: float,
    winner_fold_returns: dict[int, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold_id in sorted(set(fold_ids)):
        indices = [index for index, value in enumerate(fold_ids) if int(value) == int(fold_id)]
        if not indices:
            continue
        fold_values = values[indices]
        fold_equity = float(initial_capital) * np.cumprod(1.0 + fold_values)
        peaks = np.maximum.accumulate(fold_equity)
        drawdown = fold_equity / peaks - 1.0
        rows.append({
            "fold_id": int(fold_id),
            "strategy_return": float(fold_equity[-1] / float(initial_capital) - 1.0),
            "maximum_drawdown": float(np.min(drawdown)),
            "benchmark_return": float(winner_fold_returns.get(int(fold_id), 0.0)),
        })
    return rows


def replay_absolute_opportunity_reentry_gate(
    observations: dict[str, dict[str, Any]],
    winner_rows: list[dict[str, Any]],
    *,
    initial_capital: float,
    one_side_cost: float,
    timing_settings: dict[str, Any] | None = None,
    gate_settings: dict[str, Any] | None = None,
    winner_fold_returns: dict[int, float] | None = None,
    gate_enabled: bool = True,
) -> dict[str, Any]:
    """Counterfactual replay of a causal Absolute Opportunity + CASH Re-entry gate.

    The existing Winner-Anchored Temporal target is calculated first. The gate
    may only replace that market target with CASH. It never sees the realized
    interval return before making the decision.
    """
    timing = dict(DEFAULT_TIMING_SETTINGS)
    timing.update(dict(timing_settings or {}))
    gate = dict(gate_settings or {})
    entry_threshold = float(gate.get("absolute_entry_threshold", 0.50))
    exit_discount = max(0.0, float(gate.get("absolute_exit_discount", 0.10)))
    exit_threshold = max(0.0, min(entry_threshold, entry_threshold - exit_discount))
    reentry_premium = max(0.0, float(gate.get("cash_reentry_premium", 0.05)))
    minimum_risk = max(0.0, float(gate.get("minimum_risk_safety", 0.20)))
    minimum_agreement = max(0.0, float(gate.get("minimum_horizon_agreement", 0.55)))
    confirmation_sessions = max(1, int(round(float(gate.get("reentry_confirmation_sessions", 1)))))

    winner_by_decision = _winner_lookup(winner_rows)
    ordered_keys = sorted(observations, key=lambda item: pd.Timestamp(item))

    net_returns: list[float] = []
    fold_ids: list[int] = []
    dates: list[str] = []
    intervals: list[dict[str, Any]] = []
    current_fold: int | None = None
    current_symbol: str | None = None
    cash_qualification_streak = 0
    switch_count = 0
    exposure_days = 0
    intervention_counts: defaultdict[str, int] = defaultdict(int)
    loss_avoided_usd = 0.0
    profit_missed_usd = 0.0
    prior_equity = float(initial_capital)

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
            if current_symbol is not None and net_returns:
                net_returns[-1] = max(1e-9, (1.0 + net_returns[-1]) * max(1e-9, 1.0 - float(one_side_cost))) - 1.0
                prior_equity *= max(1e-9, 1.0 - float(one_side_cost))
            current_symbol = None
            cash_qualification_streak = 0
        current_fold = fold_id

        baseline_symbol, baseline_meta = _baseline_target(rows_by_symbol, winner_row, timing)
        target_symbol = baseline_symbol
        target_row = rows_by_symbol.get(target_symbol) if target_symbol else None
        absolute_score = absolute_opportunity_score(target_row)
        risk_safety = _finite((target_row or {}).get("all_horizon_risk_safety"))
        horizon_agreement = _finite((target_row or {}).get("horizon_agreement"))
        gate_reason = "baseline_temporal"
        qualified = True

        if gate_enabled and target_symbol is not None:
            score_value = -math.inf if absolute_score is None else float(absolute_score)
            risk_value = -math.inf if risk_safety is None else float(risk_safety)
            agreement_value = -math.inf if horizon_agreement is None else float(horizon_agreement)
            if current_symbol is None:
                required_score = min(1.0, entry_threshold + reentry_premium)
                qualified = bool(
                    score_value >= required_score
                    and risk_value >= minimum_risk
                    and agreement_value >= minimum_agreement
                )
                if qualified:
                    cash_qualification_streak += 1
                else:
                    cash_qualification_streak = 0
                if not qualified:
                    target_symbol = None
                    gate_reason = "cash_reentry_absolute_gate_reject"
                    intervention_counts[gate_reason] += 1
                elif cash_qualification_streak < confirmation_sessions:
                    target_symbol = None
                    gate_reason = "cash_reentry_confirmation_wait"
                    intervention_counts[gate_reason] += 1
                else:
                    gate_reason = "cash_reentry_gate_accept"
                    intervention_counts[gate_reason] += 1
                    cash_qualification_streak = 0
            elif target_symbol != current_symbol:
                cash_qualification_streak = 0
                qualified = bool(
                    score_value >= entry_threshold
                    and risk_value >= minimum_risk
                    and agreement_value >= minimum_agreement
                )
                if not qualified:
                    target_symbol = None
                    gate_reason = "rotation_absolute_gate_to_cash"
                    intervention_counts[gate_reason] += 1
                else:
                    gate_reason = "rotation_absolute_gate_accept"
            else:
                cash_qualification_streak = 0
                qualified = bool(score_value >= exit_threshold)
                if not qualified:
                    target_symbol = None
                    gate_reason = "absolute_opportunity_exit_to_cash"
                    intervention_counts[gate_reason] += 1
                else:
                    gate_reason = "absolute_opportunity_hold"
        elif target_symbol is None:
            cash_qualification_streak = 0
            gate_reason = "baseline_cash"

        def interval_return(symbol: str | None) -> float | None:
            if symbol is None:
                return 0.0
            return _finite((rows_by_symbol.get(symbol) or {}).get("open_to_open_return"))

        baseline_return = interval_return(baseline_symbol)
        target_return = interval_return(target_symbol)
        if target_symbol is not None and target_return is None:
            target_symbol = None
            target_return = 0.0
            gate_reason = "market_path_unavailable_to_cash"
            intervention_counts[gate_reason] += 1

        previous_symbol = current_symbol
        if previous_symbol is None and target_symbol is not None:
            cost_sides = 1
            switch_count += 1
        elif previous_symbol is not None and target_symbol is None:
            cost_sides = 1
            switch_count += 1
        elif previous_symbol is not None and target_symbol is not None and previous_symbol != target_symbol:
            cost_sides = 2
            switch_count += 1
        else:
            cost_sides = 0

        if target_symbol is not None:
            exposure_days += 1
        gross_return = float(target_return or 0.0)
        factor = max(1e-9, 1.0 - float(cost_sides) * float(one_side_cost)) * max(1e-9, 1.0 + gross_return)
        net_return = factor - 1.0

        avoided = 0.0
        missed = 0.0
        if baseline_symbol is not None and target_symbol is None and baseline_return is not None:
            if float(baseline_return) < 0.0:
                avoided = float(prior_equity) * -float(baseline_return)
                loss_avoided_usd += avoided
            elif float(baseline_return) > 0.0:
                missed = float(prior_equity) * float(baseline_return)
                profit_missed_usd += missed

        net_returns.append(net_return)
        fold_ids.append(fold_id)
        dates.append(key)
        execution_date = _timestamp_key((target_row or next(iter(rows_by_symbol.values()), {})).get("execution_date")) or key
        next_execution_date = _timestamp_key((target_row or next(iter(rows_by_symbol.values()), {})).get("next_execution_date")) or key
        intervals.append({
            "decision_date": key,
            "execution_date": execution_date,
            "next_execution_date": next_execution_date,
            "fold_id": fold_id,
            "from_asset": previous_symbol or "CASH",
            "baseline_target_asset": baseline_symbol or "CASH",
            "selected_asset": target_symbol or "CASH",
            "gate_reason": gate_reason,
            "absolute_opportunity_score": absolute_score,
            "all_horizon_risk_safety": risk_safety,
            "horizon_agreement": horizon_agreement,
            "absolute_entry_threshold": entry_threshold,
            "absolute_exit_threshold": exit_threshold,
            "cash_reentry_threshold": min(1.0, entry_threshold + reentry_premium),
            "minimum_risk_safety": minimum_risk,
            "minimum_horizon_agreement": minimum_agreement,
            "reentry_confirmation_sessions": confirmation_sessions,
            "baseline_gross_return": baseline_return,
            "gross_return": gross_return,
            "net_return": net_return,
            "cost_sides": cost_sides,
            "loss_avoided_by_cash_usd": avoided,
            "profit_missed_by_cash_usd": missed,
            **baseline_meta,
        })
        prior_equity *= factor
        current_symbol = target_symbol

    if current_symbol is not None and net_returns:
        net_returns[-1] = max(1e-9, (1.0 + net_returns[-1]) * max(1e-9, 1.0 - float(one_side_cost))) - 1.0

    if not net_returns:
        raise ValueError("Counterfactual Temporal replay has no usable decision intervals.")

    values = np.asarray(net_returns, dtype=float)
    equity = float(initial_capital) * np.cumprod(1.0 + values)
    peaks = np.maximum.accumulate(equity)
    drawdowns = equity / peaks - 1.0
    years = max(len(values) / 252.0, 1.0 / 252.0)
    ending_capital = float(equity[-1])
    cagr = (ending_capital / float(initial_capital)) ** (1.0 / years) - 1.0 if ending_capital > 0 else -1.0
    volatility = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    sharpe = float(np.mean(values) / volatility * math.sqrt(252.0)) if volatility > 1e-12 else 0.0
    folds = _fold_metrics(values, fold_ids, float(initial_capital), dict(winner_fold_returns or {}))
    fold_returns = [float(row["strategy_return"]) for row in folds]

    for index, interval in enumerate(intervals):
        interval["strategy_equity"] = float(equity[index])
        interval["strategy_drawdown"] = float(drawdowns[index])

    monthly: dict[str, dict[str, Any]] = {}
    for interval in intervals:
        timestamp = pd.Timestamp(interval["next_execution_date"] or interval["decision_date"])
        month = timestamp.strftime("%Y-%m")
        row = monthly.setdefault(month, {
            "month": month,
            "loss_avoided_by_cash_usd": 0.0,
            "profit_missed_by_cash_usd": 0.0,
            "cash_intervention_sessions": 0,
            "cash_reentry_rejects": 0,
            "cash_reentry_waits": 0,
            "rotation_to_cash_blocks": 0,
            "absolute_exits_to_cash": 0,
        })
        avoided = float(interval.get("loss_avoided_by_cash_usd") or 0.0)
        missed = float(interval.get("profit_missed_by_cash_usd") or 0.0)
        row["loss_avoided_by_cash_usd"] += avoided
        row["profit_missed_by_cash_usd"] += missed
        if interval.get("baseline_target_asset") != "CASH" and interval.get("selected_asset") == "CASH":
            row["cash_intervention_sessions"] += 1
        reason = str(interval.get("gate_reason") or "")
        if reason == "cash_reentry_absolute_gate_reject":
            row["cash_reentry_rejects"] += 1
        elif reason == "cash_reentry_confirmation_wait":
            row["cash_reentry_waits"] += 1
        elif reason == "rotation_absolute_gate_to_cash":
            row["rotation_to_cash_blocks"] += 1
        elif reason == "absolute_opportunity_exit_to_cash":
            row["absolute_exits_to_cash"] += 1
    monthly_rows = []
    for month in sorted(monthly):
        row = monthly[month]
        row["net_cash_edge_usd"] = float(row["loss_avoided_by_cash_usd"] - row["profit_missed_by_cash_usd"])
        monthly_rows.append(row)

    return {
        "settings": {
            "gate_enabled": bool(gate_enabled),
            "absolute_entry_threshold": entry_threshold,
            "absolute_exit_discount": exit_discount,
            "absolute_exit_threshold": exit_threshold,
            "cash_reentry_premium": reentry_premium,
            "minimum_risk_safety": minimum_risk,
            "minimum_horizon_agreement": minimum_agreement,
            "reentry_confirmation_sessions": confirmation_sessions,
            "absolute_opportunity_weights": dict(ABSOLUTE_OPPORTUNITY_WEIGHTS),
            **{name: float(timing[name]) for name in DEFAULT_TIMING_SETTINGS},
        },
        "metrics": {
            "initial_capital": float(initial_capital),
            "ending_capital": ending_capital,
            "strategy_return": ending_capital / float(initial_capital) - 1.0,
            "cagr": float(cagr),
            "sharpe": float(sharpe),
            "maximum_drawdown": float(np.min(drawdowns)),
            "capital_rotations": int(switch_count),
            "market_exposure": float(exposure_days / max(1, len(values))),
            "cash_days": int(len(values) - exposure_days),
            "worst_fold_return": min(fold_returns) if fold_returns else None,
            "eligible": bool(fold_returns) and all(value > 0.0 for value in fold_returns),
            "loss_avoided_by_cash_usd": float(loss_avoided_usd),
            "profit_missed_by_cash_usd": float(profit_missed_usd),
            "net_cash_edge_usd": float(loss_avoided_usd - profit_missed_usd),
            "cash_intervention_sessions": int(sum(1 for row in intervals if row["baseline_target_asset"] != "CASH" and row["selected_asset"] == "CASH")),
            "intervention_counts": dict(intervention_counts),
            "folds": folds,
        },
        "equity": [
            {
                "timestamp": intervals[index]["next_execution_date"],
                "strategy_equity": float(equity[index]),
                "drawdown": float(drawdowns[index]),
                "selected_asset": intervals[index]["selected_asset"],
                "gate_reason": intervals[index]["gate_reason"],
            }
            for index in range(len(intervals))
        ],
        "intervals": intervals,
        "monthly_attribution": monthly_rows,
    }


def compile_absolute_opportunity_context(
    observations: dict[str, dict[str, Any]],
    winner_rows: list[dict[str, Any]],
    *,
    timing_settings: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Precompile the immutable Winner-Anchored target path for large searches."""
    timing = dict(DEFAULT_TIMING_SETTINGS)
    timing.update(dict(timing_settings or {}))
    winner_by_decision = _winner_lookup(winner_rows)
    compiled: list[dict[str, Any]] = []
    for key in sorted(observations, key=lambda item: pd.Timestamp(item)):
        payload = observations[key]
        fold_id = int(payload.get("fold_id") or 0)
        rows_by_symbol = payload.get("rows_by_symbol") or {}
        winner_row = winner_by_decision.get(key)
        if fold_id <= 0 or winner_row is None or not rows_by_symbol:
            continue
        if not any(_finite(row.get("open_to_open_return")) is not None for row in rows_by_symbol.values()):
            continue
        baseline_symbol, baseline_meta = _baseline_target(rows_by_symbol, winner_row, timing)
        target_row = rows_by_symbol.get(baseline_symbol) if baseline_symbol else None
        sample_row = target_row or next(iter(rows_by_symbol.values()), {})
        compiled.append({
            "decision_date": key,
            "execution_date": _timestamp_key((sample_row or {}).get("execution_date")) or key,
            "next_execution_date": _timestamp_key((sample_row or {}).get("next_execution_date")) or key,
            "fold_id": fold_id,
            "baseline_target_asset": baseline_symbol or "CASH",
            "baseline_symbol": baseline_symbol,
            "baseline_gross_return": 0.0 if baseline_symbol is None else _finite((target_row or {}).get("open_to_open_return")),
            "absolute_opportunity_score": absolute_opportunity_score(target_row),
            "all_horizon_risk_safety": _finite((target_row or {}).get("all_horizon_risk_safety")),
            "horizon_agreement": _finite((target_row or {}).get("horizon_agreement")),
            **baseline_meta,
        })
    if not compiled:
        raise ValueError("Counterfactual Temporal replay has no usable decision intervals.")
    return compiled


def replay_compiled_absolute_opportunity_reentry_gate(
    compiled: list[dict[str, Any]],
    *,
    initial_capital: float,
    one_side_cost: float,
    gate_settings: dict[str, Any] | None = None,
    winner_fold_returns: dict[int, float] | None = None,
    gate_enabled: bool = True,
) -> dict[str, Any]:
    """Fast replay for large counterfactual searches over one frozen target path."""
    gate = dict(gate_settings or {})
    entry_threshold = float(gate.get("absolute_entry_threshold", 0.50))
    exit_discount = max(0.0, float(gate.get("absolute_exit_discount", 0.10)))
    exit_threshold = max(0.0, min(entry_threshold, entry_threshold - exit_discount))
    reentry_premium = max(0.0, float(gate.get("cash_reentry_premium", 0.05)))
    minimum_risk = max(0.0, float(gate.get("minimum_risk_safety", 0.20)))
    minimum_agreement = max(0.0, float(gate.get("minimum_horizon_agreement", 0.55)))
    confirmation_sessions = max(1, int(round(float(gate.get("reentry_confirmation_sessions", 1)))))

    net_returns: list[float] = []
    fold_ids: list[int] = []
    intervals: list[dict[str, Any]] = []
    current_fold: int | None = None
    current_symbol: str | None = None
    cash_qualification_streak = 0
    switch_count = 0
    exposure_days = 0
    intervention_counts: defaultdict[str, int] = defaultdict(int)
    loss_avoided_usd = 0.0
    profit_missed_usd = 0.0
    prior_equity = float(initial_capital)

    for source in compiled:
        fold_id = int(source["fold_id"])
        if current_fold is not None and fold_id != current_fold:
            if current_symbol is not None and net_returns:
                net_returns[-1] = max(1e-9, (1.0 + net_returns[-1]) * max(1e-9, 1.0 - float(one_side_cost))) - 1.0
                prior_equity *= max(1e-9, 1.0 - float(one_side_cost))
            current_symbol = None
            cash_qualification_streak = 0
        current_fold = fold_id

        baseline_symbol = source.get("baseline_symbol")
        target_symbol = baseline_symbol
        baseline_return = _finite(source.get("baseline_gross_return"))
        absolute_score = _finite(source.get("absolute_opportunity_score"))
        risk_safety = _finite(source.get("all_horizon_risk_safety"))
        horizon_agreement = _finite(source.get("horizon_agreement"))
        gate_reason = "baseline_temporal"

        if gate_enabled and target_symbol is not None:
            score_value = -math.inf if absolute_score is None else absolute_score
            risk_value = -math.inf if risk_safety is None else risk_safety
            agreement_value = -math.inf if horizon_agreement is None else horizon_agreement
            if current_symbol is None:
                required_score = min(1.0, entry_threshold + reentry_premium)
                qualified = score_value >= required_score and risk_value >= minimum_risk and agreement_value >= minimum_agreement
                cash_qualification_streak = cash_qualification_streak + 1 if qualified else 0
                if not qualified:
                    target_symbol = None
                    gate_reason = "cash_reentry_absolute_gate_reject"
                    intervention_counts[gate_reason] += 1
                elif cash_qualification_streak < confirmation_sessions:
                    target_symbol = None
                    gate_reason = "cash_reentry_confirmation_wait"
                    intervention_counts[gate_reason] += 1
                else:
                    gate_reason = "cash_reentry_gate_accept"
                    intervention_counts[gate_reason] += 1
                    cash_qualification_streak = 0
            elif target_symbol != current_symbol:
                cash_qualification_streak = 0
                qualified = score_value >= entry_threshold and risk_value >= minimum_risk and agreement_value >= minimum_agreement
                if not qualified:
                    target_symbol = None
                    gate_reason = "rotation_absolute_gate_to_cash"
                    intervention_counts[gate_reason] += 1
                else:
                    gate_reason = "rotation_absolute_gate_accept"
            else:
                cash_qualification_streak = 0
                if score_value < exit_threshold:
                    target_symbol = None
                    gate_reason = "absolute_opportunity_exit_to_cash"
                    intervention_counts[gate_reason] += 1
                else:
                    gate_reason = "absolute_opportunity_hold"
        elif target_symbol is None:
            cash_qualification_streak = 0
            gate_reason = "baseline_cash"

        target_return = 0.0 if target_symbol is None else baseline_return
        if target_symbol is not None and target_return is None:
            target_symbol = None
            target_return = 0.0
            gate_reason = "market_path_unavailable_to_cash"
            intervention_counts[gate_reason] += 1

        previous_symbol = current_symbol
        if previous_symbol is None and target_symbol is not None:
            cost_sides = 1
            switch_count += 1
        elif previous_symbol is not None and target_symbol is None:
            cost_sides = 1
            switch_count += 1
        elif previous_symbol is not None and target_symbol is not None and previous_symbol != target_symbol:
            cost_sides = 2
            switch_count += 1
        else:
            cost_sides = 0
        if target_symbol is not None:
            exposure_days += 1

        gross_return = float(target_return or 0.0)
        factor = max(1e-9, 1.0 - float(cost_sides) * float(one_side_cost)) * max(1e-9, 1.0 + gross_return)
        net_return = factor - 1.0
        avoided = 0.0
        missed = 0.0
        if baseline_symbol is not None and target_symbol is None and baseline_return is not None:
            if baseline_return < 0.0:
                avoided = prior_equity * -baseline_return
                loss_avoided_usd += avoided
            elif baseline_return > 0.0:
                missed = prior_equity * baseline_return
                profit_missed_usd += missed

        net_returns.append(net_return)
        fold_ids.append(fold_id)
        intervals.append({
            **source,
            "from_asset": previous_symbol or "CASH",
            "selected_asset": target_symbol or "CASH",
            "gate_reason": gate_reason,
            "absolute_entry_threshold": entry_threshold,
            "absolute_exit_threshold": exit_threshold,
            "cash_reentry_threshold": min(1.0, entry_threshold + reentry_premium),
            "minimum_risk_safety": minimum_risk,
            "minimum_horizon_agreement": minimum_agreement,
            "reentry_confirmation_sessions": confirmation_sessions,
            "gross_return": gross_return,
            "net_return": net_return,
            "cost_sides": cost_sides,
            "loss_avoided_by_cash_usd": avoided,
            "profit_missed_by_cash_usd": missed,
        })
        prior_equity *= factor
        current_symbol = target_symbol

    if current_symbol is not None and net_returns:
        net_returns[-1] = max(1e-9, (1.0 + net_returns[-1]) * max(1e-9, 1.0 - float(one_side_cost))) - 1.0

    values = np.asarray(net_returns, dtype=float)
    equity = float(initial_capital) * np.cumprod(1.0 + values)
    peaks = np.maximum.accumulate(equity)
    drawdowns = equity / peaks - 1.0
    years = max(len(values) / 252.0, 1.0 / 252.0)
    ending_capital = float(equity[-1])
    cagr = (ending_capital / float(initial_capital)) ** (1.0 / years) - 1.0 if ending_capital > 0 else -1.0
    volatility = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    sharpe = float(np.mean(values) / volatility * math.sqrt(252.0)) if volatility > 1e-12 else 0.0
    folds = _fold_metrics(values, fold_ids, float(initial_capital), dict(winner_fold_returns or {}))
    fold_returns = [float(row["strategy_return"]) for row in folds]
    for index, interval in enumerate(intervals):
        interval["strategy_equity"] = float(equity[index])
        interval["strategy_drawdown"] = float(drawdowns[index])

    monthly: dict[str, dict[str, Any]] = {}
    for interval in intervals:
        timestamp = pd.Timestamp(interval["next_execution_date"] or interval["decision_date"])
        month = timestamp.strftime("%Y-%m")
        row = monthly.setdefault(month, {
            "month": month, "loss_avoided_by_cash_usd": 0.0, "profit_missed_by_cash_usd": 0.0,
            "cash_intervention_sessions": 0, "cash_reentry_rejects": 0, "cash_reentry_waits": 0,
            "rotation_to_cash_blocks": 0, "absolute_exits_to_cash": 0,
        })
        row["loss_avoided_by_cash_usd"] += float(interval.get("loss_avoided_by_cash_usd") or 0.0)
        row["profit_missed_by_cash_usd"] += float(interval.get("profit_missed_by_cash_usd") or 0.0)
        if interval.get("baseline_target_asset") != "CASH" and interval.get("selected_asset") == "CASH":
            row["cash_intervention_sessions"] += 1
        reason = str(interval.get("gate_reason") or "")
        if reason == "cash_reentry_absolute_gate_reject": row["cash_reentry_rejects"] += 1
        elif reason == "cash_reentry_confirmation_wait": row["cash_reentry_waits"] += 1
        elif reason == "rotation_absolute_gate_to_cash": row["rotation_to_cash_blocks"] += 1
        elif reason == "absolute_opportunity_exit_to_cash": row["absolute_exits_to_cash"] += 1
    monthly_rows = []
    for month in sorted(monthly):
        row = monthly[month]
        row["net_cash_edge_usd"] = float(row["loss_avoided_by_cash_usd"] - row["profit_missed_by_cash_usd"])
        monthly_rows.append(row)

    return {
        "settings": {
            "gate_enabled": bool(gate_enabled), "absolute_entry_threshold": entry_threshold,
            "absolute_exit_discount": exit_discount, "absolute_exit_threshold": exit_threshold,
            "cash_reentry_premium": reentry_premium, "minimum_risk_safety": minimum_risk,
            "minimum_horizon_agreement": minimum_agreement, "reentry_confirmation_sessions": confirmation_sessions,
            "absolute_opportunity_weights": dict(ABSOLUTE_OPPORTUNITY_WEIGHTS),
        },
        "metrics": {
            "initial_capital": float(initial_capital), "ending_capital": ending_capital,
            "strategy_return": ending_capital / float(initial_capital) - 1.0, "cagr": float(cagr),
            "sharpe": float(sharpe), "maximum_drawdown": float(np.min(drawdowns)),
            "capital_rotations": int(switch_count), "market_exposure": float(exposure_days / max(1, len(values))),
            "cash_days": int(len(values) - exposure_days), "worst_fold_return": min(fold_returns) if fold_returns else None,
            "eligible": bool(fold_returns) and all(value > 0.0 for value in fold_returns),
            "loss_avoided_by_cash_usd": float(loss_avoided_usd), "profit_missed_by_cash_usd": float(profit_missed_usd),
            "net_cash_edge_usd": float(loss_avoided_usd - profit_missed_usd),
            "cash_intervention_sessions": int(sum(1 for row in intervals if row["baseline_target_asset"] != "CASH" and row["selected_asset"] == "CASH")),
            "intervention_counts": dict(intervention_counts), "folds": folds,
        },
        "equity": [
            {"timestamp": intervals[index]["next_execution_date"], "strategy_equity": float(equity[index]),
             "drawdown": float(drawdowns[index]), "selected_asset": intervals[index]["selected_asset"],
             "gate_reason": intervals[index]["gate_reason"]}
            for index in range(len(intervals))
        ],
        "intervals": intervals, "monthly_attribution": monthly_rows,
    }
