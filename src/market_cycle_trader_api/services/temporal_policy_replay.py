from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _timestamp_key(value: Any) -> str | None:
    if value is None:
        return None
    try:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        return timestamp.isoformat()
    except Exception:
        raw = str(value).strip()
        return raw or None


def _replay_rows(
    observations: dict[str, dict[str, Any]],
    winner_rows: list[dict[str, Any]],
    *,
    initial_capital: float,
    one_side_cost: float,
    settings: dict[str, Any],
    winner_fold_returns: dict[int, float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    winner_by_decision = {
        key: row
        for row in winner_rows
        if isinstance(row, dict) and (key := _timestamp_key(row.get("decision_date")))
    }
    weak_threshold = float(settings["timing_base_weak_threshold"])
    challenger_minimum = float(settings["timing_challenger_minimum"])
    minimum_advantage = float(settings["timing_minimum_advantage"])
    maximum_advantage = float(settings.get("timing_maximum_advantage", 1.0))
    returns: list[float] = []
    return_folds: list[int] = []
    return_dates: list[str] = []
    fold_close_flags: list[bool] = []
    exposure_days = 0
    switch_count = 0
    override_count = 0
    current_fold: int | None = None
    current_symbol: str | None = None

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
            if current_symbol is not None and returns:
                returns[-1] = max(1e-9, (1.0 + returns[-1]) * max(1e-9, 1.0 - one_side_cost)) - 1.0
                fold_close_flags[-1] = True
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
        override = False
        if base_short is not None and challenger_short is not None:
            override = bool(
                base_symbol == top1_symbol
                and challenger_symbol != base_symbol
                and base_short < weak_threshold
                and challenger_short >= challenger_minimum
                and (challenger_short - base_short) >= minimum_advantage
                and (challenger_short - base_short) <= maximum_advantage
            )
        target_symbol = challenger_symbol if override else base_symbol

        def interval_return(symbol: str | None) -> float | None:
            if symbol is None:
                return 0.0
            row = rows_by_symbol.get(symbol)
            return _finite((row or {}).get("open_to_open_return"))

        target_return = interval_return(target_symbol)
        if target_symbol is not None and target_return is None:
            base_return = interval_return(base_symbol)
            if override and base_symbol is not None and base_return is not None:
                target_symbol = base_symbol
                target_return = base_return
                override = False
            else:
                target_symbol = None
                target_return = 0.0

        if override:
            override_count += 1
        if current_symbol is None and target_symbol is not None:
            cost_sides = 1
            switch_count += 1
        elif current_symbol is not None and target_symbol is None:
            cost_sides = 1
            switch_count += 1
        elif current_symbol is not None and target_symbol is not None and current_symbol != target_symbol:
            cost_sides = 2
            switch_count += 1
        else:
            cost_sides = 0

        if target_symbol is not None and target_return is not None:
            exposure_days += 1
        gross_return = float(target_return or 0.0)
        factor = max(1e-9, 1.0 - float(cost_sides) * one_side_cost) * max(1e-9, 1.0 + gross_return)
        returns.append(factor - 1.0)
        return_folds.append(fold_id)
        return_dates.append(key)
        fold_close_flags.append(False)
        current_symbol = target_symbol

    if current_symbol is not None and returns:
        returns[-1] = max(1e-9, (1.0 + returns[-1]) * max(1e-9, 1.0 - one_side_cost)) - 1.0
        fold_close_flags[-1] = True
    if not returns:
        raise ValueError("Frozen Temporal replay has no usable decision intervals.")

    values = np.asarray(returns, dtype=float)
    equity = float(initial_capital) * np.cumprod(1.0 + values)
    peaks = np.maximum.accumulate(equity)
    drawdowns = equity / peaks - 1.0
    years = max(len(values) / 252.0, 1.0 / 252.0)
    ending_capital = float(equity[-1])
    cagr = (ending_capital / float(initial_capital)) ** (1.0 / years) - 1.0 if ending_capital > 0 else -1.0
    volatility = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    sharpe = float(np.mean(values) / volatility * math.sqrt(252.0)) if volatility > 1e-12 else 0.0

    folds: list[dict[str, Any]] = []
    for fold_id in sorted(set(return_folds)):
        indices = [index for index, value in enumerate(return_folds) if value == fold_id]
        fold_values = values[indices]
        fold_equity = float(initial_capital) * np.cumprod(1.0 + fold_values)
        fold_peaks = np.maximum.accumulate(fold_equity)
        fold_drawdown = fold_equity / fold_peaks - 1.0
        fold_return = float(fold_equity[-1] / float(initial_capital) - 1.0)
        folds.append({
            "fold_id": int(fold_id),
            "strategy_return": fold_return,
            "maximum_drawdown": float(np.min(fold_drawdown)),
            "benchmark_return": float(winner_fold_returns.get(fold_id, 0.0)),
        })
    fold_returns = [row["strategy_return"] for row in folds]
    metrics = {
        "initial_capital": float(initial_capital),
        "ending_capital": ending_capital,
        "strategy_return": ending_capital / float(initial_capital) - 1.0,
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "maximum_drawdown": float(np.min(drawdowns)),
        "risk_adjusted_compound_score": float(cagr + 0.15 * sharpe + 0.35 * float(np.min(drawdowns))),
        "turnover_ratio": float(switch_count / max(1, len(values))),
        "capital_rotations": int(switch_count),
        "average_holding_days": float(len(values) / max(1, switch_count)),
        "market_exposure": float(exposure_days / max(1, len(values))),
        "cash_days": int(len(values) - exposure_days),
        "benchmark_ending_capital": 0.0,
        "market_data_signature_sha256": None,
        "market_data_last_timestamp": return_dates[-1] if return_dates else None,
        "folds": folds,
        "worst_fold_return": min(fold_returns) if fold_returns else None,
        "eligible": bool(fold_returns) and all(value > 0 for value in fold_returns),
        "timing_override_count": int(override_count),
    }
    preview_indices = np.linspace(0, len(equity) - 1, min(500, len(equity)), dtype=int)
    preview = [
        {
            "timestamp": return_dates[int(index)],
            "simulation_equity": float(equity[int(index)]),
            "reference_equity": None,
            "selected_asset": None,
            "trade_action": None,
            "cash_edge": None,
        }
        for index in preview_indices
    ]
    return metrics, preview



def replay_temporal_policy_details(
    observations: dict[str, dict[str, Any]],
    winner_rows: list[dict[str, Any]],
    *,
    initial_capital: float,
    one_side_cost: float,
    settings: dict[str, Any],
    winner_fold_returns: dict[int, float],
) -> dict[str, Any]:
    """Replay a frozen Temporal policy and retain the full analytical path.

    The metric result is delegated to ``_replay_rows`` so tuning and validated
    analytics share exactly the same economic score. The second pass retains
    every interval and allocation change for Dashboard drill-downs.
    """
    metrics, _ = _replay_rows(
        observations,
        winner_rows,
        initial_capital=initial_capital,
        one_side_cost=one_side_cost,
        settings=settings,
        winner_fold_returns=winner_fold_returns,
    )
    winner_by_decision = {
        key: row
        for row in winner_rows
        if isinstance(row, dict) and (key := _timestamp_key(row.get("decision_date")))
    }
    weak_threshold = float(settings["timing_base_weak_threshold"])
    challenger_minimum = float(settings["timing_challenger_minimum"])
    minimum_advantage = float(settings["timing_minimum_advantage"])
    maximum_advantage = float(settings.get("timing_maximum_advantage", 1.0))

    returns: list[float] = []
    intervals: list[dict[str, Any]] = []
    current_fold: int | None = None
    current_symbol: str | None = None

    for key in sorted(observations, key=lambda item: pd.Timestamp(item)):
        payload = observations[key]
        fold_id = int(payload.get("fold_id") or 0)
        rows_by_symbol = payload.get("rows_by_symbol") or {}
        winner_row = winner_by_decision.get(key)
        if fold_id <= 0 or winner_row is None or not rows_by_symbol:
            continue
        if not any(_finite(row.get("open_to_open_return")) is not None for row in rows_by_symbol.values()):
            continue

        if current_fold is not None and fold_id != current_fold:
            if current_symbol is not None and returns:
                returns[-1] = max(1e-9, (1.0 + returns[-1]) * max(1e-9, 1.0 - one_side_cost)) - 1.0
                intervals[-1]["fold_close_cost_applied"] = True
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
        override = False
        if base_short is not None and challenger_short is not None:
            override = bool(
                base_symbol == top1_symbol
                and challenger_symbol != base_symbol
                and base_short < weak_threshold
                and challenger_short >= challenger_minimum
                and (challenger_short - base_short) >= minimum_advantage
                and (challenger_short - base_short) <= maximum_advantage
            )
        target_symbol = challenger_symbol if override else base_symbol

        def interval_return(symbol: str | None) -> float | None:
            if symbol is None:
                return 0.0
            row = rows_by_symbol.get(symbol)
            return _finite((row or {}).get("open_to_open_return"))

        target_return = interval_return(target_symbol)
        if target_symbol is not None and target_return is None:
            base_return = interval_return(base_symbol)
            if override and base_symbol is not None and base_return is not None:
                target_symbol = base_symbol
                target_return = base_return
                override = False
            else:
                target_symbol = None
                target_return = 0.0

        previous_symbol = current_symbol
        if current_symbol is None and target_symbol is not None:
            cost_sides = 1
        elif current_symbol is not None and target_symbol is None:
            cost_sides = 1
        elif current_symbol is not None and target_symbol is not None and current_symbol != target_symbol:
            cost_sides = 2
        else:
            cost_sides = 0

        target_row = rows_by_symbol.get(target_symbol) if target_symbol else None
        previous_row = rows_by_symbol.get(previous_symbol) if previous_symbol else None
        sample_row = target_row or base_row or challenger_row or next(iter(rows_by_symbol.values()), {})
        gross_return = float(target_return or 0.0)
        factor = max(1e-9, 1.0 - float(cost_sides) * one_side_cost) * max(1e-9, 1.0 + gross_return)
        returns.append(factor - 1.0)
        intervals.append({
            "decision_date": key,
            "execution_date": _timestamp_key((sample_row or {}).get("execution_date")) or key,
            "next_execution_date": _timestamp_key((sample_row or {}).get("next_execution_date")) or key,
            "fold_id": fold_id,
            "from_asset": previous_symbol or "CASH",
            "selected_asset": target_symbol or "CASH",
            "base_asset": base_symbol or "CASH",
            "top_1_asset": top1_symbol or "CASH",
            "top_2_asset": challenger_symbol or "CASH",
            "timing_override": bool(override),
            "gross_return": gross_return,
            "cost_sides": int(cost_sides),
            "buy_execution_price": _finite((target_row or {}).get("execution_open")),
            "sell_execution_price": _finite((previous_row or {}).get("execution_open")),
            "fold_close_cost_applied": False,
        })
        current_symbol = target_symbol

    if current_symbol is not None and returns:
        returns[-1] = max(1e-9, (1.0 + returns[-1]) * max(1e-9, 1.0 - one_side_cost)) - 1.0
        intervals[-1]["fold_close_cost_applied"] = True

    values = np.asarray(returns, dtype=float)
    equity_values = float(initial_capital) * np.cumprod(1.0 + values)
    peaks = np.maximum.accumulate(equity_values)
    drawdowns = equity_values / peaks - 1.0

    equity_rows: list[dict[str, Any]] = []
    rotations: list[dict[str, Any]] = []
    entry_equity: float | None = None
    entry_session: int | None = None
    entry_price: float | None = None
    prior_equity = float(initial_capital)

    for index, interval in enumerate(intervals):
        current_equity = float(equity_values[index])
        timestamp = interval["next_execution_date"] or interval["execution_date"]
        selected_asset = str(interval["selected_asset"] or "CASH")
        equity_rows.append({
            "timestamp": timestamp,
            "simulation_equity": current_equity,
            "reference_equity": None,
            "drawdown": float(drawdowns[index]),
            "selected_asset": selected_asset,
            "trade_action": (
                "HOLD"
                if str(interval["from_asset"] or "CASH") == selected_asset
                else ("BUY" if str(interval["from_asset"] or "CASH") == "CASH" else ("SELL" if selected_asset == "CASH" else "ROTATE"))
            ),
            "timing_override": bool(interval["timing_override"]),
        })

        from_asset = str(interval["from_asset"] or "CASH")
        to_asset = selected_asset
        if from_asset != to_asset:
            realized_pnl: float | None = None
            position_return: float | None = None
            holding_days: int | None = None
            if from_asset != "CASH" and entry_equity not in {None, 0.0}:
                position_return = prior_equity / float(entry_equity) - 1.0
                realized_pnl = prior_equity - float(entry_equity)
                holding_days = max(1, index - int(entry_session or index) + 1)
            transaction_fees = max(0.0, prior_equity * float(interval["cost_sides"]) * float(one_side_cost))
            rotations.append({
                "sequence": len(rotations) + 1,
                "executed_at": interval["execution_date"],
                "from_asset": from_asset,
                "to_asset": to_asset,
                "holding_days": holding_days,
                "position_return": position_return,
                "realized_pnl": realized_pnl,
                "transaction_fees": transaction_fees,
                "sell_execution_price": interval.get("sell_execution_price"),
                "buy_execution_price": interval.get("buy_execution_price"),
                "timing_override": bool(interval["timing_override"]),
            })
            if to_asset != "CASH":
                buy_fee = prior_equity * float(one_side_cost)
                entry_equity = max(1e-9, prior_equity - buy_fee)
                entry_session = index
                entry_price = interval.get("buy_execution_price")
            else:
                entry_equity = None
                entry_session = None
                entry_price = None

        prior_equity = current_equity

    return {
        "metrics": metrics,
        "equity": equity_rows,
        "rotations": rotations,
        "intervals": intervals,
    }
