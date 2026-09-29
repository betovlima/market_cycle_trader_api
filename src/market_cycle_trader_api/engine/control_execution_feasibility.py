"""Research-only, state-aware Control next-session execution feasibility scenario.

NOT a live-execution simulator or guaranteed fill model. The OOS LightGBM
policy is queried each close using *actual feasible position and holding days*.
Next-session realized volume is used only ex post to constrain intraday fills,
never to select, rank, or size a pre-session signal. No shorting, leverage,
overlapping positions, forced liquidation or orders.
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..tcc_v106_reference.capital_rotation import (
    RotationRunResult,
    _annualized_sharpe,
    _cagr,
    _curve_risk_adjusted_score,
    _equal_weight_benchmark,
    _maximum_drawdown,
)

# Frozen *scenario assumptions*. They are not Alpaca quote data, a calibrated
# impact model, or instructions to place an order.
SCENARIO = {
    "participation_rate": 0.10,
    "prior_volume_lookback": 20,
    "minimum_prior_volume_observations": 5,
    "assumed_full_spread_bps": 15.0,
    "assumed_impact_coefficient_bps": 20.0,
    "whole_shares_only": True,
    "execution_reference": "next_session_open_plus_adverse_scenario_cost",
    "volume_observation": "same_session_realized_volume_ex_post_fill_constraint_only",
}


def _nonnegative(value: Any) -> float:
    try:
        number = float(value)
    except (ValueError, TypeError):
        return 0.0
    return number if math.isfinite(number) and number > 0 else 0.0


def _volume_capacity(frame: pd.DataFrame, date: pd.Timestamp, config: dict[str, Any]) -> dict[str, float]:
    """Previous volumes forecast capacity; realized volume limits actual fills.

    An open-time decision never reads execution-day volume. A historical
    end-of-day audit can use that actual volume to measure what could have
    filled throughout the next session; the fill price is only a scenario.
    """
    index = frame.index.get_indexer([date])
    if len(index) != 1 or index[0] < 0:
        raise ValueError("Execution date missing from the frozen Control calendar.")
    i = int(index[0])
    prior = pd.to_numeric(
        frame["volume"].iloc[max(0, i - int(config["prior_volume_lookback"])):i],
        errors="coerce",
    )
    # Zero-volume sessions are real evidence of illiquidity, not missing
    # observations. Include them in the historical median; only NaN/negative
    # input rows are discarded.
    prior = prior[np.isfinite(prior) & (prior >= 0)]
    realized = _nonnegative(frame["volume"].iloc[i])
    usable = len(prior) >= int(config["minimum_prior_volume_observations"])
    historical = float(prior.median()) if usable else 0.0
    expected_cap = math.floor(historical * float(config["participation_rate"]))
    realized_cap = math.floor(realized * float(config["participation_rate"]))
    capacity = max(0, min(expected_cap, realized_cap))
    return {
        "prior_median_volume": historical,
        "realized_daily_volume": realized,
        "expected_capacity_shares": expected_cap,
        "realized_capacity_shares": realized_cap,
        "ex_post_fill_capacity_shares": capacity,
        "prior_observation_count": len(prior),
    }


def _fill_price(open_price: float, side: str, shares: int, daily_volume: float,
                scenario: dict[str, Any]) -> tuple[float, float]:
    base = _nonnegative(open_price)
    if base == 0 or shares <= 0 or daily_volume <= 0:
        raise ValueError("Cannot price a trade with no open, size, or daily volume.")
    participation = shares / daily_volume
    if participation > float(scenario["participation_rate"]) + 1e-12:
        raise ValueError("Attempted fill above capped realized volume.")
    spread = float(scenario["assumed_full_spread_bps"]) / 2.0
    impact = float(scenario["assumed_impact_coefficient_bps"]) * math.sqrt(participation)
    cost_bps = spread + impact
    factor = 1.0 + cost_bps / 10_000.0 if side == "BUY" else 1.0 - cost_bps / 10_000.0
    if factor <= 0:
        raise ValueError("Unreasonable execution cost scenario.")
    return base * factor, cost_bps


def _affordable_shares(cash: float, limit: int, open_price: float, volume: float,
                       scenario: dict[str, Any], fee_calculator: Callable,
                       config: Any) -> tuple[int, float, float, dict[str, float]]:
    """Integer binary search; fees and size-dependent impact are included."""
    low, high = 0, max(0, limit)
    winner = (0, 0.0, 0.0, fee_calculator("BUY", 0, open_price, config))
    while low <= high:
        mid = (low + high) // 2
        if mid == 0:
            low = 1
            continue
        price, cost_bps = _fill_price(open_price, "BUY", mid, volume, scenario)
        fees = fee_calculator("BUY", mid, price, config)
        spent = mid * price + float(fees["total_fee"])
        if spent <= cash + 1e-9:
            winner = (mid, price, cost_bps, fees)
            low = mid + 1
        else:
            high = mid - 1
    return winner


def simulate_feasible_control(
    backend: str,
    policy: Callable[[pd.Timestamp, int, int], tuple[int, float]],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    decision_dates: pd.DatetimeIndex,
    config: Any,
    fee_calculator: Callable,
    slippage: Callable,
    decision_metadata: dict[pd.Timestamp, dict[str, Any]] | None = None,
    policy_decision_diagnostics: dict[pd.Timestamp, dict[str, Any]] | None = None,
    trade_callback: Callable[[dict[str, Any]], None] | None = None,
    *,
    model_label: str = "LightGBM Utility",
    method_line: str | None = None,
    simulation_progress_callback: Callable[[float, str], None] | None = None,
) -> RotationRunResult:
    """Model-policy counterfactual, not a fixed tape of original decisions.

    Persist a single partially filled position; sell residual shares before
    buying any other asset. Cash stays in CASH. A new policy call is made
    at each close with the actual position and holding history. Cash can be
    topped up only while that policy continues to select the held asset.
    """
    if len(decision_dates) < 2:
        raise ValueError("Feasibility replay needs at least one next-session execution.")
    if not symbols or any(s not in frames for s in symbols):
        raise ValueError("Invalid frozen eligible symbol list.")
    if bool(getattr(config, "rotation_model_repetitions", 1) != 1):
        raise ValueError("Execution feasibility scenario requires a single Control policy repetition.")
    scenario = dict(SCENARIO)
    started = time.perf_counter()
    capital = float(config.initial_capital)
    if not (math.isfinite(capital) and capital > 0):
        raise ValueError("Initial capital must be strictly positive.")
    cash, position, qty, holding = capital, 0, 0, 0
    total_fees, modeled_price_cost, rejected_qty = 0.0, 0.0, 0.0
    blocked_sessions = partial_sessions = zero_volume_blocks = 0
    trades: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    execution_dates = decision_dates[1:]
    benchmark = _equal_weight_benchmark(
        frames, symbols, execution_dates, capital, config, fee_calculator, slippage,
    )
    stride = max(1, len(execution_dates) // 20)
    last_price: dict[str, float] = {}
    for i, (decision_date, execution_date) in enumerate(zip(decision_dates[:-1], execution_dates)):
        old_position = position
        chosen_position, score = policy(decision_date, position, holding)
        if not isinstance(chosen_position, (int, np.integer)) or not 0 <= chosen_position <= len(symbols):
            raise ValueError("Control policy returned invalid target.")
        fold_id = (decision_metadata or {}).get(pd.Timestamp(decision_date), {}).get("fold_id")
        chosen = symbols[chosen_position - 1] if chosen_position else "CASH"
        before = symbols[position - 1] if position else "CASH"
        cash_before = cash
        new_trades = []
        intended = 0
        executed = 0
        reason = "HOLD"
        market_data: dict[str, dict[str, float]] = {}

        def data(symbol: str) -> tuple[float, dict[str, float]]:
            row = frames[symbol].loc[execution_date]
            open_ = _nonnegative(row.get("open"))
            cap = _volume_capacity(frames[symbol], execution_date, scenario)
            market_data[symbol] = cap
            return open_, cap

        def save_trade(side: str, symbol: str, amount: int, price: float,
                       cost_bps: float, fees: dict[str, float], cap: dict[str, float],
                       requested: int, reason_value: str) -> None:
            nonlocal total_fees, modeled_price_cost
            total_fees += float(fees["total_fee"])
            modeled_price_cost += amount * abs(price - _nonnegative(frames[symbol].loc[execution_date, "open"]))
            trade = {
                "timestamp": execution_date,
                "decision_timestamp": decision_date,
                "walk_forward_fold": fold_id,
                "action": side,
                "asset": symbol,
                "reason": reason_value,
                "requested_quantity": int(requested),
                "quantity": int(amount),
                "unfilled_quantity": int(max(0, requested - amount)),
                "execution_price": float(price),
                "gross_trade_value": float(amount * price),
                **fees,
                "scenario_price_cost_bps": cost_bps,
                "realized_daily_volume": cap["realized_daily_volume"],
                "prior_median_volume": cap["prior_median_volume"],
                "maximum_fill_quantity": cap["ex_post_fill_capacity_shares"],
                "realized_volume_participation": amount / cap["realized_daily_volume"],
                "cash_after_trade": float(cash),
                "shares_after_trade": int(qty),
                "same_session_volume_used_for_signal": False,
            }
            new_trades.append(trade)
            trades.append(trade)
            if trade_callback is not None:
                trade_callback({**trade, "backend": backend, "model": model_label})

        # Sell up to the cap, retaining an old position until it is fully
        # liquidated. Partial proceeds can never purchase another asset while
        # the old asset remains: the Control action space is single-position.
        if position and chosen_position != position:
            symbol = symbols[position - 1]
            open_, cap = data(symbol)
            intended += qty
            capacity = int(cap["ex_post_fill_capacity_shares"]) if open_ else 0
            sold = min(qty, capacity)
            if sold:
                price, cost = _fill_price(open_, "SELL", sold, cap["realized_daily_volume"], scenario)
                fees = fee_calculator("SELL", sold, price, config)
                cash += sold * price - float(fees["total_fee"])
                if cash < -1e-7:
                    raise AssertionError("Sell fees would create negative portfolio cash.")
                cash = max(0.0, cash)
                qty -= sold
                executed += sold
                reason = "PARTIAL_SELL" if qty else "SELL_FILLED"
                save_trade("SELL", symbol, sold, price, cost, fees, cap, qty + sold, reason)
                if qty == 0:
                    position = 0
                    holding = 0
            else:
                reason = "SELL_BLOCKED"
                if cap["realized_daily_volume"] <= 0:
                    zero_volume_blocks += 1

        # If the old position was fully sold, or the model still wants the
        # position partially held, spend only available cash and daily cap.
        if chosen_position and (position == 0 or chosen_position == position):
            symbol = symbols[chosen_position - 1]
            open_, cap = data(symbol)
            if open_ and cap["ex_post_fill_capacity_shares"] > 0 and cash > open_:
                maximum = int(cap["ex_post_fill_capacity_shares"])
                approximate_target = int(cash // open_)
                intended += approximate_target
                buy_qty, price, cost, fees = _affordable_shares(
                    cash, maximum, open_, cap["realized_daily_volume"],
                    scenario, fee_calculator, config,
                )
                if buy_qty:
                    cash -= buy_qty * price + float(fees["total_fee"])
                    if cash < -1e-7:
                        raise AssertionError("Execution feasibility attempted negative cash.")
                    cash = max(0.0, cash)
                    qty += buy_qty
                    executed += buy_qty
                    position = chosen_position
                    if holding == 0:
                        holding = 1
                    else:
                        holding += 1
                    reason = "PARTIAL_BUY" if buy_qty < approximate_target else "BUY_FILLED"
                    save_trade("BUY", symbol, buy_qty, price, cost, fees, cap, approximate_target, reason)
                else:
                    reason = "BUY_BLOCKED"
            elif position == 0 or cash > open_:
                if cap["realized_daily_volume"] <= 0:
                    zero_volume_blocks += 1
                reason = "BUY_BLOCKED"
        if position > 0 and qty > 0 and not any(t["action"] == "BUY" for t in new_trades):
            holding += 1
        if position == 0:
            qty = 0
            holding = 0
        if intended > executed:
            rejected_qty += intended - executed
            if executed == 0:
                blocked_sessions += 1
            else:
                partial_sessions += 1
        # Mark to market at completed day's close; no hypothetical terminal
        # sale of an illiquid residual position. Missing daily bars retain
        # last *known* close, and are audited as stale.
        held = symbols[position - 1] if position else "CASH"
        stale_price = False
        if position:
            close = _nonnegative(frames[held].loc[execution_date, "close"])
            if close:
                last_price[held] = close
            else:
                close = last_price.get(held, 0.0)
                stale_price = True
            if close <= 0:
                raise ValueError("Cannot mark an illiquid held position without a past close.")
            equity = cash + qty * close
        else:
            equity = cash
        if equity < 0 or not math.isfinite(equity):
            raise AssertionError("Invalid constrained portfolio equity.")
        rows.append({
            "timestamp": execution_date,
            "decision_date": decision_date,
            "strategy_equity": equity,
            "buy_hold_equity": float(benchmark.loc[execution_date]),
            "selected_asset": held,
            "signal_asset": chosen,
            "previous_asset": before,
            "holding_days": holding,
            "shares": int(qty),
            "cash": float(cash),
            "cash_weight": float(cash / equity) if equity else 1.0,
            "trade_action": "|".join(t["action"] for t in new_trades),
            "trade_reason": reason,
            "requested_quantity": intended,
            "executed_quantity": executed,
            "unfilled_quantity": max(0, intended - executed),
            "mark_stale": stale_price,
            "walk_forward_fold": fold_id,
            "decision_score": float(score),
            "actual_position_used_in_policy": True,
        })
        if simulation_progress_callback is not None and ((i + 1) % stride == 0 or i + 1 == len(execution_dates)):
            simulation_progress_callback(
                (i + 1) / len(execution_dates),
                f"Feasibility portfolio replay {i+1}/{len(execution_dates)}",
            )
    predictions = pd.DataFrame(rows).set_index("timestamp")
    predictions.index = pd.DatetimeIndex(pd.to_datetime(predictions.index, utc=True))
    predictions.index.name = "timestamp"
    trade_df = pd.DataFrame(trades)
    end = float(predictions["strategy_equity"].iloc[-1])
    equity_curve = predictions["strategy_equity"].astype(float)
    benchmark_curve = predictions["buy_hold_equity"].astype(float)
    days = max(1, (pd.Timestamp(execution_dates[-1]) - pd.Timestamp(execution_dates[0])).days)
    years = max(days / 365.25, 1 / 365.25)
    buy_assets = (
        trade_df.loc[trade_df["action"] == "BUY", "asset"].tolist()
        if not trade_df.empty else []
    )
    completed_asset_rotations = sum(
        previous != current
        for previous, current in zip(buy_assets, buy_assets[1:])
    )
    policy_target_changes = int(sum(
        a != b for a, b in zip(predictions["previous_asset"], predictions["signal_asset"])
    ))
    cash_exposure = float(predictions["cash_weight"].mean())
    metrics = {
        "initial_capital": capital,
        "strategy_ending_capital": end,
        "strategy_return": end / capital - 1.0,
        "strategy_cagr": _cagr(equity_curve, capital),
        "strategy_sharpe": _annualized_sharpe(equity_curve, 252.0),
        "strategy_maximum_drawdown": _maximum_drawdown(equity_curve),
        "risk_adjusted_compound_score": _curve_risk_adjusted_score(equity_curve, config),
        "buy_hold_ending_capital": float(benchmark_curve.iloc[-1]),
        "buy_hold_return": float(benchmark_curve.iloc[-1] / capital - 1.0),
        "market_exposure": 1.0 - cash_exposure,
        "cash_weight_mean": cash_exposure,
        "cash_days": int((predictions["shares"] == 0).sum()),
        "capital_rotations": int(completed_asset_rotations),
        "policy_target_changes": policy_target_changes,
        "simulated_buys": int((trade_df["action"] == "BUY").sum()) if not trade_df.empty else 0,
        "simulated_sells": int((trade_df["action"] == "SELL").sum()) if not trade_df.empty else 0,
        "execution_scenario": scenario,
        "blocked_sessions": int(blocked_sessions),
        "partial_sessions": int(partial_sessions),
        "zero_volume_block_sessions": int(zero_volume_blocks),
        "unfilled_requested_shares": float(rejected_qty),
        "modeled_price_cost_usd": float(modeled_price_cost),
        "total_transaction_fees": float(total_fees),
        "terminal_holdings_asset": symbols[position - 1] if position else "CASH",
        "terminal_holdings_shares": int(qty),
        "terminal_cash": float(cash),
        "terminal_mark_to_market_only": True,
        "stale_mark_sessions": int(predictions["mark_stale"].sum()),
        "session_count": len(execution_dates),
        "simulation_profile": {"session_count": len(execution_dates), "total_seconds": time.perf_counter()-started},
        "research_only": True,
        "no_order_submission": True,
        "no_lookahead_policy_decision": True,
        "daily_volume_is_ex_post_realized_fill_cap_not_pretrade_signal": True,
        "unconstrained_benchmark_not_capacity_checked": True,
    }
    summary = (
        "CONTROL EXECUTION FEASIBILITY — scenario, not broker fill guarantee. "
        "Same chronological LightGBM OOS policy retrained by fold, "
        "state-aware, capital-only, capped next-session fills, no terminal liquidation. "
        f"Capital {capital:,.2f} -> {end:,.2f}; source cutoff {execution_dates[-1]}."
    )
    return RotationRunResult(backend=backend, predictions=predictions,
                             trades=trade_df, summary=summary, metrics=metrics)
