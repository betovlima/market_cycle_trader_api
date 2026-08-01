from __future__ import annotations

from typing import Any

import pandas as pd
from pymongo.database import Database

from .common import (
    _classify_relative_episode,
    _diagnostic_trade_frame,
    _future_market_prices,
    _market_close_series,
    _safe_float,
)


def build_performance_diagnostics(
    db: Database,
    prediction_rows: list[dict[str, Any]],
    trade_rows: list[dict[str, Any]],
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not prediction_rows:
        return {}

    frame = pd.DataFrame(prediction_rows).copy()
    required = {"timestamp", "strategy_equity", "buy_hold_equity"}
    if not required.issubset(frame.columns):
        return {}

    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"],
        utc=True,
        errors="coerce",
    )
    frame["strategy_equity"] = pd.to_numeric(
        frame["strategy_equity"],
        errors="coerce",
    )
    frame["buy_hold_equity"] = pd.to_numeric(
        frame["buy_hold_equity"],
        errors="coerce",
    )
    frame = frame.dropna(
        subset=["timestamp", "strategy_equity", "buy_hold_equity"]
    ).sort_values("timestamp")
    frame = frame.loc[frame["buy_hold_equity"] > 0].reset_index(drop=True)

    if frame.empty:
        return {}

    frame["relative_ratio"] = (
        frame["strategy_equity"] / frame["buy_hold_equity"]
    )
    frame["relative_peak"] = frame["relative_ratio"].cummax()
    frame["relative_drawdown"] = (
        frame["relative_ratio"] / frame["relative_peak"] - 1.0
    )

    trade_frame = _diagnostic_trade_frame(trade_rows)
    is_day_trade = str((metrics or {}).get("strategy_mode", "")).startswith("COMPOUND_ROTATION_DAY_TRADE")
    day_trade_diagnostics = {}




    episodes: list[dict[str, Any]] = []
    peak_idx = 0
    in_episode = False
    trough_idx = 0

    for idx in range(1, len(frame)):
        ratio = float(frame.loc[idx, "relative_ratio"])
        peak_ratio = float(frame.loc[peak_idx, "relative_ratio"])

        if ratio >= peak_ratio:
            if in_episode:
                start_idx = peak_idx
                end_idx = idx
                depth = float(frame.loc[trough_idx, "relative_drawdown"])
                if depth <= -0.03:
                    episodes.append(
                        {
                            "_start_idx": start_idx,
                            "_trough_idx": trough_idx,
                            "_end_idx": end_idx,
                            "_recovered": True,
                        }
                    )
                in_episode = False
            peak_idx = idx
            trough_idx = idx
            continue

        if not in_episode:
            in_episode = True
            trough_idx = idx
        elif (
            float(frame.loc[idx, "relative_drawdown"])
            < float(frame.loc[trough_idx, "relative_drawdown"])
        ):
            trough_idx = idx

    if in_episode:
        depth = float(frame.loc[trough_idx, "relative_drawdown"])
        if depth <= -0.03:
            episodes.append(
                {
                    "_start_idx": peak_idx,
                    "_trough_idx": trough_idx,
                    "_end_idx": len(frame) - 1,
                    "_recovered": False,
                }
            )

    enriched_episodes: list[dict[str, Any]] = []
    for episode in episodes:
        start_idx = int(episode["_start_idx"])
        trough_idx = int(episode["_trough_idx"])
        end_idx = int(episode["_end_idx"])
        start_row = frame.loc[start_idx]
        trough_row = frame.loc[trough_idx]
        end_row = frame.loc[end_idx]

        window = frame.iloc[start_idx : trough_idx + 1]
        start_time = pd.Timestamp(start_row["timestamp"])
        trough_time = pd.Timestamp(trough_row["timestamp"])
        end_time = pd.Timestamp(end_row["timestamp"])

        strategy_start = float(start_row["strategy_equity"])
        strategy_trough = float(trough_row["strategy_equity"])
        benchmark_start = float(start_row["buy_hold_equity"])
        benchmark_trough = float(trough_row["buy_hold_equity"])

        strategy_return = (
            strategy_trough / strategy_start - 1.0
            if strategy_start > 0
            else 0.0
        )
        benchmark_return = (
            benchmark_trough / benchmark_start - 1.0
            if benchmark_start > 0
            else 0.0
        )
        cash_days = (
            int(
                window.get(
                    "selected_asset",
                    pd.Series(index=window.index, dtype=object),
                )
                .fillna("")
                .astype(str)
                .eq("CASH")
                .sum()
            )
            if len(window)
            else 0
        )
        duration_sessions = max(1, len(window))
        cash_share = cash_days / duration_sessions

        window_actions = (
            window.get(
                "trade_action",
                pd.Series(index=window.index, dtype=object),
            )
            .fillna("")
            .astype(str)
        )
        rotations = int(window_actions.eq("ROTATE").sum())

        episode_trades = pd.DataFrame()
        if not trade_frame.empty:
            episode_trades = trade_frame.loc[
                (trade_frame["timestamp"] >= start_time)
                & (trade_frame["timestamp"] <= trough_time)
            ]

        buys = (
            int(episode_trades["action"].astype(str).eq("BUY").sum())
            if not episode_trades.empty and "action" in episode_trades
            else 0
        )
        sells = (
            int(
                episode_trades["action"]
                .astype(str)
                .isin(["SELL", "FINAL_SELL"])
                .sum()
            )
            if not episode_trades.empty and "action" in episode_trades
            else 0
        )
        average_holding: float | None = None
        if (
            not episode_trades.empty
            and "holding_bars" in episode_trades
            and "action" in episode_trades
        ):
            completed = episode_trades.loc[
                episode_trades["action"].astype(str).isin(
                    ["SELL", "FINAL_SELL"]
                )
            ]
            values = pd.to_numeric(
                completed.get("holding_bars"),
                errors="coerce",
            ).dropna()
            if not values.empty:
                average_holding = float(values.mean())

        selected_assets = (
            window.get(
                "selected_asset",
                pd.Series(index=window.index, dtype=object),
            )
            .fillna("UNKNOWN")
            .astype(str)
        )
        dominant_assets = [
            {"asset": str(asset), "days": int(days)}
            for asset, days in selected_assets.value_counts()
            .head(3)
            .items()
        ]

        likely_cause = _classify_relative_episode(
            benchmark_return=benchmark_return,
            strategy_return=strategy_return,
            cash_share=cash_share,
            rotations=rotations,
            duration_sessions=duration_sessions,
            average_holding=average_holding,
        )

        enriched_episodes.append(
            {
                "start": start_time,
                "trough": trough_time,
                "recovery": end_time if episode["_recovered"] else None,
                "recovered": bool(episode["_recovered"]),
                "relative_drawdown": float(
                    trough_row["relative_drawdown"]
                ),
                "relative_ratio_start": float(start_row["relative_ratio"]),
                "relative_ratio_trough": float(
                    trough_row["relative_ratio"]
                ),
                "strategy_return_to_trough": strategy_return,
                "buy_hold_return_to_trough": benchmark_return,
                "relative_return_gap": strategy_return - benchmark_return,
                "strategy_capital_start": strategy_start,
                "strategy_capital_trough": strategy_trough,
                "buy_hold_capital_start": benchmark_start,
                "buy_hold_capital_trough": benchmark_trough,
                "duration_sessions_to_trough": duration_sessions,
                "cash_days": cash_days,
                "cash_share": cash_share,
                "rotations": rotations,
                "buys": buys,
                "sells": sells,
                "average_holding_days": average_holding,
                "dominant_assets": dominant_assets,
                "likely_cause": likely_cause,
            }
        )

    enriched_episodes = sorted(
        enriched_episodes,
        key=lambda item: float(item["relative_drawdown"]),
    )[:10]



    exit_diagnostics: list[dict[str, Any]] = []
    if (not is_day_trade) and (not trade_frame.empty) and "action" in trade_frame:
        sells = trade_frame.loc[
            trade_frame["action"].astype(str).eq("SELL")
        ].copy()

        if not sells.empty:
            min_date = pd.Timestamp(sells["timestamp"].min())
            max_date = pd.Timestamp(sells["timestamp"].max()) + pd.Timedelta(
                days=45
            )
            asset_series: dict[str, pd.Series] = {}

            for asset in sorted(
                {
                    str(value)
                    for value in sells.get("asset", pd.Series(dtype=object))
                    .dropna()
                    .tolist()
                }
            ):
                asset_series[asset] = _market_close_series(
                    db,
                    asset,
                    min_date,
                    max_date,
                )

            for _, trade in sells.iterrows():
                asset = str(trade.get("asset") or "")
                sold_at = pd.Timestamp(trade["timestamp"])
                sale_price = _safe_float(trade.get("execution_price"))
                if not asset or sale_price is None or sale_price <= 0:
                    continue

                prices = asset_series.get(asset)
                future = _future_market_prices(prices, sold_at)
                if future.empty:
                    continue

                returns: dict[int, float | None] = {}
                for horizon in (1, 5, 10, 20):
                    if len(future) >= horizon:
                        future_price = float(future.iloc[horizon - 1])
                        returns[horizon] = future_price / sale_price - 1.0
                    else:
                        returns[horizon] = None

                available = [
                    value for value in returns.values() if value is not None
                ]
                max_future_return = max(available) if available else None

                r10 = returns.get(10)
                r20 = returns.get(20)
                if (
                    (r10 is not None and r10 >= 0.08)
                    or (r20 is not None and r20 >= 0.12)
                ):
                    classification = "EARLY_EXIT_CANDIDATE"
                elif (
                    (r10 is not None and r10 <= -0.05)
                    or (r20 is not None and r20 <= -0.08)
                ):
                    classification = "DEFENSIVE_EXIT"
                else:
                    classification = "NEUTRAL_EXIT"

                exit_diagnostics.append(
                    {
                        "timestamp": sold_at,
                        "asset": asset,
                        "reason": trade.get("reason") or "",
                        "sale_price": sale_price,
                        "realized_pnl": _safe_float(
                            trade.get("realized_pnl")
                        ),
                        "holding_bars": _safe_float(
                            trade.get("holding_bars")
                        ),
                        "return_after_1d": returns.get(1),
                        "return_after_5d": returns.get(5),
                        "return_after_10d": returns.get(10),
                        "return_after_20d": returns.get(20),
                        "max_future_return_20d": max_future_return,
                        "classification": classification,
                    }
                )

    exit_diagnostics = sorted(
        exit_diagnostics,
        key=lambda item: (
            item.get("max_future_return_20d")
            if item.get("max_future_return_20d") is not None
            else -999.0
        ),
        reverse=True,
    )

    worst_idx = int(frame["relative_ratio"].idxmin())
    worst_row = frame.loc[worst_idx]
    ending_ratio = float(frame.iloc[-1]["relative_ratio"])
    below_days = int((frame["relative_ratio"] < 1.0).sum())
    total_days = int(len(frame))

    return {
        "summary": {
            "ending_relative_ratio": ending_ratio,
            "ending_relative_excess": ending_ratio - 1.0,
            "worst_relative_ratio": float(worst_row["relative_ratio"]),
            "worst_relative_date": pd.Timestamp(worst_row["timestamp"]),
            "worst_relative_drawdown": float(
                frame["relative_drawdown"].min()
            ),
            "days_below_buy_hold": below_days,
            "observations_below_buy_hold": below_days,
            "comparison_observation_label": "sessions",
            "total_sessions": total_days,
            "total_observations": total_days,
            "share_sessions_below_buy_hold": (
                below_days / total_days if total_days else 0.0
            ),
            "episode_count": len(enriched_episodes),
            "early_exit_candidates": int(
                sum(
                    item["classification"] == "EARLY_EXIT_CANDIDATE"
                    for item in exit_diagnostics
                )
            ),
            "defensive_exits": int(
                sum(
                    item["classification"] == "DEFENSIVE_EXIT"
                    for item in exit_diagnostics
                )
            ),
        },
        "underperformance_periods": enriched_episodes,
        "exit_diagnostics": exit_diagnostics[:30],
        **day_trade_diagnostics,
    }
