from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


def _timestamp(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return parsed.tz_localize("UTC")
    return parsed.tz_convert("UTC")


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _frame_at(frames: dict[str, pd.DataFrame], symbol: str) -> pd.DataFrame | None:
    frame = frames.get(str(symbol or ""))
    if frame is None or frame.empty:
        return None
    if not isinstance(frame.index, pd.DatetimeIndex):
        return None
    if frame.index.tz is None:
        frame = frame.copy()
        frame.index = frame.index.tz_localize("UTC")
    return frame.sort_index()


def _open_to_open_return(
    frames: dict[str, pd.DataFrame],
    symbol: str,
    start_at: Any,
    end_at: Any,
) -> float | None:
    frame = _frame_at(frames, symbol)
    start = _timestamp(start_at)
    end = _timestamp(end_at)
    if frame is None or start is None or end is None or end <= start:
        return None
    try:
        start_open = _number(frame.loc[start, "open"])
        end_open = _number(frame.loc[end, "open"])
    except (KeyError, TypeError):
        return None
    if start_open in {None, 0.0} or end_open is None:
        return None
    return float(end_open / start_open - 1.0)


def _position_excursions(
    frames: dict[str, pd.DataFrame],
    symbol: str,
    entry_at: Any,
    exit_at: Any,
    entry_price: Any,
) -> dict[str, float | None]:
    frame = _frame_at(frames, symbol)
    start = _timestamp(entry_at)
    end = _timestamp(exit_at)
    base = _number(entry_price)
    if frame is None or start is None or end is None or base in {None, 0.0} or end < start:
        return {
            "maximum_favorable_excursion": None,
            "maximum_adverse_excursion": None,
        }

    holding = frame.loc[(frame.index >= start) & (frame.index < end)]
    highs = pd.to_numeric(holding.get("high"), errors="coerce").dropna() if not holding.empty else pd.Series(dtype=float)
    lows = pd.to_numeric(holding.get("low"), errors="coerce").dropna() if not holding.empty else pd.Series(dtype=float)

    exit_open = None
    try:
        exit_open = _number(frame.loc[end, "open"])
    except (KeyError, TypeError):
        pass

    high_values = highs.tolist()
    low_values = lows.tolist()
    if exit_open is not None:
        high_values.append(exit_open)
        low_values.append(exit_open)
    if not high_values or not low_values:
        return {
            "maximum_favorable_excursion": None,
            "maximum_adverse_excursion": None,
        }

    return {
        "maximum_favorable_excursion": max(0.0, float(max(high_values) / base - 1.0)),
        "maximum_adverse_excursion": min(0.0, float(min(low_values) / base - 1.0)),
    }


def _peak_exit_diagnostics(
    frames: dict[str, pd.DataFrame],
    symbol: str,
    entry_at: Any,
    exit_at: Any,
    entry_price: Any,
    exit_price: Any,
    action: str,
) -> dict[str, Any]:
    """Measure how much of the intraposition peak was captured at exit.

    Normal SELL orders execute at the exit-session open, so their held-period
    peak ends on the previous session. FINAL_SELL is bookkeeping at the final
    close, so the final session remains part of the held period.

    Post-exit windows start on the exit session for SELL and on the following
    session for FINAL_SELL. A horizon is reported only when the full number of
    sessions is available, avoiding partial-window bias near the backtest end.
    """

    empty: dict[str, Any] = {
        "peak_exit_diagnostics_schema_version": 1,
        "peak_price_while_held": None,
        "peak_timestamp_while_held": None,
        "exit_distance_from_peak_pct": None,
        "peak_capture_pct": None,
        "max_runup_pct": None,
        "days_from_peak_to_exit": None,
        "post_exit_peak_5d_pct": None,
        "post_exit_peak_10d_pct": None,
        "post_exit_peak_20d_pct": None,
    }
    frame = _frame_at(frames, symbol)
    start = _timestamp(entry_at)
    end = _timestamp(exit_at)
    base = _number(entry_price)
    execution = _number(exit_price)
    normalized_action = str(action or "").upper()
    if (
        frame is None
        or start is None
        or end is None
        or base in {None, 0.0}
        or end < start
        or normalized_action not in {"SELL", "FINAL_SELL"}
    ):
        return empty

    if execution is None:
        execution_column = "close" if normalized_action == "FINAL_SELL" else "open"
        try:
            execution = _number(frame.loc[end, execution_column])
        except (KeyError, TypeError):
            execution = None

    include_exit_session = normalized_action == "FINAL_SELL"
    holding_mask = (frame.index >= start) & (
        (frame.index <= end) if include_exit_session else (frame.index < end)
    )
    holding = frame.loc[holding_mask]
    high_column = holding.get("high")
    highs = (
        pd.to_numeric(high_column, errors="coerce").dropna()
        if high_column is not None
        else pd.Series(dtype=float)
    )

    peak_price = float(base)
    peak_at = start
    if not highs.empty:
        candidate_at = highs.idxmax()
        candidate_price = _number(highs.loc[candidate_at])
        if candidate_price is not None and candidate_price >= peak_price:
            peak_price = float(candidate_price)
            peak_at = _timestamp(candidate_at) or peak_at

    peak_gain = max(0.0, peak_price / float(base) - 1.0)
    exit_gain = (
        float(execution) / float(base) - 1.0
        if execution is not None
        else None
    )
    exit_distance = (
        max(0.0, (peak_price - float(execution)) / peak_price) * 100.0
        if execution is not None and peak_price > 0
        else None
    )
    peak_capture = (
        min(1.0, max(0.0, float(exit_gain)) / peak_gain) * 100.0
        if exit_gain is not None and peak_gain > 0
        else None
    )

    sessions_from_peak = int(
        ((frame.index > peak_at) & (frame.index <= end)).sum()
    )

    post_start_mask = (
        frame.index > end
        if normalized_action == "FINAL_SELL"
        else frame.index >= end
    )
    post = frame.loc[post_start_mask]

    def post_peak(horizon: int) -> float | None:
        if execution in {None, 0.0} or len(post) < horizon:
            return None
        window = post.head(horizon)
        window_high_column = window.get("high")
        if window_high_column is None:
            return None
        window_highs = pd.to_numeric(window_high_column, errors="coerce").dropna()
        if window_highs.empty:
            return None
        return max(0.0, float(window_highs.max()) / float(execution) - 1.0) * 100.0

    return {
        "peak_exit_diagnostics_schema_version": 1,
        "peak_price_while_held": peak_price,
        "peak_timestamp_while_held": peak_at.isoformat(),
        "exit_distance_from_peak_pct": exit_distance,
        "peak_capture_pct": peak_capture,
        "max_runup_pct": peak_gain * 100.0,
        "days_from_peak_to_exit": sessions_from_peak,
        "post_exit_peak_5d_pct": post_peak(5),
        "post_exit_peak_10d_pct": post_peak(10),
        "post_exit_peak_20d_pct": post_peak(20),
    }


def _next_exit(
    rows: list[dict[str, Any]],
    buy_index: int,
    symbol: str,
) -> dict[str, Any] | None:
    for row in rows[buy_index + 1 :]:
        if str(row.get("asset") or "") != symbol:
            continue
        if str(row.get("action") or "").upper() in {"SELL", "FINAL_SELL"}:
            return row
    return None


def enrich_trade_diagnostics(
    records: Iterable[dict[str, Any]],
    frames: dict[str, pd.DataFrame],
    symbols: Iterable[str],
) -> list[dict[str, Any]]:
    






    rows = [dict(row) for row in records]
    universe = [str(symbol) for symbol in symbols]

    for row in rows:
        if str(row.get("action") or "").upper() not in {"SELL", "FINAL_SELL"}:
            continue
        excursions = _position_excursions(
            frames,
            str(row.get("asset") or ""),
            row.get("entry_timestamp"),
            row.get("timestamp"),
            row.get("entry_price"),
        )
        row.update(excursions)
        row.update(
            _peak_exit_diagnostics(
                frames,
                str(row.get("asset") or ""),
                row.get("entry_timestamp"),
                row.get("timestamp"),
                row.get("entry_price"),
                row.get("execution_price"),
                str(row.get("action") or ""),
            )
        )
        realized_return = _number(row.get("position_return"))
        mfe = _number(excursions.get("maximum_favorable_excursion"))
        row["profit_capture_ratio"] = (
            max(0.0, realized_return) / mfe
            if realized_return is not None and mfe is not None and mfe > 0
            else None
        )

    for index, buy in enumerate(rows):
        if str(buy.get("action") or "").upper() != "BUY":
            continue
        rotation_id = str(buy.get("rotation_id") or "").strip()
        from_asset = str(buy.get("rotation_from_asset") or "").strip()
        to_asset = str(buy.get("rotation_to_asset") or buy.get("asset") or "").strip()
        if not rotation_id or not from_asset or not to_asset:
            continue

        exit_row = _next_exit(rows, index, to_asset)
        if exit_row is None:
            continue
        start_at = buy.get("timestamp")
        end_at = exit_row.get("timestamp")
        chosen_return = _open_to_open_return(frames, to_asset, start_at, end_at)
        previous_return = _open_to_open_return(frames, from_asset, start_at, end_at)

        alternative_returns: list[tuple[str, float]] = []
        for symbol in universe:
            value = _open_to_open_return(frames, symbol, start_at, end_at)
            if value is not None:
                alternative_returns.append((symbol, value))
        best_asset = None
        best_return = None
        if alternative_returns:
            best_asset, best_return = max(alternative_returns, key=lambda item: item[1])

        value_added = (
            chosen_return - previous_return
            if chosen_return is not None and previous_return is not None
            else None
        )
        opportunity_cost = (
            max(0.0, best_return - chosen_return)
            if best_return is not None and chosen_return is not None
            else None
        )

        buy.update(
            {
                "subsequent_position_return": _number(exit_row.get("position_return")),
                "chosen_market_return": chosen_return,
                "counterfactual_previous_asset_return": previous_return,
                "rotation_value_added": value_added,
                "rotation_regret": max(0.0, -value_added) if value_added is not None else None,
                "best_alternative_asset": best_asset,
                "best_alternative_return": best_return,
                "opportunity_cost": opportunity_cost,
                "maximum_favorable_excursion": _number(exit_row.get("maximum_favorable_excursion")),
                "maximum_adverse_excursion": _number(exit_row.get("maximum_adverse_excursion")),
                "profit_capture_ratio": _number(exit_row.get("profit_capture_ratio")),
                "peak_exit_diagnostics_schema_version": exit_row.get("peak_exit_diagnostics_schema_version"),
                "peak_price_while_held": _number(exit_row.get("peak_price_while_held")),
                "peak_timestamp_while_held": exit_row.get("peak_timestamp_while_held"),
                "exit_distance_from_peak_pct": _number(exit_row.get("exit_distance_from_peak_pct")),
                "peak_capture_pct": _number(exit_row.get("peak_capture_pct")),
                "max_runup_pct": _number(exit_row.get("max_runup_pct")),
                "days_from_peak_to_exit": _number(exit_row.get("days_from_peak_to_exit")),
                "post_exit_peak_5d_pct": _number(exit_row.get("post_exit_peak_5d_pct")),
                "post_exit_peak_10d_pct": _number(exit_row.get("post_exit_peak_10d_pct")),
                "post_exit_peak_20d_pct": _number(exit_row.get("post_exit_peak_20d_pct")),
                "subsequent_holding_days": _number(exit_row.get("holding_bars")),
            }
        )

    return rows
