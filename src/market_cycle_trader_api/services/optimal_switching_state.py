from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from ..engine.capital_rotation import _proportional_switch_cost

ASSET_STATE_FEATURES = (
    "return_5",
    "return_20",
    "return_60",
    "vol_20",
    "ema_5_vs_20",
    "ema_slope_20_5",
    "atr_pct_14",
    "trend_efficiency_20",
    "channel_position_20",
)

POSITION_HISTORY_FEATURES = (
    "holding_days",
    "position_return_since_entry",
    "position_peak_return",
    "position_drawdown_from_peak",
    "position_mfe_so_far",
    "position_mae_so_far",
    "score_change_from_entry",
    "days_current_not_top1",
    "consecutive_days_current_not_top1",
    "fraction_days_current_not_top1",
)

ACTION_STATE_FEATURES = (
    "current_is_cash",
    "target_is_cash",
    "action_hold",
    "action_enter",
    "action_switch",
    "action_exit",
    "switch_cost",
    "current_utility",
    "target_utility",
    "utility_gap",
    "current_rank_fraction",
    "target_rank_fraction",
    "universe_score_mean",
    "universe_score_std",
    "universe_best_score",
    "universe_positive_fraction",
    "breadth_return_20_positive",
    "median_return_20",
    "median_vol_20",
    *POSITION_HISTORY_FEATURES,
    *(f"current_{name}" for name in ASSET_STATE_FEATURES),
    *(f"target_{name}" for name in ASSET_STATE_FEATURES),
)


def finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(number) if np.isfinite(number) else float(default)


def ranked_positions(utilities: np.ndarray) -> list[int]:
    rows = [
        (position, float(utilities[position]))
        for position in range(1, len(utilities))
        if np.isfinite(utilities[position])
    ]
    rows.sort(key=lambda item: (-item[1], item[0]))
    return [position for position, _ in rows]


def candidate_targets(
    current_position: int,
    utilities: np.ndarray,
    top_k: int,
) -> list[int]:
    values = [0]
    if current_position > 0 and np.isfinite(utilities[current_position]):
        values.append(int(current_position))
    values.extend(ranked_positions(utilities)[:top_k])
    return list(dict.fromkeys(values))


def next_holding_days(
    current_position: int,
    target_position: int,
    holding_days: int,
) -> int:
    if target_position <= 0:
        return 0
    if current_position > 0 and target_position == current_position:
        return max(1, int(holding_days) + 1)
    return 1


def _row(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    timestamp: pd.Timestamp,
    position: int,
) -> dict[str, float]:
    if position <= 0:
        return {name: 0.0 for name in ASSET_STATE_FEATURES}
    frame = frames.get(symbols[position - 1])
    if frame is None or timestamp not in frame.index:
        return {name: 0.0 for name in ASSET_STATE_FEATURES}
    row = frame.loc[timestamp]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]
    return {name: finite(row.get(name)) for name in ASSET_STATE_FEATURES}


def market_context(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    timestamp: pd.Timestamp,
    utilities: np.ndarray,
) -> dict[str, Any]:
    positions = [
        position
        for position in range(1, len(utilities))
        if np.isfinite(utilities[position])
    ]
    scores = np.asarray([float(utilities[p]) for p in positions], dtype=float)
    ranked = ranked_positions(utilities)
    ranks = {position: rank for rank, position in enumerate(ranked, start=1)}

    return_20: list[float] = []
    vol_20: list[float] = []
    for position in positions:
        frame = frames.get(symbols[position - 1])
        if frame is None or timestamp not in frame.index:
            continue
        row = frame.loc[timestamp]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        r20 = finite(row.get("return_20"), float("nan"))
        v20 = finite(row.get("vol_20"), float("nan"))
        if np.isfinite(r20):
            return_20.append(r20)
        if np.isfinite(v20):
            vol_20.append(v20)

    return {
        "ranks": ranks,
        "size": max(1, len(positions)),
        "mean": float(np.mean(scores)) if len(scores) else 0.0,
        "std": float(np.std(scores)) if len(scores) else 0.0,
        "best": float(np.max(scores)) if len(scores) else 0.0,
        "positive_fraction": float(np.mean(scores > 0.0)) if len(scores) else 0.0,
        "breadth20": (
            float(np.mean(np.asarray(return_20) > 0.0)) if return_20 else 0.0
        ),
        "median_return20": float(np.median(return_20)) if return_20 else 0.0,
        "median_vol20": float(np.median(vol_20)) if vol_20 else 0.0,
    }


def empty_position_history() -> dict[str, float]:
    return {name: 0.0 for name in POSITION_HISTORY_FEATURES}


def position_history_features(
    *,
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    timestamp: pd.Timestamp,
    current_position: int,
    holding_days: int,
    utilities_for_timestamp: Callable[[pd.Timestamp], np.ndarray],
) -> dict[str, float]:
    if current_position <= 0 or holding_days <= 0:
        return empty_position_history()

    frame = frames.get(symbols[current_position - 1])
    if frame is None or frame.empty:
        return empty_position_history()

    timestamp = pd.Timestamp(timestamp)
    index = pd.DatetimeIndex(frame.index)
    try:
        current_index = int(index.get_loc(timestamp))
    except (KeyError, TypeError):
        return empty_position_history()

    span = max(1, int(holding_days))
    entry_index = max(0, current_index - span + 1)
    entry_timestamp = pd.Timestamp(index[entry_index])
    history_index = index[entry_index : current_index + 1]
    history = frame.loc[history_index]
    if history.empty:
        return empty_position_history()

    entry_price = finite(history.iloc[0].get("open"), float("nan"))
    current_close = finite(history.iloc[-1].get("close"), float("nan"))
    peak_high = finite(pd.to_numeric(history["high"], errors="coerce").max(), float("nan"))
    low_price = finite(pd.to_numeric(history["low"], errors="coerce").min(), float("nan"))

    if not (np.isfinite(entry_price) and entry_price > 0.0):
        return empty_position_history()

    return_since_entry = (
        current_close / entry_price - 1.0
        if np.isfinite(current_close) and current_close > 0.0
        else 0.0
    )
    peak_return = (
        peak_high / entry_price - 1.0
        if np.isfinite(peak_high) and peak_high > 0.0
        else 0.0
    )
    drawdown_from_peak = (
        current_close / peak_high - 1.0
        if (
            np.isfinite(current_close)
            and current_close > 0.0
            and np.isfinite(peak_high)
            and peak_high > 0.0
        )
        else 0.0
    )
    mae = (
        low_price / entry_price - 1.0
        if np.isfinite(low_price) and low_price > 0.0
        else 0.0
    )

    current_utilities = utilities_for_timestamp(timestamp)
    current_score = (
        float(current_utilities[current_position])
        if (
            current_position < len(current_utilities)
            and np.isfinite(current_utilities[current_position])
        )
        else 0.0
    )

    entry_decision_index = max(0, entry_index - 1)
    entry_decision_timestamp = pd.Timestamp(index[entry_decision_index])
    entry_utilities = utilities_for_timestamp(entry_decision_timestamp)
    entry_score = (
        float(entry_utilities[current_position])
        if (
            current_position < len(entry_utilities)
            and np.isfinite(entry_utilities[current_position])
        )
        else current_score
    )

    not_top1 = 0
    consecutive_not_top1 = 0
    for history_timestamp in history_index:
        utilities = utilities_for_timestamp(pd.Timestamp(history_timestamp))
        ranked = ranked_positions(utilities)
        is_top1 = bool(ranked and ranked[0] == current_position)
        if is_top1:
            consecutive_not_top1 = 0
        else:
            not_top1 += 1
            consecutive_not_top1 += 1

    effective_holding = int(len(history_index))
    return {
        "holding_days": float(effective_holding),
        "position_return_since_entry": float(return_since_entry),
        "position_peak_return": float(peak_return),
        "position_drawdown_from_peak": float(drawdown_from_peak),
        "position_mfe_so_far": float(peak_return),
        "position_mae_so_far": float(mae),
        "score_change_from_entry": float(current_score - entry_score),
        "days_current_not_top1": float(not_top1),
        "consecutive_days_current_not_top1": float(consecutive_not_top1),
        "fraction_days_current_not_top1": float(
            not_top1 / max(1, effective_holding)
        ),
    }


def action_features(
    *,
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    timestamp: pd.Timestamp,
    utilities: np.ndarray,
    context: dict[str, Any],
    current_position: int,
    target_position: int,
    config: Any,
    position_history: dict[str, float] | None = None,
) -> dict[str, float]:
    current_utility = (
        float(utilities[current_position])
        if current_position > 0 and np.isfinite(utilities[current_position])
        else 0.0
    )
    target_utility = (
        float(utilities[target_position])
        if target_position > 0 and np.isfinite(utilities[target_position])
        else 0.0
    )
    size = float(context["size"])
    ranks = context["ranks"]
    current_rank = float(ranks.get(current_position, size))
    target_rank = float(ranks.get(target_position, size))
    current = _row(frames, symbols, timestamp, current_position)
    target = _row(frames, symbols, timestamp, target_position)
    history = empty_position_history()
    if position_history:
        history.update(
            {
                name: finite(position_history.get(name))
                for name in POSITION_HISTORY_FEATURES
            }
        )

    values = {
        "current_is_cash": float(current_position == 0),
        "target_is_cash": float(target_position == 0),
        "action_hold": float(current_position > 0 and current_position == target_position),
        "action_enter": float(current_position == 0 and target_position > 0),
        "action_switch": float(
            current_position > 0
            and target_position > 0
            and current_position != target_position
        ),
        "action_exit": float(current_position > 0 and target_position == 0),
        "switch_cost": float(
            _proportional_switch_cost(config, current_position, target_position)
        ),
        "current_utility": current_utility,
        "target_utility": target_utility,
        "utility_gap": target_utility - current_utility,
        "current_rank_fraction": current_rank / size,
        "target_rank_fraction": target_rank / size,
        "universe_score_mean": float(context["mean"]),
        "universe_score_std": float(context["std"]),
        "universe_best_score": float(context["best"]),
        "universe_positive_fraction": float(context["positive_fraction"]),
        "breadth_return_20_positive": float(context["breadth20"]),
        "median_return_20": float(context["median_return20"]),
        "median_vol_20": float(context["median_vol20"]),
        **history,
    }
    values.update({f"current_{key}": value for key, value in current.items()})
    values.update({f"target_{key}": value for key, value in target.items()})
    return values
