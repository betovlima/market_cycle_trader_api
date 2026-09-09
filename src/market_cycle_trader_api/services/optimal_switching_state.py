from __future__ import annotations

from typing import Any

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
    }
    values.update({f"current_{key}": value for key, value in current.items()})
    values.update({f"target_{key}": value for key, value in target.items()})
    return values
