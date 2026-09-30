"""Research-only temporal pairwise ranker for Control candidate selection.

This is a NEW hypothesis after TinyTCN regression failed to learn useful
cross-sectional ordering. The network is trained only to order assets within
the same completed decision date. Absolute entry/exit thresholds and switch
margins remain anchored to the original LightGBM Control utility (after the
unchanged v10.8.44 prior-close liquidity overlay). Thus Deep Learning does not
invent a new utility scale.

No operational strategy registration, no TCC source edit and no live orders.
Historical OOS is exploratory because it informed this research direction.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from types import FunctionType
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..tcc_v106_reference import research_challengers as scientific
from ..tcc_v106_reference.capital_rotation import _utility_policy as frozen_utility_policy
from .control_deep_learning_tcn import FEATURES, WINDOW, TinyTCN, _feature_matrix
from .control_execution_feasibility import simulate_feasible_control
from .control_liquidity_policy import _CapitalAwareUtilityCache

MODE = "MCT_RESEARCH_DEEP_PAIRWISE_RANK_LIQUIDITY_V1"
TARGET = "forward_risk_adjusted_utility"
HORIZON = 60
TRAIN_DATE_STRIDE = 3
MIN_ASSETS_PER_DATE = 25
MAX_EPOCHS = 6
PATIENCE = 2
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
RANDOM_SEED = 20260930
Progress = Callable[[float, str], None]


@dataclass(frozen=True)
class FeatureScale:
    mean: np.ndarray
    std: np.ndarray


def _scale_from_dates(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    dates: pd.DatetimeIndex,
) -> FeatureScale:
    accepted = set(pd.DatetimeIndex(dates))
    blocks: list[np.ndarray] = []
    for symbol in symbols:
        frame = frames[symbol]
        mask = frame.index.isin(accepted)
        values = _feature_matrix(frame)[mask]
        if len(values):
            blocks.append(values)
    if not blocks:
        raise ValueError("No training features for deep rank scaler.")
    raw = np.vstack(blocks)
    if not np.isfinite(raw).all():
        raise ValueError("Non-finite training feature in rank scaler.")
    return FeatureScale(
        mean=raw.mean(axis=0),
        std=np.maximum(raw.std(axis=0), 1e-6),
    )


def _scaled_arrays(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    scale: FeatureScale,
) -> dict[str, np.ndarray]:
    output = {}
    for symbol in symbols:
        values = _feature_matrix(frames[symbol])
        output[symbol] = np.clip(
            (values - scale.mean) / scale.std, -8.0, 8.0,
        ).astype(np.float32)
    return output


def _date_groups(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    dates: pd.DatetimeIndex,
    *,
    maturity_before: pd.Timestamp,
    date_stride: int,
) -> list[tuple[pd.Timestamp, list[tuple[str, int, float]]]]:
    """Return same-date ranking groups with completely matured 60-day labels."""
    if date_stride < 1:
        raise ValueError("date_stride must be positive.")
    accepted = list(pd.DatetimeIndex(dates))
    cutoff = pd.Timestamp(maturity_before)
    groups = []
    for date_index, date in enumerate(accepted):
        if date_index % date_stride:
            continue
        date = pd.Timestamp(date)
        members: list[tuple[str, int, float]] = []
        for symbol in symbols:
            frame = frames[symbol]
            locs = frame.index.get_indexer([date])
            if len(locs) != 1 or int(locs[0]) < WINDOW - 1:
                continue
            loc = int(locs[0])
            if loc + HORIZON >= len(frame):
                continue
            mature_date = pd.Timestamp(frame.index[loc + HORIZON])
            if mature_date >= cutoff:
                continue
            window = _feature_matrix(frame)[loc-WINDOW+1:loc+1]
            value = float(frame[TARGET].iloc[loc])
            if not np.isfinite(window).all() or not math.isfinite(value):
                continue
            members.append((symbol, loc, value))
        if len(members) >= MIN_ASSETS_PER_DATE:
            groups.append((date, members))
    if not groups:
        raise ValueError("No strict label-mature cross-sectional ranking dates.")
    return groups


def _group_inputs(
    arrays: dict[str, np.ndarray],
    members: list[tuple[str, int, float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    x = np.stack([
        arrays[symbol][loc-WINDOW+1:loc+1].T
        for symbol, loc, _ in members
    ]).astype(np.float32)
    y = np.asarray([target for _, _, target in members], dtype=np.float32)
    return torch.from_numpy(x), torch.from_numpy(y)


def _pairwise_loss(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Unweighted all-pairs logistic ordering loss (ties ignored)."""
    if scores.ndim != 1 or targets.ndim != 1 or len(scores) != len(targets):
        raise ValueError("Pairwise ranking expects same-length 1D score/target.")
    target_delta = targets[:, None] - targets[None, :]
    score_delta = scores[:, None] - scores[None, :]
    upper = torch.triu(
        torch.ones_like(target_delta, dtype=torch.bool), diagonal=1,
    )
    sign = torch.sign(target_delta)
    mask = upper & (sign != 0)
    if not bool(mask.any()):
        return scores.sum() * 0.0
    return nn.functional.softplus(
        -sign[mask] * score_delta[mask],
    ).mean()


def _ranking_accuracy(scores: torch.Tensor, targets: torch.Tensor) -> tuple[int, int]:
    delta_y = targets[:, None] - targets[None, :]
    delta_s = scores[:, None] - scores[None, :]
    mask = torch.triu(torch.ones_like(delta_y, dtype=torch.bool), 1) & (delta_y != 0)
    total = int(mask.sum().item())
    if total == 0:
        return 0, 0
    correct = int((torch.sign(delta_y[mask]) == torch.sign(delta_s[mask])).sum().item())
    return correct, total


def _epoch(
    model: TinyTCN,
    arrays: dict[str, np.ndarray],
    groups: list[tuple[pd.Timestamp, list[tuple[str, int, float]]]],
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    model.train(optimizer is not None)
    loss_sum = 0.0
    group_count = 0
    correct = 0
    total_pairs = 0
    for _, members in groups:
        x, y = _group_inputs(arrays, members)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            scores = model(x)
            loss = _pairwise_loss(scores, y)
            if optimizer is not None:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        loss_sum += float(loss.detach())
        group_count += 1
        c, t = _ranking_accuracy(scores.detach(), y)
        correct += c
        total_pairs += t
    return loss_sum / max(group_count, 1), correct / max(total_pairs, 1)


def _fit_ranker(
    arrays: dict[str, np.ndarray],
    training: list[tuple[pd.Timestamp, list[tuple[str, int, float]]]],
    *,
    seed: int,
    epochs: int,
    validation: list[tuple[pd.Timestamp, list[tuple[str, int, float]]]] | None,
) -> tuple[TinyTCN, int, dict[str, float | int | None]]:
    if not training or epochs < 1:
        raise ValueError("Deep rank fit requires data and positive epochs.")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = TinyTCN().to("cpu")
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        )
        best_epoch = epochs
        best_loss = float("inf")
        best_accuracy = 0.0
        stale = 0
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = _epoch(model, arrays, training, optimizer)
            if validation is None:
                continue
            valid_loss, valid_acc = _epoch(model, arrays, validation, None)
            if not math.isfinite(valid_loss):
                raise ValueError("Deep rank validation loss is non-finite.")
            if valid_loss < best_loss - 1e-6:
                best_loss = valid_loss
                best_accuracy = valid_acc
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
                if stale >= PATIENCE:
                    break
        return model, best_epoch, {
            "best_validation_pairwise_loss": (
                best_loss if validation is not None else None
            ),
            "best_validation_pairwise_accuracy": (
                best_accuracy if validation is not None else None
            ),
            "last_train_pairwise_loss": train_loss,
            "last_train_pairwise_accuracy": train_acc,
        }


def _predict_rank_scores(
    model: TinyTCN,
    arrays: dict[str, np.ndarray],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    dates: list[pd.Timestamp],
) -> tuple[dict[pd.Timestamp, np.ndarray], list[dict[str, Any]]]:
    cache = {}
    rows = []
    model.eval()
    with torch.no_grad():
        for date in dates:
            windows = []
            positions = []
            for index, symbol in enumerate(symbols, start=1):
                frame = frames[symbol]
                locs = frame.index.get_indexer([date])
                if len(locs) != 1 or int(locs[0]) < WINDOW - 1:
                    continue
                loc = int(locs[0])
                window = arrays[symbol][loc-WINDOW+1:loc+1]
                if not np.isfinite(window).all():
                    continue
                windows.append(window.T)
                positions.append((index, symbol, loc))
            if len(windows) != len(symbols):
                raise ValueError(
                    f"Rank OOS date {date} lacks complete 55-asset causal windows."
                )
            tensor = torch.from_numpy(np.stack(windows).astype(np.float32))
            predictions = model(tensor).cpu().numpy().astype(float)
            values = np.full(len(symbols)+1, -np.inf, dtype=np.float64)
            values[0] = -np.inf
            for score, (position, symbol, loc) in zip(predictions, positions):
                if not math.isfinite(float(score)):
                    raise ValueError("Non-finite deep ranking OOS score.")
                values[position] = float(score)
                mature = loc + HORIZON < len(frames[symbol])
                realized = (
                    float(frames[symbol][TARGET].iloc[loc])
                    if mature and np.isfinite(frames[symbol][TARGET].iloc[loc])
                    else None
                )
                rows.append({
                    "decision_date": date,
                    "asset": symbol,
                    "rank_score": float(score),
                    "realized_utility_if_mature": realized,
                    "label_mature_at_snapshot": realized is not None,
                })
            cache[pd.Timestamp(date)] = values
    return cache, rows


def _deep_rank_policy(
    *,
    original_utility_cache: dict[pd.Timestamp, np.ndarray],
    rank_cache: dict[pd.Timestamp, np.ndarray],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    config: Any,
    switch_margin: float,
    account: dict[str, Any],
    diagnostics: dict[pd.Timestamp, dict[str, Any]] | None,
    fold_id: int,
) -> Callable[[pd.Timestamp, int, int], tuple[int, float]]:
    """Deep network orders candidates; LightGBM+liquidity owns absolute scale."""
    liquidity_cache = _CapitalAwareUtilityCache(
        original_utility_cache, frames, symbols, account,
    )

    def policy(timestamp: pd.Timestamp, current_position: int,
               holding_days: int) -> tuple[int, float]:
        key = pd.Timestamp(timestamp)
        utilities = np.asarray(liquidity_cache.get(key), dtype=float)
        ranks = np.asarray(rank_cache.get(key), dtype=float)
        if len(utilities) != len(symbols)+1 or len(ranks) != len(symbols)+1:
            raise ValueError("Deep rank / LightGBM utility cache shape mismatch.")
        current_value = (
            float(utilities[current_position])
            if 0 <= current_position < len(utilities)
            and np.isfinite(utilities[current_position])
            else 0.0
        )
        minimum = float(config.rotation_cash_threshold)
        entry_threshold = minimum + float(config.rotation_min_expected_edge)
        required = max(
            float(config.rotation_switch_margin), float(switch_margin),
        )
        # Deep ranker is allowed to choose only candidates that the original
        # LightGBM+liquidity absolute utility regards as above CASH.
        candidates = [
            position for position in range(1, len(utilities))
            if np.isfinite(utilities[position])
            and np.isfinite(ranks[position])
            and float(utilities[position]) > minimum
        ]
        candidates.sort(
            key=lambda p: (-float(ranks[p]), symbols[p-1]),
        )
        best = candidates[0] if candidates else 0
        best_value = float(utilities[best]) if best else 0.0
        reason = ""
        target = current_position
        if (
            current_position > 0
            and holding_days < int(config.rotation_min_holding_days)
        ):
            target, reason = current_position, "MIN_HOLD_GUARD"
        elif best == 0:
            target, reason = 0, "NO_LIGHTGBM_POSITIVE_CANDIDATE"
        elif current_position == 0:
            if best_value >= entry_threshold:
                target, reason = best, "DEEP_RANK_ENTER"
            else:
                target, reason = 0, "MIN_EXPECTED_EDGE_GUARD"
        elif best == current_position:
            target, reason = current_position, "DEEP_RANK_HOLD_TOP"
        elif best_value >= current_value + required:
            target, reason = best, "DEEP_RANK_ROTATE"
        else:
            target, reason = current_position, "LIGHTGBM_SWITCH_MARGIN_GUARD"
        final_score = (
            float(utilities[target])
            if target > 0 and np.isfinite(utilities[target]) else 0.0
        )
        if diagnostics is not None:
            diagnostics[key] = {
                "decision_fold_id": int(fold_id),
                "deep_rank_best_asset": symbols[best-1] if best else "CASH",
                "deep_rank_best_score": float(ranks[best]) if best else None,
                "deep_rank_best_lightgbm_liquidity_utility": best_value,
                "current_asset": symbols[current_position-1] if current_position else "CASH",
                "current_lightgbm_liquidity_utility": current_value,
                "final_action_asset": symbols[target-1] if target else "CASH",
                "final_action_score": final_score,
                "decision_reason": reason,
                "effective_switch_margin": required,
                "finite_positive_lightgbm_candidate_count": len(candidates),
            }
        return target, final_score

    return policy


def run_deep_rank_hybrid(
    bars: dict[str, pd.DataFrame],
    config: Any,
    fee_calculator: Callable,
    slippage: Callable,
    *,
    progress_callback: Callable[[float, str, int], None] | None = None,
) -> tuple[Any, pd.DataFrame, list[dict[str, Any]], dict[pd.Timestamp, dict[str, Any]]]:
    """Train pairwise ranker and reuse original LightGBM for absolute utility.

    LightGBM calibration/fits are unchanged. Only FINAL OOS policy ordering is
    overridden when the final utility cache exists; calibration margin search
    still uses the frozen Control policy.
    """
    (
        frames, common_dates, symbols, folds, all_decision_dates,
        decision_to_fold, decision_metadata,
    ) = scientific._build_execution_context(bars, config)
    if len(symbols) != 55 or len(folds) != 3:
        raise ValueError("Deep ranking expects original 55 assets / 3 folds.")
    rank_cache: dict[pd.Timestamp, np.ndarray] = {}
    fold_reports = []
    score_rows = []
    for fold in folds:
        fid = int(fold["fold_id"])
        train_dates = common_dates[:int(fold["train_end_index"])]
        calibration_dates = common_dates[
            int(fold["calibration_start_index"]):int(fold["calibration_end_index"])
        ]
        final_dates = common_dates[:int(fold["final_fit_end_index"])]
        train_groups = _date_groups(
            frames, symbols, train_dates,
            maturity_before=fold["calibration_start"],
            date_stride=TRAIN_DATE_STRIDE,
        )
        calibration_groups = _date_groups(
            frames, symbols, calibration_dates,
            maturity_before=fold["test_start"],
            date_stride=1,
        )
        scale = _scale_from_dates(frames, symbols, train_dates)
        arrays = _scaled_arrays(frames, symbols, scale)
        if progress_callback:
            progress_callback(
                5.0 + 15.0*(fid-1),
                f"Deep ranking fold {fid}: calibration training", 0,
            )
        _, epochs, validation = _fit_ranker(
            arrays, train_groups, seed=RANDOM_SEED+fid,
            epochs=MAX_EPOCHS, validation=calibration_groups,
        )
        final_groups = _date_groups(
            frames, symbols, final_dates,
            maturity_before=fold["test_start"],
            date_stride=TRAIN_DATE_STRIDE,
        )
        final_scale = _scale_from_dates(frames, symbols, final_dates)
        final_arrays = _scaled_arrays(frames, symbols, final_scale)
        final_model, _, train_diagnostics = _fit_ranker(
            final_arrays, final_groups, seed=RANDOM_SEED+100+fid,
            epochs=epochs, validation=None,
        )
        dates = [
            pd.Timestamp(x) for x in fold["decision_dates"][:-1]
        ]
        fold_cache, rows = _predict_rank_scores(
            final_model, final_arrays, frames, symbols, dates,
        )
        if set(fold_cache).intersection(rank_cache):
            raise ValueError("Overlapping deep ranking fold OOS dates.")
        rank_cache.update(fold_cache)
        for row in rows:
            row["fold_id"] = fid
        score_rows.extend(rows)
        fold_reports.append({
            "fold_id": fid,
            "train_ranking_dates": len(train_groups),
            "calibration_ranking_dates": len(calibration_groups),
            "final_train_ranking_dates": len(final_groups),
            "chosen_epochs_calibration_only": epochs,
            **validation,
            **{f"final_{k}": v for k, v in train_diagnostics.items()},
            "test_start": pd.Timestamp(fold["test_start"]).isoformat(),
            "test_end": pd.Timestamp(fold["test_end"]).isoformat(),
        })
    expected_dates = {
        pd.Timestamp(x) for fold in folds for x in fold["decision_dates"][:-1]
    }
    if set(rank_cache) != expected_dates:
        raise ValueError("Deep rank cache differs from exact OOS decision calendar.")

    account: dict[str, Any] = {"enabled": True, "audit": {}}
    policy_audit: dict[pd.Timestamp, dict[str, Any]] = {}
    original_runner = scientific._run_lightgbm
    original_policy = original_runner.__globals__.get("_utility_policy")
    original_simulator = original_runner.__globals__.get("_simulate_exact")
    if original_policy is not frozen_utility_policy or original_simulator is not scientific._simulate_exact:
        raise RuntimeError("Frozen LightGBM runner bindings changed unexpectedly.")

    def policy_wrapper(models, panel, labels, settings, switch_margin, **kwargs):
        utility_cache = kwargs.get("utility_cache")
        fold_id = kwargs.get("fold_id")
        diagnostics = kwargs.get("decision_diagnostics")
        if utility_cache is None or fold_id is None:
            return original_policy(
                models, panel, labels, settings, switch_margin, **kwargs,
            )
        fid = int(fold_id)
        fold_dates = {
            pd.Timestamp(x) for x in folds[fid-1]["decision_dates"][:-1]
        }
        if not fold_dates.issubset(rank_cache):
            return original_policy(
                models, panel, labels, settings, switch_margin, **kwargs,
            )
        return _deep_rank_policy(
            original_utility_cache=utility_cache,
            rank_cache={date: rank_cache[date] for date in fold_dates},
            frames=panel, symbols=labels, config=settings,
            switch_margin=switch_margin, account=account,
            diagnostics=diagnostics if diagnostics is not None else policy_audit,
            fold_id=fid,
        )

    captured = {}
    def simulator_wrapper(*args, **kwargs):
        if captured:
            raise ValueError("Deep rank experiment expected a single OOS replay.")
        def prepare(date, position, holding, cash, shares, equity):
            account.update({
                "decision_timestamp": pd.Timestamp(date),
                "position": int(position), "holding_days": int(holding),
                "cash": float(cash), "shares": int(shares), "equity": float(equity),
            })
        run = simulate_feasible_control(
            *args, **{**kwargs, "decision_prepare": prepare},
        )
        captured["run"] = run
        return run

    isolated_globals = dict(original_runner.__globals__)
    isolated_globals["_utility_policy"] = policy_wrapper
    isolated_globals["_simulate_exact"] = simulator_wrapper
    isolated = FunctionType(
        original_runner.__code__, isolated_globals, original_runner.__name__,
        original_runner.__defaults__, original_runner.__closure__,
    )
    isolated.__kwdefaults__ = dict(original_runner.__kwdefaults__ or {})
    result = isolated(
        bars, config, fee_calculator, slippage,
        progress_callback=progress_callback,
        trade_callback=None, progress_detail_callback=None,
        technical_log_callback=None,
    )
    if len(result) != 1 or result[0] is not captured.get("run"):
        raise ValueError("Deep rank hybrid did not produce exactly one OOS run.")
    if (
        original_runner.__globals__.get("_utility_policy") is not original_policy
        or original_runner.__globals__.get("_simulate_exact") is not original_simulator
    ):
        raise RuntimeError("Frozen TCC module was mutated by rank experiment.")
    return result[0], pd.DataFrame(score_rows), fold_reports, dict(account["audit"])
