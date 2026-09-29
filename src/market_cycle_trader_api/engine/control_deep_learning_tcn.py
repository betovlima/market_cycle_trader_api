"""Research-only causal TCN utility challenger for the frozen Control snapshot.

This module never modifies the vendored TCC, live strategies or training files.
A pooled, small CPU temporal convolutional network learns the ORIGINAL
forward risk-adjusted utility from strictly historical feature windows.
Calibration chooses epochs before each fold. Final fit and normalization use
only labels mature before test start. Actual equity feeds the unchanged
v10.8.44 liquidity overlay and v10.8.42 fill/accounting engine.

Historical OOS is EXPLORATORY because this period informed study design.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..tcc_v106_reference import research_challengers as scientific
from ..tcc_v106_reference.capital_rotation import _utility_policy
from .control_execution_feasibility import simulate_feasible_control
from .control_liquidity_policy import _CapitalAwareUtilityCache

MODE = "MCT_RESEARCH_TCN_LIQUIDITY_AWARE_V1"
FEATURES = (
    "return_1", "return_5", "return_20", "return_60",
    "vol_20", "vol_60", "ema_distance_20", "ema_distance_50",
    "rsi_14", "atr_pct_14", "channel_position_50", "volume_zscore_20",
)
TARGET = "forward_risk_adjusted_utility"
WINDOW = 40
HIDDEN = 16
DILATIONS = (1, 2, 4, 8, 16)
EPOCHS = 8
PATIENCE = 2
BATCH_SIZE = 256
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
FIXED_SWITCH_MARGIN = 0.0025
RANDOM_SEED = 42
MAX_PREDICT_ABS = 3.0
Progress = Callable[[float, str], None]


class CausalBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, dilation: int):
        super().__init__()
        padding = 2 * dilation
        self.conv = nn.Conv1d(channels_in, channels_out, 3, padding=padding,
                              dilation=dilation)
        self.pad = padding
        self.skip = (nn.Identity() if channels_in == channels_out
                     else nn.Conv1d(channels_in, channels_out, 1))
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(.10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Remove right-side future padding; final output sees only <= current t.
        output = self.conv(x)[:, :, :-self.pad]
        return self.activation(self.dropout(output) + self.skip(x))


class TinyTCN(nn.Module):
    def __init__(self, feature_count: int = len(FEATURES)):
        super().__init__()
        blocks = []
        width = feature_count
        for dilation in DILATIONS:
            blocks.append(CausalBlock(width, HIDDEN, dilation))
            width = HIDDEN
        self.encoder = nn.Sequential(*blocks)
        self.head = nn.Linear(HIDDEN, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != len(FEATURES):
            raise ValueError("Expected [batch, 12 predeclared features, history].")
        return self.head(self.encoder(x)[:, :, -1]).squeeze(-1)


@dataclass(frozen=True)
class FoldScale:
    mean: np.ndarray
    std: np.ndarray
    y_mean: float
    y_std: float


class WindowDataset(Dataset):
    """Reference views only; no windows beginning after the decision timestamp."""
    def __init__(self, arrays: dict[str, np.ndarray],
                 targets: dict[str, np.ndarray],
                 rows: list[tuple[str, int]]):
        self.arrays = arrays
        self.targets = targets
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        symbol, loc = self.rows[index]
        features = self.arrays[symbol][loc - WINDOW + 1:loc + 1]
        return (
            torch.from_numpy(features.T.copy()),
            torch.tensor(self.targets[symbol][loc], dtype=torch.float32),
        )


def _feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    if any(name not in frame for name in FEATURES):
        raise ValueError("Missing causally constructed feature column.")
    matrix = frame.loc[:, FEATURES].to_numpy(dtype=np.float64)
    return matrix


def _dataset_rows(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    dates: pd.DatetimeIndex,
    *,
    maturity_before: pd.Timestamp,
    horizon: int,
) -> list[tuple[str, int]]:
    """A sample is trainable only if its COMPLETE horizon ends before cutoff.

    Labels are observed at t+h; the policy's next-open entry and longest
    60-session target both mature BEFORE calibration/test boundaries.
    """
    accepted = set(pd.DatetimeIndex(dates))
    cutoff = pd.Timestamp(maturity_before)
    rows: list[tuple[str, int]] = []
    for symbol in symbols:
        frame = frames[symbol]
        if TARGET not in frame:
            raise ValueError("Scientific target absent from prepared frame.")
        x = _feature_matrix(frame)
        y = pd.to_numeric(frame[TARGET], errors="coerce").to_numpy(dtype=float)
        for loc in range(WINDOW - 1, len(frame) - horizon):
            date = pd.Timestamp(frame.index[loc])
            if date not in accepted:
                continue
            mature_date = pd.Timestamp(frame.index[loc + horizon])
            if mature_date >= cutoff:
                continue
            if not np.isfinite(y[loc]) or not np.isfinite(x[loc-WINDOW+1:loc+1]).all():
                continue
            rows.append((symbol, loc))
    if not rows:
        raise ValueError("No label-mature windows in strict chronological segment.")
    return rows


def _scaling(
    frames: dict[str, pd.DataFrame], symbols: list[str],
    dates: pd.DatetimeIndex, rows: list[tuple[str, int]],
) -> FoldScale:
    allowed = set(pd.DatetimeIndex(dates))
    features = []
    for symbol in symbols:
        frame = frames[symbol]
        mask = frame.index.isin(allowed)
        part = _feature_matrix(frame)[mask]
        if len(part):
            features.append(part)
    if not features:
        raise ValueError("No training-only feature observations for scaler.")
    raw = np.vstack(features)
    if not np.isfinite(raw).all():
        raise ValueError("Training features contain non-finite values.")
    mean = raw.mean(axis=0)
    std = np.maximum(raw.std(axis=0), 1e-6)
    y = np.asarray([
        float(frames[symbol][TARGET].iloc[loc]) for symbol, loc in rows
    ], dtype=np.float64)
    y_mean = float(np.mean(y))
    y_std = max(float(np.std(y)), 1e-6)
    return FoldScale(mean, std, y_mean, y_std)


def _arrays(frames: dict[str, pd.DataFrame], symbols: list[str],
            scale: FoldScale) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    x_arrays = {}
    y_arrays = {}
    for symbol in symbols:
        x = _feature_matrix(frames[symbol])
        x_arrays[symbol] = np.clip(
            (x - scale.mean) / scale.std, -8.0, 8.0,
        ).astype(np.float32)
        y = pd.to_numeric(
            frames[symbol][TARGET], errors="coerce",
        ).to_numpy(dtype=np.float64)
        y_arrays[symbol] = np.clip(
            (y - scale.y_mean) / scale.y_std, -5.0, 5.0,
        ).astype(np.float32)
    return x_arrays, y_arrays


def _train_epoch(model: TinyTCN, loader: DataLoader,
                 optimizer: torch.optim.Optimizer | None) -> float:
    model.train(optimizer is not None)
    losses, count = 0.0, 0
    for x, y in loader:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            estimate = model(x)
            loss = nn.functional.smooth_l1_loss(estimate, y)
            if optimizer is not None:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        n = len(y)
        losses += float(loss.detach()) * n
        count += n
    return losses / max(count, 1)


def _fit(
    arrays: dict[str, np.ndarray], targets: dict[str, np.ndarray],
    rows: list[tuple[str, int]], seed: int, epochs: int,
    valid: list[tuple[str, int]] | None = None,
) -> tuple[TinyTCN, int, float | None]:
    if epochs < 1 or not rows:
        raise ValueError("Expected nonempty train segment and positive epochs.")
    if valid is not None and not valid:
        raise ValueError("Validation set cannot be empty.")
    # fork_rng restores process-wide torch RNG; explicit CPU-only research.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = TinyTCN().to("cpu")
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        )
        generator = torch.Generator(device="cpu").manual_seed(seed)
        train_loader = DataLoader(
            WindowDataset(arrays, targets, rows), batch_size=BATCH_SIZE,
            shuffle=True, generator=generator, num_workers=0,
        )
        validation_loader = (
            DataLoader(WindowDataset(arrays, targets, valid),
                       batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
            if valid is not None else None
        )
        best_loss = float("inf")
        best_epoch = epochs
        no_improvement = 0
        for epoch in range(1, epochs + 1):
            _train_epoch(model, train_loader, optimizer)
            if validation_loader is None:
                continue
            validation_loss = _train_epoch(model, validation_loader, None)
            if not math.isfinite(validation_loss):
                raise ValueError("Non-finite validation loss.")
            if validation_loss < best_loss - 1e-6:
                best_loss = validation_loss
                best_epoch = epoch
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= PATIENCE:
                    break
        return model, best_epoch, (
            best_loss if validation_loader is not None else None
        )


def _predict(
    model: TinyTCN, arrays: dict[str, np.ndarray],
    rows: list[tuple[str, int]], scale: FoldScale,
) -> dict[tuple[str, int], float]:
    dataset = WindowDataset(
        arrays, {symbol: np.zeros(len(a), dtype=np.float32)
                 for symbol, a in arrays.items()}, rows,
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=0)
    preds = []
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            preds.extend(model(x).cpu().numpy().tolist())
    if len(preds) != len(rows):
        raise AssertionError("TCN predicted a different number of samples.")
    output = {}
    for row, prediction in zip(rows, preds):
        score = float(np.clip(
            scale.y_mean + scale.y_std * prediction,
            -MAX_PREDICT_ABS, MAX_PREDICT_ABS,
        ))
        if not math.isfinite(score):
            raise ValueError("Non-finite TCN OOS score.")
        output[row] = score
    return output


def run_tcn_challenger(
    frames: dict[str, pd.DataFrame], common_dates: pd.DatetimeIndex,
    symbols: list[str], folds: list[dict[str, Any]],
    all_decision_dates: pd.DatetimeIndex,
    decision_to_fold: dict[pd.Timestamp, int],
    decision_metadata: dict[pd.Timestamp, dict[str, Any]],
    config: Any, fee_calculator: Callable, slippage: Callable,
    *, progress: Progress | None = None,
) -> tuple[Any, pd.DataFrame, list[dict[str, Any]], dict[pd.Timestamp, dict[str, Any]]]:
    """Train/cross-check on locked folds; no access to OOS for fit/epoch choice."""
    if len(folds) != 3 or len(symbols) != 55:
        raise ValueError("Expected original 55-symbol, three-fold research context.")
    horizon = max(int(x) for x in config.rotation_target_horizons)
    if horizon != 60:
        raise ValueError("TCN predeclared target maturity must be 60 sessions.")
    score_cache = {}
    fold_reports = []
    out_rows = []
    date_keys = set(pd.DatetimeIndex(all_decision_dates[:-1]))
    for fold in folds:
        fid = int(fold["fold_id"])
        train_dates = common_dates[:int(fold["train_end_index"])]
        calibration_dates = common_dates[
            int(fold["calibration_start_index"]):
            int(fold["calibration_end_index"])
        ]
        final_dates = common_dates[:int(fold["final_fit_end_index"])]
        training = _dataset_rows(
            frames, symbols, train_dates,
            maturity_before=fold["calibration_start"], horizon=horizon,
        )
        validation = _dataset_rows(
            frames, symbols, calibration_dates,
            maturity_before=fold["test_start"], horizon=horizon,
        )
        first_scale = _scaling(frames, symbols, train_dates, training)
        x, y = _arrays(frames, symbols, first_scale)
        if progress:
            progress((fid - 1) / len(folds), f"TCN fold {fid}: calibration train/validation")
        _, chosen_epochs, loss = _fit(
            x, y, training, seed=RANDOM_SEED+fid,
            epochs=EPOCHS, valid=validation,
        )
        final_training = _dataset_rows(
            frames, symbols, final_dates,
            maturity_before=fold["test_start"], horizon=horizon,
        )
        final_scale = _scaling(frames, symbols, final_dates, final_training)
        xfinal, yfinal = _arrays(frames, symbols, final_scale)
        if progress:
            progress((fid-.5)/len(folds), f"TCN fold {fid}: final training (epochs={chosen_epochs})")
        final_model, _, _ = _fit(
            xfinal, yfinal, final_training,
            seed=RANDOM_SEED+100+fid,
            epochs=chosen_epochs, valid=None,
        )
        # A decision at t predicts the next open from data through t, without
        # testing presence/price/volume of the next execution session.
        dates = [
            pd.Timestamp(d) for d in fold["decision_dates"][:-1]
            if pd.Timestamp(d) in date_keys
        ]
        per_fold = {}
        for symbol in symbols:
            frame = frames[symbol]
            xmatrix = _feature_matrix(frame)
            positions = frame.index.get_indexer(dates)
            pairs = [
                (symbol, int(position)) for position in positions
                if position >= WINDOW - 1
                and np.isfinite(
                    xmatrix[int(position)-WINDOW+1:int(position)+1]
                ).all()
            ]
            predictions = _predict(final_model, xfinal, pairs, final_scale)
            for (s, pos), score in predictions.items():
                timestamp = pd.Timestamp(frame.index[pos])
                values = per_fold.setdefault(
                    timestamp,
                    np.full(len(symbols)+1, -np.inf, dtype=np.float64),
                )
                values[0] = 0.0
                values[symbols.index(s)+1] = score
                # The scientific utility is an ex post report target only.
                mature = (
                    pos+horizon < len(frame)
                    and pd.Timestamp(frame.index[pos+horizon]) <=
                    pd.Timestamp(common_dates[-1])
                )
                realized = (
                    float(frame[TARGET].iloc[pos]) if mature
                    and np.isfinite(frame[TARGET].iloc[pos]) else None
                )
                out_rows.append({
                    "fold_id": fid, "decision_date": timestamp,
                    "asset": symbol, "predicted_utility": score,
                    "realized_utility_if_mature": realized,
                    "label_mature_at_cutoff": realized is not None,
                })
        if len(per_fold) != len(dates):
            raise ValueError(f"TCN fold {fid} has missing OOS decision dates.")
        if set(per_fold).intersection(score_cache):
            raise ValueError("Overlapping temporal fold predictions.")
        score_cache.update(per_fold)
        fold_reports.append({
            "fold_id": fid,
            "train_rows": len(training),
            "calibration_rows": len(validation),
            "final_train_rows": len(final_training),
            "chosen_epochs_calibration_only": chosen_epochs,
            "calibration_huber_normalized": loss,
            "test_start": pd.Timestamp(fold["test_start"]).isoformat(),
            "test_end": pd.Timestamp(fold["test_end"]).isoformat(),
            "prediction_count": sum(len(row) > 0 for row in per_fold.values()),
            "feature_scaler_scope": "expanding_final_fit_only",
            "label_maturity": "strictly before next segment start",
        })
    if set(score_cache) != date_keys:
        raise ValueError("TCN score cache is not precisely the original OOS calendar.")
    account: dict[str, Any] = {"enabled": True, "audit": {}}
    policies = {}
    # The scientifically frozen function itself stays unchanged. Only the
    # cache of forecast utilities comes from the new, independently trained TCN.
    for fold in folds:
        fid = int(fold["fold_id"])
        keys = {pd.Timestamp(d) for d in fold["decision_dates"][:-1]}
        cache = _CapitalAwareUtilityCache(
            {date: score_cache[date] for date in keys},
            frames, symbols, account,
        )
        policies[fid] = _utility_policy(
            {}, frames, symbols, config, FIXED_SWITCH_MARGIN,
            utility_cache=cache,
        )
    def scheduled(date: pd.Timestamp, position: int, holding: int):
        key = pd.Timestamp(date)
        if key not in decision_to_fold:
            raise ValueError("TCN policy has no fold for decision date.")
        return policies[int(decision_to_fold[key])](key, position, holding)

    def prepare(date, position, holding, cash, shares, equity):
        account.update({
            "decision_timestamp": pd.Timestamp(date),
            "position": position, "holding_days": holding,
            "cash": cash, "shares": shares, "equity": equity,
        })
    run = simulate_feasible_control(
        "tcn_utility", scheduled, frames, symbols, all_decision_dates,
        config, fee_calculator, slippage,
        decision_metadata=decision_metadata, decision_prepare=prepare,
        model_label="Tiny causal TCN · liquidity aware (research only)",
        simulation_progress_callback=(
            lambda ratio, stage: progress(.95+.05*ratio, stage)
            if progress else None
        ),
    )
    return run, pd.DataFrame(out_rows), fold_reports, account["audit"]
