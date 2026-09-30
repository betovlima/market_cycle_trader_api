"""Control-first Deep Learning meta-veto research.

The neural model never chooses an asset. It may only veto rotations already
proposed by the original LightGBM Control plus the unchanged v10.8.44
liquidity overlay. A fold cannot intervene unless pre-OOS validation earns
that right; otherwise it falls back exactly to the base policy.
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
from torch.utils.data import DataLoader, Dataset

from ..tcc_v106_reference import research_challengers as scientific
from ..tcc_v106_reference.capital_rotation import (
    _precompute_model_utilities,
    _utility_policy as frozen_utility_policy,
)
from .control_deep_learning_tcn import FEATURES, WINDOW, HIDDEN, TinyTCN, _feature_matrix
from .control_execution_feasibility import simulate_feasible_control
from .control_liquidity_policy import _CapitalAwareUtilityCache

MODE = "MCT_RESEARCH_CONTROL_META_VETO_V1"
TARGET = "forward_risk_adjusted_utility"
HORIZON = 60
MAX_EPOCHS = 6
PATIENCE = 2
BATCH_SIZE = 256
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
RANDOM_SEED = 20260930
VALIDATION_DATE_FRACTION = 0.30
VETO_THRESHOLDS = (0.10, 0.20, 0.30, 0.40)
MIN_VETO_PRECISION = 0.60
MIN_VETO_VALIDATION_SAMPLES = 50
Progress = Callable[[float, str], None]


@dataclass(frozen=True)
class MetaScale:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    context_mean: np.ndarray
    context_std: np.ndarray


@dataclass(frozen=True)
class PairRow:
    decision_date: pd.Timestamp
    candidate: str
    candidate_loc: int
    incumbent: str
    incumbent_loc: int
    candidate_utility: float
    incumbent_utility: float
    label_accept: float


class MetaVetoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal = TinyTCN().encoder
        self.head = nn.Sequential(
            nn.Linear(HIDDEN * 3 + 3, HIDDEN),
            nn.ReLU(),
            nn.Dropout(.10),
            nn.Linear(HIDDEN, 1),
        )

    def forward(
        self,
        candidate: torch.Tensor,
        incumbent: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        c = self.temporal(candidate)[:, :, -1]
        h = self.temporal(incumbent)[:, :, -1]
        return self.head(torch.cat((c, h, c - h, context), dim=1)).squeeze(-1)


class PairDataset(Dataset):
    def __init__(self, rows: list[PairRow], arrays: dict[str, np.ndarray], scale: MetaScale):
        self.rows = rows
        self.arrays = arrays
        self.scale = scale

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        candidate = self.arrays[row.candidate][row.candidate_loc-WINDOW+1:row.candidate_loc+1].T
        incumbent = self.arrays[row.incumbent][row.incumbent_loc-WINDOW+1:row.incumbent_loc+1].T
        raw = np.asarray([
            row.candidate_utility,
            row.incumbent_utility,
            row.candidate_utility-row.incumbent_utility,
        ], dtype=np.float64)
        context = np.clip(
            (raw-self.scale.context_mean)/self.scale.context_std, -8.0, 8.0,
        ).astype(np.float32)
        return (
            torch.from_numpy(candidate.copy()),
            torch.from_numpy(incumbent.copy()),
            torch.from_numpy(context),
            torch.tensor(row.label_accept, dtype=torch.float32),
        )


def _build_pair_rows(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    calibration_dates: pd.DatetimeIndex,
    utility_cache: dict[pd.Timestamp, np.ndarray],
    *,
    test_start: pd.Timestamp,
) -> list[PairRow]:
    """Top pre-OOS Control proposal vs lower-ranked possible incumbents."""
    cutoff = pd.Timestamp(test_start)
    output: list[PairRow] = []
    for date in pd.DatetimeIndex(calibration_dates):
        key = pd.Timestamp(date)
        utilities = np.asarray(utility_cache.get(key), dtype=np.float64)
        if len(utilities) != len(symbols)+1 or not np.isfinite(utilities[1:]).any():
            continue
        candidate_position = int(
            np.argmax(np.where(np.isfinite(utilities[1:]), utilities[1:], -np.inf))
        ) + 1
        candidate = symbols[candidate_position-1]
        cframe = frames[candidate]
        cidx = cframe.index.get_indexer([key])
        if len(cidx) != 1 or int(cidx[0]) < WINDOW-1:
            continue
        cloc = int(cidx[0])
        if cloc + HORIZON >= len(cframe):
            continue
        if pd.Timestamp(cframe.index[cloc+HORIZON]) >= cutoff:
            continue
        creal = float(cframe[TARGET].iloc[cloc])
        cwindow = _feature_matrix(cframe)[cloc-WINDOW+1:cloc+1]
        if not math.isfinite(creal) or not np.isfinite(cwindow).all():
            continue
        cu = float(utilities[candidate_position])

        for incumbent_position, incumbent in enumerate(symbols, start=1):
            if incumbent_position == candidate_position:
                continue
            iu = float(utilities[incumbent_position])
            if not math.isfinite(iu) or cu <= iu:
                continue
            iframe = frames[incumbent]
            iidx = iframe.index.get_indexer([key])
            if len(iidx) != 1 or int(iidx[0]) < WINDOW-1:
                continue
            iloc = int(iidx[0])
            if iloc + HORIZON >= len(iframe):
                continue
            if pd.Timestamp(iframe.index[iloc+HORIZON]) >= cutoff:
                continue
            ireal = float(iframe[TARGET].iloc[iloc])
            iwindow = _feature_matrix(iframe)[iloc-WINDOW+1:iloc+1]
            if not math.isfinite(ireal) or not np.isfinite(iwindow).all():
                continue
            output.append(PairRow(
                key, candidate, cloc, incumbent, iloc, cu, iu,
                float(creal > ireal),
            ))
    if not output:
        raise ValueError("No fully matured pre-OOS Control proposal pairs.")
    return output


def _split_pairs(rows: list[PairRow]) -> tuple[list[PairRow], list[PairRow]]:
    dates = sorted({row.decision_date for row in rows})
    if len(dates) < 10:
        raise ValueError("Too few chronological meta-veto dates.")
    n_valid = max(3, int(math.ceil(len(dates)*VALIDATION_DATE_FRACTION)))
    valid_dates = set(dates[-n_valid:])
    train = [row for row in rows if row.decision_date not in valid_dates]
    valid = [row for row in rows if row.decision_date in valid_dates]
    if not train or not valid:
        raise ValueError("Meta-veto chronological split is empty.")
    return train, valid


def _scale(frames: dict[str, pd.DataFrame], rows: list[PairRow]) -> MetaScale:
    blocks = []
    contexts = []
    for row in rows:
        blocks.append(_feature_matrix(frames[row.candidate])[row.candidate_loc-WINDOW+1:row.candidate_loc+1])
        blocks.append(_feature_matrix(frames[row.incumbent])[row.incumbent_loc-WINDOW+1:row.incumbent_loc+1])
        contexts.append([
            row.candidate_utility,
            row.incumbent_utility,
            row.candidate_utility-row.incumbent_utility,
        ])
    features = np.vstack(blocks)
    context = np.asarray(contexts, dtype=np.float64)
    return MetaScale(
        feature_mean=features.mean(axis=0),
        feature_std=np.maximum(features.std(axis=0), 1e-6),
        context_mean=context.mean(axis=0),
        context_std=np.maximum(context.std(axis=0), 1e-6),
    )


def _arrays(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    scale: MetaScale,
) -> dict[str, np.ndarray]:
    return {
        symbol: np.clip(
            (_feature_matrix(frames[symbol])-scale.feature_mean)/scale.feature_std,
            -8.0, 8.0,
        ).astype(np.float32)
        for symbol in symbols
    }


def _epoch(
    model: MetaVetoNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, list[float], list[int]]:
    model.train(optimizer is not None)
    total_loss = 0.0
    count = 0
    probabilities: list[float] = []
    labels: list[int] = []
    for candidate, incumbent, context, y in loader:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(candidate, incumbent, context)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
            if optimizer is not None:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        n = len(y)
        total_loss += float(loss.detach())*n
        count += n
        probabilities.extend(torch.sigmoid(logits.detach()).cpu().tolist())
        labels.extend(y.to(torch.int64).cpu().tolist())
    return total_loss/max(1,count), probabilities, labels


def _choose_threshold(
    probabilities: list[float],
    labels: list[int],
) -> tuple[float | None, list[dict[str, Any]]]:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=int)
    diagnostics = []
    qualified = []
    for threshold in VETO_THRESHOLDS:
        veto = p < threshold
        count = int(veto.sum())
        precision = float(np.mean(y[veto] == 0)) if count else None
        row = {
            "threshold": float(threshold),
            "veto_validation_count": count,
            "bad_rotation_precision": precision,
            "qualified": bool(
                count >= MIN_VETO_VALIDATION_SAMPLES
                and precision is not None
                and precision >= MIN_VETO_PRECISION
            ),
        }
        diagnostics.append(row)
        if row["qualified"]:
            qualified.append(row)
    return max(
        (float(row["threshold"]) for row in qualified),
        default=None,
    ), diagnostics


def _fit(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    training: list[PairRow],
    validation: list[PairRow],
    *,
    seed: int,
) -> tuple[MetaVetoNet, MetaScale, dict[str, Any]]:
    train_scale = _scale(frames, training)
    train_arrays = _arrays(frames, symbols, train_scale)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = MetaVetoNet().to("cpu")
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        )
        generator = torch.Generator(device="cpu").manual_seed(seed)
        train_loader = DataLoader(
            PairDataset(training, train_arrays, train_scale),
            batch_size=BATCH_SIZE, shuffle=True, generator=generator,
            num_workers=0,
        )
        valid_loader = DataLoader(
            PairDataset(validation, train_arrays, train_scale),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        )
        best_loss = float("inf")
        best_epoch = 1
        stale = 0
        best_probs: list[float] = []
        best_labels: list[int] = []
        for epoch in range(1, MAX_EPOCHS+1):
            _epoch(model, train_loader, optimizer)
            valid_loss, probs, labels = _epoch(model, valid_loader, None)
            if valid_loss < best_loss - 1e-6:
                best_loss = valid_loss
                best_epoch = epoch
                best_probs = probs
                best_labels = labels
                stale = 0
            else:
                stale += 1
                if stale >= PATIENCE:
                    break
        threshold, threshold_diagnostics = _choose_threshold(
            best_probs, best_labels,
        )

    all_rows = training + validation
    final_scale = _scale(frames, all_rows)
    final_arrays = _arrays(frames, symbols, final_scale)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed+1000)
        final_model = MetaVetoNet().to("cpu")
        optimizer = torch.optim.AdamW(
            final_model.parameters(), lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        generator = torch.Generator(device="cpu").manual_seed(seed+1000)
        loader = DataLoader(
            PairDataset(all_rows, final_arrays, final_scale),
            batch_size=BATCH_SIZE, shuffle=True, generator=generator,
            num_workers=0,
        )
        for _ in range(best_epoch):
            _epoch(final_model, loader, optimizer)

    return final_model, final_scale, {
        "train_pairs": len(training),
        "validation_pairs": len(validation),
        "train_dates": len({row.decision_date for row in training}),
        "validation_dates": len({row.decision_date for row in validation}),
        "train_accept_rate": float(np.mean([
            row.label_accept for row in training
        ])),
        "validation_accept_rate": float(np.mean([
            row.label_accept for row in validation
        ])),
        "chosen_epochs": best_epoch,
        "best_validation_bce": best_loss,
        "threshold_candidates": threshold_diagnostics,
        "chosen_veto_threshold": threshold,
        "intervention_enabled": threshold is not None,
        "minimum_veto_precision_required": MIN_VETO_PRECISION,
        "minimum_veto_samples_required": MIN_VETO_VALIDATION_SAMPLES,
    }


def _probability(
    model: MetaVetoNet,
    scale: MetaScale,
    arrays: dict[str, np.ndarray],
    frames: dict[str, pd.DataFrame],
    *,
    date: pd.Timestamp,
    candidate: str,
    incumbent: str,
    candidate_utility: float,
    incumbent_utility: float,
) -> float | None:
    cframe = frames[candidate]
    iframe = frames[incumbent]
    cidx = cframe.index.get_indexer([date])
    iidx = iframe.index.get_indexer([date])
    if (
        len(cidx) != 1 or len(iidx) != 1
        or int(cidx[0]) < WINDOW-1 or int(iidx[0]) < WINDOW-1
    ):
        return None
    cloc, iloc = int(cidx[0]), int(iidx[0])
    candidate_window = arrays[candidate][cloc-WINDOW+1:cloc+1].T
    incumbent_window = arrays[incumbent][iloc-WINDOW+1:iloc+1].T
    if not np.isfinite(candidate_window).all() or not np.isfinite(incumbent_window).all():
        return None
    raw = np.asarray([
        candidate_utility,
        incumbent_utility,
        candidate_utility-incumbent_utility,
    ], dtype=np.float64)
    context = np.clip(
        (raw-scale.context_mean)/scale.context_std, -8.0, 8.0,
    ).astype(np.float32)
    model.eval()
    with torch.no_grad():
        logit = model(
            torch.from_numpy(candidate_window[None].astype(np.float32)),
            torch.from_numpy(incumbent_window[None].astype(np.float32)),
            torch.from_numpy(context[None]),
        )
        value = float(torch.sigmoid(logit).item())
    return value if math.isfinite(value) else None


def run_meta_veto_pair(
    bars: dict[str, pd.DataFrame],
    config: Any,
    fee_calculator: Callable,
    slippage: Callable,
    *,
    progress_callback: Callable[[float, str, int], None] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[pd.Timestamp, dict[str, Any]]]:
    """Fit meta-veto on pre-OOS calibration and replay paired OOS paths."""
    (
        frames, common_dates, symbols, folds, _all_decision_dates,
        _decision_to_fold, _decision_metadata,
    ) = scientific._build_execution_context(bars, config)
    if len(symbols) != 55 or len(folds) != 3:
        raise ValueError("Meta-veto requires original 55 assets and 3 folds.")

    fold_models: dict[int, MetaVetoNet] = {}
    fold_scales: dict[int, MetaScale] = {}
    fold_thresholds: dict[int, float | None] = {}
    fold_reports: list[dict[str, Any]] = []

    for fold in folds:
        fid = int(fold["fold_id"])
        train_dates = common_dates[:int(fold["train_end_index"])]
        calibration_dates = common_dates[
            int(fold["calibration_start_index"]):int(fold["calibration_end_index"])
        ]
        if progress_callback:
            progress_callback(
                2.0 + 5.0*(fid-1),
                f"Meta-veto fold {fid}: fitting pre-OOS proposal model",
                0,
            )
        calibration_models = scientific._lightgbm_fit_models(
            frames, symbols, train_dates, config,
            phase=f"meta_veto_fold_{fid}_proposal_source",
        )
        calibration_cache, _ = _precompute_model_utilities(
            calibration_models, frames, symbols, calibration_dates, config,
        )
        rows = _build_pair_rows(
            frames, symbols, calibration_dates, calibration_cache,
            test_start=pd.Timestamp(fold["test_start"]),
        )
        training, validation = _split_pairs(rows)
        model, scale, report = _fit(
            frames, symbols, training, validation,
            seed=RANDOM_SEED+fid,
        )
        fold_models[fid] = model
        fold_scales[fid] = scale
        fold_thresholds[fid] = report["chosen_veto_threshold"]
        fold_reports.append({
            "fold_id": fid,
            "test_start": pd.Timestamp(fold["test_start"]).isoformat(),
            "test_end": pd.Timestamp(fold["test_end"]).isoformat(),
            "proposal_pair_source": "top calibration LightGBM candidate versus lower-ranked possible incumbents",
            "meta_label": "candidate forward_risk_adjusted_utility > incumbent forward_risk_adjusted_utility",
            **report,
        })

    original_runner = scientific._run_lightgbm
    original_predict = scientific._precompute_model_utilities
    original_policy = original_runner.__globals__.get("_utility_policy")
    original_simulator = original_runner.__globals__.get("_simulate_exact")
    if (
        original_policy is not frozen_utility_policy
        or original_simulator is not scientific._simulate_exact
        or original_runner.__globals__.get("_precompute_model_utilities") is not original_predict
    ):
        raise RuntimeError("Frozen Control bindings differ from expected source.")

    account: dict[str, Any] = {
        "enabled": True,
        "veto_enabled": False,
        "audit": {},
        "veto_audit": {},
    }
    arrays_by_fold = {
        fid: _arrays(frames, symbols, fold_scales[fid])
        for fid in fold_scales
    }

    def liquidity_utilities(models, panel, labels, timestamps, settings):
        cached, profile = original_predict(
            models, panel, labels, timestamps, settings,
        )
        return _CapitalAwareUtilityCache(cached, panel, labels, account), profile


    def policy_wrapper(
        models, panel, labels, settings, switch_margin, **kwargs,
    ):
        base_policy = original_policy(
            models, panel, labels, settings, switch_margin, **kwargs,
        )
        cache = kwargs.get("utility_cache")
        fold_id = kwargs.get("fold_id")
        if cache is None or fold_id is None:
            return base_policy
        fid = int(fold_id)
        if fid not in fold_models:
            return base_policy

        def policy(
            timestamp: pd.Timestamp,
            current_position: int,
            holding_days: int,
        ) -> tuple[int, float]:
            base_target, base_score = base_policy(
                timestamp, current_position, holding_days,
            )
            key = pd.Timestamp(timestamp)
            audit = {
                "fold_id": fid,
                "base_target": labels[base_target-1] if base_target else "CASH",
                "current_asset": labels[current_position-1] if current_position else "CASH",
                "veto_enabled": bool(account.get("veto_enabled", False)),
                "fold_threshold": fold_thresholds[fid],
                "probability_candidate_better": None,
                "veto_applied": False,
                "final_target": labels[base_target-1] if base_target else "CASH",
            }
            threshold = fold_thresholds[fid]
            if (
                not account.get("veto_enabled", False)
                or threshold is None
                or current_position <= 0
                or base_target <= 0
                or base_target == current_position
            ):
                if account.get("veto_enabled", False):
                    account["veto_audit"][key] = audit
                return int(base_target), float(base_score)

            effective = np.asarray(cache.get(key), dtype=float)
            candidate_utility = float(effective[base_target])
            incumbent_utility = float(effective[current_position])
            probability = _probability(
                fold_models[fid],
                fold_scales[fid],
                arrays_by_fold[fid],
                frames,
                date=key,
                candidate=labels[base_target-1],
                incumbent=labels[current_position-1],
                candidate_utility=candidate_utility,
                incumbent_utility=incumbent_utility,
            )
            audit["probability_candidate_better"] = probability
            if probability is not None and probability < float(threshold):
                audit["veto_applied"] = True
                audit["final_target"] = labels[current_position-1]
                account["veto_audit"][key] = audit
                return current_position, incumbent_utility
            account["veto_audit"][key] = audit
            return int(base_target), float(base_score)

        return policy
