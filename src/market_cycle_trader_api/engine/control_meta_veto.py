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
