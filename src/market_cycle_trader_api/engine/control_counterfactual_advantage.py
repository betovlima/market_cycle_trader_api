"""Research-only Control counterfactual advantage meta-veto for v10.8.47.

The Control/Liquidity-Aware policy remains the default action.  This module
learns only whether an already proposed asset-to-asset rotation looks harmful
relative to keeping the incumbent.  It never ranks the 55-asset universe.

Labels are paired and causal at training time: a decision at t is trainable
only when the longest forward horizon has fully matured before the
calibration/test boundary.  The label compares the SAME date, candidate and
incumbent using weighted forward returns.  It is a local action-advantage
target, not a claim of exact portfolio-level DeltaCapital.

If pre-OOS calibration skill is insufficient, the model is disabled and the
caller MUST execute the original Control action.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from .control_deep_learning_tcn import FEATURES, WINDOW, TinyTCN, _feature_matrix

MODE = "MCT_RESEARCH_CONTROL_COUNTERFACTUAL_ADVANTAGE_V1"
HORIZONS = (5, 10, 20, 40, 60)
HORIZON_WEIGHTS = (0.10, 0.15, 0.20, 0.30, 0.25)
MAX_HORIZON = max(HORIZONS)
TRAIN_STRIDE = 3
MAX_EPOCHS = 8
PATIENCE = 2
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
RANDOM_SEED = 20260930

# Fail-safe contract.  The neural model is allowed to veto only when its
# validation skill is non-trivial and its probability is strongly negative.
MIN_CALIBRATION_BALANCED_ACCURACY = 0.55
MIN_CALIBRATION_SAMPLES = 30
VETO_PROBABILITY_MAX = 0.40


@dataclass(frozen=True)
class PairScale:
    mean: np.ndarray
    std: np.ndarray


@dataclass(frozen=True)
class PairSample:
    decision_date: pd.Timestamp
    incumbent: str
    candidate: str
    incumbent_loc: int
    candidate_loc: int
    advantage: float

    @property
    def label(self) -> float:
        return 1.0 if self.advantage > 0.0 else 0.0


def _weighted_forward_return(frame: pd.DataFrame, loc: int) -> float:
    """Weighted close-to-close forward return over the frozen Control horizons."""
    start = float(frame["close"].iloc[loc])
    if not math.isfinite(start) or start <= 0:
        raise ValueError("Invalid decision close in counterfactual label.")
    value = 0.0
    for horizon, weight in zip(HORIZONS, HORIZON_WEIGHTS):
        end_loc = loc + horizon
        if end_loc >= len(frame):
            raise ValueError("Counterfactual label has not fully matured.")
        end = float(frame["close"].iloc[end_loc])
        if not math.isfinite(end) or end <= 0:
            raise ValueError("Invalid matured close in counterfactual label.")
        value += float(weight) * (end / start - 1.0)
    return float(value)


def _loc(frame: pd.DataFrame, date: pd.Timestamp) -> int:
    locations = frame.index.get_indexer([pd.Timestamp(date)])
    if len(locations) != 1 or int(locations[0]) < 0:
        return -1
    return int(locations[0])


def build_pair_samples(
    frames: dict[str, pd.DataFrame],
    decisions: pd.DataFrame,
    *,
    maturity_before: pd.Timestamp,
    stride: int = TRAIN_STRIDE,
) -> list[PairSample]:
    """Build incumbent-vs-Control-candidate samples without future leakage.

    Required decision columns:
      decision_timestamp, held_asset_at_decision, liquidity_policy_target.

    CASH transitions are deliberately excluded in v10.8.47.  The model may
    only veto asset-to-asset rotations; entries/exits remain pure Control.
    """
    required = {
        "decision_timestamp", "held_asset_at_decision", "liquidity_policy_target",
    }
    if not required.issubset(decisions.columns):
        raise ValueError(f"Missing decision columns: {sorted(required-set(decisions.columns))}")
    cutoff = pd.Timestamp(maturity_before)
    rows = decisions.copy()
    rows["decision_timestamp"] = pd.to_datetime(
        rows["decision_timestamp"], utc=True, errors="coerce",
    )
    rows = rows.dropna(subset=["decision_timestamp"]).sort_values("decision_timestamp")
    samples: list[PairSample] = []
    for ordinal, row in enumerate(rows.itertuples(index=False)):
        if ordinal % max(1, int(stride)):
            continue
        date = pd.Timestamp(row.decision_timestamp)
        incumbent = str(row.held_asset_at_decision).upper()
        candidate = str(row.liquidity_policy_target).upper()
        if (
            incumbent == "CASH" or candidate == "CASH"
            or incumbent == candidate
            or incumbent not in frames or candidate not in frames
        ):
            continue
        iloc = _loc(frames[incumbent], date)
        cloc = _loc(frames[candidate], date)
        if min(iloc, cloc) < WINDOW - 1:
            continue
        if iloc + MAX_HORIZON >= len(frames[incumbent]) or cloc + MAX_HORIZON >= len(frames[candidate]):
            continue
        incumbent_mature = pd.Timestamp(frames[incumbent].index[iloc + MAX_HORIZON])
        candidate_mature = pd.Timestamp(frames[candidate].index[cloc + MAX_HORIZON])
        if max(incumbent_mature, candidate_mature) >= cutoff:
            continue
        advantage = (
            _weighted_forward_return(frames[candidate], cloc)
            - _weighted_forward_return(frames[incumbent], iloc)
        )
        if math.isfinite(advantage):
            samples.append(PairSample(
                decision_date=date,
                incumbent=incumbent,
                candidate=candidate,
                incumbent_loc=iloc,
                candidate_loc=cloc,
                advantage=float(advantage),
            ))
    return samples


def fit_scale(
    frames: dict[str, pd.DataFrame], samples: list[PairSample],
) -> PairScale:
    blocks = []
    for sample in samples:
        c = _feature_matrix(frames[sample.candidate])[
            sample.candidate_loc-WINDOW+1:sample.candidate_loc+1
        ]
        i = _feature_matrix(frames[sample.incumbent])[
            sample.incumbent_loc-WINDOW+1:sample.incumbent_loc+1
        ]
        blocks.append(c - i)
    if not blocks:
        raise ValueError("No counterfactual pair features.")
    raw = np.vstack(blocks)
    if not np.isfinite(raw).all():
        raise ValueError("Non-finite counterfactual pair feature.")
    return PairScale(raw.mean(axis=0), np.maximum(raw.std(axis=0), 1e-6))


def _tensor(
    frames: dict[str, pd.DataFrame],
    samples: list[PairSample],
    scale: PairScale,
) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = [], []
    for sample in samples:
        candidate = _feature_matrix(frames[sample.candidate])[
            sample.candidate_loc-WINDOW+1:sample.candidate_loc+1
        ]
        incumbent = _feature_matrix(frames[sample.incumbent])[
            sample.incumbent_loc-WINDOW+1:sample.incumbent_loc+1
        ]
        difference = np.clip((candidate-incumbent-scale.mean)/scale.std, -8.0, 8.0)
        x.append(difference.T)
        y.append(sample.label)
    if not x:
        raise ValueError("No counterfactual samples.")
    return (
        torch.from_numpy(np.stack(x).astype(np.float32)),
        torch.tensor(y, dtype=torch.float32),
    )


def balanced_accuracy(probabilities: np.ndarray, labels: np.ndarray) -> float:
    pred = probabilities >= 0.5
    truth = labels >= 0.5
    pos = truth.sum()
    neg = (~truth).sum()
    if pos == 0 or neg == 0:
        return 0.5
    tpr = float((pred & truth).sum()) / float(pos)
    tnr = float(((~pred) & (~truth)).sum()) / float(neg)
    return 0.5 * (tpr + tnr)


def fit_meta_veto(
    frames: dict[str, pd.DataFrame],
    training: list[PairSample],
    validation: list[PairSample],
    *,
    seed: int,
) -> tuple[TinyTCN, PairScale, int, dict[str, Any]]:
    if not training or not validation:
        raise ValueError("Meta-veto requires chronological train and calibration samples.")
    scale = fit_scale(frames, training)
    x_train, y_train = _tensor(frames, training, scale)
    x_valid, y_valid = _tensor(frames, validation, scale)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = TinyTCN().cpu()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        )
        criterion = nn.BCEWithLogitsLoss()
        best_state = None
        best_loss = float("inf")
        best_epoch = 1
        stale = 0
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_train), y_train)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.eval()
            with torch.no_grad():
                valid_loss = float(criterion(model(x_valid), y_valid))
            if valid_loss < best_loss - 1e-6:
                best_loss = valid_loss
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= PATIENCE:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(x_valid)).numpy()
        labels = y_valid.numpy()
        skill = balanced_accuracy(probs, labels)
        enabled = (
            len(validation) >= MIN_CALIBRATION_SAMPLES
            and skill >= MIN_CALIBRATION_BALANCED_ACCURACY
        )
        report = {
            "validation_samples": len(validation),
            "validation_positive_rate": float(labels.mean()),
            "validation_bce": best_loss,
            "validation_balanced_accuracy": float(skill),
            "chosen_epochs": int(best_epoch),
            "model_enabled": bool(enabled),
            "activation_rule": (
                f"n>={MIN_CALIBRATION_SAMPLES} and balanced_accuracy>="
                f"{MIN_CALIBRATION_BALANCED_ACCURACY:.2f}"
            ),
        }
        return model, scale, best_epoch, report


def predict_rotation_advantage_probability(
    model: TinyTCN,
    scale: PairScale,
    frames: dict[str, pd.DataFrame],
    *,
    date: pd.Timestamp,
    incumbent: str,
    candidate: str,
) -> float | None:
    if incumbent == "CASH" or candidate == "CASH" or incumbent == candidate:
        return None
    if incumbent not in frames or candidate not in frames:
        return None
    iloc = _loc(frames[incumbent], date)
    cloc = _loc(frames[candidate], date)
    if min(iloc, cloc) < WINDOW - 1:
        return None
    c = _feature_matrix(frames[candidate])[cloc-WINDOW+1:cloc+1]
    i = _feature_matrix(frames[incumbent])[iloc-WINDOW+1:iloc+1]
    diff = np.clip((c-i-scale.mean)/scale.std, -8.0, 8.0)
    tensor = torch.from_numpy(diff.T[None, :, :].astype(np.float32))
    model.eval()
    with torch.no_grad():
        probability = float(torch.sigmoid(model(tensor))[0])
    return probability if math.isfinite(probability) else None


def apply_meta_veto(
    *,
    current_position: int,
    control_target: int,
    probability_positive_advantage: float | None,
    model_enabled: bool,
) -> tuple[int, str]:
    """Fail-safe policy: only a strong negative signal may change Control."""
    if not model_enabled:
        return control_target, "CONTROL_FALLBACK_MODEL_DISABLED"
    if current_position <= 0 or control_target <= 0 or current_position == control_target:
        return control_target, "CONTROL_FALLBACK_NON_ROTATION"
    if probability_positive_advantage is None:
        return control_target, "CONTROL_FALLBACK_NO_SCORE"
    if probability_positive_advantage <= VETO_PROBABILITY_MAX:
        return current_position, "META_VETO_STRONG_NEGATIVE_ADVANTAGE"
    return control_target, "CONTROL_FALLBACK_NO_VETO"
