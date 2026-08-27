from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..classification_evaluation import roc_curve_payload
from ..engine.capital_rotation import ROTATION_FEATURES
from ..engine.temporal_intelligence import _fit_binary_classifier_relaxed
from .metrics import finite
from .threshold_selection import select_threshold


@dataclass(frozen=True)
class RelativeProbabilityCalibrator:
    model: Any | None

    def transform(self, raw_probability: np.ndarray) -> np.ndarray:
        values = np.clip(np.asarray(raw_probability, dtype=float), 1e-6, 1.0 - 1e-6)
        if self.model is None:
            return values
        logits = np.log(values / (1.0 - values)).reshape(-1, 1)
        return np.clip(self.model.predict_proba(logits)[:, 1], 1e-6, 1.0 - 1e-6)


def _fit_probability_calibrator(raw_probability: np.ndarray, relative_edge_values: np.ndarray, *, random_state: int, minimum_samples: int) -> RelativeProbabilityCalibrator:
    from sklearn.linear_model import LogisticRegression

    probabilities = np.clip(np.asarray(raw_probability, dtype=float), 1e-6, 1.0 - 1e-6)
    labels = np.asarray(relative_edge_values > 0.0, dtype=int)
    if len(probabilities) < int(minimum_samples) or len(np.unique(labels)) < 2:
        return RelativeProbabilityCalibrator(None)
    logits = np.log(probabilities / (1.0 - probabilities)).reshape(-1, 1)
    model = LogisticRegression(random_state=int(random_state), solver="lbfgs")
    model.fit(logits, labels)
    return RelativeProbabilityCalibrator(model)


def _stamp(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _asset(value: Any) -> str:
    text = str(value or "CASH").strip().upper()
    return text or "CASH"


def _pair_columns() -> list[str]:
    return [f"delta_{name}" for name in ROTATION_FEATURES] + [f"mean_{name}" for name in ROTATION_FEATURES]


def _pair_vector(control: np.ndarray, challenger: np.ndarray) -> np.ndarray:
    return np.concatenate((challenger - control, (challenger + control) * 0.5))


def _pair_frame(control: pd.Series, challenger: pd.Series) -> pd.DataFrame | None:
    left = pd.to_numeric(control.reindex(ROTATION_FEATURES), errors="coerce").to_numpy(dtype=float)
    right = pd.to_numeric(challenger.reindex(ROTATION_FEATURES), errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return None
    return pd.DataFrame([_pair_vector(left, right)], columns=_pair_columns())


def _return_lookup(target_matrix: pd.DataFrame, timestamp: pd.Timestamp, symbol: str) -> float | None:
    try:
        return finite(target_matrix.at[timestamp, symbol])
    except (KeyError, TypeError):
        return None


def relative_edge(
    target_matrix: pd.DataFrame,
    timestamp: pd.Timestamp,
    control_symbol: str,
    challenger_symbol: str,
    *,
    round_trip_cost_rate: float,
) -> float | None:
    control_return = _return_lookup(target_matrix, timestamp, control_symbol)
    challenger_return = _return_lookup(target_matrix, timestamp, challenger_symbol)
    if control_return is None or challenger_return is None:
        return None
    gross_relative = float(np.expm1(float(challenger_return) - float(control_return)))
    return gross_relative - max(0.0, float(round_trip_cost_rate))


def _sample_seed(base_seed: int, timestamp: pd.Timestamp, fold_id: int, horizon: int, split_name: str) -> int:
    token = f"{int(base_seed)}|{timestamp.isoformat()}|{int(fold_id)}|{int(horizon)}|{split_name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big", signed=False)


def build_pair_dataset(
    training: dict[str, Any],
    *,
    fold_id: int,
    horizon: int,
    split_name: str,
    round_trip_cost_rate: float,
    max_pairs_per_timestamp: int,
    random_state: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    fold_context = training["fold_contexts"][int(fold_id)]
    split = fold_context["splits"][split_name]
    x = split["x"].reset_index(drop=True)
    metadata = split["metadata"].reset_index(drop=True)
    target_matrix = training["targets_by_horizon"][int(horizon)]["return"]
    if x.empty or metadata.empty:
        return pd.DataFrame(columns=_pair_columns()), np.asarray([], dtype=float)

    frame = x.copy()
    frame["timestamp"] = pd.to_datetime(metadata["timestamp"], utc=True)
    frame["symbol"] = metadata["symbol"].astype(str).str.upper()
    feature_values: list[np.ndarray] = []
    targets: list[float] = []

    for timestamp, group in frame.groupby("timestamp", sort=True):
        timestamp = _stamp(timestamp)
        if timestamp is None:
            continue
        valid_rows: list[tuple[str, np.ndarray, float]] = []
        for _, row in group.iterrows():
            symbol = _asset(row.get("symbol"))
            values = pd.to_numeric(row.reindex(ROTATION_FEATURES), errors="coerce").to_numpy(dtype=float)
            realized = _return_lookup(target_matrix, timestamp, symbol)
            if symbol == "CASH" or realized is None or not np.isfinite(values).all():
                continue
            valid_rows.append((symbol, values, float(realized)))
        valid_rows.sort(key=lambda item: item[0])
        if len(valid_rows) < 2:
            continue

        pairs = list(itertools.combinations(range(len(valid_rows)), 2))
        limit = int(max_pairs_per_timestamp)
        if limit > 0 and len(pairs) > limit:
            rng = np.random.default_rng(_sample_seed(random_state, timestamp, fold_id, horizon, split_name))
            selected = np.sort(rng.choice(len(pairs), size=limit, replace=False))
            pairs = [pairs[int(position)] for position in selected]

        for left_index, right_index in pairs:
            _, left_features, left_return = valid_rows[left_index]
            _, right_features, right_return = valid_rows[right_index]
            for control_features, control_return, challenger_features, challenger_return in (
                (left_features, left_return, right_features, right_return),
                (right_features, right_return, left_features, left_return),
            ):
                edge = float(np.expm1(challenger_return - control_return)) - max(0.0, float(round_trip_cost_rate))
                feature_values.append(_pair_vector(control_features, challenger_features))
                targets.append(edge)

    if not feature_values:
        return pd.DataFrame(columns=_pair_columns()), np.asarray([], dtype=float)
    return pd.DataFrame(np.vstack(feature_values), columns=_pair_columns()), np.asarray(targets, dtype=float)


def fit_fold_horizon(
    training: dict[str, Any],
    config: Any,
    *,
    fold: dict[str, Any],
    horizon: int,
    settings: dict[str, Any],
    round_trip_cost_rate: float,
) -> dict[str, Any]:
    fold_id = int(fold["fold_id"])
    max_pairs = int(settings["max_pairs_per_timestamp"])
    train_x, train_edge = build_pair_dataset(
        training,
        fold_id=fold_id,
        horizon=horizon,
        split_name="train",
        round_trip_cost_rate=round_trip_cost_rate,
        max_pairs_per_timestamp=max_pairs,
        random_state=int(config.random_state),
    )
    calibration_x, calibration_edge = build_pair_dataset(
        training,
        fold_id=fold_id,
        horizon=horizon,
        split_name="calibration",
        round_trip_cost_rate=round_trip_cost_rate,
        max_pairs_per_timestamp=max_pairs,
        random_state=int(config.random_state),
    )
    labels = np.asarray(calibration_edge > 0.0, dtype=int)
    positive_count = int((labels == 1).sum())
    negative_count = int((labels == 0).sum())
    eligible = bool(
        len(labels) >= int(settings["minimum_calibration_samples"])
        and positive_count >= int(settings["minimum_class_samples"])
        and negative_count >= int(settings["minimum_class_samples"])
        and len(train_x) >= int(settings["minimum_training_samples"])
        and len(np.unique(np.asarray(train_edge > 0.0, dtype=int))) == 2
    )
    base = {
        "fold_id": fold_id,
        "horizon": int(horizon),
        "target": "challenger_relative_outperformance_net_rotation_cost",
        "training_samples": int(len(train_edge)),
        "calibration_samples": int(len(labels)),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "round_trip_cost_rate": float(round_trip_cost_rate),
        "max_pairs_per_timestamp": max_pairs,
    }
    if not eligible:
        return {**base, "eligible": False, "reason": "insufficient_relative_pair_sample"}

    model = _fit_binary_classifier_relaxed(train_x, train_edge, config)
    raw_calibration = model.predict_proba(calibration_x)[:, 1]
    calibrator = _fit_probability_calibrator(
        raw_calibration,
        calibration_edge,
        random_state=int(config.random_state),
        minimum_samples=int(settings["minimum_calibration_samples"]),
    )
    probabilities = np.clip(calibrator.transform(raw_calibration), 0.0, 1.0)
    selected = select_threshold(labels, probabilities, metric=str(settings["selection_metric"]))
    auc = float(roc_auc_score(labels, probabilities))
    roc = roc_curve_payload(
        labels,
        probabilities,
        operating_threshold=float(selected["threshold"]),
        operating_point_role="relative_rotation_threshold",
        threshold_origin="chronological_relative_pair_calibration",
        validation_metric_name=str(settings["selection_metric"]),
        validation_metric_value=float(selected["selection_score"]),
        max_points=int(settings["max_curve_points"]),
    )
    return {
        **base,
        "eligible": True,
        "threshold": float(selected["threshold"]),
        "selection_metric": str(settings["selection_metric"]),
        "selection_score": float(selected["selection_score"]),
        "calibration_auc": auc,
        "calibration_roc": roc,
        "_model": model,
        "_calibrator": calibrator,
    }


def score_pair(
    training: dict[str, Any],
    calibration: dict[str, Any],
    *,
    decision_timestamp: Any,
    control_symbol: str,
    challenger_symbol: str,
    horizon: int,
    round_trip_cost_rate: float,
) -> dict[str, Any] | None:
    timestamp = _stamp(decision_timestamp)
    control_symbol = _asset(control_symbol)
    challenger_symbol = _asset(challenger_symbol)
    if timestamp is None or control_symbol == "CASH" or challenger_symbol in {"CASH", control_symbol}:
        return None
    control_frame = training["frames"].get(control_symbol)
    challenger_frame = training["frames"].get(challenger_symbol)
    if control_frame is None or challenger_frame is None or timestamp not in control_frame.index or timestamp not in challenger_frame.index:
        return None
    pair = _pair_frame(control_frame.loc[timestamp], challenger_frame.loc[timestamp])
    if pair is None:
        return None
    model = calibration.get("_model")
    calibrator = calibration.get("_calibrator")
    if model is None or calibrator is None:
        return None
    raw_probability = float(model.predict_proba(pair)[:, 1][0])
    probability = float(np.clip(calibrator.transform(np.asarray([raw_probability], dtype=float))[0], 0.0, 1.0))
    threshold = float(calibration["threshold"])
    edge = relative_edge(
        training["targets_by_horizon"][int(horizon)]["return"],
        timestamp,
        control_symbol,
        challenger_symbol,
        round_trip_cost_rate=round_trip_cost_rate,
    )
    return {
        "horizon": int(horizon),
        "probability": probability,
        "threshold": threshold,
        "margin": probability - threshold,
        "realized_relative_edge": edge,
        "realized_outperformance": None if edge is None else bool(edge > 0.0),
    }
