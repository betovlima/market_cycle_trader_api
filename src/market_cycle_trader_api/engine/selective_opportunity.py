from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple

import numpy as np
import pandas as pd

SELECTIVE_ROTATION_MODE = "COMPOUND_ROTATION_SWING_SELECTIVE"
OPTIMIZED_ALLOCATION_MODE = "COMPOUND_ROTATION_SWING_OPTIMIZED_ALLOCATION"

OPPORTUNITY_FEATURES = (
    "best_score",
    "second_score",
    "best_vs_second_gap",
    "universe_score_mean",
    "universe_score_std",
    "best_score_zscore",
    "positive_score_fraction",
    "best_return_5",
    "best_return_20",
    "best_return_60",
    "best_vol_20",
    "best_vol_60",
    "best_trend_efficiency_20",
    "best_trend_efficiency_60",
    "universe_breadth_5",
    "universe_breadth_20",
)


class OpportunityEvaluation(NamedTuple):
    probability: float
    confidence: float
    accepted: bool
    features: dict[str, float]
    best_position: int


@dataclass(frozen=True)
class SelectiveOpportunityGate:
    model: Any | None
    threshold: float
    constant_probability: float | None
    training_rows: int
    positive_rate: float
    threshold_validation_rows: int
    threshold_validation_score: float
    reference_probabilities: tuple[float, ...] = ()
    threshold_validation_accepted: int = 0
    calibration_method: str = "prequential_relative_confidence_v2"

    def probability(self, values: dict[str, float]) -> float:
        vector = pd.DataFrame([[float(values[name]) for name in OPPORTUNITY_FEATURES]], columns=list(OPPORTUNITY_FEATURES), dtype=float)
        if not np.isfinite(vector.to_numpy(dtype=float)).all():
            return 0.0
        if self.model is None:
            return float(self.constant_probability or 0.0)
        probability = float(self.model.predict_proba(vector)[0, 1])
        return min(1.0, max(0.0, probability))

    def confidence_from_probability(self, probability: float) -> float:
        value = min(1.0, max(0.0, float(probability)))
        if self.model is None or not self.reference_probabilities:
            return value
        reference = np.asarray(self.reference_probabilities, dtype=float)
        rank = int(np.searchsorted(reference, value, side="right"))
        return min(1.0, max(0.0, float(rank / len(reference))))

    def confidence(self, values: dict[str, float]) -> float:
        return self.confidence_from_probability(self.probability(values))


def selective_opportunity_enabled(config: Any) -> bool:
    return str(getattr(config, "strategy_mode", "")) in {SELECTIVE_ROTATION_MODE, OPTIMIZED_ALLOCATION_MODE}


def opportunity_features(
    utilities: np.ndarray,
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    timestamp: pd.Timestamp,
) -> tuple[dict[str, float], int] | None:
    ranked = sorted(
        (
            position
            for position in range(1, len(utilities))
            if np.isfinite(utilities[position])
        ),
        key=lambda position: (-float(utilities[position]), symbols[position - 1]),
    )
    if not ranked:
        return None

    best_position = ranked[0]
    best_score = float(utilities[best_position])
    second_score = float(utilities[ranked[1]]) if len(ranked) > 1 else best_score
    finite_scores = np.asarray([float(utilities[position]) for position in ranked], dtype=float)
    score_mean = float(np.mean(finite_scores))
    score_std = float(np.std(finite_scores))
    best_z = float((best_score - score_mean) / score_std) if score_std > 1e-12 else 0.0
    positive_fraction = float(np.mean(finite_scores > 0.0))

    best_symbol = symbols[best_position - 1]
    best_frame = frames.get(best_symbol)
    if best_frame is None or timestamp not in best_frame.index:
        return None
    row = best_frame.loc[timestamp]

    breadth_5_values: list[float] = []
    breadth_20_values: list[float] = []
    for symbol in symbols:
        frame = frames.get(symbol)
        if frame is None or timestamp not in frame.index:
            continue
        current = frame.loc[timestamp]
        value_5 = float(current.get("return_5", float("nan")))
        value_20 = float(current.get("return_20", float("nan")))
        if np.isfinite(value_5):
            breadth_5_values.append(value_5)
        if np.isfinite(value_20):
            breadth_20_values.append(value_20)

    values = {
        "best_score": best_score,
        "second_score": second_score,
        "best_vs_second_gap": float(best_score - second_score),
        "universe_score_mean": score_mean,
        "universe_score_std": score_std,
        "best_score_zscore": best_z,
        "positive_score_fraction": positive_fraction,
        "best_return_5": float(row.get("return_5", float("nan"))),
        "best_return_20": float(row.get("return_20", float("nan"))),
        "best_return_60": float(row.get("return_60", float("nan"))),
        "best_vol_20": float(row.get("vol_20", float("nan"))),
        "best_vol_60": float(row.get("vol_60", float("nan"))),
        "best_trend_efficiency_20": float(row.get("trend_efficiency_20", float("nan"))),
        "best_trend_efficiency_60": float(row.get("trend_efficiency_60", float("nan"))),
        "universe_breadth_5": float(np.mean(np.asarray(breadth_5_values) > 0.0)) if breadth_5_values else float("nan"),
        "universe_breadth_20": float(np.mean(np.asarray(breadth_20_values) > 0.0)) if breadth_20_values else float("nan"),
    }
    if not all(np.isfinite(values[name]) for name in OPPORTUNITY_FEATURES):
        return None
    return values, best_position


def build_opportunity_samples(
    models: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    dates: pd.DatetimeIndex,
    utilities_for_timestamp: Callable[[dict[str, Any], dict[str, pd.DataFrame], list[str], pd.Timestamp], np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for timestamp in dates:
        ts = pd.Timestamp(timestamp)
        utilities = utilities_for_timestamp(models, frames, symbols, ts)
        built = opportunity_features(utilities, frames, symbols, ts)
        if built is None:
            continue
        features, best_position = built
        symbol = symbols[best_position - 1]
        target = float(frames[symbol].loc[ts].get("forward_net_log_return", float("nan")))
        if not np.isfinite(target):
            continue
        rows.append(
            {
                "timestamp": ts,
                **features,
                "realized_net_log_return": target,
                "label": int(target > 0.0),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["timestamp", *OPPORTUNITY_FEATURES, "realized_net_log_return", "label"])
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def _fit_classifier(samples: pd.DataFrame, random_state: int) -> Any | None:
    labels = samples["label"].astype(int)
    if labels.nunique() < 2:
        return None
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(max_iter=2000, random_state=int(random_state))),
        ]
    )
    model.fit(samples[list(OPPORTUNITY_FEATURES)], labels)
    return model


def _probabilities(model: Any | None, constant: float | None, samples: pd.DataFrame) -> np.ndarray:
    if samples.empty:
        return np.asarray([], dtype=float)
    if model is None:
        return np.full(len(samples), float(constant or 0.0), dtype=float)
    return np.asarray(model.predict_proba(samples[list(OPPORTUNITY_FEATURES)])[:, 1], dtype=float)


def _relative_confidence(probability: float, reference_probabilities: np.ndarray, *, model_is_constant: bool = False) -> float:
    value = min(1.0, max(0.0, float(probability)))
    finite = np.sort(reference_probabilities[np.isfinite(reference_probabilities)])
    if model_is_constant or not len(finite):
        return value
    rank = int(np.searchsorted(finite, value, side="right"))
    return min(1.0, max(0.0, float(rank / len(finite))))


def _prequential_validation(samples: pd.DataFrame, random_state: int, label_horizon: int) -> pd.DataFrame:
    gap = max(1, int(label_horizon))
    minimum_training_rows = max(24, len(OPPORTUNITY_FEATURES) * 2)
    first_validation_index = gap + minimum_training_rows
    if first_validation_index >= len(samples):
        return pd.DataFrame(columns=["timestamp", "probability", "confidence", "realized_net_log_return", "label"])

    rows: list[dict[str, Any]] = []
    for index in range(first_validation_index, len(samples)):
        training_end = index - gap
        training = samples.iloc[:training_end]
        current = samples.iloc[[index]]
        model = _fit_classifier(training, int(random_state))
        constant = float(training["label"].mean()) if model is None else None
        probability = float(_probabilities(model, constant, current)[0])
        reference_probabilities = _probabilities(model, constant, training)
        confidence = _relative_confidence(
            probability,
            reference_probabilities,
            model_is_constant=model is None,
        )
        rows.append(
            {
                "timestamp": current.iloc[0]["timestamp"],
                "probability": probability,
                "confidence": confidence,
                "realized_net_log_return": float(current.iloc[0]["realized_net_log_return"]),
                "label": int(current.iloc[0]["label"]),
            }
        )
    return pd.DataFrame(rows)


def _confidence_threshold_candidates() -> tuple[float, ...]:
    return tuple(float(value) for value in np.linspace(0.0, 0.90, 19))


def _calibrate_confidence_threshold(validation: pd.DataFrame) -> tuple[float, float, int]:
    if validation.empty:
        return 0.5, float("nan"), 0

    minimum_accepted = max(8, int(np.ceil(np.sqrt(len(validation)))))
    realized = validation["realized_net_log_return"].to_numpy(dtype=float)
    confidence = validation["confidence"].to_numpy(dtype=float)
    best_threshold = 0.0
    best_score = float("-inf")
    best_accepted = len(validation)

    for threshold in _confidence_threshold_candidates():
        accepted = confidence >= float(threshold)
        accepted_count = int(np.sum(accepted))
        if accepted_count < minimum_accepted:
            continue
        accepted_returns = realized[accepted]
        total = float(np.sum(accepted_returns))
        uncertainty = (
            float(np.std(accepted_returns, ddof=1) * np.sqrt(accepted_count))
            if accepted_count > 1
            else 0.0
        )
        score = total - uncertainty
        if score > best_score + 1e-12 or (
            abs(score - best_score) <= 1e-12 and float(threshold) < best_threshold
        ):
            best_threshold = float(threshold)
            best_score = float(score)
            best_accepted = accepted_count

    if not np.isfinite(best_score):
        return 0.0, float(np.sum(realized)), len(validation)
    return best_threshold, best_score, best_accepted


def fit_selective_opportunity_gate(
    models: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    calibration_dates: pd.DatetimeIndex,
    utilities_for_timestamp: Callable[[dict[str, Any], dict[str, pd.DataFrame], list[str], pd.Timestamp], np.ndarray],
    *,
    random_state: int,
    label_horizon: int,
) -> SelectiveOpportunityGate:
    samples = build_opportunity_samples(
        models,
        frames,
        symbols,
        calibration_dates,
        utilities_for_timestamp,
    )
    if len(samples) < 30:
        raise ValueError(
            "Selective Opportunity requires at least 30 valid calibration decisions; "
            f"only {len(samples)} are available."
        )

    validation = _prequential_validation(samples, int(random_state), int(label_horizon))
    best_threshold, best_score, best_accepted = _calibrate_confidence_threshold(validation)

    final_model = _fit_classifier(samples, int(random_state))
    constant_probability = float(samples["label"].mean()) if final_model is None else None
    reference_probabilities = (
        tuple(float(value) for value in np.sort(_probabilities(final_model, constant_probability, samples)))
        if final_model is not None
        else ()
    )
    if final_model is None:
        best_threshold = 0.5

    return SelectiveOpportunityGate(
        model=final_model,
        threshold=float(best_threshold),
        constant_probability=constant_probability,
        training_rows=int(len(samples)),
        positive_rate=float(samples["label"].mean()),
        threshold_validation_rows=int(len(validation)),
        threshold_validation_score=float(best_score),
        reference_probabilities=reference_probabilities,
        threshold_validation_accepted=int(best_accepted),
        calibration_method="prequential_relative_confidence_v2",
    )


def evaluate_opportunity(
    gate: SelectiveOpportunityGate,
    utilities: np.ndarray,
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    timestamp: pd.Timestamp,
) -> OpportunityEvaluation | None:
    built = opportunity_features(utilities, frames, symbols, timestamp)
    if built is None:
        return None
    features, best_position = built
    probability = gate.probability(features)
    confidence = gate.confidence_from_probability(probability)
    return OpportunityEvaluation(
        probability=float(probability),
        confidence=float(confidence),
        accepted=bool(confidence >= gate.threshold),
        features=features,
        best_position=int(best_position),
    )
