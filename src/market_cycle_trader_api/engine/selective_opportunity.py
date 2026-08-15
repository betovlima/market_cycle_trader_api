from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple

import numpy as np
import pandas as pd

SELECTIVE_ROTATION_MODE = "COMPOUND_ROTATION_SWING_SELECTIVE"
OPPORTUNITY_CASH_GATE_MODE = "COMPOUND_ROTATION_SWING_OPPORTUNITY_CASH_GATE"
OPTIMIZED_ALLOCATION_MODE = "COMPOUND_ROTATION_SWING_OPTIMIZED_ALLOCATION"
CONCENTRATED_ALLOCATION_MODE = "COMPOUND_ROTATION_SWING_CONCENTRATED_ALLOCATION"

CASH_GATE_V2_REFRESH_SESSIONS = 21
CASH_GATE_V2_ROLLING_SAMPLE_WINDOW = 252
CASH_GATE_V2_MIN_VALIDATION_ROWS = 60
CASH_GATE_V2_MIN_MARKET_EXPOSURE_RATIO = 0.60

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
    entry_threshold: float | None = None
    exit_threshold: float | None = None
    threshold_validation_transitions: int = 0
    threshold_basis: str = "relative_confidence"
    target_basis: str = "weighted_forward_net_log_return"
    target_horizon_sessions: int | None = None
    regularized_to_base_policy: bool = False
    threshold_validation_alpha: float | None = None
    threshold_validation_exposure_ratio: float | None = None

    def active_threshold(self, current_position: int | None = None) -> float:
        if self.entry_threshold is None or self.exit_threshold is None or current_position is None:
            return float(self.threshold)
        return float(self.exit_threshold if int(current_position) > 0 else self.entry_threshold)

    def decision_value(self, probability: float, confidence: float) -> float:
        if self.threshold_basis == "absolute_probability":
            return min(1.0, max(0.0, float(probability)))
        return min(1.0, max(0.0, float(confidence)))

    def accepts(
        self,
        probability: float,
        confidence: float,
        current_position: int | None = None,
    ) -> bool:
        return bool(
            self.decision_value(probability, confidence) >= self.active_threshold(current_position)
        )

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


@dataclass
class AdaptiveOpportunityCashGate:
    """Online, no-look-ahead CASH gate for the protected B0 policy.

    The utility/ranking model remains frozen for each walk-forward fold.  Only
    this small logistic gate is refreshed as one-session B0 outcomes mature.
    A shared OOS history lets later folds learn from earlier *realized* OOS
    decisions without peeking at future outcomes.
    """

    initial_samples: pd.DataFrame
    shared_history: list[dict[str, Any]]
    random_state: int
    fold_id: int | None = None
    refresh_interval: int = CASH_GATE_V2_REFRESH_SESSIONS
    rolling_window: int = CASH_GATE_V2_ROLLING_SAMPLE_WINDOW
    gate: SelectiveOpportunityGate | None = None
    last_refit_history_count: int = 0
    refresh_count: int = 0

    def __post_init__(self) -> None:
        self.gate = _fit_cash_gate_v2_from_samples(self.initial_samples, self.random_state)
        self.last_refit_history_count = len(self.shared_history)

    def _combined_samples(self) -> pd.DataFrame:
        history = pd.DataFrame(self.shared_history)
        pieces = [frame for frame in (self.initial_samples, history) if not frame.empty]
        if not pieces:
            return pd.DataFrame(columns=["timestamp", *OPPORTUNITY_FEATURES, "realized_net_log_return", "label"])
        combined = pd.concat(pieces, ignore_index=True)
        combined = combined.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
        if len(combined) > int(self.rolling_window):
            combined = combined.iloc[-int(self.rolling_window):]
        return combined.reset_index(drop=True)

    def record_matured_sample(self, sample: dict[str, Any]) -> None:
        self.shared_history.append(dict(sample))

    def refresh_if_needed(self, *, force: bool = False) -> bool:
        history_count = len(self.shared_history)
        if not force and history_count - self.last_refit_history_count < int(self.refresh_interval):
            return False
        combined = self._combined_samples()
        if len(combined) < 30:
            return False
        self.gate = _fit_cash_gate_v2_from_samples(combined, self.random_state)
        self.last_refit_history_count = history_count
        self.refresh_count += 1
        return True

    @property
    def current(self) -> SelectiveOpportunityGate:
        if self.gate is None:
            raise RuntimeError("Adaptive Opportunity Cash Gate has not been fitted.")
        return self.gate

    # Delegate the policy-facing API so existing evaluation code stays simple.
    @property
    def threshold(self) -> float:
        return float(self.current.threshold)

    @property
    def entry_threshold(self) -> float | None:
        return self.current.entry_threshold

    @property
    def exit_threshold(self) -> float | None:
        return self.current.exit_threshold

    @property
    def threshold_basis(self) -> str:
        return self.current.threshold_basis

    @property
    def calibration_method(self) -> str:
        return self.current.calibration_method

    @property
    def training_rows(self) -> int:
        return int(self.current.training_rows)

    @property
    def positive_rate(self) -> float:
        return float(self.current.positive_rate)

    @property
    def threshold_validation_rows(self) -> int:
        return int(self.current.threshold_validation_rows)

    @property
    def threshold_validation_score(self) -> float:
        return float(self.current.threshold_validation_score)

    @property
    def threshold_validation_accepted(self) -> int:
        return int(self.current.threshold_validation_accepted)

    @property
    def threshold_validation_transitions(self) -> int:
        return int(self.current.threshold_validation_transitions)

    @property
    def target_basis(self) -> str:
        return self.current.target_basis

    @property
    def target_horizon_sessions(self) -> int | None:
        return self.current.target_horizon_sessions

    @property
    def regularized_to_base_policy(self) -> bool:
        return bool(self.current.regularized_to_base_policy)

    @property
    def threshold_validation_alpha(self) -> float | None:
        return self.current.threshold_validation_alpha

    @property
    def threshold_validation_exposure_ratio(self) -> float | None:
        return self.current.threshold_validation_exposure_ratio

    def active_threshold(self, current_position: int | None = None) -> float:
        return self.current.active_threshold(current_position)

    def decision_value(self, probability: float, confidence: float) -> float:
        return self.current.decision_value(probability, confidence)

    def accepts(self, probability: float, confidence: float, current_position: int | None = None) -> bool:
        return self.current.accepts(probability, confidence, current_position)

    def probability(self, values: dict[str, float]) -> float:
        return self.current.probability(values)

    def confidence_from_probability(self, probability: float) -> float:
        return self.current.confidence_from_probability(probability)

    def confidence(self, values: dict[str, float]) -> float:
        return self.current.confidence(values)


def opportunity_cash_gate_enabled(config: Any) -> bool:
    return str(getattr(config, "strategy_mode", "")) == OPPORTUNITY_CASH_GATE_MODE


def selective_opportunity_enabled(config: Any) -> bool:
    return str(getattr(config, "strategy_mode", "")) in {
        SELECTIVE_ROTATION_MODE,
        OPPORTUNITY_CASH_GATE_MODE,
        OPTIMIZED_ALLOCATION_MODE,
        CONCENTRATED_ALLOCATION_MODE,
    }


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



def build_base_policy_opportunity_samples(
    models: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    dates: pd.DatetimeIndex,
    utilities_for_timestamp: Callable[[dict[str, Any], dict[str, pd.DataFrame], list[str], pd.Timestamp], np.ndarray],
    base_policy: Callable[[pd.Timestamp, int, int], tuple[int, float]],
    realized_action_return: Callable[[pd.Timestamp, pd.Timestamp, int, int], float],
) -> pd.DataFrame:
    """Build one-session labels for the *protected B0 action*.

    The decision at close(t) can only control exposure after the next open, so
    the v2 target intentionally avoids the old 5–60 session utility label.  It
    asks whether following B0's risky action during the next execution session
    produced positive net log growth versus holding dollars.
    """
    rows: list[dict[str, Any]] = []
    position = 0
    holding_days = 0
    ordered = pd.DatetimeIndex(dates)
    for index in range(max(0, len(ordered) - 1)):
        timestamp = pd.Timestamp(ordered[index])
        next_timestamp = pd.Timestamp(ordered[index + 1])
        utilities = utilities_for_timestamp(models, frames, symbols, timestamp)
        built = opportunity_features(utilities, frames, symbols, timestamp)
        position_before = int(position)
        action, _ = base_policy(timestamp, position_before, holding_days)
        action = int(action)
        if built is not None and action > 0:
            features, _ = built
            target = float(realized_action_return(timestamp, next_timestamp, position_before, action))
            if np.isfinite(target):
                rows.append(
                    {
                        "timestamp": timestamp,
                        **features,
                        "realized_net_log_return": target,
                        "label": int(target > 0.0),
                    }
                )
        if action == position_before:
            holding_days = holding_days + 1 if action > 0 else 0
        else:
            position = action
            holding_days = 1 if action > 0 else 0
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


def _calibrate_hysteresis_thresholds(
    validation: pd.DataFrame,
    *,
    signal_column: str = "probability",
) -> tuple[float, float, float, int, int]:
    """Calibrate a stateful CASH↔MARKET gate without using test-period data.

    Entry and exit thresholds are selected only from the requested prequential
    signal.  Opportunity Cash Gate uses absolute P(positive net return), while
    legacy Selective keeps its relative-confidence calibration unchanged.  The
    constraint exit <= entry creates hysteresis. Transition count is only a
    deterministic tie-breaker so churn is never rewarded by the calibration.
    """
    if validation.empty:
        return 0.5, 0.5, float("nan"), 0, 0
    if signal_column not in validation.columns:
        raise ValueError(f"Opportunity hysteresis signal column is missing: {signal_column}")

    realized = validation["realized_net_log_return"].to_numpy(dtype=float)
    signal = validation[signal_column].to_numpy(dtype=float)
    minimum_exposed = max(8, int(np.ceil(np.sqrt(len(validation)))))

    best_entry = 0.0
    best_exit = 0.0
    best_score = float("-inf")
    best_exposed = len(validation)
    best_transitions = 0

    candidates = _confidence_threshold_candidates()
    for entry_threshold in candidates:
        for exit_threshold in candidates:
            if float(exit_threshold) > float(entry_threshold) + 1e-12:
                continue

            invested = False
            exposed = np.zeros(len(validation), dtype=bool)
            transitions = 0
            for index, value in enumerate(signal):
                previous = invested
                if invested:
                    if float(value) < float(exit_threshold):
                        invested = False
                elif float(value) >= float(entry_threshold):
                    invested = True
                if invested != previous:
                    transitions += 1
                exposed[index] = invested

            exposed_count = int(np.sum(exposed))
            if exposed_count < minimum_exposed:
                continue
            accepted_returns = realized[exposed]
            total = float(np.sum(accepted_returns))
            uncertainty = (
                float(np.std(accepted_returns, ddof=1) * np.sqrt(exposed_count))
                if exposed_count > 1
                else 0.0
            )
            score = total - uncertainty
            better = score > best_score + 1e-12
            tied = abs(score - best_score) <= 1e-12
            if better or (
                tied
                and (
                    transitions < best_transitions
                    or (transitions == best_transitions and float(entry_threshold) < best_entry)
                    or (
                        transitions == best_transitions
                        and abs(float(entry_threshold) - best_entry) <= 1e-12
                        and float(exit_threshold) > best_exit
                    )
                )
            ):
                best_entry = float(entry_threshold)
                best_exit = float(exit_threshold)
                best_score = float(score)
                best_exposed = exposed_count
                best_transitions = int(transitions)

    if not np.isfinite(best_score):
        total = float(np.sum(realized))
        return 0.0, 0.0, total, len(validation), 0
    return best_entry, best_exit, best_score, best_exposed, best_transitions



def _cash_gate_v2_threshold_candidates() -> tuple[float, ...]:
    # 0.0 is the protected B0 fallback (always allow market exposure).
    return (0.0, *tuple(float(value) for value in np.arange(0.30, 0.751, 0.05)))


def _calibrate_cash_gate_v2_thresholds(
    validation: pd.DataFrame,
) -> tuple[float, float, float, int, int, float, float, bool]:
    """Select CASH interventions only when validation evidence beats B0.

    Score is incremental log-growth of the gate versus always following B0,
    penalized by uncertainty.  At least 60% of validation sessions must remain
    exposed.  If no candidate has positive conservative alpha, thresholds are
    forced to zero so B0 remains fully in control.
    """
    if len(validation) < CASH_GATE_V2_MIN_VALIDATION_ROWS:
        realized = validation.get("realized_net_log_return", pd.Series(dtype=float)).to_numpy(dtype=float)
        total = float(np.sum(realized)) if len(realized) else float("nan")
        return 0.0, 0.0, 0.0, len(validation), 0, 0.0, 1.0, True

    realized = validation["realized_net_log_return"].to_numpy(dtype=float)
    probabilities = validation["probability"].to_numpy(dtype=float)
    count = len(validation)
    minimum_exposed = max(24, int(np.ceil(CASH_GATE_V2_MIN_MARKET_EXPOSURE_RATIO * count)))

    best: tuple[float, float, float, int, int, float, float] | None = None
    candidates = _cash_gate_v2_threshold_candidates()
    for entry_threshold in candidates:
        for exit_threshold in candidates:
            if float(exit_threshold) > float(entry_threshold) + 1e-12:
                continue
            invested = False
            exposed = np.zeros(count, dtype=bool)
            transitions = 0
            for index, probability in enumerate(probabilities):
                previous = invested
                if invested:
                    if float(probability) < float(exit_threshold):
                        invested = False
                elif float(probability) >= float(entry_threshold):
                    invested = True
                if invested != previous:
                    transitions += 1
                exposed[index] = invested

            exposed_count = int(np.sum(exposed))
            if exposed_count < minimum_exposed:
                continue
            cash_count = count - exposed_count
            intervention = np.where(exposed, 0.0, -realized)
            alpha = float(np.sum(intervention))
            uncertainty = (
                float(np.std(intervention, ddof=1) * np.sqrt(count))
                if count > 1
                else 0.0
            )
            conservative_score = float(alpha - 0.50 * uncertainty)
            exposure_ratio = float(exposed_count / count)
            candidate = (
                float(entry_threshold),
                float(exit_threshold),
                conservative_score,
                exposed_count,
                int(transitions),
                alpha,
                exposure_ratio,
            )
            if best is None:
                best = candidate
                continue
            _, _, best_score, best_exposed, best_transitions, best_alpha, _ = best
            better = conservative_score > best_score + 1e-12
            tied = abs(conservative_score - best_score) <= 1e-12
            if better or (
                tied
                and (
                    alpha > best_alpha + 1e-12
                    or (abs(alpha - best_alpha) <= 1e-12 and transitions < best_transitions)
                    or (
                        abs(alpha - best_alpha) <= 1e-12
                        and transitions == best_transitions
                        and exposed_count > best_exposed
                    )
                )
            ):
                best = candidate

    if best is None:
        return 0.0, 0.0, 0.0, count, 0, 0.0, 1.0, True
    entry, exit, score, exposed, transitions, alpha, exposure_ratio = best
    # B0 is the statistical prior.  CASH must earn the right to intervene.
    if score <= 0.0 or alpha <= 0.0:
        return 0.0, 0.0, 0.0, count, 0, 0.0, 1.0, True
    return entry, exit, score, exposed, transitions, alpha, exposure_ratio, False


def _fit_cash_gate_v2_from_samples(samples: pd.DataFrame, random_state: int) -> SelectiveOpportunityGate:
    if len(samples) < 30:
        # Conservative fail-open behavior: preserve B0, never default to CASH.
        return SelectiveOpportunityGate(
            model=None,
            threshold=0.0,
            constant_probability=1.0,
            training_rows=int(len(samples)),
            positive_rate=float(samples["label"].mean()) if len(samples) else float("nan"),
            threshold_validation_rows=0,
            threshold_validation_score=0.0,
            entry_threshold=0.0,
            exit_threshold=0.0,
            calibration_method="adaptive_base_policy_one_step_hysteresis_v2",
            threshold_basis="absolute_probability",
            target_basis="protected_base_policy_next_session_open_to_close_net_log_return",
            target_horizon_sessions=1,
            regularized_to_base_policy=True,
            threshold_validation_alpha=0.0,
            threshold_validation_exposure_ratio=1.0,
        )

    validation = _prequential_validation(samples, int(random_state), 1)
    entry, exit, score, exposed, transitions, alpha, exposure_ratio, regularized = (
        _calibrate_cash_gate_v2_thresholds(validation)
    )
    final_model = _fit_classifier(samples, int(random_state))
    constant_probability = float(samples["label"].mean()) if final_model is None else None
    reference_probabilities = (
        tuple(float(value) for value in np.sort(_probabilities(final_model, constant_probability, samples)))
        if final_model is not None
        else ()
    )
    # A constant classifier carries no cross-sectional/regime information.
    if final_model is None:
        entry = exit = 0.0
        regularized = True
        alpha = 0.0
        exposure_ratio = 1.0

    return SelectiveOpportunityGate(
        model=final_model,
        threshold=float(entry),
        constant_probability=constant_probability,
        training_rows=int(len(samples)),
        positive_rate=float(samples["label"].mean()),
        threshold_validation_rows=int(len(validation)),
        threshold_validation_score=float(score),
        reference_probabilities=reference_probabilities,
        threshold_validation_accepted=int(exposed),
        entry_threshold=float(entry),
        exit_threshold=float(exit),
        threshold_validation_transitions=int(transitions),
        calibration_method="adaptive_base_policy_one_step_hysteresis_v2",
        threshold_basis="absolute_probability",
        target_basis="protected_base_policy_next_session_open_to_close_net_log_return",
        target_horizon_sessions=1,
        regularized_to_base_policy=bool(regularized),
        threshold_validation_alpha=float(alpha),
        threshold_validation_exposure_ratio=float(exposure_ratio),
    )


def fit_adaptive_opportunity_cash_gate(
    initial_samples: pd.DataFrame,
    *,
    random_state: int,
    shared_history: list[dict[str, Any]],
    fold_id: int | None = None,
    refresh_interval: int = CASH_GATE_V2_REFRESH_SESSIONS,
    rolling_window: int = CASH_GATE_V2_ROLLING_SAMPLE_WINDOW,
) -> AdaptiveOpportunityCashGate:
    return AdaptiveOpportunityCashGate(
        initial_samples=initial_samples.copy(),
        shared_history=shared_history,
        random_state=int(random_state),
        fold_id=fold_id,
        refresh_interval=int(refresh_interval),
        rolling_window=int(rolling_window),
    )


def fit_selective_opportunity_gate(
    models: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    calibration_dates: pd.DatetimeIndex,
    utilities_for_timestamp: Callable[[dict[str, Any], dict[str, pd.DataFrame], list[str], pd.Timestamp], np.ndarray],
    *,
    random_state: int,
    label_horizon: int,
    hysteresis: bool = False,
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
    if hysteresis:
        entry_threshold, exit_threshold, best_score, best_accepted, transition_count = (
            _calibrate_hysteresis_thresholds(validation, signal_column="probability")
        )
        best_threshold = float(entry_threshold)
    else:
        best_threshold, best_score, best_accepted = _calibrate_confidence_threshold(validation)
        entry_threshold = None
        exit_threshold = None
        transition_count = 0

    final_model = _fit_classifier(samples, int(random_state))
    constant_probability = float(samples["label"].mean()) if final_model is None else None
    reference_probabilities = (
        tuple(float(value) for value in np.sort(_probabilities(final_model, constant_probability, samples)))
        if final_model is not None
        else ()
    )
    if final_model is None:
        best_threshold = 0.5
        if hysteresis:
            entry_threshold = 0.5
            exit_threshold = 0.5

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
        calibration_method=(
            "prequential_absolute_probability_hysteresis_v1"
            if hysteresis
            else "prequential_relative_confidence_v2"
        ),
        entry_threshold=float(entry_threshold) if entry_threshold is not None else None,
        exit_threshold=float(exit_threshold) if exit_threshold is not None else None,
        threshold_validation_transitions=int(transition_count),
        threshold_basis=("absolute_probability" if hysteresis else "relative_confidence"),
        target_basis="weighted_forward_net_log_return",
        target_horizon_sessions=int(label_horizon),
    )


def evaluate_opportunity(
    gate: SelectiveOpportunityGate,
    utilities: np.ndarray,
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    timestamp: pd.Timestamp,
    *,
    current_position: int | None = None,
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
        accepted=gate.accepts(probability, confidence, current_position),
        features=features,
        best_position=int(best_position),
    )
