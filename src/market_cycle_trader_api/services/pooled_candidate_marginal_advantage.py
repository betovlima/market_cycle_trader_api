from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler


@dataclass
class PooledCandidateMarginalModel:
    feature_names: list[str]
    scaler: StandardScaler
    regressor: BayesianRidge
    training_rows: int
    training_decision_dates: int

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        matrix = frame[self.feature_names].to_numpy(dtype=float)
        scaled = self.scaler.transform(matrix)
        mean, std = self.regressor.predict(scaled, return_std=True)
        return np.asarray(mean, dtype=float), np.asarray(std, dtype=float)

    def diagnostics(self) -> dict[str, Any]:
        coefficients = np.asarray(self.regressor.coef_, dtype=float)
        return {
            "training_rows": int(self.training_rows),
            "training_decision_dates": int(self.training_decision_dates),
            "feature_count": int(len(self.feature_names)),
            "posterior_noise_precision": float(self.regressor.alpha_),
            "posterior_weight_precision": float(self.regressor.lambda_),
            "coefficient_l2_norm": float(np.linalg.norm(coefficients)),
            "intercept": float(self.regressor.intercept_),
        }


def _finite_training_frame(
    samples: pd.DataFrame,
    feature_names: Iterable[str],
    *,
    target_column: str,
) -> pd.DataFrame:
    features = list(feature_names)
    required = ["fold", "timestamp", "candidate", target_column, *features]
    missing = [column for column in required if column not in samples.columns]
    if missing:
        raise RuntimeError(
            "Pooled candidate-advantage samples are missing required columns: "
            + ", ".join(missing)
        )

    frame = samples[required].copy()
    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"], utc=True, format="mixed", errors="coerce"
    )
    frame[target_column] = pd.to_numeric(frame[target_column], errors="coerce")
    for column in features:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    finite = np.isfinite(frame[target_column].to_numpy(dtype=float))
    for column in features:
        finite &= np.isfinite(frame[column].to_numpy(dtype=float))
    finite &= frame["timestamp"].notna().to_numpy(dtype=bool)
    frame = frame.loc[finite].copy()
    if frame.empty:
        raise RuntimeError("No finite pooled candidate-advantage training samples remain.")
    return frame.reset_index(drop=True)


def fit_pooled_candidate_marginal_model(
    samples: pd.DataFrame,
    feature_names: Iterable[str],
    *,
    target_column: str = "marginal_next_session_net_log_return",
) -> PooledCandidateMarginalModel:
    features = list(feature_names)
    frame = _finite_training_frame(
        samples,
        features,
        target_column=target_column,
    )

    group_sizes = frame.groupby(["fold", "timestamp"])["candidate"].transform("count")
    weights = 1.0 / group_sizes.astype(float).clip(lower=1.0)

    matrix = frame[features].to_numpy(dtype=float)
    target = frame[target_column].to_numpy(dtype=float)
    sample_weight = weights.to_numpy(dtype=float)

    scaler = StandardScaler()
    scaler.fit(matrix, sample_weight=sample_weight)
    scaled = scaler.transform(matrix)
    regressor = BayesianRidge(compute_score=True)
    regressor.fit(scaled, target, sample_weight=sample_weight)

    return PooledCandidateMarginalModel(
        feature_names=features,
        scaler=scaler,
        regressor=regressor,
        training_rows=int(len(frame)),
        training_decision_dates=int(frame[["fold", "timestamp"]].drop_duplicates().shape[0]),
    )


def score_candidate_samples(
    model: PooledCandidateMarginalModel,
    samples: pd.DataFrame,
    *,
    target_column: str = "marginal_next_session_net_log_return",
) -> pd.DataFrame:
    frame = _finite_training_frame(
        samples,
        model.feature_names,
        target_column=target_column,
    )
    mean, std = model.predict(frame)
    frame["predicted_marginal_advantage"] = mean
    frame["predicted_marginal_std"] = std
    return frame


def choose_daily_candidate_overrides(
    scored_samples: pd.DataFrame,
    *,
    target_column: str = "marginal_next_session_net_log_return",
) -> pd.DataFrame:
    required = {
        "fold",
        "timestamp",
        "candidate",
        target_column,
        "predicted_marginal_advantage",
        "predicted_marginal_std",
    }
    missing = sorted(required.difference(scored_samples.columns))
    if missing:
        raise RuntimeError(
            "Scored candidate samples are missing required columns: "
            + ", ".join(missing)
        )

    rows: list[dict[str, Any]] = []
    for (fold, timestamp), group in scored_samples.groupby(
        ["fold", "timestamp"], sort=True
    ):
        ordered = group.sort_values(
            ["predicted_marginal_advantage", "candidate"],
            ascending=[False, True],
        )
        best = ordered.iloc[0]
        predicted = float(best["predicted_marginal_advantage"])
        if np.isfinite(predicted) and predicted > 0.0:
            chosen_candidate = str(best["candidate"])
            realized = float(best[target_column])
            chosen_std = float(best["predicted_marginal_std"])
            override = True
        else:
            chosen_candidate = "BASELINE"
            realized = 0.0
            chosen_std = float(best["predicted_marginal_std"])
            override = False

        oracle = pd.to_numeric(group[target_column], errors="coerce")
        oracle_value = float(max(0.0, oracle.max())) if oracle.notna().any() else 0.0
        rows.append(
            {
                "fold": int(fold),
                "timestamp": pd.Timestamp(timestamp),
                "chosen_candidate": chosen_candidate,
                "override_baseline": bool(override),
                "predicted_marginal_advantage": predicted,
                "predicted_marginal_std": chosen_std,
                "realized_marginal_next_session_net_log_return": realized,
                "oracle_best_positive_marginal_next_session_net_log_return": oracle_value,
                "candidate_count": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values(["fold", "timestamp"]).reset_index(drop=True)


def summarize_override_decisions(decisions: pd.DataFrame) -> dict[str, Any]:
    if decisions.empty:
        return {
            "decision_dates": 0,
            "override_count": 0,
            "override_rate": 0.0,
            "realized_marginal_log_sum": 0.0,
            "realized_marginal_log_mean_when_overridden": None,
            "positive_realized_override_rate": None,
            "oracle_positive_marginal_log_sum": 0.0,
        }

    realized = pd.to_numeric(
        decisions["realized_marginal_next_session_net_log_return"], errors="coerce"
    ).fillna(0.0)
    override_mask = decisions["override_baseline"].astype(bool)
    override_realized = realized.loc[override_mask]
    oracle = pd.to_numeric(
        decisions["oracle_best_positive_marginal_next_session_net_log_return"],
        errors="coerce",
    ).fillna(0.0)
    return {
        "decision_dates": int(len(decisions)),
        "override_count": int(override_mask.sum()),
        "override_rate": float(override_mask.mean()),
        "realized_marginal_log_sum": float(realized.sum()),
        "realized_marginal_log_mean_when_overridden": (
            float(override_realized.mean()) if not override_realized.empty else None
        ),
        "positive_realized_override_rate": (
            float((override_realized > 0.0).mean()) if not override_realized.empty else None
        ),
        "oracle_positive_marginal_log_sum": float(oracle.sum()),
    }
