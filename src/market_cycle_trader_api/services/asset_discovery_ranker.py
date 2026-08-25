from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd

FEATURE_COLUMNS = (
    "return_5",
    "return_10",
    "return_20",
    "return_40",
    "return_60",
    "volatility_10",
    "volatility_20",
    "volatility_60",
    "drawdown_20",
    "drawdown_60",
    "position_60",
    "trend_efficiency_20",
    "volume_ratio_20",
    "dollar_volume_log",
)
TARGET_HORIZON = 20
MIN_FEATURE_SESSIONS = 126
VALIDATION_FOLDS = 4
INITIAL_TRAIN_FRACTION = 0.50
RANDOM_BASELINE_REPEATS = 24


@dataclass(frozen=True)
class RankerBundle:
    model: Any
    diagnostics: dict[str, Any]


def _clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    required = ["open", "high", "low", "close", "volume"]
    if any(column not in frame.columns for column in required):
        return pd.DataFrame()
    result = frame.copy().sort_index()
    result = result[~result.index.duplicated(keep="last")]
    for column in required:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    result = result[(result[["open", "high", "low", "close"]] > 0).all(axis=1)]
    result = result[result["volume"] >= 0]
    return result


def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    source = _clean_frame(frame)
    if source.empty:
        return pd.DataFrame(columns=list(FEATURE_COLUMNS))

    close = source["close"].astype(float)
    volume = source["volume"].astype(float)
    ret1 = close.pct_change()
    features = pd.DataFrame(index=source.index)
    for horizon in (5, 10, 20, 40, 60):
        features[f"return_{horizon}"] = close.pct_change(horizon)
    for horizon in (10, 20, 60):
        features[f"volatility_{horizon}"] = ret1.rolling(horizon).std(ddof=0) * sqrt(252.0)

    rolling_high_20 = close.rolling(20).max()
    rolling_high_60 = close.rolling(60).max()
    rolling_low_60 = close.rolling(60).min()
    features["drawdown_20"] = close / rolling_high_20 - 1.0
    features["drawdown_60"] = close / rolling_high_60 - 1.0
    spread_60 = (rolling_high_60 - rolling_low_60).replace(0.0, np.nan)
    features["position_60"] = (close - rolling_low_60) / spread_60

    gross_path_20 = ret1.abs().rolling(20).sum().replace(0.0, np.nan)
    features["trend_efficiency_20"] = close.pct_change(20).abs() / gross_path_20
    median_volume_20 = volume.rolling(20).median().replace(0.0, np.nan)
    features["volume_ratio_20"] = volume / median_volume_20
    features["dollar_volume_log"] = np.log1p((close * volume).clip(lower=0.0))
    return features.replace([np.inf, -np.inf], np.nan)


def _future_utility(frame: pd.DataFrame, horizon: int = TARGET_HORIZON) -> pd.Series:
    source = _clean_frame(frame)
    close = source["close"].astype(float)
    forward_return = close.shift(-horizon) / close - 1.0
    path = pd.concat(
        [(close.shift(-step) / close - 1.0).rename(step) for step in range(1, horizon + 1)],
        axis=1,
    )
    adverse_excursion = path.min(axis=1)
    return forward_return + 0.45 * adverse_excursion.clip(upper=0.0)


def build_training_dataset(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for symbol, frame in frames.items():
        features = feature_frame(frame)
        if features.empty:
            continue
        utility = _future_utility(frame).reindex(features.index)
        combined = features.copy()
        combined["utility"] = utility
        combined["symbol"] = str(symbol).upper()
        combined["date"] = pd.to_datetime(combined.index, utc=True).normalize()
        rows.append(combined.reset_index(drop=True))
    if not rows:
        return pd.DataFrame()

    dataset = pd.concat(rows, ignore_index=True)
    dataset = dataset.dropna(subset=[*FEATURE_COLUMNS, "utility", "date"])
    if dataset.empty:
        return dataset
    group_sizes = dataset.groupby("date")["symbol"].transform("count")
    dataset = dataset.loc[group_sizes >= 3].copy()
    if dataset.empty:
        return dataset
    percentile = dataset.groupby("date")["utility"].rank(method="average", pct=True)
    dataset["relevance"] = np.minimum(np.floor(percentile * 5.0), 4).astype(int)
    return dataset.sort_values(["date", "symbol"]).reset_index(drop=True)


def _group_sizes(dataset: pd.DataFrame) -> list[int]:
    if dataset.empty:
        return []
    return dataset.groupby("date", sort=False).size().astype(int).tolist()


def _average_ndcg(dataset: pd.DataFrame, predictions: np.ndarray, k: int = 5) -> float | None:
    if dataset.empty or len(dataset) != len(predictions):
        return None
    try:
        from sklearn.metrics import ndcg_score
    except ImportError:
        return None

    values: list[float] = []
    offset = 0
    for _, group in dataset.groupby("date", sort=False):
        size = len(group)
        if size < 2:
            offset += size
            continue
        truth = group["relevance"].to_numpy(dtype=float)
        pred = np.asarray(predictions[offset: offset + size], dtype=float)
        offset += size
        try:
            values.append(float(ndcg_score([truth], [pred], k=min(k, size))))
        except ValueError:
            continue
    return float(np.mean(values)) if values else None


def _new_ranker(random_state: int) -> Any:
    try:
        from lightgbm import LGBMRanker
    except ImportError as exc:
        raise RuntimeError("LightGBM is required for Asset Discovery Learning-to-Rank.") from exc

    return LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=320,
        learning_rate=0.035,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=30,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_alpha=0.05,
        reg_lambda=0.20,
        random_state=int(random_state),
        n_jobs=1,
        verbosity=-1,
    )


def _fit_ranker(dataset: pd.DataFrame, *, random_state: int) -> Any:
    model = _new_ranker(random_state)
    model.fit(
        dataset[list(FEATURE_COLUMNS)],
        dataset["relevance"],
        group=_group_sizes(dataset),
    )
    return model


def _random_baseline_ndcg(dataset: pd.DataFrame, *, random_state: int, repeats: int = RANDOM_BASELINE_REPEATS) -> float | None:
    if dataset.empty:
        return None
    rng = np.random.default_rng(int(random_state))
    values: list[float] = []
    for _ in range(max(1, int(repeats))):
        predictions = rng.random(len(dataset))
        score = _average_ndcg(dataset, predictions, 5)
        if score is not None:
            values.append(float(score))
    return float(np.mean(values)) if values else None


def _median(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.median(clean)) if clean else None


def _minimum(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(min(clean)) if clean else None


def _walk_forward_validation(dataset: pd.DataFrame, *, random_state: int) -> dict[str, Any]:
    unique_dates = list(pd.unique(dataset["date"])) if not dataset.empty else []
    if len(unique_dates) < 360:
        raise RuntimeError("Learning-to-Rank walk-forward validation requires at least 360 chronological sessions.")

    initial_train_sessions = max(180, int(len(unique_dates) * INITIAL_TRAIN_FRACTION))
    remaining_sessions = len(unique_dates) - initial_train_sessions
    if remaining_sessions < VALIDATION_FOLDS * 20:
        raise RuntimeError("Learning-to-Rank walk-forward validation does not have enough out-of-sample sessions.")

    boundaries = np.linspace(initial_train_sessions, len(unique_dates), VALIDATION_FOLDS + 1, dtype=int)
    folds: list[dict[str, Any]] = []
    for fold_index in range(VALIDATION_FOLDS):
        validation_start_index = int(boundaries[fold_index])
        validation_end_index = int(boundaries[fold_index + 1])
        validation_dates = unique_dates[validation_start_index:validation_end_index]
        purge_end_index = max(0, validation_start_index - TARGET_HORIZON)
        train_dates = unique_dates[:purge_end_index]
        if len(train_dates) < 180 or len(validation_dates) < 20:
            continue

        train = dataset[dataset["date"].isin(set(train_dates))].copy()
        validation = dataset[dataset["date"].isin(set(validation_dates))].copy()
        if train.empty or validation.empty:
            continue

        fold_seed = int(random_state) + fold_index + 1
        model = _fit_ranker(train, random_state=fold_seed)
        ranker_predictions = model.predict(validation[list(FEATURE_COLUMNS)])
        ranker_ndcg = _average_ndcg(validation, ranker_predictions, 5)
        momentum_ndcg = _average_ndcg(
            validation,
            validation["return_20"].to_numpy(dtype=float),
            5,
        )
        random_ndcg = _random_baseline_ndcg(validation, random_state=fold_seed + 10_000)
        beats_momentum = (
            ranker_ndcg is not None
            and momentum_ndcg is not None
            and float(ranker_ndcg) > float(momentum_ndcg)
        )
        folds.append(
            {
                "fold": fold_index + 1,
                "train_start": pd.Timestamp(train_dates[0]).date().isoformat(),
                "train_end": pd.Timestamp(train_dates[-1]).date().isoformat(),
                "validation_start": pd.Timestamp(validation_dates[0]).date().isoformat(),
                "validation_end": pd.Timestamp(validation_dates[-1]).date().isoformat(),
                "train_rows": int(len(train)),
                "train_sessions": int(train["date"].nunique()),
                "validation_rows": int(len(validation)),
                "validation_sessions": int(validation["date"].nunique()),
                "ranker_ndcg_at_5": ranker_ndcg,
                "momentum_ndcg_at_5": momentum_ndcg,
                "random_ndcg_at_5": random_ndcg,
                "beats_momentum": bool(beats_momentum),
            }
        )

    if len(folds) != VALIDATION_FOLDS:
        raise RuntimeError("Learning-to-Rank walk-forward validation could not build all chronological folds.")

    ranker_scores = [item.get("ranker_ndcg_at_5") for item in folds]
    momentum_scores = [item.get("momentum_ndcg_at_5") for item in folds]
    random_scores = [item.get("random_ndcg_at_5") for item in folds]
    comparable = [item for item in folds if item.get("ranker_ndcg_at_5") is not None and item.get("momentum_ndcg_at_5") is not None]
    wins = sum(1 for item in comparable if bool(item.get("beats_momentum")))
    return {
        "method": "purged_expanding_walk_forward",
        "purge_sessions": TARGET_HORIZON,
        "fold_count": len(folds),
        "folds": folds,
        "summary": {
            "ranker_median_ndcg_at_5": _median(ranker_scores),
            "ranker_worst_ndcg_at_5": _minimum(ranker_scores),
            "momentum_median_ndcg_at_5": _median(momentum_scores),
            "random_median_ndcg_at_5": _median(random_scores),
            "win_rate_vs_momentum": float(wins / len(comparable)) if comparable else None,
        },
    }


def train_ranker(frames: dict[str, pd.DataFrame], *, random_state: int) -> RankerBundle:
    dataset = build_training_dataset(frames)
    unique_dates = list(pd.unique(dataset["date"])) if not dataset.empty else []
    if len(unique_dates) < 360 or len(dataset) < 2_000:
        raise RuntimeError(
            "The selected Strategy Research baseline does not provide enough cross-sectional history for Learning-to-Rank."
        )

    validation = _walk_forward_validation(dataset, random_state=random_state)
    final_model = _fit_ranker(dataset, random_state=random_state)
    summary = validation.get("summary") or {}

    diagnostics = {
        "family": "lightgbm_lambdamart",
        "objective": "lambdarank",
        "target_horizon_sessions": TARGET_HORIZON,
        "feature_count": len(FEATURE_COLUMNS),
        "features": list(FEATURE_COLUMNS),
        "validation_method": validation.get("method"),
        "validation_fold_count": validation.get("fold_count"),
        "validation_purge_sessions": validation.get("purge_sessions"),
        "validation_folds": validation.get("folds") or [],
        "validation_summary": summary,
        "ndcg_at_5": summary.get("ranker_median_ndcg_at_5"),
        "refit_rows": int(len(dataset)),
        "refit_sessions": int(dataset["date"].nunique()),
    }
    return RankerBundle(model=final_model, diagnostics=diagnostics)

def latest_feature_snapshot(frame: pd.DataFrame) -> tuple[pd.Series, pd.Timestamp]:
    features = feature_frame(frame).dropna(subset=list(FEATURE_COLUMNS))
    if features.empty:
        raise RuntimeError("The asset does not have enough usable sessions for the ranker features.")
    row = features.iloc[-1]
    stamp = pd.Timestamp(features.index[-1])
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return row, stamp


def market_quality(frame: pd.DataFrame) -> dict[str, float]:
    source = _clean_frame(frame)
    if len(source) < MIN_FEATURE_SESSIONS:
        raise RuntimeError("insufficient_history")
    recent = source.tail(min(63, len(source)))
    close = recent["close"].astype(float)
    volume = recent["volume"].astype(float)
    latest_close = float(close.iloc[-1])
    median_dollar_volume = float((close * volume).median())
    nonzero_volume_ratio = float((volume > 0).mean())
    if latest_close < 5.0:
        raise RuntimeError("price_filter")
    if median_dollar_volume < 10_000_000.0:
        raise RuntimeError("liquidity_filter")
    if nonzero_volume_ratio < 0.98:
        raise RuntimeError("volume_quality_filter")
    return {
        "latest_close": latest_close,
        "median_dollar_volume": median_dollar_volume,
        "nonzero_volume_ratio": nonzero_volume_ratio,
    }
