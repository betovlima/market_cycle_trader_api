from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from pymongo import MongoClient

import research_contextual_signature_protocol as protocol
import research_contextual_signature_storage as storage

TARGET = "source_direct_delta_log_capital"
PRIMARY = "Original25"
ROBUST = "Original24_MinusADM"
HOLDOUT = 2026
MIN_STATES = 8
TOL = 1e-12
EXPERIMENT_NAME = "contextual_marginal_signature_temporal_trajectory_tournament"
COLLECTION = "research_contextual_signature_tournaments"
TRAJECTORY_SOURCE_FEATURES = (
    "return_5",
    "return_20",
    "return_60",
    "vol_20",
    "ema_distance_20",
    "ema_20_vs_50",
    "rsi_14",
    "atr_pct_14",
    "trend_efficiency_20",
    "momentum_acceleration_5_20",
)
TRAJECTORY_LOOKBACKS = (5, 20, 60)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: Callable[[], Any]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy-sequence", type=int, default=10)
    parser.add_argument("--strategy-id")
    parser.add_argument("--env-file")
    parser.add_argument("--mongo-uri")
    parser.add_argument("--database")
    return parser


def _models() -> list[ModelSpec]:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def scaled(model: Any) -> Pipeline:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )

    models = [
        ModelSpec("ridge", lambda: scaled(Ridge(alpha=10.0))),
        ModelSpec(
            "elastic_net",
            lambda: scaled(
                ElasticNet(
                    alpha=0.01,
                    l1_ratio=0.25,
                    max_iter=10000,
                    random_state=42,
                )
            ),
        ),
        ModelSpec(
            "mlp",
            lambda: scaled(
                MLPRegressor(
                    hidden_layer_sizes=(16,),
                    solver="lbfgs",
                    alpha=0.01,
                    max_iter=2000,
                    random_state=42,
                )
            ),
        ),
    ]
    try:
        from lightgbm import LGBMRegressor

        models.insert(
            2,
            ModelSpec(
                "lightgbm",
                lambda: LGBMRegressor(
                    n_estimators=120,
                    learning_rate=0.03,
                    num_leaves=7,
                    min_child_samples=10,
                    reg_lambda=1.0,
                    random_state=42,
                    n_jobs=1,
                    verbosity=-1,
                    deterministic=True,
                    force_col_wise=True,
                ),
            ),
        )
    except Exception:
        pass
    return models


def _num(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _stamp(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def _at(frame: pd.DataFrame, date: Any) -> pd.Series:
    index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True))
    decision = _stamp(date)
    position = int(index.searchsorted(decision, side="right") - 1)
    if position < 0:
        raise RuntimeError(f"No feature row before {decision.date()}")
    return frame.iloc[position]


def build_matrix(
    observations: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    features: list[str],
) -> pd.DataFrame:
    universes = {
        str(item["name"]): list(item["assets"])
        for item in protocol.UNIVERSES
    }
    rows: list[dict[str, Any]] = []
    spy_available = "SPY" in frames

    for observation in observations.to_dict(orient="records"):
        candidate = str(observation["candidate"])
        universe_name = str(observation["universe_name"])
        decision = pd.Timestamp(observation["decision_date"])
        candidate_row = _at(frames[candidate], decision)
        reference_assets = universes[universe_name]
        reference_rows = pd.DataFrame(
            [_at(frames[symbol], decision) for symbol in reference_assets],
            index=reference_assets,
        )
        row: dict[str, Any] = {
            "run_id": observation.get("run_id"),
            "decision_date": decision.date().isoformat(),
            "universe_name": universe_name,
            "candidate": candidate,
            TARGET: float(observation[TARGET]),
        }
        for feature in features:
            candidate_value = _num(candidate_row.get(feature))
            values = (
                pd.to_numeric(reference_rows.get(feature), errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            mean = float(values.mean()) if len(values) else float("nan")
            std = float(values.std(ddof=0)) if len(values) else float("nan")
            row[f"candidate__{feature}"] = candidate_value
            row[f"relative__{feature}"] = candidate_value - mean
            row[f"z__{feature}"] = (
                (candidate_value - mean) / std
                if math.isfinite(std) and std > TOL
                else 0.0
            )
            row[f"rankpct__{feature}"] = (
                float((values < candidate_value).mean())
                if len(values) and math.isfinite(candidate_value)
                else float("nan")
            )
            row[f"universe_mean__{feature}"] = mean
            row[f"universe_std__{feature}"] = std
            if spy_available:
                row[f"market_spy__{feature}"] = _num(
                    _at(frames["SPY"], decision).get(feature)
                )
        rows.append(row)
    return pd.DataFrame(rows)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    skip = {
        "run_id",
        "decision_date",
        "universe_name",
        "candidate",
        TARGET,
    }
    return [
        column
        for column in frame.columns
        if column not in skip
        and pd.api.types.is_numeric_dtype(frame[column])
        and frame[column].notna().any()
    ]


def _safe_rank_corr(prediction: Any, target: Any) -> float | None:
    predicted = pd.to_numeric(pd.Series(prediction), errors="coerce")
    observed = pd.to_numeric(pd.Series(target), errors="coerce")
    valid = predicted.notna() & observed.notna()
    if int(valid.sum()) < 2:
        return None
    predicted_rank = predicted[valid].rank()
    observed_rank = observed[valid].rank()
    if (
        int(predicted_rank.nunique(dropna=True)) < 2
        or int(observed_rank.nunique(dropna=True)) < 2
    ):
        return None
    value = predicted_rank.corr(observed_rank)
    if pd.isna(value):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _numeric_series(frame: pd.DataFrame, feature: str) -> pd.Series:
    if feature not in frame.columns:
        return pd.Series(dtype=float)
    series = pd.to_numeric(frame[feature], errors="coerce")
    series.index = pd.to_datetime(series.index, utc=True)
    return series.replace([np.inf, -np.inf], np.nan).sort_index()


def _path_stats(
    series: pd.Series,
    decision: Any,
    lookback: int,
) -> tuple[float, float, float]:
    if series.empty:
        return float("nan"), float("nan"), float("nan")
    cutoff = _stamp(decision)
    path = series.loc[series.index <= cutoff].dropna().tail(int(lookback) + 1)
    if len(path) < 2:
        return float("nan"), float("nan"), float("nan")
    values = path.to_numpy(dtype=float)
    delta = float(values[-1] - values[0])
    x = np.arange(len(values), dtype=float)
    slope = float(np.polyfit(x, values, 1)[0]) if len(values) >= 2 else float("nan")
    std = float(np.std(values, ddof=0))
    return delta, slope, std


def _trajectory_cache(
    frames: dict[str, pd.DataFrame],
) -> dict[tuple[str, str, str], dict[str, pd.Series]]:
    universes = {
        str(item["name"]): list(item["assets"])
        for item in protocol.UNIVERSES
    }
    output: dict[tuple[str, str, str], dict[str, pd.Series]] = {}
    market_by_feature = {
        feature: _numeric_series(frames["SPY"], feature)
        if "SPY" in frames
        else pd.Series(dtype=float)
        for feature in TRAJECTORY_SOURCE_FEATURES
    }

    for universe_name, reference_assets in universes.items():
        for feature in TRAJECTORY_SOURCE_FEATURES:
            reference_series = {
                symbol: _numeric_series(frames[symbol], feature)
                for symbol in reference_assets
                if symbol in frames
            }
            reference_panel = pd.concat(reference_series, axis=1).sort_index()
            for candidate in protocol.CANDIDATES:
                if candidate not in frames:
                    continue
                candidate_series = _numeric_series(frames[candidate], feature)
                index = reference_panel.index.union(candidate_series.index).sort_values()
                panel = reference_panel.reindex(index).ffill()
                candidate_aligned = candidate_series.reindex(index).ffill()
                universe_mean = panel.mean(axis=1, skipna=True)
                valid_count = panel.notna().sum(axis=1).replace(0, np.nan)
                relative = candidate_aligned - universe_mean
                rankpct = panel.lt(candidate_aligned, axis=0).sum(axis=1) / valid_count
                market = market_by_feature[feature].reindex(index).ffill()
                output[(universe_name, candidate, feature)] = {
                    "candidate": candidate_aligned,
                    "relative": relative,
                    "rankpct": rankpct,
                    "market": market,
                }
    return output


def augment_with_trajectory(
    snapshot: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    cache = _trajectory_cache(frames)
    trajectory_rows: list[dict[str, float]] = []

    for row in snapshot.to_dict(orient="records"):
        universe_name = str(row["universe_name"])
        candidate = str(row["candidate"])
        decision = row["decision_date"]
        additions: dict[str, float] = {}
        for feature in TRAJECTORY_SOURCE_FEATURES:
            paths = cache.get((universe_name, candidate, feature))
            if paths is None:
                continue
            for lookback in TRAJECTORY_LOOKBACKS:
                candidate_delta, candidate_slope, _ = _path_stats(
                    paths["candidate"], decision, lookback
                )
                relative_delta, relative_slope, relative_std = _path_stats(
                    paths["relative"], decision, lookback
                )
                rank_delta, rank_slope, _ = _path_stats(
                    paths["rankpct"], decision, lookback
                )
                market_delta, market_slope, _ = _path_stats(
                    paths["market"], decision, lookback
                )
                prefix = f"trajectory_{lookback}__{feature}"
                additions[f"{prefix}__candidate_delta"] = candidate_delta
                additions[f"{prefix}__candidate_slope"] = candidate_slope
                additions[f"{prefix}__relative_delta"] = relative_delta
                additions[f"{prefix}__relative_slope"] = relative_slope
                additions[f"{prefix}__relative_std"] = relative_std
                additions[f"{prefix}__rank_delta"] = rank_delta
                additions[f"{prefix}__rank_slope"] = rank_slope
                additions[f"{prefix}__market_delta"] = market_delta
                additions[f"{prefix}__market_slope"] = market_slope
        trajectory_rows.append(additions)

    trajectory_frame = pd.DataFrame(trajectory_rows, index=snapshot.index)
    return pd.concat([snapshot.copy(), trajectory_frame], axis=1)


def _pick(group: pd.DataFrame) -> tuple[str | None, float, float]:
    best = group.sort_values("prediction", ascending=False).iloc[0]
    prediction = float(best["prediction"])
    oracle = max(0.0, float(group[TARGET].max()))
    if prediction <= 0.0:
        return None, 0.0, oracle
    return str(best["candidate"]), float(best[TARGET]), oracle


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "contexts": 0,
            "interventions": 0,
            "positive_interventions": 0,
            "cumulative_direct_log_gain": 0.0,
            "capital_multiplier_vs_no_insert": 1.0,
            "oracle_log_gain": 0.0,
            "mean_context_spearman": None,
        }
    frame = pd.DataFrame(rows)
    gain = float(frame["realized"].sum())
    correlations = pd.to_numeric(frame["rho"], errors="coerce").dropna()
    return {
        "contexts": int(len(frame)),
        "interventions": int(frame["selected"].notna().sum()),
        "positive_interventions": int(
            ((frame["realized"] > TOL) & frame["selected"].notna()).sum()
        ),
        "cumulative_direct_log_gain": gain,
        "capital_multiplier_vs_no_insert": float(math.exp(gain)),
        "oracle_log_gain": float(frame["oracle"].sum()),
        "mean_context_spearman": (
            float(correlations.mean()) if len(correlations) else None
        ),
    }


def walk_forward(
    frame: pd.DataFrame,
    columns: list[str],
    spec: ModelSpec,
) -> dict[str, Any]:
    data = frame.loc[frame["universe_name"] == PRIMARY].copy()
    data["decision_date"] = pd.to_datetime(data["decision_date"])
    dates = sorted(
        value
        for value in data["decision_date"].unique()
        if pd.Timestamp(value).year < HOLDOUT
    )
    rows: list[dict[str, Any]] = []
    for index in range(MIN_STATES, len(dates)):
        train = data.loc[data["decision_date"].isin(dates[:index])]
        test = data.loc[data["decision_date"] == dates[index]].copy()
        model = spec.factory()
        model.fit(train[columns], train[TARGET])
        test["prediction"] = model.predict(test[columns])
        selected, realized, oracle = _pick(test)
        rows.append(
            {
                "date": str(pd.Timestamp(dates[index]).date()),
                "selected": selected,
                "realized": realized,
                "oracle": oracle,
                "rho": _safe_rank_corr(test["prediction"], test[TARGET]),
            }
        )
    return _metrics(rows)


def mean_baseline(frame: pd.DataFrame) -> dict[str, Any]:
    data = frame.loc[frame["universe_name"] == PRIMARY].copy()
    data["decision_date"] = pd.to_datetime(data["decision_date"])
    dates = sorted(
        value
        for value in data["decision_date"].unique()
        if pd.Timestamp(value).year < HOLDOUT
    )
    rows: list[dict[str, Any]] = []
    for index in range(MIN_STATES, len(dates)):
        train = data.loc[data["decision_date"].isin(dates[:index])]
        test = data.loc[data["decision_date"] == dates[index]].copy()
        means = train.groupby("candidate")[TARGET].mean()
        test["prediction"] = test["candidate"].map(means).fillna(0.0)
        selected, realized, oracle = _pick(test)
        rows.append(
            {
                "selected": selected,
                "realized": realized,
                "oracle": oracle,
                "rho": _safe_rank_corr(test["prediction"], test[TARGET]),
            }
        )
    return _metrics(rows)


def holdout(
    frame: pd.DataFrame,
    columns: list[str],
    spec: ModelSpec,
    universe: str,
) -> dict[str, Any]:
    data = frame.copy()
    data["decision_date"] = pd.to_datetime(data["decision_date"])
    train = data.loc[
        (data["universe_name"] == PRIMARY)
        & (data["decision_date"].dt.year < HOLDOUT)
    ]
    test = data.loc[
        (data["universe_name"] == universe)
        & (data["decision_date"].dt.year >= HOLDOUT)
    ].copy()
    rows: list[dict[str, Any]] = []
    if test.empty:
        return _metrics(rows)
    model = spec.factory()
    model.fit(train[columns], train[TARGET])
    test["prediction"] = model.predict(test[columns])
    for _, group in test.groupby("decision_date", sort=True):
        selected, realized, oracle = _pick(group)
        rows.append(
            {
                "selected": selected,
                "realized": realized,
                "oracle": oracle,
                "rho": _safe_rank_corr(group["prediction"], group[TARGET]),
            }
        )
    return _metrics(rows)


def snapshot_status(
    development: dict[str, Any],
    baseline: dict[str, Any],
    holdout_metrics: dict[str, Any],
    robust_metrics: dict[str, Any],
) -> str:
    development_gain = development["cumulative_direct_log_gain"]
    baseline_gain = baseline["cumulative_direct_log_gain"]
    if development_gain <= TOL or development_gain <= baseline_gain + TOL:
        return "NO_CONTEXTUAL_SIGNAL"
    if holdout_metrics["cumulative_direct_log_gain"] <= TOL:
        return "DEVELOPMENT_SIGNAL_NOT_CONFIRMED"
    if robust_metrics["cumulative_direct_log_gain"] <= TOL:
        return "HOLDOUT_SIGNAL_NOT_ROBUST"
    return "CONFIRMED_LIMITED_NEXT_SYSTEM_BACKTEST"


def trajectory_status(
    trajectory_development: dict[str, Any],
    snapshot_development: dict[str, Any],
    baseline: dict[str, Any],
    holdout_metrics: dict[str, Any],
    robust_metrics: dict[str, Any],
) -> str:
    trajectory_gain = trajectory_development["cumulative_direct_log_gain"]
    snapshot_gain = snapshot_development["cumulative_direct_log_gain"]
    baseline_gain = baseline["cumulative_direct_log_gain"]
    if trajectory_gain <= snapshot_gain + TOL:
        return "TRAJECTORY_ADDED_VALUE_NOT_FOUND"
    if trajectory_gain <= baseline_gain + TOL:
        return "TRAJECTORY_CONTEXTUAL_SIGNAL_NOT_FOUND"
    if holdout_metrics["cumulative_direct_log_gain"] <= TOL:
        return "TRAJECTORY_SIGNAL_NOT_CONFIRMED"
    if robust_metrics["cumulative_direct_log_gain"] <= TOL:
        return "TRAJECTORY_HOLDOUT_NOT_ROBUST"
    return "TRAJECTORY_CONFIRMED_LIMITED_NEXT_SYSTEM_BACKTEST"


def _winner(
    specs: list[ModelSpec],
    results: dict[str, dict[str, Any]],
) -> ModelSpec:
    return max(
        specs,
        key=lambda spec: results[spec.name]["cumulative_direct_log_gain"],
    )


def main(*, script_version: str) -> int:
    protocol._install_console_logging()
    args = _parser().parse_args()
    protocol.load_project_environment(args.env_file)

    from market_cycle_trader_api.engine import capital_rotation, market_data
    from market_cycle_trader_api.infrastructure.persistence import mongo_repository
    from market_cycle_trader_api.schemas.requests import BacktestRequest

    mongo_uri, database_name = storage.runtime_mongo_settings(
        args,
        mongo_repository,
        protocol,
    )
    client = MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=3000,
        connectTimeoutMS=3000,
        retryWrites=False,
    )
    try:
        storage.mongo_retry(lambda: client.admin.command("ping"))
        db = client[database_name]
        strategy = protocol._strategy_document(
            db,
            args.strategy_sequence,
            args.strategy_id,
        )
        strategy_id = str(strategy["_id"])
        config = BacktestRequest.model_validate(
            protocol._configuration(strategy)
        ).model_copy(
            update={
                "end_date": protocol.SNAPSHOT_END,
                "research_market_data_mode": "database_only",
                "mongo_cache_enabled": True,
                "market_data_require_complete_history": True,
            }
        )

        expected = (
            len(protocol.DECISION_DATES)
            * len(protocol.UNIVERSES)
            * len(protocol.CANDIDATES)
        )
        source_run = db[protocol.RUNS_COLLECTION].find_one(
            {
                "experiment": protocol.EXPERIMENT_NAME,
                "strategy_id": strategy_id,
                "status": "completed",
                "completed_observations": {"$gte": expected},
            },
            sort=[("completed_utc", -1), ("updated_utc", -1)],
        )
        if not source_run:
            raise RuntimeError("No completed 23x2x7 Mongo campaign found.")

        source_run_id = str(source_run["_id"])
        observations = pd.DataFrame(
            list(
                db[protocol.OBSERVATIONS_COLLECTION].find(
                    {"run_id": source_run_id},
                    {"_id": 0},
                )
            )
        )
        if len(observations) != expected or TARGET not in observations:
            raise RuntimeError(
                f"Expected {expected} observations with {TARGET}."
            )

        bars, _ = protocol._load_market_frames(config, market_data)
        feature_frames = {
            symbol: capital_rotation.build_rotation_frame(frame, config)
            for symbol, frame in bars.items()
        }
        snapshot = build_matrix(
            observations,
            feature_frames,
            list(capital_rotation.ROTATION_FEATURES),
        )
        snapshot_columns = feature_columns(snapshot)
        protocol.live.console_log(
            f"[tournament] rows={len(snapshot)} | "
            f"features={len(snapshot_columns)} | target={TARGET} | "
            "candidate identity excluded"
        )

        baseline = mean_baseline(snapshot)
        specs = _models()
        snapshot_development = {
            spec.name: walk_forward(snapshot, snapshot_columns, spec)
            for spec in specs
        }
        snapshot_winner = _winner(specs, snapshot_development)
        snapshot_holdout = holdout(
            snapshot,
            snapshot_columns,
            snapshot_winner,
            PRIMARY,
        )
        snapshot_robust = holdout(
            snapshot,
            snapshot_columns,
            snapshot_winner,
            ROBUST,
        )
        snapshot_decision = snapshot_status(
            snapshot_development[snapshot_winner.name],
            baseline,
            snapshot_holdout,
            snapshot_robust,
        )
        for name, metrics in snapshot_development.items():
            protocol.live.console_log(
                f"[development] {name}: "
                f"{metrics['capital_multiplier_vs_no_insert']:.4f}x"
            )
        protocol.live.console_log(
            f"[winner] {snapshot_winner.name} | "
            f"dev={snapshot_development[snapshot_winner.name]['capital_multiplier_vs_no_insert']:.4f}x | "
            f"2026={snapshot_holdout['capital_multiplier_vs_no_insert']:.4f}x | "
            f"robust={snapshot_robust['capital_multiplier_vs_no_insert']:.4f}x | "
            f"{snapshot_decision}"
        )

        trajectory = augment_with_trajectory(snapshot, feature_frames)
        trajectory_columns = feature_columns(trajectory)
        trajectory_added_columns = sorted(
            set(trajectory_columns).difference(snapshot_columns)
        )
        protocol.live.console_log(
            f"[trajectory] rows={len(trajectory)} | "
            f"snapshot_features={len(snapshot_columns)} | "
            f"added_features={len(trajectory_added_columns)} | "
            f"total_features={len(trajectory_columns)} | "
            f"lookbacks={','.join(str(x) for x in TRAJECTORY_LOOKBACKS)} | "
            f"source_features={len(TRAJECTORY_SOURCE_FEATURES)}"
        )

        trajectory_development = {
            spec.name: walk_forward(trajectory, trajectory_columns, spec)
            for spec in specs
        }
        trajectory_winner = _winner(specs, trajectory_development)
        trajectory_holdout = holdout(
            trajectory,
            trajectory_columns,
            trajectory_winner,
            PRIMARY,
        )
        trajectory_robust = holdout(
            trajectory,
            trajectory_columns,
            trajectory_winner,
            ROBUST,
        )
        decision = trajectory_status(
            trajectory_development[trajectory_winner.name],
            snapshot_development[snapshot_winner.name],
            baseline,
            trajectory_holdout,
            trajectory_robust,
        )
        for name, metrics in trajectory_development.items():
            protocol.live.console_log(
                f"[trajectory-development] {name}: "
                f"{metrics['capital_multiplier_vs_no_insert']:.4f}x"
            )
        protocol.live.console_log(
            f"[trajectory-winner] {trajectory_winner.name} | "
            f"dev={trajectory_development[trajectory_winner.name]['capital_multiplier_vs_no_insert']:.4f}x | "
            f"2026={trajectory_holdout['capital_multiplier_vs_no_insert']:.4f}x | "
            f"robust={trajectory_robust['capital_multiplier_vs_no_insert']:.4f}x | "
            f"{decision}"
        )

        snapshot_gain = snapshot_development[snapshot_winner.name][
            "cumulative_direct_log_gain"
        ]
        trajectory_gain = trajectory_development[trajectory_winner.name][
            "cumulative_direct_log_gain"
        ]
        protocol.live.console_log(
            f"[comparison] snapshot={snapshot_winner.name} "
            f"{snapshot_development[snapshot_winner.name]['capital_multiplier_vs_no_insert']:.4f}x | "
            f"trajectory={trajectory_winner.name} "
            f"{trajectory_development[trajectory_winner.name]['capital_multiplier_vs_no_insert']:.4f}x | "
            f"delta_log={trajectory_gain - snapshot_gain:+.6f}"
        )
        protocol.live.console_log(f"[decision] {decision}")

        tournament_id = storage.canonical_hash(
            {
                "source": source_run_id,
                "version": script_version,
                "snapshot_features": snapshot_columns,
                "trajectory_source_features": list(TRAJECTORY_SOURCE_FEATURES),
                "trajectory_lookbacks": list(TRAJECTORY_LOOKBACKS),
                "trajectory_features": trajectory_added_columns,
            }
        )
        summary = {
            "_id": tournament_id,
            "schema_version": 2,
            "script_version": script_version,
            "experiment": EXPERIMENT_NAME,
            "status": "completed",
            "source_run_id": source_run_id,
            "strategy_id": strategy_id,
            "target": TARGET,
            "target_definition": (
                "log(W_policy_with_candidate_added / "
                "W_baseline_policy_without_candidate)"
            ),
            "rows": int(len(trajectory)),
            "independent_primary_temporal_states": int(
                trajectory.loc[
                    trajectory["universe_name"] == PRIMARY,
                    "decision_date",
                ].nunique()
            ),
            "candidate_identity_feature": False,
            "validation": (
                "Original25 expanding walk-forward through 2025; "
                "2026 untouched holdout; MinusADM robustness only"
            ),
            "historical_candidate_mean_baseline": baseline,
            "snapshot_feature_count": int(len(snapshot_columns)),
            "development_models": snapshot_development,
            "development_winner": snapshot_winner.name,
            "final_holdout_2026": snapshot_holdout,
            "robustness_holdout_2026": snapshot_robust,
            "snapshot_decision_status": snapshot_decision,
            "trajectory_source_features": list(TRAJECTORY_SOURCE_FEATURES),
            "trajectory_lookbacks_sessions": list(TRAJECTORY_LOOKBACKS),
            "trajectory_added_feature_count": int(
                len(trajectory_added_columns)
            ),
            "trajectory_total_feature_count": int(
                len(trajectory_columns)
            ),
            "trajectory_development_models": trajectory_development,
            "trajectory_development_winner": trajectory_winner.name,
            "trajectory_final_holdout_2026": trajectory_holdout,
            "trajectory_robustness_holdout_2026": trajectory_robust,
            "representation_comparison": {
                "snapshot_winner": snapshot_winner.name,
                "snapshot_development_log_gain": float(snapshot_gain),
                "trajectory_winner": trajectory_winner.name,
                "trajectory_development_log_gain": float(trajectory_gain),
                "trajectory_minus_snapshot_log_gain": float(
                    trajectory_gain - snapshot_gain
                ),
            },
            "decision_status": decision,
            "next_step": (
                "Only TRAJECTORY_CONFIRMED_LIMITED_NEXT_SYSTEM_BACKTEST "
                "proceeds to one frozen system-level capital backtest. "
                "Otherwise do not add more tabular models to this frozen "
                "trajectory experiment merely to manufacture a winner."
            ),
            "created_utc": pd.Timestamp.now(tz="UTC").to_pydatetime(),
        }
        storage.mongo_retry(
            lambda: db[COLLECTION].replace_one(
                {"_id": tournament_id},
                storage.mongo_value(summary),
                upsert=True,
            )
        )
        export_summary = {
            key: value for key, value in summary.items() if key != "_id"
        }
        export_dir, zip_path = storage.export_result(
            project_root=protocol.PROJECT_ROOT,
            run_id=tournament_id,
            script_version=script_version,
            dataset=trajectory,
            summary=export_summary,
            export_folder_name=protocol.EXPORT_FOLDER_NAME,
            export_zip_name=protocol.EXPORT_ZIP_NAME,
            collections=(
                protocol.RUNS_COLLECTION,
                protocol.OBSERVATIONS_COLLECTION,
                COLLECTION,
            ),
        )
        protocol.live.console_log(
            f"[complete] tournament persisted | collection={COLLECTION}"
        )
        protocol.live.console_log(
            f"[complete] export folder={export_dir}"
        )
        protocol.live.console_log(f"[complete] ZIP={zip_path}")
        protocol.live.console_log(f"[complete] decision={decision}")
        return 0
    finally:
        client.close()
