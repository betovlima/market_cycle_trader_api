from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymongo import MongoClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import research_asset_signature_leave_one_out as common  # noqa: E402
import research_sequential_exact_marginal_search as sequential  # noqa: E402
import research_sequential_exact_marginal_search_v110 as accelerator  # noqa: E402
from market_cycle_trader_api.core.config import API_VERSION  # noqa: E402
from market_cycle_trader_api.engine import capital_rotation as rotation  # noqa: E402
from market_cycle_trader_api.engine.market_data import validate_and_clean_bars  # noqa: E402
from market_cycle_trader_api.schemas.requests import BacktestRequest  # noqa: E402
from market_cycle_trader_api.services import asset_discovery as discovery  # noqa: E402

SCRIPT_VERSION = "contextual-marginal-signature-v1.0.2"
EXPERIMENT_NAME = "contextual_marginal_signature"
DEFAULT_WORKERS = 4

BASE_FEATURES = [
    "return_5",
    "return_20",
    "return_60",
    "vol_20",
    "vol_60",
    "ema_distance_20",
    "trend_efficiency_20",
    "momentum_acceleration_5_20",
]

MODEL_FEATURES = [
    "relative__return_5",
    "relative__return_20",
    "relative__return_60",
    "relative__vol_20",
    "relative__vol_60",
    "relative__ema_distance_20",
    "relative__trend_efficiency_20",
    "relative__momentum_acceleration_5_20",
    "corr_20_to_universe",
    "corr_60_to_universe",
    "universe_dispersion_return_20",
    "universe_dispersion_vol_20",
    "candidate_rank_return_20",
    "candidate_rank_return_60",
    "market_return_20",
    "market_vol_20",
]


def _log(message: str) -> None:
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{stamp}] {message}", flush=True)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def _safe_fresh(path: Path) -> None:
    root = (PROJECT_ROOT / "research_output").resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"Refusing --fresh-run outside {root}: {resolved}") from exc
    if not relative.parts or not relative.parts[0].startswith("contextual_marginal_signature_"):
        raise RuntimeError(f"Refusing to delete unexpected research output: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a small point-in-time counterfactual panel to test whether information available at t "
            "contains a stable mathematical signal about exact marginal capital contribution over a future window."
        )
    )
    parser.add_argument("--strategy-sequence", type=int, default=10)
    parser.add_argument("--strategy-id", default=None)
    parser.add_argument("--history-start", required=True)
    parser.add_argument("--snapshot-end", required=True)
    parser.add_argument("--seed-assets", nargs="+", required=True)
    parser.add_argument("--candidate-symbols", nargs="+", required=True)
    parser.add_argument("--decision-start", required=True)
    parser.add_argument("--decision-end", required=True)
    parser.add_argument("--decision-count", type=int, default=20)
    parser.add_argument("--horizon-sessions", type=int, default=40)
    parser.add_argument("--validation-start", required=True)
    parser.add_argument("--market-proxy", default="SPY")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--mongo-uri", default=None)
    parser.add_argument("--database", default=None)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--allow-remote-mongo", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--fresh-run", action="store_true")
    return parser


def _utc_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _utc_index(values: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(pd.DatetimeIndex(values), utc=True)).sort_values()


def select_decision_sessions(
    sessions: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
    count: int,
    horizon_sessions: int,
) -> list[pd.Timestamp]:
    ordered = _utc_index(sessions)
    start = _utc_timestamp(start)
    end = _utc_timestamp(end)
    count = max(1, int(count))
    horizon_sessions = max(1, int(horizon_sessions))

    monthly: list[pd.Timestamp] = []
    seen_months: set[tuple[int, int]] = set()
    for position, session in enumerate(ordered):
        ts = _utc_timestamp(session)
        if ts < start or ts > end:
            continue
        if position + horizon_sessions - 1 >= len(ordered):
            continue
        key = (ts.year, ts.month)
        if key in seen_months:
            continue
        seen_months.add(key)
        monthly.append(ts)

    if not monthly:
        raise RuntimeError("No eligible monthly decision sessions remain for the requested window/horizon.")
    if len(monthly) <= count:
        return monthly

    positions = np.linspace(0, len(monthly) - 1, num=count)
    selected_indices: list[int] = []
    for value in positions:
        index = int(round(float(value)))
        if index not in selected_indices:
            selected_indices.append(index)
    if len(selected_indices) < count:
        for index in range(len(monthly)):
            if index not in selected_indices:
                selected_indices.append(index)
            if len(selected_indices) == count:
                break
    return [monthly[index] for index in sorted(selected_indices[:count])]


def _horizon_end(
    sessions: pd.DatetimeIndex,
    decision_session: pd.Timestamp,
    horizon_sessions: int,
) -> pd.Timestamp:
    ordered = _utc_index(sessions)
    decision = _utc_timestamp(decision_session)
    position = int(ordered.searchsorted(decision, side="left"))
    if position >= len(ordered) or pd.Timestamp(ordered[position]) != decision:
        raise RuntimeError(f"Decision session is not in the expected calendar: {decision.date()}")
    end_position = position + max(1, int(horizon_sessions)) - 1
    if end_position >= len(ordered):
        raise RuntimeError(f"Horizon exceeds snapshot for decision session {decision.date()}")
    return pd.Timestamp(ordered[end_position])


def panel_completeness_issues(
    frame: pd.DataFrame,
    decision_sessions: list[pd.Timestamp],
    candidates: list[str],
) -> list[str]:
    expected = {
        (pd.Timestamp(date).date().isoformat(), str(symbol).strip().upper())
        for date in decision_sessions
        for symbol in candidates
    }
    observed: dict[tuple[str, str], int] = {}
    non_completed: list[str] = []
    for row in frame.to_dict(orient="records"):
        key = (str(row.get("decision_date") or ""), str(row.get("candidate") or "").upper())
        observed[key] = observed.get(key, 0) + 1
        if str(row.get("evaluation_status") or "").lower() != "completed":
            non_completed.append(f"{key[0]}:{key[1]}={row.get('evaluation_status')}")
    missing = sorted(expected.difference(observed))
    duplicates = sorted(key for key, count in observed.items() if count != 1)
    unexpected = sorted(set(observed).difference(expected))

    issues: list[str] = []
    if missing:
        issues.append("missing:" + ",".join(f"{d}:{s}" for d, s in missing))
    if unexpected:
        issues.append("unexpected:" + ",".join(f"{d}:{s}" for d, s in unexpected))
    if duplicates:
        issues.append("duplicates:" + ",".join(f"{d}:{s}" for d, s in duplicates))
    if non_completed:
        issues.append("non_completed:" + ",".join(sorted(non_completed)))
    return issues


def _asof_rotation_row(frame: pd.DataFrame, decision: pd.Timestamp) -> pd.Series:
    decision = _utc_timestamp(decision)
    index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True))
    normalized = frame.copy()
    normalized.index = index
    eligible = normalized.loc[normalized.index <= decision]
    if eligible.empty:
        raise RuntimeError(f"No point-in-time feature row is available at {decision.date()}")
    return eligible.iloc[-1]


def _equal_weight_return_series(
    raw_frames: dict[str, pd.DataFrame],
    symbols: list[str],
    decision: pd.Timestamp,
) -> pd.Series:
    decision = _utc_timestamp(decision)
    columns: dict[str, pd.Series] = {}
    for symbol in symbols:
        frame = raw_frames[symbol].copy()
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True))
        frame = frame.loc[frame.index <= decision]
        close = pd.to_numeric(frame["close"], errors="coerce")
        columns[symbol] = close.pct_change()
    return pd.DataFrame(columns).mean(axis=1, skipna=True)


def _candidate_correlation(
    raw_frames: dict[str, pd.DataFrame],
    seed_assets: list[str],
    candidate: str,
    decision: pd.Timestamp,
    window: int,
) -> float:
    decision = _utc_timestamp(decision)
    market = _equal_weight_return_series(raw_frames, seed_assets, decision)
    candidate_frame = raw_frames[candidate].copy()
    candidate_frame.index = pd.DatetimeIndex(pd.to_datetime(candidate_frame.index, utc=True))
    candidate_frame = candidate_frame.loc[candidate_frame.index <= decision]
    candidate_return = pd.to_numeric(candidate_frame["close"], errors="coerce").pct_change()
    aligned = pd.concat([candidate_return.rename("candidate"), market.rename("universe")], axis=1).dropna().tail(window)
    if len(aligned) < max(10, window // 2):
        return float("nan")
    return float(aligned["candidate"].corr(aligned["universe"]))


def build_feature_snapshot(
    *,
    candidate: str,
    seed_assets: list[str],
    rotation_frames: dict[str, pd.DataFrame],
    raw_frames: dict[str, pd.DataFrame],
    decision: pd.Timestamp,
    market_proxy: str,
) -> dict[str, float]:
    candidate_row = _asof_rotation_row(rotation_frames[candidate], decision)
    universe_rows = pd.DataFrame(
        {
            symbol: _asof_rotation_row(rotation_frames[symbol], decision)
            for symbol in seed_assets
        }
    ).T

    result: dict[str, float] = {}
    for feature in BASE_FEATURES:
        if feature not in rotation.ROTATION_FEATURES:
            raise RuntimeError(f"Feature is not backward-looking ROTATION_FEATURES input: {feature}")
        candidate_value = float(candidate_row.get(feature, float("nan")))
        universe_mean = float(pd.to_numeric(universe_rows[feature], errors="coerce").mean())
        result[f"candidate__{feature}"] = candidate_value
        result[f"universe_mean__{feature}"] = universe_mean
        result[f"relative__{feature}"] = candidate_value - universe_mean

    combined = pd.concat(
        [
            pd.to_numeric(universe_rows["return_20"], errors="coerce"),
            pd.Series({candidate: float(candidate_row.get("return_20", float("nan")))}, dtype=float),
        ]
    )
    result["candidate_rank_return_20"] = float(combined.rank(pct=True).get(candidate, float("nan")))
    combined_60 = pd.concat(
        [
            pd.to_numeric(universe_rows["return_60"], errors="coerce"),
            pd.Series({candidate: float(candidate_row.get("return_60", float("nan")))}, dtype=float),
        ]
    )
    result["candidate_rank_return_60"] = float(combined_60.rank(pct=True).get(candidate, float("nan")))
    result["universe_dispersion_return_20"] = float(pd.to_numeric(universe_rows["return_20"], errors="coerce").std())
    result["universe_dispersion_vol_20"] = float(pd.to_numeric(universe_rows["vol_20"], errors="coerce").std())
    result["corr_20_to_universe"] = _candidate_correlation(raw_frames, seed_assets, candidate, decision, 20)
    result["corr_60_to_universe"] = _candidate_correlation(raw_frames, seed_assets, candidate, decision, 60)

    proxy = str(market_proxy).strip().upper()
    if proxy in rotation_frames:
        proxy_row = _asof_rotation_row(rotation_frames[proxy], decision)
        result["market_return_20"] = float(proxy_row.get("return_20", float("nan")))
        result["market_vol_20"] = float(proxy_row.get("vol_20", float("nan")))
    else:
        result["market_return_20"] = float(pd.to_numeric(universe_rows["return_20"], errors="coerce").mean())
        result["market_vol_20"] = float(pd.to_numeric(universe_rows["vol_20"], errors="coerce").mean())
    return result


def _window_request(
    *,
    db: Any,
    config: BacktestRequest,
    strategy_id: str,
    assets: list[str],
    reference_assets: list[str],
    candidate_assets: list[str],
    decision_session: pd.Timestamp,
    horizon_end: pd.Timestamp,
):
    return discovery._marginal_execution_request(
        db,
        config,
        {"id": strategy_id},
        config,
        horizon_end.date().isoformat(),
        assets=assets,
        reference_assets=reference_assets,
        candidate_assets=candidate_assets,
        analysis_start_date=decision_session.date().isoformat(),
        analysis_end_date=horizon_end.date().isoformat(),
    )


def _evaluate_candidate_window(
    *,
    db: Any,
    config: BacktestRequest,
    strategy_id: str,
    seed_assets: list[str],
    seed_frames: dict[str, pd.DataFrame],
    baseline_metrics: dict[str, Any],
    baseline_sessions: pd.DatetimeIndex,
    candidate: str,
    candidate_frame: pd.DataFrame,
    decision_session: pd.Timestamp,
    horizon_end: pd.Timestamp,
    feature_snapshot: dict[str, float],
) -> dict[str, Any]:
    started = time.perf_counter()
    row: dict[str, Any] = {
        "decision_date": decision_session.date().isoformat(),
        "horizon_end": horizon_end.date().isoformat(),
        "candidate": candidate,
        "evaluation_status": "running",
        **feature_snapshot,
    }
    try:
        assets = [*seed_assets, candidate]
        frames = dict(seed_frames)
        frames[candidate] = candidate_frame
        request = _window_request(
            db=db,
            config=config,
            strategy_id=strategy_id,
            assets=assets,
            reference_assets=seed_assets,
            candidate_assets=[candidate],
            decision_session=decision_session,
            horizon_end=horizon_end,
        )
        metrics, sessions = discovery._run_rotation_replay(frames, request)
        context = discovery._research_context_compatibility(baseline_sessions, sessions)
        row.update(context)
        if not bool(context.get("research_context_compatible")):
            row.update({"evaluation_status": "context_rejected", "error": "research_context_incomplete"})
        else:
            baseline_capital = discovery._finite_number(baseline_metrics.get("ending_capital"))
            candidate_capital = discovery._finite_number(metrics.get("ending_capital"))
            if baseline_capital is None or candidate_capital is None or baseline_capital <= 0 or candidate_capital <= 0:
                raise RuntimeError("Exact window returned non-positive/invalid ending capital.")
            delta_rate = discovery._capital_delta_rate(candidate_capital, baseline_capital)
            row.update(
                {
                    "evaluation_status": "completed",
                    "baseline_ending_capital": baseline_capital,
                    "candidate_ending_capital": candidate_capital,
                    "ending_capital_delta_rate": delta_rate,
                    "delta_log_capital": float(math.log(candidate_capital / baseline_capital)),
                    "candidate_cagr": metrics.get("cagr"),
                    "candidate_sharpe": metrics.get("sharpe"),
                    "candidate_maximum_drawdown": metrics.get("maximum_drawdown"),
                    "candidate_switches": metrics.get("switches"),
                }
            )
    except Exception as exc:
        row.update({"evaluation_status": "failed", "error": str(exc)[:700]})
    row["total_seconds"] = float(time.perf_counter() - started)
    return row


def feature_report(dataset: pd.DataFrame) -> pd.DataFrame:
    completed = dataset.loc[dataset["evaluation_status"].astype(str).str.lower() == "completed"].copy()
    rows: list[dict[str, Any]] = []
    y = pd.to_numeric(completed["delta_log_capital"], errors="coerce")
    for feature in MODEL_FEATURES:
        x = pd.to_numeric(completed[feature], errors="coerce")
        pair = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
        if len(pair) < 3:
            rows.append({"feature": feature, "n": len(pair), "pearson": None, "spearman": None})
            continue
        median = float(pair["x"].median())
        low = pair.loc[pair["x"] <= median, "y"]
        high = pair.loc[pair["x"] > median, "y"]
        rows.append(
            {
                "feature": feature,
                "n": int(len(pair)),
                "pearson": float(pair["x"].corr(pair["y"], method="pearson")),
                "spearman": float(pair["x"].corr(pair["y"], method="spearman")),
                "median_split": median,
                "mean_delta_log_low": float(low.mean()) if len(low) else None,
                "mean_delta_log_high": float(high.mean()) if len(high) else None,
                "positive_rate_low": float((low > 0).mean()) if len(low) else None,
                "positive_rate_high": float((high > 0).mean()) if len(high) else None,
            }
        )
    return pd.DataFrame(rows).sort_values("spearman", key=lambda s: s.abs(), ascending=False, na_position="last")


def temporal_ols_validation(
    dataset: pd.DataFrame,
    validation_start: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    completed = dataset.loc[dataset["evaluation_status"].astype(str).str.lower() == "completed"].copy()
    completed["decision_ts"] = pd.to_datetime(completed["decision_date"], utc=True)
    validation_start = _utc_timestamp(validation_start)
    train = completed.loc[completed["decision_ts"] < validation_start].copy()
    test = completed.loc[completed["decision_ts"] >= validation_start].copy()
    if train.empty or test.empty:
        return pd.DataFrame(), pd.DataFrame(), {
            "status": "insufficient_temporal_split",
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
        }

    train_x = train.loc[:, MODEL_FEATURES].apply(pd.to_numeric, errors="coerce")
    test_x = test.loc[:, MODEL_FEATURES].apply(pd.to_numeric, errors="coerce")
    medians = train_x.median(axis=0)
    train_x = train_x.fillna(medians)
    test_x = test_x.fillna(medians)
    means = train_x.mean(axis=0)
    stds = train_x.std(axis=0, ddof=0)
    active = [feature for feature in MODEL_FEATURES if np.isfinite(stds[feature]) and float(stds[feature]) > 1e-12]
    if not active:
        return pd.DataFrame(), pd.DataFrame(), {"status": "no_nonconstant_features"}

    x_train = ((train_x[active] - means[active]) / stds[active]).to_numpy(dtype=float)
    x_test = ((test_x[active] - means[active]) / stds[active]).to_numpy(dtype=float)
    y_train = pd.to_numeric(train["delta_log_capital"], errors="coerce").to_numpy(dtype=float)
    y_test = pd.to_numeric(test["delta_log_capital"], errors="coerce").to_numpy(dtype=float)

    design_train = np.column_stack([np.ones(len(x_train), dtype=float), x_train])
    design_test = np.column_stack([np.ones(len(x_test), dtype=float), x_test])
    beta, *_ = np.linalg.lstsq(design_train, y_train, rcond=None)
    prediction = design_test @ beta

    predictions = test[["decision_date", "candidate", "ending_capital_delta_rate", "delta_log_capital"]].copy()
    predictions["predicted_delta_log_capital"] = prediction
    predictions["predicted_positive"] = prediction > 0
    predictions["actual_positive"] = y_test > 0

    coefficients = pd.DataFrame({"feature": ["intercept", *active], "standardized_ols_coefficient": beta})

    mae = float(np.mean(np.abs(prediction - y_test)))
    pearson = float(pd.Series(prediction).corr(pd.Series(y_test), method="pearson")) if len(test) > 1 else float("nan")
    spearman = float(pd.Series(prediction).corr(pd.Series(y_test), method="spearman")) if len(test) > 1 else float("nan")
    sign_accuracy = float(np.mean((prediction > 0) == (y_test > 0)))
    majority_positive = bool(np.mean(y_train > 0) >= 0.5)
    baseline_sign_accuracy = float(np.mean((y_test > 0) == majority_positive))

    per_date_rows: list[dict[str, Any]] = []
    for decision_date, group in predictions.groupby("decision_date", sort=True):
        chosen = group.sort_values("predicted_delta_log_capital", ascending=False).iloc[0]
        oracle = group.sort_values("delta_log_capital", ascending=False).iloc[0]
        per_date_rows.append(
            {
                "decision_date": decision_date,
                "chosen_candidate": chosen["candidate"],
                "chosen_actual_delta_rate": float(chosen["ending_capital_delta_rate"]),
                "chosen_actual_delta_log": float(chosen["delta_log_capital"]),
                "mean_candidate_delta_log": float(group["delta_log_capital"].mean()),
                "oracle_candidate": oracle["candidate"],
                "oracle_delta_log": float(oracle["delta_log_capital"]),
                "rank_spearman": float(group["predicted_delta_log_capital"].corr(group["delta_log_capital"], method="spearman")),
            }
        )
    per_date = pd.DataFrame(per_date_rows)

    summary = {
        "status": "completed",
        "validation_start": validation_start.date().isoformat(),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_dates": int(train["decision_date"].nunique()),
        "test_dates": int(test["decision_date"].nunique()),
        "active_feature_count": int(len(active)),
        "mae_delta_log_capital": mae,
        "pearson_prediction_vs_actual": pearson,
        "spearman_prediction_vs_actual": spearman,
        "sign_accuracy": sign_accuracy,
        "majority_sign_baseline_accuracy": baseline_sign_accuracy,
        "mean_cross_sectional_rank_spearman": float(per_date["rank_spearman"].mean()) if not per_date.empty else None,
        "top1_positive_rate": float((per_date["chosen_actual_delta_log"] > 0).mean()) if not per_date.empty else None,
        "mean_top1_actual_delta_log": float(per_date["chosen_actual_delta_log"].mean()) if not per_date.empty else None,
        "mean_candidate_delta_log": float(per_date["mean_candidate_delta_log"].mean()) if not per_date.empty else None,
        "mean_oracle_delta_log": float(per_date["oracle_delta_log"].mean()) if not per_date.empty else None,
    }
    return predictions, coefficients, summary


def main() -> int:
    args = _parser().parse_args()
    load_project_environment = common.load_project_environment if hasattr(common, "load_project_environment") else None
    if load_project_environment is not None:
        load_project_environment(args.env_file)
    else:
        from market_cycle_trader_api.core.environment import load_project_environment as _load
        _load(args.env_file)

    if args.mongo_uri:
        os.environ["MONGO_URI"] = str(args.mongo_uri)
        os.environ["MONGO_URL"] = str(args.mongo_uri)
    if args.database:
        os.environ["MONGO_DATABASE"] = str(args.database)

    mongo_uri = str(args.mongo_uri or os.getenv("MONGO_URL") or os.getenv("MONGO_URI") or "mongodb://localhost:27017").strip()
    database_name = str(args.database or os.getenv("MONGO_DATABASE") or "").strip()
    if not database_name:
        raise RuntimeError("MONGO_DATABASE is required.")
    common._assert_local_mongo(mongo_uri, bool(args.allow_remote_mongo))

    workers = max(1, int(args.workers or 1))
    horizon_sessions = max(1, int(args.horizon_sessions or 1))
    seed_assets = sequential.normalize_symbols(args.seed_assets)
    candidates = sequential.normalize_symbols(args.candidate_symbols)
    overlap = sorted(set(seed_assets).intersection(candidates))
    if overlap:
        raise RuntimeError(f"Candidates must be outside the seed universe: {overlap}")
    if len(seed_assets) < 2:
        raise RuntimeError("At least two seed assets are required.")
    if not candidates:
        raise RuntimeError("At least one candidate symbol is required.")

    client = MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=3_000,
        connectTimeoutMS=3_000,
        maxPoolSize=max(8, workers + 4),
        retryWrites=False,
    )
    try:
        client.admin.command("ping")
        db = client[database_name]
        strategy = common._strategy_document(db, args.strategy_sequence, args.strategy_id)
        stored_configuration = common._configuration(strategy)
        strategy_id = str(strategy.get("_id") or "").strip()
        strategy_sequence = int(strategy.get("strategy_sequence") or args.strategy_sequence)
        strategy_assets = sequential.normalize_symbols(list(stored_configuration.get("assets") or []))
        requested_symbols = [*seed_assets, *candidates]
        missing = sorted(set(requested_symbols).difference(strategy_assets))
        if missing:
            raise RuntimeError(f"Requested symbols are not present in Strategy #{strategy_sequence}: {missing}")

        history_start = common._normalize_date(args.history_start)
        configured_start = common._normalize_date(stored_configuration.get("start_date"))
        if history_start != configured_start:
            raise RuntimeError(
                f"--history-start must match Strategy start date: strategy={configured_start.date()}, requested={history_start.date()}"
            )
        snapshot_end = common._normalize_date(args.snapshot_end)
        decision_start = common._normalize_date(args.decision_start)
        decision_end = common._normalize_date(args.decision_end)
        validation_start = common._normalize_date(args.validation_start)

        config = BacktestRequest.model_validate(stored_configuration).model_copy(
            update={"assets": list(seed_assets), "end_date": snapshot_end.date().isoformat()}
        )
        identity = common._market_identity(stored_configuration)
        collection = db[common.ALPACA_MARKET_BARS_COLLECTION]
        expected_sessions = _utc_index(common._expected_sessions(history_start, snapshot_end))
        decision_sessions = select_decision_sessions(
            expected_sessions,
            decision_start,
            decision_end,
            int(args.decision_count),
            horizon_sessions,
        )

        output_dir = Path(
            args.output_dir
            or PROJECT_ROOT / "research_output" / f"contextual_marginal_signature_strategy_{strategy_sequence}_{snapshot_end.date().isoformat()}"
        ).resolve()
        if args.fresh_run:
            _safe_fresh(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        accelerator._OUTPUT_DIR = output_dir
        accelerator._INDEX_ROWS.clear()
        accelerator._REPLAY_COUNTER = 0
        accelerator.install_v110(parity_mode=False)

        manifest = {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "api_version": API_VERSION,
            "experiment": EXPERIMENT_NAME,
            "strategy_id": strategy_id,
            "strategy_sequence": strategy_sequence,
            "history_start": history_start.date().isoformat(),
            "snapshot_end": snapshot_end.date().isoformat(),
            "seed_assets": seed_assets,
            "candidate_symbols": candidates,
            "decision_dates": [date.date().isoformat() for date in decision_sessions],
            "decision_count": len(decision_sessions),
            "horizon_sessions": horizon_sessions,
            "validation_start": validation_start.date().isoformat(),
            "market_proxy": str(args.market_proxy).strip().upper(),
            "workers": workers,
            "label": "exact ending-capital delta for candidate admission over the future analysis window",
            "feature_policy": "only backward-looking ROTATION_FEATURES and relative/context features known at decision time",
            "market_data_mode": "local_mongodb_only",
            "preselector_used": False,
            "purpose": "small proof-of-structure experiment; not a production selector and not a generalization claim",
        }
        _write_json(output_dir / "contextual_signature_manifest.json", manifest)

        _log(
            f"Contextual Marginal Signature v1.0.2: seed={len(seed_assets)}, candidates={len(candidates)}, "
            f"dates={len(decision_sessions)}, horizon={horizon_sessions} sessions."
        )
        _log("Loading one frozen local MongoDB market panel; no transient Alpaca candidate fetch is used.")
        raw_frames = common._load_frames_once(
            collection,
            requested_symbols,
            identity,
            history_start,
            snapshot_end,
        )
        common._validate_complete_history(raw_frames, requested_symbols, expected_sessions)
        cleaned_frames = {
            symbol: validate_and_clean_bars(frame, config)
            for symbol, frame in raw_frames.items()
        }
        rotation_frames = {
            symbol: rotation.build_rotation_frame(frame.copy(), config)
            for symbol, frame in cleaned_frames.items()
        }

        seed_frames = {symbol: cleaned_frames[symbol] for symbol in seed_assets}
        all_rows: list[dict[str, Any]] = []
        abort_reason: str | None = None

        for date_index, decision_session in enumerate(decision_sessions, start=1):
            horizon_end = _horizon_end(expected_sessions, decision_session, horizon_sessions)
            _log(
                f"Decision {date_index}/{len(decision_sessions)}: {decision_session.date()} -> {horizon_end.date()}. "
                f"Running exact seed baseline."
            )
            baseline_request = _window_request(
                db=db,
                config=config,
                strategy_id=strategy_id,
                assets=seed_assets,
                reference_assets=seed_assets,
                candidate_assets=[],
                decision_session=decision_session,
                horizon_end=horizon_end,
            )
            baseline_started = time.perf_counter()
            baseline_metrics, baseline_sessions = discovery._run_rotation_replay(seed_frames, baseline_request)
            baseline_seconds = float(time.perf_counter() - baseline_started)
            baseline_capital = discovery._finite_number(baseline_metrics.get("ending_capital"))
            _log(f"Decision {decision_session.date()}: baseline={baseline_capital}, elapsed={baseline_seconds:.1f}s.")

            snapshots = {
                candidate: build_feature_snapshot(
                    candidate=candidate,
                    seed_assets=seed_assets,
                    rotation_frames=rotation_frames,
                    raw_frames=cleaned_frames,
                    decision=decision_session,
                    market_proxy=args.market_proxy,
                )
                for candidate in candidates
            }

            date_rows: list[dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=min(workers, len(candidates))) as executor:
                futures = {
                    executor.submit(
                        _evaluate_candidate_window,
                        db=db,
                        config=config,
                        strategy_id=strategy_id,
                        seed_assets=seed_assets,
                        seed_frames=seed_frames,
                        baseline_metrics=baseline_metrics,
                        baseline_sessions=baseline_sessions,
                        candidate=candidate,
                        candidate_frame=cleaned_frames[candidate],
                        decision_session=decision_session,
                        horizon_end=horizon_end,
                        feature_snapshot=snapshots[candidate],
                    ): candidate
                    for candidate in candidates
                }
                for future in as_completed(futures):
                    candidate = futures[future]
                    try:
                        row = future.result()
                    except Exception as exc:
                        row = {
                            "decision_date": decision_session.date().isoformat(),
                            "horizon_end": horizon_end.date().isoformat(),
                            "candidate": candidate,
                            "evaluation_status": "failed",
                            "error": str(exc)[:700],
                        }
                    date_rows.append(row)
                    all_rows.append(row)
                    delta = row.get("ending_capital_delta_rate")
                    delta_text = "n/a" if delta is None or pd.isna(delta) else f"{float(delta):+.4%}"
                    _log(
                        f"Decision {decision_session.date()} candidate {candidate}: "
                        f"{row.get('evaluation_status')} ΔCapital={delta_text}"
                    )
                    _write_frame(output_dir / "contextual_signature_dataset.csv", pd.DataFrame(all_rows))

            incomplete = [
                row for row in date_rows
                if str(row.get("evaluation_status") or "").lower() != "completed"
            ]
            if incomplete:
                abort_reason = (
                    f"ContextualSignatureDateIncomplete: {decision_session.date()} "
                    + ", ".join(
                        f"{row.get('candidate')}={row.get('evaluation_status')}:{row.get('error') or ''}"
                        for row in incomplete
                    )
                )
                _log(abort_reason)
                break

        dataset = pd.DataFrame(all_rows)
        _write_frame(output_dir / "contextual_signature_dataset.csv", dataset)

        expected_completed_dates = decision_sessions[: dataset["decision_date"].nunique()] if not dataset.empty else []
        issues = panel_completeness_issues(dataset, expected_completed_dates, candidates) if expected_completed_dates else []
        report = feature_report(dataset) if not dataset.empty else pd.DataFrame()
        _write_frame(output_dir / "contextual_signature_feature_report.csv", report)

        predictions, coefficients, validation = temporal_ols_validation(dataset, validation_start) if not dataset.empty else (
            pd.DataFrame(), pd.DataFrame(), {"status": "empty_dataset"}
        )
        _write_frame(output_dir / "contextual_signature_validation_predictions.csv", predictions)
        _write_frame(output_dir / "contextual_signature_ols_coefficients.csv", coefficients)

        completed = dataset.loc[dataset.get("evaluation_status", pd.Series(dtype=str)).astype(str).str.lower() == "completed"] if not dataset.empty else pd.DataFrame()
        summary = {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "api_version": API_VERSION,
            "status": "aborted" if abort_reason else "completed",
            "abort_reason": abort_reason,
            "panel_completeness_issues": issues,
            "rows": int(len(dataset)),
            "completed_rows": int(len(completed)),
            "completed_dates": int(completed["decision_date"].nunique()) if not completed.empty else 0,
            "candidate_count": len(candidates),
            "positive_label_rate": float((completed["delta_log_capital"] > 0).mean()) if not completed.empty else None,
            "mean_delta_log_capital": float(completed["delta_log_capital"].mean()) if not completed.empty else None,
            "median_delta_log_capital": float(completed["delta_log_capital"].median()) if not completed.empty else None,
            "top_absolute_spearman_features": (
                report.loc[:, ["feature", "spearman"]].head(8).to_dict(orient="records") if not report.empty else []
            ),
            "temporal_ols_validation": validation,
            "interpretation_rule": (
                "Evidence for a useful mathematical signature requires relationships discovered before validation_start "
                "to retain sign/rank information after validation_start. In-sample correlation alone is not sufficient."
            ),
        }
        _write_json(output_dir / "contextual_signature_summary.json", summary)
        _log(f"Completed with status={summary['status']}. Outputs: {output_dir}")
        return 2 if abort_reason else 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
