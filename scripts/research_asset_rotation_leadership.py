from __future__ import annotations

import argparse
import hashlib
import json
import os
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
import research_asset_timing_vs_buyhold as timing  # noqa: E402
from research_asset_timing_vs_buyhold_execution import _immutable_model_snapshot  # noqa: E402
from market_cycle_trader_api.engine.capital_rotation import (  # noqa: E402
    ROTATION_FEATURES,
    _build_walk_forward_folds,
    prepare_rotation_panel,
)
from market_cycle_trader_api.engine.research_challengers import _lightgbm_fit_models  # noqa: E402
from market_cycle_trader_api.schemas.requests import BacktestExecutionRequest, BacktestRequest  # noqa: E402
from market_cycle_trader_api.services.model_research import apply_execution_profile  # noqa: E402

SCRIPT_VERSION = "asset-rotation-leadership-v1.0"
EXPERIMENT_NAME = "point_in_time_intrinsic_timing_plus_rotation_leadership"
DEFAULT_WORKERS = 4


def _log(message: str) -> None:
    stamp = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{stamp}] {message}", flush=True)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Revalidate the complete research universe using two independent OOS evidences: "
            "(1) intrinsic same-asset timing vs Buy & Hold already frozen by the previous study, "
            "and (2) useful cross-sectional LightGBM leadership episodes. "
            "No full Strategy Backtest is used for qualification."
        )
    )
    parser.add_argument("--strategy-sequence", type=int, default=10)
    parser.add_argument("--strategy-id", default=None)
    parser.add_argument("--snapshot-end", required=True)
    parser.add_argument("--mongo-uri", default=None)
    parser.add_argument("--database", default=None)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--candidate-symbols", nargs="*", default=None)
    parser.add_argument("--timing-output-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--allow-remote-mongo", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser


def _verify_timing_snapshot(
    frozen: dict[str, Any],
    strategy: dict[str, Any],
    snapshot_end: pd.Timestamp,
) -> None:
    expected = str(frozen.get("decision_snapshot_sha256") or "")
    canonical = dict(frozen)
    canonical.pop("decision_snapshot_sha256", None)
    actual = _sha256_json(canonical)
    if not expected or expected != actual:
        raise RuntimeError("Prior timing snapshot hash mismatch.")

    checks = {
        "strategy_id": str(frozen.get("strategy_id") or "")
        == str(strategy.get("_id") or ""),
        "strategy_revision": int(frozen.get("strategy_revision") or 0)
        == int(strategy.get("revision") or 0),
        "strategy_configuration_hash": str(
            frozen.get("strategy_configuration_hash") or ""
        )
        == str(strategy.get("configuration_hash") or ""),
        "snapshot_end": str(frozen.get("snapshot_end") or "")
        == snapshot_end.date().isoformat(),
        "backtest_independent": frozen.get("full_strategy_backtest_used_for_selection")
        is False,
    }
    failed = [name for name, accepted in checks.items() if not accepted]
    if failed:
        raise RuntimeError(
            "Prior timing evidence does not match this Strategy/snapshot: "
            + ", ".join(failed)
        )


def _execution_config(
    configuration: dict[str, Any],
    research_assets: list[str],
    baseline_assets: list[str],
    snapshot_end: pd.Timestamp,
    family: str,
    settings: dict[str, Any],
) -> BacktestExecutionRequest:
    base = BacktestRequest.model_validate(configuration)
    locked = base.model_copy(
        update={
            "assets": list(research_assets),
            "end_date": snapshot_end.date().isoformat(),
        }
    )
    locked = apply_execution_profile(locked, family, settings)
    return BacktestExecutionRequest.model_validate(
        {
            **locked.model_dump(mode="python"),
            "analysis_start_date": locked.start_date,
            "analysis_end_date": snapshot_end.date().isoformat(),
            "calendar_anchor_assets": list(baseline_assets),
            "research_reference_assets": list(baseline_assets),
            "research_candidate_assets": [
                symbol for symbol in research_assets if symbol not in set(baseline_assets)
            ],
            "research_model_family": family,
            "research_model_settings": settings,
            "research_market_data_mode": "database_only",
        }
    )


def _intrinsic_qualification(row: dict[str, Any]) -> tuple[bool, list[str]]:
    checks = {
        "timing_compound_return_positive": float(row["compound_oos_timing_return"]) > 0.0,
        "timing_beats_buy_hold_compound": float(row["compound_oos_excess_return"]) > 0.0,
        "median_fold_excess_positive": float(row["median_fold_excess_return"]) > 0.0,
        "majority_of_folds_beat_buy_hold": float(row["beat_buy_hold_fold_rate"]) > 0.5,
    }
    failed = [name for name, accepted in checks.items() if not accepted]
    return bool(all(checks.values())), failed


def _leadership_qualification(row: dict[str, Any]) -> tuple[bool, list[str]]:
    checks = {
        "leadership_event_exists": int(row["leadership_event_count"]) > 0,
        "leadership_present_in_majority_of_folds": float(
            row["leadership_event_fold_rate"]
        )
        > 0.5,
        "event_median_realized_utility_positive": float(
            row["leadership_event_realized_utility_median"]
        )
        > 0.0,
        "majority_of_events_have_positive_realized_utility": float(
            row["leadership_event_positive_utility_rate"]
        )
        > 0.5,
        "event_mean_forward_net_log_return_positive": float(
            row["leadership_event_forward_net_log_return_mean"]
        )
        > 0.0,
    }
    failed = [name for name, accepted in checks.items() if not accepted]
    return bool(all(checks.values())), failed


def _fit_one_asset(
    symbol: str,
    frame: pd.DataFrame,
    train_dates: pd.DatetimeIndex,
    config: BacktestExecutionRequest,
    fold_id: int,
) -> Any:
    fitted = _lightgbm_fit_models(
        {symbol: frame},
        [symbol],
        train_dates,
        config,
        phase=f"rotation_leadership_{symbol}_fold_{fold_id}_final",
    )
    model = fitted.get(symbol)
    if model is None:
        raise RuntimeError(f"{symbol}: LightGBM model was not fitted for fold {fold_id}.")
    return model


def _predict_fold(
    fold: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    common_dates: pd.DatetimeIndex,
    symbols: list[str],
    config: BacktestExecutionRequest,
    workers: int,
) -> pd.DataFrame:
    fold_id = int(fold["fold_id"])
    final_fit_dates = common_dates[: int(fold["final_fit_end_index"])]
    rep_config = config.model_copy(update={"random_state": int(config.random_state) + fold_id})

    _log(
        f"Fold {fold_id}: fitting {len(symbols)} final LightGBM asset models "
        f"with {min(workers, len(symbols))} workers..."
    )
    models: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(symbols)))) as executor:
        futures = {
            executor.submit(
                _fit_one_asset,
                symbol,
                frames[symbol],
                final_fit_dates,
                rep_config,
                fold_id,
            ): symbol
            for symbol in symbols
        }
        completed = 0
        for future in as_completed(futures):
            symbol = futures[future]
            models[symbol] = future.result()
            completed += 1
            if completed == 1 or completed % 10 == 0 or completed == len(symbols):
                _log(f"Fold {fold_id}: trained {completed}/{len(symbols)} models.")

    decision_dates = pd.DatetimeIndex(fold["decision_dates"]).sort_values()[:-1]
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        frame = frames[symbol]
        dates = decision_dates.intersection(pd.DatetimeIndex(frame.index))
        if dates.empty:
            continue
        features = frame.loc[dates, ROTATION_FEATURES]
        valid_mask = ~features.isna().any(axis=1)
        features = features.loc[valid_mask]
        if features.empty:
            continue
        dates = pd.DatetimeIndex(features.index)
        predicted = np.asarray(models[symbol].predict(features), dtype=float)
        realized_utility = pd.to_numeric(
            frame.loc[dates, "forward_risk_adjusted_utility"], errors="coerce"
        ).to_numpy(dtype=float)
        realized_net = pd.to_numeric(
            frame.loc[dates, "forward_net_log_return"], errors="coerce"
        ).to_numpy(dtype=float)

        for timestamp, utility, future_utility, future_net in zip(
            dates, predicted, realized_utility, realized_net, strict=True
        ):
            if not np.isfinite(utility):
                continue
            rows.append(
                {
                    "fold": fold_id,
                    "timestamp": pd.Timestamp(timestamp).isoformat(),
                    "symbol": symbol,
                    "predicted_utility": float(utility),
                    "realized_utility": (
                        float(future_utility) if np.isfinite(future_utility) else None
                    ),
                    "forward_net_log_return": (
                        float(future_net) if np.isfinite(future_net) else None
                    ),
                }
            )
    return pd.DataFrame(rows)


def _rank_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["predicted_rank"] = frame.groupby(
        ["fold", "timestamp"]
    )["predicted_utility"].rank(method="min", ascending=False)
    group_size = frame.groupby(["fold", "timestamp"])["symbol"].transform("count")
    denominator = (group_size - 1).clip(lower=1)
    frame["predicted_rank_percentile"] = 1.0 - (
        (frame["predicted_rank"] - 1.0) / denominator
    )
    frame["predicted_top1"] = frame["predicted_rank"] <= 1.0
    frame["predicted_top3"] = frame["predicted_rank"] <= 3.0
    frame = frame.sort_values(["symbol", "fold", "timestamp"]).reset_index(drop=True)
    previous_top3 = frame.groupby(["symbol", "fold"])["predicted_top3"].shift(
        1, fill_value=False
    )
    frame["leadership_event_start"] = frame["predicted_top3"] & ~previous_top3
    return frame


def _leadership_summary(
    ranked: pd.DataFrame,
    symbols: list[str],
    fold_count: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        asset = ranked.loc[ranked["symbol"] == symbol].copy()
        events = asset.loc[asset["leadership_event_start"]].copy()
        top3 = asset.loc[asset["predicted_top3"]].copy()

        event_utility = pd.to_numeric(
            events["realized_utility"], errors="coerce"
        ).dropna()
        event_net = pd.to_numeric(
            events["forward_net_log_return"], errors="coerce"
        ).dropna()
        top3_utility = pd.to_numeric(top3["realized_utility"], errors="coerce").dropna()
        top3_net = pd.to_numeric(
            top3["forward_net_log_return"], errors="coerce"
        ).dropna()

        event_folds = int(events["fold"].nunique()) if not events.empty else 0
        observations = int(len(asset))
        top1_count = int(asset["predicted_top1"].sum())
        top3_count = int(asset["predicted_top3"].sum())
        rows.append(
            {
                "symbol": symbol,
                "oos_observation_count": observations,
                "top1_count": top1_count,
                "top1_frequency": float(top1_count / max(1, observations)),
                "top3_count": top3_count,
                "top3_frequency": float(top3_count / max(1, observations)),
                "leadership_event_count": int(len(events)),
                "leadership_event_fold_count": event_folds,
                "leadership_event_fold_rate": float(event_folds / max(1, fold_count)),
                "leadership_event_realized_utility_mean": (
                    float(event_utility.mean()) if not event_utility.empty else 0.0
                ),
                "leadership_event_realized_utility_median": (
                    float(event_utility.median()) if not event_utility.empty else 0.0
                ),
                "leadership_event_positive_utility_rate": (
                    float((event_utility > 0.0).mean()) if not event_utility.empty else 0.0
                ),
                "leadership_event_forward_net_log_return_mean": (
                    float(event_net.mean()) if not event_net.empty else 0.0
                ),
                "leadership_event_positive_net_return_rate": (
                    float((event_net > 0.0).mean()) if not event_net.empty else 0.0
                ),
                "top3_realized_utility_mean": (
                    float(top3_utility.mean()) if not top3_utility.empty else 0.0
                ),
                "top3_positive_utility_rate": (
                    float((top3_utility > 0.0).mean()) if not top3_utility.empty else 0.0
                ),
                "top3_forward_net_log_return_mean": (
                    float(top3_net.mean()) if not top3_net.empty else 0.0
                ),
                "predicted_rank_percentile_mean": float(
                    pd.to_numeric(
                        asset["predicted_rank_percentile"], errors="coerce"
                    ).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = _parser().parse_args()
    common.load_project_environment(args.env_file)
    os.environ.setdefault("MCT_MODEL_THREADS_OVERRIDE", "1")

    mongo_uri = str(
        args.mongo_uri
        or os.getenv("MONGO_URL")
        or os.getenv("MONGO_URI")
        or "mongodb://localhost:27017"
    ).strip()
    database_name = str(args.database or os.getenv("MONGO_DATABASE") or "").strip()
    if not database_name:
        raise RuntimeError("MONGO_DATABASE is required in .env or via --database.")
    common._assert_local_mongo(mongo_uri, bool(args.allow_remote_mongo))
    workers = max(1, int(args.workers or DEFAULT_WORKERS))

    client = MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=3_000,
        connectTimeoutMS=3_000,
        maxPoolSize=max(8, workers + 2),
        retryWrites=False,
    )
    client.admin.command("ping")
    db = client[database_name]

    strategy = common._strategy_document(db, args.strategy_sequence, args.strategy_id)
    configuration = common._configuration(strategy)
    base_config = BacktestRequest.model_validate(configuration)
    baseline_assets = list(
        dict.fromkeys(
            str(symbol).strip().upper()
            for symbol in base_config.assets
            if str(symbol).strip()
        )
    )
    baseline_set = set(baseline_assets)
    snapshot_end = common._normalize_date(args.snapshot_end)
    start_date = common._normalize_date(base_config.start_date)
    identity = common._market_identity(configuration)
    collection = db[common.ALPACA_MARKET_BARS_COLLECTION]

    if args.candidate_symbols:
        external = sorted(
            {
                str(item).strip().upper()
                for item in args.candidate_symbols
                if str(item).strip()
            }
            - baseline_set
        )
        candidate_source = "explicit_candidate_symbols"
    else:
        cached = {
            str(item).strip().upper()
            for item in collection.distinct("symbol", identity)
            if str(item).strip()
        }
        external = sorted(cached - baseline_set)
        candidate_source = "all_local_mongo_cached_symbols"

    universe = [*baseline_assets, *external]
    _log(
        f"Loading Full Strategy History for existing={len(baseline_assets)}, "
        f"external_cached={len(external)}."
    )
    frames = timing._load_frames_allow_incomplete(
        collection, universe, identity, start_date, snapshot_end
    )
    expected = common._expected_sessions(start_date, snapshot_end)
    history_rows, complete_assets = timing._history_diagnostics(
        frames, universe, baseline_set, expected
    )
    complete_set = set(complete_assets)
    missing_existing = [
        symbol for symbol in baseline_assets if symbol not in complete_set
    ]
    if missing_existing:
        raise RuntimeError(
            "Existing Strategy assets failed Full Strategy History: "
            + ", ".join(missing_existing)
        )

    valid_candidates = [symbol for symbol in external if symbol in complete_set]
    research_assets = [*baseline_assets, *valid_candidates]
    research_frames = {symbol: frames[symbol] for symbol in research_assets}

    family, settings, settings_hash = _immutable_model_snapshot(strategy)
    config = _execution_config(
        configuration,
        research_assets,
        baseline_assets,
        snapshot_end,
        family,
        settings,
    )

    strategy_sequence = int(
        strategy.get("strategy_sequence") or args.strategy_sequence
    )
    output_dir = Path(
        args.output_dir
        or PROJECT_ROOT
        / "research_output"
        / f"asset_rotation_leadership_strategy_{strategy_sequence}_{snapshot_end.date().isoformat()}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "asset_history_integrity.csv", pd.DataFrame(history_rows))

    timing_output_dir = Path(
        args.timing_output_dir
        or PROJECT_ROOT
        / "research_output"
        / f"asset_timing_strategy_{strategy_sequence}_{snapshot_end.date().isoformat()}"
    ).resolve()
    timing_snapshot_path = timing_output_dir / "timing_validation_snapshot_frozen.json"
    timing_summary_path = timing_output_dir / "asset_timing_summary.csv"
    if not timing_snapshot_path.exists() or not timing_summary_path.exists():
        raise RuntimeError(
            "Prior point-in-time timing evidence is required and was not found. "
            f"Expected {timing_snapshot_path} and {timing_summary_path}."
        )
    timing_snapshot = json.loads(timing_snapshot_path.read_text(encoding="utf-8"))
    _verify_timing_snapshot(timing_snapshot, strategy, snapshot_end)
    timing_summary = pd.read_csv(timing_summary_path)
    timing_summary["symbol"] = timing_summary["symbol"].astype(str).str.upper()
    timing_symbols = set(timing_summary["symbol"])
    missing_timing = [
        symbol for symbol in research_assets if symbol not in timing_symbols
    ]
    extra_timing = sorted(timing_symbols - set(research_assets))
    if missing_timing or extra_timing:
        raise RuntimeError(
            "Prior timing evidence universe differs from this research universe. "
            f"missing={missing_timing}, extra={extra_timing}."
        )

    client.close()
    _log(
        "MongoDB closed. Leadership training uses only the in-memory OHLCV snapshot. "
        f"model={family}, settings_hash={settings_hash}."
    )

    _log("Building the exact Strategy rotation feature panel...")
    aligned_frames, common_dates = prepare_rotation_panel(research_frames, config)
    symbols = sorted(aligned_frames)
    folds = _build_walk_forward_folds(common_dates, config)
    if not folds:
        raise RuntimeError("No walk-forward folds were produced.")

    prediction_path = output_dir / "leadership_predictions.csv"
    existing_predictions = pd.DataFrame()
    completed_folds: set[int] = set()
    if not args.no_resume and prediction_path.exists():
        existing_predictions = pd.read_csv(prediction_path)
        if not existing_predictions.empty:
            for fold_id, group in existing_predictions.groupby("fold"):
                if set(group["symbol"].astype(str).str.upper()) == set(symbols):
                    completed_folds.add(int(fold_id))
            if completed_folds:
                _log(f"Resume: complete leadership folds already present={sorted(completed_folds)}.")

    prediction_frames: list[pd.DataFrame] = []
    if not existing_predictions.empty:
        prediction_frames.append(existing_predictions)

    for fold in folds:
        fold_id = int(fold["fold_id"])
        if fold_id in completed_folds:
            continue
        started = time.perf_counter()
        fold_predictions = _predict_fold(
            fold,
            aligned_frames,
            common_dates,
            symbols,
            config,
            workers,
        )
        prediction_frames = [
            frame
            for frame in prediction_frames
            if "fold" not in frame.columns
            or not (pd.to_numeric(frame["fold"], errors="coerce") == fold_id).any()
        ]
        prediction_frames.append(fold_predictions)
        combined_predictions = pd.concat(prediction_frames, ignore_index=True)
        combined_predictions = combined_predictions.sort_values(
            ["fold", "timestamp", "symbol"]
        ).reset_index(drop=True)
        _write_csv(prediction_path, combined_predictions)
        _log(
            f"Fold {fold_id}: leadership predictions completed in "
            f"{time.perf_counter() - started:.1f}s."
        )

    predictions = pd.read_csv(prediction_path)
    ranked = _rank_predictions(predictions)
    _write_csv(output_dir / "leadership_ranked_predictions.csv", ranked)

    leadership = _leadership_summary(ranked, symbols, len(folds))
    leadership_rows: list[dict[str, Any]] = []
    for row in leadership.to_dict(orient="records"):
        qualified, failures = _leadership_qualification(row)
        row["leadership_qualified"] = qualified
        row["leadership_fail_reasons"] = "|".join(failures)
        leadership_rows.append(row)
    leadership = pd.DataFrame(leadership_rows)
    _write_csv(output_dir / "leadership_summary.csv", leadership)

    timing_records: list[dict[str, Any]] = []
    for row in timing_summary.to_dict(orient="records"):
        corrected, failures = _intrinsic_qualification(row)
        row["intrinsic_timing_qualified_v2"] = corrected
        row["intrinsic_timing_fail_reasons_v2"] = "|".join(failures)
        timing_records.append(row)
    timing_v2 = pd.DataFrame(timing_records)
    _write_csv(output_dir / "timing_metrics_revalidated.csv", timing_v2)

    combined = timing_v2.merge(leadership, on="symbol", how="inner", validate="one_to_one")
    if len(combined) != len(research_assets):
        raise RuntimeError(
            f"Qualification merge produced {len(combined)} rows for "
            f"{len(research_assets)} research assets."
        )
    combined["qualified"] = (
        combined["intrinsic_timing_qualified_v2"].astype(bool)
        | combined["leadership_qualified"].astype(bool)
    )
    combined["qualification_path"] = np.select(
        [
            combined["intrinsic_timing_qualified_v2"].astype(bool)
            & combined["leadership_qualified"].astype(bool),
            combined["intrinsic_timing_qualified_v2"].astype(bool),
            combined["leadership_qualified"].astype(bool),
        ],
        ["intrinsic_and_leadership", "intrinsic_timing", "rotation_leadership"],
        default="rejected",
    )
    combined["source"] = np.where(
        combined["symbol"].isin(baseline_set), "strategy_existing", "candidate"
    )
    combined = combined.sort_values(
        [
            "qualified",
            "leadership_qualified",
            "intrinsic_timing_qualified_v2",
            "leadership_event_realized_utility_median",
            "compound_oos_excess_return",
            "symbol",
        ],
        ascending=[False, False, False, False, False, True],
    ).reset_index(drop=True)
    combined["research_rank"] = np.arange(1, len(combined) + 1)
    _write_csv(output_dir / "asset_rotation_qualification.csv", combined)

    qualified_assets = combined.loc[combined["qualified"], "symbol"].astype(str).tolist()
    qualified_set = set(qualified_assets)
    retained_existing = [
        symbol for symbol in baseline_assets if symbol in qualified_set
    ]
    removed_existing = [
        symbol for symbol in baseline_assets if symbol not in qualified_set
    ]
    added_candidates = [
        symbol for symbol in qualified_assets if symbol not in baseline_set
    ]
    rejected_candidates = [
        symbol for symbol in valid_candidates if symbol not in qualified_set
    ]

    ranking_records = combined.to_dict(orient="records")
    ranking_sha256 = _sha256_json(ranking_records)
    qualified_assets_sha256 = _sha256_json(qualified_assets)
    frozen = {
        "schema_version": 3,
        "experiment": EXPERIMENT_NAME,
        "script_version": SCRIPT_VERSION,
        "strategy_id": str(strategy.get("_id") or ""),
        "strategy_sequence": strategy_sequence,
        "strategy_revision": int(strategy.get("revision") or 0),
        "strategy_configuration_hash": strategy.get("configuration_hash"),
        "strategy_model_family": family,
        "strategy_model_settings_hash": settings_hash,
        "strategy_mode": str(base_config.strategy_mode),
        "strategy_start": start_date.date().isoformat(),
        "snapshot_end": snapshot_end.date().isoformat(),
        "candidate_source": candidate_source,
        "selection_rule": "intrinsic_timing_qualified_v2 OR rotation_leadership_qualified_v1",
        "intrinsic_rule": {
            "compound_oos_timing_return_positive": True,
            "compound_oos_excess_return_positive": True,
            "median_fold_excess_return_positive": True,
            "majority_folds_beat_buy_hold": True,
            "removed_market_vs_cash_auxiliary_gate": True,
        },
        "leadership_rule": {
            "top3_event_definition": (
                "first OOS session of each contiguous predicted Top-3 leadership episode"
            ),
            "event_exists": True,
            "event_fold_rate_gt": 0.5,
            "event_median_realized_utility_gt": 0.0,
            "event_positive_utility_rate_gt": 0.5,
            "event_mean_forward_net_log_return_gt": 0.0,
        },
        "baseline_distribution_used_as_threshold": False,
        "correlation_used_for_selection": False,
        "full_strategy_backtest_used_for_selection": False,
        "prior_full_strategy_backtest_results_used_for_selection": False,
        "same_asset_buy_hold_used_as_intrinsic_benchmark": True,
        "cross_sectional_predicted_utility_used_for_leadership": True,
        "existing_strategy_assets_revalidated": True,
        "original_assets": baseline_assets,
        "complete_external_candidates": valid_candidates,
        "qualified_assets": qualified_assets,
        "retained_existing_assets": retained_existing,
        "removed_existing_assets": removed_existing,
        "added_candidate_assets": added_candidates,
        "rejected_candidate_assets": rejected_candidates,
        "prior_timing_snapshot_sha256": timing_snapshot["decision_snapshot_sha256"],
        "ranking_sha256": ranking_sha256,
        "qualified_assets_sha256": qualified_assets_sha256,
    }
    frozen["decision_snapshot_sha256"] = _sha256_json(frozen)
    frozen_path = output_dir / "rotation_leadership_snapshot_frozen.json"
    _write_json(frozen_path, frozen)

    manifest = {
        "schema_version": 3,
        "experiment": EXPERIMENT_NAME,
        "script_version": SCRIPT_VERSION,
        "data_source": "local_mongodb_then_ram_cpu",
        "mongo_writes": False,
        "alpaca_network_used": False,
        "full_strategy_backtest_run": False,
        "full_strategy_backtest_used_for_selection": False,
        "prior_full_strategy_backtest_results_used_for_selection": False,
        "full_strategy_history_required": True,
        "expected_xnys_sessions": int(len(expected)),
        "walk_forward_fold_count": int(len(folds)),
        "rotation_features": list(ROTATION_FEATURES),
        "model_family": family,
        "model_settings_hash": settings_hash,
        "workers": workers,
        "model_threads_per_worker": int(
            os.getenv("MCT_MODEL_THREADS_OVERRIDE") or 1
        ),
        "result_counts": {
            "original_strategy_assets": len(baseline_assets),
            "complete_external_candidates": len(valid_candidates),
            "evaluated": len(combined),
            "intrinsic_timing_qualified": int(
                combined["intrinsic_timing_qualified_v2"].sum()
            ),
            "rotation_leadership_qualified": int(
                combined["leadership_qualified"].sum()
            ),
            "qualified_union": len(qualified_assets),
            "retained_existing": len(retained_existing),
            "removed_existing": len(removed_existing),
            "added_candidates": len(added_candidates),
            "rejected_candidates": len(rejected_candidates),
        },
        "ranking_sha256": ranking_sha256,
        "qualified_assets_sha256": qualified_assets_sha256,
        "decision_snapshot_sha256": frozen["decision_snapshot_sha256"],
    }
    _write_json(output_dir / "experiment_manifest.json", manifest)

    _log(
        f"Frozen universe: {len(qualified_assets)} assets = "
        f"{len(retained_existing)} retained existing + {len(added_candidates)} new."
    )
    if removed_existing:
        _log("Existing rejected by both evidences: " + ", ".join(removed_existing))
    if added_candidates:
        _log("New qualified assets: " + ", ".join(added_candidates))
    _log(f"Frozen snapshot: {frozen_path}")
    _log(
        "Qualification finished. Only now may one full Strategy Backtest be run "
        "against the frozen universe."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
