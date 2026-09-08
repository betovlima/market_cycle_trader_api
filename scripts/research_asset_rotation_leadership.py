from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

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
from market_cycle_trader_api.engine.absolute_utility_cash_gate import (  # noqa: E402
    absolute_utility_cash_gate_enabled,
)
from market_cycle_trader_api.engine.capital_rotation import (  # noqa: E402
    ROTATION_FEATURES,
    _build_walk_forward_folds,
    _simple_policy_growth,
    _utility_policy,
    prepare_rotation_panel,
)
from market_cycle_trader_api.engine.research_challengers import _lightgbm_fit_models  # noqa: E402
from market_cycle_trader_api.schemas.requests import BacktestExecutionRequest, BacktestRequest  # noqa: E402
from market_cycle_trader_api.services.model_research import apply_execution_profile  # noqa: E402

SCRIPT_VERSION = "asset-rotation-leadership-v1.1"
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
            "Revalidate all complete-history assets from scratch using the immutable Strategy "
            "LightGBM snapshot. Qualification combines (a) intrinsic OOS timing vs same-asset "
            "Buy & Hold and (b) useful OOS cross-sectional leadership. No full Strategy "
            "Backtest result is read or used before the universe is frozen."
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
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--allow-remote-mongo", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser


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


def _fit_model(
    symbol: str,
    frame: pd.DataFrame,
    train_dates: pd.DatetimeIndex,
    config: BacktestExecutionRequest,
    *,
    phase: str,
) -> Any:
    fitted = _lightgbm_fit_models(
        {symbol: frame},
        [symbol],
        train_dates,
        config,
        phase=phase,
    )
    model = fitted.get(symbol)
    if model is None:
        raise RuntimeError(f"{symbol}: LightGBM did not fit a model for {phase}.")
    return model


def _calibrate_and_fit(
    symbol: str,
    frame: pd.DataFrame,
    common_dates: pd.DatetimeIndex,
    fold: dict[str, Any],
    config: BacktestExecutionRequest,
) -> tuple[Callable[[pd.Timestamp, int, int], tuple[int, float]], Any, float, float, float]:
    fold_id = int(fold["fold_id"])

    # Match the Strategy engine: one repetition seed is kept constant across folds.
    rep_config = config.model_copy(update={"random_state": int(config.random_state)})
    train_dates = common_dates[: int(fold["train_end_index"])]
    calibration_dates = common_dates[
        int(fold["calibration_start_index"]): int(fold["calibration_end_index"])
    ]
    final_fit_dates = common_dates[: int(fold["final_fit_end_index"])]

    calibration_model = _fit_model(
        symbol,
        frame,
        train_dates,
        rep_config,
        phase=f"rotation_leadership_{symbol}_fold_{fold_id}_calibration",
    )
    calibration_models = {symbol: calibration_model}

    candidate_margins = tuple(
        float(value) for value in rep_config.rotation_switch_margin_candidates
    )
    if not candidate_margins:
        raise RuntimeError(f"{symbol}: rotation_switch_margin_candidates is empty.")

    best_candidate = candidate_margins[0]
    best_score = float("-inf")
    margin_config = (
        rep_config.model_copy(
            update={"strategy_mode": "COMPOUND_ROTATION_SWING_XGBOOST"}
        )
        if absolute_utility_cash_gate_enabled(rep_config)
        else rep_config
    )
    for candidate in candidate_margins:
        calibration_policy = _utility_policy(
            calibration_models,
            {symbol: frame},
            [symbol],
            margin_config,
            candidate,
        )
        score = _simple_policy_growth(
            calibration_policy,
            {symbol: frame},
            [symbol],
            calibration_dates,
            rep_config,
        )
        if float(score) > best_score:
            best_score = float(score)
            best_candidate = float(candidate)

    final_model = _fit_model(
        symbol,
        frame,
        final_fit_dates,
        rep_config,
        phase=f"rotation_leadership_{symbol}_fold_{fold_id}_final",
    )
    effective_margin = max(
        float(rep_config.rotation_switch_margin), float(best_candidate)
    )
    policy = _utility_policy(
        {symbol: final_model},
        {symbol: frame},
        [symbol],
        rep_config,
        effective_margin,
        cash_gate_base_state=(
            {"position": 0, "holding_days": 0, "pending_sample": None}
            if absolute_utility_cash_gate_enabled(rep_config)
            else None
        ),
        fold_id=fold_id,
        calibrated_switch_margin=float(best_candidate),
    )
    return (
        policy,
        final_model,
        float(best_candidate),
        float(effective_margin),
        float(best_score),
    )


def _prediction_rows(
    symbol: str,
    frame: pd.DataFrame,
    fold: dict[str, Any],
    model: Any,
) -> list[dict[str, Any]]:
    fold_id = int(fold["fold_id"])
    decision_dates = pd.DatetimeIndex(fold["decision_dates"]).sort_values()[:-1]
    dates = decision_dates.intersection(pd.DatetimeIndex(frame.index))
    if dates.empty:
        return []

    features = frame.loc[dates, ROTATION_FEATURES]
    valid = ~features.isna().any(axis=1)
    features = features.loc[valid]
    if features.empty:
        return []
    dates = pd.DatetimeIndex(features.index)
    predicted = np.asarray(model.predict(features), dtype=float)
    realized_utility = pd.to_numeric(
        frame.loc[dates, "forward_risk_adjusted_utility"], errors="coerce"
    ).to_numpy(dtype=float)
    realized_net = pd.to_numeric(
        frame.loc[dates, "forward_net_log_return"], errors="coerce"
    ).to_numpy(dtype=float)

    rows: list[dict[str, Any]] = []
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
    return rows


def _analyse_asset(
    symbol: str,
    source: str,
    raw: pd.DataFrame,
    frame: pd.DataFrame,
    common_dates: pd.DatetimeIndex,
    folds: list[dict[str, Any]],
    config: BacktestExecutionRequest,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    started = time.perf_counter()
    fold_metrics: list[Any] = []
    predictions: list[dict[str, Any]] = []

    for fold in folds:
        policy, final_model, calibrated, effective, calibration_score = _calibrate_and_fit(
            symbol,
            frame,
            common_dates,
            fold,
            config,
        )
        fold_metric = timing._simulate_fold(
            symbol,
            raw,
            frame,
            fold,
            policy,
            config,
            calibrated,
            effective,
            calibration_score,
        )
        fold_metrics.append(fold_metric)
        predictions.extend(_prediction_rows(symbol, frame, fold, final_model))

    aggregate = timing._aggregate(symbol, source, fold_metrics)
    aggregate["elapsed_seconds"] = float(time.perf_counter() - started)
    return (
        aggregate,
        [asdict(row) for row in fold_metrics],
        predictions,
    )


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

    client.close()
    _log(
        "MongoDB closed. All qualification below runs from the immutable in-memory market snapshot. "
        f"model={family}, settings_hash={settings_hash}."
    )

    _log("Building the exact Strategy rotation feature panel and walk-forward folds...")
    aligned_frames, common_dates = prepare_rotation_panel(research_frames, config)
    symbols = sorted(aligned_frames)
    folds = _build_walk_forward_folds(common_dates, config)
    if not folds:
        raise RuntimeError("No walk-forward folds were produced.")

    intrinsic_path = output_dir / "intrinsic_timing_summary.csv"
    fold_path = output_dir / "intrinsic_timing_folds.csv"
    raw_prediction_path = output_dir / "leadership_predictions_raw.csv"

    intrinsic_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    completed: set[str] = set()

    if not args.no_resume and intrinsic_path.exists() and raw_prediction_path.exists():
        intrinsic_existing = pd.read_csv(intrinsic_path)
        prediction_existing = pd.read_csv(raw_prediction_path)
        fold_existing = (
            pd.read_csv(fold_path) if fold_path.exists() else pd.DataFrame()
        )
        if not intrinsic_existing.empty:
            intrinsic_existing["symbol"] = (
                intrinsic_existing["symbol"].astype(str).str.upper()
            )
        if not prediction_existing.empty:
            prediction_existing["symbol"] = (
                prediction_existing["symbol"].astype(str).str.upper()
            )

        intrinsic_symbols = set(intrinsic_existing.get("symbol", []))
        prediction_fold_counts = (
            prediction_existing.groupby("symbol")["fold"].nunique().to_dict()
            if not prediction_existing.empty
            else {}
        )
        completed = {
            symbol
            for symbol in symbols
            if symbol in intrinsic_symbols
            and int(prediction_fold_counts.get(symbol, 0)) == len(folds)
        }
        if completed:
            intrinsic_rows = intrinsic_existing[
                intrinsic_existing["symbol"].isin(completed)
            ].to_dict(orient="records")
            prediction_rows = prediction_existing[
                prediction_existing["symbol"].isin(completed)
            ].to_dict(orient="records")
            if not fold_existing.empty and "symbol" in fold_existing.columns:
                fold_existing["symbol"] = fold_existing["symbol"].astype(str).str.upper()
                fold_rows = fold_existing[
                    fold_existing["symbol"].isin(completed)
                ].to_dict(orient="records")
            _log(f"Resume: {len(completed)}/{len(symbols)} assets already complete.")

    pending = [symbol for symbol in symbols if symbol not in completed]
    _log(
        f"Point-in-time intrinsic timing + leadership model generation: "
        f"pending={len(pending)}, workers={min(workers, max(1, len(pending)))}."
    )
    with ThreadPoolExecutor(
        max_workers=min(workers, max(1, len(pending)))
    ) as executor:
        futures = {
            executor.submit(
                _analyse_asset,
                symbol,
                "strategy_existing" if symbol in baseline_set else "candidate",
                research_frames[symbol],
                aligned_frames[symbol],
                common_dates,
                folds,
                config,
            ): symbol
            for symbol in pending
        }
        done_count = len(completed)
        for future in as_completed(futures):
            symbol = futures[future]
            aggregate, asset_folds, asset_predictions = future.result()

            intrinsic_rows = [
                row for row in intrinsic_rows
                if str(row.get("symbol") or "").upper() != symbol
            ]
            fold_rows = [
                row for row in fold_rows
                if str(row.get("symbol") or "").upper() != symbol
            ]
            prediction_rows = [
                row for row in prediction_rows
                if str(row.get("symbol") or "").upper() != symbol
            ]
            intrinsic_rows.append(aggregate)
            fold_rows.extend(asset_folds)
            prediction_rows.extend(asset_predictions)
            done_count += 1

            intrinsic_frame = pd.DataFrame(intrinsic_rows).sort_values(
                "symbol"
            ).reset_index(drop=True)
            fold_frame = pd.DataFrame(fold_rows).sort_values(
                ["symbol", "fold"]
            ).reset_index(drop=True)
            prediction_frame = pd.DataFrame(prediction_rows).sort_values(
                ["fold", "timestamp", "symbol"]
            ).reset_index(drop=True)
            _write_csv(intrinsic_path, intrinsic_frame)
            _write_csv(fold_path, fold_frame)
            _write_csv(raw_prediction_path, prediction_frame)

            _log(
                f"Asset {done_count}/{len(symbols)} - {symbol}: "
                f"beat_BH={float(aggregate['beat_buy_hold_fold_rate']):.1%}, "
                f"timing={float(aggregate['compound_oos_timing_return']):.2%}, "
                f"BH={float(aggregate['compound_oos_buy_hold_return']):.2%}, "
                f"excess={float(aggregate['compound_oos_excess_return']):.2%}."
            )

    intrinsic = pd.read_csv(intrinsic_path)
    intrinsic["symbol"] = intrinsic["symbol"].astype(str).str.upper()
    intrinsic_records: list[dict[str, Any]] = []
    for row in intrinsic.to_dict(orient="records"):
        qualified, failures = _intrinsic_qualification(row)
        row["intrinsic_timing_qualified"] = qualified
        row["intrinsic_timing_fail_reasons"] = "|".join(failures)
        intrinsic_records.append(row)
    intrinsic = pd.DataFrame(intrinsic_records)
    _write_csv(intrinsic_path, intrinsic)

    raw_predictions = pd.read_csv(raw_prediction_path)
    ranked = _rank_predictions(raw_predictions)
    _write_csv(output_dir / "leadership_ranked_predictions.csv", ranked)

    leadership = _leadership_summary(ranked, symbols, len(folds))
    leadership_records: list[dict[str, Any]] = []
    for row in leadership.to_dict(orient="records"):
        qualified, failures = _leadership_qualification(row)
        row["leadership_qualified"] = qualified
        row["leadership_fail_reasons"] = "|".join(failures)
        leadership_records.append(row)
    leadership = pd.DataFrame(leadership_records)
    _write_csv(output_dir / "leadership_summary.csv", leadership)

    combined = intrinsic.merge(
        leadership, on="symbol", how="inner", validate="one_to_one"
    )
    if len(combined) != len(symbols):
        raise RuntimeError(
            f"Qualification merge produced {len(combined)} rows for {len(symbols)} assets."
        )

    combined["qualified"] = (
        combined["intrinsic_timing_qualified"].astype(bool)
        | combined["leadership_qualified"].astype(bool)
    )
    combined["qualification_path"] = np.select(
        [
            combined["intrinsic_timing_qualified"].astype(bool)
            & combined["leadership_qualified"].astype(bool),
            combined["intrinsic_timing_qualified"].astype(bool),
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
            "intrinsic_timing_qualified",
            "leadership_event_realized_utility_median",
            "compound_oos_excess_return",
            "symbol",
        ],
        ascending=[False, False, False, False, False, True],
    ).reset_index(drop=True)
    combined["research_rank"] = np.arange(1, len(combined) + 1)
    _write_csv(output_dir / "asset_rotation_qualification.csv", combined)

    qualified_assets = combined.loc[
        combined["qualified"], "symbol"
    ].astype(str).tolist()
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
        "schema_version": 4,
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
        "selection_rule": "intrinsic_timing_qualified OR rotation_leadership_qualified",
        "intrinsic_rule": {
            "compound_oos_timing_return_gt": 0.0,
            "compound_oos_excess_return_gt": 0.0,
            "median_fold_excess_return_gt": 0.0,
            "beat_buy_hold_fold_rate_gt": 0.5,
            "market_vs_cash_auxiliary_gate_used": False,
        },
        "leadership_rule": {
            "top3_event_definition": (
                "first OOS session of each contiguous predicted Top-3 leadership episode"
            ),
            "event_count_gt": 0,
            "event_fold_rate_gt": 0.5,
            "event_median_realized_utility_gt": 0.0,
            "event_positive_utility_rate_gt": 0.5,
            "event_mean_forward_net_log_return_gt": 0.0,
        },
        "strategy_seed_policy": (
            "same immutable Strategy repetition seed across all walk-forward folds"
        ),
        "baseline_distribution_used_as_threshold": False,
        "correlation_used_for_selection": False,
        "full_strategy_backtest_used_for_selection": False,
        "prior_full_strategy_backtest_results_read": False,
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
        "ranking_sha256": ranking_sha256,
        "qualified_assets_sha256": qualified_assets_sha256,
    }
    frozen["decision_snapshot_sha256"] = _sha256_json(frozen)
    frozen_path = output_dir / "rotation_leadership_snapshot_frozen.json"
    _write_json(frozen_path, frozen)

    manifest = {
        "schema_version": 4,
        "experiment": EXPERIMENT_NAME,
        "script_version": SCRIPT_VERSION,
        "data_source": "local_mongodb_once_then_ram_cpu",
        "mongo_writes": False,
        "alpaca_network_used": False,
        "full_strategy_backtest_run": False,
        "full_strategy_backtest_used_for_selection": False,
        "prior_full_strategy_backtest_results_read": False,
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
                combined["intrinsic_timing_qualified"].sum()
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
        "against this immutable universe."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
