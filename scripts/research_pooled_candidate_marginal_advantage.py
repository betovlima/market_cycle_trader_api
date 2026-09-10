from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymongo import MongoClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for _path in (SRC_ROOT, SCRIPT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import research_asset_signature_leave_one_out as common  # noqa: E402
import research_asset_timing_vs_buyhold as timing  # noqa: E402
import research_asset_rotation_leadership as leadership  # noqa: E402
import research_asset_rotation_independent_then_validate as independent  # noqa: E402
import research_asset_marginal_rotation_validation as validation  # noqa: E402
import research_windows_file_io as file_io  # noqa: E402
from research_asset_timing_vs_buyhold_execution import _immutable_model_snapshot  # noqa: E402
from market_cycle_trader_api.engine.absolute_utility_cash_gate import (  # noqa: E402
    absolute_utility_cash_gate_enabled,
)
from market_cycle_trader_api.engine.capital_rotation import (  # noqa: E402
    ROTATION_FEATURES,
    _build_walk_forward_folds,
    _model_utilities,
    _simple_policy_growth,
    _training_transition_log_return,
    _utility_policy,
    prepare_rotation_panel,
)
from market_cycle_trader_api.engine.research_challengers import _lightgbm_fit_models  # noqa: E402
from market_cycle_trader_api.services.pooled_candidate_marginal_advantage import (  # noqa: E402
    choose_daily_candidate_overrides,
    fit_pooled_candidate_marginal_model,
    score_candidate_samples,
    summarize_override_decisions,
)

SCRIPT_VERSION = "pooled-candidate-marginal-advantage-v1.0.0"
EXPERIMENT = "pooled_asset_agnostic_candidate_marginal_advantage"
DEFAULT_VALIDATION_SESSIONS = 252

BASE_FEATURES = [
    "candidate_score",
    "candidate_score_z_train",
    "candidate_score_percentile_train",
    "baseline_target_score",
    "baseline_target_score_z_cross_section",
    "baseline_current_score",
    "baseline_target_vs_current_gap",
    "baseline_score_mean",
    "baseline_score_std",
    "baseline_positive_score_fraction",
    "baseline_holding_days",
    "baseline_target_is_cash",
]
DELTA_FEATURES = [f"delta_{name}" for name in ROTATION_FEATURES]
MODEL_FEATURES = [*BASE_FEATURES, *DELTA_FEATURES]
TARGET_COLUMN = "marginal_next_session_net_log_return"


def _log(message: str) -> None:
    stamp = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{stamp}] {message}", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Learn one pooled, asset-agnostic model of the exact next-session marginal "
            "economic advantage of an external candidate versus the immutable Strategy "
            "baseline action. This v1 is a research signal test; it does not replace the "
            "production policy or run a full candidate-overlay capital backtest."
        )
    )
    parser.add_argument("--strategy-sequence", type=int, default=10)
    parser.add_argument("--strategy-id", default=None)
    parser.add_argument("--history-start", required=True)
    parser.add_argument("--snapshot-end", required=True)
    parser.add_argument("--validation-sessions", type=int, default=DEFAULT_VALIDATION_SESSIONS)
    parser.add_argument("--mongo-uri", default=None)
    parser.add_argument("--database", default=None)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--fresh-run",
        action="store_true",
        help=(
            "Delete only this experiment's output directory and recompute everything from "
            "the local MongoDB market cache. No prior research artifacts are read."
        ),
    )
    return parser


def _assert_safe_fresh_root(root: Path) -> None:
    research_root = (PROJECT_ROOT / "research_output").resolve()
    resolved = root.resolve()
    try:
        relative = resolved.relative_to(research_root)
    except ValueError as exc:
        raise RuntimeError(
            "--fresh-run only deletes this experiment under PROJECT_ROOT/research_output: "
            f"{resolved}"
        ) from exc
    if not relative.parts or not relative.parts[0].startswith(
        "pooled_candidate_marginal_advantage_strategy_"
    ):
        raise RuntimeError(f"Refusing to delete unexpected research path: {resolved}")


def _progress(label: str):
    def callback(position: int, total: int, device: str) -> None:
        if position == 1 or position % 10 == 0 or position == total:
            _log(f"{label}: LightGBM {position}/{total} models ready ({device.upper()}).")
    return callback


def _calibrate_baseline_margin(
    frames: dict[str, pd.DataFrame],
    baseline_symbols: list[str],
    train_dates: pd.DatetimeIndex,
    calibration_dates: pd.DatetimeIndex,
    config: Any,
    *,
    phase_prefix: str,
) -> tuple[dict[str, Any], float, float]:
    models = _lightgbm_fit_models(
        {symbol: frames[symbol] for symbol in baseline_symbols},
        baseline_symbols,
        train_dates,
        config,
        phase=f"{phase_prefix}_baseline_calibration",
        progress_callback=_progress(f"{phase_prefix} baseline calibration"),
    )
    missing = sorted(set(baseline_symbols) - set(models))
    if missing:
        raise RuntimeError(
            f"{phase_prefix}: immutable baseline assets failed calibration model fit: "
            + ", ".join(missing)
        )

    margins = tuple(float(x) for x in config.rotation_switch_margin_candidates)
    if not margins:
        raise RuntimeError("rotation_switch_margin_candidates is empty.")
    best_margin = margins[0]
    best_score = float("-inf")
    margin_config = (
        config.model_copy(update={"strategy_mode": "COMPOUND_ROTATION_SWING_XGBOOST"})
        if absolute_utility_cash_gate_enabled(config)
        else config
    )
    baseline_frames = {symbol: frames[symbol] for symbol in baseline_symbols}
    for candidate in margins:
        policy = _utility_policy(
            models,
            baseline_frames,
            baseline_symbols,
            margin_config,
            candidate,
        )
        score = _simple_policy_growth(
            policy,
            baseline_frames,
            baseline_symbols,
            calibration_dates,
            config,
        )
        if float(score) > best_score:
            best_score = float(score)
            best_margin = float(candidate)
    return models, float(best_margin), float(best_score)


def _fit_final_models(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    dates: pd.DatetimeIndex,
    config: Any,
    *,
    phase: str,
    require_all: bool,
) -> dict[str, Any]:
    symbols = [symbol for symbol in symbols if symbol in frames]
    if not symbols:
        return {}
    models = _lightgbm_fit_models(
        {symbol: frames[symbol] for symbol in symbols},
        symbols,
        dates,
        config,
        phase=phase,
        progress_callback=_progress(phase),
    )
    if require_all:
        missing = sorted(set(symbols) - set(models))
        if missing:
            raise RuntimeError(
                f"{phase}: immutable baseline assets failed final model fit: "
                + ", ".join(missing)
            )
    return models


def _score_reference(
    model: Any,
    frame: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> tuple[float, float, np.ndarray]:
    available = pd.DatetimeIndex(dates).intersection(pd.DatetimeIndex(frame.index))
    if available.empty:
        return 0.0, 1.0, np.asarray([], dtype=float)
    features = frame.loc[available, ROTATION_FEATURES].copy()
    features = features.loc[~features.isna().any(axis=1)]
    if features.empty:
        return 0.0, 1.0, np.asarray([], dtype=float)
    values = np.asarray(model.predict(features), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0, values
    mean = float(np.mean(values))
    std = float(np.std(values))
    if not np.isfinite(std) or std <= 1e-12:
        std = 1.0
    return mean, std, np.sort(values)


def _score_percentile(sorted_reference: np.ndarray, value: float) -> float:
    if sorted_reference.size == 0 or not np.isfinite(value):
        return 0.5
    return float(np.searchsorted(sorted_reference, value, side="right") / len(sorted_reference))


def _candidate_score(model: Any, frame: pd.DataFrame, timestamp: pd.Timestamp) -> float | None:
    if model is None or timestamp not in frame.index:
        return None
    row = frame.loc[[timestamp], ROTATION_FEATURES]
    if row.empty or row.isna().any(axis=None):
        return None
    value = float(model.predict(row)[0])
    return value if np.isfinite(value) else None


def _state_update(current_position: int, holding_days: int, target_position: int) -> tuple[int, int]:
    if int(target_position) == int(current_position):
        return int(current_position), int(holding_days + 1) if int(current_position) > 0 else 0
    return int(target_position), 1 if int(target_position) > 0 else 0


def _build_samples(
    *,
    fold_id: int,
    frames: dict[str, pd.DataFrame],
    baseline_symbols: list[str],
    candidate_symbols: list[str],
    baseline_models: dict[str, Any],
    candidate_models: dict[str, Any],
    candidate_references: dict[str, tuple[float, float, np.ndarray]],
    decision_dates: pd.DatetimeIndex,
    config: Any,
    effective_margin: float,
    calibrated_margin: float,
    phase_label: str,
    include_future_targets: bool = True,
) -> pd.DataFrame:
    baseline_frames = {symbol: frames[symbol] for symbol in baseline_symbols}
    policy = _utility_policy(
        baseline_models,
        baseline_frames,
        baseline_symbols,
        config,
        effective_margin,
        fold_id=fold_id,
        calibrated_switch_margin=calibrated_margin,
    )
    combined_symbols = [*baseline_symbols, *candidate_symbols]
    combined_position = {symbol: idx + 1 for idx, symbol in enumerate(combined_symbols)}
    current_position = 0
    holding_days = 0
    rows: list[dict[str, Any]] = []

    ordered_dates = pd.DatetimeIndex(decision_dates).sort_values()
    for timestamp, next_timestamp in zip(ordered_dates[:-1], ordered_dates[1:]):
        timestamp = pd.Timestamp(timestamp)
        next_timestamp = pd.Timestamp(next_timestamp)
        baseline_utilities = _model_utilities(
            baseline_models,
            baseline_frames,
            baseline_symbols,
            timestamp,
            config,
        )
        finite_scores = baseline_utilities[1:][np.isfinite(baseline_utilities[1:])]
        if finite_scores.size == 0:
            continue

        target_position, target_score = policy(timestamp, current_position, holding_days)
        current_score = (
            float(baseline_utilities[current_position])
            if current_position > 0 and np.isfinite(baseline_utilities[current_position])
            else 0.0
        )
        baseline_mean = float(np.mean(finite_scores))
        baseline_std = float(np.std(finite_scores))
        stable_std = baseline_std if baseline_std > 1e-12 else 1.0
        positive_fraction = float(np.mean(finite_scores > 0.0))
        target_score_float = float(target_score) if np.isfinite(target_score) else 0.0
        target_z = float((target_score_float - baseline_mean) / stable_std)
        target_symbol = baseline_symbols[int(target_position) - 1] if int(target_position) > 0 else None
        target_features = (
            frames[target_symbol].loc[timestamp, ROTATION_FEATURES]
            if target_symbol is not None
            else pd.Series(0.0, index=ROTATION_FEATURES, dtype=float)
        )
        if target_symbol is not None and target_features.isna().any():
            current_position, holding_days = _state_update(current_position, holding_days, int(target_position))
            continue

        from_combined = (
            combined_position[baseline_symbols[current_position - 1]]
            if current_position > 0
            else 0
        )
        base_to_combined = combined_position[target_symbol] if target_symbol else 0
        if include_future_targets:
            base_return = _training_transition_log_return(
                frames, combined_symbols, timestamp, next_timestamp,
                from_combined, base_to_combined, config,
            )
            if not np.isfinite(base_return):
                current_position, holding_days = _state_update(current_position, holding_days, int(target_position))
                continue

        for candidate in candidate_symbols:
            model = candidate_models.get(candidate)
            if model is None:
                continue
            candidate_frame = frames[candidate]
            score = _candidate_score(model, candidate_frame, timestamp)
            if score is None:
                continue
            candidate_features = candidate_frame.loc[timestamp, ROTATION_FEATURES]
            if candidate_features.isna().any():
                continue
            if include_future_targets:
                candidate_return = _training_transition_log_return(
                    frames, combined_symbols, timestamp, next_timestamp,
                    from_combined, combined_position[candidate], config,
                )
                if not np.isfinite(candidate_return):
                    continue

            mean, std, reference = candidate_references[candidate]
            row: dict[str, Any] = {
                "phase": phase_label,
                "fold": int(fold_id),
                "timestamp": timestamp,
                "next_timestamp": next_timestamp,
                "candidate": candidate,
                "baseline_current_asset": baseline_symbols[current_position - 1] if current_position > 0 else "CASH",
                "baseline_target_asset": target_symbol or "CASH",
                "candidate_score": float(score),
                "candidate_score_z_train": float((score - mean) / std),
                "candidate_score_percentile_train": _score_percentile(reference, score),
                "baseline_target_score": target_score_float,
                "baseline_target_score_z_cross_section": target_z,
                "baseline_current_score": current_score,
                "baseline_target_vs_current_gap": float(target_score_float - current_score),
                "baseline_score_mean": baseline_mean,
                "baseline_score_std": baseline_std,
                "baseline_positive_score_fraction": positive_fraction,
                "baseline_holding_days": int(holding_days),
                "baseline_target_is_cash": int(target_symbol is None),
            }
            if include_future_targets:
                row.update({
                    "baseline_next_session_net_log_return": float(base_return),
                    "candidate_next_session_net_log_return": float(candidate_return),
                    TARGET_COLUMN: float(candidate_return - base_return),
                })
            delta = candidate_features.astype(float) - target_features.astype(float)
            for feature in ROTATION_FEATURES:
                row[f"delta_{feature}"] = float(delta[feature])
            rows.append(row)

        current_position, holding_days = _state_update(current_position, holding_days, int(target_position))

    frame = pd.DataFrame(rows)
    _log(
        f"{phase_label}: generated {len(frame):,} candidate-date samples from "
        f"{frame['timestamp'].nunique() if not frame.empty else 0} baseline decisions and "
        f"{frame['candidate'].nunique() if not frame.empty else 0} trainable candidates."
    )
    return frame


def _fold_dates(common_dates: pd.DatetimeIndex, fold: dict[str, Any]) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]:
    train_dates = common_dates[: int(fold["train_end_index"])]
    calibration_dates = common_dates[
        int(fold["calibration_start_index"]): int(fold["calibration_end_index"])
    ]
    final_fit_dates = common_dates[: int(fold["final_fit_end_index"])]
    decision_dates = pd.DatetimeIndex(fold["decision_dates"]).sort_values()
    return train_dates, calibration_dates, final_fit_dates, decision_dates


def _candidate_summary(decisions: pd.DataFrame, phase: str) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame()
    chosen = decisions.loc[decisions["override_baseline"].astype(bool)].copy()
    rows: list[dict[str, Any]] = []
    for candidate, group in chosen.groupby("chosen_candidate"):
        realized = pd.to_numeric(group["realized_marginal_next_session_net_log_return"], errors="coerce").dropna()
        predicted = pd.to_numeric(group["predicted_marginal_advantage"], errors="coerce").dropna()
        rows.append(
            {
                "phase": phase,
                "candidate": candidate,
                "chosen_count": int(len(group)),
                "predicted_marginal_advantage_mean": float(predicted.mean()) if not predicted.empty else None,
                "realized_marginal_log_sum": float(realized.sum()),
                "realized_marginal_log_mean": float(realized.mean()) if not realized.empty else None,
                "positive_realized_rate": float((realized > 0.0).mean()) if not realized.empty else None,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = _parser().parse_args()
    common.load_project_environment(args.env_file)
    os.environ.setdefault("MCT_MODEL_THREADS_OVERRIDE", "1")

    history_start = common._normalize_date(args.history_start)
    selection_end_text, validation_start_text, validation_end_text = independent._split(
        history_start.date().isoformat(), args.snapshot_end, int(args.validation_sessions)
    )
    selection_end = common._normalize_date(selection_end_text)
    validation_start = common._normalize_date(validation_start_text)
    validation_end = common._normalize_date(validation_end_text)

    output_dir = Path(
        args.output_dir
        or PROJECT_ROOT / "research_output" / (
            f"pooled_candidate_marginal_advantage_strategy_{args.strategy_sequence}_"
            f"{validation_start.date().isoformat()}_to_{validation_end.date().isoformat()}"
        )
    ).resolve()
    if args.fresh_run:
        _assert_safe_fresh_root(output_dir)
        if file_io.exists(output_dir):
            _log(f"FRESH RUN: deleting prior PCMA artifacts only from {output_dir}")
            file_io.remove_tree(output_dir)
    elif file_io.exists(output_dir):
        raise RuntimeError(
            f"Output directory already exists: {output_dir}. This research does not resume. "
            "Use --fresh-run to delete only this experiment and recompute from MongoDB."
        )
    file_io.ensure_dir(output_dir)

    mongo_uri = str(args.mongo_uri or os.getenv("MONGO_URL") or os.getenv("MONGO_URI") or "mongodb://localhost:27017").strip()
    database_name = str(args.database or os.getenv("MONGO_DATABASE") or "").strip()
    if not database_name:
        raise RuntimeError("MONGO_DATABASE is required in .env or via --database.")
    common._assert_local_mongo(mongo_uri, False)

    client = MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=3_000,
        connectTimeoutMS=3_000,
        maxPoolSize=8,
        retryWrites=False,
    )
    client.admin.command("ping")
    db = client[database_name]
    strategy = common._strategy_document(db, args.strategy_sequence, args.strategy_id)
    configuration = common._configuration(strategy)
    base_config = leadership.BacktestRequest.model_validate(configuration)
    baseline_assets = list(dict.fromkeys(str(x).strip().upper() for x in base_config.assets if str(x).strip()))
    if common._normalize_date(base_config.start_date) != history_start:
        raise RuntimeError(
            "Strategy start_date does not match --history-start: "
            f"strategy={common._normalize_date(base_config.start_date).date()}, requested={history_start.date()}."
        )

    identity = common._market_identity(configuration)
    collection = db[common.ALPACA_MARKET_BARS_COLLECTION]
    baseline_set = set(baseline_assets)
    cached = {str(item).strip().upper() for item in collection.distinct("symbol", identity) if str(item).strip()}
    external = sorted(cached - baseline_set)
    _log(
        f"Candidate discovery from local MongoDB only: baseline={len(baseline_assets)}, "
        f"external_cached={len(external)}, selection_end={selection_end.date()}."
    )

    selection_universe = [*baseline_assets, *external]
    raw_selection_all = timing._load_frames_allow_incomplete(
        collection, selection_universe, identity, history_start, selection_end
    )
    expected_selection = common._expected_sessions(history_start, selection_end)
    history_rows, complete_assets = timing._history_diagnostics(
        raw_selection_all, selection_universe, baseline_set, expected_selection
    )
    file_io.write_csv(output_dir / "pcma_selection_history_integrity.csv", pd.DataFrame(history_rows))
    complete_set = set(complete_assets)
    missing_baseline = [x for x in baseline_assets if x not in complete_set]
    if missing_baseline:
        raise RuntimeError(
            "Immutable Strategy baseline failed Full Strategy History: " + ", ".join(missing_baseline)
        )
    candidate_assets = [x for x in external if x in complete_set]
    research_assets = [*baseline_assets, *candidate_assets]
    raw_selection = {x: raw_selection_all[x] for x in research_assets}
    _log(f"Frozen pre-validation candidate pool: {len(candidate_assets)} complete external assets.")

    raw_holdout = timing._load_frames_allow_incomplete(
        collection, research_assets, identity, history_start, validation_end
    )
    family, settings, settings_hash = _immutable_model_snapshot(strategy)
    client.close()
    _log(
        "MongoDB closed. Candidate pool is frozen; all remaining work uses immutable in-memory data. "
        f"model={family}, settings_hash={settings_hash}."
    )

    selection_config = leadership._execution_config(
        configuration, research_assets, baseline_assets, selection_end, family, settings
    )
    selection_frames, selection_dates = prepare_rotation_panel(raw_selection, selection_config)
    folds = _build_walk_forward_folds(selection_dates, selection_config)
    if len(folds) < 3:
        raise RuntimeError(f"PCMA requires at least three pre-validation walk-forward folds; got {len(folds)}.")

    all_fold_samples: list[pd.DataFrame] = []
    fold_model_rows: list[dict[str, Any]] = []
    baseline_symbols = sorted(baseline_assets)
    selection_candidates = sorted([x for x in candidate_assets if x in selection_frames])
    for fold in folds:
        fold_id = int(fold["fold_id"])
        train_dates, calibration_dates, final_fit_dates, decision_dates = _fold_dates(selection_dates, fold)
        _log(
            f"PREVALIDATION FOLD {fold_id}: train={len(train_dates)}, calibration={len(calibration_dates)}, "
            f"final_fit={len(final_fit_dates)}, decisions={max(0, len(decision_dates)-1)}."
        )
        _, calibrated_margin, calibration_score = _calibrate_baseline_margin(
            selection_frames,
            baseline_symbols,
            train_dates,
            calibration_dates,
            selection_config,
            phase_prefix=f"pcma_fold_{fold_id}",
        )
        baseline_models = _fit_final_models(
            selection_frames,
            baseline_symbols,
            final_fit_dates,
            selection_config,
            phase=f"pcma_fold_{fold_id}_baseline_final",
            require_all=True,
        )
        candidate_models = _fit_final_models(
            selection_frames,
            selection_candidates,
            final_fit_dates,
            selection_config,
            phase=f"pcma_fold_{fold_id}_candidate_final",
            require_all=False,
        )
        trainable_candidates = sorted(candidate_models)
        references = {
            candidate: _score_reference(candidate_models[candidate], selection_frames[candidate], final_fit_dates)
            for candidate in trainable_candidates
        }
        effective_margin = max(float(selection_config.rotation_switch_margin), float(calibrated_margin))
        samples = _build_samples(
            fold_id=fold_id,
            frames=selection_frames,
            baseline_symbols=baseline_symbols,
            candidate_symbols=trainable_candidates,
            baseline_models=baseline_models,
            candidate_models=candidate_models,
            candidate_references=references,
            decision_dates=decision_dates,
            config=selection_config,
            effective_margin=effective_margin,
            calibrated_margin=calibrated_margin,
            phase_label=f"prevalidation_fold_{fold_id}",
        )
        all_fold_samples.append(samples)
        fold_model_rows.append(
            {
                "fold": fold_id,
                "calibrated_switch_margin": calibrated_margin,
                "effective_switch_margin": effective_margin,
                "baseline_calibration_score": calibration_score,
                "candidate_pool_count": len(candidate_assets),
                "trainable_candidate_count": len(trainable_candidates),
                "sample_count": len(samples),
                "decision_date_count": int(samples["timestamp"].nunique()) if not samples.empty else 0,
            }
        )

    prevalidation_samples = pd.concat(all_fold_samples, ignore_index=True)
    file_io.write_csv(output_dir / "pcma_prevalidation_samples.csv", prevalidation_samples)
    file_io.write_csv(output_dir / "pcma_fold_model_diagnostics.csv", pd.DataFrame(fold_model_rows))

    crossfit_scored: list[pd.DataFrame] = []
    crossfit_decisions: list[pd.DataFrame] = []
    crossfit_stages: list[dict[str, Any]] = []
    fold_ids = sorted(int(x) for x in prevalidation_samples["fold"].unique())
    for test_fold in fold_ids[1:]:
        train = prevalidation_samples.loc[prevalidation_samples["fold"] < test_fold].copy()
        test = prevalidation_samples.loc[prevalidation_samples["fold"] == test_fold].copy()
        if train.empty or test.empty:
            continue
        model = fit_pooled_candidate_marginal_model(train, MODEL_FEATURES, target_column=TARGET_COLUMN)
        scored = score_candidate_samples(model, test, target_column=TARGET_COLUMN)
        decisions = choose_daily_candidate_overrides(scored, target_column=TARGET_COLUMN)
        summary = summarize_override_decisions(decisions)
        crossfit_stages.append(
            {
                "test_fold": int(test_fold),
                "training_folds": sorted(int(x) for x in train["fold"].unique()),
                **model.diagnostics(),
                **summary,
            }
        )
        scored["evaluation_stage"] = f"test_fold_{test_fold}"
        decisions["evaluation_stage"] = f"test_fold_{test_fold}"
        crossfit_scored.append(scored)
        crossfit_decisions.append(decisions)
        _log(
            f"CROSSFIT fold {test_fold}: overrides={summary['override_count']}/{summary['decision_dates']}, "
            f"realized_marginal_log_sum={summary['realized_marginal_log_sum']:+.6f}."
        )

    crossfit_scored_frame = pd.concat(crossfit_scored, ignore_index=True) if crossfit_scored else pd.DataFrame()
    crossfit_decisions_frame = pd.concat(crossfit_decisions, ignore_index=True) if crossfit_decisions else pd.DataFrame()
    file_io.write_csv(output_dir / "pcma_crossfit_scored_samples.csv", crossfit_scored_frame)
    file_io.write_csv(output_dir / "pcma_crossfit_decisions.csv", crossfit_decisions_frame)
    file_io.write_json(output_dir / "pcma_crossfit_summary.json", crossfit_stages)

    holdout_config = leadership._execution_config(
        configuration, research_assets, baseline_assets, validation_end, family, settings
    ).model_copy(
        update={
            "analysis_start_date": validation_start.date().isoformat(),
            "analysis_end_date": validation_end.date().isoformat(),
        }
    )
    holdout_frames, holdout_dates = prepare_rotation_panel(raw_holdout, holdout_config)
    holdout_train_dates, holdout_calibration_dates, holdout_final_fit_dates, holdout_purge, validation_index = validation._strict_training_windows(
        holdout_dates, validation_start, holdout_config
    )
    holdout_decision_dates = validation._decision_dates(holdout_dates, validation_index, validation_end)
    _, holdout_calibrated_margin, holdout_calibration_score = _calibrate_baseline_margin(
        holdout_frames,
        baseline_symbols,
        holdout_train_dates,
        holdout_calibration_dates,
        holdout_config,
        phase_prefix="pcma_holdout",
    )
    holdout_baseline_models = _fit_final_models(
        holdout_frames,
        baseline_symbols,
        holdout_final_fit_dates,
        holdout_config,
        phase="pcma_holdout_baseline_final",
        require_all=True,
    )
    holdout_candidate_models = _fit_final_models(
        holdout_frames,
        sorted([x for x in candidate_assets if x in holdout_frames]),
        holdout_final_fit_dates,
        holdout_config,
        phase="pcma_holdout_candidate_final",
        require_all=False,
    )
    holdout_candidates = sorted(holdout_candidate_models)
    holdout_references = {
        candidate: _score_reference(holdout_candidate_models[candidate], holdout_frames[candidate], holdout_final_fit_dates)
        for candidate in holdout_candidates
    }
    holdout_effective_margin = max(float(holdout_config.rotation_switch_margin), float(holdout_calibrated_margin))
    holdout_samples = _build_samples(
        fold_id=max(fold_ids) + 1,
        frames=holdout_frames,
        baseline_symbols=baseline_symbols,
        candidate_symbols=holdout_candidates,
        baseline_models=holdout_baseline_models,
        candidate_models=holdout_candidate_models,
        candidate_references=holdout_references,
        decision_dates=holdout_decision_dates,
        config=holdout_config,
        effective_margin=holdout_effective_margin,
        calibrated_margin=holdout_calibrated_margin,
        phase_label="untouched_holdout",
    )
    file_io.write_csv(output_dir / "pcma_holdout_samples.csv", holdout_samples)

    final_model = fit_pooled_candidate_marginal_model(prevalidation_samples, MODEL_FEATURES, target_column=TARGET_COLUMN)
    holdout_scored = score_candidate_samples(final_model, holdout_samples, target_column=TARGET_COLUMN)
    holdout_decisions = choose_daily_candidate_overrides(holdout_scored, target_column=TARGET_COLUMN)
    holdout_summary = summarize_override_decisions(holdout_decisions)
    file_io.write_csv(output_dir / "pcma_holdout_scored_samples.csv", holdout_scored)
    file_io.write_csv(output_dir / "pcma_holdout_decisions.csv", holdout_decisions)

    candidate_summary = pd.concat(
        [
            _candidate_summary(crossfit_decisions_frame, "prevalidation_crossfit"),
            _candidate_summary(holdout_decisions, "untouched_holdout"),
        ],
        ignore_index=True,
    )
    file_io.write_csv(output_dir / "pcma_candidate_summary.csv", candidate_summary)

    crossfit_total = float(sum(float(stage["realized_marginal_log_sum"]) for stage in crossfit_stages))
    result = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "experiment": EXPERIMENT,
        "strategy_sequence": int(args.strategy_sequence),
        "strategy_id": str(strategy.get("_id") or ""),
        "strategy_revision": int(strategy.get("revision") or 0),
        "strategy_configuration_hash": strategy.get("configuration_hash"),
        "model_family": family,
        "model_settings_hash": settings_hash,
        "baseline_asset_count": len(baseline_assets),
        "candidate_pool_count": len(candidate_assets),
        "candidate_pool_source": "all local MongoDB symbols with Full Strategy History through selection_end",
        "candidate_identity_feature_used": False,
        "baseline_policy_modified": False,
        "full_candidate_overlay_backtest_run": False,
        "target": (
            "exact next-session net-log-return advantage of choosing candidate instead of "
            "the immutable baseline action from the same baseline current state"
        ),
        "decision_threshold": 0.0,
        "decision_threshold_basis": "economic indifference; no manual confidence or score margin",
        "pooled_estimator": "StandardScaler + BayesianRidge",
        "equal_total_weight_per_decision_date": True,
        "feature_count": len(MODEL_FEATURES),
        "features": MODEL_FEATURES,
        "prevalidation_fold_count": len(folds),
        "prevalidation_crossfit_stages": crossfit_stages,
        "prevalidation_crossfit_realized_marginal_log_sum": crossfit_total,
        "prevalidation_crossfit_incremental_factor": float(math.exp(crossfit_total) - 1.0),
        "validation_start": validation_start.date().isoformat(),
        "validation_end": validation_end.date().isoformat(),
        "validation_sessions": int(len(holdout_decision_dates) - 1),
        "holdout_purge_sessions": int(holdout_purge),
        "holdout_calibrated_switch_margin": float(holdout_calibrated_margin),
        "holdout_effective_switch_margin": float(holdout_effective_margin),
        "holdout_baseline_calibration_score": float(holdout_calibration_score),
        "holdout_trainable_candidate_count": len(holdout_candidates),
        "holdout_model_diagnostics": final_model.diagnostics(),
        "holdout_summary": holdout_summary,
        "holdout_incremental_factor": float(math.exp(float(holdout_summary["realized_marginal_log_sum"])) - 1.0),
        "untouched_holdout_signal_positive": bool(float(holdout_summary["realized_marginal_log_sum"]) > 0.0),
        "selection_used_holdout_period": False,
        "training_used_holdout_period": False,
        "research_next_step_if_positive": (
            "Integrate this pooled candidate-advantage model as a research-only overlay and "
            "run an exact capital backtest versus the immutable 56-asset baseline."
        ),
    }
    file_io.write_json(output_dir / "pcma_result.json", result)
    file_io.write_json(
        output_dir / "experiment_manifest.json",
        {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "experiment": EXPERIMENT,
            "fresh_run": bool(args.fresh_run),
            "prior_research_artifacts_read": False,
            "mongo_writes": False,
            "alpaca_network_used": False,
            "candidate_pool_frozen_before_holdout_load": True,
            "baseline_assets": baseline_assets,
            "candidate_assets": candidate_assets,
            "rotation_features": list(ROTATION_FEATURES),
            "pooled_features": MODEL_FEATURES,
            "target_column": TARGET_COLUMN,
            "model": "BayesianRidge",
        },
    )

    _log("=== POOLED CANDIDATE MARGINAL ADVANTAGE V1 RESULT ===")
    _log(
        f"Prevalidation cross-fit marginal log sum: {crossfit_total:+.6f} "
        f"(factor={math.exp(crossfit_total)-1.0:+.2%})."
    )
    _log(
        f"Untouched holdout: overrides={holdout_summary['override_count']}/{holdout_summary['decision_dates']}, "
        f"marginal_log_sum={float(holdout_summary['realized_marginal_log_sum']):+.6f}, "
        f"incremental_factor={math.exp(float(holdout_summary['realized_marginal_log_sum']))-1.0:+.2%}."
    )
    _log(
        "This v1 tests whether an asset-agnostic relative-state signature generalizes. "
        "It intentionally does not claim portfolio-capital improvement until the next exact overlay backtest."
    )
    _log(f"Result directory: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
