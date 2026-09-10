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
import research_pooled_candidate_marginal_advantage as pcma  # noqa: E402
import research_pooled_candidate_episode_advantage as pcea  # noqa: E402
import research_windows_file_io as file_io  # noqa: E402
from research_asset_timing_vs_buyhold_execution import _immutable_model_snapshot  # noqa: E402
from market_cycle_trader_api.engine.capital_rotation import (  # noqa: E402
    _build_walk_forward_folds,
    prepare_rotation_panel,
)
from market_cycle_trader_api.services.asset_marginal_score_replay import (  # noqa: E402
    ScoreReplayPanel,
    replay_policy_settings,
)
from market_cycle_trader_api.services.dense_counterfactual_candidate_advantage import (  # noqa: E402
    DCCA_EPISODE_DEFINITION_VERSION,
    build_dense_counterfactual_episode_samples,
    oracle_non_overlapping_summary,
)
from market_cycle_trader_api.services.pooled_candidate_episode_advantage import (  # noqa: E402
    TARGET_COLUMN,
    choose_non_overlapping_episode_overrides,
    fit_episode_model,
    score_episode_samples,
    summarize_episode_decisions,
)

SCRIPT_VERSION = "dense-counterfactual-candidate-advantage-v1.0.0"
EXPERIMENT = "dense_forced_candidate_stateful_episode_advantage"
DEFAULT_VALIDATION_SESSIONS = 252
MODEL_FEATURES = list(pcma.MODEL_FEATURES)


def _log(message: str) -> None:
    stamp = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{stamp}] {message}", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Research whether dense, ticker-agnostic forced candidate interventions can "
            "learn when an external asset adds stateful economic value over the immutable "
            "56-asset Utility policy. Every run is rebuilt from local MongoDB data."
        )
    )
    parser.add_argument("--strategy-sequence", type=int, default=10)
    parser.add_argument("--strategy-id", default=None)
    parser.add_argument("--history-start", required=True)
    parser.add_argument("--snapshot-end", required=True)
    parser.add_argument(
        "--validation-sessions", type=int, default=DEFAULT_VALIDATION_SESSIONS
    )
    parser.add_argument("--mongo-uri", default=None)
    parser.add_argument("--database", default=None)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--fresh-run",
        action="store_true",
        help=(
            "Delete only this DCCA experiment output and recompute from the local MongoDB "
            "market cache. No prior PCMA/PCEA/marginal artifacts are read."
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
        "dense_counterfactual_candidate_advantage_strategy_"
    ):
        raise RuntimeError(f"Refusing to delete unexpected research path: {resolved}")


def _signal_diagnostics(scored: pd.DataFrame) -> dict[str, Any]:
    if scored.empty:
        return {
            "scored_rows": 0,
            "prediction_realized_pearson": None,
            "prediction_realized_spearman": None,
            "realized_positive_rate": None,
            "prediction_mean": None,
            "realized_mean": None,
        }
    predicted = pd.to_numeric(
        scored["predicted_marginal_advantage"], errors="coerce"
    )
    realized = pd.to_numeric(scored[TARGET_COLUMN], errors="coerce")
    mask = predicted.notna() & realized.notna()
    predicted = predicted.loc[mask]
    realized = realized.loc[mask]
    return {
        "scored_rows": int(len(predicted)),
        "prediction_realized_pearson": (
            float(predicted.corr(realized, method="pearson"))
            if len(predicted) > 1
            else None
        ),
        "prediction_realized_spearman": (
            float(predicted.corr(realized, method="spearman"))
            if len(predicted) > 1
            else None
        ),
        "realized_positive_rate": (
            float((realized > 0.0).mean()) if not realized.empty else None
        ),
        "prediction_mean": float(predicted.mean()) if not predicted.empty else None,
        "realized_mean": float(realized.mean()) if not realized.empty else None,
    }


def _candidate_summary(decisions: pd.DataFrame, phase: str) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame()
    chosen = decisions.loc[decisions["override_baseline"].astype(bool)].copy()
    rows: list[dict[str, Any]] = []
    for candidate, group in chosen.groupby("chosen_candidate"):
        realized = pd.to_numeric(
            group["realized_marginal_episode_net_log_return"], errors="coerce"
        ).dropna()
        predicted = pd.to_numeric(
            group["predicted_marginal_advantage"], errors="coerce"
        ).dropna()
        rows.append(
            {
                "phase": phase,
                "candidate": candidate,
                "chosen_episode_count": int(len(group)),
                "predicted_episode_advantage_mean": (
                    float(predicted.mean()) if not predicted.empty else None
                ),
                "realized_episode_log_sum": (
                    float(realized.sum()) if not realized.empty else None
                ),
                "realized_episode_log_mean": (
                    float(realized.mean()) if not realized.empty else None
                ),
                "positive_realized_episode_rate": (
                    float((realized > 0.0).mean()) if not realized.empty else None
                ),
            }
        )
    return pd.DataFrame(rows)


def _period_samples(
    *,
    fold_id: int,
    frames: dict[str, pd.DataFrame],
    baseline_symbols: list[str],
    candidate_symbols: list[str],
    baseline_models: dict[str, Any],
    candidate_models: dict[str, Any],
    candidate_references: dict[str, tuple[float, float, np.ndarray]],
    decision_dates: pd.DatetimeIndex,
    replay_fold: dict[str, Any],
    config: Any,
    effective_margin: float,
    calibrated_margin: float,
    maximum_label_horizon: int,
    phase_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    daily_features = pcma._build_samples(
        fold_id=fold_id,
        frames=frames,
        baseline_symbols=baseline_symbols,
        candidate_symbols=candidate_symbols,
        baseline_models=baseline_models,
        candidate_models=candidate_models,
        candidate_references=candidate_references,
        decision_dates=decision_dates,
        config=config,
        effective_margin=effective_margin,
        calibrated_margin=calibrated_margin,
        phase_label=phase_label,
    )

    all_models = {**baseline_models, **candidate_models}
    tape = pcea._replay_tape(
        frames=frames,
        models=all_models,
        symbols=[*baseline_symbols, *candidate_symbols],
        fold=replay_fold,
        maximum_label_horizon=maximum_label_horizon,
    )
    if tape.empty:
        raise RuntimeError(f"{phase_label}: no cached-score replay tape was produced.")

    panel_end = pd.to_datetime(
        tape["execution_timestamp"], utc=True, format="mixed"
    ).max()
    panel = ScoreReplayPanel(
        tape,
        baseline_symbols,
        pd.Timestamp(panel_end).date().isoformat(),
    )
    complete_candidates = [
        candidate for candidate in candidate_symbols if panel.complete(candidate)
    ]
    if not complete_candidates:
        raise RuntimeError(
            f"{phase_label}: no candidate has complete cached-score replay coverage."
        )

    settings = replay_policy_settings(config)
    settings["rotation_switch_margin"] = float(effective_margin)
    baseline_replay = panel.replay(baseline_symbols, settings)

    def progress(position: int, total: int) -> None:
        _log(
            f"{phase_label}: dense counterfactual episodes "
            f"{position:,}/{total:,} candidate-date interventions."
        )

    samples, episodes, diagnostics = build_dense_counterfactual_episode_samples(
        panel=panel,
        baseline_replay=baseline_replay,
        daily_features=daily_features,
        candidate_symbols=complete_candidates,
        settings=settings,
        fold_id=fold_id,
        progress_callback=progress,
    )
    diagnostics.update(
        {
            "phase": phase_label,
            "baseline_replay_ending_capital": float(
                baseline_replay.ending_capital
            ),
            "baseline_replay_net_log_growth": float(
                baseline_replay.net_log_growth
            ),
            "effective_switch_margin": float(effective_margin),
            "calibrated_switch_margin": float(calibrated_margin),
            "candidate_models": int(len(candidate_symbols)),
            "replay_complete_candidates": int(len(complete_candidates)),
        }
    )
    _log(
        f"{phase_label}: dense samples={len(samples):,}, "
        f"forced episodes={len(episodes):,}, "
        f"right-censored={diagnostics['right_censored_episode_rows']:,}."
    )
    return samples, episodes, diagnostics


def main() -> int:
    args = _parser().parse_args()
    common.load_project_environment(args.env_file)
    os.environ.setdefault("MCT_MODEL_THREADS_OVERRIDE", "1")

    history_start = common._normalize_date(args.history_start)
    selection_end_text, validation_start_text, validation_end_text = independent._split(
        history_start.date().isoformat(),
        args.snapshot_end,
        int(args.validation_sessions),
    )
    selection_end = common._normalize_date(selection_end_text)
    validation_start = common._normalize_date(validation_start_text)
    validation_end = common._normalize_date(validation_end_text)

    output_dir = Path(
        args.output_dir
        or PROJECT_ROOT
        / "research_output"
        / (
            f"dense_counterfactual_candidate_advantage_strategy_{args.strategy_sequence}_"
            f"{validation_start.date().isoformat()}_to_{validation_end.date().isoformat()}"
        )
    ).resolve()
    if args.fresh_run:
        _assert_safe_fresh_root(output_dir)
        if file_io.exists(output_dir):
            _log(f"FRESH RUN: deleting prior DCCA artifacts only from {output_dir}")
            file_io.remove_tree(output_dir)
    elif file_io.exists(output_dir):
        raise RuntimeError(
            f"Output directory already exists: {output_dir}. "
            "Use --fresh-run to recompute from MongoDB."
        )
    file_io.ensure_dir(output_dir)

    mongo_uri = str(
        args.mongo_uri
        or os.getenv("MONGO_URL")
        or os.getenv("MONGO_URI")
        or "mongodb://localhost:27017"
    ).strip()
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
    strategy = common._strategy_document(
        db, args.strategy_sequence, args.strategy_id
    )
    configuration = common._configuration(strategy)
    base_config = leadership.BacktestRequest.model_validate(configuration)
    baseline_assets = list(
        dict.fromkeys(
            str(value).strip().upper()
            for value in base_config.assets
            if str(value).strip()
        )
    )
    if common._normalize_date(base_config.start_date) != history_start:
        raise RuntimeError(
            "Strategy start_date does not match --history-start: "
            f"strategy={common._normalize_date(base_config.start_date).date()}, "
            f"requested={history_start.date()}."
        )

    identity = common._market_identity(configuration)
    collection = db[common.ALPACA_MARKET_BARS_COLLECTION]
    baseline_set = set(baseline_assets)
    cached = {
        str(item).strip().upper()
        for item in collection.distinct("symbol", identity)
        if str(item).strip()
    }
    external = sorted(cached - baseline_set)
    _log(
        f"Candidate discovery from local MongoDB only: baseline={len(baseline_assets)}, "
        f"external_cached={len(external)}, selection_end={selection_end.date()}."
    )

    selection_universe = [*baseline_assets, *external]
    raw_selection_all = timing._load_frames_allow_incomplete(
        collection,
        selection_universe,
        identity,
        history_start,
        selection_end,
    )
    expected_selection = common._expected_sessions(history_start, selection_end)
    history_rows, complete_assets = timing._history_diagnostics(
        raw_selection_all,
        selection_universe,
        baseline_set,
        expected_selection,
    )
    file_io.write_csv(
        output_dir / "dcca_selection_history_integrity.csv",
        pd.DataFrame(history_rows),
    )
    complete_set = set(complete_assets)
    missing_baseline = [
        symbol for symbol in baseline_assets if symbol not in complete_set
    ]
    if missing_baseline:
        raise RuntimeError(
            "Immutable Strategy baseline failed Full Strategy History: "
            + ", ".join(missing_baseline)
        )

    candidate_assets = [
        symbol for symbol in external if symbol in complete_set
    ]
    research_assets = [*baseline_assets, *candidate_assets]
    raw_selection = {
        symbol: raw_selection_all[symbol] for symbol in research_assets
    }
    _log(
        f"Frozen candidate pool: {len(candidate_assets)} complete external assets."
    )

    raw_validation = timing._load_frames_allow_incomplete(
        collection,
        research_assets,
        identity,
        history_start,
        validation_end,
    )
    family, settings, settings_hash = _immutable_model_snapshot(strategy)
    client.close()
    _log(
        "MongoDB closed. Candidate pool is frozen; remaining work uses immutable "
        f"in-memory data. model={family}, settings_hash={settings_hash}."
    )

    selection_config = leadership._execution_config(
        configuration,
        research_assets,
        baseline_assets,
        selection_end,
        family,
        settings,
    )
    selection_frames, selection_dates = prepare_rotation_panel(
        raw_selection, selection_config
    )
    folds = _build_walk_forward_folds(selection_dates, selection_config)
    if len(folds) < 3:
        raise RuntimeError(
            f"DCCA v1 requires at least three pre-validation folds; got {len(folds)}."
        )

    maximum_label_horizon = max(
        int(value) for value in selection_config.rotation_target_horizons
    )
    baseline_symbols = sorted(baseline_assets)
    selection_candidates = sorted(
        [
            symbol
            for symbol in candidate_assets
            if symbol in selection_frames
        ]
    )

    all_fold_samples: list[pd.DataFrame] = []
    all_fold_episodes: list[pd.DataFrame] = []
    fold_diagnostics: list[dict[str, Any]] = []

    for fold in folds:
        fold_id = int(fold["fold_id"])
        (
            train_dates,
            calibration_dates,
            final_fit_dates,
            decision_dates,
        ) = pcma._fold_dates(selection_dates, fold)
        _log(
            f"PREVALIDATION FOLD {fold_id}: train={len(train_dates)}, "
            f"calibration={len(calibration_dates)}, final_fit={len(final_fit_dates)}, "
            f"decisions={max(0, len(decision_dates)-1)}."
        )
        _, calibrated_margin, calibration_score = pcma._calibrate_baseline_margin(
            selection_frames,
            baseline_symbols,
            train_dates,
            calibration_dates,
            selection_config,
            phase_prefix=f"dcca_fold_{fold_id}",
        )
        baseline_models = pcma._fit_final_models(
            selection_frames,
            baseline_symbols,
            final_fit_dates,
            selection_config,
            phase=f"dcca_fold_{fold_id}_baseline_final",
            require_all=True,
        )
        candidate_models = pcma._fit_final_models(
            selection_frames,
            selection_candidates,
            final_fit_dates,
            selection_config,
            phase=f"dcca_fold_{fold_id}_candidate_final",
            require_all=False,
        )
        trainable_candidates = sorted(candidate_models)
        references = {
            candidate: pcma._score_reference(
                candidate_models[candidate],
                selection_frames[candidate],
                final_fit_dates,
            )
            for candidate in trainable_candidates
        }
        effective_margin = max(
            float(selection_config.rotation_switch_margin),
            float(calibrated_margin),
        )

        samples, episodes, diagnostics = _period_samples(
            fold_id=fold_id,
            frames=selection_frames,
            baseline_symbols=baseline_symbols,
            candidate_symbols=trainable_candidates,
            baseline_models=baseline_models,
            candidate_models=candidate_models,
            candidate_references=references,
            decision_dates=decision_dates,
            replay_fold=fold,
            config=selection_config,
            effective_margin=effective_margin,
            calibrated_margin=calibrated_margin,
            maximum_label_horizon=maximum_label_horizon,
            phase_label=f"prevalidation_fold_{fold_id}",
        )
        diagnostics["baseline_calibration_score"] = float(calibration_score)
        all_fold_samples.append(samples)
        all_fold_episodes.append(episodes)
        fold_diagnostics.append(diagnostics)

    usable_samples = [frame for frame in all_fold_samples if not frame.empty]
    if not usable_samples:
        raise RuntimeError("No usable dense counterfactual pre-validation samples.")
    prevalidation_samples = pd.concat(usable_samples, ignore_index=True)
    prevalidation_episodes = pd.concat(
        [frame for frame in all_fold_episodes if not frame.empty],
        ignore_index=True,
    )

    file_io.write_csv(
        output_dir / "dcca_prevalidation_episode_samples.csv",
        prevalidation_samples,
    )
    file_io.write_csv(
        output_dir / "dcca_prevalidation_all_forced_episodes.csv",
        prevalidation_episodes,
    )
    file_io.write_csv(
        output_dir / "dcca_fold_diagnostics.csv",
        pd.DataFrame(fold_diagnostics),
    )

    crossfit_scored: list[pd.DataFrame] = []
    crossfit_decisions: list[pd.DataFrame] = []
    crossfit_stages: list[dict[str, Any]] = []
    fold_ids = sorted(
        int(value) for value in prevalidation_samples["fold"].unique()
    )

    for test_fold in fold_ids[1:]:
        train = prevalidation_samples.loc[
            prevalidation_samples["fold"] < test_fold
        ].copy()
        test = prevalidation_samples.loc[
            prevalidation_samples["fold"] == test_fold
        ].copy()
        if train.empty or test.empty:
            continue

        model = fit_episode_model(train, MODEL_FEATURES)
        scored = score_episode_samples(model, test)
        decisions = choose_non_overlapping_episode_overrides(scored)
        summary = summarize_episode_decisions(decisions)
        signal = _signal_diagnostics(scored)
        oracle = oracle_non_overlapping_summary(test)
        stage = {
            "test_fold": int(test_fold),
            "training_folds": sorted(
                int(value) for value in train["fold"].unique()
            ),
            **model.diagnostics(),
            **summary,
            **signal,
            **oracle,
        }
        crossfit_stages.append(stage)
        scored["evaluation_stage"] = f"test_fold_{test_fold}"
        decisions["evaluation_stage"] = f"test_fold_{test_fold}"
        crossfit_scored.append(scored)
        crossfit_decisions.append(decisions)
        _log(
            f"CROSSFIT fold {test_fold}: overrides="
            f"{summary['override_count']}/{summary['decision_episode_starts']}, "
            f"realized_marginal_log_sum="
            f"{summary['realized_marginal_log_sum']:+.6f}, "
            f"spearman={signal['prediction_realized_spearman']}."
        )

    crossfit_scored_frame = (
        pd.concat(crossfit_scored, ignore_index=True)
        if crossfit_scored
        else pd.DataFrame()
    )
    crossfit_decisions_frame = (
        pd.concat(crossfit_decisions, ignore_index=True)
        if crossfit_decisions
        else pd.DataFrame()
    )
    file_io.write_csv(
        output_dir / "dcca_crossfit_scored_episodes.csv",
        crossfit_scored_frame,
    )
    file_io.write_csv(
        output_dir / "dcca_crossfit_decisions.csv",
        crossfit_decisions_frame,
    )
    file_io.write_json(
        output_dir / "dcca_crossfit_summary.json",
        crossfit_stages,
    )

    validation_config = leadership._execution_config(
        configuration,
        research_assets,
        baseline_assets,
        validation_end,
        family,
        settings,
    ).model_copy(
        update={
            "analysis_start_date": validation_start.date().isoformat(),
            "analysis_end_date": validation_end.date().isoformat(),
        }
    )
    validation_frames, validation_dates = prepare_rotation_panel(
        raw_validation, validation_config
    )
    (
        validation_train_dates,
        validation_calibration_dates,
        validation_final_fit_dates,
        validation_purge,
        validation_index,
    ) = validation._strict_training_windows(
        validation_dates,
        validation_start,
        validation_config,
    )
    validation_decision_dates = validation._decision_dates(
        validation_dates,
        validation_index,
        validation_end,
    )

    (
        _,
        validation_calibrated_margin,
        validation_calibration_score,
    ) = pcma._calibrate_baseline_margin(
        validation_frames,
        baseline_symbols,
        validation_train_dates,
        validation_calibration_dates,
        validation_config,
        phase_prefix="dcca_validation",
    )
    validation_baseline_models = pcma._fit_final_models(
        validation_frames,
        baseline_symbols,
        validation_final_fit_dates,
        validation_config,
        phase="dcca_validation_baseline_final",
        require_all=True,
    )
    validation_candidate_models = pcma._fit_final_models(
        validation_frames,
        sorted(
            [
                symbol
                for symbol in candidate_assets
                if symbol in validation_frames
            ]
        ),
        validation_final_fit_dates,
        validation_config,
        phase="dcca_validation_candidate_final",
        require_all=False,
    )
    validation_candidates = sorted(validation_candidate_models)
    validation_references = {
        candidate: pcma._score_reference(
            validation_candidate_models[candidate],
            validation_frames[candidate],
            validation_final_fit_dates,
        )
        for candidate in validation_candidates
    }
    validation_effective_margin = max(
        float(validation_config.rotation_switch_margin),
        float(validation_calibrated_margin),
    )
    validation_fold_id = max(fold_ids) + 1
    validation_replay_fold = pcea._synthetic_holdout_fold(
        fold_id=validation_fold_id,
        decision_dates=validation_decision_dates,
        common_dates=validation_dates,
        final_fit_dates=validation_final_fit_dates,
    )
    (
        validation_samples,
        validation_episodes,
        validation_diagnostics,
    ) = _period_samples(
        fold_id=validation_fold_id,
        frames=validation_frames,
        baseline_symbols=baseline_symbols,
        candidate_symbols=validation_candidates,
        baseline_models=validation_baseline_models,
        candidate_models=validation_candidate_models,
        candidate_references=validation_references,
        decision_dates=validation_decision_dates,
        replay_fold=validation_replay_fold,
        config=validation_config,
        effective_margin=validation_effective_margin,
        calibrated_margin=validation_calibrated_margin,
        maximum_label_horizon=max(
            int(value)
            for value in validation_config.rotation_target_horizons
        ),
        phase_label="retrospective_validation",
    )
    if validation_samples.empty:
        raise RuntimeError(
            "Validation period produced no usable dense counterfactual samples."
        )

    file_io.write_csv(
        output_dir / "dcca_validation_episode_samples.csv",
        validation_samples,
    )
    file_io.write_csv(
        output_dir / "dcca_validation_all_forced_episodes.csv",
        validation_episodes,
    )

    final_model = fit_episode_model(
        prevalidation_samples,
        MODEL_FEATURES,
    )
    validation_scored = score_episode_samples(
        final_model, validation_samples
    )
    validation_decisions = choose_non_overlapping_episode_overrides(
        validation_scored
    )
    validation_summary = summarize_episode_decisions(validation_decisions)
    validation_signal = _signal_diagnostics(validation_scored)
    validation_oracle = oracle_non_overlapping_summary(validation_samples)

    file_io.write_csv(
        output_dir / "dcca_validation_scored_episodes.csv",
        validation_scored,
    )
    file_io.write_csv(
        output_dir / "dcca_validation_decisions.csv",
        validation_decisions,
    )

    candidate_summary = pd.concat(
        [
            _candidate_summary(
                crossfit_decisions_frame,
                "prevalidation_crossfit",
            ),
            _candidate_summary(
                validation_decisions,
                "retrospective_validation",
            ),
        ],
        ignore_index=True,
    )
    file_io.write_csv(
        output_dir / "dcca_candidate_summary.csv",
        candidate_summary,
    )

    crossfit_total = float(
        sum(
            float(stage["realized_marginal_log_sum"])
            for stage in crossfit_stages
        )
    )
    validation_total = float(
        validation_summary["realized_marginal_log_sum"]
    )
    result = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "experiment": EXPERIMENT,
        "base_research": "PCEA v2.0.4 policy-sufficient episode definition",
        "strategy_sequence": int(args.strategy_sequence),
        "strategy_id": str(strategy.get("_id") or ""),
        "strategy_revision": int(strategy.get("revision") or 0),
        "strategy_configuration_hash": strategy.get("configuration_hash"),
        "model_family": family,
        "model_settings_hash": settings_hash,
        "baseline_asset_count": len(baseline_assets),
        "candidate_pool_count": len(candidate_assets),
        "candidate_identity_feature_used": False,
        "baseline_policy_modified_before_intervention": False,
        "minimum_holding_guard_respected": True,
        "raw_candidate_score_required_to_win_before_sampling": False,
        "sampling_change_from_pcea_v2": (
            "replace naturally occurring candidate-led divergence episodes with "
            "dense forced candidate-date interventions from the exact immutable "
            "baseline prefix state"
        ),
        "pooled_estimator": "StandardScaler + BayesianRidge",
        "pooled_estimator_changed_from_pcea_v2": False,
        "feature_count": len(MODEL_FEATURES),
        "features": MODEL_FEATURES,
        "target_column": TARGET_COLUMN,
        "episode_definition_version": DCCA_EPISODE_DEFINITION_VERSION,
        "decision_threshold": 0.0,
        "decision_threshold_basis": (
            "economic indifference; no confidence cutoff or hand-tuned margin"
        ),
        "prevalidation_fold_count": len(folds),
        "prevalidation_crossfit_stages": crossfit_stages,
        "prevalidation_crossfit_realized_marginal_log_sum": crossfit_total,
        "prevalidation_crossfit_incremental_factor": float(
            math.exp(crossfit_total) - 1.0
        ),
        "validation_start": validation_start.date().isoformat(),
        "validation_end": validation_end.date().isoformat(),
        "validation_sessions": int(len(validation_decision_dates) - 1),
        "validation_period_previously_observed_by_researcher": True,
        "validation_training_used_validation_period": False,
        "validation_purge_sessions": int(validation_purge),
        "validation_calibrated_switch_margin": float(
            validation_calibrated_margin
        ),
        "validation_effective_switch_margin": float(
            validation_effective_margin
        ),
        "validation_baseline_calibration_score": float(
            validation_calibration_score
        ),
        "validation_episode_diagnostics": validation_diagnostics,
        "validation_model_diagnostics": final_model.diagnostics(),
        "validation_signal_diagnostics": validation_signal,
        "validation_oracle_diagnostic": validation_oracle,
        "validation_summary": validation_summary,
        "validation_incremental_factor": float(
            math.exp(validation_total) - 1.0
        ),
        "validation_signal_positive": bool(validation_total > 0.0),
        "full_stateful_overlay_backtest_run": False,
        "next_step_if_signal_generalizes": (
            "Integrate DCCA admission into one exact stateful overlay account and "
            "compare final capital with the immutable 56-asset baseline."
        ),
    }
    file_io.write_json(output_dir / "dcca_result.json", result)
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
            "candidate_pool_frozen_before_validation_load": True,
            "baseline_assets": baseline_assets,
            "candidate_assets": candidate_assets,
            "pooled_features": MODEL_FEATURES,
            "target_column": TARGET_COLUMN,
            "pooled_model": "BayesianRidge",
            "counterfactual_sampling": "forced_candidate_date",
            "raw_candidate_score_required_to_win": False,
            "minimum_holding_guard_respected": True,
            "episode_definition_version": DCCA_EPISODE_DEFINITION_VERSION,
            "validation_period_previously_observed_by_researcher": True,
        },
    )

    _log("=== DENSE COUNTERFACTUAL CANDIDATE ADVANTAGE V1 RESULT ===")
    _log(
        f"Prevalidation cross-fit marginal log sum: {crossfit_total:+.6f} "
        f"(factor={math.exp(crossfit_total)-1.0:+.2%})."
    )
    _log(
        f"Retrospective validation: overrides="
        f"{validation_summary['override_count']}/"
        f"{validation_summary['decision_episode_starts']}, "
        f"marginal_log_sum={validation_total:+.6f}, "
        f"factor={math.exp(validation_total)-1.0:+.2%}, "
        f"spearman={validation_signal['prediction_realized_spearman']}."
    )
    _log(
        "DCCA v1 changes one failure family only: sampling density/selection bias. "
        "The pooled estimator, features, economic zero boundary and immutable "
        "baseline remain unchanged."
    )
    _log(f"Result directory: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
