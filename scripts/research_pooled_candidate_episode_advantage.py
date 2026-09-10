from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

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
import research_windows_file_io as file_io  # noqa: E402
from research_marginal_reproducibility import code_identity, market_data_hashes  # noqa: E402
from research_asset_timing_vs_buyhold_execution import _immutable_model_snapshot  # noqa: E402
from market_cycle_trader_api.engine.capital_rotation import (  # noqa: E402
    _build_walk_forward_folds,
    prepare_rotation_panel,
)
from market_cycle_trader_api.services.asset_marginal_score_replay import (  # noqa: E402
    ScoreReplayPanel,
    export_execution_rows,
    replay_policy_settings,
)
from market_cycle_trader_api.services.pooled_candidate_episode_advantage import (  # noqa: E402
    TARGET_COLUMN,
    EPISODE_COLUMNS,
    EPISODE_DEFINITION_VERSION,
    LABEL_AVAILABLE_COLUMN,
    OBSERVED_TARGET_COLUMN,
    build_episode_samples,
    choose_non_overlapping_episode_overrides,
    extract_candidate_divergence_episodes,
    fit_episode_model,
    score_episode_samples,
    summarize_episode_decisions,
    training_episode_samples,
)

SCRIPT_VERSION = "pooled-candidate-episode-advantage-v2.1.1"
EXPERIMENT = "pooled_asset_agnostic_stateful_divergence_episode_advantage"
DEFAULT_VALIDATION_SESSIONS = 252
MODEL_FEATURES = list(pcma.MODEL_FEATURES)


def _log(message: str) -> None:
    stamp = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{stamp}] {message}", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Learn an asset-agnostic candidate signature using non-overlapping, stateful "
            "divergence episodes from exact cached-score account replays. The 56-asset "
            "baseline, LightGBM models/features, costs and pooled Bayesian estimator remain "
            "unchanged from PCMA v1; only the marginal target/sampling unit changes."
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
            "Delete only this experiment's output directory and recompute from the local "
            "MongoDB market cache. No prior PCMA or marginal-research artifacts are read."
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
        "pooled_candidate_episode_advantage_strategy_"
    ):
        raise RuntimeError(f"Refusing to delete unexpected research path: {resolved}")


def _replay_tape(
    *,
    frames: dict[str, pd.DataFrame],
    models: dict[str, Any],
    symbols: list[str],
    fold: dict[str, Any],
    maximum_label_horizon: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        model = models.get(symbol)
        if model is None or symbol not in frames:
            continue
        scored = leadership._prediction_rows(symbol, frames[symbol], fold, model)
        rows.extend(
            export_execution_rows(
                scored,
                frames[symbol],
                fold,
                maximum_label_horizon,
            )
        )
    return pd.DataFrame(rows)


def _episode_samples_for_period(
    *,
    fold_id: int,
    frames: dict[str, pd.DataFrame],
    baseline_symbols: list[str],
    candidate_symbols: list[str],
    baseline_models: dict[str, Any],
    candidate_models: dict[str, Any],
    candidate_references: dict[str, tuple[float, float, Any]],
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
        include_future_targets=False,
    )

    all_models = {**baseline_models, **candidate_models}
    tape = _replay_tape(
        frames=frames,
        models=all_models,
        symbols=[*baseline_symbols, *candidate_symbols],
        fold=replay_fold,
        maximum_label_horizon=maximum_label_horizon,
    )
    if tape.empty:
        raise RuntimeError(f"{phase_label}: no cached-score replay tape was produced.")

    panel_end = pd.to_datetime(tape["execution_timestamp"], utc=True, format="mixed").max()
    panel = ScoreReplayPanel(
        tape,
        baseline_symbols,
        pd.Timestamp(panel_end).date().isoformat(),
    )
    settings = replay_policy_settings(config)
    # The margin was calibrated strictly before this evaluation period. Using it here
    # keeps the replay aligned with the protected baseline policy used for features.
    settings["rotation_switch_margin"] = float(effective_margin)

    baseline_replay = panel.replay(baseline_symbols, settings)
    episode_frames: list[pd.DataFrame] = []
    complete_candidates: list[str] = []
    for candidate in candidate_symbols:
        if not panel.complete(candidate):
            continue
        complete_candidates.append(candidate)
        expanded = panel.replay([*baseline_symbols, candidate], settings)
        episodes = extract_candidate_divergence_episodes(
            baseline_daily=baseline_replay.daily,
            expanded_daily=expanded.daily,
            candidate=candidate,
            fold_id=fold_id,
        )
        if not episodes.empty:
            episode_frames.append(episodes)

    episodes = (
        pd.concat(episode_frames, ignore_index=True)
        if episode_frames
        else pd.DataFrame(columns=EPISODE_COLUMNS)
    )
    samples = build_episode_samples(
        daily_features=daily_features,
        episodes=episodes,
    )
    diagnostics = {
        "fold": int(fold_id),
        "phase": phase_label,
        "decision_dates": int(max(0, len(decision_dates) - 1)),
        "candidate_models": int(len(candidate_symbols)),
        "replay_complete_candidates": int(len(complete_candidates)),
        "raw_episode_count": int(len(episodes)),
        "right_censored_episode_count": (
            int(episodes["right_censored"].astype(bool).sum())
            if not episodes.empty
            else 0
        ),
        "scoring_episode_count": int(len(samples)),
        "usable_episode_count": int((~samples["right_censored"]).sum()) if not samples.empty else 0,
        "baseline_replay_ending_capital": float(baseline_replay.ending_capital),
        "baseline_replay_net_log_growth": float(baseline_replay.net_log_growth),
        "effective_switch_margin": float(effective_margin),
    }
    _log(
        f"{phase_label}: stateful replay produced {len(samples)} scorable divergence "
        f"episodes ({diagnostics['right_censored_episode_count']} right-censored; training only excludes them) "
        f"from {len(complete_candidates)} candidates."
    )
    return samples, episodes, diagnostics


def _synthetic_holdout_fold(
    *,
    fold_id: int,
    decision_dates: pd.DatetimeIndex,
    common_dates: pd.DatetimeIndex,
    final_fit_dates: pd.DatetimeIndex,
) -> dict[str, Any]:
    if final_fit_dates.empty:
        raise RuntimeError("Holdout final-fit window is empty.")
    last = pd.Timestamp(final_fit_dates[-1])
    matches = common_dates.get_indexer([last])
    if len(matches) != 1 or int(matches[0]) < 0:
        raise RuntimeError("Could not locate holdout final-fit endpoint in aligned calendar.")
    return {
        "fold_id": int(fold_id),
        "decision_dates": pd.DatetimeIndex(decision_dates).sort_values(),
        "final_fit_end_index": int(matches[0]) + 1,
    }


def _candidate_summary(decisions: pd.DataFrame, phase: str) -> pd.DataFrame:
    columns = [
        "phase", "candidate", "chosen_episode_count", "predicted_episode_advantage_mean",
        *summarize_episode_decisions(pd.DataFrame()).keys(),
    ]
    if decisions.empty:
        return pd.DataFrame(columns=columns)
    chosen = decisions.loc[decisions["override_baseline"].astype(bool)].copy()
    rows: list[dict[str, Any]] = []
    for candidate, group in chosen.groupby("chosen_candidate"):
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
                **summarize_episode_decisions(group),
            }
        )
    return pd.DataFrame(rows, columns=columns)


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
            f"pooled_candidate_episode_advantage_strategy_{args.strategy_sequence}_v2_1_"
            f"{validation_start.date().isoformat()}_to_{validation_end.date().isoformat()}"
        )
    ).resolve()
    if args.fresh_run:
        _assert_safe_fresh_root(output_dir)
        if file_io.exists(output_dir):
            _log(f"FRESH RUN: deleting prior PCEA artifacts only from {output_dir}")
            file_io.remove_tree(output_dir)
    elif file_io.exists(output_dir):
        raise RuntimeError(
            f"Output directory already exists: {output_dir}. This research does not resume. "
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
    strategy = common._strategy_document(db, args.strategy_sequence, args.strategy_id)
    configuration = common._configuration(strategy)
    base_config = leadership.BacktestRequest.model_validate(configuration)
    baseline_assets = list(
        dict.fromkeys(
            str(x).strip().upper() for x in base_config.assets if str(x).strip()
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
        output_dir / "pcea_selection_history_integrity.csv",
        pd.DataFrame(history_rows),
    )
    complete_set = set(complete_assets)
    missing_baseline = [x for x in baseline_assets if x not in complete_set]
    if missing_baseline:
        raise RuntimeError(
            "Immutable Strategy baseline failed Full Strategy History: "
            + ", ".join(missing_baseline)
        )
    candidate_assets = [x for x in external if x in complete_set]
    research_assets = [*baseline_assets, *candidate_assets]
    raw_selection = {x: raw_selection_all[x] for x in research_assets}
    _log(
        f"Frozen pre-validation candidate pool: {len(candidate_assets)} complete external assets."
    )

    raw_holdout = timing._load_frames_allow_incomplete(
        collection,
        research_assets,
        identity,
        history_start,
        validation_end,
    )
    family, settings, settings_hash = _immutable_model_snapshot(strategy)
    client.close()
    _log(
        "MongoDB closed. Candidate pool is frozen; all remaining work uses immutable "
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
        raw_selection,
        selection_config,
    )
    folds = _build_walk_forward_folds(selection_dates, selection_config)
    if len(folds) < 3:
        raise RuntimeError(
            f"PCEA v2 requires at least three pre-validation walk-forward folds; got {len(folds)}."
        )

    maximum_label_horizon = max(
        int(h) for h in selection_config.rotation_target_horizons
    )
    baseline_symbols = sorted(baseline_assets)
    selection_candidates = sorted(
        [x for x in candidate_assets if x in selection_frames]
    )
    all_fold_samples: list[pd.DataFrame] = []
    all_fold_episodes: list[pd.DataFrame] = []
    fold_diagnostics: list[dict[str, Any]] = []

    for fold in folds:
        fold_id = int(fold["fold_id"])
        train_dates, calibration_dates, final_fit_dates, decision_dates = pcma._fold_dates(
            selection_dates,
            fold,
        )
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
            phase_prefix=f"pcea_fold_{fold_id}",
        )
        baseline_models = pcma._fit_final_models(
            selection_frames,
            baseline_symbols,
            final_fit_dates,
            selection_config,
            phase=f"pcea_fold_{fold_id}_baseline_final",
            require_all=True,
        )
        candidate_models = pcma._fit_final_models(
            selection_frames,
            selection_candidates,
            final_fit_dates,
            selection_config,
            phase=f"pcea_fold_{fold_id}_candidate_final",
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
        samples, episodes, diag = _episode_samples_for_period(
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
        diag["calibrated_switch_margin"] = float(calibrated_margin)
        diag["baseline_calibration_score"] = float(calibration_score)
        all_fold_samples.append(samples)
        all_fold_episodes.append(episodes)
        fold_diagnostics.append(diag)

    usable_sample_frames = [x for x in all_fold_samples if not x.empty]
    usable_episode_frames = [x for x in all_fold_episodes if not x.empty]
    if not usable_sample_frames:
        raise RuntimeError("No usable pre-validation divergence episodes were generated.")
    prevalidation_samples = pd.concat(
        usable_sample_frames,
        ignore_index=True,
    )
    prevalidation_episodes = (
        pd.concat(usable_episode_frames, ignore_index=True)
        if usable_episode_frames
        else pd.DataFrame()
    )

    file_io.write_csv(
        output_dir / "pcea_prevalidation_episode_samples.csv",
        prevalidation_samples,
    )
    file_io.write_csv(
        output_dir / "pcea_prevalidation_all_episodes.csv",
        prevalidation_episodes,
    )
    file_io.write_csv(
        output_dir / "pcea_fold_diagnostics.csv",
        pd.DataFrame(fold_diagnostics),
    )

    crossfit_scored: list[pd.DataFrame] = []
    crossfit_decisions: list[pd.DataFrame] = []
    crossfit_stages: list[dict[str, Any]] = []
    fold_ids = sorted(int(fold["fold_id"]) for fold in folds)
    for test_fold in fold_ids[1:]:
        train = prevalidation_samples.loc[
            prevalidation_samples["fold"] < test_fold
        ].copy()
        test = prevalidation_samples.loc[
            prevalidation_samples["fold"] == test_fold
        ].copy()
        if train.empty or test.empty:
            continue
        cutoff = next(
            pd.DatetimeIndex(fold["decision_dates"]).min()
            for fold in folds if int(fold["fold_id"]) == test_fold
        )
        mature_train = training_episode_samples(train, available_before=cutoff)
        model = fit_episode_model(train, MODEL_FEATURES, available_before=cutoff)
        scored = score_episode_samples(model, test)
        decisions = choose_non_overlapping_episode_overrides(scored)
        summary = summarize_episode_decisions(decisions)
        crossfit_stages.append(
            {
                "test_fold": int(test_fold),
                "training_folds": sorted(
                    int(x) for x in mature_train["fold"].unique()
                ),
                "training_cutoff_exclusive": str(cutoff),
                "training_last_label_available_at": str(mature_train[LABEL_AVAILABLE_COLUMN].max()),
                "excluded_training_episode_count": int(len(train) - len(mature_train)),
                **model.diagnostics(),
                **summary,
            }
        )
        scored["evaluation_stage"] = f"test_fold_{test_fold}"
        decisions["evaluation_stage"] = f"test_fold_{test_fold}"
        crossfit_scored.append(scored)
        crossfit_decisions.append(decisions)
        _log(
            f"CROSSFIT fold {test_fold}: episode_overrides="
            f"{summary['override_count']}/{summary['decision_episode_starts']}, "
            f"observed_marginal_log_sum={summary['observed_marginal_log_sum']:+.6f}, "
            f"open_overrides={summary['right_censored_override_count']}."
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
        output_dir / "pcea_crossfit_scored_episodes.csv",
        crossfit_scored_frame,
    )
    file_io.write_csv(
        output_dir / "pcea_crossfit_decisions.csv",
        crossfit_decisions_frame,
    )
    file_io.write_json(
        output_dir / "pcea_crossfit_summary.json",
        crossfit_stages,
    )

    holdout_config = leadership._execution_config(
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
    holdout_frames, holdout_dates = prepare_rotation_panel(
        raw_holdout,
        holdout_config,
    )
    (
        holdout_train_dates,
        holdout_calibration_dates,
        holdout_final_fit_dates,
        holdout_purge,
        validation_index,
    ) = validation._strict_training_windows(
        holdout_dates,
        validation_start,
        holdout_config,
    )
    holdout_decision_dates = validation._decision_dates(
        holdout_dates,
        validation_index,
        validation_end,
    )
    _, holdout_calibrated_margin, holdout_calibration_score = (
        pcma._calibrate_baseline_margin(
            holdout_frames,
            baseline_symbols,
            holdout_train_dates,
            holdout_calibration_dates,
            holdout_config,
            phase_prefix="pcea_holdout",
        )
    )
    holdout_baseline_models = pcma._fit_final_models(
        holdout_frames,
        baseline_symbols,
        holdout_final_fit_dates,
        holdout_config,
        phase="pcea_holdout_baseline_final",
        require_all=True,
    )
    holdout_candidate_models = pcma._fit_final_models(
        holdout_frames,
        sorted([x for x in candidate_assets if x in holdout_frames]),
        holdout_final_fit_dates,
        holdout_config,
        phase="pcea_holdout_candidate_final",
        require_all=False,
    )
    holdout_candidates = sorted(holdout_candidate_models)
    holdout_references = {
        candidate: pcma._score_reference(
            holdout_candidate_models[candidate],
            holdout_frames[candidate],
            holdout_final_fit_dates,
        )
        for candidate in holdout_candidates
    }
    holdout_effective_margin = max(
        float(holdout_config.rotation_switch_margin),
        float(holdout_calibrated_margin),
    )
    holdout_fold_id = max(fold_ids) + 1
    holdout_replay_fold = _synthetic_holdout_fold(
        fold_id=holdout_fold_id,
        decision_dates=holdout_decision_dates,
        common_dates=holdout_dates,
        final_fit_dates=holdout_final_fit_dates,
    )
    holdout_samples, holdout_episodes, holdout_diag = _episode_samples_for_period(
        fold_id=holdout_fold_id,
        frames=holdout_frames,
        baseline_symbols=baseline_symbols,
        candidate_symbols=holdout_candidates,
        baseline_models=holdout_baseline_models,
        candidate_models=holdout_candidate_models,
        candidate_references=holdout_references,
        decision_dates=holdout_decision_dates,
        replay_fold=holdout_replay_fold,
        config=holdout_config,
        effective_margin=holdout_effective_margin,
        calibrated_margin=holdout_calibrated_margin,
        maximum_label_horizon=max(
            int(h) for h in holdout_config.rotation_target_horizons
        ),
        phase_label="retrospective_holdout",
    )

    file_io.write_csv(
        output_dir / "pcea_holdout_episode_samples.csv",
        holdout_samples,
    )
    file_io.write_csv(
        output_dir / "pcea_holdout_all_episodes.csv",
        holdout_episodes,
    )

    final_model = fit_episode_model(
        prevalidation_samples,
        MODEL_FEATURES,
        available_before=holdout_decision_dates.min(),
    )
    holdout_scored = score_episode_samples(final_model, holdout_samples)
    holdout_decisions = choose_non_overlapping_episode_overrides(holdout_scored)
    holdout_summary = summarize_episode_decisions(holdout_decisions)
    file_io.write_csv(
        output_dir / "pcea_holdout_scored_episodes.csv",
        holdout_scored,
    )
    file_io.write_csv(
        output_dir / "pcea_holdout_decisions.csv",
        holdout_decisions,
    )

    candidate_frames = [
        _candidate_summary(crossfit_decisions_frame, "prevalidation_crossfit"),
        _candidate_summary(holdout_decisions, "retrospective_holdout"),
    ]
    nonempty_candidate_frames = [frame for frame in candidate_frames if not frame.empty]
    candidate_summary = (
        pd.concat(nonempty_candidate_frames, ignore_index=True)
        if nonempty_candidate_frames else candidate_frames[0]
    )
    file_io.write_csv(
        output_dir / "pcea_candidate_summary.csv",
        candidate_summary,
    )

    crossfit_summary = summarize_episode_decisions(crossfit_decisions_frame)
    crossfit_total = crossfit_summary["realized_marginal_log_sum"]
    holdout_total = holdout_summary["realized_marginal_log_sum"]
    source_identity = code_identity()
    final_training = training_episode_samples(
        prevalidation_samples, available_before=holdout_decision_dates.min()
    )
    result = {
        "schema_version": 2,
        "script_version": SCRIPT_VERSION,
        "episode_definition_version": EPISODE_DEFINITION_VERSION,
        "code_identity": source_identity,
        "experiment": EXPERIMENT,
        "strategy_sequence": int(args.strategy_sequence),
        "strategy_id": str(strategy.get("_id") or ""),
        "strategy_revision": int(strategy.get("revision") or 0),
        "strategy_configuration_hash": strategy.get("configuration_hash"),
        "model_family": family,
        "model_settings_hash": settings_hash,
        "baseline_asset_count": len(baseline_assets),
        "candidate_pool_count": len(candidate_assets),
        "candidate_identity_feature_used": False,
        "baseline_policy_modified": False,
        "pooled_estimator_changed_from_v1": False,
        "pcma_v1_failure_addressed": (
            "replace noisy one-session marginal labels with non-overlapping stateful "
            "candidate-divergence episode log-growth labels"
        ),
        "target": (
            "exact cumulative net-log-growth difference between baseline and "
            "baseline-plus-candidate cached-score replays over one state divergence "
            "episode, ending when policy state reconverges"
        ),
        "right_censored_episodes_used_for_training": False,
        "right_censored_episodes_excluded_from_evaluation": False,
        "inference_requires_future_labels": False,
        "episode_factors_are_portfolio_capital_returns": False,
        "decision_threshold": 0.0,
        "decision_threshold_basis": (
            "economic indifference; no manual confidence or score margin"
        ),
        "pooled_estimator": "StandardScaler + BayesianRidge",
        "feature_count": len(MODEL_FEATURES),
        "features": MODEL_FEATURES,
        "prevalidation_fold_count": len(folds),
        "prevalidation_crossfit_stages": crossfit_stages,
        "prevalidation_crossfit_summary": crossfit_summary,
        "prevalidation_crossfit_realized_marginal_log_sum": crossfit_total,
        "prevalidation_crossfit_incremental_factor": float(
            math.expm1(crossfit_total)
        ) if crossfit_total is not None else None,
        "validation_start": validation_start.date().isoformat(),
        "validation_end": validation_end.date().isoformat(),
        "validation_sessions": int(len(holdout_decision_dates) - 1),
        "holdout_purge_sessions": int(holdout_purge),
        "holdout_calibrated_switch_margin": float(holdout_calibrated_margin),
        "holdout_effective_switch_margin": float(holdout_effective_margin),
        "holdout_baseline_calibration_score": float(
            holdout_calibration_score
        ),
        "holdout_episode_diagnostics": holdout_diag,
        "holdout_model_diagnostics": final_model.diagnostics(),
        "holdout_training_cutoff_exclusive": str(holdout_decision_dates.min()),
        "holdout_training_last_label_available_at": str(final_training[LABEL_AVAILABLE_COLUMN].max()),
        "holdout_excluded_training_episode_count": int(len(prevalidation_samples) - len(final_training)),
        "holdout_summary": holdout_summary,
        "holdout_incremental_factor": float(math.expm1(holdout_total)) if holdout_total is not None else None,
        "holdout_is_untouched": False,
        "holdout_interpretation": "retrospective correctness comparison; period already inspected in prior experiments",
        "selection_used_holdout_period": False,
        "training_used_holdout_period": False,
        "full_stateful_overlay_backtest_run": False,
        "research_next_step_if_positive": (
            "Integrate episode admission into a research-only exact stateful overlay "
            "and compare final capital against the immutable 56-asset baseline."
        ),
    }
    file_io.write_json(output_dir / "pcea_result.json", result)
    file_io.write_json(
        output_dir / "experiment_manifest.json",
        {
            "schema_version": 2,
            "script_version": SCRIPT_VERSION,
            "episode_definition_version": EPISODE_DEFINITION_VERSION,
            "code_identity": source_identity,
            "selection_market_data_sha256_by_asset": market_data_hashes(raw_selection),
            "holdout_market_data_sha256_by_asset": market_data_hashes(raw_holdout),
            "experiment": EXPERIMENT,
            "fresh_run": bool(args.fresh_run),
            "prior_research_artifacts_read": False,
            "mongo_writes": False,
            "alpaca_network_used": False,
            "candidate_pool_frozen_before_holdout_load": True,
            "baseline_assets": baseline_assets,
            "candidate_assets": candidate_assets,
            "pooled_features": MODEL_FEATURES,
            "target_column": TARGET_COLUMN,
            "model": "BayesianRidge",
            "score_replay_engine": "asset_marginal_score_replay v2",
            "non_overlapping_episode_evaluation": True,
            "right_censored_episode_exclusion_from_training": True,
            "right_censored_episode_exclusion_from_evaluation": False,
            "label_available_at_basis": "last execution session of a completed episode",
            "pooled_training_rule": "completed labels available strictly before the scoring session",
            "holdout_is_untouched": False,
            "observed_target_column": OBSERVED_TARGET_COLUMN,
        },
    )

    _log("=== POOLED CANDIDATE EPISODE ADVANTAGE V2.1.1 RESULT ===")
    _log(
        f"Prevalidation cross-fit observed episode log sum: "
        f"{crossfit_summary['observed_marginal_log_sum']:+.6f}; "
        f"open overrides={crossfit_summary['right_censored_override_count']}."
    )
    _log(
        f"Retrospective holdout: episode_overrides={holdout_summary['override_count']}/"
        f"{holdout_summary['decision_episode_starts']}, "
        f"observed_log_sum={holdout_summary['observed_marginal_log_sum']:+.6f}, "
        f"open_overrides={holdout_summary['right_censored_override_count']}."
    )
    _log(
        "Episode diagnostics are not the return of a fully executed overlay portfolio. "
        "PCEA v2.1.1 keeps the v2.1 censoring/label contract and adds a writable "
        "scoring mask for pandas Copy-on-Write compatibility."
    )
    _log(f"Result directory: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
