from __future__ import annotations

import argparse
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
import research_windows_file_io as file_io  # noqa: E402
from research_asset_timing_vs_buyhold_execution import _immutable_model_snapshot  # noqa: E402
from research_marginal_reproducibility import code_identity, market_data_hashes  # noqa: E402
from market_cycle_trader_api.engine.capital_rotation import (  # noqa: E402
    _build_walk_forward_folds,
    _utility_policy,
    prepare_rotation_panel,
)
from market_cycle_trader_api.services.cross_asset_utility_signature import (  # noqa: E402
    PREDICTION_COLUMN,
    TARGET_COLUMN,
    choose_candidate_overrides_on_common_scale,
    common_scale_signal_metrics,
    fit_cross_asset_utility_model,
    score_common_scale_utility,
    summarize_common_scale_overrides,
)

SCRIPT_VERSION = "cross-asset-utility-signature-v1.0.0"
EXPERIMENT = "baseline_trained_transferable_common_scale_utility_signature"
DEFAULT_VALIDATION_SESSIONS = 252


def _log(message: str) -> None:
    stamp = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{stamp}] {message}", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Research whether the protected 56-asset universe contains a transferable "
            "cross-asset Utility signature. One pooled LightGBM is trained only on baseline "
            "assets and predicts the same forward_risk_adjusted_utility for baseline and "
            "external assets on one common score scale."
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
            "Delete only this experiment output directory and rebuild from the local "
            "MongoDB market cache. No PCMA/PCEA/DCCA/marginal artifacts are read."
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
        "cross_asset_utility_signature_strategy_"
    ):
        raise RuntimeError(f"Refusing to delete unexpected research path: {resolved}")


def _assert_label_maturity(
    common_dates: pd.DatetimeIndex,
    training_dates: pd.DatetimeIndex,
    decision_dates: pd.DatetimeIndex,
    maximum_label_horizon: int,
    *,
    phase: str,
) -> None:
    if training_dates.empty or decision_dates.empty:
        raise RuntimeError(f"{phase}: empty training or decision window.")
    last_train = pd.Timestamp(training_dates[-1])
    first_decision = pd.Timestamp(decision_dates[0])
    train_location = common_dates.get_indexer([last_train])
    decision_location = common_dates.get_indexer([first_decision])
    if len(train_location) != 1 or len(decision_location) != 1:
        raise RuntimeError(f"{phase}: training/decision boundary is not on aligned calendar.")
    last_label_location = int(train_location[0]) + int(maximum_label_horizon)
    if last_label_location > int(decision_location[0]):
        raise RuntimeError(
            f"{phase}: pooled Utility training label horizon crosses the first OOS decision."
        )


def _baseline_action_path(
    *,
    frames: dict[str, pd.DataFrame],
    baseline_symbols: list[str],
    baseline_models: dict[str, Any],
    decision_dates: pd.DatetimeIndex,
    config: Any,
    effective_margin: float,
    calibrated_margin: float,
    fold_id: int,
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
    current_position = 0
    holding_days = 0
    rows: list[dict[str, Any]] = []
    for timestamp in pd.DatetimeIndex(decision_dates).sort_values()[:-1]:
        timestamp = pd.Timestamp(timestamp)
        target_position, target_raw_score = policy(
            timestamp, current_position, holding_days
        )
        target_position = int(target_position)
        target_asset = (
            baseline_symbols[target_position - 1]
            if target_position > 0
            else "CASH"
        )
        rows.append(
            {
                "timestamp": timestamp,
                "baseline_current_asset": (
                    baseline_symbols[current_position - 1]
                    if current_position > 0
                    else "CASH"
                ),
                "baseline_holding_days": int(holding_days),
                "baseline_target_asset": target_asset,
                "baseline_independent_raw_score": (
                    float(target_raw_score)
                    if np.isfinite(float(target_raw_score))
                    else np.nan
                ),
            }
        )
        current_position, holding_days = pcma._state_update(
            current_position, holding_days, target_position
        )
    return pd.DataFrame(rows)


def _baseline_action_common_scores(
    actions: pd.DataFrame,
    baseline_scored: pd.DataFrame,
) -> pd.DataFrame:
    scored = baseline_scored.copy()
    scored["timestamp"] = pd.to_datetime(
        scored["timestamp"], utc=True, format="mixed", errors="raise"
    )
    lookup = scored.set_index(["timestamp", "symbol"])
    rows: list[dict[str, Any]] = []
    for row in actions.to_dict(orient="records"):
        timestamp = pd.Timestamp(row["timestamp"])
        symbol = str(row["baseline_target_asset"]).upper()
        if symbol == "CASH":
            predicted = 0.0
            realized = 0.0
        else:
            key = (timestamp, symbol)
            if key not in lookup.index:
                continue
            value = lookup.loc[key]
            if isinstance(value, pd.DataFrame):
                raise RuntimeError(
                    f"Duplicate common-scale baseline score for {timestamp} {symbol}."
                )
            predicted = float(value[PREDICTION_COLUMN])
            realized_raw = pd.to_numeric(
                pd.Series([value["realized_forward_risk_adjusted_utility"]]),
                errors="coerce",
            ).iloc[0]
            realized = float(realized_raw) if pd.notna(realized_raw) else np.nan
        rows.append(
            {
                **row,
                "baseline_common_scale_predicted_utility": predicted,
                "baseline_realized_forward_risk_adjusted_utility": realized,
            }
        )
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def _period(
    *,
    fold_id: int,
    frames: dict[str, pd.DataFrame],
    common_dates: pd.DatetimeIndex,
    baseline_symbols: list[str],
    candidate_symbols: list[str],
    train_dates: pd.DatetimeIndex,
    calibration_dates: pd.DatetimeIndex,
    final_fit_dates: pd.DatetimeIndex,
    decision_dates: pd.DatetimeIndex,
    config: Any,
    phase: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    maximum_label_horizon = max(int(x) for x in config.rotation_target_horizons)
    _assert_label_maturity(
        common_dates,
        final_fit_dates,
        decision_dates,
        maximum_label_horizon,
        phase=phase,
    )

    _, calibrated_margin, calibration_score = pcma._calibrate_baseline_margin(
        frames,
        baseline_symbols,
        train_dates,
        calibration_dates,
        config,
        phase_prefix=f"{phase}_baseline",
    )
    effective_margin = max(
        float(config.rotation_switch_margin),
        float(calibrated_margin),
    )
    independent_models = pcma._fit_final_models(
        frames,
        baseline_symbols,
        final_fit_dates,
        config,
        phase=f"{phase}_protected_baseline_final",
        require_all=True,
    )
    pooled_model = fit_cross_asset_utility_model(
        frames,
        baseline_symbols,
        final_fit_dates,
        config,
    )
    eval_dates = pd.DatetimeIndex(decision_dates).sort_values()[:-1]
    baseline_scored = score_common_scale_utility(
        pooled_model,
        frames,
        baseline_symbols,
        eval_dates,
        role="protected_baseline",
    )
    candidate_scored = score_common_scale_utility(
        pooled_model,
        frames,
        candidate_symbols,
        eval_dates,
        role="external_candidate",
    )
    actions = _baseline_action_path(
        frames=frames,
        baseline_symbols=baseline_symbols,
        baseline_models=independent_models,
        decision_dates=decision_dates,
        config=config,
        effective_margin=effective_margin,
        calibrated_margin=calibrated_margin,
        fold_id=fold_id,
    )
    baseline_action_scores = _baseline_action_common_scores(
        actions,
        baseline_scored,
    )
    decisions = choose_candidate_overrides_on_common_scale(
        candidate_scored,
        baseline_action_scores,
    )

    baseline_metrics = common_scale_signal_metrics(baseline_scored)
    candidate_metrics = common_scale_signal_metrics(candidate_scored)
    decision_summary = summarize_common_scale_overrides(decisions)
    summary = {
        "fold": int(fold_id),
        "phase": phase,
        "calibrated_switch_margin": float(calibrated_margin),
        "effective_switch_margin": float(effective_margin),
        "baseline_calibration_score": float(calibration_score),
        "candidate_count": int(candidate_scored["symbol"].nunique())
        if not candidate_scored.empty
        else 0,
        "decision_date_count": int(len(eval_dates)),
        "pooled_model_trained_on_external_candidates": False,
        **{f"baseline_{key}": value for key, value in baseline_metrics.items()},
        **{f"candidate_{key}": value for key, value in candidate_metrics.items()},
        **{f"override_{key}": value for key, value in decision_summary.items()},
        **{f"model_{key}": value for key, value in pooled_model.diagnostics().items()},
    }
    _log(
        f"{phase}: pooled baseline daily-rank={baseline_metrics['mean_daily_cross_section_spearman']}, "
        f"candidate daily-rank={candidate_metrics['mean_daily_cross_section_spearman']}, "
        f"overrides={decision_summary['override_count']}/{decision_summary['decision_dates']}, "
        f"realized utility advantage sum={decision_summary['realized_utility_advantage_sum']:+.6f}."
    )
    return baseline_scored, candidate_scored, decisions, summary


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
            f"cross_asset_utility_signature_strategy_{args.strategy_sequence}_"
            f"{validation_start.date().isoformat()}_to_{validation_end.date().isoformat()}"
        )
    ).resolve()
    if args.fresh_run:
        _assert_safe_fresh_root(output_dir)
        if file_io.exists(output_dir):
            _log(f"FRESH RUN: deleting prior CAUS artifacts only from {output_dir}")
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
        output_dir / "caus_selection_history_integrity.csv",
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
        f"Frozen candidate pool: {len(candidate_assets)} complete external assets. "
        "The pooled Utility model will train on baseline assets only."
    )

    raw_validation = timing._load_frames_allow_incomplete(
        collection,
        research_assets,
        identity,
        history_start,
        validation_end,
    )
    family, settings, settings_hash = _immutable_model_snapshot(strategy)
    if str(family) != "lightgbm_utility":
        raise RuntimeError(
            f"CAUS v1 is defined for the LightGBM Utility baseline; got {family}."
        )
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
        raw_selection,
        selection_config,
    )
    folds = _build_walk_forward_folds(selection_dates, selection_config)
    if len(folds) < 3:
        raise RuntimeError(
            f"CAUS v1 requires at least three pre-validation folds; got {len(folds)}."
        )

    baseline_symbols = sorted(baseline_assets)
    selection_candidates = sorted(
        symbol for symbol in candidate_assets if symbol in selection_frames
    )
    fold_summaries: list[dict[str, Any]] = []
    fold_baseline_scores: list[pd.DataFrame] = []
    fold_candidate_scores: list[pd.DataFrame] = []
    fold_decisions: list[pd.DataFrame] = []

    for fold in folds:
        fold_id = int(fold["fold_id"])
        train_dates, calibration_dates, final_fit_dates, decision_dates = pcma._fold_dates(
            selection_dates, fold
        )
        _log(
            f"PREVALIDATION FOLD {fold_id}: train={len(train_dates)}, "
            f"calibration={len(calibration_dates)}, final_fit={len(final_fit_dates)}, "
            f"decisions={max(0, len(decision_dates)-1)}."
        )
        baseline_scored, candidate_scored, decisions, summary = _period(
            fold_id=fold_id,
            frames=selection_frames,
            common_dates=selection_dates,
            baseline_symbols=baseline_symbols,
            candidate_symbols=selection_candidates,
            train_dates=train_dates,
            calibration_dates=calibration_dates,
            final_fit_dates=final_fit_dates,
            decision_dates=decision_dates,
            config=selection_config,
            phase=f"prevalidation_fold_{fold_id}",
        )
        baseline_scored["fold"] = fold_id
        candidate_scored["fold"] = fold_id
        decisions["fold"] = fold_id
        fold_baseline_scores.append(baseline_scored)
        fold_candidate_scores.append(candidate_scored)
        fold_decisions.append(decisions)
        fold_summaries.append(summary)

    pre_baseline = pd.concat(fold_baseline_scores, ignore_index=True)
    pre_candidates = pd.concat(fold_candidate_scores, ignore_index=True)
    pre_decisions = pd.concat(fold_decisions, ignore_index=True)
    file_io.write_csv(
        output_dir / "caus_prevalidation_baseline_common_scores.csv", pre_baseline
    )
    file_io.write_csv(
        output_dir / "caus_prevalidation_candidate_common_scores.csv", pre_candidates
    )
    file_io.write_csv(
        output_dir / "caus_prevalidation_decisions.csv", pre_decisions
    )
    file_io.write_json(
        output_dir / "caus_prevalidation_fold_summary.json", fold_summaries
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
    validation_fold = max(int(fold["fold_id"]) for fold in folds) + 1
    validation_candidates = sorted(
        symbol for symbol in candidate_assets if symbol in validation_frames
    )
    (
        validation_baseline_scored,
        validation_candidate_scored,
        validation_decisions,
        validation_summary,
    ) = _period(
        fold_id=validation_fold,
        frames=validation_frames,
        common_dates=validation_dates,
        baseline_symbols=baseline_symbols,
        candidate_symbols=validation_candidates,
        train_dates=validation_train_dates,
        calibration_dates=validation_calibration_dates,
        final_fit_dates=validation_final_fit_dates,
        decision_dates=validation_decision_dates,
        config=validation_config,
        phase="retrospective_validation",
    )
    validation_summary["purge_sessions"] = int(validation_purge)

    file_io.write_csv(
        output_dir / "caus_validation_baseline_common_scores.csv",
        validation_baseline_scored,
    )
    file_io.write_csv(
        output_dir / "caus_validation_candidate_common_scores.csv",
        validation_candidate_scored,
    )
    file_io.write_csv(
        output_dir / "caus_validation_decisions.csv",
        validation_decisions,
    )

    pre_override_sum = float(
        sum(
            float(stage["override_realized_utility_advantage_sum"])
            for stage in fold_summaries
        )
    )
    pre_positive_folds = int(
        sum(
            float(stage["override_realized_utility_advantage_sum"]) > 0.0
            for stage in fold_summaries
        )
    )
    source_identity = code_identity()
    result = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "experiment": EXPERIMENT,
        "code_identity": source_identity,
        "strategy_sequence": int(args.strategy_sequence),
        "strategy_id": str(strategy.get("_id") or ""),
        "strategy_revision": int(strategy.get("revision") or 0),
        "strategy_configuration_hash": strategy.get("configuration_hash"),
        "model_family": family,
        "model_settings_hash": settings_hash,
        "baseline_asset_count": len(baseline_assets),
        "candidate_pool_count": len(candidate_assets),
        "signature_training_assets": "protected baseline assets only",
        "pooled_model_trained_on_external_candidates": False,
        "candidate_identity_feature_used": False,
        "pooled_target": TARGET_COLUMN,
        "pooled_features": list(pcma.ROTATION_FEATURES),
        "common_scale_model": "LightGBM pooled across protected baseline assets",
        "individual_baseline_policy_preserved_for_reference_action": True,
        "external_candidate_individual_models_required": False,
        "decision_rule": (
            "choose the external candidate with highest common-scale predicted Utility; "
            "override reference action only when candidate predicted Utility exceeds the "
            "reference action predicted Utility on the same pooled model scale"
        ),
        "decision_boundary": 0.0,
        "decision_boundary_basis": "common-scale predicted Utility indifference",
        "prevalidation_fold_summaries": fold_summaries,
        "prevalidation_override_realized_utility_advantage_sum": pre_override_sum,
        "prevalidation_positive_fold_count": pre_positive_folds,
        "validation_start": validation_start.date().isoformat(),
        "validation_end": validation_end.date().isoformat(),
        "validation_sessions": int(len(validation_decision_dates) - 1),
        "validation_period_previously_observed_by_researcher": True,
        "validation_training_used_validation_period": False,
        "validation_summary": validation_summary,
        "full_stateful_overlay_backtest_run": False,
        "utility_advantage_sum_is_portfolio_return": False,
        "next_step_if_common_scale_signal_generalizes": (
            "Run one research-only stateful overlay account using the common-scale "
            "candidate gate and compare exact final capital with the immutable 56-asset baseline."
        ),
    }
    file_io.write_json(output_dir / "caus_result.json", result)
    file_io.write_json(
        output_dir / "experiment_manifest.json",
        {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "experiment": EXPERIMENT,
            "code_identity": source_identity,
            "fresh_run": bool(args.fresh_run),
            "prior_research_artifacts_read": False,
            "mongo_writes": False,
            "alpaca_network_used": False,
            "candidate_pool_frozen_before_validation_load": True,
            "pooled_model_training_universe": baseline_assets,
            "candidate_assets": candidate_assets,
            "candidate_identity_feature_used": False,
            "external_candidate_targets_used_for_model_training": False,
            "target_column": TARGET_COLUMN,
            "market_selection_sha256_by_asset": market_data_hashes(raw_selection),
            "market_validation_sha256_by_asset": market_data_hashes(raw_validation),
        },
    )

    _log("=== CROSS-ASSET UTILITY SIGNATURE V1 RESULT ===")
    for stage in fold_summaries:
        _log(
            f"Fold {stage['fold']}: candidate daily-rank="
            f"{stage['candidate_mean_daily_cross_section_spearman']}, "
            f"override utility advantage sum="
            f"{stage['override_realized_utility_advantage_sum']:+.6f}."
        )
    _log(
        f"Retrospective validation: candidate daily-rank="
        f"{validation_summary['candidate_mean_daily_cross_section_spearman']}, "
        f"overrides={validation_summary['override_override_count']}/"
        f"{validation_summary['override_decision_dates']}, "
        f"utility advantage sum="
        f"{validation_summary['override_realized_utility_advantage_sum']:+.6f}."
    )
    _log(
        "These Utility advantages are signal diagnostics, not compounded portfolio returns."
    )
    _log(f"Result directory: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
