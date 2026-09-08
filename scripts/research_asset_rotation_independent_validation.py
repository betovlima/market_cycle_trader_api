from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
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
import research_asset_rotation_leadership as leadership  # noqa: E402
from research_asset_timing_vs_buyhold_execution import _immutable_model_snapshot  # noqa: E402
from market_cycle_trader_api.engine.absolute_utility_cash_gate import (  # noqa: E402
    absolute_utility_cash_gate_enabled,
)
from market_cycle_trader_api.engine.capital_rotation import (  # noqa: E402
    _simple_policy_growth,
    _simulate_exact,
    _utility_policy,
    prepare_rotation_panel,
)
from market_cycle_trader_api.engine.compound_rotation_backtest import (  # noqa: E402
    apply_slippage,
    calculate_reference_fees,
)
from market_cycle_trader_api.engine.research_challengers import _lightgbm_fit_models  # noqa: E402
from market_cycle_trader_api.engine.selective_opportunity import (  # noqa: E402
    opportunity_cash_gate_enabled,
    selective_opportunity_enabled,
)
from market_cycle_trader_api.engine.concentrated_allocation import (  # noqa: E402
    portfolio_allocation_enabled,
)
from market_cycle_trader_api.engine.compound_risk_overlay import (  # noqa: E402
    compound_risk_overlay_enabled,
)

SCRIPT_VERSION = "asset-rotation-independent-validation-v1.0"


def _log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate one frozen rotation-qualified universe on a final chronological period "
            "that was never used for asset selection, LightGBM training, or margin calibration."
        )
    )
    parser.add_argument("--strategy-sequence", type=int, default=10)
    parser.add_argument("--strategy-id", default=None)
    parser.add_argument("--selection-end", required=True)
    parser.add_argument("--validation-start", required=True)
    parser.add_argument("--validation-end", required=True)
    parser.add_argument("--frozen-snapshot", required=True)
    parser.add_argument("--mongo-uri", default=None)
    parser.add_argument("--database", default=None)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--allow-remote-mongo", action="store_true")
    return parser


def _verify_frozen_snapshot(
    frozen: dict[str, Any],
    strategy: dict[str, Any],
    selection_end: pd.Timestamp,
) -> None:
    expected = str(frozen.get("decision_snapshot_sha256") or "")
    canonical = dict(frozen)
    canonical.pop("decision_snapshot_sha256", None)
    actual = leadership._sha256_json(canonical)
    if not expected or expected != actual:
        raise RuntimeError("Frozen rotation selection snapshot hash mismatch.")

    checks = {
        "strategy_id": str(frozen.get("strategy_id") or "")
        == str(strategy.get("_id") or ""),
        "strategy_revision": int(frozen.get("strategy_revision") or 0)
        == int(strategy.get("revision") or 0),
        "strategy_configuration_hash": str(
            frozen.get("strategy_configuration_hash") or ""
        )
        == str(strategy.get("configuration_hash") or ""),
        "selection_snapshot_end": str(frozen.get("snapshot_end") or "")
        == selection_end.date().isoformat(),
        "full_backtest_not_used_for_selection": frozen.get(
            "full_strategy_backtest_used_for_selection"
        )
        is False,
        "rotation_opportunity_used_for_selection": frozen.get(
            "rotation_opportunity_used_for_selection"
        )
        is True,
        "intrinsic_timing_not_used_for_selection": frozen.get(
            "intrinsic_timing_used_for_selection"
        )
        is False,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(
            "Frozen selection is not valid for independent validation: "
            + ", ".join(failed)
        )


def _progress(label: str):
    def callback(position: int, total: int, device: str) -> None:
        if position == 1 or position % 5 == 0 or position == total:
            _log(f"{label}: LightGBM {position}/{total} models ready ({device.upper()}).")

    return callback


def _strict_training_windows(
    common_dates: pd.DatetimeIndex,
    validation_start: pd.Timestamp,
    config: Any,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex, int, int]:
    normalized = pd.DatetimeIndex(common_dates).tz_convert("UTC")
    target = pd.Timestamp(validation_start, tz="UTC") if validation_start.tzinfo is None else validation_start.tz_convert("UTC")
    matches = np.flatnonzero(normalized == target)
    if len(matches) != 1:
        raise RuntimeError(
            f"Validation start {validation_start.date().isoformat()} is not an aligned market session."
        )
    validation_index = int(matches[0])
    purge = max(
        int(config.rotation_purge_days),
        max(int(item) for item in config.rotation_target_horizons),
    )
    calibration_days = int(config.rotation_walk_forward_calibration_days)

    # Last model-fit label must mature strictly before the first validation session.
    final_fit_end = validation_index - purge
    calibration_start = final_fit_end - calibration_days
    train_end = calibration_start - purge
    if train_end <= 0 or calibration_start <= 0 or final_fit_end <= calibration_start:
        raise RuntimeError("Insufficient pre-validation history for strict train/calibration/purge split.")

    train_dates = normalized[:train_end]
    calibration_dates = normalized[calibration_start:final_fit_end]
    final_fit_dates = normalized[:final_fit_end]
    minimum_rows = int(config.rotation_minimum_training_rows)
    if len(train_dates) < minimum_rows:
        raise RuntimeError(
            f"Strict training window has {len(train_dates)} sessions; at least {minimum_rows} are required."
        )
    return train_dates, calibration_dates, final_fit_dates, purge, validation_index


def _decision_dates(
    common_dates: pd.DatetimeIndex,
    validation_index: int,
    validation_end: pd.Timestamp,
) -> pd.DatetimeIndex:
    normalized = pd.DatetimeIndex(common_dates).tz_convert("UTC")
    target_end = pd.Timestamp(validation_end, tz="UTC") if validation_end.tzinfo is None else validation_end.tz_convert("UTC")
    end_matches = np.flatnonzero(normalized == target_end)
    if len(end_matches) != 1:
        raise RuntimeError(
            f"Validation end {validation_end.date().isoformat()} is not an aligned market session."
        )
    end_index = int(end_matches[0])
    if validation_index < 1 or end_index <= validation_index:
        raise RuntimeError("Independent validation interval is too short.")
    # Include the session immediately before validation so the first decision executes
    # at the first validation-session open, exactly like the production walk-forward engine.
    return normalized[validation_index - 1 : end_index + 1]


def main() -> int:
    args = _parser().parse_args()
    common.load_project_environment(args.env_file)
    os.environ.setdefault("MCT_MODEL_THREADS_OVERRIDE", "1")

    selection_end = common._normalize_date(args.selection_end)
    validation_start = common._normalize_date(args.validation_start)
    validation_end = common._normalize_date(args.validation_end)
    if not (selection_end < validation_start <= validation_end):
        raise RuntimeError("Expected selection_end < validation_start <= validation_end.")

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

    output_dir = Path(
        args.output_dir
        or PROJECT_ROOT
        / "research_output"
        / f"asset_rotation_independent_validation_strategy_{args.strategy_sequence}_{validation_end.date().isoformat()}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frozen_path = Path(args.frozen_snapshot).resolve()
    if not frozen_path.exists():
        raise RuntimeError(f"Frozen rotation selection snapshot not found: {frozen_path}")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))

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
    _verify_frozen_snapshot(frozen, strategy, selection_end)

    qualified_assets = [
        str(item).strip().upper()
        for item in frozen.get("qualified_assets") or []
        if str(item).strip()
    ]
    qualified_assets = list(dict.fromkeys(qualified_assets))
    if len(qualified_assets) < 2:
        raise RuntimeError("Frozen rotation-qualified universe has fewer than two assets.")

    configuration = common._configuration(strategy)
    base_config = leadership.BacktestRequest.model_validate(configuration)
    baseline_assets = list(
        dict.fromkeys(
            str(item).strip().upper()
            for item in base_config.assets
            if str(item).strip()
        )
    )
    start_date = common._normalize_date(base_config.start_date)
    identity = common._market_identity(configuration)
    collection = db[common.ALPACA_MARKET_BARS_COLLECTION]

    _log(
        f"Loading immutable market history for frozen universe={len(qualified_assets)}: "
        f"{start_date.date().isoformat()} -> {validation_end.date().isoformat()}."
    )
    raw_frames = timing._load_frames_allow_incomplete(
        collection,
        qualified_assets,
        identity,
        start_date,
        validation_end,
    )
    expected = common._expected_sessions(start_date, validation_end)
    history_rows, complete_assets = timing._history_diagnostics(
        raw_frames,
        qualified_assets,
        set(qualified_assets),
        expected,
    )
    missing = [asset for asset in qualified_assets if asset not in set(complete_assets)]
    if missing:
        raise RuntimeError(
            "Frozen asset lost Full Strategy History before independent validation: "
            + ", ".join(missing)
        )
    _write_csv(output_dir / "independent_validation_history_integrity.csv", pd.DataFrame(history_rows))

    family, settings, settings_hash = _immutable_model_snapshot(strategy)
    config = leadership._execution_config(
        configuration,
        qualified_assets,
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
    client.close()
    _log(
        "MongoDB closed. Independent validation now runs entirely from the in-memory snapshot. "
        f"model={family}, settings_hash={settings_hash}."
    )

    if portfolio_allocation_enabled(config) or compound_risk_overlay_enabled(config):
        raise RuntimeError(
            "This independent validator currently targets the single-position rotation policy used by Strategy #10; "
            "allocation/risk-overlay modes require their own frozen pre-validation calibrators."
        )
    if selective_opportunity_enabled(config) or opportunity_cash_gate_enabled(config):
        raise RuntimeError(
            "This independent validator currently targets the Strategy #10 utility rotation policy; "
            "selective opportunity-gate modes require a separate pre-validation gate freeze."
        )

    _log("Building Strategy rotation features once for the frozen universe...")
    frames, common_dates = prepare_rotation_panel(raw_frames, config)
    symbols = sorted(frames)
    if set(symbols) != set(qualified_assets):
        absent = sorted(set(qualified_assets) - set(symbols))
        raise RuntimeError("Rotation panel omitted frozen assets: " + ", ".join(absent))

    train_dates, calibration_dates, final_fit_dates, purge, validation_index = _strict_training_windows(
        common_dates,
        validation_start,
        config,
    )
    validation_decision_dates = _decision_dates(
        common_dates,
        validation_index,
        validation_end,
    )

    _log(
        "STRICT CHRONOLOGICAL SPLIT: "
        f"training <= {pd.Timestamp(train_dates[-1]).date()}, "
        f"calibration={pd.Timestamp(calibration_dates[0]).date()} -> {pd.Timestamp(calibration_dates[-1]).date()}, "
        f"final model labels <= {pd.Timestamp(final_fit_dates[-1]).date()}, "
        f"purge={purge} sessions, validation={validation_start.date()} -> {validation_end.date()}."
    )
    _log(
        "The validation-period returns/utilities are NOT used for asset selection, training, or switch-margin calibration."
    )

    calibration_models = _lightgbm_fit_models(
        frames,
        symbols,
        train_dates,
        config,
        phase="independent_validation_calibration",
        progress_callback=_progress("Calibration training"),
    )
    if set(calibration_models) != set(symbols):
        missing_models = sorted(set(symbols) - set(calibration_models))
        raise RuntimeError(
            "Frozen universe is not fully trainable before validation (calibration models): "
            + ", ".join(missing_models)
        )

    candidate_margins = tuple(float(value) for value in config.rotation_switch_margin_candidates)
    if not candidate_margins:
        raise RuntimeError("rotation_switch_margin_candidates is empty.")
    best_candidate = candidate_margins[0]
    best_score = float("-inf")
    margin_config = (
        config.model_copy(update={"strategy_mode": "COMPOUND_ROTATION_SWING_XGBOOST"})
        if absolute_utility_cash_gate_enabled(config)
        else config
    )
    for candidate in candidate_margins:
        policy = _utility_policy(
            calibration_models,
            frames,
            symbols,
            margin_config,
            candidate,
        )
        score = _simple_policy_growth(
            policy,
            frames,
            symbols,
            calibration_dates,
            config,
        )
        _log(f"Pre-validation margin calibration: candidate={candidate:.6f}, score={score:.6f}.")
        if float(score) > best_score:
            best_score = float(score)
            best_candidate = float(candidate)

    final_models = _lightgbm_fit_models(
        frames,
        symbols,
        final_fit_dates,
        config,
        phase="independent_validation_final_fit",
        progress_callback=_progress("Final pre-validation training"),
    )
    if set(final_models) != set(symbols):
        missing_models = sorted(set(symbols) - set(final_models))
        raise RuntimeError(
            "Frozen universe is not fully trainable before validation (final models): "
            + ", ".join(missing_models)
        )

    effective_margin = max(float(config.rotation_switch_margin), float(best_candidate))
    diagnostics: dict[pd.Timestamp, dict[str, Any]] = {}
    final_policy = _utility_policy(
        final_models,
        frames,
        symbols,
        config,
        effective_margin,
        cash_gate_base_state=(
            {"position": 0, "holding_days": 0, "pending_sample": None}
            if absolute_utility_cash_gate_enabled(config)
            else None
        ),
        decision_diagnostics=diagnostics,
        fold_id=1,
        calibrated_switch_margin=float(best_candidate),
    )

    decision_metadata = {
        pd.Timestamp(ts): {
            "fold_id": 1,
            "test_start": validation_start.date().isoformat(),
            "test_end": validation_end.date().isoformat(),
        }
        for ts in validation_decision_dates[:-1]
    }
    _log(
        f"Starting untouched final validation: {len(validation_decision_dates) - 1} sessions, "
        f"frozen assets={len(symbols)}, initial capital=${float(config.initial_capital):,.2f}."
    )
    result = _simulate_exact(
        "lightgbm_utility_independent_validation",
        final_policy,
        frames,
        symbols,
        validation_decision_dates,
        config,
        calculate_reference_fees,
        apply_slippage,
        decision_metadata=decision_metadata,
        policy_decision_diagnostics=diagnostics,
        model_label="LightGBM Utility · independent final period",
        method_line=(
            "- Asset universe, models, and switch margin were frozen before this validation period; "
            "the benchmark is equal-weight Buy & Hold over the same assets and execution dates."
        ),
    )

    predictions = result.predictions.reset_index()
    trades = result.trades.copy()
    _write_csv(output_dir / "independent_validation_predictions.csv", predictions)
    _write_csv(output_dir / "independent_validation_trades.csv", trades)

    metrics = dict(result.metrics)
    metrics.update(
        {
            "script_version": SCRIPT_VERSION,
            "experiment": "frozen_rotation_universe_independent_final_period",
            "selection_end": selection_end.date().isoformat(),
            "validation_start": validation_start.date().isoformat(),
            "validation_end": validation_end.date().isoformat(),
            "validation_sessions": int(len(validation_decision_dates) - 1),
            "selection_snapshot_sha256": frozen["decision_snapshot_sha256"],
            "qualified_assets_sha256": frozen.get("qualified_assets_sha256"),
            "selection_used_validation_period": False,
            "training_used_validation_period": False,
            "calibration_used_validation_period": False,
            "purge_sessions_before_validation": int(purge),
            "training_last_session": pd.Timestamp(train_dates[-1]).date().isoformat(),
            "calibration_first_session": pd.Timestamp(calibration_dates[0]).date().isoformat(),
            "calibration_last_session": pd.Timestamp(calibration_dates[-1]).date().isoformat(),
            "final_fit_last_labeled_session": pd.Timestamp(final_fit_dates[-1]).date().isoformat(),
            "calibrated_switch_margin": float(best_candidate),
            "effective_switch_margin": float(effective_margin),
            "calibration_score": float(best_score),
            "primary_benchmark": "Equal-weight Buy & Hold across the same frozen assets",
            "success_definition": "intelligent_rotation_final_capital > equal_weight_buy_hold_final_capital",
        }
    )
    passed = float(metrics["strategy_ending_capital"]) > float(metrics["buy_hold_ending_capital"])
    metrics["independent_validation_passed"] = bool(passed)
    _write_json(output_dir / "independent_validation_result.json", metrics)
    _write_json(
        output_dir / "independent_validation_manifest.json",
        {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "strategy_id": str(strategy.get("_id") or ""),
            "strategy_sequence": int(strategy.get("strategy_sequence") or args.strategy_sequence),
            "strategy_revision": int(strategy.get("revision") or 0),
            "strategy_configuration_hash": strategy.get("configuration_hash"),
            "model_family": family,
            "model_settings_hash": settings_hash,
            "frozen_asset_count": len(symbols),
            "frozen_assets": symbols,
            "selection_end": selection_end.date().isoformat(),
            "validation_start": validation_start.date().isoformat(),
            "validation_end": validation_end.date().isoformat(),
            "validation_sessions": int(len(validation_decision_dates) - 1),
            "selection_snapshot_sha256": frozen["decision_snapshot_sha256"],
            "no_validation_data_used_for_selection": True,
            "no_validation_data_used_for_training": True,
            "no_validation_data_used_for_calibration": True,
            "benchmark": "equal_weight_buy_and_hold_same_frozen_universe",
            "independent_validation_passed": bool(passed),
        },
    )

    rotation_end = float(metrics["strategy_ending_capital"])
    buy_hold_end = float(metrics["buy_hold_ending_capital"])
    _log("FINAL INDEPENDENT VALIDATION")
    _log(
        f"Intelligent rotation: ${float(config.initial_capital):,.2f} -> ${rotation_end:,.2f} "
        f"({float(metrics['strategy_return']):.2%})."
    )
    _log(
        f"Equal-weight Buy & Hold: ${float(config.initial_capital):,.2f} -> ${buy_hold_end:,.2f} "
        f"({float(metrics['buy_hold_return']):.2%})."
    )
    _log(
        f"Excess return={float(metrics['excess_return']):.2%}; "
        f"rotation Sharpe={float(metrics['strategy_sharpe']):.3f}; "
        f"Buy & Hold Sharpe={float(metrics['buy_hold_sharpe']):.3f}; "
        f"rotation MaxDD={float(metrics['strategy_maximum_drawdown']):.2%}; "
        f"Buy & Hold MaxDD={float(metrics['buy_hold_maximum_drawdown']):.2%}."
    )
    _log(
        "RESULT: PASS - intelligent rotation beat Buy & Hold on the untouched final period."
        if passed
        else "RESULT: FAIL - intelligent rotation did not beat Buy & Hold on the untouched final period."
    )
    _log(f"Artifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
