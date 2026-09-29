"""Offline validation for a preexisting MCT Control Shadow snapshot.

No Alpaca fetch, no TCC CSVs, no database mutation, no trading code. The
science reference calibrator/replay is reused verbatim on the same SHA-audited
MCT data; generated CSV/JSON reports live *beside* the source snapshot.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..tcc_v106_reference.capital_rotation import (
    _risk_adjusted_reward,
    _simple_policy_growth,
    _training_transition_log_return,
    _utility_policy,
)
from ..tcc_v106_reference.config import (
    ASSETS, CONFIG, REFERENCE_ASSETS, build_control_config,
)
from ..tcc_v106_reference.execution import apply_slippage, calculate_reference_fees
from ..tcc_v106_reference.research_challengers import (
    _lightgbm_fit_models,
    run_research_challenger,
)
from .control_shadow_market_data import DATA_DIRECTORY, SOURCE_CONTRACT
from .operational_control_contract import prepare_operational_control_panel

Progress = Callable[[str, int, int], None]
_ID = re.compile(r"control-shadow-[a-f0-9]{16}\Z")
_METRICS = (
    "strategy_ending_capital", "strategy_return", "strategy_cagr",
    "strategy_sharpe", "strategy_maximum_drawdown", "capital_rotations",
    "simulated_buys", "simulated_sells", "market_exposure",
    "buy_hold_ending_capital", "buy_hold_return", "risk_adjusted_compound_score",
)


def read_verified_control_snapshot(
    source_job_id: str, *, snapshot_root: Path | None = None,
    expected_sha256: str | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], Path]:
    """Verify manifest, complete hashes and data before loading normalized CSVs."""
    if not _ID.fullmatch(source_job_id):
        raise ValueError("Expected server-generated Control Shadow job ID.")
    root = (snapshot_root if snapshot_root is not None else DATA_DIRECTORY).resolve()
    source = root / source_job_id
    if not source.is_dir() or source.is_symlink() or source.resolve() != root / source_job_id:
        raise FileNotFoundError("MCT Control Shadow snapshot not found in configured data directory.")
    manifest_file = source / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    required = {
        "schema_version": 1, "source_contract": SOURCE_CONTRACT,
        "source": "alpaca", "feed": "sip", "download_adjustment": "raw",
        "effective_adjustment": "raw_plus_split_normalization",
        "dividend_adjustment_applied": False,
    }
    if any(manifest.get(k) != v for k, v in required.items()):
        raise ValueError("Snapshot provenance does not match the current MCT Control data contract.")
    cutoff = str(manifest.get("completed_session") or "")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", cutoff):
        raise ValueError("Invalid snapshot completed session.")
    stamp = pd.Timestamp(cutoff)
    if stamp.strftime("%Y-%m-%d") != cutoff:
        raise ValueError("Invalid snapshot cutoff.")
    canonical = {k: v for k, v in manifest.items() if k != "snapshot_sha256"}
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    if manifest.get("snapshot_sha256") != digest or (
        expected_sha256 and digest != expected_sha256
    ):
        raise ValueError("Control snapshot SHA-256 manifest mismatch.")
    if manifest.get("requested_assets") != list(ASSETS):
        raise ValueError("Control universe differs from recorded reference assets.")
    eligible = manifest.get("eligible_assets")
    if not isinstance(eligible, list) or len(eligible) < 2 or len(eligible) != len(set(eligible)):
        raise ValueError("Invalid eligible asset list.")
    if sorted(eligible) != eligible or not set(eligible).issubset(ASSETS):
        raise ValueError("Unexpected Control assets in the snapshot.")
    if not set(REFERENCE_ASSETS).issubset(eligible):
        raise ValueError("Missing Control reference calendar assets.")
    records = manifest.get("per_asset")
    hashes = manifest.get("file_hashes")
    if not isinstance(records, dict) or not isinstance(hashes, dict):
        raise ValueError("Incomplete snapshot audit.")
    expected_files = set()
    for symbol in ASSETS:
        expected_files.update({f"raw_bars/{symbol}.csv", f"corporate_actions/{symbol}.json"})
        record = records.get(symbol)
        if not isinstance(record, dict):
            raise ValueError(f"{symbol}: missing per-asset provenance.")
        if symbol in eligible:
            if record.get("status") != "eligible":
                raise ValueError(f"{symbol}: manifest eligibility mismatch.")
            expected_files.add(f"normalized_bars/{symbol}.csv")
        elif record.get("status") != "excluded_structural_identity":
            raise ValueError(f"{symbol}: missing structural exclusion reason.")
    if set(hashes) != expected_files:
        raise ValueError("Snapshot file set mismatch.")
    for relative, expected in hashes.items():
        path = source / relative
        if path.is_symlink() or not path.is_file() or path.resolve().parent.parent != source:
            raise ValueError(f"Missing or unsafe snapshot file: {relative}.")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Snapshot file hash changed: {relative}.")
    frames = {}
    for symbol in eligible:
        frame = pd.read_csv(source / f"normalized_bars/{symbol}.csv", index_col="timestamp")
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True))
        frame.index.name = "timestamp"
        if frame.empty or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
            raise ValueError(f"{symbol}: invalid normalized session index.")
        if frame.index[-1].date() > stamp.date():
            raise ValueError(f"{symbol}: future bars in normalized snapshot.")
        record = records[symbol]
        if len(frame) != int(record["normalized_rows"]):
            raise ValueError(f"{symbol}: normalized row count changed.")
        if frame.index[0].date().isoformat() != record["first_session"] or (
            frame.index[-1].date().isoformat() != record["last_session"]
        ):
            raise ValueError(f"{symbol}: normalized date range changed.")
        frames[symbol] = frame
    return frames, manifest, source


def _calibration_curve(
    policy: Callable[[pd.Timestamp, int, int], tuple[int, float]],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    dates: pd.DatetimeIndex,
    config: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Recreate scientific _simple_policy_growth with observable path."""
    capital = float(config.initial_capital)
    wealth, peak, position, holding, score = 1.0, 1.0, 0, 0, 0.0
    rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for i in range(len(dates) - 1):
        now, nxt = dates[i], dates[i + 1]
        action, predicted_utility = policy(now, position, holding)
        log_return = _training_transition_log_return(
            frames, symbols, now, nxt, position, action, config,
        )
        reward, wealth, peak = _risk_adjusted_reward(
            log_return, wealth, peak, config,
        )
        score += reward
        previous = position
        if action == position:
            holding = holding + 1 if action > 0 else 0
        else:
            position = action
            holding = 1 if action > 0 else 0
        rows.append({
            "date": nxt.date().isoformat(),
            "equity": capital * wealth,
            "drawdown": wealth / peak - 1.0,
            "log_return": log_return,
            "risk_adjusted_reward": reward,
            "cumulative_calibration_score": score,
        })
        decisions.append({
            "decision_date": now.date().isoformat(),
            "execution_date": nxt.date().isoformat(),
            "from_asset": symbols[previous - 1] if previous else "CASH",
            "target_asset": symbols[action - 1] if action else "CASH",
            "predicted_utility": float(predicted_utility),
            "holding_sessions": holding,
            "log_return": log_return,
        })
    return pd.DataFrame(rows), pd.DataFrame(decisions), score


def _json_float(v: Any) -> float | None:
    if v is None:
        return None
    n = float(v)
    return n if math.isfinite(n) else None


def run_control_snapshot_validation(
    *,
    source_job_id: str,
    expected_sha256: str,
    original_calibration_score: float | None,
    original_candidate_margin: float | None,
    progress: Progress | None = None,
    snapshot_root: Path | None = None,
) -> dict[str, Any]:
    """Calibrate four margins and replay official OOS on immutable source snapshot."""
    def emit(stage: str, completed: int, total: int = 1) -> None:
        if progress is not None:
            progress(stage, completed, total)

    emit("verify_snapshot", 0)
    bars, manifest, directory = read_verified_control_snapshot(
        source_job_id, snapshot_root=snapshot_root, expected_sha256=expected_sha256,
    )
    completed_session = str(manifest["completed_session"])
    # Scientific analysis_end_date is a UTC instant, whereas MCT cutoff is
    # an inclusive XNYS session DATE. Use the v10.8.38 adapter convention.
    inclusive_utc = (
        pd.Timestamp(completed_session, tz="UTC")
        + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    ).isoformat()
    config = build_control_config(CONFIG).model_copy(update={
        "analysis_end_date": inclusive_utc,
        "end_date": completed_session,
    })
    frames, dates, audit = prepare_operational_control_panel(
        bars, completed_session=completed_session, config=config,
    )
    symbols = sorted(frames)
    purge = max(int(config.rotation_purge_days), max(config.rotation_target_horizons))
    calibration_days = int(config.rotation_walk_forward_calibration_days)
    execution_index = len(dates)
    calibration_end = execution_index - purge
    calibration_start = calibration_end - calibration_days
    train_end = calibration_start - purge
    if train_end < int(config.rotation_minimum_training_rows):
        raise ValueError("Insufficient available history for Control shadow calibration.")
    train_dates = dates[:train_end]
    calibration_dates = dates[calibration_start:calibration_end]
    emit("calibration_fit", 0, len(symbols))
    model = _lightgbm_fit_models(
        frames, symbols, train_dates, config,
        phase="control_snapshot_validation_calibration",
        progress_callback=lambda n, total, _device: emit("calibration_fit", n, total),
    )
    if not model:
        raise ValueError("No calibrated Control models for snapshot validation.")
    calibration_summary: list[dict[str, Any]] = []
    calibration_paths: list[pd.DataFrame] = []
    calibration_decisions: list[pd.DataFrame] = []
    candidate_margins = tuple(float(x) for x in config.rotation_switch_margin_candidates)
    for i, candidate in enumerate(candidate_margins, 1):
        policy = _utility_policy(model, frames, symbols, config, candidate)
        score = _simple_policy_growth(policy, frames, symbols, calibration_dates, config)
        curve, decisions, reconstructed_score = _calibration_curve(
            policy, frames, symbols, calibration_dates, config,
        )
        if not math.isfinite(score) or not math.isclose(
            score, reconstructed_score, rel_tol=0, abs_tol=1e-9,
        ):
            raise ValueError("Calibration curve differs from the scientific score.")
        effective = max(float(config.rotation_switch_margin), candidate)
        calibration_summary.append({
            "candidate_margin": candidate,
            "effective_margin": effective,
            "risk_adjusted_score": score,
            "ending_equity": float(curve["equity"].iloc[-1]),
            "maximum_drawdown": float(curve["drawdown"].min()),
            "transitions": len(curve),
            "rotations": int((decisions["from_asset"] != decisions["target_asset"]).sum()),
        })
        curve.insert(0, "candidate_margin", candidate)
        decisions.insert(0, "candidate_margin", candidate)
        calibration_paths.append(curve)
        calibration_decisions.append(decisions)
        emit("margin_diagnostics", i, len(candidate_margins))
    chosen = max(calibration_summary, key=lambda x: x["risk_adjusted_score"])
    score_difference = (
        None if original_calibration_score is None
        else chosen["risk_adjusted_score"] - float(original_calibration_score)
    )
    original_match = (
        original_candidate_margin is not None
        and float(original_candidate_margin) == chosen["candidate_margin"]
        and score_difference is not None and abs(score_difference) <= 1e-8
    )

    emit("oos_replay", 0, 100)
    def oos_progress(percent: float, stage: str, _completed: int) -> None:
        emit("oos_replay:" + str(stage)[:100], min(100, max(0, int(percent))), 100)
    results = run_research_challenger(
        "lightgbm_utility", bars, config,
        calculate_reference_fees, apply_slippage,
        progress_callback=oos_progress,
    )
    if len(results) != 1 or results[0].predictions.empty:
        raise ValueError("Scientific OOS replay returned no complete Control result.")
    replay = results[0]
    if replay.predictions.index[-1].date().isoformat() != completed_session:
        raise ValueError("OOS replay did not reach the source snapshot cutoff.")
    metrics = {key: _json_float(replay.metrics.get(key)) for key in _METRICS}
    metrics["walk_forward_fold_count"] = int(replay.metrics["walk_forward_fold_count"])
    metrics["walk_forward_folds"] = [
        {
            key: (_json_float(value) if isinstance(value, (float, np.floating)) else
                  value.isoformat() if isinstance(value, pd.Timestamp) else value)
            for key, value in row.items()
            if key in {
                "fold_id", "test_start", "test_end", "train_end",
                "calibration_start", "calibration_end", "calibrated_candidate_margin",
                "effective_switch_margin", "calibration_risk_adjusted_score",
                "ending_capital", "return", "max_drawdown", "sharpe",
            }
        }
        for row in replay.metrics["walk_forward_folds"]
    ]
    oos_curve = replay.predictions[[
        "strategy_equity", "buy_hold_equity", "selected_asset", "decision_date",
        "walk_forward_fold", "trade_action",
    ]].copy()
    oos_curve["drawdown"] = oos_curve["strategy_equity"] / oos_curve["strategy_equity"].cummax() - 1
    report = {
        "schema_version": 1,
        "source_job_id": source_job_id,
        "source_snapshot_sha256": manifest["snapshot_sha256"],
        "source_kind": "fresh_alpaca_raw_sip_local_mct_snapshot",
        "completed_session": completed_session,
        "source_unchanged": True,
        "snapshot_assets": len(bars),
        "calendar_sessions": audit.calendar_sessions,
        "calibration_window": {
            "train_end": train_dates[-1].date().isoformat(),
            "calibration_start": calibration_dates[0].date().isoformat(),
            "calibration_end": calibration_dates[-1].date().isoformat(),
            "purge_sessions": purge,
            "initial_equity": float(config.initial_capital),
            "scoring_basis": "sum of risk_adjusted_reward; NOT percentage return",
        },
        "margin_candidates": calibration_summary,
        "calibration_selected": chosen,
        "original_shadow": {
            "calibration_score": original_calibration_score,
            "calibrated_candidate_margin": original_candidate_margin,
            "score_difference": score_difference,
            "reproduced": bool(original_match),
        },
        "oos": metrics,
        "caveats": [
            "Calibration uses the same holdout window to choose a margin; it is not independent OOS.",
            "OOS scientific replay uses expanding walk-forward folds and next-open execution, not live orders.",
            "Historical capital on refreshed Alpaca data is not the frozen TCC v1.0.6 result.",
            "A separate Control shadow prospective decision may use a later final fit than the last historical OOS fold.",
            "No order eligibility, Winner promotion or trading integration.",
        ],
        "order_eligible": False, "order_submission": "never",
    }

    # Write reports only after all provenance checks and scientific simulations pass.
    output = directory / "validation" / "v10.8.40"
    output.mkdir(parents=True, exist_ok=True)
    pd.concat(calibration_paths, ignore_index=True).to_csv(
        output / "calibration_curves.csv", index=False, float_format="%.17g",
    )
    pd.concat(calibration_decisions, ignore_index=True).to_csv(
        output / "calibration_decisions.csv", index=False, float_format="%.17g",
    )
    oos_curve.to_csv(output / "oos_capital_curve.csv", float_format="%.17g")
    replay.predictions.to_csv(output / "oos_decisions.csv", float_format="%.17g")
    replay.trades.to_csv(output / "oos_trades.csv", index=False, float_format="%.17g")
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report["report_directory"] = str(output)
    report["artifacts"] = sorted(x.name for x in output.iterdir())
    emit("completed", 1)
    return report
