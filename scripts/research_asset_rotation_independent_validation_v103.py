from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import research_asset_rotation_independent_validation as base  # noqa: E402
import research_asset_rotation_independent_validation_v101 as v101  # noqa: E402

SCRIPT_VERSION = "asset-rotation-independent-validation-v1.0.3"


def _bootstrap_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--history-start", required=True)
    args, _ = parser.parse_known_args()
    return args


def _utc_index(values: Any) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(values)
    if index.tz is None:
        return index.tz_localize("UTC")
    return index.tz_convert("UTC")


def _utc_day(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.normalize()


def _aligned_session_position(
    common_dates: pd.DatetimeIndex,
    requested: pd.Timestamp,
    *,
    label: str,
) -> tuple[pd.DatetimeIndex, int]:
    aligned = _utc_index(common_dates)
    aligned_days = aligned.normalize()
    target_day = _utc_day(requested)
    matches = np.flatnonzero(aligned_days == target_day)
    if len(matches) == 1:
        return aligned, int(matches[0])

    before = aligned_days[aligned_days < target_day]
    after = aligned_days[aligned_days > target_day]
    previous_day = before[-1].date().isoformat() if len(before) else "none"
    next_day = after[0].date().isoformat() if len(after) else "none"
    if len(matches) > 1:
        raise RuntimeError(
            f"{label} {target_day.date().isoformat()} maps to {len(matches)} aligned rows; "
            "expected exactly one daily market session."
        )
    raise RuntimeError(
        f"{label} {target_day.date().isoformat()} is not present in the aligned daily market panel. "
        f"Nearest aligned session dates: previous={previous_day}, next={next_day}."
    )


def _strict_training_windows(
    common_dates: pd.DatetimeIndex,
    validation_start: pd.Timestamp,
    config: Any,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex, int, int]:
    normalized, validation_index = _aligned_session_position(
        common_dates,
        validation_start,
        label="Validation start",
    )
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
    normalized, end_index = _aligned_session_position(
        common_dates,
        validation_end,
        label="Validation end",
    )
    if validation_index < 1 or end_index <= validation_index:
        raise RuntimeError("Independent validation interval is too short.")

    validation_sessions = normalized[validation_index : end_index + 1]
    expected_start = normalized[validation_index].normalize().tz_localize(None)
    expected_end = _utc_day(validation_end).tz_localize(None)
    expected = _utc_index(
        base.common._expected_sessions(expected_start, expected_end)
    ).normalize()
    actual = validation_sessions.normalize()
    if not actual.equals(expected):
        missing = expected.difference(actual)
        extra = actual.difference(expected)
        raise RuntimeError(
            "Independent validation aligned-session coverage mismatch: "
            f"expected={len(expected)}, actual={len(actual)}, "
            f"missing={','.join(item.date().isoformat() for item in missing[:5]) or 'none'}, "
            f"extra={','.join(item.date().isoformat() for item in extra[:5]) or 'none'}."
        )

    # Include the session immediately before validation so the first decision executes
    # at the first validation-session open, exactly like the production walk-forward engine.
    return normalized[validation_index - 1 : end_index + 1]


def _install_history_start_contract(history_start: str) -> None:
    requested_start = base.common._normalize_date(history_start)
    original_parser = base._parser
    original_verify = base._verify_frozen_snapshot

    def parser() -> argparse.ArgumentParser:
        result = original_parser()
        result.add_argument("--history-start", required=True)
        return result

    def verify_frozen_snapshot(
        frozen: dict[str, Any],
        strategy: dict[str, Any],
        selection_end: pd.Timestamp,
    ) -> None:
        original_verify(frozen, strategy, selection_end)
        frozen_start = base.common._normalize_date(frozen.get("strategy_start"))
        if frozen_start != requested_start:
            raise RuntimeError(
                "Frozen selection history start does not match --history-start: "
                f"frozen={frozen_start.date().isoformat()}, "
                f"requested={requested_start.date().isoformat()}."
            )

        configuration = base.common._configuration(strategy)
        strategy_start = base.common._normalize_date(configuration.get("start_date"))
        if strategy_start != requested_start:
            raise RuntimeError(
                "Strategy research start does not match --history-start: "
                f"strategy={strategy_start.date().isoformat()}, "
                f"requested={requested_start.date().isoformat()}. "
                "Independent validation refuses to silently truncate or extend the Strategy history."
            )

    base._parser = parser
    base._verify_frozen_snapshot = verify_frozen_snapshot


def main() -> int:
    args = _bootstrap_args()
    _install_history_start_contract(args.history_start)
    base._strict_training_windows = _strict_training_windows
    base._decision_dates = _decision_dates

    # Keep all v1.0.2 protections: Windows long-path-safe writes and frozen-universe anchors.
    v101.SCRIPT_VERSION = SCRIPT_VERSION
    base.SCRIPT_VERSION = SCRIPT_VERSION
    return int(v101.main())


if __name__ == "__main__":
    raise SystemExit(main())
