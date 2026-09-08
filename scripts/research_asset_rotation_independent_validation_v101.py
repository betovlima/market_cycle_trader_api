from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

import research_asset_rotation_independent_validation as base
import research_asset_rotation_leadership as leadership

SCRIPT_VERSION = "asset-rotation-independent-validation-v1.0.2"


def _windows_long_path(path: Path) -> str:
    resolved = str(Path(path).resolve())
    if os.name != "nt" or resolved.startswith("\\\\?\\"):
        return resolved
    if resolved.startswith("\\\\"):
        return "\\\\?\\UNC\\" + resolved[2:]
    return "\\\\?\\" + resolved


def _ensure_parent(path: Path) -> None:
    os.makedirs(_windows_long_path(path.parent), exist_ok=True)


def _temporary_path(path: Path) -> Path:
    # Keep the temporary filename deliberately short. The research output tree is
    # already deep on Windows and appending '.tmp' to the full descriptive name
    # can cross the legacy MAX_PATH boundary.
    return path.parent / ".mct_write.tmp"


def _write_json(path: Path, value: Any) -> None:
    path = Path(path).resolve()
    _ensure_parent(path)
    temporary = _temporary_path(path)
    with open(_windows_long_path(temporary), "w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    os.replace(_windows_long_path(temporary), _windows_long_path(path))


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path = Path(path).resolve()
    _ensure_parent(path)
    temporary = _temporary_path(path)
    frame.to_csv(_windows_long_path(temporary), index=False)
    os.replace(_windows_long_path(temporary), _windows_long_path(path))


def _install_frozen_universe_execution_config() -> None:
    original_execution_config = leadership._execution_config

    def frozen_universe_execution_config(
        configuration: dict[str, Any],
        research_assets: list[str],
        baseline_assets: list[str],
        snapshot_end: pd.Timestamp,
        family: str,
        settings: dict[str, Any],
    ):
        # The independent final test runs only the already-frozen selected universe.
        # BacktestExecutionRequest requires calendar/reference assets to be members
        # of that universe, so preserve only Strategy reference assets that actually
        # survived the pre-validation selection. This changes metadata/alignment
        # only; it does not add or remove any frozen asset and does not touch the
        # validation-period data, model, policy, costs, or benchmark.
        frozen_set = {
            str(symbol).strip().upper()
            for symbol in research_assets
            if str(symbol).strip()
        }
        frozen_reference_assets = [
            str(symbol).strip().upper()
            for symbol in baseline_assets
            if str(symbol).strip().upper() in frozen_set
        ]
        frozen_reference_assets = list(dict.fromkeys(frozen_reference_assets))
        if len(frozen_reference_assets) < 2:
            raise RuntimeError(
                "Independent validation requires at least two frozen Strategy reference assets "
                "to anchor the execution calendar."
            )
        return original_execution_config(
            configuration,
            research_assets,
            frozen_reference_assets,
            snapshot_end,
            family,
            settings,
        )

    leadership._execution_config = frozen_universe_execution_config


def main() -> int:
    base.SCRIPT_VERSION = SCRIPT_VERSION
    base._write_json = _write_json
    base._write_csv = _write_csv
    _install_frozen_universe_execution_config()
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())
