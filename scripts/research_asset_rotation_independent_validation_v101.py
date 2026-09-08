from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

import research_asset_rotation_independent_validation as base

SCRIPT_VERSION = "asset-rotation-independent-validation-v1.0.1"


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


def main() -> int:
    base.SCRIPT_VERSION = SCRIPT_VERSION
    base._write_json = _write_json
    base._write_csv = _write_csv
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())
