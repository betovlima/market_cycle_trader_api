from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


def windows_long_path(path: str | os.PathLike[str] | Path) -> str:
    resolved = str(Path(path).resolve())
    if os.name != "nt" or resolved.startswith("\\\\?\\"):
        return resolved
    if resolved.startswith("\\\\"):
        return "\\\\?\\UNC\\" + resolved[2:]
    return "\\\\?\\" + resolved


def ensure_dir(path: str | os.PathLike[str] | Path) -> None:
    os.makedirs(windows_long_path(path), exist_ok=True)


def exists(path: str | os.PathLike[str] | Path) -> bool:
    return os.path.exists(windows_long_path(path))


def read_text(path: str | os.PathLike[str] | Path) -> str:
    with open(windows_long_path(path), "r", encoding="utf-8") as handle:
        return handle.read()


def read_json(path: str | os.PathLike[str] | Path) -> Any:
    return json.loads(read_text(path))


def read_csv(path: str | os.PathLike[str] | Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(windows_long_path(path), **kwargs)


def _temporary_path(path: Path) -> Path:
    return path.parent / ".mct_write.tmp"


def write_text(path: str | os.PathLike[str] | Path, content: str) -> None:
    target = Path(path).resolve()
    ensure_dir(target.parent)
    temporary = _temporary_path(target)
    with open(windows_long_path(temporary), "w", encoding="utf-8", newline="") as handle:
        handle.write(content)
    os.replace(windows_long_path(temporary), windows_long_path(target))


def write_json(path: str | os.PathLike[str] | Path, value: Any) -> None:
    write_text(
        path,
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
    )


def write_csv(
    path: str | os.PathLike[str] | Path,
    frame: pd.DataFrame,
) -> None:
    target = Path(path).resolve()
    ensure_dir(target.parent)
    temporary = _temporary_path(target)
    frame.to_csv(windows_long_path(temporary), index=False)
    os.replace(windows_long_path(temporary), windows_long_path(target))
