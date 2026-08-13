from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values

ENV_FILE_VARIABLE = "MARKET_CYCLE_TRADER_ENV_FILE"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"


def _candidate_environment_files(
    explicit_path: str | os.PathLike[str] | None = None,
) -> list[Path]:
    candidates: list[Path] = []

    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())

    configured_path = str(os.getenv(ENV_FILE_VARIABLE) or "").strip()
    if configured_path:
        candidates.append(Path(configured_path).expanduser())

    candidates.append(Path.cwd() / ".env")
    candidates.append(DEFAULT_ENV_FILE)

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        key = os.path.normcase(str(resolved))
        if key in seen:
            continue
        seen.add(key)
        unique.append(resolved)
    return unique


def load_project_environment(
    explicit_path: str | os.PathLike[str] | None = None,
) -> tuple[Path, ...]:
    






    loaded: list[Path] = []
    for candidate in _candidate_environment_files(explicit_path):
        if not candidate.is_file():
            continue

        values = dotenv_values(candidate)
        for key, raw_value in values.items():
            if not key or raw_value is None:
                continue
            current = os.getenv(key)
            if current is None or not str(current).strip():
                os.environ[key] = str(raw_value).strip()

        os.environ[ENV_FILE_VARIABLE] = str(candidate)
        loaded.append(candidate)
        break

    return tuple(loaded)


def build_subprocess_environment(
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    

    load_project_environment()
    child_environment = dict(os.environ)
    child_environment.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
        }
    )
    if extra:
        child_environment.update(
            {str(key): str(value) for key, value in extra.items()}
        )
    return child_environment
