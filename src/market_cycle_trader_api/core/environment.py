from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

API_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_project_environment() -> tuple[Path, ...]:








    candidates = (
        Path.cwd() / ".env",
        API_PROJECT_ROOT / ".env",
    )
    loaded: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if not resolved.is_file():
            continue
        load_dotenv(dotenv_path=resolved, override=False)
        loaded.append(resolved)
    return tuple(loaded)
