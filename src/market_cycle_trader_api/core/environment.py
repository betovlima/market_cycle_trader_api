from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

API_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_project_environment() -> tuple[Path, ...]:
    """Load local .env files without overriding real system variables.

    Railway injects variables directly into the process environment, so those
    values always win. During local development, both the current working
    directory and the API project directory are checked so the application can
    be started either from the repository root or from market_cycle_trader_api.
    """

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
