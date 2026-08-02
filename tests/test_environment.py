from __future__ import annotations

import os
from pathlib import Path

from market_cycle_trader_api.core.environment import (
    ENV_FILE_VARIABLE,
    build_subprocess_environment,
    load_project_environment,
)


def test_empty_process_values_are_filled_from_env_file(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "ALPACA_API_KEY_ID=local-key\n"
        "ALPACA_SECRET_KEY=local-secret\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ALPACA_API_KEY_ID", "")
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.delenv(ENV_FILE_VARIABLE, raising=False)

    loaded = load_project_environment(env_file)

    assert loaded == (env_file.resolve(),)
    assert os.environ["ALPACA_API_KEY_ID"] == "local-key"
    assert os.environ["ALPACA_SECRET_KEY"] == "local-secret"


def test_non_empty_system_values_keep_priority(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("ALPACA_API_KEY_ID=file-key\n", encoding="utf-8")
    monkeypatch.setenv("ALPACA_API_KEY_ID", "system-key")
    monkeypatch.delenv(ENV_FILE_VARIABLE, raising=False)

    load_project_environment(env_file)

    assert os.environ["ALPACA_API_KEY_ID"] == "system-key"


def test_child_environment_contains_refreshed_values(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("ALPACA_API_KEY_ID=child-key\n", encoding="utf-8")
    monkeypatch.setenv(ENV_FILE_VARIABLE, str(env_file))
    monkeypatch.setenv("ALPACA_API_KEY_ID", "")

    child = build_subprocess_environment({"PYTHONPATH": "src"})

    assert child["ALPACA_API_KEY_ID"] == "child-key"
    assert child["PYTHONPATH"] == "src"
    assert child["PYTHONIOENCODING"] == "utf-8"
