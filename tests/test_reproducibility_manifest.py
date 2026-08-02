from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from market_cycle_trader_api.schemas.requests import BacktestRequest
from market_cycle_trader_api.services.reproducibility import (
    build_reproducibility_manifest,
)


def _config() -> BacktestRequest:
    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "market_cycle_trader_api"
        / "parameterizations"
        / "001_xgboost_multihorizon_champion_cpu.json"
    )
    return BacktestRequest.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _bars(close_values: list[float]) -> dict[str, pd.DataFrame]:
    index = pd.date_range("2026-01-02", periods=len(close_values), freq="B", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": close_values,
            "high": [value + 1 for value in close_values],
            "low": [value - 1 for value in close_values],
            "close": close_values,
            "volume": [1000.0] * len(close_values),
        },
        index=index,
    )
    return {"AAPL": frame}


def test_manifest_is_stable_for_same_configuration_and_bars() -> None:
    first = build_reproducibility_manifest(_config(), _bars([100.0, 101.0, 102.0]))
    second = build_reproducibility_manifest(_config(), _bars([100.0, 101.0, 102.0]))

    assert first["strategy_configuration_sha256"] == second["strategy_configuration_sha256"]
    assert first["market_data_signature_sha256"] == second["market_data_signature_sha256"]
    assert first["deterministic_execution"] is False
    assert first["numeric_thread_limit"] == 1
    assert first["xgb_n_jobs"] == -1


def test_market_data_signature_changes_when_price_changes() -> None:
    first = build_reproducibility_manifest(_config(), _bars([100.0, 101.0, 102.0]))
    second = build_reproducibility_manifest(_config(), _bars([100.0, 101.0, 103.0]))

    assert first["market_data_signature_sha256"] != second["market_data_signature_sha256"]
