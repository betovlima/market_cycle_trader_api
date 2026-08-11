from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from market_cycle_trader_api.schemas.requests import BacktestExecutionRequest, BacktestRequest
from market_cycle_trader_api.services.reproducibility import (
    build_reproducibility_manifest,
    strategy_configuration_fingerprint,
)


def _config() -> BacktestRequest:
    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "market_cycle_trader_api"
        / "parameterizations"
        / "winner-v1.13.2.json"
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
    assert first["reproducibility_schema_version"] == 2
    assert first["api_version"] == "1.13.44"
    assert first["runtime_fingerprint_sha256"]
    assert first["engine_source_sha256"]
    assert first["package_source_sha256"]
    assert "xgboost_build_info" in first
    assert "numeric_thread_environment" in first
    assert "threadpool_runtime" in first


def test_market_data_signature_changes_when_price_changes() -> None:
    first = build_reproducibility_manifest(_config(), _bars([100.0, 101.0, 102.0]))
    second = build_reproducibility_manifest(_config(), _bars([100.0, 101.0, 103.0]))

    assert first["market_data_signature_sha256"] != second["market_data_signature_sha256"]


def test_research_reference_is_independent_from_calendar_anchors_and_configuration_hash() -> None:
    base = _config()
    assets = list(base.assets[:4])
    execution = BacktestExecutionRequest.model_validate(
        {
            **base.model_dump(mode="python"),
            "assets": assets,
            "analysis_start_date": base.start_date,
            "analysis_end_date": base.end_date,
            "calendar_anchor_assets": assets[:2],
            "research_reference_assets": assets[:3],
            "research_candidate_assets": assets[3:],
        }
    )
    base_same_strategy = BacktestRequest.model_validate(
        {**base.model_dump(mode="python"), "assets": assets}
    )

    manifest = build_reproducibility_manifest(
        execution,
        {assets[0]: next(iter(_bars([100.0, 101.0, 102.0]).values()))},
    )

    assert manifest["research_reference_assets"] == assets[:3]
    assert manifest["research_candidate_assets"] == assets[3:]
    assert strategy_configuration_fingerprint(execution) == strategy_configuration_fingerprint(base_same_strategy)


def test_execution_model_is_not_part_of_strategy_fingerprint() -> None:
    base = _config()
    common = {
        **base.model_dump(mode="python"),
        "analysis_start_date": base.start_date,
        "analysis_end_date": base.end_date,
        "calendar_anchor_assets": list(base.assets),
    }
    lightgbm = BacktestExecutionRequest.model_validate({
        **common,
        "research_model_family": "lightgbm_utility",
        "research_model_settings": {},
    })
    iqn = BacktestExecutionRequest.model_validate({
        **common,
        "research_model_family": "iqn",
        "research_model_settings": {"iqn": {"hidden_dim": 128}},
    })

    baseline_hash = strategy_configuration_fingerprint(base)
    assert strategy_configuration_fingerprint(lightgbm) == baseline_hash
    assert strategy_configuration_fingerprint(iqn) == baseline_hash
