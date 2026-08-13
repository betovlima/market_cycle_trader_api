from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from market_cycle_trader_api.schemas.requests import BacktestExecutionRequest, BacktestRequest
from market_cycle_trader_api.services.reproducibility import (
    build_reproducibility_manifest,
    market_data_research_signature_from_manifests,
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
    assert first["reproducibility_schema_version"] == 3
    assert first["api_version"] == "2.0.16"
    assert first["runtime_fingerprint_sha256"]
    assert first["engine_source_sha256"]
    assert first["package_source_sha256"]
    assert "xgboost_build_info" in first
    assert "numeric_thread_environment" in first
    assert "threadpool_runtime" in first




def test_research_signature_ignores_load_path_audit_metadata() -> None:
    first_bars = _bars([100.0, 101.0, 102.0])
    second_bars = _bars([100.0, 101.0, 102.0])
    first_frame = first_bars["AAPL"]
    second_frame = second_bars["AAPL"]
    first_frame.attrs["market_data_provenance"] = {
        "historical_feed": "sip",
        "adjustment": "all",
        "initial_rows": 2666,
        "history_backfill_rows": 0,
        "history_backfill_provider": None,
    }
    second_frame.attrs["market_data_provenance"] = {
        "historical_feed": "sip",
        "adjustment": "all",
        "initial_rows": 3000,
        "history_backfill_rows": 334,
        "history_backfill_provider": "alpaca",
    }

    first = build_reproducibility_manifest(_config(), first_bars)
    second = build_reproducibility_manifest(_config(), second_bars)

    assert first["market_data_signature_sha256"] == second["market_data_signature_sha256"]
    assert first["market_data_audit_signature_sha256"] != second["market_data_audit_signature_sha256"]


def test_research_signature_can_be_reconstructed_from_retained_symbol_manifests() -> None:
    manifest = build_reproducibility_manifest(_config(), _bars([100.0, 101.0, 102.0]))
    reconstructed = market_data_research_signature_from_manifests(manifest["market_data_signatures"])
    assert reconstructed == manifest["market_data_signature_sha256"]


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


def test_execution_signature_guard_accepts_only_sha256_hex() -> None:
    base = _config()
    common = {
        **base.model_dump(mode="python"),
        "analysis_start_date": base.start_date,
        "analysis_end_date": base.end_date,
        "calendar_anchor_assets": list(base.assets),
    }
    digest = "b" * 64
    execution = BacktestExecutionRequest.model_validate({
        **common,
        "expected_market_data_signature_sha256": digest,
    })
    assert execution.expected_market_data_signature_sha256 == digest


def test_engine_rejects_signature_mismatch_before_model_training() -> None:
    from unittest.mock import patch
    from market_cycle_trader_api.engine.compound_rotation_backtest import run_job

    base = _config()
    assets = ["AAPL", "MSFT"]
    execution = BacktestExecutionRequest.model_validate({
        **base.model_dump(mode="python"),
        "assets": assets,
        "analysis_start_date": base.start_date,
        "analysis_end_date": base.end_date,
        "calendar_anchor_assets": assets,
        "research_reference_assets": assets,
        "research_candidate_assets": [],
        "expected_market_data_signature_sha256": "0" * 64,
    })
    aapl = next(iter(_bars([100.0, 101.0, 102.0]).values()))
    msft = next(iter(_bars([200.0, 201.0, 202.0]).values()))
    for frame in (aapl, msft):
        frame.attrs["market_data_provenance"] = {
            "historical_feed": "sip",
            "adjustment": "all",
            "history_complete": True,
            "research_access_path": "mongodb_only",
        }

    with (
        patch(
            "market_cycle_trader_api.engine.compound_rotation_backtest.load_market_bars",
            side_effect=[aapl, msft],
        ),
        patch(
            "market_cycle_trader_api.engine.compound_rotation_backtest.validate_and_clean_bars",
            side_effect=lambda frame, config: frame,
        ),
        patch(
            "market_cycle_trader_api.engine.compound_rotation_backtest.run_rotation_models"
        ) as trainer,
    ):
        try:
            run_job("job-1", execution, object())
            raised = False
        except RuntimeError as exc:
            raised = True
            assert "MarketDataSignatureMismatch" in str(exc)

    assert raised
    trainer.assert_not_called()


def test_research_signature_ignores_optional_non_model_bar_fields() -> None:
    first_bars = _bars([100.0, 101.0, 102.0])
    second_bars = _bars([100.0, 101.0, 102.0])
    first_bars["AAPL"]["vwap"] = [100.1, 101.1, 102.1]
    first_bars["AAPL"]["trade_count"] = [100.0, 110.0, 120.0]
    second_bars["AAPL"]["vwap"] = [999.0, 998.0, 997.0]
    second_bars["AAPL"]["trade_count"] = [1.0, 2.0, 3.0]

    first = build_reproducibility_manifest(_config(), first_bars)
    second = build_reproducibility_manifest(_config(), second_bars)

    assert first["market_data_signature_sha256"] == second["market_data_signature_sha256"]
    assert first["market_data_audit_signature_sha256"] != second["market_data_audit_signature_sha256"]
    assert first["market_data_signature_schema_version"] == 4


def test_market_data_signature_is_dtype_and_datetime_unit_canonical() -> None:
    from market_cycle_trader_api.services.reproducibility import market_data_manifest

    base_index = pd.date_range("2016-01-04", periods=4, freq="B", tz="UTC")
    source_index = base_index.as_unit("us")
    restored_index = base_index.as_unit("ns")

    source = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            "volume": [1000, 1100, 1200, 1300],
        },
        index=source_index,
    )
    restored = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            "volume": [1000.0, 1100.0, 1200.0, 1300.0],
        },
        index=restored_index,
    )
    provenance = {
        "history_complete": True,
        "historical_feed": "sip",
        "adjustment": "all",
    }
    source.attrs["market_data_provenance"] = dict(provenance)
    restored.attrs["market_data_provenance"] = dict(provenance)

    source_signature, source_manifests = market_data_manifest({"AAPL": source})
    restored_signature, restored_manifests = market_data_manifest({"AAPL": restored})

    assert source_manifests["AAPL"]["sha256"] == restored_manifests["AAPL"]["sha256"]
    assert source_signature == restored_signature
