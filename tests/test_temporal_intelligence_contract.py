from __future__ import annotations

import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np
import pytest
import pandas as pd

from market_cycle_trader_api.auth.capabilities import capabilities_for_role
from market_cycle_trader_api.engine.temporal_intelligence import (
    _BinaryModelBundle,
    _PlattCalibrator,
    _binary_quality_weight,
    _apply_online_matured_quality,
    _classification_metrics,
    _future_target_matrices,
    _shadow_capital_study,
    _decision_components,
    _multi_horizon_components,
    _multi_horizon_frame,
    _multi_horizon_observation_rows,
    _compressed_artifact_documents,
    _externalize_result_diagnostics,
    _winner_anchored_temporal_study,
)
from market_cycle_trader_api.services.temporal_intelligence import build_temporal_intelligence_export

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "market_cycle_trader_api"
FRONT = ROOT.parent / "market_cycle_trader" / "src"


def _bars(open_values: list[float], close_values: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2026-01-05", periods=len(open_values), freq="B", tz="UTC")
    return pd.DataFrame({"open": open_values, "close": close_values}, index=dates)


def test_temporal_targets_start_at_next_open_and_use_relative_equal_weight_alpha() -> None:
    a = _bars([100, 101, 102, 103, 104, 105], [100, 102, 104, 106, 108, 110])
    b = _bars([100, 100, 100, 100, 100, 100], [100, 100, 100, 100, 100, 100])
    dates = a.index
    targets = _future_target_matrices({"AAA": a, "BBB": b}, dates, ["AAA", "BBB"], [2])[2]

    expected_a_return = math.log(104 / 101)
    expected_b_return = 0.0
    expected_benchmark = (expected_a_return + expected_b_return) / 2.0
    assert targets["return"].at[dates[0], "AAA"] == pytest.approx(expected_a_return)
    assert targets["benchmark"].at[dates[0], "benchmark_return"] == pytest.approx(expected_benchmark)
    assert targets["alpha"].at[dates[0], "AAA"] == pytest.approx(expected_a_return - expected_benchmark)
    assert targets["alpha"].at[dates[0], "BBB"] == pytest.approx(-expected_benchmark)
    assert targets["drawdown"].at[dates[0], "AAA"] == pytest.approx(0.0)


def test_temporal_probability_metrics_are_calibration_aware() -> None:
    realized_alpha = np.asarray([0.10, -0.05, 0.03, -0.02], dtype=float)
    probability = np.asarray([0.90, 0.10, 0.80, 0.20], dtype=float)
    metrics = _classification_metrics(realized_alpha, probability)
    assert 0.0 <= metrics["brier"] <= 1.0
    assert metrics["auc"] == pytest.approx(1.0)
    assert metrics["calibration_error"] is not None


def test_temporal_intelligence_is_shadow_only_and_capability_driven() -> None:
    engine = (SRC / "engine" / "temporal_intelligence.py").read_text(encoding="utf-8")
    service = (SRC / "services" / "temporal_intelligence.py").read_text(encoding="utf-8")
    router = (SRC / "api" / "routers" / "temporal_intelligence.py").read_text(encoding="utf-8")
    panel = (FRONT / "features" / "TemporalIntelligencePanel.jsx").read_text(encoding="utf-8")
    backtest = (FRONT / "features" / "backtest" / "components" / "BacktestPage.jsx").read_text(encoding="utf-8")

    assert '"shadow_only": True' in engine
    assert '"affects_strategy_decisions": False' in engine
    assert '"affects_winner": False' in engine
    assert '"affects_paper_trading": False' in engine
    assert 'get_trader_winner_context(db)' in service
    assert 'get_trader_winner_model_snapshot(db)' in service
    assert '"experiment": "temporal_decision_intelligence_v8_winner_anchored_timing"' in service
    assert 'temporal_intelligence_decision_diagnostics.csv' in service
    assert 'temporal_intelligence_multi_horizon_daily_assets.csv' in service
    assert 'require_capability("temporal_intelligence.view")' in router
    assert 'require_capability("temporal_intelligence.start")' in router
    assert 'require_capability("temporal_intelligence.stop")' in router
    assert 'require_capability("temporal_intelligence.export")' in router
    assert 'build_temporal_intelligence_export' in router
    assert '/export.zip' in router
    assert 'downloadFile' in panel
    assert "require_admin_session" not in router
    assert "TemporalIntelligencePanel" in backtest
    assert "temporal_intelligence.view" in backtest
    assert "/temporal-intelligence" in panel

    viewer = capabilities_for_role("viewer")
    trader = capabilities_for_role("trader")
    admin = capabilities_for_role("admin")
    assert viewer["temporal_intelligence.view"] is True
    assert viewer["temporal_intelligence.start"] is False
    assert trader["temporal_intelligence.view"] is True
    assert trader["temporal_intelligence.start"] is False
    assert admin["temporal_intelligence.start"] is True
    assert admin["temporal_intelligence.stop"] is True
    assert viewer["temporal_intelligence.export"] is False
    assert trader["temporal_intelligence.export"] is False
    assert admin["temporal_intelligence.export"] is True


class _TemporalCollection:
    def __init__(self, document):
        self.document = document

    def find_one(self, query, projection=None):
        del projection
        return self.document if query.get("id") == self.document.get("id") else None


class _EmptyObservationCursor(list):
    def sort(self, *args, **kwargs):
        del args, kwargs
        return self


class _EmptyObservationCollection:
    def __init__(self, items=None):
        self.items = list(items or [])

    def find(self, query=None, *args, **kwargs):
        del args, kwargs
        query = query or {}
        items = [
            item for item in self.items
            if all(item.get(key) == value for key, value in query.items() if key in item)
        ]
        return _EmptyObservationCursor(items)


class _TemporalDb:
    def __init__(self, document, observations=None, artifacts=None):
        self.collection = _TemporalCollection(document)
        self.observations = _EmptyObservationCollection(observations)
        self.artifacts = _EmptyObservationCollection(artifacts)

    def __getitem__(self, name):
        if name == "temporal_intelligence_runs":
            return self.collection
        if name == "temporal_intelligence_observations":
            return self.observations
        if name == "temporal_intelligence_artifacts":
            return self.artifacts
        raise AssertionError(name)


def test_temporal_intelligence_export_contains_evaluation_files() -> None:
    document = {
        "id": "temporal-test",
        "status": "completed",
        "strategy_profile_name": "Test Strategy",
        "horizons": [5],
        "request": {"rotation_target_horizons": [5]},
        "result": {
            "horizons": [5],
            "asset_count": 2,
            "feature_count": 3,
            "walk_forward_fold_count": 1,
            "horizon_metrics": [{
                "horizon": 5,
                "samples": 20,
                "brier": 0.2,
                "confidence_bins": [{"from_probability": 0.7, "to_probability": 0.8, "samples": 5}],
                "risk_buckets": [{"bucket": "low", "samples": 7}],
            }],
            "fold_metrics": [{"fold_id": 1, "horizon": 5, "samples": 20}],
            "latest_forecasts": [{"symbol": "AAA", "horizon": 5, "expected_alpha": 0.02}],
            "shadow_only": True,
        },
    }
    content = build_temporal_intelligence_export(_TemporalDb(document), "temporal-test")
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = set(archive.namelist())
        assert "temporal_intelligence_summary.csv" in names
        assert "temporal_intelligence_horizons.csv" in names
        assert "temporal_intelligence_folds.csv" in names
        assert "temporal_intelligence_confidence_bins.csv" in names
        assert "temporal_intelligence_risk_buckets.csv" in names
        assert "temporal_intelligence_latest_forecasts.csv" in names
        assert "temporal_intelligence_manifest.json" in names
        assert "AAA" in archive.read("temporal_intelligence_latest_forecasts.csv").decode("utf-8")


def test_temporal_decision_targets_identify_profit_first_bottom_and_top() -> None:
    dates = pd.date_range("2026-01-05", periods=6, freq="B", tz="UTC")
    bottom_bars = pd.DataFrame({
        "open": [100, 100, 101, 102, 103, 104],
        "high": [100, 103, 104, 105, 106, 107],
        "low": [100, 99.5, 100, 101, 102, 103],
        "close": [100, 102, 103, 104, 105, 106],
    }, index=dates)
    top_bars = pd.DataFrame({
        "open": [100, 100, 99, 98, 97, 96],
        "high": [100, 100.5, 100.4, 99, 98, 97],
        "low": [100, 98, 97, 96, 95, 94],
        "close": [100, 99, 98, 97, 96, 95],
    }, index=dates)
    targets = _future_target_matrices({"AAA": bottom_bars, "BBB": top_bars}, dates, ["AAA", "BBB"], [2])[2]
    assert targets["profit_before_loss"].at[dates[0], "AAA"] == pytest.approx(1.0)
    assert targets["bottom"].at[dates[0], "AAA"] == pytest.approx(1.0)
    assert targets["top"].at[dates[0], "AAA"] == pytest.approx(0.0)
    assert targets["profit_before_loss"].at[dates[0], "BBB"] == pytest.approx(0.0)
    assert targets["top"].at[dates[0], "BBB"] == pytest.approx(1.0)


def test_temporal_decision_export_contains_signal_capital_and_diagnostic_files() -> None:
    document = {
        "id": "temporal-decision-test",
        "status": "completed",
        "experiment": "temporal_decision_intelligence_v2",
        "strategy_profile_name": "Test Strategy",
        "horizons": [20],
        "request": {"rotation_target_horizons": [20]},
        "result": {
            "experiment": "temporal_decision_intelligence_v2",
            "horizons": [20],
            "asset_count": 2,
            "feature_count": 52,
            "walk_forward_fold_count": 1,
            "horizon_metrics": [{
                "horizon": 20,
                "samples": 30,
                "signal_metrics": [{"signal": "profit_before_loss", "auc": 0.61}],
                "risk_buckets": [{"bucket": "low", "samples": 10}],
                "shadow_capital": {
                    "ending_capital": 120000.0,
                    "cagr": 0.20,
                    "action_counts": {"buy": 4, "hold": 10},
                    "decision_diagnostics": [{"timestamp": "2026-01-05", "best_symbol": "AAA", "action": "BUY", "entry_score": 0.01}],
                },
            }],
            "fold_metrics": [{
                "fold_id": 1,
                "horizon": 20,
                "samples": 30,
                "shadow_capital": {"ending_capital": 120000.0, "action_counts": {"buy": 4}},
            }],
            "latest_forecasts": [{"symbol": "AAA", "horizon": 20, "decision_score": 0.12, "shadow_target": True}],
            "shadow_only": True,
        },
    }
    content = build_temporal_intelligence_export(_TemporalDb(document), "temporal-decision-test")
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = set(archive.namelist())
        assert "temporal_intelligence_signal_metrics.csv" in names
        assert "temporal_intelligence_shadow_capital.csv" in names
        assert "temporal_intelligence_shadow_capital_folds.csv" in names
        assert "temporal_intelligence_decision_diagnostics.csv" in names
        assert "profit_before_loss" in archive.read("temporal_intelligence_signal_metrics.csv").decode("utf-8")
        assert "120000" in archive.read("temporal_intelligence_shadow_capital.csv").decode("utf-8")
        assert "AAA" in archive.read("temporal_intelligence_decision_diagnostics.csv").decode("utf-8")


class _CapitalConfig:
    initial_capital = 10000.0
    slippage_bps = 0.0
    commission_rate = 0.0


def test_temporal_v5_externalizes_large_decision_diagnostics_before_run_persistence() -> None:
    diagnostic = {
        "timestamp": "2026-01-05",
        "best_symbol": "AAA",
        "action": "HOLD",
        **{f"metric_{index}": float(index) / 100.0 for index in range(60)},
    }
    result = {
        "horizon_metrics": [
            {
                "horizon": horizon,
                "shadow_capital": {
                    "ending_capital": 10000.0 + horizon,
                    "decision_diagnostics": [dict(diagnostic, timestamp=f"2026-01-{day:02d}") for day in range(1, 29)],
                },
            }
            for horizon in (5, 10, 20, 40, 60)
        ],
        "multi_horizon_metrics": {
            "shadow_capital": {
                "ending_capital": 20000.0,
                "decision_diagnostics": [dict(diagnostic, timestamp=f"2026-02-{day:02d}") for day in range(1, 29)],
            }
        },
    }

    rows, counts = _externalize_result_diagnostics(result)

    assert counts["horizon_decision_diagnostics"] == 140
    assert counts["multi_horizon_decision_diagnostics"] == 28
    assert len(rows) == 168
    assert all("decision_diagnostics" not in item["shadow_capital"] for item in result["horizon_metrics"])
    assert "decision_diagnostics" not in result["multi_horizon_metrics"]["shadow_capital"]

    documents = _compressed_artifact_documents("run-1", "decision_diagnostics", rows, chunk_size=25)
    assert documents
    assert max(document["row_count"] for document in documents) <= 25
    assert all(document["encoding"] == "zlib-json-v1" for document in documents)
    assert all(len(document["payload"]) < 16 * 1024 * 1024 for document in documents)


def test_temporal_export_rehydrates_externalized_decision_diagnostics() -> None:
    horizon_row = {
        "artifact_kind": "horizon_decision_diagnostics",
        "horizon": 5,
        "timestamp": "2026-01-05",
        "best_symbol": "AAA",
        "action": "BUY",
    }
    multi_row = {
        "artifact_kind": "multi_horizon_decision_diagnostics",
        "timestamp": "2026-01-05",
        "best_symbol": "BBB",
        "action": "ROTATE",
    }
    artifacts = _compressed_artifact_documents(
        "temporal-external-test",
        "decision_diagnostics",
        [horizon_row, multi_row],
        chunk_size=10,
    )
    document = {
        "id": "temporal-external-test",
        "status": "completed",
        "experiment": "temporal_decision_intelligence_v5_trend_capture_hysteresis",
        "horizons": [5],
        "request": {"rotation_target_horizons": [5]},
        "result": {
            "experiment": "temporal_decision_intelligence_v5_trend_capture_hysteresis",
            "horizons": [5],
            "horizon_metrics": [{"horizon": 5, "shadow_capital": {"ending_capital": 12000.0}}],
            "fold_metrics": [],
            "latest_forecasts": [],
            "multi_horizon_metrics": {"shadow_capital": {"ending_capital": 13000.0}},
            "multi_horizon_fold_metrics": [],
            "multi_horizon_latest_forecasts": [],
            "shadow_only": True,
        },
    }

    content = build_temporal_intelligence_export(
        _TemporalDb(document, artifacts=artifacts),
        "temporal-external-test",
    )
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        horizon_csv = archive.read("temporal_intelligence_decision_diagnostics.csv").decode("utf-8")
        multi_csv = archive.read("temporal_intelligence_multi_horizon_decision_diagnostics.csv").decode("utf-8")
        assert "AAA" in horizon_csv
        assert "BBB" in multi_csv


def test_temporal_shadow_capital_study_trades_top_positive_decision_score() -> None:
    dates = pd.date_range("2026-01-05", periods=5, freq="B", tz="UTC")
    open_prices = pd.DataFrame({"AAA": [100.0, 100.0, 102.0, 104.0, 106.0]}, index=dates)
    predictions = pd.DataFrame({
        "timestamp": [dates[0], dates[1], dates[2]],
        "symbol": ["AAA", "AAA", "AAA"],
        "fold_id": [1, 1, 1],
        "entry_score": [0.020, 0.018, 0.015],
        "hold_score": [0.018, 0.016, 0.014],
        "entry_threshold": [0.0, 0.0, 0.0],
        "exit_threshold": [-0.002, -0.002, -0.002],
        "rotation_hurdle": [0.002, 0.002, 0.002],
        "cash_score": [0.0, 0.0, 0.0],
        "profit_before_loss_probability": [0.80, 0.78, 0.75],
    })
    metrics = _shadow_capital_study(predictions, open_prices, dates, _CapitalConfig(), include_diagnostics=True)
    assert metrics["ending_capital"] > metrics["initial_capital"]
    assert metrics["exposure"] > 0.0
    assert metrics["action_counts"]["buy"] == 1
    assert metrics["action_counts"]["hold"] >= 1
    assert metrics["decision_diagnostics"][0]["action"] == "BUY"


def test_temporal_decision_quality_gate_disables_unskilled_optional_signal() -> None:
    unskilled = _BinaryModelBundle(
        model=object(),
        calibrator=_PlattCalibrator(None),
        baseline_probability=0.5,
        validation_auc=0.49,
        validation_brier_skill=-0.01,
        validation_samples=500,
    )
    skilled = _BinaryModelBundle(
        model=object(),
        calibrator=_PlattCalibrator(None),
        baseline_probability=0.5,
        validation_auc=0.62,
        validation_brier_skill=0.05,
        validation_samples=500,
    )
    assert _binary_quality_weight(unskilled) == 0.0
    assert _binary_quality_weight(skilled) == pytest.approx(1.0)
    assert _binary_quality_weight(unskilled, primary=True) == pytest.approx(0.35)


def test_temporal_decision_policy_separates_entry_hold_and_exit() -> None:
    dates = pd.date_range("2026-01-05", periods=6, freq="B", tz="UTC")
    open_prices = pd.DataFrame({"AAA": [100, 100, 101, 102, 103, 104], "BBB": [100, 100, 100, 100, 100, 100]}, index=dates)
    predictions = pd.DataFrame({
        "timestamp": [dates[0], dates[0], dates[1], dates[1], dates[2], dates[2]],
        "symbol": ["AAA", "BBB"] * 3,
        "fold_id": [1] * 6,
        "entry_score": [0.02, 0.01, 0.005, 0.004, -0.01, 0.015],
        "hold_score": [0.018, 0.008, 0.003, 0.003, -0.01, 0.012],
        "entry_threshold": [0.0] * 6,
        "exit_threshold": [-0.002] * 6,
        "rotation_hurdle": [0.003] * 6,
        "cash_score": [0.0] * 6,
        "profit_before_loss_probability": [0.7, 0.6, 0.6, 0.58, 0.4, 0.7],
    })
    metrics = _shadow_capital_study(predictions, open_prices, dates, _CapitalConfig(), include_diagnostics=True)
    actions = [row["action"] for row in metrics["decision_diagnostics"]]
    assert actions[0] == "BUY"
    assert actions[1] == "HOLD"
    assert actions[2] == "ROTATE"


def test_temporal_v3_ranker_uses_relative_profit_and_relative_risk() -> None:
    timestamp = pd.Timestamp("2026-01-05", tz="UTC")
    frame = pd.DataFrame({
        "timestamp": [timestamp] * 4,
        "profit_before_loss_probability": [0.32, 0.24, 0.20, 0.18],
        "baseline_profit_before_loss_probability": [0.40] * 4,
        "profit_before_loss_quality_weight": [0.9] * 4,
        "predicted_drawdown": [0.06, 0.10, 0.08, 0.12],
        "drawdown_quality_weight": [0.9] * 4,
        "bottom_probability": [0.5] * 4,
        "baseline_bottom_probability": [0.5] * 4,
        "bottom_quality_weight": [0.0] * 4,
        "top_probability": [0.5] * 4,
        "baseline_top_probability": [0.5] * 4,
        "top_quality_weight": [0.0] * 4,
        "trend_direction": [1.0] * 4,
        "trend_persistence_probability": [0.5] * 4,
        "trend_persistence_quality_weight": [0.0] * 4,
    })
    components = _decision_components(frame, profit_barrier=0.08, loss_barrier=0.05, one_side_cost=0.0)
    assert components.loc[0, "profit_percentile"] == pytest.approx(1.0)
    assert components.loc[0, "risk_safety_percentile"] == pytest.approx(1.0)
    assert components.loc[0, "asset_rank_score"] == pytest.approx(components["asset_rank_score"].max())
    assert components.loc[0, "opportunity_gate_score"] > components.loc[0, "entry_threshold"]


def test_temporal_v3_cash_gate_does_not_require_absolute_barrier_breakeven() -> None:
    timestamp = pd.Timestamp("2026-01-05", tz="UTC")
    frame = pd.DataFrame({
        "timestamp": [timestamp] * 4,
        "profit_before_loss_probability": [0.32, 0.20, 0.18, 0.15],
        "baseline_profit_before_loss_probability": [0.42] * 4,
        "profit_before_loss_quality_weight": [1.0] * 4,
        "predicted_drawdown": [0.04, 0.09, 0.10, 0.12],
        "drawdown_quality_weight": [1.0] * 4,
        "bottom_probability": [0.5] * 4, "baseline_bottom_probability": [0.5] * 4, "bottom_quality_weight": [0.0] * 4,
        "top_probability": [0.5] * 4, "baseline_top_probability": [0.5] * 4, "top_quality_weight": [0.0] * 4,
        "trend_direction": [1.0] * 4, "trend_persistence_probability": [0.5] * 4, "trend_persistence_quality_weight": [0.0] * 4,
    })
    components = _decision_components(frame, profit_barrier=0.08, loss_barrier=0.05, one_side_cost=0.0)
    assert frame.loc[0, "profit_before_loss_probability"] < components.loc[0, "breakeven_probability"]
    assert components.loc[0, "opportunity_gate_score"] > components.loc[0, "entry_threshold"]


def test_temporal_v3_export_contains_winner_reference_files() -> None:
    document = {
        "id": "temporal-v3-test", "status": "completed", "experiment": "temporal_decision_intelligence_v3",
        "strategy_profile_name": "Winner #3", "horizons": [20], "request": {"rotation_target_horizons": [20]},
        "result": {
            "experiment": "temporal_decision_intelligence_v3", "horizons": [20], "asset_count": 2, "feature_count": 52, "walk_forward_fold_count": 1,
            "horizon_metrics": [{"horizon": 20, "samples": 30, "shadow_capital": {"ending_capital": 15000.0, "action_counts": {"buy": 2}}}],
            "fold_metrics": [], "latest_forecasts": [], "shadow_only": True,
            "winner_reference": {"ending_capital": 18000.0, "total_return": 0.8, "folds": [{"fold_id": 1, "strategy_return": 0.2}]},
        },
    }
    content = build_temporal_intelligence_export(_TemporalDb(document), "temporal-v3-test")
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = set(archive.namelist())
        assert "temporal_intelligence_winner_reference.csv" in names
        assert "temporal_intelligence_winner_reference_folds.csv" in names
        assert "18000" in archive.read("temporal_intelligence_winner_reference.csv").decode("utf-8")


def test_temporal_v4_multi_horizon_uses_short_entry_and_long_hold_roles() -> None:
    timestamp = pd.Timestamp("2026-01-05", tz="UTC")
    rows = []
    for symbol, short_profit, long_profit, risk in (
        ("AAA", 0.95, 0.80, 0.90),
        ("BBB", 0.70, 0.65, 0.70),
        ("CCC", 0.40, 0.45, 0.50),
        ("DDD", 0.10, 0.20, 0.30),
    ):
        row = {"timestamp": timestamp, "symbol": symbol, "fold_id": 1}
        for horizon in (5, 10, 20, 40, 60):
            percentile = short_profit if horizon in (5, 10) else long_profit
            row[f"profit_percentile_h{horizon}"] = percentile
            row[f"profit_before_loss_probability_h{horizon}"] = 0.30 + 0.20 * percentile
            row[f"baseline_profit_before_loss_probability_h{horizon}"] = 0.40
            row[f"profit_before_loss_quality_weight_h{horizon}"] = 0.9
            row[f"predicted_drawdown_h{horizon}"] = 0.05 + 0.10 * (1.0 - risk)
            row[f"risk_safety_percentile_h{horizon}"] = risk
            row[f"drawdown_quality_weight_h{horizon}"] = 0.9
            row[f"bottom_probability_h{horizon}"] = 0.55
            row[f"baseline_bottom_probability_h{horizon}"] = 0.50
            row[f"bottom_quality_weight_h{horizon}"] = 0.2 if horizon == 5 else 0.0
            row[f"top_probability_h{horizon}"] = 0.45
            row[f"baseline_top_probability_h{horizon}"] = 0.50
            row[f"top_quality_weight_h{horizon}"] = 0.0
            row[f"trend_direction_h{horizon}"] = 1.0
            row[f"trend_persistence_probability_h{horizon}"] = 0.65
            row[f"baseline_trend_persistence_probability_h{horizon}"] = 0.50
            row[f"trend_persistence_quality_weight_h{horizon}"] = 0.2 if horizon in (20, 40, 60) else 0.0
        rows.append(row)
    components = _multi_horizon_components(pd.DataFrame(rows), [5, 10, 20, 40, 60], one_side_cost=0.0)
    leader = components.loc[components["symbol"] == "AAA"].iloc[0]
    assert leader["asset_rank_score"] == pytest.approx(components["asset_rank_score"].max())
    assert leader["short_horizon_agreement"] == pytest.approx(1.0)
    assert leader["horizon_agreement"] > 0.8
    assert leader["entry_score"] > leader["entry_threshold"]
    assert leader["hold_score"] > leader["exit_threshold"]


def test_temporal_v4_multi_horizon_penalizes_isolated_long_horizon_leader() -> None:
    timestamp = pd.Timestamp("2026-01-05", tz="UTC")
    rows = []
    definitions = {
        "GOOD": {5: 0.95, 10: 0.90, 20: 0.75, 40: 0.70, 60: 0.65},
        "LONG_ONLY": {5: 0.15, 10: 0.20, 20: 1.0, 40: 1.0, 60: 1.0},
        "MID": {5: 0.55, 10: 0.50, 20: 0.50, 40: 0.50, 60: 0.50},
    }
    for symbol, values in definitions.items():
        row = {"timestamp": timestamp, "symbol": symbol, "fold_id": 1}
        for horizon, percentile in values.items():
            row[f"profit_percentile_h{horizon}"] = percentile
            row[f"profit_before_loss_probability_h{horizon}"] = 0.25 + 0.20 * percentile
            row[f"baseline_profit_before_loss_probability_h{horizon}"] = 0.40
            row[f"profit_before_loss_quality_weight_h{horizon}"] = 0.9
            row[f"predicted_drawdown_h{horizon}"] = 0.07
            row[f"risk_safety_percentile_h{horizon}"] = 0.8
            row[f"drawdown_quality_weight_h{horizon}"] = 0.9
            row[f"bottom_probability_h{horizon}"] = 0.5
            row[f"baseline_bottom_probability_h{horizon}"] = 0.5
            row[f"bottom_quality_weight_h{horizon}"] = 0.0
            row[f"top_probability_h{horizon}"] = 0.5
            row[f"baseline_top_probability_h{horizon}"] = 0.5
            row[f"top_quality_weight_h{horizon}"] = 0.0
            row[f"trend_direction_h{horizon}"] = 1.0
            row[f"trend_persistence_probability_h{horizon}"] = 0.5
            row[f"baseline_trend_persistence_probability_h{horizon}"] = 0.5
            row[f"trend_persistence_quality_weight_h{horizon}"] = 0.0
        rows.append(row)
    components = _multi_horizon_components(pd.DataFrame(rows), [5, 10, 20, 40, 60], one_side_cost=0.0)
    good = components.loc[components["symbol"] == "GOOD"].iloc[0]
    long_only = components.loc[components["symbol"] == "LONG_ONLY"].iloc[0]
    assert good["asset_rank_score"] > long_only["asset_rank_score"]
    assert good["short_profit_consensus"] > long_only["short_profit_consensus"]
    assert good["horizon_agreement"] > long_only["horizon_agreement"]




def test_temporal_v5_risk_adjusted_entry_reduces_high_risk_authority() -> None:
    timestamp = pd.Timestamp("2026-01-05", tz="UTC")
    rows = []
    for symbol, risk in (("RISKY", 0.02), ("SAFE", 0.85), ("MID", 0.50)):
        row = {"timestamp": timestamp, "symbol": symbol, "fold_id": 1}
        for horizon in (5, 10, 20, 40, 60):
            percentile = 1.0 if symbol == "RISKY" else 0.90 if symbol == "SAFE" else 0.50
            row[f"profit_percentile_h{horizon}"] = percentile
            row[f"profit_before_loss_probability_h{horizon}"] = 0.30 + 0.20 * percentile
            row[f"baseline_profit_before_loss_probability_h{horizon}"] = 0.40
            row[f"profit_before_loss_quality_weight_h{horizon}"] = 0.9
            row[f"predicted_drawdown_h{horizon}"] = 0.05 + 0.10 * (1.0 - risk)
            row[f"risk_safety_percentile_h{horizon}"] = risk
            row[f"drawdown_quality_weight_h{horizon}"] = 0.9
            row[f"bottom_probability_h{horizon}"] = 0.5
            row[f"baseline_bottom_probability_h{horizon}"] = 0.5
            row[f"bottom_quality_weight_h{horizon}"] = 0.0
            row[f"top_probability_h{horizon}"] = 0.5
            row[f"baseline_top_probability_h{horizon}"] = 0.5
            row[f"top_quality_weight_h{horizon}"] = 0.0
            row[f"trend_direction_h{horizon}"] = 1.0
            row[f"trend_persistence_probability_h{horizon}"] = 0.6
            row[f"baseline_trend_persistence_probability_h{horizon}"] = 0.5
            row[f"trend_persistence_quality_weight_h{horizon}"] = 0.2
        rows.append(row)
    components = _multi_horizon_components(pd.DataFrame(rows), [5, 10, 20, 40, 60], one_side_cost=0.0)
    risky = components.loc[components["symbol"] == "RISKY"].iloc[0]
    safe = components.loc[components["symbol"] == "SAFE"].iloc[0]
    assert risky["entry_risk_multiplier"] < safe["entry_risk_multiplier"]
    assert risky["risk_adjusted_entry_score"] < risky["opportunity_gate_score"]
    assert safe["risk_adjusted_entry_score"] > risky["risk_adjusted_entry_score"]


def _v5_policy_row(timestamp: pd.Timestamp, symbol: str, *, rank: float, entry: float, persistence: float, risk: float, short_profit: float, long_profit: float) -> dict:
    return {
        "timestamp": timestamp, "fold_id": 1, "symbol": symbol,
        "asset_rank_score": rank, "entry_score": entry, "hold_score": persistence,
        "entry_threshold": 0.34, "exit_threshold": 0.37, "rotation_hurdle": 0.03,
        "incumbent_persistence_score": persistence, "all_horizon_risk_safety": risk,
        "short_risk_safety": risk, "short_profit_consensus": short_profit,
        "long_profit_confirmation": long_profit, "short_horizon_agreement": 0.9,
        "horizon_agreement": 0.9, "profit_before_loss_probability": 0.5,
        "reentry_margin": 0.11, "reentry_decay_sessions": 5.0,
    }


def test_temporal_v5_protects_incumbent_until_challenger_is_meaningfully_better() -> None:
    from types import SimpleNamespace
    dates = pd.date_range("2026-01-05", periods=5, freq="B", tz="UTC")
    rows = [
        _v5_policy_row(dates[0], "AAA", rank=0.80, entry=0.60, persistence=0.80, risk=0.80, short_profit=0.90, long_profit=0.85),
        _v5_policy_row(dates[0], "BBB", rank=0.70, entry=0.55, persistence=0.70, risk=0.70, short_profit=0.80, long_profit=0.75),
        _v5_policy_row(dates[1], "AAA", rank=0.80, entry=0.55, persistence=0.80, risk=0.80, short_profit=0.90, long_profit=0.85),
        _v5_policy_row(dates[1], "BBB", rank=0.85, entry=0.60, persistence=0.75, risk=0.80, short_profit=0.95, long_profit=0.85),
        _v5_policy_row(dates[2], "AAA", rank=0.45, entry=0.42, persistence=0.65, risk=0.55, short_profit=0.60, long_profit=0.60),
        _v5_policy_row(dates[2], "BBB", rank=0.98, entry=0.75, persistence=0.85, risk=0.90, short_profit=0.98, long_profit=0.95),
    ]
    frame = pd.DataFrame(rows)
    open_prices = pd.DataFrame(1.0, index=dates, columns=["AAA", "BBB"])
    config = SimpleNamespace(slippage_bps=0.0, commission_rate=0.0, initial_capital=10000.0)
    result = _shadow_capital_study(frame, open_prices, dates, config, include_diagnostics=True, decision_policy="trend_capture_hysteresis")
    actions = [item["action"] for item in result["decision_diagnostics"]]
    targets = [item["target_symbol"] for item in result["decision_diagnostics"]]
    assert actions == ["BUY", "HOLD", "ROTATE"]
    assert targets == ["AAA", "AAA", "BBB"]


def test_temporal_v5_severe_risk_can_exit_immediately_and_reentry_requires_recovery() -> None:
    from types import SimpleNamespace
    dates = pd.date_range("2026-01-05", periods=6, freq="B", tz="UTC")
    rows = [
        _v5_policy_row(dates[0], "AAA", rank=0.90, entry=0.65, persistence=0.80, risk=0.80, short_profit=0.90, long_profit=0.85),
        _v5_policy_row(dates[0], "BBB", rank=0.50, entry=0.30, persistence=0.50, risk=0.60, short_profit=0.50, long_profit=0.50),
        _v5_policy_row(dates[1], "AAA", rank=0.85, entry=0.40, persistence=0.80, risk=0.01, short_profit=0.70, long_profit=0.80),
        _v5_policy_row(dates[1], "BBB", rank=0.50, entry=0.30, persistence=0.50, risk=0.50, short_profit=0.50, long_profit=0.50),
        _v5_policy_row(dates[2], "AAA", rank=0.90, entry=0.40, persistence=0.70, risk=0.70, short_profit=0.80, long_profit=0.75),
        _v5_policy_row(dates[2], "BBB", rank=0.50, entry=0.30, persistence=0.50, risk=0.50, short_profit=0.50, long_profit=0.50),
        _v5_policy_row(dates[3], "AAA", rank=0.95, entry=0.65, persistence=0.80, risk=0.80, short_profit=0.90, long_profit=0.85),
        _v5_policy_row(dates[3], "BBB", rank=0.50, entry=0.30, persistence=0.50, risk=0.50, short_profit=0.50, long_profit=0.50),
    ]
    frame = pd.DataFrame(rows)
    open_prices = pd.DataFrame(1.0, index=dates, columns=["AAA", "BBB"])
    config = SimpleNamespace(slippage_bps=0.0, commission_rate=0.0, initial_capital=10000.0)
    result = _shadow_capital_study(frame, open_prices, dates, config, include_diagnostics=True, decision_policy="trend_capture_hysteresis")
    diagnostics = result["decision_diagnostics"]
    assert [item["action"] for item in diagnostics] == ["BUY", "SELL", "CASH", "BUY"]
    assert diagnostics[1]["reason"] == "severe_risk_exit"
    assert diagnostics[2]["active_reentry_margin"] > 0
    assert result["median_cash_days"] == pytest.approx(2.0)
    assert result["next_day_reentry_count"] == 0


def test_temporal_v5_state_duration_metrics_measure_hysteresis() -> None:
    from market_cycle_trader_api.engine.temporal_intelligence import _state_duration_metrics
    metrics = _state_duration_metrics([
        (1, "AAA"), (1, "AAA"), (1, "AAA"), (1, "CASH"), (1, "CASH"),
        (1, "BBB"), (1, "BBB"), (2, "CCC"),
    ])
    assert metrics["median_holding_days"] == pytest.approx(2.0)
    assert metrics["short_holding_ratio_2d"] == pytest.approx(2.0 / 3.0)
    assert metrics["median_cash_days"] == pytest.approx(2.0)


def test_temporal_v5_export_schema_is_v6() -> None:
    document = {
        "id": "temporal-v5-test", "status": "completed",
        "experiment": "temporal_decision_intelligence_v5_trend_capture_hysteresis",
        "strategy_profile_name": "Winner #3", "horizons": [5, 10, 20, 40, 60],
        "request": {"rotation_target_horizons": [5, 10, 20, 40, 60]},
        "result": {
            "experiment": "temporal_decision_intelligence_v5_trend_capture_hysteresis",
            "horizons": [5, 10, 20, 40, 60], "asset_count": 2, "feature_count": 52,
            "walk_forward_fold_count": 1, "horizon_metrics": [], "fold_metrics": [],
            "latest_forecasts": [], "multi_horizon_metrics": {
                "shadow_capital": {
                    "ending_capital": 12000.0,
                    "cost_stress": [
                        {"one_side_cost_bps": 0.0, "ending_capital": 12000.0, "total_return": 0.2, "sharpe": 1.5, "max_drawdown": -0.1, "switch_cost_events": 4},
                        {"one_side_cost_bps": 5.0, "ending_capital": 11500.0, "total_return": 0.15, "sharpe": 1.4, "max_drawdown": -0.11, "switch_cost_events": 4},
                    ],
                },
                "winner_anchor_replay": {
                    "ending_capital": 11000.0,
                    "cost_stress": [
                        {"one_side_cost_bps": 0.0, "ending_capital": 11000.0, "total_return": 0.1, "sharpe": 1.3, "max_drawdown": -0.12, "switch_cost_events": 3},
                        {"one_side_cost_bps": 5.0, "ending_capital": 10800.0, "total_return": 0.08, "sharpe": 1.2, "max_drawdown": -0.13, "switch_cost_events": 3},
                    ],
                },
            },
            "multi_horizon_fold_metrics": [], "multi_horizon_latest_forecasts": [],
            "winner_reference": {"ending_capital": 10000.0, "folds": []}, "shadow_only": True,
        },
    }
    content = build_temporal_intelligence_export(_TemporalDb(document), "temporal-v5-test")
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        manifest = json.loads(archive.read("temporal_intelligence_manifest.json").decode("utf-8"))
        assert manifest["schema_version"] == "temporal_intelligence_export_v6"

def test_temporal_v4_export_contains_multi_horizon_files() -> None:
    document = {
        "id": "temporal-v4-test",
        "status": "completed",
        "experiment": "temporal_decision_intelligence_v4_multi_horizon",
        "strategy_profile_name": "Winner #3",
        "horizons": [5, 10, 20, 40, 60],
        "request": {"rotation_target_horizons": [5, 10, 20, 40, 60]},
        "result": {
            "experiment": "temporal_decision_intelligence_v4_multi_horizon",
            "horizons": [5, 10, 20, 40, 60],
            "asset_count": 2,
            "feature_count": 52,
            "walk_forward_fold_count": 1,
            "horizon_metrics": [],
            "fold_metrics": [],
            "latest_forecasts": [],
            "multi_horizon_metrics": {
                "entry_horizons": [5, 10],
                "hold_horizons": [20, 40, 60],
                "risk_horizons": [5, 10, 20, 40, 60],
                "shadow_capital": {"ending_capital": 25000.0, "action_counts": {"buy": 2}},
            },
            "multi_horizon_fold_metrics": [{"fold_id": 1, "shadow_capital": {"ending_capital": 25000.0}}],
            "multi_horizon_latest_forecasts": [{"symbol": "AAA", "asset_rank_score": 0.9, "shadow_target": True}],
            "winner_reference": {"ending_capital": 30000.0, "folds": []},
            "shadow_only": True,
        },
    }
    observations = [{
        "run_id": "temporal-v4-test",
        "timestamp": pd.Timestamp("2026-01-05", tz="UTC"),
        "rows": [{"symbol": "AAA", "entry_rank_score": 0.9, "short_profit_consensus": 0.8}],
    }]
    content = build_temporal_intelligence_export(_TemporalDb(document, observations), "temporal-v4-test")
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = set(archive.namelist())
        assert "temporal_intelligence_multi_horizon.csv" in names
        assert "temporal_intelligence_multi_horizon_folds.csv" in names
        assert "temporal_intelligence_multi_horizon_decision_diagnostics.csv" in names
        assert "temporal_intelligence_multi_horizon_latest_forecasts.csv" in names
        assert "temporal_intelligence_multi_horizon_daily_assets.csv" in names
        assert "25000" in archive.read("temporal_intelligence_multi_horizon.csv").decode("utf-8")
        assert "AAA" in archive.read("temporal_intelligence_multi_horizon_latest_forecasts.csv").decode("utf-8")
        assert "AAA" in archive.read("temporal_intelligence_multi_horizon_daily_assets.csv").decode("utf-8")


def test_temporal_v6_online_quality_uses_only_matured_labels() -> None:
    dates = pd.date_range("2025-01-02", periods=40, freq="B", tz="UTC")
    rows = []
    for date_index, timestamp in enumerate(dates[:30]):
        for symbol_index, symbol in enumerate(("AAA", "BBB", "CCC", "DDD")):
            probability = 0.80 if symbol_index % 2 == 0 else 0.20
            realized = 1.0 if (date_index < 12 and symbol_index % 2 == 0) else 0.0 if date_index < 12 else 1.0 - float(symbol_index % 2 == 0)
            rows.append({
                "timestamp": timestamp, "symbol": symbol, "fold_id": 1,
                "realized_profit_before_loss": realized,
                "profit_before_loss_probability": probability,
                "raw_profit_before_loss_probability": probability,
                "baseline_profit_before_loss_probability": 0.5,
                "realized_bottom": realized, "bottom_probability": probability,
                "raw_bottom_probability": probability, "baseline_bottom_probability": 0.5,
                "realized_top": 1.0 - realized, "top_probability": 1.0 - probability,
                "raw_top_probability": 1.0 - probability, "baseline_top_probability": 0.5,
                "realized_trend_persistence": realized, "trend_persistence_probability": probability,
                "raw_trend_persistence_probability": probability, "baseline_trend_persistence_probability": 0.5,
                "realized_drawdown": 0.02 if realized else 0.12,
                "predicted_drawdown": 0.03 if probability > 0.5 else 0.10,
                "baseline_drawdown": 0.08,
                "profit_before_loss_quality_weight": 0.35, "bottom_quality_weight": 0.0,
                "top_quality_weight": 0.0, "trend_persistence_quality_weight": 0.0,
                "drawdown_quality_weight": 0.0, "quality_history_samples": 0,
            })
    frame = pd.DataFrame(rows)
    adaptive = _apply_online_matured_quality(
        frame, dates, 5, window_sessions=10, update_every_sessions=1,
        minimum_history_samples=8, full_trust_samples=16, smoothing_alpha=1.0,
    )
    before_reversal_matures = adaptive.loc[adaptive["timestamp"] == dates[14]].iloc[0]
    after_reversal_matures = adaptive.loc[adaptive["timestamp"] == dates[24]].iloc[0]
    assert before_reversal_matures["profit_before_loss_quality_weight"] > after_reversal_matures["profit_before_loss_quality_weight"]
    assert pd.Timestamp(before_reversal_matures["online_quality_maturity_cutoff"]) <= dates[9]
    assert pd.Timestamp(after_reversal_matures["online_quality_maturity_cutoff"]) <= dates[19]
    assert after_reversal_matures["quality_source"] == "online_matured_oos"


def test_temporal_v6_extreme_risk_raises_entry_threshold_and_reduces_incumbent_protection() -> None:
    timestamp = pd.Timestamp("2026-01-05", tz="UTC")
    rows = []
    for symbol, risk in (("RISKY", 0.03), ("SAFE", 0.85)):
        row = {"timestamp": timestamp, "symbol": symbol, "fold_id": 1}
        for horizon in (5, 10, 20, 40, 60):
            row[f"profit_percentile_h{horizon}"] = 0.95
            row[f"profit_before_loss_probability_h{horizon}"] = 0.60
            row[f"baseline_profit_before_loss_probability_h{horizon}"] = 0.40
            row[f"profit_before_loss_quality_weight_h{horizon}"] = 0.9
            row[f"predicted_drawdown_h{horizon}"] = 0.05 if risk > 0.5 else 0.20
            row[f"risk_safety_percentile_h{horizon}"] = risk
            row[f"drawdown_quality_weight_h{horizon}"] = 0.9
            row[f"bottom_probability_h{horizon}"] = 0.5
            row[f"baseline_bottom_probability_h{horizon}"] = 0.5
            row[f"bottom_quality_weight_h{horizon}"] = 0.0
            row[f"top_probability_h{horizon}"] = 0.5
            row[f"baseline_top_probability_h{horizon}"] = 0.5
            row[f"top_quality_weight_h{horizon}"] = 0.0
            row[f"trend_direction_h{horizon}"] = 1.0
            row[f"trend_persistence_probability_h{horizon}"] = 0.80
            row[f"baseline_trend_persistence_probability_h{horizon}"] = 0.5
            row[f"trend_persistence_quality_weight_h{horizon}"] = 0.8
            row[f"quality_history_samples_h{horizon}"] = 1000
        rows.append(row)
    components = _multi_horizon_components(pd.DataFrame(rows), [5, 10, 20, 40, 60], one_side_cost=0.0)
    risky = components.loc[components["symbol"] == "RISKY"].iloc[0]
    safe = components.loc[components["symbol"] == "SAFE"].iloc[0]
    assert risky["risk_entry_threshold_penalty"] > safe["risk_entry_threshold_penalty"]
    assert risky["entry_threshold"] > safe["entry_threshold"]
    assert risky["incumbent_risk_health"] < safe["incumbent_risk_health"]
    assert risky["incumbent_persistence_score"] < safe["incumbent_persistence_score"]


def test_temporal_v6_risk_break_can_exit_even_when_incumbent_persistence_is_high() -> None:
    from types import SimpleNamespace
    dates = pd.date_range("2026-01-05", periods=5, freq="B", tz="UTC")
    rows = [
        {**_v5_policy_row(dates[0], "AAA", rank=0.90, entry=0.65, persistence=0.80, risk=0.80, short_profit=0.90, long_profit=0.85), "long_risk_safety": 0.80, "incumbent_risk_health": 1.0},
        {**_v5_policy_row(dates[0], "BBB", rank=0.50, entry=0.30, persistence=0.50, risk=0.60, short_profit=0.50, long_profit=0.50), "long_risk_safety": 0.60, "incumbent_risk_health": 1.0},
        {**_v5_policy_row(dates[1], "AAA", rank=0.80, entry=0.45, persistence=0.78, risk=0.08, short_profit=0.70, long_profit=0.75), "short_risk_safety": 0.10, "long_risk_safety": 0.12, "incumbent_risk_health": 0.0},
        {**_v5_policy_row(dates[1], "BBB", rank=0.55, entry=0.30, persistence=0.55, risk=0.50, short_profit=0.50, long_profit=0.50), "long_risk_safety": 0.50, "incumbent_risk_health": 1.0},
    ]
    frame = pd.DataFrame(rows)
    open_prices = pd.DataFrame(1.0, index=dates, columns=["AAA", "BBB"])
    config = SimpleNamespace(slippage_bps=0.0, commission_rate=0.0, initial_capital=10000.0)
    result = _shadow_capital_study(frame, open_prices, dates, config, include_diagnostics=True, decision_policy="adaptive_trend_capture")
    diagnostics = result["decision_diagnostics"]
    assert diagnostics[0]["action"] == "BUY"
    assert diagnostics[1]["action"] == "SELL"
    assert diagnostics[1]["reason"] == "risk_break_exit"
    assert diagnostics[1]["risk_break_exit"] is True


def test_temporal_v6_export_schema_is_v7() -> None:
    document = {
        "id": "temporal-v6-test", "status": "completed",
        "experiment": "temporal_decision_intelligence_v6_adaptive_trend_capture",
        "strategy_profile_name": "Winner #3", "horizons": [5, 10, 20, 40, 60],
        "request": {"rotation_target_horizons": [5, 10, 20, 40, 60]},
        "result": {
            "experiment": "temporal_decision_intelligence_v6_adaptive_trend_capture",
            "horizons": [5, 10, 20, 40, 60], "asset_count": 2, "feature_count": 52,
            "walk_forward_fold_count": 1, "horizon_metrics": [], "fold_metrics": [],
            "latest_forecasts": [], "multi_horizon_metrics": {"shadow_capital": {"ending_capital": 10000.0}},
            "multi_horizon_fold_metrics": [], "multi_horizon_latest_forecasts": [],
            "winner_reference": {"ending_capital": 10000.0, "folds": []}, "shadow_only": True,
        },
    }
    content = build_temporal_intelligence_export(_TemporalDb(document), "temporal-v6-test")
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        manifest = json.loads(archive.read("temporal_intelligence_manifest.json").decode("utf-8"))
        assert manifest["schema_version"] == "temporal_intelligence_export_v7"



def test_temporal_v7_soft_exit_holds_when_incumbent_still_clears_entry() -> None:
    from types import SimpleNamespace
    dates = pd.date_range("2026-01-05", periods=5, freq="B", tz="UTC")
    first_aaa = _v5_policy_row(dates[0], "AAA", rank=0.90, entry=0.65, persistence=0.80, risk=0.80, short_profit=0.90, long_profit=0.85)
    first_bbb = _v5_policy_row(dates[0], "BBB", rank=0.50, entry=0.30, persistence=0.50, risk=0.60, short_profit=0.50, long_profit=0.50)
    soft_aaa = _v5_policy_row(dates[1], "AAA", rank=0.92, entry=0.58, persistence=0.30, risk=0.20, short_profit=0.25, long_profit=0.75)
    soft_aaa.update({"incumbent_risk_health": 0.30, "long_risk_safety": 0.25})
    soft_bbb = _v5_policy_row(dates[1], "BBB", rank=0.60, entry=0.40, persistence=0.50, risk=0.50, short_profit=0.55, long_profit=0.55)
    soft_bbb.update({"incumbent_risk_health": 0.80, "long_risk_safety": 0.50})
    frame = pd.DataFrame([first_aaa, first_bbb, soft_aaa, soft_bbb])
    open_prices = pd.DataFrame(1.0, index=dates, columns=["AAA", "BBB"])
    config = SimpleNamespace(slippage_bps=0.0, commission_rate=0.0, initial_capital=10000.0)
    result = _shadow_capital_study(frame, open_prices, dates, config, include_diagnostics=True, decision_policy="adaptive_rotation_before_cash")
    diagnostics = result["decision_diagnostics"]
    assert diagnostics[0]["action"] == "BUY"
    assert diagnostics[1]["action"] == "HOLD"
    assert diagnostics[1]["reason"] == "incumbent_entry_signal_overrides_soft_exit"
    assert result["incumbent_entry_recovery_hold_count"] == 1


def test_temporal_v7_soft_exit_rotates_before_cash_when_challenger_is_valid() -> None:
    from types import SimpleNamespace
    dates = pd.date_range("2026-01-05", periods=5, freq="B", tz="UTC")
    first_aaa = _v5_policy_row(dates[0], "AAA", rank=0.90, entry=0.65, persistence=0.80, risk=0.80, short_profit=0.90, long_profit=0.85)
    first_bbb = _v5_policy_row(dates[0], "BBB", rank=0.50, entry=0.30, persistence=0.50, risk=0.60, short_profit=0.50, long_profit=0.50)
    soft_aaa = _v5_policy_row(dates[1], "AAA", rank=0.45, entry=0.38, persistence=0.30, risk=0.20, short_profit=0.25, long_profit=0.55)
    soft_aaa.update({"incumbent_risk_health": 0.25, "long_risk_safety": 0.25})
    strong_bbb = _v5_policy_row(dates[1], "BBB", rank=0.92, entry=0.62, persistence=0.75, risk=0.75, short_profit=0.92, long_profit=0.85)
    strong_bbb.update({"incumbent_risk_health": 0.90, "long_risk_safety": 0.75})
    frame = pd.DataFrame([first_aaa, first_bbb, soft_aaa, strong_bbb])
    open_prices = pd.DataFrame(1.0, index=dates, columns=["AAA", "BBB"])
    config = SimpleNamespace(slippage_bps=0.0, commission_rate=0.0, initial_capital=10000.0)
    result = _shadow_capital_study(frame, open_prices, dates, config, include_diagnostics=True, decision_policy="adaptive_rotation_before_cash")
    diagnostics = result["decision_diagnostics"]
    assert diagnostics[1]["action"] == "ROTATE"
    assert diagnostics[1]["target_symbol"] == "BBB"
    assert diagnostics[1]["reason"] == "opportunity_exit_rotates_before_cash"
    assert result["rotation_before_cash_count"] == 1
    assert result["opportunity_exit_cash_count"] == 0


def test_temporal_v7_defensive_exit_keeps_cash_when_challenger_risk_is_not_safe() -> None:
    from types import SimpleNamespace
    dates = pd.date_range("2026-01-05", periods=5, freq="B", tz="UTC")
    first_aaa = _v5_policy_row(dates[0], "AAA", rank=0.90, entry=0.65, persistence=0.80, risk=0.80, short_profit=0.90, long_profit=0.85)
    first_bbb = _v5_policy_row(dates[0], "BBB", rank=0.50, entry=0.30, persistence=0.50, risk=0.60, short_profit=0.50, long_profit=0.50)
    broken_aaa = _v5_policy_row(dates[1], "AAA", rank=0.40, entry=0.30, persistence=0.70, risk=0.08, short_profit=0.60, long_profit=0.65)
    broken_aaa.update({"short_risk_safety": 0.10, "long_risk_safety": 0.10, "incumbent_risk_health": 0.0})
    risky_bbb = _v5_policy_row(dates[1], "BBB", rank=0.95, entry=0.65, persistence=0.80, risk=0.20, short_profit=0.95, long_profit=0.90)
    risky_bbb.update({"short_risk_safety": 0.20, "long_risk_safety": 0.20, "incumbent_risk_health": 0.30})
    frame = pd.DataFrame([first_aaa, first_bbb, broken_aaa, risky_bbb])
    open_prices = pd.DataFrame(1.0, index=dates, columns=["AAA", "BBB"])
    config = SimpleNamespace(slippage_bps=0.0, commission_rate=0.0, initial_capital=10000.0)
    result = _shadow_capital_study(frame, open_prices, dates, config, include_diagnostics=True, decision_policy="adaptive_rotation_before_cash")
    diagnostics = result["decision_diagnostics"]
    assert diagnostics[1]["action"] == "SELL"
    assert diagnostics[1]["reason"] == "risk_break_exit"
    assert diagnostics[1]["cash_recovery_mode"] == "defensive"
    assert result["defensive_exit_cash_count"] == 1


def test_temporal_v8_uses_winner_anchor_for_overall_and_fold_replay() -> None:
    engine = (SRC / "engine" / "temporal_intelligence.py").read_text(encoding="utf-8")
    assert engine.count('_winner_anchored_temporal_study(') >= 5
    assert 'temporal_decision_intelligence_v8_winner_anchored_timing' in engine
    assert 'timing_minimum_advantage' in engine



def test_temporal_v8_winner_anchor_overrides_top1_only_with_strong_top2_timing() -> None:
    from types import SimpleNamespace
    dates = pd.date_range("2026-01-05", periods=5, freq="B", tz="UTC")
    rows = []
    for date, aaa_short, bbb_short in ((dates[0], 0.30, 0.80), (dates[1], 0.70, 0.90)):
        aaa = _v5_policy_row(date, "AAA", rank=0.90, entry=0.60, persistence=0.70, risk=0.70, short_profit=aaa_short, long_profit=0.70)
        bbb = _v5_policy_row(date, "BBB", rank=0.70, entry=0.55, persistence=0.65, risk=0.30, short_profit=bbb_short, long_profit=0.65)
        rows.extend([aaa, bbb])
    frame = pd.DataFrame(rows)
    open_prices = pd.DataFrame({"AAA": [100.0, 100.0, 101.0, 102.0, 102.0], "BBB": [100.0, 100.0, 110.0, 111.0, 111.0]}, index=dates)
    winner_rows = [
        {"decision_date": dates[0], "selected_asset": "AAA", "top_1_asset": "AAA", "top_2_asset": "BBB", "top_1_score": 0.9, "top_2_score": 0.8},
        {"decision_date": dates[1], "selected_asset": "AAA", "top_1_asset": "AAA", "top_2_asset": "BBB", "top_1_score": 0.9, "top_2_score": 0.8},
    ]
    config = SimpleNamespace(slippage_bps=0.0, commission_rate=0.0, initial_capital=10000.0)
    result = _winner_anchored_temporal_study(frame, winner_rows, open_prices, dates, config, include_diagnostics=True)
    diagnostics = result["decision_diagnostics"]
    assert diagnostics[0]["target_symbol"] == "BBB"
    assert diagnostics[0]["temporal_timing_override"] is True
    assert diagnostics[0]["reason"] == "temporal_short_timing_overrides_winner_top1_with_top2"
    assert diagnostics[0]["temporal_timing_candidate"] is True
    assert diagnostics[0]["winner_anchor_interval_return"] == pytest.approx(0.01)
    assert diagnostics[0]["winner_top2_interval_return"] == pytest.approx(0.10)
    assert diagnostics[0]["temporal_timing_alpha_return"] == pytest.approx(0.09)
    assert diagnostics[0]["temporal_timing_alpha_log_return"] > 0.0
    assert diagnostics[1]["target_symbol"] == "AAA"
    assert diagnostics[1]["temporal_timing_override"] is False
    assert result["timing_override_count"] == 1
    cost_stress = {row["one_side_cost_bps"]: row for row in result["cost_stress"]}
    assert set(cost_stress) == {0.0, 1.0, 2.0, 5.0, 10.0}
    assert cost_stress[0.0]["ending_capital"] > cost_stress[10.0]["ending_capital"]
    assert cost_stress[10.0]["switch_cost_events"] >= 1


def test_temporal_v8_winner_anchor_supports_legacy_raw_best_and_second_fields() -> None:
    from types import SimpleNamespace
    dates = pd.date_range("2026-01-05", periods=4, freq="B", tz="UTC")
    aaa = _v5_policy_row(dates[0], "AAA", rank=0.90, entry=0.60, persistence=0.70, risk=0.70, short_profit=0.30, long_profit=0.70)
    bbb = _v5_policy_row(dates[0], "BBB", rank=0.70, entry=0.55, persistence=0.65, risk=0.30, short_profit=0.80, long_profit=0.65)
    frame = pd.DataFrame([aaa, bbb])
    open_prices = pd.DataFrame({"AAA": [100.0, 100.0, 102.0, 102.0], "BBB": [100.0, 100.0, 110.0, 110.0]}, index=dates)
    winner_rows = [{
        "decision_date": dates[0], "selected_asset": "AAA",
        "raw_best_asset": "AAA", "second_asset": "BBB",
        "raw_best_score": 0.9, "second_score": 0.8,
    }]
    config = SimpleNamespace(slippage_bps=0.0, commission_rate=0.0, initial_capital=10000.0)
    result = _winner_anchored_temporal_study(frame, winner_rows, open_prices, dates, config, include_diagnostics=True)
    assert result["decision_diagnostics"][0]["winner_top1_symbol"] == "AAA"
    assert result["decision_diagnostics"][0]["winner_top2_symbol"] == "BBB"
    assert result["decision_diagnostics"][0]["target_symbol"] == "BBB"
    assert result["timing_override_count"] == 1


def test_temporal_v8_winner_anchor_does_not_override_without_25_point_advantage() -> None:
    from types import SimpleNamespace
    dates = pd.date_range("2026-01-05", periods=4, freq="B", tz="UTC")
    aaa = _v5_policy_row(dates[0], "AAA", rank=0.90, entry=0.60, persistence=0.70, risk=0.70, short_profit=0.45, long_profit=0.70)
    bbb = _v5_policy_row(dates[0], "BBB", rank=0.70, entry=0.55, persistence=0.65, risk=0.30, short_profit=0.65, long_profit=0.65)
    frame = pd.DataFrame([aaa, bbb])
    open_prices = pd.DataFrame({"AAA": [100.0, 100.0, 102.0, 102.0], "BBB": [100.0, 100.0, 110.0, 110.0]}, index=dates)
    winner_rows = [{"decision_date": dates[0], "selected_asset": "AAA", "top_1_asset": "AAA", "top_2_asset": "BBB"}]
    config = SimpleNamespace(slippage_bps=0.0, commission_rate=0.0, initial_capital=10000.0)
    result = _winner_anchored_temporal_study(frame, winner_rows, open_prices, dates, config, include_diagnostics=True)
    assert result["decision_diagnostics"][0]["target_symbol"] == "AAA"
    assert result["timing_override_count"] == 0


def test_temporal_metrics_ignore_unmatured_tail_labels() -> None:
    classification = _classification_metrics(
        np.asarray([1.0, 0.0, np.nan], dtype=float),
        np.asarray([0.8, 0.2, 0.9], dtype=float),
    )
    assert classification["auc"] == pytest.approx(1.0)
    engine = (SRC / "engine" / "temporal_intelligence.py").read_text(encoding="utf-8")
    assert 'valid &= np.isfinite(values)' not in engine


def test_temporal_v8_export_schema_is_v11() -> None:
    document = {
        "id": "temporal-v8-test", "status": "completed",
        "experiment": "temporal_decision_intelligence_v8_winner_anchored_timing",
        "strategy_profile_name": "Winner #3", "horizons": [5, 10, 20, 40, 60],
        "request": {"rotation_target_horizons": [5, 10, 20, 40, 60]},
        "result": {
            "experiment": "temporal_decision_intelligence_v8_winner_anchored_timing",
            "horizons": [5, 10, 20, 40, 60], "asset_count": 2, "feature_count": 52,
            "walk_forward_fold_count": 1, "horizon_metrics": [], "fold_metrics": [],
            "latest_forecasts": [], "multi_horizon_metrics": {
                "shadow_capital": {
                    "ending_capital": 12000.0,
                    "cost_stress": [
                        {"one_side_cost_bps": 0.0, "ending_capital": 12000.0, "total_return": 0.2, "sharpe": 1.5, "max_drawdown": -0.1, "switch_cost_events": 4},
                        {"one_side_cost_bps": 5.0, "ending_capital": 11500.0, "total_return": 0.15, "sharpe": 1.4, "max_drawdown": -0.11, "switch_cost_events": 4},
                    ],
                },
                "winner_anchor_replay": {
                    "ending_capital": 11000.0,
                    "cost_stress": [
                        {"one_side_cost_bps": 0.0, "ending_capital": 11000.0, "total_return": 0.1, "sharpe": 1.3, "max_drawdown": -0.12, "switch_cost_events": 3},
                        {"one_side_cost_bps": 5.0, "ending_capital": 10800.0, "total_return": 0.08, "sharpe": 1.2, "max_drawdown": -0.13, "switch_cost_events": 3},
                    ],
                },
            },
            "multi_horizon_fold_metrics": [], "multi_horizon_latest_forecasts": [],
            "winner_reference": {"ending_capital": 10000.0, "folds": []}, "shadow_only": True,
        },
    }
    content = build_temporal_intelligence_export(_TemporalDb(document), "temporal-v8-test")
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        manifest = json.loads(archive.read("temporal_intelligence_manifest.json").decode("utf-8"))
        assert manifest["schema_version"] == "temporal_intelligence_export_v11"
        names = set(archive.namelist())
        assert "temporal_intelligence_cost_stress.csv" in names
        assert "temporal_intelligence_timing_override_attribution.csv" in names
        stress = archive.read("temporal_intelligence_cost_stress.csv").decode("utf-8")
        assert "capital_lift_vs_winner_anchor" in stress
        assert "12000.0" in stress


def test_temporal_v7_export_schema_is_v9() -> None:
    document = {
        "id": "temporal-v7-test", "status": "completed",
        "experiment": "temporal_decision_intelligence_v7_rotation_before_cash",
        "strategy_profile_name": "Winner #3", "horizons": [5, 10, 20, 40, 60],
        "request": {"rotation_target_horizons": [5, 10, 20, 40, 60]},
        "result": {
            "experiment": "temporal_decision_intelligence_v7_rotation_before_cash",
            "horizons": [5, 10, 20, 40, 60], "asset_count": 2, "feature_count": 52,
            "walk_forward_fold_count": 1, "horizon_metrics": [], "fold_metrics": [],
            "latest_forecasts": [], "multi_horizon_metrics": {"shadow_capital": {"ending_capital": 10000.0}},
            "multi_horizon_fold_metrics": [], "multi_horizon_latest_forecasts": [],
            "winner_reference": {"ending_capital": 10000.0, "folds": []}, "shadow_only": True,
        },
    }
    content = build_temporal_intelligence_export(_TemporalDb(document), "temporal-v7-test")
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        manifest = json.loads(archive.read("temporal_intelligence_manifest.json").decode("utf-8"))
        assert manifest["schema_version"] == "temporal_intelligence_export_v9"


def test_temporal_v7_observation_export_includes_realized_labels_and_execution_path() -> None:
    dates = pd.date_range("2026-01-05", periods=4, freq="B", tz="UTC")
    frame = pd.DataFrame([{
        "timestamp": dates[0], "fold_id": 1, "symbol": "AAA",
        "realized_profit_before_loss_h5": 1.0, "realized_bottom_h5": 1.0,
        "realized_top_h5": 0.0, "realized_trend_persistence_h5": 1.0,
        "realized_drawdown_h5": 0.04,
    }])
    bars = pd.DataFrame({
        "open": [100.0, 101.0, 103.0, 104.0],
        "high": [101.0, 104.0, 105.0, 106.0],
        "low": [99.0, 100.0, 102.0, 103.0],
        "close": [100.5, 103.0, 104.0, 105.0],
        "volume": [1000.0, 1100.0, 1200.0, 1300.0],
    }, index=dates)
    rows = _multi_horizon_observation_rows(
        frame, [5], frames_by_symbol={"AAA": bars}, common_dates=dates
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["realized_profit_before_loss_h5"] == 1.0
    assert row["realized_drawdown_h5"] == pytest.approx(0.04)
    assert row["execution_date"] == dates[1]
    assert row["next_execution_date"] == dates[2]
    assert row["execution_open"] == pytest.approx(101.0)
    assert row["next_open"] == pytest.approx(103.0)
    assert row["open_to_open_return"] == pytest.approx(103.0 / 101.0 - 1.0)


def test_temporal_v7_shadow_economic_curve_contains_exact_net_capital_path() -> None:
    from types import SimpleNamespace
    dates = pd.date_range("2026-01-05", periods=5, freq="B", tz="UTC")
    rows = []
    for date in dates[:2]:
        rows.append(_v5_policy_row(date, "AAA", rank=0.9, entry=0.7, persistence=0.8, risk=0.8, short_profit=0.8, long_profit=0.8))
        rows.append(_v5_policy_row(date, "BBB", rank=0.4, entry=0.3, persistence=0.5, risk=0.5, short_profit=0.5, long_profit=0.5))
    frame = pd.DataFrame(rows)
    open_prices = pd.DataFrame({"AAA": [100.0, 100.0, 110.0, 121.0, 121.0], "BBB": [100.0]*5}, index=dates)
    config = SimpleNamespace(slippage_bps=0.0, commission_rate=0.0, initial_capital=10000.0)
    result = _shadow_capital_study(
        frame, open_prices, dates, config, include_economic_curve=True, decision_policy="adaptive_rotation_before_cash"
    )
    curve = result["economic_curve"]
    assert curve
    assert curve[0]["gross_interval_return"] == pytest.approx(0.10)
    assert curve[0]["strategy_equity"] == pytest.approx(11000.0)
    assert "strategy_drawdown" in curve[0]


def test_temporal_v7_export_contains_offline_replay_and_winner_daily_files() -> None:
    document = {
        "id": "temporal-v7-replay", "status": "completed",
        "experiment": "temporal_decision_intelligence_v7_rotation_before_cash",
        "strategy_profile_name": "Winner #3", "horizons": [5, 10, 20, 40, 60],
        "request": {"rotation_target_horizons": [5, 10, 20, 40, 60]},
        "result": {
            "experiment": "temporal_decision_intelligence_v7_rotation_before_cash",
            "horizons": [5, 10, 20, 40, 60], "asset_count": 1, "feature_count": 52,
            "walk_forward_fold_count": 1, "horizon_metrics": [], "fold_metrics": [],
            "latest_forecasts": [], "multi_horizon_metrics": {"shadow_capital": {"ending_capital": 12000.0}},
            "multi_horizon_fold_metrics": [], "multi_horizon_latest_forecasts": [],
            "winner_reference": {"ending_capital": 15000.0, "folds": []}, "shadow_only": True,
        },
    }
    observations = [{
        "run_id": "temporal-v7-replay", "timestamp": "2026-01-05T00:00:00+00:00",
        "rows": [{
            "fold_id": 1, "symbol": "AAA", "execution_date": "2026-01-06T00:00:00+00:00",
            "next_execution_date": "2026-01-07T00:00:00+00:00", "execution_open": 100.0,
            "next_open": 105.0, "open_to_open_return": 0.05, "realized_profit_before_loss_h5": 1.0,
        }],
    }]
    artifacts = [
        {"run_id": "temporal-v7-replay", "kind": "decision_diagnostics", "sequence": 0, "rows": [
            {"artifact_kind": "multi_horizon_equity_curve", "decision_timestamp": "2026-01-05", "strategy_equity": 10500.0}
        ]},
        {"run_id": "temporal-v7-replay", "kind": "winner_reference_daily", "sequence": 0, "rows": [
            {"timestamp": "2026-01-06", "strategy_equity": 11000.0, "buy_hold_equity": 10300.0, "selected_asset": "AAA"}
        ]},
        {"run_id": "temporal-v7-replay", "kind": "winner_reference_trades", "sequence": 0, "rows": [
            {"timestamp": "2026-01-06", "action": "BUY", "asset": "AAA", "execution_price": 100.0}
        ]},
    ]
    content = build_temporal_intelligence_export(_TemporalDb(document, observations, artifacts), "temporal-v7-replay")
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = set(archive.namelist())
        assert "temporal_intelligence_multi_horizon_market_replay.csv" in names
        assert "temporal_intelligence_multi_horizon_equity_curve.csv" in names
        assert "temporal_intelligence_winner_reference_daily.csv" in names
        assert "temporal_intelligence_winner_reference_trades.csv" in names
        market = archive.read("temporal_intelligence_multi_horizon_market_replay.csv").decode("utf-8")
        equity = archive.read("temporal_intelligence_multi_horizon_equity_curve.csv").decode("utf-8")
        winner_daily = archive.read("temporal_intelligence_winner_reference_daily.csv").decode("utf-8")
        winner_trades = archive.read("temporal_intelligence_winner_reference_trades.csv").decode("utf-8")
        assert "open_to_open_return" in market and "0.05" in market
        assert "strategy_equity" in equity and "10500" in equity
        assert "buy_hold_equity" in winner_daily and "11000" in winner_daily
        assert "execution_price" in winner_trades and "BUY" in winner_trades
