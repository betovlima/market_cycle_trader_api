"""Checks for U75-only collective cash protection and one-call onboarding."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from market_cycle_trader_api.core.config import (
    TCC_U75_DAILY_PROTECTION_MODE, RESEARCH_ONLY_SWING_STRATEGY_MODES,
)
from market_cycle_trader_api.engine.live_lightgbm_signal import LiveLightGBMDecision
from market_cycle_trader_api.engine.u75_market_protection import (
    ENGINE_BINDING, _indicators, apply_live_u75_protection,
    daily_u75_market_signals, overlay_u75_policy,
    u75_contract_issues, u75_strategy_updates,
)
from market_cycle_trader_api.engine.tcc_u67_operational_runtime import tcc_u67_model_values
from market_cycle_trader_api.services.model_research import (
    execution_settings_from_values, model_execution_snapshot,
)
from market_cycle_trader_api.services.strategy_lab import _trader_runtime_compatibility
from market_cycle_trader_api.api.routers import u75_protection
from market_cycle_trader_api.main import create_app


def _frames(one_up=False, n=90):
    dates = pd.date_range("2025-01-02", periods=n, freq="B", tz="UTC")
    symbols = [f"A{i:02d}" for i in range(75)]
    frames = {}
    for i, symbol in enumerate(symbols):
        factor = 1.01 if i == 0 and one_up else 0.99
        prices = 100.0 * factor ** np.arange(n)
        frames[symbol] = pd.DataFrame({
            "close": prices, "open": prices,
        }, index=dates)
    return frames, symbols, dates


def test_whole_universe_decline_and_no_technical_candidate():
    bars, symbols, dates = _frames()
    signal = _indicators(bars, symbols, dates[-1])
    assert signal["valid_assets"] == 75
    assert signal["universe_stress"]
    assert signal["no_technical_candidate"]
    assert signal["stress_without_candidate"]


def test_one_positive_asset_prevents_whole_universe_cash():
    bars, symbols, dates = _frames(one_up=True)
    signal = _indicators(bars, symbols, dates[-1])
    assert signal["universe_stress"]
    assert signal["technical_candidates_count"] == 1
    assert not signal["stress_without_candidate"]


def test_missing_assets_never_prove_no_opportunities():
    bars, symbols, dates = _frames()
    for sym in symbols[-7:]:
        bars[sym] = bars[sym].iloc[:5]
    signal = _indicators(bars, symbols, dates[-1])
    assert signal["valid_assets"] == 68
    assert signal["universe_stress"]
    assert not signal["no_technical_candidate"]
    assert not signal["stress_without_candidate"]


def test_today_decision_does_not_read_future_bars():
    bars, symbols, dates = _frames(n=95)
    date = dates[85]
    first = _indicators(bars, symbols, date)
    for sym in symbols:
        bars[sym].loc[dates[86]:, "close"] = 10000
    assert _indicators(bars, symbols, date) == first


def test_cash_duration_is_inferred_each_day_not_fixed():
    dates = pd.date_range("2025-01-02", periods=20, freq="B", tz="UTC")
    signal = {
        date: {"stress_without_candidate": i in {2, 4, 5, 6, 7, 8, 9, 10, 11}}
        for i, date in enumerate(dates)
    }
    positions = []
    def base(_date, pos, hold):
        positions.append((pos, hold))
        return (1 if pos == 0 else 2, 0.9)
    wrapped = overlay_u75_policy(base, signal)
    assert wrapped(dates[2], 1, 3)[0] == 0
    assert wrapped(dates[3], 0, 0)[0] == 1  # reentry after just one session
    for i in range(4, 12):
        assert wrapped(dates[i], 0, 0)[0] == 0
    assert wrapped(dates[12], 0, 0)[0] == 1  # reentry after 8 sessions
    assert positions == [(1, 3), (0, 0), (0, 0), (0, 0)]


def test_original_policy_is_kept_and_diagnostics_explain_override():
    dates = pd.date_range("2025-01-02", periods=2, freq="B", tz="UTC")
    signal = {dates[0]: {
        "stress_without_candidate": True, "universe_stress": True,
        "valid_assets": 75, "technical_candidates_count": 0,
    }}
    diagnostic = {}
    wrapped = overlay_u75_policy(
        lambda dt, p, h: (2, .5), signal, diagnostics=diagnostic,
    )
    assert wrapped(dates[0], 1, 4) == (0, 0.0)
    assert diagnostic[dates[0]]["decision_reason"] == "U75_COLLECTIVE_STRESS_NO_CANDIDATE"
    assert diagnostic[dates[0]]["u75_cash_override"] is True


def test_u75_catalog_contract_and_winner_process_compatibility():
    settings = execution_settings_from_values(
        "lightgbm_utility", tcc_u67_model_values(),
        settings_revision=1, profile_id="u75-v101",
    )
    model = model_execution_snapshot("lightgbm_utility", settings)
    config = u75_strategy_updates()
    assert config["strategy_mode"] == TCC_U75_DAILY_PROTECTION_MODE
    assert len(config["assets"]) == 77
    assert not u75_contract_issues(config)
    assert TCC_U75_DAILY_PROTECTION_MODE not in RESEARCH_ONLY_SWING_STRATEGY_MODES
    profile = {
        "backtest_engine_binding": ENGINE_BINDING,
        "configuration": config, "research_model_snapshot": model,
        "strategy_kind": "standard",
    }
    result = _trader_runtime_compatibility(profile)
    assert result["eligible"] and result["code"] == "u75_daily_protection_live_runtime_ready"
    bad = {**profile, "backtest_engine_binding": "unsupported"}
    assert not _trader_runtime_compatibility(bad)["eligible"]


def _decision(date):
    return LiveLightGBMDecision(
        decision_date=date, current_asset="A00", target_asset="A01",
        raw_best_asset="A01", selected_utility=.3,
        utilities={"A01": .3}, cash_edges={},
        opportunity_probability=None, opportunity_confidence=None,
        opportunity_threshold=None, opportunity_accepted=None,
        effective_switch_margin=0.001, calibrated_candidate_margin=0.,
        calibration_score=0.1, training_end=date, calibration_start=date,
        calibration_end=date, final_fit_end=date,
        effective_compute_device="cpu", compute_fallback_reason=None,
        random_state=42,
    )


def test_live_u75_policy_only_changes_decision_when_collective_signal():
    from market_cycle_trader_api.engine.tcc_u77_static_challenger import U77_REQUESTED_ASSETS
    bars, _, dates = _frames()
    remapped = {symbol: frame for symbol, frame in zip(U77_REQUESTED_ASSETS[:75], bars.values())}
    original = _decision(dates[-1])
    changed = apply_live_u75_protection(remapped, original)
    assert changed.target_asset == "CASH"
    assert changed.selected_utility == 0.
    for x in remapped.values():
        pass
    # One resilient asset keeps original decision, no individual stop rule.
    first_key = next(iter(remapped))
    remapped[first_key] = remapped[first_key].copy()
    remapped[first_key]["close"] = 100 * 1.01 ** np.arange(len(dates))
    unchanged = apply_live_u75_protection(remapped, original)
    assert unchanged == original


def test_one_endpoint_registered_and_annual_replay_routes_preserved():
    app = create_app()
    routes = {(r.path, next(iter(r.methods)) if len(r.methods) == 1 else "")
              for r in app.routes if getattr(r, "methods", None)}
    assert ("/api/admin/strategies/u75-protection/install-and-backtest", "POST") in routes
    assert ("/api/analytics/backtests/{job_id}", "GET") in routes
    assert ("/api/analytics/backtests/{job_id}/files", "GET") in routes


def test_one_call_reuses_existing_job_without_installing_or_promoting():
    db = MagicMock()
    db.__getitem__.return_value.find_one.return_value = {
        "id": "existing-job", "status": "running",
        "backtest_engine_binding": ENGINE_BINDING,
    }
    with (patch.object(u75_protection, "database", return_value=db),
          patch.object(u75_protection, "install_u75_protection_strategy") as install,
          patch.object(u75_protection, "queue_backtest_job") as queue):
        result = u75_protection.install_and_backtest(SimpleNamespace(email="admin@example.org"))
    assert result["already_running"]
    assert result["job_id"] == "existing-job"
    assert not result["winner_changed"]
    install.assert_not_called()
    queue.assert_not_called()
