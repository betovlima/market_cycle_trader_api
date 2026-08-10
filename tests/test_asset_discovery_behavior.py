from __future__ import annotations

import numpy as np
import pandas as pd

from market_cycle_trader_api.services.asset_discovery_behavior import behavior_risk_profile


SETTINGS = {
    "behavior_lookback_sessions": 756,
    "behavior_min_sessions": 63,
    "behavior_max_downside_tail_1pct": 0.12,
    "behavior_max_gap_downside_tail_1pct": 0.10,
    "behavior_max_annualized_volatility": 0.90,
    "behavior_max_drawdown": 0.75,
    "behavior_max_single_day_loss": 0.30,
    "behavior_max_single_gap_loss": 0.25,
    "behavior_max_10_session_loss": 0.35,
}


def _frame(closes: np.ndarray, opens: np.ndarray | None = None) -> pd.DataFrame:
    index = pd.date_range("2023-01-02", periods=len(closes), freq="B", tz="UTC")
    open_values = closes.copy() if opens is None else opens
    return pd.DataFrame(
        {
            "open": open_values,
            "high": np.maximum(open_values, closes) * 1.01,
            "low": np.minimum(open_values, closes) * 0.99,
            "close": closes,
            "volume": np.full(len(closes), 1_000_000.0),
        },
        index=index,
    )


def test_stable_history_passes_behavior_gate() -> None:
    closes = 100.0 * np.cumprod(np.full(260, 1.001))
    result = behavior_risk_profile(_frame(closes), SETTINGS)
    assert result["sample_ready"] is True
    assert result["passed"] is True
    assert result["reason_codes"] == ["behavior_risk_ready"]


def test_large_price_shock_is_rejected() -> None:
    closes = 100.0 * np.cumprod(np.full(260, 1.001))
    closes[180:] *= 0.55
    result = behavior_risk_profile(_frame(closes), SETTINGS)
    assert result["sample_ready"] is True
    assert result["passed"] is False
    assert "single_day_shock_failed" in result["reason_codes"]


def test_short_history_remains_eligible_for_watchlist_path() -> None:
    closes = 100.0 * np.cumprod(np.full(30, 1.001))
    result = behavior_risk_profile(_frame(closes), SETTINGS)
    assert result["sample_ready"] is False
    assert result["passed"] is True
    assert result["reason_codes"] == ["behavior_sample_limited"]
