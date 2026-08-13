from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

TRADING_SESSIONS_PER_YEAR = 252
ASSET_DISCOVERY_EVALUATION_POLICY_VERSION = "behavior-risk-v1"


def _nonnegative_magnitude(value: float) -> float:
    return float(abs(min(0.0, float(value))))


def _series_quantile(series: pd.Series, quantile: float) -> float:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return 0.0
    return float(clean.quantile(quantile))


def behavior_risk_profile(frame: pd.DataFrame, settings: dict[str, Any]) -> dict[str, Any]:
    







    if frame is None or frame.empty:
        return {
            "sample_ready": False,
            "passed": True,
            "lookback_sessions": 0,
            "reason_codes": ["behavior_sample_limited"],
        }

    lookback_sessions = max(1, int(settings["behavior_lookback_sessions"]))
    sample = frame.tail(min(lookback_sessions, len(frame))).copy()
    close = pd.to_numeric(sample.get("close"), errors="coerce")
    open_price = pd.to_numeric(sample.get("open"), errors="coerce")
    returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()
    gaps = (open_price / close.shift(1) - 1.0).replace([np.inf, -np.inf], np.nan).dropna()

    clean_close = close.replace([np.inf, -np.inf], np.nan).dropna()
    if clean_close.empty:
        return {
            "sample_ready": False,
            "passed": True,
            "lookback_sessions": 0,
            "reason_codes": ["behavior_sample_limited"],
        }

    drawdown = clean_close / clean_close.cummax() - 1.0
    downside_tail_1pct = _nonnegative_magnitude(_series_quantile(returns, 0.01))
    gap_downside_tail_1pct = _nonnegative_magnitude(_series_quantile(gaps, 0.01))
    annualized_volatility = (
        float(returns.std(ddof=1)) * float(np.sqrt(TRADING_SESSIONS_PER_YEAR))
        if len(returns) >= 2
        else 0.0
    )
    max_drawdown = _nonnegative_magnitude(float(drawdown.min())) if not drawdown.empty else 0.0
    worst_daily_loss = _nonnegative_magnitude(float(returns.min())) if not returns.empty else 0.0
    worst_gap_loss = _nonnegative_magnitude(float(gaps.min())) if not gaps.empty else 0.0

    rolling_10_return = (close / close.shift(10) - 1.0).replace([np.inf, -np.inf], np.nan).dropna()
    worst_10_session_loss = (
        _nonnegative_magnitude(float(rolling_10_return.min())) if not rolling_10_return.empty else 0.0
    )

    sample_ready = len(clean_close) >= int(settings["behavior_min_sessions"])
    metrics = {
        "lookback_sessions": int(len(clean_close)),
        "downside_tail_1pct": downside_tail_1pct,
        "gap_downside_tail_1pct": gap_downside_tail_1pct,
        "annualized_volatility": annualized_volatility,
        "max_drawdown": max_drawdown,
        "worst_daily_loss": worst_daily_loss,
        "worst_gap_loss": worst_gap_loss,
        "worst_10_session_loss": worst_10_session_loss,
    }

    if not sample_ready:
        return {
            "sample_ready": False,
            "passed": True,
            **metrics,
            "reason_codes": ["behavior_sample_limited"],
        }

    checks = {
        "downside_tail_risk_failed": downside_tail_1pct
        <= float(settings["behavior_max_downside_tail_1pct"]),
        "gap_downside_risk_failed": gap_downside_tail_1pct
        <= float(settings["behavior_max_gap_downside_tail_1pct"]),
        "volatility_stability_failed": annualized_volatility
        <= float(settings["behavior_max_annualized_volatility"]),
        "drawdown_stability_failed": max_drawdown
        <= float(settings["behavior_max_drawdown"]),
        "single_day_shock_failed": worst_daily_loss
        <= float(settings["behavior_max_single_day_loss"]),
        "single_gap_shock_failed": worst_gap_loss
        <= float(settings["behavior_max_single_gap_loss"]),
        "short_window_shock_failed": worst_10_session_loss
        <= float(settings["behavior_max_10_session_loss"]),
    }
    failures = [reason_code for reason_code, passed in checks.items() if not passed]
    return {
        "sample_ready": True,
        "passed": not failures,
        **metrics,
        "reason_codes": failures or ["behavior_risk_ready"],
    }
