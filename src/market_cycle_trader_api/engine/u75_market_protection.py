"""U75-wide daily cash protection (research-only until certified).

Exact U75 membership is required before declaring absence of opportunity.
All inputs are completed daily bars at the decision close or earlier.
Numeric stress thresholds are frozen from TCC v1.0.1 exploratory research,
NOT statistically validated. No holding-period lock is imposed.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..core.config import TCC_U75_DAILY_PROTECTION_MODE, TCC_U77_FIXED_CHALLENGER_MODE
from .tcc_u77_static_challenger import U77_REQUESTED_ASSETS, tcc_u77_contract_issues

ENGINE_BINDING = "u75_daily_cash_protection_v101"
SOURCE_REPOSITORY = "betovlima/tcc_mba_usp_data_science_analytics"
SOURCE_COMMIT = "a899dd258a83b09df05a2db25177737f4a1c6b36"
SOURCE_BRANCH = "research/u75-stress-dynamic-cash-v1"
SOURCE_VERSION = "u75-universe-stress-diagnostics-v1.0.1"
REFERENCE_CONTROL_CAPITAL = 80685454.78647688
ENGINE_VERSION = "u75-daily-market-protection-v1"
EXPECTED_VALID_UNIVERSE = 75


@dataclass(frozen=True)
class U75Thresholds:
    # The following numerical thresholds are unvalidated research hypotheses.
    min_coverage: int = 68
    share_positive_5_max: float = 0.35
    share_positive_20_max: float = 0.45
    share_below_ma50_min: float = 0.55
    median_return_20_max: float = -0.015


def u75_contract_issues(config: Any) -> list[str]:
    payload = (
        config.model_dump(mode="python")
        if hasattr(config, "model_dump")
        else dict(config) if isinstance(config, dict) else vars(config)
    )
    problems = []
    if payload.get("strategy_mode") != TCC_U75_DAILY_PROTECTION_MODE:
        problems.append("strategy_mode: unexpected U75 cash protection mode")
    inherited = dict(payload)
    inherited["strategy_mode"] = TCC_U77_FIXED_CHALLENGER_MODE
    problems.extend(tcc_u77_contract_issues(inherited))
    return problems


def assert_u75_contract(config: Any) -> None:
    issues = u75_contract_issues(config)
    if issues:
        raise ValueError("U75 daily protection contract: " + "; ".join(issues))


def u75_strategy_updates() -> dict[str, Any]:
    from .tcc_u77_static_challenger import tcc_u77_strategy_updates
    result = tcc_u77_strategy_updates()
    result["strategy_mode"] = TCC_U75_DAILY_PROTECTION_MODE
    return result


def _indicators(
    frames: dict[str, pd.DataFrame], symbols: list[str], date: pd.Timestamp,
) -> dict[str, float | int | bool]:
    # Require exactly the same complete effective 75-asset universe. Never
    # interpret missing candidates in missing data as lack of opportunity.
    if len(symbols) != EXPECTED_VALID_UNIVERSE or len(set(symbols)) != EXPECTED_VALID_UNIVERSE:
        raise ValueError("U75 protection requires exactly 75 distinct effective assets.")
    rows = []
    for symbol in symbols:
        df = frames.get(symbol)
        if df is None or "close" not in df:
            raise ValueError(f"U75 missing close history: {symbol}")
        close = pd.to_numeric(df["close"], errors="coerce")
        close = close.loc[close.index <= date].dropna()
        if len(close) < 50 or not np.isfinite(float(close.iloc[-1])) or float(close.iloc[-1]) <= 0:
            continue
        if pd.Timestamp(close.index[-1]) != pd.Timestamp(date):
            continue
        p = float(close.iloc[-1])
        r5 = p / float(close.iloc[-6]) - 1.0
        r20 = p / float(close.iloc[-21]) - 1.0
        ma20 = float(close.iloc[-20:].mean())
        ma50 = float(close.iloc[-50:].mean())
        if not np.isfinite([r5, r20, ma20, ma50]).all() or ma50 <= 0 or ma20 <= 0:
            continue
        rows.append((r5, r20, p < ma50, r5 > 0 and r20 > 0 and p > ma20 and p > ma50))
    eligible = len(rows)
    t = U75Thresholds()
    if eligible < t.min_coverage:
        return {"date": str(date), "valid_assets": eligible,
                "universe_stress": False, "no_technical_candidate": False,
                "stress_without_candidate": False,
                "universe_coverage_complete": False,
                "technical_candidates_count": 0}
    r = np.asarray([item[0] for item in rows], dtype=float)
    m = np.asarray([item[1] for item in rows], dtype=float)
    above5 = float(np.mean(r > 0))
    above20 = float(np.mean(m > 0))
    below_ma50 = float(np.mean([item[2] for item in rows]))
    median = float(np.median(m))
    candidates = int(sum(int(item[3]) for item in rows))
    stress = bool(
        above5 <= t.share_positive_5_max
        and above20 <= t.share_positive_20_max
        and below_ma50 >= t.share_below_ma50_min
        and median <= t.median_return_20_max
    )
    complete = eligible == EXPECTED_VALID_UNIVERSE
    no_candidate = complete and candidates == 0
    return {
        "date": str(date), "valid_assets": eligible,
        "share_positive_5": above5, "share_positive_20": above20,
        "share_below_ma50": below_ma50, "median_return_20": median,
        "technical_candidates_count": candidates,
        "universe_coverage_complete": complete,
        "universe_stress": stress,
        "no_technical_candidate": no_candidate,
        "stress_without_candidate": bool(stress and no_candidate),
    }


def daily_u75_market_signals(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    dates: pd.DatetimeIndex,
) -> dict[pd.Timestamp, dict[str, Any]]:
    index = pd.DatetimeIndex(dates)
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("U75 decision dates must be unique and ordered.")
    # Calculate exactly once per date. No future market data is inspected.
    return {pd.Timestamp(dt): _indicators(frames, symbols, pd.Timestamp(dt))
            for dt in index}


def overlay_u75_policy(
    base_policy: Callable[[pd.Timestamp, int, int], tuple[int, float]],
    signals: dict[pd.Timestamp, dict[str, Any]],
    *,
    diagnostics: dict[pd.Timestamp, dict[str, Any]] | None = None,
) -> Callable[[pd.Timestamp, int, int], tuple[int, float]]:
    """Cash only when the WHOLE U75 is distressed and has no candidate.

    After the signal clears, immediately delegate to the base strategy
    using the true cash/asset position. One action per session.
    """
    def policy(date: pd.Timestamp, position: int, holding: int) -> tuple[int, float]:
        key = pd.Timestamp(date)
        if key not in signals:
            raise KeyError(f"Missing U75 signal for {key}.")
        original = base_policy(key, position, holding)
        signal = signals[key]
        if diagnostics is not None:
            diagnostic = diagnostics.setdefault(key, {})
            diagnostic.update({
                "u75_market_stress": bool(signal["universe_stress"]),
                "u75_effective_coverage": int(signal["valid_assets"]),
                "u75_technical_candidates": int(signal["technical_candidates_count"]),
                "u75_cash_override": bool(signal["stress_without_candidate"]),
            })
        if signal["stress_without_candidate"]:
            if diagnostics is not None:
                diagnostics[key].update({
                    "decision_reason": "U75_COLLECTIVE_STRESS_NO_CANDIDATE",
                    "final_action_asset": "CASH", "final_action_score": 0.0,
                })
            return 0, 0.0
        return original
    return policy


def apply_live_u75_protection(bars: dict[str, pd.DataFrame], decision: Any) -> Any:
    """Protected live decision at the latest COMPLETED session."""
    date = pd.Timestamp(decision.decision_date)
    symbols = [s for s in U77_REQUESTED_ASSETS if s in bars]
    signal = _indicators(bars, symbols, date)
    if not signal["stress_without_candidate"]:
        return decision
    return replace(decision, target_asset="CASH", selected_utility=0.0)
