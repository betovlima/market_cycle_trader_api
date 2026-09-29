"""A separate, predeclared *research* Control liquidity-aware policy.

The frozen TCC LightGBM model, labels, calibrator, thresholds and original
utility policy are untouched. At a prior completed close, discount each
candidate's full-deployment utility by the fraction of current equity that
could hypothetically pass both the incumbent's exit and candidate's entry
volume forecasts. The original Control policy then applies its unchanged
CASH, min-hold and switch-margin rules on these effective utilities.

The capital-dependent account context is handed to the cached utility view
only at decision time. The execution day's OHLCV is never read by this view.
This is a hypothesis, not a trained liquidity model or broker fill guarantee.
"""
from __future__ import annotations

import math
from types import FunctionType
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..tcc_v106_reference import research_challengers as scientific
from .control_execution_feasibility import SCENARIO, simulate_feasible_control

RESEARCH_STRATEGY_MODE = "MCT_RESEARCH_CONTROL_LIQUIDITY_AWARE_V1"
POLICY_SPEC = {
    "strategy_mode": RESEARCH_STRATEGY_MODE,
    "reference_strategy_mode": "COMPOUND_ROTATION_SWING_LIGHTGBM",
    "participation_rate": 0.10,
    "historical_sessions": 20,
    "minimum_observations": 5,
    "utility_transformation": "full_deployment_utility_times_forecast_entry_and_exit_capacity_fraction",
    "calibrated_on_oos": False,
    "capacity_source": "prior_completed_session_close_and_trailing_volume_only",
}


def _known_capacity_dollars(
    frame: pd.DataFrame, timestamp: pd.Timestamp, spec: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    """Inclusive trailing *completed* dates. Never inspect timestamp+1."""
    key = pd.Timestamp(timestamp)
    location = frame.index.get_indexer([key])
    if len(location) != 1 or int(location[0]) < 0:
        raise ValueError("Liquidity policy decision date missing from canonical panel.")
    i = int(location[0])
    end = i + 1
    prior = pd.to_numeric(
        frame["volume"].iloc[max(0, end - int(spec["historical_sessions"])):end],
        errors="coerce",
    )
    prior = prior[np.isfinite(prior) & (prior >= 0)]
    close = float(frame["close"].iloc[i])
    if not math.isfinite(close) or close <= 0:
        return 0.0, {
            "known_close": close if math.isfinite(close) else 0.0,
            "median_volume": 0.0, "observations": len(prior),
        }
    if len(prior) < int(spec["minimum_observations"]):
        median = 0.0
    else:
        median = float(prior.median())
    if not math.isfinite(median) or median < 0:
        median = 0.0
    capacity = median * float(spec["participation_rate"]) * close
    return capacity, {
        "known_close": close, "median_volume": median,
        "observations": len(prior),
    }


class _CapitalAwareUtilityCache(dict):
    """Same precomputed model predictions; only the returned view changes.

    The original dictionary values are never modified. The Context object
    belongs exclusively to one isolated research job and is inactive during
    the exact reference replay.
    """

    def __init__(self, cached: dict, frames: dict[str, pd.DataFrame],
                 symbols: list[str], account: dict[str, Any]):
        super().__init__(cached)
        self._frames = frames
        self._symbols = tuple(symbols)
        self._account = account

    def get(self, key: Any, default: Any = None):
        key = pd.Timestamp(key)
        original = dict.get(self, key, default)
        if original is default or not self._account.get("enabled", False):
            return original
        if pd.Timestamp(self._account.get("decision_timestamp")) != key:
            raise ValueError("Liquidity policy account context date is stale.")
        raw = np.asarray(original, dtype=np.float64)
        if len(raw) != len(self._symbols) + 1:
            raise ValueError("Frozen utility vector has unexpected asset count.")
        equity = float(self._account["equity"])
        position = int(self._account["position"])
        shares = int(self._account["shares"])
        if not (math.isfinite(equity) and equity > 0 and 0 <= position <= len(self._symbols)):
            raise ValueError("Invalid live research portfolio context.")
        incumbent_exit = 1.0
        if position > 0 and shares > 0:
            symbol = self._symbols[position - 1]
            forecast, details = _known_capacity_dollars(
                self._frames[symbol], key, POLICY_SPEC,
            )
            holding_value = shares * float(details["known_close"])
            incumbent_exit = min(1.0, forecast / holding_value) if holding_value > 0 else 0.0
        effective = raw.copy()
        details_by_symbol = {}
        for i, symbol in enumerate(self._symbols, 1):
            if i == position:
                fraction = 1.0  # No entry/exit required to keep existing shares.
                forecast_dollars = 0.0
            else:
                forecast_dollars, _ = _known_capacity_dollars(
                    self._frames[symbol], key, POLICY_SPEC,
                )
                entry_fraction = min(1.0, forecast_dollars / equity)
                fraction = min(entry_fraction, incumbent_exit)
            if not math.isfinite(fraction) or fraction < 0 or fraction > 1:
                raise AssertionError("Invalid decision-time liquidity fraction.")
            if np.isfinite(raw[i]):
                effective[i] = raw[i] * fraction
            details_by_symbol[symbol] = {
                "capacity_dollars": float(forecast_dollars),
                "capacity_fraction": float(fraction),
                "raw_utility": float(raw[i]) if np.isfinite(raw[i]) else None,
                "effective_utility": (
                    float(effective[i]) if np.isfinite(effective[i]) else None
                ),
            }
        if not math.isclose(float(effective[0]), float(raw[0]), abs_tol=0, rel_tol=0):
            raise AssertionError("Liquidity-aware candidate calculation modified CASH baseline.")
        self._account["audit"][key] = {
            "timestamp": key,
            "equity_at_decision": equity,
            "cash_at_decision": float(self._account["cash"]),
            "shares_at_decision": shares,
            "held_asset_at_decision": (
                self._symbols[position - 1] if position else "CASH"
            ),
            "incumbent_exit_fraction": float(incumbent_exit),
            "raw_best_asset": (
                self._symbols[int(np.argmax(np.where(np.isfinite(raw[1:]), raw[1:], -np.inf)))] if np.isfinite(raw[1:]).any()
                else None
            ),
            "adjusted_best_asset": (
                self._symbols[int(np.argmax(np.where(np.isfinite(effective[1:]), effective[1:], -np.inf)))] if np.isfinite(effective[1:]).any()
                else None
            ),
            "raw_positive_candidate_count": int((raw[1:] > 0).sum()),
            "effective_positive_candidate_count": int((effective[1:] > 0).sum()),
            "effective_candidate_count": int(
                sum(np.isfinite(effective[1:]) & (effective[1:] != 0.0))
            ),
            "candidate_liquidity_detail": details_by_symbol,
        }
        return effective


def run_control_liquidity_pair(
    bars: dict[str, pd.DataFrame],
    config: Any,
    fees: Callable,
    slippage: Callable,
    *,
    progress_callback: Callable | None = None,
) -> tuple[dict[str, Any], Any, dict[pd.Timestamp, dict[str, Any]]]:
    """Fit original Control once; replay old and liquidity-aware policy.

    Both use the same *unchanged* 10%-cap and additional-price-cost execution
    scenario. The old path is first checked against the known v10.8.42 result
    by the orchestrator; any disagreement aborts the research job.
    """
    if scientific.allocation_execution_enabled(config):
        raise ValueError("Liquidity-aware research requires single-position Control.")
    original_runner = scientific._run_lightgbm
    original_predict = scientific._precompute_model_utilities
    if original_runner.__globals__.get("_simulate_exact") is not scientific._simulate_exact:
        raise RuntimeError("Frozen TCC simulator global has been altered.")
    if original_runner.__globals__.get("_precompute_model_utilities") is not original_predict:
        raise RuntimeError("Frozen TCC utility cache binding has been altered.")
    account: dict[str, Any] = {"enabled": False, "audit": {}}
    cache_count = [0]

    def audited_utilities(models, frames, symbols, timestamps, settings):
        cached, profile = original_predict(
            models, frames, symbols, timestamps, settings,
        )
        cache_count[0] += 1
        return _CapitalAwareUtilityCache(cached, frames, symbols, account), profile

    captured: dict[str, Any] = {}
    def replay_pair(*args, **kwargs):
        if captured:
            raise ValueError("Pair experiment requires exactly one Control repetition.")
        callback = kwargs.get("simulation_progress_callback")
        def emit(fraction, stage, *, section: str, offset: float):
            if callback is not None:
                callback(
                    offset + .5 * max(0.0, min(1.0, float(fraction))),
                    f"{section}: {stage}",
                )
        captured["control_reference"] = simulate_feasible_control(
            *args,
            **{**kwargs,
               "simulation_progress_callback": lambda f, s:
               emit(f, s, section="control_reference", offset=0.0)},
        )
        account["enabled"] = True
        account["audit"] = {}
        def prepare(date, position, holding, cash, shares, equity):
            account.update({
                "decision_timestamp": pd.Timestamp(date),
                "position": position, "holding_days": holding,
                "cash": cash, "shares": shares, "equity": equity,
            })
        captured["liquidity_aware"] = simulate_feasible_control(
            *args,
            **{**kwargs, "decision_prepare": prepare,
               "simulation_progress_callback": lambda f, s:
               emit(f, s, section="liquidity_aware", offset=.5)},
        )
        return captured["control_reference"]

    isolated_globals = dict(original_runner.__globals__)
    isolated_globals["_precompute_model_utilities"] = audited_utilities
    isolated_globals["_simulate_exact"] = replay_pair
    isolated = FunctionType(
        original_runner.__code__, isolated_globals, original_runner.__name__,
        original_runner.__defaults__, original_runner.__closure__,
    )
    isolated.__kwdefaults__ = dict(original_runner.__kwdefaults__ or {})
    results = isolated(
        bars, config, fees, slippage,
        progress_callback=progress_callback,
        trade_callback=None,
        progress_detail_callback=None,
        technical_log_callback=None,
    )
    if (
        len(results) != 1
        or results[0] is not captured.get("control_reference")
        or set(captured) != {"control_reference", "liquidity_aware"}
        or cache_count[0] != 3
    ):
        raise ValueError("Expected one unchanged model fit and three fold utility caches.")
    if (
        original_runner.__globals__.get("_precompute_model_utilities") is not original_predict
        or original_runner.__globals__.get("_simulate_exact") is not scientific._simulate_exact
    ):
        raise RuntimeError("Scientific TCC module bindings changed during the research.")
    return captured, results[0], dict(account["audit"])
