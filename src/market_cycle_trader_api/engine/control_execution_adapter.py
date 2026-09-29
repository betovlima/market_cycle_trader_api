"""Isolated MCT-only substitution of Control portfolio accounting.

The vendored TCC v1.0.6 module has a strict source-blob regression contract.
This adapter clones the *function object* for its LightGBM research runner
with an independent global symbol mapping. No vendored file, global module
variable, or original function object is modified. Every scientific fold,
calibration, model fit and policy function remains the original reference.
Only the final OOS portfolio simulator is replaced in this isolated call.
"""
from __future__ import annotations

from types import FunctionType
from typing import Any, Callable

from ..tcc_v106_reference import research_challengers as scientific
from .control_execution_feasibility import simulate_feasible_control


def run_feasible_lightgbm(
    bars: dict[str, Any],
    config: Any,
    fees: Callable,
    slippage: Callable,
    *,
    progress_callback: Callable | None = None,
):
    if scientific.allocation_execution_enabled(config):
        raise ValueError(
            "Execution feasibility requires the single-position Control policy."
        )
    original = scientific._run_lightgbm
    if original.__globals__.get("_simulate_exact") is not scientific._simulate_exact:
        raise RuntimeError("Unexpected modified scientific simulator binding.")
    isolated_globals = dict(original.__globals__)
    isolated_globals["_simulate_exact"] = simulate_feasible_control
    isolated = FunctionType(
        original.__code__, isolated_globals,
        original.__name__, original.__defaults__, original.__closure__,
    )
    isolated.__kwdefaults__ = dict(original.__kwdefaults__ or {})
    result = isolated(
        bars, config, fees, slippage,
        progress_callback=progress_callback,
        trade_callback=None,
        progress_detail_callback=None,
        technical_log_callback=None,
    )
    if original.__globals__.get("_simulate_exact") is not scientific._simulate_exact:
        raise RuntimeError("Scientific simulator binding changed unexpectedly.")
    return result
