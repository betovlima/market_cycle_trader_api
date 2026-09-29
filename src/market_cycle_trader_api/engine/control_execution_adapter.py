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


# Predeclared fixed scenarios. No tuning or ranking on OOS results.
SENSITIVITY_SCENARIOS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("cap10_cost", {
        "participation_rate": 0.10, "unlimited_capacity": False,
        "assumed_full_spread_bps": 15.0,
        "assumed_impact_coefficient_bps": 20.0,
    }),
    ("no_cap_no_cost", {
        "participation_rate": 1.0, "unlimited_capacity": True,
        "assumed_full_spread_bps": 0.0,
        "assumed_impact_coefficient_bps": 0.0,
    }),
    ("cap10_no_cost", {
        "participation_rate": 0.10, "unlimited_capacity": False,
        "assumed_full_spread_bps": 0.0,
        "assumed_impact_coefficient_bps": 0.0,
    }),
    ("cap05_cost", {
        "participation_rate": 0.05, "unlimited_capacity": False,
        "assumed_full_spread_bps": 15.0,
        "assumed_impact_coefficient_bps": 20.0,
    }),
    ("cap01_cost", {
        "participation_rate": 0.01, "unlimited_capacity": False,
        "assumed_full_spread_bps": 15.0,
        "assumed_impact_coefficient_bps": 20.0,
    }),
)


def run_sensitivity_lightgbm(
    bars: dict[str, Any],
    config: Any,
    fees: Callable,
    slippage: Callable,
    *,
    progress_callback: Callable | None = None,
):
    """Fit the SAME immutable Control models once; independently replay five
    state-aware accounting paths at the identical OOS fold boundary.

    This works only at the isolated simulator injection point; no monkeypatch
    to any frozen module global or changes to TCC files. Each scenario starts
    from fresh $10k CASH and calls the policy with its own portfolio state.
    """
    if scientific.allocation_execution_enabled(config):
        raise ValueError("Sensitivity requires single-position Control.")
    original = scientific._run_lightgbm
    if original.__globals__.get("_simulate_exact") is not scientific._simulate_exact:
        raise RuntimeError("Unexpected altered frozen research simulator.")
    captured = {}

    def replay_all(*args, **kwargs):
        if captured:
            raise ValueError("Sensitivity requires precisely one Control repetition.")
        base_progress = kwargs.get("simulation_progress_callback")
        primary = None
        count = len(SENSITIVITY_SCENARIOS)
        for index, (name, override) in enumerate(SENSITIVITY_SCENARIOS):
            callback = None
            if base_progress is not None:
                def callback(fraction, stage, *, _index=index, _name=name):
                    base_progress(
                        (_index + max(0.0, min(1.0, float(fraction)))) / count,
                        f"{_name}: {stage}",
                    )
            item = simulate_feasible_control(
                *args,
                **{**kwargs, "simulation_progress_callback": callback,
                   "scenario_override": override},
            )
            captured[name] = item
            if primary is None:
                primary = item
        return primary

    isolated_globals = dict(original.__globals__)
    isolated_globals["_simulate_exact"] = replay_all
    isolated = FunctionType(
        original.__code__, isolated_globals, original.__name__,
        original.__defaults__, original.__closure__,
    )
    isolated.__kwdefaults__ = dict(original.__kwdefaults__ or {})
    results = isolated(
        bars, config, fees, slippage,
        progress_callback=progress_callback,
        trade_callback=None,
        progress_detail_callback=None,
        technical_log_callback=None,
    )
    if (len(results) != 1 or results[0] is not captured.get("cap10_cost")
            or tuple(captured) != tuple(name for name, _ in SENSITIVITY_SCENARIOS)):
        raise ValueError("Sensitivity scientific run did not produce exact declared cases.")
    if original.__globals__.get("_simulate_exact") is not scientific._simulate_exact:
        raise RuntimeError("Frozen simulator binding changed.")
    return captured, results[0]
