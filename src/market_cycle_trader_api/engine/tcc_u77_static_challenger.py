"""Immutable U67+10 static Challenger contract, isolated from the U67 Winner.

Research/backtest only until independently validated live runtime is delivered.
"""
from __future__ import annotations

from typing import Any

from ..core.config import TCC_U67_CONTROL_OPERATIONAL_MODE, TCC_U77_FIXED_CHALLENGER_MODE
from ..tcc_u67_v1210_reference.contract import U67_REQUESTED_ASSETS
from .tcc_u67_operational_runtime import (
    tcc_u67_contract_issues,
    tcc_u67_strategy_updates,
)

SOURCE_REPOSITORY = "betovlima/tcc_mba_usp_data_science_analytics"
SOURCE_COMMIT = "0719366790c56bfe3935aaa556fe45211b9965da"
SOURCE_TAG = "research-u67plus10-challenger-v1.4.1"
SOURCE_VERSION = "u67-asset-expansion-v1.4.1"
REFERENCE_END_DATE = "2026-10-06"
REFERENCE_CAPITAL = 587_165_941.1330212
STATIC_ADDITIONS = (
    "CRT", "NBR", "CAPR", "PNBK", "PENN",
    "JBL", "STAG", "MCHB", "LAD", "TOWN",
)
U77_REQUESTED_ASSETS = (*U67_REQUESTED_ASSETS, *STATIC_ADDITIONS)
BACKTEST_ENGINE_BINDING = "tcc_u77_fixed_challenger_backtest"
REFERENCE_ENGINE_ID = "tcc-u67-plus10-v1.4.1-fixed-control"


def tcc_u77_strategy_updates() -> dict[str, Any]:
    updates = tcc_u67_strategy_updates()
    updates["assets"] = list(U77_REQUESTED_ASSETS)
    updates["strategy_mode"] = TCC_U77_FIXED_CHALLENGER_MODE
    return updates


def tcc_u77_contract_issues(config: Any) -> list[str]:
    source = (
        config.model_dump(mode="python")
        if hasattr(config, "model_dump")
        else dict(config)
        if isinstance(config, dict)
        else dict(vars(config))
    )
    issues: list[str] = []
    if source.get("strategy_mode") != TCC_U77_FIXED_CHALLENGER_MODE:
        issues.append("strategy_mode: expected fixed U77 Challenger mode")
    if tuple(source.get("assets") or ()) != U77_REQUESTED_ASSETS:
        issues.append("assets: expected exact U67+10 scientific order")
    # Verify every inherited financial/market-data invariant without allowing
    # a modification to the independently protected U67 contract.
    inherited = dict(source)
    inherited["strategy_mode"] = TCC_U67_CONTROL_OPERATIONAL_MODE
    inherited["assets"] = list(U67_REQUESTED_ASSETS)
    issues.extend(tcc_u67_contract_issues(inherited))
    return issues


def assert_tcc_u77_contract(config: Any) -> None:
    issues = tcc_u77_contract_issues(config)
    if issues:
        raise ValueError(
            "TCC U77 fixed Challenger contract mismatch: "
            + "; ".join(issues)
        )
