"""Guardrails for the strategy-neutral operational snapshot interface.

This module never connects to Alpaca or submits orders.
"""
from __future__ import annotations

from inspect import signature

from market_cycle_trader_api.engine import operational_market_snapshot as shared
from market_cycle_trader_api.engine import tcc_u67_operational_market_data as legacy
from market_cycle_trader_api.engine.tcc_u77_static_challenger import U77_REQUESTED_ASSETS
from market_cycle_trader_api.tcc_u67_v1210_reference.contract import U67_REQUESTED_ASSETS


def test_generic_snapshot_interface_is_shared_across_u67_u77() -> None:
    assert len(U67_REQUESTED_ASSETS) == 67
    assert len(U77_REQUESTED_ASSETS) == 77
    assert U77_REQUESTED_ASSETS[:67] == U67_REQUESTED_ASSETS
    parameters = signature(shared.download_operational_market_snapshot).parameters
    assert "requested_assets" in parameters
    # The U67 default remains for old callers, while U77 passes the full tuple.
    assert parameters["requested_assets"].default == U67_REQUESTED_ASSETS


def test_historical_module_remains_compatible() -> None:
    assert legacy.download_u67_operational_snapshot is shared.download_operational_market_snapshot
    assert legacy.append_u67_current_session_intraday is shared.append_current_session_intraday
    assert legacy.U67OperationalSnapshot is shared.OperationalMarketSnapshot
    assert legacy._structural_issue is shared._structural_issue


def test_structural_identity_exclusions_remain_centralized() -> None:
    assert legacy.KNOWN_U67_STRUCTURAL_EXCLUSIONS is shared.KNOWN_OPERATIONAL_STRUCTURAL_EXCLUSIONS
    issue = shared._structural_issue("CLMT", [])
    assert issue is not None and issue["reason"] == "structural_identity_change"
