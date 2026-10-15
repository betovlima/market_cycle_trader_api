"""Live/Paper isolation guardrails (no real account network calls)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from market_cycle_trader_api.infrastructure.trading.alpaca_live import (
    live_credentials,
)
from market_cycle_trader_api.services import live_portfolio


def test_live_creds_are_never_taken_from_paper_environment(monkeypatch):
    monkeypatch.delenv("ALPACA_LIVE_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_LIVE_SECRET_KEY", raising=False)
    monkeypatch.setenv("ALPACA_API_KEY_ID", "paper-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "paper-secret")
    with pytest.raises(RuntimeError, match="ALPACA_LIVE"):
        live_credentials()


def test_live_creds_must_differ_from_paper(monkeypatch):
    monkeypatch.setenv("ALPACA_LIVE_API_KEY_ID", "paper-key")
    monkeypatch.setenv("ALPACA_LIVE_SECRET_KEY", "live-secret")
    monkeypatch.setenv("ALPACA_API_KEY_ID", "paper-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "paper-secret")
    with pytest.raises(RuntimeError, match="different"):
        live_credentials()


def test_live_broker_fill_never_implies_decision_or_realized_pnl():
    order = SimpleNamespace(
        client_order_id="live-order-1", symbol="CAPR", side="buy",
        status="filled", qty="3", filled_qty="3", filled_avg_price="6.5",
        submitted_at=None, created_at=None, filled_at=None,
    )
    result = live_portfolio._order(order)
    assert result["filled_value"] == 19.5
    assert result["decision_audit"] is None
    assert result["realized_pnl"] is None
    assert result["execution_origin"] == "live_broker_history"


def test_live_robot_is_explicitly_readonly(monkeypatch):
    monkeypatch.setattr(
        live_portfolio, "get_trader_winner_summary",
        lambda db: {"name": "Strategy #30"},
    )
    response = live_portfolio.live_robot_status(object())
    assert response["execution_enabled"] is False
    assert response["status"] == "stopped"
    assert response["scheduler_alive"] is False
    assert response["trader_winner"]["name"] == "Strategy #30"
    assert response["active_run"] is None
