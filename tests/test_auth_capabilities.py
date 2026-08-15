from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from market_cycle_trader_api.api.routers.auth import _session_response
from market_cycle_trader_api.auth.capabilities import capabilities_for_role


def test_viewer_capabilities_are_read_only_for_backtest_and_tuning() -> None:
    capabilities = capabilities_for_role("viewer")
    assert capabilities["dashboard.view"] is True
    assert capabilities["backtest.view"] is True
    assert capabilities["tuning.view"] is True
    assert capabilities["research_models.view"] is True
    assert capabilities["analytics.view"] is True
    assert capabilities["backtest.start"] is False
    assert capabilities["backtest.export"] is False
    assert capabilities["tuning.start"] is False
    assert capabilities["tuning.stop"] is False
    assert capabilities["tuning.export"] is False
    assert capabilities["tuning.logs.view"] is False
    assert capabilities["tuning.promote"] is False
    assert capabilities["portfolio.view"] is False
    assert capabilities["administration.view"] is False
    assert capabilities["settings.view"] is False


def test_trader_and_admin_capabilities_match_backend_access_policy() -> None:
    trader = capabilities_for_role("trader")
    admin = capabilities_for_role("admin")

    assert trader["backtest.start"] is True
    assert trader["portfolio.view"] is True
    assert trader["dashboard.strategy_intelligence.view"] is True
    assert trader["tuning.start"] is False
    assert trader["administration.view"] is False

    assert all(admin.values())


def test_session_response_returns_backend_capabilities() -> None:
    identity = SimpleNamespace(
        role="viewer",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        display_name="Read Only",
        email="viewer@example.com",
    )
    response = _session_response(identity)
    assert response.authenticated is True
    assert response.capabilities["backtest.view"] is True
    assert response.capabilities["tuning.view"] is True
    assert response.capabilities["backtest.start"] is False
    assert response.capabilities["portfolio.view"] is False
