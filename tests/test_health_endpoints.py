from __future__ import annotations

from market_cycle_trader_api.api.routers import health
from market_cycle_trader_api.core import runtime


def test_liveness_is_independent_from_mongodb_readiness(monkeypatch) -> None:
    monkeypatch.setitem(runtime.MONGO_STATUS, "available", False)
    monkeypatch.setitem(runtime.MONGO_STATUS, "configuration_available", False)

    response = health.liveness()

    assert response["status"] == "ok"


def test_readiness_remains_strict(monkeypatch) -> None:
    monkeypatch.setitem(runtime.MONGO_STATUS, "available", False)
    monkeypatch.setitem(runtime.MONGO_STATUS, "configuration_available", False)

    response = health.readiness()

    assert response.status_code == 503
