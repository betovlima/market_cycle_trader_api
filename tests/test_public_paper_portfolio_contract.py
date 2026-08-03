from market_cycle_trader_api.services.public_paper_portfolio import _safe_order


def test_public_order_omits_internal_identifiers() -> None:
    result = _safe_order(
        {
            "client_order_id": "secret-client-id",
            "plan_id": "secret-plan-id",
            "symbol": "AAPL",
            "side": "buy",
            "status": "filled",
            "quantity": 1.0,
        }
    )
    assert result == {
        "symbol": "AAPL",
        "side": "buy",
        "status": "filled",
        "quantity": 1.0,
    }
