from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from ..infrastructure.persistence.mongo_repository import (
    PAPER_TRADING_SETTINGS_COLLECTION,
    PAPER_TRADING_STATE_COLLECTION,
    get_paper_trading_state,
    utc_now,
)
from ..infrastructure.trading.alpaca_paper import (
    account_snapshot,
    assert_account_can_trade,
    clock_snapshot,
    create_unverified_paper_trading_client,
)
from ..schemas.paper_trading import PaperTradingState
from .paper_market_scheduler import arm_next_session, latest_paper_market_run
from .paper_trading import initialize_paper_state
from .strategy_lab import trader_winner_requires_state_reinitialization
from .parameter_bootstrap import (
    bootstrap_missing_parameterizations,
    parameterization_status,
)


def _masked_account(account: dict[str, Any]) -> dict[str, Any]:
    number = str(account.get("account_number") or "")
    return {
        "id": str(account.get("id") or ""),
        "account_number": f"***{number[-4:]}" if number else "",
        "status": account.get("status"),
        "cash": account.get("cash"),
        "equity": account.get("equity"),
        "buying_power": account.get("buying_power"),
        "trading_blocked": account.get("trading_blocked"),
        "account_blocked": account.get("account_blocked"),
        "trade_suspended_by_user": account.get("trade_suspended_by_user"),
    }


def setup_status(db: Any) -> dict[str, Any]:
    parameters = parameterization_status(db)
    state_document = db[PAPER_TRADING_STATE_COLLECTION].find_one({"_id": "default"})
    state_status: dict[str, Any]
    if state_document is None:
        state_status = {"status": "missing", "valid": False}
    else:
        try:
            state = PaperTradingState.model_validate(get_paper_trading_state(db))
            state_status = {
                "status": "ready",
                "valid": True,
                "initial_capital": state.initial_capital,
                "strategy_cash": state.strategy_cash,
                "managed_symbol": state.managed_symbol,
            }
        except (RuntimeError, ValidationError) as exc:
            state_status = {"status": "invalid", "valid": False, "detail": str(exc)}

    account_status: dict[str, Any]
    try:
        client = create_unverified_paper_trading_client()
        account = account_snapshot(client)
        assert_account_can_trade(account)
        settings = db[PAPER_TRADING_SETTINGS_COLLECTION].find_one(
            {"_id": "default"}, {"paper_account_id": 1}
        ) or {}
        bound = str(settings.get("paper_account_id") or "")
        account_status = {
            "status": "ready" if bound == account["id"] else "not_bound",
            "bound": bound == account["id"],
            "account": _masked_account(account),
            "clock": clock_snapshot(client),
        }
    except Exception as exc:
        account_status = {"status": "unavailable", "bound": False, "detail": str(exc)}

    return {
        "parameters": parameters,
        "alpaca_paper": account_status,
        "paper_state": state_status,
        "next_session_run": latest_paper_market_run(db),
    }


def initialize_application(db: Any, *, arm_market: bool) -> dict[str, Any]:
    bootstrap = bootstrap_missing_parameterizations(db, source="admin-setup-api")
    invalid = [item for item in bootstrap["results"] if not item["valid"]]
    if invalid:
        raise RuntimeError(
            "One or more existing parameter documents are invalid and were preserved: "
            + "; ".join(f"{item['collection']}/{item['document_id']}: {item['message']}" for item in invalid)
        )

    client = create_unverified_paper_trading_client()
    account = account_snapshot(client)
    assert_account_can_trade(account)
    actual_id = str(account["id"])
    now = utc_now()
    db[PAPER_TRADING_SETTINGS_COLLECTION].update_one(
        {"_id": "default"},
        {"$set": {"paper_account_id": actual_id, "updated_at": now}},
    )

    existing_state = db[PAPER_TRADING_STATE_COLLECTION].find_one({"_id": "default"})
    requires_winner_reset = trader_winner_requires_state_reinitialization(db)
    if existing_state is None:
        state = initialize_paper_state(db, replace=False)
        state_action = "initialized"
    elif requires_winner_reset:
        state = initialize_paper_state(db, replace=True)
        state_action = "reinitialized_for_promoted_winner"
    else:
        validated = PaperTradingState.model_validate(get_paper_trading_state(db))
        state = validated.model_dump(mode="python")
        state_action = "preserved_existing_valid"

    market_run = None
    if arm_market:
        current = latest_paper_market_run(db)
        if current and current.get("status") in {
            "armed", "preparing", "prepared", "executing"
        }:
            market_run = current
        else:
            market_run = arm_next_session(db)

    return {
        "status": "ready",
        "bootstrap": bootstrap,
        "alpaca_paper": {
            "bound": True,
            "account": _masked_account(account),
            "clock": clock_snapshot(client),
        },
        "paper_state": {"action": state_action, **state},
        "next_session_run": market_run,
    }
