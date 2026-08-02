from __future__ import annotations

import math
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from pydantic import ValidationError
from pymongo import ReturnDocument

from ..engine.live_xgboost_signal import build_live_xgboost_decision
from ..engine.market_data import load_market_bars, validate_and_clean_bars
from ..infrastructure.persistence.mongo_repository import (
    PAPER_TRADE_ORDERS_COLLECTION,
    PAPER_TRADE_PLANS_COLLECTION,
    PAPER_TRADING_STATE_COLLECTION,
    bson_value,
    get_paper_trading_settings,
    get_paper_trading_state,
    get_settings,
    insert_paper_trade_order,
    insert_paper_trade_plan,
    replace_paper_trading_state,
    update_paper_trade_order,
    update_paper_trade_plan,
    utc_now,
)
from ..infrastructure.trading.alpaca_paper import (
    account_snapshot,
    assert_account_can_trade,
    asset_snapshot,
    calendar_session_dates,
    clock_snapshot,
    create_paper_trading_client,
    open_order_snapshots,
    position_snapshots,
    submit_market_buy_notional,
    submit_market_sell,
    wait_for_order,
)
from ..schemas.paper_trading import PaperTradePlan, PaperTradingSettings, PaperTradingState
from ..schemas.requests import BacktestRequest

EASTERN = ZoneInfo("America/New_York")


def _iso(value: Any) -> str:
    return pd.Timestamp(value).isoformat()


def _et_date(value: Any) -> str:
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return stamp.tz_convert(EASTERN).date().isoformat()


def _validated_context(db: Any) -> tuple[BacktestRequest, PaperTradingSettings, PaperTradingState]:
    try:
        strategy = BacktestRequest.model_validate(get_settings(db))
        settings = PaperTradingSettings.model_validate(get_paper_trading_settings(db))
        state = PaperTradingState.model_validate(get_paper_trading_state(db))
    except (RuntimeError, ValidationError) as exc:
        raise RuntimeError(f"Paper-trading configuration is unavailable or invalid: {exc}") from exc

    if not settings.enabled:
        raise RuntimeError("Paper trading is disabled in MongoDB.")
    if strategy.strategy_mode != "COMPOUND_ROTATION_SWING_XGBOOST":
        raise RuntimeError("Paper trading requires the locked XGBoost swing strategy.")
    if strategy.rotation_models != ["xgboost_utility"]:
        raise RuntimeError("Paper trading requires rotation_models=['xgboost_utility'].")
    if strategy.market_data_provider != "alpaca":
        raise RuntimeError("Paper trading requires market_data_provider='alpaca'.")
    if strategy.end_date is not None:
        raise RuntimeError(
            "Paper trading requires the locked historical end_date to be empty so the latest completed session is loaded."
        )
    if strategy.whole_shares:
        raise RuntimeError("Paper trading requires fractional shares for the isolated strategy sleeve.")
    if not math.isclose(
        float(state.initial_capital),
        float(strategy.initial_capital),
        rel_tol=0,
        abs_tol=0.01,
    ):
        raise RuntimeError(
            "Paper state initial capital differs from the locked strategy capital: "
            f"state={state.initial_capital:.2f}, locked={strategy.initial_capital:.2f}."
        )
    return strategy, settings, state



def paper_market_readiness(db: Any) -> dict[str, Any]:
    """Validate every dependency required to arm next-session paper execution."""

    strategy, settings, state = _validated_context(db)
    client = create_paper_trading_client(db)
    account = account_snapshot(client)
    assert_account_can_trade(account)
    _assert_no_conflicting_orders(client, assets=strategy.assets)
    _reconcile_state_with_account(
        client,
        assets=strategy.assets,
        state=state,
    )
    clock = clock_snapshot(client)
    if clock.get("timestamp") is None or clock.get("next_open") is None:
        raise RuntimeError("Alpaca market clock did not return timestamp and next_open.")
    return {
        "clock": clock,
        "settings": settings.model_dump(mode="python"),
        "strategy_cash": float(state.strategy_cash),
        "managed_symbol": state.managed_symbol,
        "holding_sessions": int(state.holding_sessions),
        "paper_account_id": account["id"],
    }

def _trim_incomplete_daily_session(
    frame: pd.DataFrame,
    *,
    clock: dict[str, Any],
) -> pd.DataFrame:
    if frame.empty:
        return frame
    timestamp = pd.Timestamp(clock["timestamp"])
    timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
    next_open = pd.Timestamp(clock["next_open"])
    next_open = next_open.tz_localize("UTC") if next_open.tzinfo is None else next_open.tz_convert("UTC")

    # During regular hours, and before today's open, today's daily candle is not
    # a completed decision candle and must never be used by the model.
    incomplete_session: str | None = None
    if bool(clock["is_open"]) or timestamp.tz_convert(EASTERN).date() == next_open.tz_convert(EASTERN).date():
        incomplete_session = timestamp.tz_convert(EASTERN).date().isoformat()

    if incomplete_session is None:
        return frame

    session_dates = pd.DatetimeIndex(frame.index).tz_convert(EASTERN).date
    mask = [item.isoformat() != incomplete_session for item in session_dates]
    return frame.loc[mask]


def _assert_no_conflicting_orders(
    client: Any,
    *,
    assets: list[str],
) -> None:
    open_orders = open_order_snapshots(client, assets)
    if open_orders:
        details = ", ".join(
            f"{item['symbol']}:{item['side']}:{item['status']}:{item['client_order_id']}"
            for item in open_orders
        )
        raise RuntimeError(
            "Open Alpaca orders already exist for the locked asset universe. "
            f"Resolve them before continuing: {details}"
        )


def _reconcile_state_with_account(
    client: Any,
    *,
    assets: list[str],
    state: PaperTradingState,
) -> dict[str, dict[str, Any]]:
    positions = position_snapshots(client)
    strategy_positions = {
        symbol: item
        for symbol, item in positions.items()
        if symbol in set(assets) and abs(float(item["quantity"])) > 1e-9
    }

    if state.managed_symbol is None:
        if strategy_positions:
            raise RuntimeError(
                "The Alpaca paper account has positions in the strategy universe, "
                "but the strategy state is in cash. Use a clean paper account or "
                "reinitialize the state after closing those positions."
            )
        return positions

    unmanaged = sorted(set(strategy_positions) - {state.managed_symbol})
    if unmanaged:
        raise RuntimeError(
            "The paper account contains unmanaged strategy positions: "
            + ", ".join(unmanaged)
        )
    actual = strategy_positions.get(state.managed_symbol)
    if actual is None:
        raise RuntimeError(
            f"MongoDB says the strategy owns {state.managed_symbol}, but Alpaca has no such position."
        )
    if str(actual.get("side") or "").lower() not in {"long", ""}:
        raise RuntimeError("Short positions are not supported by this paper strategy.")
    actual_quantity = float(actual["quantity"])
    tolerance = max(1e-6, abs(float(state.managed_quantity)) * 1e-6)
    if abs(actual_quantity - float(state.managed_quantity)) > tolerance:
        raise RuntimeError(
            "Managed quantity differs between MongoDB and Alpaca: "
            f"symbol={state.managed_symbol}, state={state.managed_quantity:.9f}, "
            f"alpaca={actual_quantity:.9f}."
        )
    return positions


def initialize_paper_state(db: Any, *, replace: bool = False) -> dict[str, Any]:
    strategy = BacktestRequest.model_validate(get_settings(db))
    settings = PaperTradingSettings.model_validate(get_paper_trading_settings(db))
    if not settings.enabled:
        raise RuntimeError("Paper trading is disabled in MongoDB.")
    if strategy.strategy_mode != "COMPOUND_ROTATION_SWING_XGBOOST":
        raise RuntimeError("The locked strategy is not the XGBoost swing strategy.")

    existing = db[PAPER_TRADING_STATE_COLLECTION].find_one({"_id": "default"})
    if existing is not None and not replace:
        raise RuntimeError(
            "Paper-trading state already exists. Use --replace only after confirming "
            "the strategy has no managed position or pending order."
        )

    client = create_paper_trading_client(db)
    account = account_snapshot(client)
    assert_account_can_trade(account)
    if float(account["cash"]) + 1e-9 < float(strategy.initial_capital):
        raise RuntimeError(
            "The Alpaca paper account does not have enough cash to initialize the strategy sleeve: "
            f"cash=${account['cash']:,.2f}, required=${strategy.initial_capital:,.2f}."
        )

    _assert_no_conflicting_orders(client, assets=strategy.assets)
    positions = position_snapshots(client)
    conflicts = sorted(
        symbol
        for symbol, item in positions.items()
        if symbol in set(strategy.assets) and abs(float(item["quantity"])) > 1e-9
    )
    if conflicts:
        raise RuntimeError(
            "Close existing positions in the locked strategy universe before initialization: "
            + ", ".join(conflicts)
        )

    for symbol in strategy.assets:
        asset = asset_snapshot(client, symbol)
        if not asset["tradable"]:
            raise RuntimeError(f"{symbol} is not tradable on the Alpaca paper account.")
        if not asset["fractionable"]:
            raise RuntimeError(f"{symbol} does not support fractional paper orders.")

    state = PaperTradingState(
        initial_capital=float(strategy.initial_capital),
        strategy_cash=float(strategy.initial_capital),
        managed_symbol=None,
        managed_quantity=0.0,
        average_entry_price=None,
        holding_sessions=0,
        realized_pnl=0.0,
        last_decision_date=None,
        last_execution_session=None,
    )
    replace_paper_trading_state(db, state.model_dump(mode="python"))
    return {
        "paper_account_id": account["id"],
        "account_cash": account["cash"],
        "strategy_initial_capital": state.initial_capital,
        "strategy_cash": state.strategy_cash,
        "managed_symbol": None,
    }


def prepare_next_paper_plan(db: Any, *, replace: bool = False) -> dict[str, Any]:
    strategy, settings, state = _validated_context(db)
    client = create_paper_trading_client(db)
    account = account_snapshot(client)
    assert_account_can_trade(account)
    clock = clock_snapshot(client)
    if bool(clock["is_open"]):
        raise RuntimeError(
            "Prepare the next-open decision only while the regular market is closed, "
            "so the latest daily candle is complete."
        )

    _assert_no_conflicting_orders(client, assets=strategy.assets)
    _reconcile_state_with_account(
        client,
        assets=strategy.assets,
        state=state,
    )

    bars_by_symbol: dict[str, pd.DataFrame] = {}
    for symbol in strategy.assets:
        bars = validate_and_clean_bars(load_market_bars(symbol, strategy), strategy)
        bars = _trim_incomplete_daily_session(bars, clock=clock)
        if bars.empty:
            raise RuntimeError(f"No completed daily bars are available for {symbol}.")
        bars_by_symbol[symbol] = bars

    decision = build_live_xgboost_decision(
        bars_by_symbol,
        strategy,
        current_asset=state.managed_symbol,
        holding_sessions=state.holding_sessions,
    )
    decision_date = _et_date(decision.decision_date)
    expected_open = pd.Timestamp(clock["next_open"])
    expected_open = expected_open.tz_localize("UTC") if expected_open.tzinfo is None else expected_open.tz_convert("UTC")
    execution_session = expected_open.tz_convert(EASTERN).date().isoformat()

    calendar_start = (expected_open.tz_convert(EASTERN).date() - timedelta(days=14))
    calendar_end = expected_open.tz_convert(EASTERN).date()
    sessions = calendar_session_dates(
        client,
        start_date=calendar_start,
        end_date=calendar_end,
    )
    completed_sessions = [item for item in sessions if item < execution_session]
    if not completed_sessions:
        raise RuntimeError("Alpaca calendar did not return a completed session before the next open.")
    expected_decision_date = completed_sessions[-1]
    if decision_date != expected_decision_date:
        raise RuntimeError(
            "The latest aligned daily candle is not the most recent completed Alpaca session: "
            f"data={decision_date}, expected={expected_decision_date}. "
            "Wait for the daily market-data bar/cache to refresh before preparing the order plan."
        )

    current = decision.current_asset
    target = decision.target_asset
    if current == target and target == "CASH":
        action = "stay_in_cash"
    elif current == target:
        action = "hold"
    elif current == "CASH":
        action = "buy"
    elif target == "CASH":
        action = "sell_to_cash"
    else:
        action = "rotate"

    plan_id = f"xgb-{decision_date}-{uuid.uuid4().hex[:8]}"
    plan = PaperTradePlan(
        plan_id=plan_id,
        status="prepared",
        decision_date=decision_date,
        expected_market_open=expected_open.isoformat(),
        execution_session=execution_session,
        current_asset=current,
        target_asset=target,
        action=action,
        random_state=decision.random_state,
        effective_switch_margin=decision.effective_switch_margin,
        calibrated_candidate_margin=decision.calibrated_candidate_margin,
        calibration_score=decision.calibration_score,
        selected_utility=decision.selected_utility,
        utilities=decision.utilities,
        training_end=_et_date(decision.training_end),
        calibration_start=_et_date(decision.calibration_start),
        calibration_end=_et_date(decision.calibration_end),
        final_fit_end=_et_date(decision.final_fit_end),
        state_snapshot=state.model_dump(mode="python"),
        created_at=utc_now().isoformat(),
    )
    document = {
        **plan.model_dump(mode="python"),
        "raw_best_asset": decision.raw_best_asset,
        "effective_compute_device": decision.effective_compute_device,
        "compute_fallback_reason": decision.compute_fallback_reason,
        "paper_account_id": account["id"],
    }
    insert_paper_trade_plan(db, document, replace=replace)
    return document


def _client_order_id(prefix: str, plan_id: str, side: str, symbol: str) -> str:
    compact_plan = plan_id.replace("_", "-")[-18:]
    value = f"{prefix}-{compact_plan}-{side.lower()}-{symbol.lower()}"
    return value[:48]


def _record_submitted_order(
    db: Any,
    *,
    plan_id: str,
    client_order_id: str,
    symbol: str,
    side: str,
    requested_quantity: float | None,
    requested_notional: float | None,
) -> None:
    insert_paper_trade_order(
        db,
        {
            "plan_id": plan_id,
            "client_order_id": client_order_id,
            "symbol": symbol,
            "side": side,
            "requested_quantity": requested_quantity,
            "requested_notional": requested_notional,
            "status": "submitting",
        },
    )


def _execute_sell(
    db: Any,
    client: Any,
    *,
    plan: dict[str, Any],
    settings: PaperTradingSettings,
    state: PaperTradingState,
) -> PaperTradingState:
    if state.managed_symbol is None or state.managed_quantity <= 0:
        return state
    client_order_id = _client_order_id(
        settings.client_order_id_prefix,
        plan["plan_id"],
        "sell",
        state.managed_symbol,
    )
    _record_submitted_order(
        db,
        plan_id=plan["plan_id"],
        client_order_id=client_order_id,
        symbol=state.managed_symbol,
        side="sell",
        requested_quantity=float(state.managed_quantity),
        requested_notional=None,
    )
    order = submit_market_sell(
        client,
        symbol=state.managed_symbol,
        quantity=float(state.managed_quantity),
        client_order_id=client_order_id,
    )
    order_id = str(getattr(order, "id", ""))
    update_paper_trade_order(
        db,
        client_order_id,
        {"order_id": order_id, "status": str(getattr(getattr(order, "status", ""), "value", getattr(order, "status", "")))},
    )
    fill = wait_for_order(
        client,
        order_id=order_id,
        timeout_seconds=settings.order_fill_timeout_seconds,
        poll_interval_seconds=settings.order_poll_interval_seconds,
    )
    update_paper_trade_order(
        db,
        client_order_id,
        {
            "status": fill.status,
            "filled_quantity": fill.filled_quantity,
            "filled_average_price": fill.filled_average_price,
            "filled_notional": fill.filled_notional,
            "filled_at": fill.filled_at,
        },
    )

    sold = min(float(state.managed_quantity), float(fill.filled_quantity))
    realized = 0.0
    if sold > 0 and fill.filled_average_price is not None:
        realized = sold * (float(fill.filled_average_price) - float(state.average_entry_price or 0))
    remaining = max(0.0, float(state.managed_quantity) - sold)
    updated = state.model_copy(
        update={
            "strategy_cash": float(state.strategy_cash) + float(fill.filled_notional),
            "managed_quantity": remaining,
            "realized_pnl": float(state.realized_pnl) + realized,
            "managed_symbol": state.managed_symbol if remaining > 1e-8 else None,
            "average_entry_price": state.average_entry_price if remaining > 1e-8 else None,
            "holding_sessions": state.holding_sessions if remaining > 1e-8 else 0,
        }
    )
    replace_paper_trading_state(db, updated.model_dump(mode="python"))
    if fill.status != "filled" or remaining > 1e-6:
        raise RuntimeError(
            "The paper sell order was not completely filled: "
            f"status={fill.status}, filled={fill.filled_quantity:.9f}, remaining={remaining:.9f}."
        )
    return PaperTradingState.model_validate(updated.model_dump(mode="python"))


def _execute_buy(
    db: Any,
    client: Any,
    *,
    plan: dict[str, Any],
    settings: PaperTradingSettings,
    state: PaperTradingState,
    target_symbol: str,
) -> PaperTradingState:
    account = account_snapshot(client)
    assert_account_can_trade(account)
    notional = max(0.0, float(state.strategy_cash) - float(settings.cash_reserve_dollars))
    if float(account["cash"]) + 0.01 < notional:
        raise RuntimeError(
            "The Alpaca paper account cash is below the strategy sleeve cash. "
            f"account_cash=${account['cash']:,.2f}, strategy_order=${notional:,.2f}."
        )
    notional = math.floor(notional * 100.0) / 100.0
    if notional < 1.0:
        raise RuntimeError(
            "The strategy sleeve does not have enough cash for a new fractional order: "
            f"strategy_cash=${state.strategy_cash:,.2f}, reserve=${settings.cash_reserve_dollars:,.2f}."
        )

    client_order_id = _client_order_id(
        settings.client_order_id_prefix,
        plan["plan_id"],
        "buy",
        target_symbol,
    )
    _record_submitted_order(
        db,
        plan_id=plan["plan_id"],
        client_order_id=client_order_id,
        symbol=target_symbol,
        side="buy",
        requested_quantity=None,
        requested_notional=notional,
    )
    order = submit_market_buy_notional(
        client,
        symbol=target_symbol,
        notional=notional,
        client_order_id=client_order_id,
    )
    order_id = str(getattr(order, "id", ""))
    update_paper_trade_order(
        db,
        client_order_id,
        {"order_id": order_id, "status": str(getattr(getattr(order, "status", ""), "value", getattr(order, "status", "")))},
    )
    fill = wait_for_order(
        client,
        order_id=order_id,
        timeout_seconds=settings.order_fill_timeout_seconds,
        poll_interval_seconds=settings.order_poll_interval_seconds,
    )
    update_paper_trade_order(
        db,
        client_order_id,
        {
            "status": fill.status,
            "filled_quantity": fill.filled_quantity,
            "filled_average_price": fill.filled_average_price,
            "filled_notional": fill.filled_notional,
            "filled_at": fill.filled_at,
        },
    )

    if fill.filled_quantity > 0 and fill.filled_average_price is not None:
        updated = state.model_copy(
            update={
                "strategy_cash": max(0.0, float(state.strategy_cash) - float(fill.filled_notional)),
                "managed_symbol": target_symbol,
                "managed_quantity": float(fill.filled_quantity),
                "average_entry_price": float(fill.filled_average_price),
                "holding_sessions": 1,
            }
        )
        replace_paper_trading_state(db, updated.model_dump(mode="python"))
        state = PaperTradingState.model_validate(updated.model_dump(mode="python"))

    if fill.status != "filled" or fill.filled_quantity <= 0:
        raise RuntimeError(
            "The paper buy order was not completely filled: "
            f"status={fill.status}, filled={fill.filled_quantity:.9f}."
        )
    return state


def execute_prepared_paper_plan(db: Any, *, plan_id: str | None = None) -> dict[str, Any]:
    strategy, settings, state = _validated_context(db)
    query: dict[str, Any]
    if plan_id:
        query = {"plan_id": plan_id}
    else:
        query = {"status": "prepared"}
    plan = db[PAPER_TRADE_PLANS_COLLECTION].find_one(query, sort=[("created_at", -1)])
    if plan is None:
        raise RuntimeError("No prepared paper-trading plan was found.")
    if plan.get("status") == "executed":
        return {key: bson_value(value) for key, value in plan.items() if key != "_id"}
    if plan.get("status") != "prepared":
        raise RuntimeError(f"Paper plan is not executable: status={plan.get('status')}.")

    client = create_paper_trading_client(db)
    account = account_snapshot(client)
    assert_account_can_trade(account)
    clock = clock_snapshot(client)
    if not bool(clock["is_open"]):
        raise RuntimeError(
            f"The Alpaca regular market is closed. Next open: {_iso(clock['next_open'])}."
        )

    now = pd.Timestamp(clock["timestamp"])
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    expected_open = pd.Timestamp(plan["expected_market_open"])
    expected_open = expected_open.tz_localize("UTC") if expected_open.tzinfo is None else expected_open.tz_convert("UTC")
    earliest = expected_open + pd.Timedelta(settings.market_open_delay_seconds, unit="s")
    if now < earliest:
        raise RuntimeError(
            "The configured opening delay has not elapsed: "
            f"now={now.isoformat()}, earliest={earliest.isoformat()}."
        )
    if now.tz_convert(EASTERN).date().isoformat() != str(plan["execution_session"]):
        raise RuntimeError(
            "The prepared plan is stale for the current market session: "
            f"plan_session={plan['execution_session']}, current_session={now.tz_convert(EASTERN).date().isoformat()}."
        )

    claimed = db[PAPER_TRADE_PLANS_COLLECTION].find_one_and_update(
        {"plan_id": plan["plan_id"], "status": "prepared"},
        {"$set": {"status": "executing", "execution_started_at": utc_now(), "updated_at": utc_now()}},
        return_document=ReturnDocument.AFTER,
    )
    if claimed is None:
        raise RuntimeError("The paper plan was already claimed by another executor.")
    plan = claimed

    try:
        _assert_no_conflicting_orders(client, assets=strategy.assets)
        _reconcile_state_with_account(client, assets=strategy.assets, state=state)

        target = str(plan["target_asset"]).upper()
        current = state.managed_symbol or "CASH"
        if current != str(plan["current_asset"]).upper():
            raise RuntimeError(
                "The prepared plan current asset no longer matches the strategy state: "
                f"plan={plan['current_asset']}, state={current}."
            )

        if current != target and current != "CASH":
            state = _execute_sell(
                db,
                client,
                plan=plan,
                settings=settings,
                state=state,
            )

        if target != "CASH" and (state.managed_symbol or "CASH") != target:
            state = _execute_buy(
                db,
                client,
                plan=plan,
                settings=settings,
                state=state,
                target_symbol=target,
            )
        elif target != "CASH" and state.managed_symbol == target:
            state = state.model_copy(update={"holding_sessions": state.holding_sessions + 1})
        elif target == "CASH":
            state = state.model_copy(update={"holding_sessions": 0})

        state = state.model_copy(
            update={
                "last_decision_date": str(plan["decision_date"]),
                "last_execution_session": str(plan["execution_session"]),
            }
        )
        replace_paper_trading_state(db, state.model_dump(mode="python"))
        report = {
            "status": "executed",
            "executed_at": utc_now(),
            "final_state": state.model_dump(mode="python"),
            "account": account_snapshot(client),
        }
        update_paper_trade_plan(db, plan["plan_id"], report)
        return {
            "plan_id": plan["plan_id"],
            "action": plan["action"],
            "target_asset": target,
            **bson_value(report),
        }
    except Exception as exc:
        update_paper_trade_plan(
            db,
            plan["plan_id"],
            {
                "status": "failed",
                "failed_at": utc_now(),
                "error": str(exc),
            },
        )
        raise
