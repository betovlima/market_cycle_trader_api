from __future__ import annotations

import math
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from pydantic import ValidationError
from pymongo import ReturnDocument

from ..core.config import RESEARCH_ONLY_SWING_STRATEGY_MODES, SWING_STRATEGY_MODES
from ..engine.live_model_signal import build_live_model_decision
from ..engine.market_data import (
    latest_safe_completed_xnys_session,
    load_market_bars,
    refresh_market_data_to_live_cutoff,
    validate_and_clean_bars,
)
from ..infrastructure.persistence.mongo_repository import (
    PAPER_TRADE_ORDERS_COLLECTION,
    PAPER_TRADE_PLANS_COLLECTION,
    PAPER_TRADING_STATE_COLLECTION,
    STRATEGY_CONTROL_COLLECTION,
    bson_value,
    get_paper_trading_settings,
    get_paper_trading_state,
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
from .model_research import apply_execution_profile
from .system_settings import apply_training_runtime_settings, get_system_settings
from .strategy_lab import (
    get_trader_winner_context,
    get_trader_winner_model_snapshot,
    mark_trader_winner_state_initialized,
    trader_winner_requires_state_reinitialization,
    update_trader_live_market_cutoff,
)

EASTERN = ZoneInfo("America/New_York")


def _iso(value: Any) -> str:
    return pd.Timestamp(value).isoformat()


def _et_date(value: Any) -> str:
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return stamp.tz_convert(EASTERN).date().isoformat()


def _acquire_live_market_refresh_lock(db: Any, *, source: str) -> bool:
    locked = db[STRATEGY_CONTROL_COLLECTION].find_one_and_update(
        {
            "_id": "default",
            "winner_promotion_in_progress": {"$ne": True},
            "live_market_refresh_in_progress": {"$ne": True},
        },
        {
            "$set": {
                "live_market_refresh_in_progress": True,
                "live_market_refresh_started_at": utc_now(),
                "live_market_refresh_source": str(source),
            }
        },
        return_document=ReturnDocument.AFTER,
    )
    return locked is not None


def _release_live_market_refresh_lock(db: Any) -> None:
    db[STRATEGY_CONTROL_COLLECTION].update_one(
        {"_id": "default", "live_market_refresh_in_progress": True},
        {
            "$set": {
                "live_market_refresh_in_progress": False,
                "live_market_refresh_started_at": None,
            }
        },
    )


def _validated_context(
    db: Any,
) -> tuple[BacktestRequest, PaperTradingSettings, PaperTradingState, dict[str, Any], dict[str, Any]]:
    strategy_control = db[STRATEGY_CONTROL_COLLECTION].find_one({"_id": "default"}) or {}
    if bool(strategy_control.get("winner_promotion_in_progress")):
        raise RuntimeError(
            "Winner promotion is in progress. The scheduled model pipeline will retry after the metadata handoff completes."
        )
    if trader_winner_requires_state_reinitialization(db):
        raise RuntimeError(
            "The Trader winner requires protected Paper state initialization before arming Trader."
        )
    try:
        winner_configuration, winner_profile = get_trader_winner_context(db)
        winner_model = get_trader_winner_model_snapshot(db)
        strategy = apply_training_runtime_settings(
            db,
            winner_configuration,
        )
        strategy = apply_execution_profile(
            strategy,
            winner_model["family"],
            winner_model.get("settings_snapshot") or {},
        )
        strategy = strategy.model_copy(
            update={
                "research_model_family": winner_model["family"],
                "research_model_settings": winner_model.get("settings_snapshot") or {},
            }
        )
        settings = PaperTradingSettings.model_validate(get_paper_trading_settings(db))
        state = PaperTradingState.model_validate(get_paper_trading_state(db))
    except (RuntimeError, ValidationError) as exc:
        raise RuntimeError(f"Paper-trading configuration is unavailable or invalid: {exc}") from exc

    if not settings.enabled:
        raise RuntimeError("Paper trading is disabled in MongoDB.")
    if strategy.strategy_mode not in SWING_STRATEGY_MODES:
        raise RuntimeError("Paper trading requires the validated compound-rotation strategy contract.")
    if strategy.strategy_mode in RESEARCH_ONLY_SWING_STRATEGY_MODES:
        raise RuntimeError("Opportunity Cash Gate / Absolute Utility Cash Gate / Portfolio Allocation / Compound Risk Overlay v3.12.0 is research/backtest-only until the compatible Paper executor is enabled.")
    if strategy.rotation_models != ["xgboost_utility"]:
        raise RuntimeError("The legacy strategy model marker changed unexpectedly.")
    if winner_model["family"] not in {"xgboost_utility", "lightgbm_utility"}:
        raise RuntimeError(
            f"Trader Winner model {winner_model['family']!r} does not have a protected live engine."
        )
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
    return strategy, settings, state, winner_profile, winner_model


def refresh_trader_live_market_data(
    db: Any,
    *,
    source: str,
    force: bool = False,
    now: datetime | pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Keep the operational Winner market data current without mutating the Winner snapshot."""
    target = latest_safe_completed_xnys_session(now).date().isoformat()
    control = db[STRATEGY_CONTROL_COLLECTION].find_one({"_id": "default"}) or {}
    current_cutoff = str(control.get("live_market_cutoff") or "").strip()
    current_winner = str(control.get("live_market_cutoff_winner_strategy_id") or "").strip()
    trader_winner_id = str(control.get("trader_winner_strategy_id") or "").strip()
    if not force and trader_winner_id and current_cutoff == target and current_winner == trader_winner_id:
        return {
            "live_market_cutoff": target,
            "target_session": target,
            "refreshed": False,
            "source": str(source),
        }

    now_utc = pd.Timestamp(now if now is not None else utc_now())
    now_utc = now_utc.tz_localize("UTC") if now_utc.tzinfo is None else now_utc.tz_convert("UTC")
    retry_target = str(control.get("live_market_refresh_target") or "").strip()
    retry_at = control.get("live_market_refresh_next_retry_at")
    retry_stamp = None
    if retry_at is not None:
        retry_stamp = pd.Timestamp(retry_at)
        retry_stamp = retry_stamp.tz_localize("UTC") if retry_stamp.tzinfo is None else retry_stamp.tz_convert("UTC")
    if not force and retry_target == target and retry_stamp is not None and now_utc < retry_stamp:
        return {
            "live_market_cutoff": current_cutoff or None,
            "target_session": target,
            "refreshed": False,
            "pending_retry": True,
            "next_retry_at": retry_stamp.isoformat(),
            "source": str(source),
        }

    if not _acquire_live_market_refresh_lock(db, source=source):
        current = db[STRATEGY_CONTROL_COLLECTION].find_one({"_id": "default"}) or {}
        return {
            "live_market_cutoff": current.get("live_market_cutoff"),
            "target_session": target,
            "refreshed": False,
            "pending_retry": True,
            "blocked_by": (
                "winner_promotion"
                if bool(current.get("winner_promotion_in_progress"))
                else "temporal_market_series_update"
            ),
            "source": str(source),
        }

    try:
        control = db[STRATEGY_CONTROL_COLLECTION].find_one({"_id": "default"}) or {}
        current_cutoff = str(control.get("live_market_cutoff") or "").strip()
        current_winner = str(control.get("live_market_cutoff_winner_strategy_id") or "").strip()
        trader_winner_id = str(control.get("trader_winner_strategy_id") or "").strip()
        if not force and trader_winner_id and current_cutoff == target and current_winner == trader_winner_id:
            return {
                "live_market_cutoff": target,
                "target_session": target,
                "refreshed": False,
                "source": str(source),
            }

        winner_configuration, winner_profile = get_trader_winner_context(db)
        winner_model = get_trader_winner_model_snapshot(db)
        strategy = apply_training_runtime_settings(db, winner_configuration)
        strategy = apply_execution_profile(
            strategy,
            winner_model["family"],
            winner_model.get("settings_snapshot") or {},
        )
        if strategy.market_data_provider != "alpaca":
            raise RuntimeError("Trader live market refresh requires market_data_provider='alpaca'.")
        if strategy.end_date is not None:
            raise RuntimeError(
                "Trader live market refresh requires end_date=None; the certified cutoff belongs to certification metadata, not the live Strategy window."
            )

        attempt_at = utc_now()
        db[STRATEGY_CONTROL_COLLECTION].update_one(
            {"_id": "default"},
            {
                "$set": {
                    "live_market_refresh_target": target,
                    "live_market_refresh_last_attempt_at": attempt_at,
                    "live_market_refresh_source": str(source),
                },
                "$unset": {"live_market_refresh_last_error": ""},
            },
        )
        try:
            refreshed = refresh_market_data_to_live_cutoff(strategy, now=now)
            metadata = update_trader_live_market_cutoff(
                db,
                cutoff=str(refreshed["live_market_cutoff"]),
                source=source,
            )
            db[STRATEGY_CONTROL_COLLECTION].update_one(
                {"_id": "default"},
                {
                    "$set": {"live_market_refresh_last_success_at": utc_now()},
                    "$unset": {
                        "live_market_refresh_next_retry_at": "",
                        "live_market_refresh_last_error": "",
                    },
                },
            )
            return {**refreshed, **metadata, "refreshed": True}
        except Exception as exc:
            next_retry = utc_now() + timedelta(minutes=5)
            db[STRATEGY_CONTROL_COLLECTION].update_one(
                {"_id": "default"},
                {
                    "$set": {
                        "live_market_refresh_next_retry_at": next_retry,
                        "live_market_refresh_last_error": str(exc)[:1000],
                    }
                },
            )
            raise
    finally:
        _release_live_market_refresh_lock(db)


def paper_market_readiness(db: Any) -> dict[str, Any]:
    

    strategy, settings, state, winner_profile, winner_model = _validated_context(db)
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
        "live_market_cutoff": (db[STRATEGY_CONTROL_COLLECTION].find_one({"_id": "default"}) or {}).get("live_market_cutoff"),
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
    strategy, _winner_profile = get_trader_winner_context(db)
    settings = PaperTradingSettings.model_validate(get_paper_trading_settings(db))
    if not settings.enabled:
        raise RuntimeError("Paper trading is disabled in MongoDB.")
    if strategy.strategy_mode not in SWING_STRATEGY_MODES:
        raise RuntimeError("The locked strategy is not the XGBoost swing strategy.")
    if strategy.strategy_mode in RESEARCH_ONLY_SWING_STRATEGY_MODES:
        raise RuntimeError("Opportunity Cash Gate / Absolute Utility Cash Gate / Portfolio Allocation / Compound Risk Overlay v3.12.0 is research/backtest-only until the compatible Paper executor is enabled.")

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
    mark_trader_winner_state_initialized(db)
    return {
        "paper_account_id": account["id"],
        "account_cash": account["cash"],
        "strategy_initial_capital": state.initial_capital,
        "strategy_cash": state.strategy_cash,
        "managed_symbol": None,
    }


def prepare_next_paper_plan(
    db: Any,
    *,
    replace: bool = False,
    allow_open_market: bool = False,
    execution_session_override: str | None = None,
    expected_market_open_override: Any | None = None,
    refresh_source: str = "premarket_plan_refresh",
) -> dict[str, Any]:
    runtime_training = get_system_settings(db)["training"]
    if not bool(runtime_training["enabled"]):
        raise RuntimeError("Model training is disabled in System Settings.")
    strategy, settings, state, winner_profile, winner_model = _validated_context(db)
    client = create_paper_trading_client(db)
    account = account_snapshot(client)
    assert_account_can_trade(account)
    clock = clock_snapshot(client)
    market_is_open = bool(clock["is_open"])
    if market_is_open and not allow_open_market:
        raise RuntimeError(
            "Prepare the next-open decision only while the regular market is closed, "
            "so the latest daily candle is complete."
        )
    if allow_open_market and not market_is_open:
        raise RuntimeError("Manual current-session recovery requires the regular market to be open.")

    live_market = refresh_trader_live_market_data(db, source=refresh_source, force=True)

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

    decision = build_live_model_decision(
        bars_by_symbol,
        strategy,
        model_family=winner_model["family"],
        current_asset=state.managed_symbol,
        holding_sessions=state.holding_sessions,
    )
    stateful_decision = None
    if (
        str(winner_profile.get("strategy_kind") or "") == "temporal_intelligence"
        and str(winner_profile.get("temporal_strategy_variant") or "") == "winner_transition_stateful"
    ):
        from .temporal_winner_transition_stateful import build_stateful_candidate_a_live_decision
        stateful_decision = build_stateful_candidate_a_live_decision(
            db,
            bars_by_symbol=bars_by_symbol,
            strategy=strategy,
            current_asset=state.managed_symbol,
            holding_sessions=state.holding_sessions,
            winner_profile=winner_profile,
            winner_model=winner_model,
            cooldown=bool(state.stateful_defer_cooldown),
        )
        if _et_date(stateful_decision["decision_date"]) != _et_date(decision.decision_date):
            raise RuntimeError("Stateful and base live decisions resolved different completed sessions.")
    decision_date = _et_date(decision.decision_date)
    if expected_market_open_override is not None:
        expected_open = pd.Timestamp(expected_market_open_override)
        expected_open = expected_open.tz_localize("UTC") if expected_open.tzinfo is None else expected_open.tz_convert("UTC")
    else:
        expected_open = pd.Timestamp(clock["next_open"])
        expected_open = expected_open.tz_localize("UTC") if expected_open.tzinfo is None else expected_open.tz_convert("UTC")
    execution_session = (
        str(execution_session_override)
        if execution_session_override is not None
        else expected_open.tz_convert(EASTERN).date().isoformat()
    )
    if execution_session != expected_open.tz_convert(EASTERN).date().isoformat():
        raise RuntimeError(
            "Execution-session override does not match the supplied market-open timestamp: "
            f"session={execution_session}, open={expected_open.isoformat()}."
        )
    if allow_open_market:
        current_session = pd.Timestamp(clock["timestamp"])
        current_session = current_session.tz_localize("UTC") if current_session.tzinfo is None else current_session.tz_convert("UTC")
        current_session = current_session.tz_convert(EASTERN).date().isoformat()
        if execution_session != current_session:
            raise RuntimeError(
                "Manual recovery may prepare only the currently open regular session: "
                f"requested={execution_session}, current={current_session}."
            )

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
    target = str((stateful_decision or {}).get("target_asset") or decision.target_asset).upper()
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

    plan_id = f"{winner_model['family'].split('_')[0]}-{decision_date}-{uuid.uuid4().hex[:8]}"
    plan = PaperTradePlan(
        plan_id=plan_id,
        status="prepared",
        winner_strategy_id=str(winner_profile["id"]),
        winner_strategy_name=str(winner_profile["name"]),
        winner_strategy_revision=int(winner_profile["revision"]),
        winner_configuration_hash=str(winner_profile["configuration_hash"]),
        winner_model_family=str(winner_model["family"]),
        winner_model_profile_id=str(winner_model["profile_id"]),
        winner_model_settings_hash=str(winner_model["settings_hash"]),
        winner_assets=list(strategy.assets),
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
        cash_edges=decision.cash_edges,
        opportunity_probability=decision.opportunity_probability,
        opportunity_confidence=decision.opportunity_confidence,
        opportunity_threshold=decision.opportunity_threshold,
        opportunity_accepted=decision.opportunity_accepted,
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
        "live_market_cutoff": live_market.get("live_market_cutoff"),
        "manual_current_session_recovery": bool(allow_open_market),
        "plan_source": refresh_source,
        "stateful_intervention": bool((stateful_decision or {}).get("stateful_intervention")),
        "stateful_control_target_asset": (stateful_decision or {}).get("control_target_asset"),
        "stateful_defer_cooldown_before": bool((stateful_decision or {}).get("stateful_cooldown_before")),
        "stateful_defer_cooldown_after": bool((stateful_decision or {}).get("stateful_cooldown_after")),
        "stateful_risk_score": (stateful_decision or {}).get("risk_score"),
        "stateful_risk_threshold": (stateful_decision or {}).get("risk_threshold"),
        "stateful_confidence_margin": (stateful_decision or {}).get("confidence_margin"),
        "stateful_confidence_threshold": (stateful_decision or {}).get("confidence_threshold"),
        "stateful_risk_family": (stateful_decision or {}).get("risk_family"),
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
    strategy, settings, state, winner_profile, winner_model = _validated_context(db)
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

    plan_winner_id = str(plan.get("winner_strategy_id") or "")
    plan_winner_revision = int(plan.get("winner_strategy_revision") or 0)
    plan_winner_hash = str(plan.get("winner_configuration_hash") or "")
    if not plan_winner_id or not plan_winner_hash:
        raise RuntimeError(
            "The prepared Paper plan predates Winner identity binding and cannot be executed safely."
        )
    if (
        plan_winner_id != str(winner_profile["id"])
        or plan_winner_revision != int(winner_profile["revision"])
        or plan_winner_hash != str(winner_profile["configuration_hash"])
    ):
        raise RuntimeError(
            "The prepared Paper plan belongs to a different Trader Winner. "
            "Discard it and let the scheduled pre-market cycle recalibrate and rebuild the plan."
        )

    plan_model_family = str(plan.get("winner_model_family") or "")
    plan_model_hash = str(plan.get("winner_model_settings_hash") or "")
    if not plan_model_family or not plan_model_hash:
        raise RuntimeError(
            "The prepared Paper plan predates Winner model binding and cannot be executed safely."
        )
    if (
        plan_model_family != str(winner_model["family"])
        or plan_model_hash != str(winner_model["settings_hash"])
    ):
        raise RuntimeError(
            "The prepared Paper plan belongs to a different Winner model snapshot. "
            "Discard it and let the scheduled pre-market cycle rebuild the plan."
        )

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
                "stateful_defer_cooldown": bool(plan.get("stateful_defer_cooldown_after")),
                "stateful_last_intervention_date": (
                    str(plan["decision_date"])
                    if bool(plan.get("stateful_intervention"))
                    else state.stateful_last_intervention_date
                ),
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
