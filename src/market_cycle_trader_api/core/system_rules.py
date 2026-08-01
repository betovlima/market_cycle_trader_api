from __future__ import annotations

from typing import Any

TRAINING_HISTORY_START = "2016-01-01"
TRAINING_HISTORY_END: str | None = None
MARKET_DATA_PROVIDER = "alpaca"
ALPACA_HISTORICAL_FEED = "sip"
ALPACA_LIVE_FEED = "iex"
DECISION_TIME_POLICY = "after_each_completed_regular_session"
EXECUTION_TIME_POLICY = "next_regular_session_open_after_configured_delay"
MODEL_REFRESH_POLICY = "retrain_after_each_completed_regular_session"
NO_LOOKAHEAD = True

IMMUTABLE_STRATEGY_FIELDS = frozenset(
    {
        "start_date",
        "end_date",
        "training_start_date",
        "training_end_date",
        "market_data_provider",
        "alpaca_historical_feed",
        "alpaca_live_feed",
    }
)


def system_rules_payload() -> dict[str, Any]:
    return {
        "training_history_start": TRAINING_HISTORY_START,
        "training_history_end": TRAINING_HISTORY_END,
        "market_data_provider": MARKET_DATA_PROVIDER,
        "alpaca_historical_feed": ALPACA_HISTORICAL_FEED,
        "alpaca_live_feed": ALPACA_LIVE_FEED,
        "decision_time_policy": DECISION_TIME_POLICY,
        "execution_time_policy": EXECUTION_TIME_POLICY,
        "model_refresh_policy": MODEL_REFRESH_POLICY,
        "no_lookahead": NO_LOOKAHEAD,
        "editable_by_frontend": False,
        "editable_by_strategy_api": False,
    }
