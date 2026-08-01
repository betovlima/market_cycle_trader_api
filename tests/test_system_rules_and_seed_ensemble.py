from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from market_cycle_trader_api.core.system_rules import TRAINING_HISTORY_START
from market_cycle_trader_api.engine.capital_rotation import _majority_vote_policy
from market_cycle_trader_api.schemas.requests import BacktestExecutionRequest, BacktestRequest
from market_cycle_trader_api.schemas.strategy_configuration import StrategyConfigurationPatchRequest


def canonical_payload() -> dict:
    path = (
        Path(__file__).parents[1]
        / "src"
        / "market_cycle_trader_api"
        / "parameterizations"
        / "001_xgboost_high_performance_seed_3042.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_training_start_and_feeds_are_fixed_properties() -> None:
    config = BacktestRequest.model_validate(canonical_payload())
    assert config.start_date == TRAINING_HISTORY_START
    assert config.alpaca_historical_feed == "sip"
    assert config.alpaca_live_feed == "iex"
    assert "start_date" not in config.model_dump()
    assert "alpaca_historical_feed" not in config.model_dump()


def test_public_analysis_window_does_not_change_training_start() -> None:
    config = BacktestExecutionRequest.model_validate(
        {
            **canonical_payload(),
            "analysis_start_date": "2024-01-01",
            "analysis_end_date": None,
        }
    )
    assert config.start_date == "2016-01-01"
    assert config.analysis_start_date == "2024-01-01"


def test_strategy_api_rejects_fixed_system_fields() -> None:
    with pytest.raises(ValidationError):
        StrategyConfigurationPatchRequest.model_validate(
            {
                "confirm_update": True,
                "expected_revision": 1,
                "note": "Attempt to change a fixed system rule.",
                "changes": {"start_date": "2020-01-01"},
            }
        )


def test_majority_vote_ensemble_uses_consensus() -> None:
    policy = _majority_vote_policy(
        [
            lambda *_: (2, 0.3),
            lambda *_: (2, 0.1),
            lambda *_: (1, 0.5),
            lambda *_: (2, 0.2),
            lambda *_: (1, 0.4),
        ],
        minimum_agreement=0.4,
    )
    position, score = policy(pd.Timestamp("2026-01-01"), 0, 0)
    assert position == 2
    assert score == pytest.approx(0.2)


def test_majority_vote_stays_put_when_agreement_is_too_low() -> None:
    policy = _majority_vote_policy(
        [lambda *_: (1, 0.2), lambda *_: (2, 0.3), lambda *_: (3, 0.4)],
        minimum_agreement=0.5,
    )
    position, _ = policy(pd.Timestamp("2026-01-01"), 0, 0)
    assert position == 0
