from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from market_cycle_trader_api.engine.capital_rotation import ROTATION_FEATURES
from market_cycle_trader_api.engine.research_challengers import (
    _horizon_consensus_guard_policy,
    _horizon_vote_snapshot,
    _horizon_voting_settings,
)


class _ConstantModel:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), self.value, dtype=float)


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        research_model_settings={
            "horizon_voting": {
                "enabled": True,
                "minimum_consensus_weight": 0.50,
                "cash_override_enabled": True,
            }
        },
        rotation_target_horizons=[5, 10, 20, 40, 60],
        rotation_target_horizon_weights=[0.10, 0.15, 0.20, 0.30, 0.25],
    )


def _frames() -> tuple[dict[str, pd.DataFrame], pd.Timestamp]:
    index = pd.to_datetime(
        ["2026-01-05", "2026-01-06"],
        utc=True,
    )
    frames: dict[str, pd.DataFrame] = {}
    for symbol in ("AAA", "BBB"):
        frame = pd.DataFrame(
            0.0,
            index=index,
            columns=ROTATION_FEATURES,
        )
        frame["open"] = 100.0
        frame["close"] = 100.0
        frames[symbol] = frame
    return frames, pd.Timestamp(index[0])


def _models(
    winners: dict[int, str],
) -> dict[int, dict[str, _ConstantModel]]:
    output: dict[int, dict[str, _ConstantModel]] = {}
    for horizon, winner in winners.items():
        if winner == "AAA":
            output[horizon] = {
                "AAA": _ConstantModel(0.30),
                "BBB": _ConstantModel(0.10),
            }
        elif winner == "BBB":
            output[horizon] = {
                "AAA": _ConstantModel(0.10),
                "BBB": _ConstantModel(0.30),
            }
        elif winner == "CASH":
            output[horizon] = {
                "AAA": _ConstantModel(-0.10),
                "BBB": _ConstantModel(-0.20),
            }
        else:
            raise AssertionError(winner)
    return output


def test_horizon_weights_are_normalized() -> None:
    settings = _horizon_voting_settings(_config())

    assert settings["enabled"] is True
    assert np.isclose(sum(settings["weights"]), 1.0)
    assert settings["minimum_consensus_weight"] == 0.50


def test_weighted_horizon_vote_can_choose_cash() -> None:
    frames, timestamp = _frames()
    models = _models(
        {
            5: "AAA",
            10: "AAA",
            20: "CASH",
            40: "CASH",
            60: "CASH",
        }
    )

    snapshot = _horizon_vote_snapshot(
        models,
        frames,
        ["AAA", "BBB"],
        timestamp,
        _config(),
        base_target=1,
        current_position=1,
    )

    assert snapshot["winner_asset"] == "CASH"
    assert np.isclose(snapshot["winner_weight"], 0.75)
    assert np.isclose(snapshot["cash_vote_weight"], 0.75)


def test_consensus_guard_overrides_asset_to_cash() -> None:
    frames, timestamp = _frames()
    diagnostics: dict[pd.Timestamp, dict] = {}
    models = _models(
        {
            5: "AAA",
            10: "AAA",
            20: "CASH",
            40: "CASH",
            60: "CASH",
        }
    )
    base_policy = lambda ts, position, holding: (1, 0.42)
    policy = _horizon_consensus_guard_policy(
        base_policy,
        models,
        frames,
        ["AAA", "BBB"],
        _config(),
        decision_diagnostics=diagnostics,
    )

    target, score = policy(timestamp, 1, 10)

    assert target == 0
    assert np.isclose(score, 0.42)
    assert (
        diagnostics[timestamp]["horizon_voting_reason"]
        == "HORIZON_CONSENSUS_CASH_OVERRIDE"
    )


def test_consensus_guard_accepts_base_asset_with_majority() -> None:
    frames, timestamp = _frames()
    diagnostics: dict[pd.Timestamp, dict] = {}
    models = _models(
        {
            5: "BBB",
            10: "BBB",
            20: "AAA",
            40: "AAA",
            60: "AAA",
        }
    )
    base_policy = lambda ts, position, holding: (1, 0.51)
    policy = _horizon_consensus_guard_policy(
        base_policy,
        models,
        frames,
        ["AAA", "BBB"],
        _config(),
        decision_diagnostics=diagnostics,
    )

    target, _ = policy(timestamp, 0, 0)

    assert target == 1
    assert (
        diagnostics[timestamp]["horizon_voting_reason"]
        == "HORIZON_CONSENSUS_ACCEPT"
    )
    assert np.isclose(
        diagnostics[timestamp]["horizon_voting_winner_weight"],
        0.75,
    )


def test_consensus_guard_blocks_switch_without_majority() -> None:
    frames, timestamp = _frames()
    diagnostics: dict[pd.Timestamp, dict] = {}
    models = _models(
        {
            5: "AAA",
            10: "BBB",
            20: "AAA",
            40: "BBB",
            60: "AAA",
        }
    )
    base_policy = lambda ts, position, holding: (2, 0.33)
    policy = _horizon_consensus_guard_policy(
        base_policy,
        models,
        frames,
        ["AAA", "BBB"],
        _config(),
        decision_diagnostics=diagnostics,
    )

    target, _ = policy(timestamp, 1, 8)

    assert target == 1
    assert (
        diagnostics[timestamp]["horizon_voting_reason"]
        == "HORIZON_CONSENSUS_BLOCK_SWITCH"
    )
