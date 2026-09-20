from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from market_cycle_trader_api.engine.capital_rotation import ROTATION_FEATURES
from market_cycle_trader_api.engine.research_challengers import (
    _horizon_rank_consensus_snapshot,
    _soft_horizon_consensus_policy,
    _soft_horizon_consensus_settings,
)


class _ConstantModel:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(
            len(frame),
            self.value,
            dtype=float,
        )


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        research_model_settings={
            "horizon_voting": {
                "enabled": False,
            },
            "soft_horizon_consensus": {
                "enabled": True,
                "penalty_strength": 1.0,
            },
        },
        rotation_target_horizons=[
            5,
            10,
            20,
            40,
            60,
        ],
        rotation_target_horizon_weights=[
            0.10,
            0.15,
            0.20,
            0.30,
            0.25,
        ],
    )


def _frames() -> tuple[
    dict[str, pd.DataFrame],
    pd.Timestamp,
]:
    index = pd.to_datetime(
        [
            "2026-01-05",
            "2026-01-06",
        ],
        utc=True,
    )
    frames: dict[str, pd.DataFrame] = {}
    for symbol in ("AAA", "BBB", "CCC"):
        frame = pd.DataFrame(
            0.0,
            index=index,
            columns=ROTATION_FEATURES,
        )
        frame["open"] = 100.0
        frame["close"] = 100.0
        frames[symbol] = frame
    return frames, pd.Timestamp(index[0])


def _horizon_models(
    values_by_horizon: dict[
        int,
        dict[str, float],
    ],
) -> dict[int, dict[str, _ConstantModel]]:
    return {
        int(horizon): {
            symbol: _ConstantModel(value)
            for symbol, value in values.items()
        }
        for horizon, values
        in values_by_horizon.items()
    }


def test_soft_consensus_uses_weighted_rank_not_exact_winner_vote() -> None:
    frames, timestamp = _frames()
    models = _horizon_models(
        {
            5: {
                "AAA": 0.30,
                "BBB": 0.20,
                "CCC": 0.10,
            },
            10: {
                "BBB": 0.30,
                "AAA": 0.20,
                "CCC": 0.10,
            },
            20: {
                "CCC": 0.30,
                "AAA": 0.20,
                "BBB": 0.10,
            },
            40: {
                "BBB": 0.30,
                "AAA": 0.20,
                "CCC": 0.10,
            },
            60: {
                "CCC": 0.30,
                "AAA": 0.20,
                "BBB": 0.10,
            },
        }
    )

    snapshot = _horizon_rank_consensus_snapshot(
        models,
        frames,
        ["AAA", "BBB", "CCC"],
        timestamp,
        _config(),
        base_target=1,
        current_position=2,
    )

    # AAA wins only the 5d horizon (10% of hard votes), but it is never
    # worse than rank 2. Weighted rank aggregation therefore preserves
    # that information instead of discarding it.
    assert snapshot["consensus_winner_asset"] == "AAA"
    assert np.isclose(
        snapshot["base_target_rank_score"],
        0.55,
    )
    assert snapshot["soft_support"] is not None
    assert float(snapshot["soft_support"]) > 0.50


def test_soft_consensus_accepts_strong_primary_gap_without_majority() -> None:
    frames, timestamp = _frames()
    horizon_models = _horizon_models(
        {
            5: {"AAA": 0.30, "BBB": 0.20, "CCC": 0.10},
            10: {"BBB": 0.30, "AAA": 0.20, "CCC": 0.10},
            20: {"CCC": 0.30, "AAA": 0.20, "BBB": 0.10},
            40: {"BBB": 0.30, "AAA": 0.20, "CCC": 0.10},
            60: {"CCC": 0.30, "AAA": 0.20, "BBB": 0.10},
        }
    )
    base_models = {
        "AAA": _ConstantModel(0.020),
        "BBB": _ConstantModel(0.010),
        "CCC": _ConstantModel(0.005),
    }
    diagnostics: dict[pd.Timestamp, dict] = {}
    base_policy = (
        lambda ts, position, holding: (1, 0.020)
    )
    policy = _soft_horizon_consensus_policy(
        base_policy,
        base_models,
        horizon_models,
        frames,
        ["AAA", "BBB", "CCC"],
        _config(),
        base_switch_margin=0.005,
        decision_diagnostics=diagnostics,
    )

    target, _ = policy(
        timestamp,
        2,
        4,
    )

    assert target == 1
    diag = diagnostics[timestamp]
    assert (
        diag["soft_horizon_consensus_reason"]
        == "SOFT_CONSENSUS_ACCEPT"
    )
    assert (
        diag[
            "soft_horizon_consensus_margin_multiplier"
        ]
        > 1.0
    )
    assert (
        diag[
            "soft_horizon_consensus_dynamic_margin"
        ]
        < 0.010
    )


def test_soft_consensus_blocks_only_marginal_low_support_switch() -> None:
    frames, timestamp = _frames()
    horizon_models = _horizon_models(
        {
            5: {"BBB": 0.30, "CCC": 0.20, "AAA": 0.10},
            10: {"BBB": 0.30, "CCC": 0.20, "AAA": 0.10},
            20: {"BBB": 0.30, "CCC": 0.20, "AAA": 0.10},
            40: {"BBB": 0.30, "CCC": 0.20, "AAA": 0.10},
            60: {"BBB": 0.30, "CCC": 0.20, "AAA": 0.10},
        }
    )
    base_models = {
        "AAA": _ConstantModel(0.016),
        "BBB": _ConstantModel(0.010),
        "CCC": _ConstantModel(0.005),
    }
    diagnostics: dict[pd.Timestamp, dict] = {}
    base_policy = (
        lambda ts, position, holding: (1, 0.016)
    )
    policy = _soft_horizon_consensus_policy(
        base_policy,
        base_models,
        horizon_models,
        frames,
        ["AAA", "BBB", "CCC"],
        _config(),
        base_switch_margin=0.005,
        decision_diagnostics=diagnostics,
    )

    target, _ = policy(
        timestamp,
        2,
        4,
    )

    assert target == 2
    diag = diagnostics[timestamp]
    assert (
        diag["soft_horizon_consensus_reason"]
        == "SOFT_CONSENSUS_BLOCK_MARGINAL_SWITCH"
    )
    assert np.isclose(
        diag[
            "soft_horizon_consensus_margin_multiplier"
        ],
        2.0,
    )
    assert np.isclose(
        diag[
            "soft_horizon_consensus_dynamic_margin"
        ],
        0.010,
    )


def test_soft_consensus_never_overrides_base_cash() -> None:
    frames, timestamp = _frames()
    diagnostics: dict[pd.Timestamp, dict] = {}
    base_policy = (
        lambda ts, position, holding: (0, 0.0)
    )
    policy = _soft_horizon_consensus_policy(
        base_policy,
        {},
        {},
        frames,
        ["AAA", "BBB", "CCC"],
        _config(),
        base_switch_margin=0.005,
        decision_diagnostics=diagnostics,
    )

    target, score = policy(
        timestamp,
        1,
        3,
    )

    assert target == 0
    assert score == 0.0
    assert (
        diagnostics[timestamp][
            "soft_horizon_consensus_reason"
        ]
        == "BASE_CASH_PRESERVED"
    )


def test_soft_consensus_settings_reuse_existing_horizon_weights() -> None:
    settings = _soft_horizon_consensus_settings(
        _config()
    )

    assert settings["enabled"] is True
    assert settings["penalty_strength"] == 1.0
    assert settings["horizons"] == [
        5,
        10,
        20,
        40,
        60,
    ]
    assert np.isclose(
        sum(settings["weights"]),
        1.0,
    )
