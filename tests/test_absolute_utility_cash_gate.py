from types import SimpleNamespace

from market_cycle_trader_api.engine.absolute_utility_cash_gate import (
    ABSOLUTE_UTILITY_CASH_GATE_MODE,
    absolute_utility_cash_gate_enabled,
    evaluate_absolute_utility_cash_gate,
)


def _config(entry: float = 0.28, exit_: float = 0.27):
    return SimpleNamespace(
        strategy_mode=ABSOLUTE_UTILITY_CASH_GATE_MODE,
        opportunity_utility_entry_threshold=entry,
        opportunity_utility_exit_threshold=exit_,
    )


def test_absolute_utility_gate_uses_entry_floor_while_in_cash() -> None:
    config = _config()
    assert absolute_utility_cash_gate_enabled(config) is True
    rejected = evaluate_absolute_utility_cash_gate(config, best_score=0.275, current_position=0)
    accepted = evaluate_absolute_utility_cash_gate(config, best_score=0.281, current_position=0)
    assert rejected.active_threshold == 0.28
    assert rejected.accepted is False
    assert rejected.hysteresis_cash_block is True
    assert accepted.accepted is True


def test_absolute_utility_gate_uses_lower_exit_floor_while_in_market() -> None:
    config = _config()
    held = evaluate_absolute_utility_cash_gate(config, best_score=0.275, current_position=3)
    exited = evaluate_absolute_utility_cash_gate(config, best_score=0.269, current_position=3)
    assert held.active_threshold == 0.27
    assert held.accepted is True
    assert held.hysteresis_market_hold is True
    assert exited.accepted is False


def test_absolute_utility_gate_requires_exit_not_above_entry() -> None:
    import json
    from pathlib import Path

    import pytest
    from pydantic import ValidationError

    from market_cycle_trader_api.schemas.requests import BacktestRequest

    payload = json.loads(
        (Path(__file__).resolve().parents[1] / "src" / "market_cycle_trader_api" / "parameterizations" / "winner-v1.13.2.json").read_text()
    )
    payload.update({
        "strategy_mode": ABSOLUTE_UTILITY_CASH_GATE_MODE,
        "opportunity_utility_entry_threshold": 0.27,
        "opportunity_utility_exit_threshold": 0.28,
    })
    with pytest.raises(ValidationError, match="opportunity_utility_exit_threshold"):
        BacktestRequest.model_validate(payload)
