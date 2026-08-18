from __future__ import annotations

from market_cycle_trader_api.services.temporal_cash_reentry_counterfactual import (
    absolute_opportunity_score,
    replay_absolute_opportunity_reentry_gate,
)


def _row(ret: float, *, score: float, risk: float = 0.8, agreement: float = 0.8):
    return {
        "short_profit_consensus": score,
        "all_horizon_risk_safety": risk,
        "short_horizon_agreement": agreement,
        "long_profit_confirmation": score,
        "horizon_agreement": agreement,
        "long_trend_support": score,
        "open_to_open_return": ret,
        "execution_date": "2026-06-02T00:00:00+00:00",
        "next_execution_date": "2026-06-03T00:00:00+00:00",
    }


def _winner(date: str, asset: str = "AAA"):
    return {
        "decision_date": date,
        "selected_asset": asset,
        "top_1_asset": asset,
        "top_2_asset": "BBB",
    }


def test_absolute_score_uses_only_prediction_components():
    row = _row(-0.99, score=0.7)
    first = absolute_opportunity_score(row)
    row["open_to_open_return"] = 5.0
    second = absolute_opportunity_score(row)
    assert first == second


def test_weak_cash_reentry_is_blocked_and_loss_is_attributed():
    observations = {
        "2026-06-01T00:00:00+00:00": {"fold_id": 1, "rows_by_symbol": {"AAA": _row(-0.10, score=0.30), "BBB": _row(-0.02, score=0.20)}},
        "2026-06-02T00:00:00+00:00": {"fold_id": 1, "rows_by_symbol": {"AAA": _row(0.01, score=0.80), "BBB": _row(0.00, score=0.20)}},
    }
    winners = [_winner(key) for key in observations]
    result = replay_absolute_opportunity_reentry_gate(
        observations, winners, initial_capital=10_000, one_side_cost=0.0,
        gate_settings={
            "absolute_entry_threshold": 0.50,
            "absolute_exit_discount": 0.10,
            "cash_reentry_premium": 0.10,
            "minimum_risk_safety": 0.20,
            "minimum_horizon_agreement": 0.50,
            "reentry_confirmation_sessions": 1,
        }, winner_fold_returns={1: 0.0},
    )
    assert result["intervals"][0]["selected_asset"] == "CASH"
    assert result["intervals"][0]["gate_reason"] == "cash_reentry_absolute_gate_reject"
    assert result["metrics"]["loss_avoided_by_cash_usd"] > 0


def test_strong_reentry_can_enter_after_confirmation():
    observations = {}
    winners = []
    for day in range(1, 4):
        key = f"2026-06-0{day}T00:00:00+00:00"
        observations[key] = {"fold_id": 1, "rows_by_symbol": {"AAA": _row(0.02, score=0.90), "BBB": _row(0.00, score=0.20)}}
        winners.append(_winner(key))
    result = replay_absolute_opportunity_reentry_gate(
        observations, winners, initial_capital=10_000, one_side_cost=0.0,
        gate_settings={
            "absolute_entry_threshold": 0.50,
            "absolute_exit_discount": 0.10,
            "cash_reentry_premium": 0.10,
            "minimum_risk_safety": 0.20,
            "minimum_horizon_agreement": 0.50,
            "reentry_confirmation_sessions": 2,
        }, winner_fold_returns={1: 0.0},
    )
    assert result["intervals"][0]["selected_asset"] == "CASH"
    assert result["intervals"][0]["gate_reason"] == "cash_reentry_confirmation_wait"
    assert result["intervals"][1]["selected_asset"] == "AAA"
    assert result["metrics"]["cash_days"] == 1


def test_weak_rotation_prefers_cash_instead_of_relative_leader():
    first = _row(0.01, score=0.90)
    second_a = _row(-0.05, score=0.25)
    second_b = _row(-0.02, score=0.45)
    observations = {
        "2026-06-01T00:00:00+00:00": {"fold_id": 1, "rows_by_symbol": {"AAA": first, "BBB": _row(0.00, score=0.10)}},
        "2026-06-02T00:00:00+00:00": {"fold_id": 1, "rows_by_symbol": {"AAA": second_a, "BBB": second_b}},
    }
    winners = [
        _winner("2026-06-01T00:00:00+00:00", "AAA"),
        {"decision_date": "2026-06-02T00:00:00+00:00", "selected_asset": "BBB", "top_1_asset": "BBB", "top_2_asset": "AAA"},
    ]
    result = replay_absolute_opportunity_reentry_gate(
        observations, winners, initial_capital=10_000, one_side_cost=0.0,
        gate_settings={
            "absolute_entry_threshold": 0.60,
            "absolute_exit_discount": 0.10,
            "cash_reentry_premium": 0.0,
            "minimum_risk_safety": 0.20,
            "minimum_horizon_agreement": 0.50,
            "reentry_confirmation_sessions": 1,
        }, winner_fold_returns={1: 0.0},
    )
    assert result["intervals"][0]["selected_asset"] == "AAA"
    assert result["intervals"][1]["selected_asset"] == "CASH"
    assert result["intervals"][1]["gate_reason"] == "rotation_absolute_gate_to_cash"
