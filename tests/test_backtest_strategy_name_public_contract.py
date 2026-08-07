from market_cycle_trader_api.services.jobs import public_job


def test_public_job_exposes_only_strategy_display_name() -> None:
    payload = public_job({
        "id": "job-1",
        "status": "queued",
        "stage": "Queued",
        "strategy_profile_name": "Drawdown Reduction Test A2 - Cash 0.005",
        "strategy_profile_id": "private-id",
        "strategy_profile_revision": 7,
        "strategy_configuration_hash": "private-hash",
        "logs": ["Backtest queued."],
    })

    assert payload is not None
    assert payload["strategy_profile_name"] == "Drawdown Reduction Test A2 - Cash 0.005"
    assert "strategy_profile_id" not in payload
    assert "strategy_profile_revision" not in payload
    assert "strategy_configuration_hash" not in payload
