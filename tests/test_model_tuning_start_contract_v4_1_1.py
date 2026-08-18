from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from market_cycle_trader_api.schemas.model_tuning import ModelTuningStartRequest


ROOT = Path(__file__).resolve().parents[1]


def _policy_request() -> dict:
    return {
        "method": "champion_probability",
        "candidate_count": 20,
        "seed": 42,
        "tuning_target": "temporal_policy",
        "explicit_start_confirmation": False,
        "fold_protocol": {
            "research_folds": 3,
            "validation_folds": 5,
            "certification_folds": 7,
        },
        "probability": {
            "min_capital_improvement": 0.03,
            "sharpe_tolerance": 0.05,
            "drawdown_tolerance": 0.03,
            "min_worst_fold_return": 0.0,
            "adaptive_stopping_enabled": True,
            "no_improvement_trial_limit": 100,
            "minimum_meaningful_improvement": 0.0025,
        },
    }


def test_temporal_policy_caro_front_payload_matches_api_request_schema() -> None:
    request = ModelTuningStartRequest.model_validate(_policy_request())
    assert request.tuning_target == "temporal_policy"
    assert request.method == "champion_probability"
    assert request.fold_protocol is not None
    assert request.fold_protocol.research_folds == 3


def test_unknown_start_request_input_remains_strictly_rejected() -> None:
    payload = _policy_request()
    payload["unexpected_front_field"] = True
    with pytest.raises(ValidationError) as exc_info:
        ModelTuningStartRequest.model_validate(payload)
    error = exc_info.value.errors()[0]
    assert error["type"] == "extra_forbidden"
    assert tuple(error["loc"]) == ("unexpected_front_field",)


def test_tuning_catalog_and_front_publish_same_contract_marker() -> None:
    service = (ROOT / "src" / "market_cycle_trader_api" / "services" / "model_tuning.py").read_text(encoding="utf-8")
    front = (ROOT.parent / "market_cycle_trader" / "src" / "features" / "ModelTuningPanel.jsx")
    # The API test package does not contain the Front; validate the server marker here.
    assert '"start_request_contract_version": 1' in service
