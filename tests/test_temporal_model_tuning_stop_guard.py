from __future__ import annotations

from pathlib import Path

import pytest

from market_cycle_trader_api.engine.temporal_intelligence import run_temporal_intelligence
from market_cycle_trader_api.schemas.model_tuning import ModelTuningStartRequest
from market_cycle_trader_api.schemas.requests import BacktestExecutionRequest


def test_temporal_model_start_requires_explicit_confirmation_in_contract() -> None:
    payload = ModelTuningStartRequest(
        tuning_target="temporal_model",
        candidate_count=8,
    )
    assert payload.explicit_start_confirmation is False

    confirmed = ModelTuningStartRequest(
        tuning_target="temporal_model",
        candidate_count=8,
        explicit_start_confirmation=True,
    )
    assert confirmed.explicit_start_confirmation is True


def test_temporal_engine_checks_cancellation_before_work_begins() -> None:
    request = BacktestExecutionRequest.model_construct(research_model_family="lightgbm_utility")

    def cancel() -> None:
        raise RuntimeError("cancel-now")

    with pytest.raises(RuntimeError, match="cancel-now"):
        run_temporal_intelligence({}, request, cancel_callback=cancel)


def test_temporal_model_ui_requires_user_confirmation_and_sends_confirmation_flag() -> None:
    root = Path(__file__).resolve().parents[1]
    panel = (root.parent / "market_cycle_trader" / "src" / "features" / "ModelTuningPanel.jsx").read_text(encoding="utf-8")
    service = (root / "src" / "market_cycle_trader_api" / "services" / "model_tuning.py").read_text(encoding="utf-8")
    model_service = (root / "src" / "market_cycle_trader_api" / "services" / "temporal_model_tuning.py").read_text(encoding="utf-8")
    engine = (root / "src" / "market_cycle_trader_api" / "engine" / "temporal_intelligence.py").read_text(encoding="utf-8")

    assert "window.confirm" in panel
    assert "body.explicit_start_confirmation = temporalModelMode" in panel
    assert "Temporal Model Tuning requires explicit start confirmation." in service
    assert "TemporalModelTuningCancelled" in service
    assert "cancel_check=temporal_cancel_requested" in service
    assert "cancel_callback" in model_service
    assert "ensure_not_cancelled()" in engine


def test_temporal_model_stop_has_own_cooperative_cancellation_phase() -> None:
    service = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "market_cycle_trader_api"
        / "services"
        / "model_tuning.py"
    ).read_text(encoding="utf-8")

    assert '"cancelling_temporal_model_candidate"' in service
    assert "next model checkpoint" in service
    assert '"candidates.$.status": "cancelled"' in service
    assert '"status": "stopped"' in service
    recovery = service[service.index("def recover_integrated_model_tuning_runs"):service.index("def request_model_tuning_stop")]
    assert '"execution_mode": "full_temporal_lightgbm_retrain"' in recovery


def test_temporal_model_stop_moves_running_candidate_to_cooperative_cancel_phase(monkeypatch) -> None:
    from copy import deepcopy

    from market_cycle_trader_api.infrastructure.persistence.mongo_repository import MODEL_TUNING_RUNS_COLLECTION
    from market_cycle_trader_api.services import model_tuning as tuning_service

    class Collection:
        def __init__(self) -> None:
            self.document = {
                "id": "temporal-stop-1",
                "status": "running",
                "phase": "running_candidate",
                "tuning_scope": "temporal_model",
                "current_candidate_id": 2,
                "current_job_id": None,
                "candidates": [{"candidate_id": 2, "status": "running"}],
            }

        def find_one(self, query, *args, **kwargs):
            return deepcopy(self.document) if query.get("id") == self.document["id"] else None

        def find_one_and_update(self, query, update, **kwargs):
            self.document.update(deepcopy(update.get("$set") or {}))
            return deepcopy(self.document)

    collection = Collection()
    db = {MODEL_TUNING_RUNS_COLLECTION: collection}
    monkeypatch.setattr(tuning_service, "_append_campaign_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(tuning_service, "public_model_tuning_run", lambda _db, document: document)

    result = tuning_service.request_model_tuning_stop(db, "temporal-stop-1")

    assert result["status"] == "stop_requested"
    assert result["phase"] == "cancelling_temporal_model_candidate"
    assert result["stop_requested"] is True
