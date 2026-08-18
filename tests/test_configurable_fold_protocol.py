from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import ValidationError

from market_cycle_trader_api.engine.capital_rotation import _build_walk_forward_folds
from market_cycle_trader_api.schemas.model_tuning import ModelTuningStartRequest
from market_cycle_trader_api.services.model_tuning import _normalized_fold_protocol

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "market_cycle_trader_api"
FRONT = ROOT.parent / "market_cycle_trader"


def _config(folds: int) -> SimpleNamespace:
    return SimpleNamespace(
        rotation_purge_days=5,
        rotation_target_horizons=[5, 10],
        rotation_walk_forward_calibration_days=20,
        rotation_walk_forward_test_days=63,
        rotation_walk_forward_min_test_days=20,
        rotation_minimum_training_rows=100,
        walk_forward_fold_count_override=folds,
    )


def test_fold_protocol_defaults_and_user_values_are_request_parameters() -> None:
    default = ModelTuningStartRequest()
    assert default.fold_protocol is None

    request = ModelTuningStartRequest(
        candidate_count=20,
        fold_protocol={
            "research_folds": 4,
            "validation_folds": 8,
            "certification_folds": 12,
        },
    )
    assert request.fold_protocol is not None
    assert request.fold_protocol.research_folds == 4
    assert request.fold_protocol.validation_folds == 8
    assert request.fold_protocol.certification_folds == 12

    # There is no arbitrary scientific ceiling. Feasibility is determined by
    # the available OOS history and the minimum test rows per fold.
    high = ModelTuningStartRequest(
        fold_protocol={
            "research_folds": 60,
            "validation_folds": 75,
            "certification_folds": 90,
        },
    )
    assert high.fold_protocol.certification_folds == 90

    with pytest.raises(ValidationError):
        ModelTuningStartRequest(
            fold_protocol={
                "research_folds": 5,
                "validation_folds": 4,
                "certification_folds": 7,
            },
        )


def test_normalized_fold_protocol_keeps_research_validation_certification_order() -> None:
    assert _normalized_fold_protocol(None) == {
        "research_folds": 3,
        "validation_folds": 5,
        "certification_folds": 7,
    }
    assert _normalized_fold_protocol({
        "research_folds": 3,
        "validation_folds": 6,
        "certification_folds": 9,
    }) == {
        "research_folds": 3,
        "validation_folds": 6,
        "certification_folds": 9,
    }


def test_walk_forward_override_builds_exact_requested_chronological_fold_count() -> None:
    dates = pd.date_range("2018-01-02", periods=700, freq="B", tz="UTC")
    folds = _build_walk_forward_folds(dates, _config(7))
    assert len(folds) == 7
    assert [item["fold_id"] for item in folds] == list(range(1, 8))
    assert all(item["test_end_index"] > item["test_start_index"] for item in folds)
    assert all(
        folds[index]["test_end_index"] == folds[index + 1]["test_start_index"]
        for index in range(len(folds) - 1)
    )
    assert folds[0]["train_end_index"] < folds[-1]["train_end_index"]


def test_walk_forward_fold_count_is_limited_by_real_oos_history_not_ui_lock() -> None:
    dates = pd.date_range("2022-01-03", periods=220, freq="B", tz="UTC")
    with pytest.raises(ValueError, match="Not enough out-of-sample history"):
        _build_walk_forward_folds(dates, _config(20))


def test_temporal_fold_validation_and_certification_are_exposed_end_to_end() -> None:
    service = (SRC / "services" / "model_tuning_validation.py").read_text(encoding="utf-8")
    tuning = (SRC / "services" / "model_tuning.py").read_text(encoding="utf-8")
    router = (SRC / "api" / "routers" / "model_tuning.py").read_text(encoding="utf-8")
    analytics = (SRC / "services" / "analytics.py").read_text(encoding="utf-8")
    panel = (FRONT / "src" / "features" / "ModelTuningPanel.jsx").read_text(encoding="utf-8")

    assert '"research_folds": DEFAULT_RESEARCH_FOLDS' in tuning
    assert '"validation_folds": DEFAULT_VALIDATION_FOLDS' in tuning
    assert '"certification_folds": DEFAULT_CERTIFICATION_FOLDS' in tuning
    assert "evaluate_temporal_model_candidate(" in service
    assert 'fold_count=fold_count' in service
    assert 'certification_processing_id' in service
    assert '@router.post("/{run_id}/candidates/{candidate_id}/certify")' in router
    assert 'caro_certification' in analytics

    assert 'Research folds' in panel
    assert 'Validation folds' in panel
    assert 'Certification folds' in panel
    assert 'body.fold_protocol' in panel
    assert 'validateFinalist(candidate)' in panel
    assert 'certifyCandidate(candidate)' in panel
    assert 'Validation Analytics' in panel
    assert 'Certification Analytics' in panel
