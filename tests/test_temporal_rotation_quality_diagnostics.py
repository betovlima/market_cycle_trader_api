from __future__ import annotations

import io
import zipfile

import pandas as pd
import pytest

from market_cycle_trader_api.schemas.temporal_rotation_quality import TemporalRotationQualityDiagnosticRequest
from market_cycle_trader_api.services.temporal_rotation_quality import ReplayInputs
from market_cycle_trader_api.services.temporal_rotation_quality_diagnostics import (
    TEMPORAL_ROTATION_QUALITY_DIAGNOSTIC_COLLECTION,
    build_rotation_quality_diagnostic,
    build_temporal_rotation_quality_diagnostic_export,
)


def _inputs() -> ReplayInputs:
    timestamps = pd.date_range("2026-01-02", periods=5, freq="B", tz="UTC")
    equity = pd.DataFrame(
        [
            {"fold_id": 1, "decision_timestamp": timestamps[0], "target_symbol": "A"},
            {"fold_id": 1, "decision_timestamp": timestamps[1], "target_symbol": "B"},
            {"fold_id": 1, "decision_timestamp": timestamps[2], "target_symbol": "C"},
            {"fold_id": 1, "decision_timestamp": timestamps[3], "target_symbol": "B"},
            {"fold_id": 1, "decision_timestamp": timestamps[4], "target_symbol": "C"},
        ]
    )
    returns = {
        (1, timestamps[0], "A"): -0.10,
        (1, timestamps[1], "A"): 0.02,
        (1, timestamps[1], "B"): -0.03,
        (1, timestamps[2], "A"): -0.03,
        (1, timestamps[2], "C"): 0.02,
        (1, timestamps[3], "A"): 0.04,
        (1, timestamps[3], "B"): -0.01,
        (1, timestamps[4], "A"): -0.02,
        (1, timestamps[4], "C"): 0.03,
    }
    rows = []
    for index, timestamp in enumerate(timestamps):
        rows.extend(
            [
                {
                    "fold_id": 1,
                    "timestamp": timestamp,
                    "symbol": "A",
                    "entry_rank_score": 0.70,
                    "hold_score": 0.60 - index * 0.01,
                    "open_to_open_return": returns.get((1, timestamp, "A"), 0.0),
                },
                {
                    "fold_id": 1,
                    "timestamp": timestamp,
                    "symbol": "B",
                    "entry_rank_score": 0.50,
                    "hold_score": 0.30 + index * 0.08,
                    "open_to_open_return": returns.get((1, timestamp, "B"), 0.0),
                },
                {
                    "fold_id": 1,
                    "timestamp": timestamp,
                    "symbol": "C",
                    "entry_rank_score": 0.50,
                    "hold_score": 0.55 - index * 0.08,
                    "open_to_open_return": returns.get((1, timestamp, "C"), 0.0),
                },
            ]
        )
    daily = pd.DataFrame(rows)
    return_map = daily.set_index(["fold_id", "timestamp", "symbol"])["open_to_open_return"].astype(float).to_dict()
    score_map = daily.set_index(["fold_id", "timestamp", "symbol"])["entry_rank_score"].astype(float).to_dict()
    return ReplayInputs(
        summary=pd.DataFrame([{"run_id": "source"}]),
        multi=pd.DataFrame([{"ending_capital": 1.0}]),
        equity=equity,
        daily_assets=daily,
        folds=pd.DataFrame([{"fold_id": 1}]),
        return_map=return_map,
        score_map=score_map,
    )


def test_diagnostic_request_is_parameter_driven() -> None:
    request = TemporalRotationQualityDiagnosticRequest(
        candidate_id="RQ-017",
        lookback_sessions=7,
        feature_names=["entry_rank_score", "hold_score"],
        minimum_group_samples=2,
        outcome_neutral_band=0.001,
        top_feature_count=12,
    )
    assert request.lookback_sessions == 7
    assert request.feature_names == ["entry_rank_score", "hold_score"]
    assert request.outcome_neutral_band == pytest.approx(0.001)


def test_diagnostic_uses_past_feature_history_and_future_only_as_label() -> None:
    request = TemporalRotationQualityDiagnosticRequest(
        candidate_id="RQ-017",
        lookback_sessions=1,
        feature_names=["entry_rank_score", "hold_score"],
        minimum_group_samples=2,
        top_feature_count=20,
    )
    result = build_rotation_quality_diagnostic(
        _inputs(),
        candidate_id="RQ-017",
        drawdown_trigger=-0.05,
        rotation_score_tolerance=-0.10,
        request=request,
    )
    assert result["blocked_rotations"] == 4
    assert result["helpful_blocks"] == 2
    assert result["harmful_blocks"] == 2
    assert result["diagnostic_policy"]["future_information_used_for_decision"] is False
    assert result["diagnostic_policy"]["future_outcome_used_as_diagnostic_label_only"] is True
    assert all("incumbent_hold_score_delta" in event for event in result["events"])
    assert any(row["feature"] == "hold_score" for row in result["feature_separation"])


def test_diagnostic_export_contains_complete_bundle() -> None:
    class Collection:
        def __init__(self, document):
            self.document = document

        def find_one(self, query, projection=None):
            if query.get("id") != self.document.get("id"):
                return None
            return dict(self.document)

    document = {
        "id": "diag-1",
        "research_id": "research-1",
        "validation_id": "validation-1",
        "source_run_id": "source-1",
        "candidate_id": "RQ-017",
        "status": "completed",
        "events": [{"fold_id": 1, "outcome_class": "helpful", "incremental_interval_return": 0.01}],
        "feature_separation": [{"feature": "hold_score", "standardized_separation": 0.5}],
        "fold_summary": [{"fold_id": 1, "blocked_rotations": 1}],
        "top_feature_separation": [{"feature": "hold_score"}],
        "diagnostic_policy": {"future_information_used_for_decision": False},
    }
    db = {TEMPORAL_ROTATION_QUALITY_DIAGNOSTIC_COLLECTION: Collection(document)}
    raw = build_temporal_rotation_quality_diagnostic_export(db, "research-1", "validation-1", "diag-1")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        assert {
            "summary.json",
            "diagnostic_policy.json",
            "blocked_rotation_diagnostics.csv",
            "feature_separation.csv",
            "fold_summary.csv",
            "metadata.json",
        }.issubset(set(archive.namelist()))
