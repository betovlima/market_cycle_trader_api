from __future__ import annotations

from pathlib import Path

import pytest

from market_cycle_trader_api.schemas.temporal_rotation_quality import TemporalRotationQualityResearchRequest
from market_cycle_trader_api.services.temporal_rotation_quality import evaluate_temporal_rotation_quality_export


FIXTURE = Path(__file__).resolve().parents[2] / "temporal_intelligence_20260816T181543-temporal-a5afd924.zip"


@pytest.mark.skipif(not FIXTURE.exists(), reason="Frozen Temporal fixture not present in repository checkout.")
def test_rotation_quality_reproduces_control_and_best_candidate() -> None:
    request = TemporalRotationQualityResearchRequest(
        source_run_id="20260816T181543-temporal-a5afd924",
        focus_month="2026-06",
    )
    result = evaluate_temporal_rotation_quality_export(FIXTURE.read_bytes(), request)
    assert result["control"]["replayed_ending_capital"] == pytest.approx(1_318_356.29, abs=0.01)
    assert result["best_candidate"]["candidate_id"] == "RQ-017"
    assert result["best_candidate"]["drawdown_trigger"] == pytest.approx(-0.05)
    assert result["best_candidate"]["rotation_score_tolerance"] == pytest.approx(-0.10)
    assert result["best_candidate"]["ending_capital"] == pytest.approx(2_013_752.51, abs=0.01)
    assert result["best_candidate"]["robust_vs_control"] is True


def test_rotation_quality_request_defaults_to_56_candidates() -> None:
    request = TemporalRotationQualityResearchRequest(source_run_id="temporal-run")
    assert len(request.drawdown_triggers) == 8
    assert len(request.rotation_score_tolerances) == 7
    assert len(request.drawdown_triggers) * len(request.rotation_score_tolerances) == 56


def test_simple_start_request_expands_to_default_grid() -> None:
    from market_cycle_trader_api.schemas.temporal_rotation_quality import (
        TemporalRotationQualityResearchStartRequest,
    )

    start = TemporalRotationQualityResearchStartRequest(
        source_run_id="20260816T181543-temporal-a5afd924"
    )
    request = start.to_research_request()
    assert len(request.drawdown_triggers) == 8
    assert len(request.rotation_score_tolerances) == 7
    assert 0.0 not in request.drawdown_triggers
    assert request.rotation_score_tolerances[-1] == 0.0


def test_simple_start_openapi_example_does_not_expose_grid_placeholder() -> None:
    from market_cycle_trader_api.schemas.temporal_rotation_quality import (
        TemporalRotationQualityResearchStartRequest,
    )

    schema = TemporalRotationQualityResearchStartRequest.model_json_schema()
    assert schema["example"] == {
        "source_run_id": "20260816T181543-temporal-a5afd924"
    }
    assert "drawdown_triggers" not in schema["properties"]
    assert "rotation_score_tolerances" not in schema["properties"]


def test_validation_request_defaults_to_five_folds_and_freezes_candidate_ids() -> None:
    from market_cycle_trader_api.schemas.temporal_rotation_quality import (
        TemporalRotationQualityValidationRequest,
    )

    request = TemporalRotationQualityValidationRequest(
        candidate_ids=["RQ-017", "RQ-038", "RQ-053"]
    )
    assert request.fold_count == 5
    assert request.candidate_ids == ["RQ-017", "RQ-038", "RQ-053"]


def test_validation_request_rejects_control_and_duplicate_candidates() -> None:
    from pydantic import ValidationError
    from market_cycle_trader_api.schemas.temporal_rotation_quality import (
        TemporalRotationQualityValidationRequest,
    )

    with pytest.raises(ValidationError):
        TemporalRotationQualityValidationRequest(candidate_ids=["CONTROL"])
    with pytest.raises(ValidationError):
        TemporalRotationQualityValidationRequest(candidate_ids=["RQ-017", "RQ-017"])


def test_validation_gate_requires_four_of_five_folds_and_no_metric_regression() -> None:
    from market_cycle_trader_api.services.temporal_rotation_quality import (
        ReplayResult,
        _validation_candidate_metrics,
    )

    control = ReplayResult(
        metrics={"ending_capital": 100.0, "sharpe": 2.0, "max_drawdown": -0.30, "switch_count": 100},
        fold_rows=[
            {"candidate_id": "CONTROL", "fold_id": i, "ending_capital": 100.0, "switch_count": 20, "max_drawdown": -0.2, "sharpe": 1.0, "initial_capital": 10.0, "total_return": 9.0, "blocked_rotations": 0}
            for i in range(1, 6)
        ],
        equity_rows=[],
        blocked_rows=[],
    )
    candidate = ReplayResult(
        metrics={"ending_capital": 120.0, "sharpe": 2.1, "max_drawdown": -0.29, "switch_count": 90},
        fold_rows=[
            {"candidate_id": "RQ-017", "fold_id": i, "ending_capital": 110.0 if i < 5 else 99.0, "switch_count": 18, "max_drawdown": -0.2, "sharpe": 1.0, "initial_capital": 10.0, "total_return": 10.0, "blocked_rotations": 1}
            for i in range(1, 6)
        ],
        equity_rows=[],
        blocked_rows=[],
    )
    result = _validation_candidate_metrics(candidate, control, fold_count=5)
    assert result["folds_beating_control"] == 4
    assert result["required_fold_wins"] == 4
    assert result["validation_pass"] is True


def test_caro_configuration_is_parameter_driven_and_validated() -> None:
    from pydantic import ValidationError
    from market_cycle_trader_api.schemas.temporal_rotation_quality import (
        TemporalRotationQualityCaroConfig,
        TemporalRotationQualityResearchRequest,
    )

    request = TemporalRotationQualityResearchRequest(
        source_run_id="temporal-run",
        search_method="caro",
        caro=TemporalRotationQualityCaroConfig(
            drawdown_trigger_min=-0.20,
            drawdown_trigger_max=-0.02,
            rotation_score_tolerance_min=-0.30,
            rotation_score_tolerance_max=0.05,
            trials=240,
            seed=91,
        ),
    )
    assert request.search_method == "caro"
    assert request.caro.trials == 240
    assert request.caro.drawdown_trigger_min == pytest.approx(-0.20)
    assert request.caro.rotation_score_tolerance_max == pytest.approx(0.05)

    with pytest.raises(ValidationError):
        TemporalRotationQualityCaroConfig(
            drawdown_trigger_min=-0.02,
            drawdown_trigger_max=-0.20,
        )


def test_manual_research_requires_explicit_candidates() -> None:
    from pydantic import ValidationError
    from market_cycle_trader_api.schemas.temporal_rotation_quality import (
        TemporalRotationQualityResearchRequest,
    )

    with pytest.raises(ValidationError):
        TemporalRotationQualityResearchRequest(
            source_run_id="temporal-run",
            search_method="manual",
            manual_candidates=[],
        )


def test_validation_and_certification_protocol_is_fully_configurable() -> None:
    from market_cycle_trader_api.schemas.temporal_rotation_quality import (
        TemporalRotationQualityValidationRequest,
    )

    request = TemporalRotationQualityValidationRequest(
        kind="certification",
        fold_count=7,
        required_fold_wins=6,
        candidate_ids=["RQ-017", "RQ-053"],
        minimum_capital_lift=0.10,
        minimum_sharpe_delta=0.02,
        minimum_max_drawdown_delta=-0.01,
    )
    assert request.kind == "certification"
    assert request.fold_count == 7
    assert request.resolved_required_fold_wins() == 6
    assert request.minimum_capital_lift == pytest.approx(0.10)
    assert request.minimum_sharpe_delta == pytest.approx(0.02)
    assert request.minimum_max_drawdown_delta == pytest.approx(-0.01)


def test_validation_gate_uses_client_supplied_thresholds() -> None:
    from market_cycle_trader_api.services.temporal_rotation_quality import (
        ReplayResult,
        _validation_candidate_metrics,
    )

    control = ReplayResult(
        metrics={"ending_capital": 100.0, "sharpe": 2.0, "max_drawdown": -0.30, "switch_count": 100},
        fold_rows=[
            {"candidate_id": "CONTROL", "fold_id": i, "ending_capital": 100.0, "switch_count": 20, "max_drawdown": -0.2, "sharpe": 1.0, "initial_capital": 10.0, "total_return": 9.0, "blocked_rotations": 0}
            for i in range(1, 8)
        ],
        equity_rows=[],
        blocked_rows=[],
    )
    candidate = ReplayResult(
        metrics={"ending_capital": 111.0, "sharpe": 2.03, "max_drawdown": -0.305, "switch_count": 90},
        fold_rows=[
            {"candidate_id": "RQ-017", "fold_id": i, "ending_capital": 101.0 if i <= 6 else 99.0, "switch_count": 18, "max_drawdown": -0.2, "sharpe": 1.0, "initial_capital": 10.0, "total_return": 10.0, "blocked_rotations": 1}
            for i in range(1, 8)
        ],
        equity_rows=[],
        blocked_rows=[],
    )
    passing = _validation_candidate_metrics(
        candidate,
        control,
        fold_count=7,
        required_fold_wins=6,
        minimum_capital_lift=0.10,
        minimum_sharpe_delta=0.02,
        minimum_max_drawdown_delta=-0.01,
    )
    assert passing["folds_beating_control"] == 6
    assert passing["validation_pass"] is True

    failing = _validation_candidate_metrics(
        candidate,
        control,
        fold_count=7,
        required_fold_wins=7,
        minimum_capital_lift=0.12,
        minimum_sharpe_delta=0.04,
        minimum_max_drawdown_delta=0.0,
    )
    assert failing["capital_pass"] is False
    assert failing["sharpe_pass"] is False
    assert failing["max_drawdown_pass"] is False
    assert failing["folds_pass"] is False
    assert failing["validation_pass"] is False


def test_validation_export_contains_complete_analysis_bundle() -> None:
    import io
    import json
    import zipfile

    from market_cycle_trader_api.services.temporal_rotation_quality import (
        TEMPORAL_ROTATION_QUALITY_VALIDATION_COLLECTION,
        build_temporal_rotation_quality_validation_export,
    )

    class Collection:
        def __init__(self, document):
            self.document = document

        def find_one(self, query, projection=None):
            if query.get("id") != self.document.get("id") or query.get("research_id") != self.document.get("research_id"):
                return None
            return dict(self.document)

    class Database(dict):
        pass

    document = {
        "id": "validation-1",
        "research_id": "research-1",
        "source_run_id": "source-1",
        "kind": "certification",
        "status": "completed",
        "fold_count": 7,
        "frozen_candidates": [{"candidate_id": "RQ-017", "drawdown_trigger": -0.05, "rotation_score_tolerance": -0.10}],
        "control": {
            "candidate_id": "CONTROL",
            "ending_capital": 100.0,
            "folds": [{"candidate_id": "CONTROL", "fold_id": 1, "ending_capital": 100.0}],
        },
        "candidates": [{
            "candidate_id": "RQ-017",
            "ending_capital": 150.0,
            "validation_pass": True,
            "folds": [{"candidate_id": "RQ-017", "fold_id": 1, "ending_capital": 110.0}],
            "blocked_rotation_details": [{"fold_id": 1, "decision_timestamp": "2026-01-02", "incumbent_symbol": "A", "original_target_symbol": "B"}],
        }],
        "validation_policy": {"required_fold_wins": 6, "future_information_used_for_decision": False},
        "passing_candidate_count": 1,
        "best_validated_candidate": {"candidate_id": "RQ-017"},
    }
    db = Database({TEMPORAL_ROTATION_QUALITY_VALIDATION_COLLECTION: Collection(document)})
    raw = build_temporal_rotation_quality_validation_export(db, "research-1", "validation-1")
    with zipfile.ZipFile(io.BytesIO(raw), "r") as archive:
        names = set(archive.namelist())
        assert {
            "summary.json",
            "control.json",
            "candidates.csv",
            "folds.csv",
            "blocked_rotations.csv",
            "validation_policy.json",
            "candidate_details.json",
        }.issubset(names)
        policy = json.loads(archive.read("validation_policy.json"))
        assert policy["required_fold_wins"] == 6
        assert policy["future_information_used_for_decision"] is False
        assert "RQ-017" in archive.read("blocked_rotations.csv").decode("utf-8")


def test_strong_challenger_override_requires_frozen_baseline_for_caro() -> None:
    from pydantic import ValidationError
    from market_cycle_trader_api.schemas.temporal_rotation_quality import (
        TemporalRotationQualityCaroConfig,
        TemporalRotationQualityResearchRequest,
    )

    with pytest.raises(ValidationError):
        TemporalRotationQualityResearchRequest(
            source_run_id="temporal-run",
            search_method="caro",
            strong_challenger_override=True,
        )

    request = TemporalRotationQualityResearchRequest(
        source_run_id="temporal-run",
        search_method="caro",
        strong_challenger_override=True,
        baseline_drawdown_trigger=-0.05,
        baseline_rotation_score_tolerance=-0.10,
        caro=TemporalRotationQualityCaroConfig(
            challenger_quality_floor_min=0.40,
            challenger_quality_floor_max=0.80,
            trials=8,
        ),
    )
    assert request.baseline_drawdown_trigger == pytest.approx(-0.05)
    assert request.baseline_rotation_score_tolerance == pytest.approx(-0.10)
    assert request.caro.challenger_quality_floor_min == pytest.approx(0.40)
    assert request.caro.challenger_quality_floor_max == pytest.approx(0.80)


def test_strong_challenger_override_allows_a_strong_absolute_challenger() -> None:
    import pandas as pd

    from market_cycle_trader_api.services.temporal_rotation_quality import ReplayInputs, _replay

    day1 = pd.Timestamp("2026-01-02T00:00:00Z")
    day2 = pd.Timestamp("2026-01-05T00:00:00Z")
    data = ReplayInputs(
        summary=pd.DataFrame(),
        multi=pd.DataFrame(),
        equity=pd.DataFrame([
            {"fold_id": 1, "decision_timestamp": day1, "target_symbol": "A"},
            {"fold_id": 1, "decision_timestamp": day2, "target_symbol": "B"},
        ]),
        daily_assets=pd.DataFrame(),
        folds=pd.DataFrame(),
        return_map={
            (1, day1, "A"): -0.10,
            (1, day2, "A"): -0.05,
            (1, day2, "B"): 0.04,
        },
        score_map={
            (1, day2, "A"): 0.80,
            (1, day2, "B"): 0.68,
        },
    )

    baseline = _replay(
        data,
        candidate_id="BASE",
        drawdown_trigger=-0.05,
        rotation_score_tolerance=-0.10,
    )
    strong = _replay(
        data,
        candidate_id="STRONG",
        drawdown_trigger=-0.05,
        rotation_score_tolerance=-0.10,
        challenger_quality_floor=0.65,
    )

    assert baseline.metrics["blocked_rotations"] == 1
    assert baseline.equity_rows[-1]["chosen_target_symbol"] == "A"
    assert strong.metrics["blocked_rotations"] == 0
    assert strong.metrics["strong_challenger_overrides"] == 1
    assert strong.equity_rows[-1]["chosen_target_symbol"] == "B"
    assert strong.equity_rows[-1]["strong_challenger_override"] is True
    assert strong.metrics["ending_capital"] > baseline.metrics["ending_capital"]


def test_manual_challenger_floor_requires_strong_override_mode() -> None:
    from pydantic import ValidationError
    from market_cycle_trader_api.schemas.temporal_rotation_quality import (
        TemporalRotationQualityManualCandidate,
        TemporalRotationQualityResearchRequest,
    )

    candidate = TemporalRotationQualityManualCandidate(
        drawdown_trigger=-0.05,
        rotation_score_tolerance=-0.10,
        challenger_quality_floor=0.60,
    )
    with pytest.raises(ValidationError):
        TemporalRotationQualityResearchRequest(
            source_run_id="temporal-run",
            search_method="manual",
            manual_candidates=[candidate],
        )
