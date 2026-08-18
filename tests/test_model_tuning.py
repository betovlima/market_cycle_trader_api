from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from market_cycle_trader_api.schemas.model_tuning import ModelTuningStartRequest
from market_cycle_trader_api.services.model_tuning import (
    _compact_historical_tuning_document,
    _format_adopted_strategy_description,
    _sanitize_tuning_log_line,
    _source_anchor,
    _source_observations,
    _rank_candidates,
    _tuning_plan,
    _tuning_target_allows_locked_strategy,
    _tuning_target_strategy,
    generate_latin_hypercube_candidates,
    get_model_tuning_campaign_log,
    get_model_tuning_candidate_log,
    list_model_tuning_history,
    tuning_catalog,
)
from market_cycle_trader_api.services.model_tuning_probability import (
    PROBABILITY_MODEL,
    evolve_probability_search,
    initial_probability_state,
    propose_champion_probability_candidate,
    propose_unified_space_filling_candidate,
    unified_caro_next_mode,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "market_cycle_trader_api"
FRONT = ROOT.parent / "market_cycle_trader"


BASE_LIGHTGBM = {
    "n_estimators": 300,
    "learning_rate": 0.035,
    "max_depth": 3,
    "num_leaves": 8,
    "min_child_samples": 20,
    "min_child_weight": 5.0,
    "subsample": 0.85,
    "subsample_freq": 0,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 2.0,
    "max_bin": 255,
    "n_jobs": -1,
    "repetitions": 1,
    "seed_step": 1000,
    "random_state": 42,
}


def test_tuning_request_bounds_candidate_count_and_prior_campaign_contract() -> None:
    default = ModelTuningStartRequest()
    assert default.method == "champion_probability"
    assert default.candidate_count == 20
    assert default.baseline_job_id is None
    probability = ModelTuningStartRequest(method="champion_probability", candidate_count=20)
    assert probability.method == "champion_probability"
    reused = ModelTuningStartRequest(
        method="champion_probability",
        candidate_count=20,
        source_tuning_run_id="lhs-1",
        anchor_candidate_id=4,
        probability={"startup_trials": 20},
    )
    assert reused.source_tuning_run_id == "lhs-1"
    assert reused.anchor_candidate_id == 4
    with pytest.raises(ValidationError):
        ModelTuningStartRequest(candidate_count=3)
    expanded = ModelTuningStartRequest(candidate_count=500)
    assert expanded.candidate_count == 500
    with pytest.raises(ValidationError):
        ModelTuningStartRequest(candidate_count=2001)
    legacy_floor = ModelTuningStartRequest(method="champion_probability", candidate_count=8, probability={"startup_trials": 8})
    assert legacy_floor.probability.startup_trials == 8
    with pytest.raises(ValidationError):
        ModelTuningStartRequest(method="latin_hypercube", source_tuning_run_id="lhs-1")
    with pytest.raises(ValidationError):
        ModelTuningStartRequest(method="champion_probability", anchor_candidate_id=4)






def test_historical_tuning_compaction_keeps_only_adopted_or_validated_candidates() -> None:
    document = {
        "id": "run-old",
        "best_candidate_id": 9,
        "best_exploratory_candidate_id": 9,
        "best_champion_beating_candidate_id": 7,
        "adopted_candidate_id": 7,
        "validated_candidate_id": 9,
        "adoption_history": [{"candidate_id": 7, "strategy_id": "strategy-7"}],
        "prior_observations": [
            {"candidate_id": -1, "status": "completed", "settings": {"x": 1}, "metrics": {"ending_capital": 1}},
            {"candidate_id": -2, "status": "completed", "settings": {"x": 2}, "metrics": {"ending_capital": 2}},
        ],
        "probability_champion_history": [
            {"candidate_id": 5, "metrics": {"ending_capital": 105}},
            {"candidate_id": 7, "metrics": {"ending_capital": 107}},
            {"candidate_id": 9, "metrics": {"ending_capital": 109}},
        ],
        "event_log": [
            {"candidate_id": None, "stage": "created"},
            {"candidate_id": 5, "stage": "candidate_completed"},
            {"candidate_id": 7, "stage": "champion_promoted"},
            {"candidate_id": 9, "stage": "candidate_completed"},
        ],
        "candidates": [
            {"candidate_id": 0, "is_control": True, "status": "completed", "settings": {}, "metrics": {"ending_capital": 100}},
            {"candidate_id": 5, "status": "completed", "settings": {"x": 5}, "metrics": {"ending_capital": 105}},
            {"candidate_id": 6, "status": "failed", "settings": {"x": 6}, "failure_message": "technical"},
            {"candidate_id": 7, "status": "completed", "settings": {"x": 7}, "metrics": {"ending_capital": 107}},
            {"candidate_id": 8, "status": "cancelled", "settings": {"x": 8}},
            {"candidate_id": 9, "status": "completed", "settings": {"x": 9}, "metrics": {"ending_capital": 109}},
        ],
    }

    update = _compact_historical_tuning_document(document, next_run_id="run-new")

    assert [item["candidate_id"] for item in update["candidates"]] == [7, 9]
    assert update["prior_observations"] == []
    assert update["retained_candidate_ids"] == [7, 9]
    assert update["retained_candidate_count"] == 2
    assert update["discarded_successful_trial_count"] == 3  # one non-retained research success + 2 prior observations
    assert update["discarded_failed_trial_count"] == 1
    assert update["discarded_cancelled_trial_count"] == 1
    assert [item["candidate_id"] for item in update["probability_champion_history"]] == [7, 9]
    assert [item.get("candidate_id") for item in update["event_log"]] == [None, 7, 9]
    assert update["best_candidate_id"] == 9
    assert update["best_champion_beating_candidate_id"] == 7


def test_historical_tuning_compaction_discards_unselected_research_trials() -> None:
    document = {
        "id": "run-old",
        "best_candidate_id": 4,
        "candidates": [
            {"candidate_id": 0, "is_control": True, "status": "completed", "settings": {}, "metrics": {"ending_capital": 100}},
            {"candidate_id": 4, "status": "completed", "settings": {"x": 4}, "metrics": {"ending_capital": 120}},
        ],
        "prior_observations": [],
        "event_log": [{"candidate_id": None, "stage": "created"}],
    }
    update = _compact_historical_tuning_document(document, next_run_id="run-new")
    assert update["candidates"] == []
    assert update["retained_candidate_count"] == 0
    assert update["best_candidate_id"] is None
    assert update["discarded_successful_trial_count"] == 1


def test_automatic_latin_hypercube_to_caro_request_contract() -> None:
    pipeline = ModelTuningStartRequest(
        method="latin_hypercube_then_caro",
        candidate_count=20,
        caro_candidate_count=24,
    )
    assert pipeline.method == "latin_hypercube_then_caro"
    assert pipeline.candidate_count == 20
    assert pipeline.caro_candidate_count == 24
    with pytest.raises(ValidationError):
        ModelTuningStartRequest(method="latin_hypercube_then_caro", source_tuning_run_id="lhs-1")
    with pytest.raises(ValidationError):
        ModelTuningStartRequest(method="latin_hypercube", caro_candidate_count=12)


def test_tuning_catalog_declares_unified_caro_and_keeps_static_lhs_for_diagnostics() -> None:
    catalog = tuning_catalog()
    assert [item["id"] for item in catalog["methods"]] == ["champion_probability", "latin_hypercube"]
    unified = catalog["methods"][0]
    assert "no fixed Hypercube" in unified["description"]
    assert catalog["unified_caro"] is True
    assert catalog["dynamic_exploration"] is True
    assert catalog["legacy_pipeline_supported"] is True
    assert catalog["automatic_lhs_to_caro_handoff"] is False

def test_completed_latin_hypercube_source_imports_all_21_observations_and_anchor() -> None:
    candidates = generate_latin_hypercube_candidates(BASE_LIGHTGBM, candidate_count=20, seed=42)
    for index, candidate in enumerate(candidates):
        candidate["status"] = "completed"
        candidate["rank"] = index + 1
        candidate["metrics"] = {
            "eligible": True,
            "ending_capital": 299_000.0 + index * 1_000.0,
            "sharpe": 1.5,
            "maximum_drawdown": -0.5,
            "worst_fold_return": 0.1,
        }
    document = {"best_candidate_id": 4, "candidates": candidates}

    observations = _source_observations(document)
    anchor = _source_anchor(document, 4)

    assert len(observations) == 21
    assert {item["source_candidate_id"] for item in observations} == set(range(21))
    assert anchor["candidate_id"] == 4
    assert anchor["settings_hash"] == candidates[4]["settings_hash"]

def test_latin_hypercube_includes_control_and_unique_candidates() -> None:
    candidates = generate_latin_hypercube_candidates(BASE_LIGHTGBM, candidate_count=20, seed=42)
    assert len(candidates) == 21
    assert candidates[0]["candidate_id"] == 0
    assert candidates[0]["is_control"] is True
    assert candidates[0]["settings"] == BASE_LIGHTGBM
    assert len({item["settings_hash"] for item in candidates}) == 21
    for candidate in candidates[1:]:
        values = candidate["settings"]
        assert 220 <= values["n_estimators"] <= 380
        assert 0.020 <= values["learning_rate"] <= 0.050
        assert 2 <= values["max_depth"] <= 4
        assert 4 <= values["num_leaves"] <= min(12, 2 ** values["max_depth"])
        assert 15 <= values["min_child_samples"] <= 30
        assert 0.75 <= values["colsample_bytree"] <= 0.95
        assert 0.0 <= values["reg_alpha"] <= 0.50
        assert 1.0 <= values["reg_lambda"] <= 4.0
        for frozen in (
            "min_child_weight", "subsample", "subsample_freq", "max_bin",
            "n_jobs", "repetitions", "seed_step", "random_state",
        ):
            assert values[frozen] == BASE_LIGHTGBM[frozen]


def test_latin_hypercube_is_reproducible_for_same_seed() -> None:
    first = generate_latin_hypercube_candidates(BASE_LIGHTGBM, candidate_count=10, seed=77)
    second = generate_latin_hypercube_candidates(BASE_LIGHTGBM, candidate_count=10, seed=77)
    assert [item["settings_hash"] for item in first] == [item["settings_hash"] for item in second]


def _synthetic_observations(count: int = 9) -> list[dict]:
    initial = generate_latin_hypercube_candidates(BASE_LIGHTGBM, candidate_count=count - 1, seed=42)
    observations = []
    for index, candidate in enumerate(initial):
        row = dict(candidate)
        row["status"] = "completed"
        settings = row["settings"]
        quality = (
            1.0
            - abs(float(settings["n_estimators"]) - 330.0) / 500.0
            - abs(float(settings["learning_rate"]) - 0.025) * 2.0
            - abs(float(settings["max_depth"]) - 3.0) * 0.03
        )
        row["metrics"] = {
            "ending_capital": 300_000.0 + 100_000.0 * quality + index * 500.0,
            "sharpe": 1.45 + 0.20 * quality,
            "maximum_drawdown": -0.58 + 0.08 * quality,
            "worst_fold_return": 0.10 + 0.30 * quality,
            "risk_adjusted_compound_score": 1.0 + quality,
        }
        observations.append(row)
    return observations


def test_tuning_catalog_declares_integrated_worker_prior_reuse_and_reproducibility_guard() -> None:
    catalog = tuning_catalog()
    assert {item["id"] for item in catalog["methods"]} == {"latin_hypercube", "champion_probability"}
    assert catalog["dedicated_worker"] is False
    assert catalog["execution_mode"] == "integrated_api_worker"
    assert catalog["market_data_access"] == "database_only"
    assert catalog["prior_campaign_reuse"] is True
    assert catalog["historical_trial_retention"] == "active_campaign_only"
    assert catalog["durable_candidate_retention"] == "adopted_or_validated_only"
    assert catalog["technical_failure_retention"] == "campaign_counters_and_summary_only"
    assert catalog["historical_cleanup_trigger"] == "next_tuning_campaign_start"
    assert catalog["reproducibility_guard"] == "frozen_execution_snapshot_and_market_data_signature"
    assert catalog["probability"]["probability_model"] == PROBABILITY_MODEL
    assert catalog["recommended_method"] == "champion_probability"
    assert catalog["probability"]["default_minimum_exploration_trials"] == 10
    assert catalog["probability"]["search_policy"] == "dynamic_space_filling_plus_sequential_adaptive_trust_region"
    assert catalog["probability"]["space_filling_sampler"] == "latin_hypercube_maximin"
    assert catalog["probability"]["global_exploration_never_zero"] is True
    assert len(catalog["search_space"]) == 8


def test_champion_probability_uses_prior_observations_and_explicit_anchor() -> None:
    catalog = tuning_catalog()
    prior = _synthetic_observations()
    anchor = max(prior, key=lambda item: item["metrics"]["ending_capital"])
    document = {
        "seed": 42,
        "search_space": catalog["search_space"],
        "base_model_values": anchor["settings"],
        "prior_observations": prior,
        "candidates": [],
        "probability_anchor": {"metrics": anchor["metrics"]},
        "probability_config": {
            "min_capital_improvement": 0.03,
            "sharpe_tolerance": 0.05,
            "drawdown_tolerance": 0.03,
            "min_worst_fold_return": 0.0,
            "candidate_pool_size": 512,
            "exploration_weight": 0.15,
        },
    }
    proposed = propose_champion_probability_candidate(document)
    assert proposed["kind"] == "champion_probability"
    assert proposed["candidate_id"] == max(item["candidate_id"] for item in prior) + 1
    assert proposed["settings_hash"] not in {item["settings_hash"] for item in prior}
    proposal = proposed["proposal"]
    assert proposal["observation_count"] == len(prior)
    assert proposal["thresholds"]["baseline_capital"] == pytest.approx(anchor["metrics"]["ending_capital"])
    assert proposal["thresholds"]["capital"] == pytest.approx(anchor["metrics"]["ending_capital"] * 1.03)
    assert 0.0 <= proposal["estimated_probability_beats_champion"] <= 1.0
    assert proposal["promising_region"]



def test_adaptive_caro_promotes_observed_champion_and_contracts_after_repeated_misses() -> None:
    catalog = tuning_catalog()
    base = _synthetic_observations(count=5)[0]
    anchor = {
        "source": "fresh_control",
        "candidate_id": 0,
        "settings": base["settings"],
        "metrics": {
            "ending_capital": 300_000.0,
            "sharpe": 1.50,
            "maximum_drawdown": -0.50,
            "worst_fold_return": 0.10,
            "risk_adjusted_compound_score": 1.0,
        },
    }
    document = {
        "search_space": catalog["search_space"],
        "base_model_values": base["settings"],
        "probability_anchor": anchor,
        "probability_state": initial_probability_state(),
        "probability_config": {
            "min_capital_improvement": 0.03,
            "sharpe_tolerance": 0.05,
            "drawdown_tolerance": 0.03,
            "min_worst_fold_return": 0.0,
        },
    }
    candidate = {
        "candidate_id": 7,
        "kind": "champion_probability",
        "settings": base["settings"],
        "settings_hash": "candidate-7",
        "job_id": "job-7",
    }
    winning_metrics = {
        "ending_capital": 315_000.0,
        "sharpe": 1.55,
        "maximum_drawdown": -0.49,
        "worst_fold_return": 0.12,
        "risk_adjusted_compound_score": 1.2,
    }
    gate = {"passed": True}
    evolved = evolve_probability_search(document, candidate, winning_metrics, gate)
    assert evolved["champion_promoted"] is True
    assert evolved["probability_anchor"]["candidate_id"] == 7
    assert evolved["state"]["champion_revision"] == 1
    assert evolved["state"]["trust_region_radius"] > initial_probability_state()["trust_region_radius"]

    miss_document = dict(document)
    miss_document["probability_state"] = initial_probability_state()
    radius_before = miss_document["probability_state"]["trust_region_radius"]
    for candidate_id in (8, 9, 10):
        miss = dict(candidate, candidate_id=candidate_id, settings_hash=f"candidate-{candidate_id}")
        result = evolve_probability_search(miss_document, miss, winning_metrics, {"passed": False})
        miss_document["probability_state"] = result["state"]
    assert miss_document["probability_state"]["trust_region_radius"] < radius_before
    assert miss_document["probability_state"]["no_improvement_streak"] == 3


def test_adaptive_caro_proposal_reports_dynamic_trust_region() -> None:
    catalog = tuning_catalog()
    prior = _synthetic_observations()
    anchor = max(prior, key=lambda item: item["metrics"]["ending_capital"])
    document = {
        "seed": 42,
        "search_space": catalog["search_space"],
        "base_model_values": anchor["settings"],
        "prior_observations": prior,
        "candidates": [],
        "probability_anchor": {"candidate_id": anchor["candidate_id"], "settings": anchor["settings"], "metrics": anchor["metrics"]},
        "probability_state": {**initial_probability_state(), "trust_region_radius": 0.11, "adaptive_trials_completed": 5},
        "probability_config": {
            "min_capital_improvement": 0.03,
            "sharpe_tolerance": 0.05,
            "drawdown_tolerance": 0.03,
            "min_worst_fold_return": 0.0,
            "candidate_pool_size": 512,
            "exploration_weight": 0.15,
        },
    }
    proposed = propose_champion_probability_candidate(document)
    assert proposed["proposal"]["trust_region_radius"] == pytest.approx(0.11)
    assert proposed["proposal"]["champion_candidate_id"] == anchor["candidate_id"]
    assert proposed["proposal"]["pool_composition"]["global_fraction"] < 0.30


def test_tuning_routes_and_integrated_execution_contract() -> None:
    main = (SRC / "main.py").read_text(encoding="utf-8")
    router = (SRC / "api" / "routers" / "model_tuning.py").read_text(encoding="utf-8")
    service = (SRC / "services" / "model_tuning.py").read_text(encoding="utf-8")
    jobs_router = (SRC / "api" / "routers" / "jobs.py").read_text(encoding="utf-8")
    strategy_lab = (SRC / "services" / "strategy_lab.py").read_text(encoding="utf-8")

    assert "application.include_router(model_tuning.router, dependencies=research_access)" in main
    assert '@router.get("/sources")' in router
    assert "threading.Thread(target=run_model_tuning" in service
    assert '"execution_mode": "integrated_api_worker"' in service
    assert "recover_integrated_model_tuning_runs" in service
    assert "champion_gate_passed" in service
    assert "execution_request_override" in jobs_router
    assert 'request_payload["research_market_data_mode"] = "database_only"' in jobs_router
    assert '"research_market_data_mode": "backtest_bootstrap_missing"' in jobs_router
    assert 'request_snapshot["research_market_data_mode"] = "database_only"' in service
    assert "MarketDataSignatureMismatch" in service
    assert '"tuning_summary_only": tuning_run_id is not None' in jobs_router
    assert '"tuning_summary_only": {"$ne": True}' in strategy_lab


def test_frontend_uses_strategy_catalog_as_research_baseline_and_keeps_advanced_controls_optional() -> None:
    panel = (FRONT / "src" / "features" / "ModelTuningPanel.jsx").read_text(encoding="utf-8")
    assert "control?.model_tuning_strategy_id || control?.candidate_strategy_id" in panel
    assert "apiFetch(`${API}/admin/strategies`)" in panel
    assert "select-for-model-tuning" in panel
    assert "strategyStatusFilter" in panel
    assert "All statuses" in panel
    assert "Choose any Strategy from the catalog. Lifecycle status is guidance, not a research gate." in panel
    assert "1. BASELINE" in panel and "2. RESEARCH" in panel and "3. TUNING" in panel and "4. RESULTS" in panel
    assert "source_tuning_run_id: run.id" in panel
    assert "anchor_candidate_id" not in panel
    assert "Clone the protected Strategy before starting model tuning." not in panel
    assert "Selected Strategy baseline" in panel
    assert "Continue Research" in panel
    assert "Advanced CARO settings" in panel
    assert "useState(PROBABILITY_METHOD)" in panel
    assert "Start Unified CARO" in panel
    assert "effectivePlan?.search_space || catalog.search_space" in panel
    assert "/admin/model-tuning/workers" not in panel
    for protected_key in (
        "n_estimators", "learning_rate", "max_depth", "num_leaves",
        "min_child_samples", "colsample_bytree", "reg_alpha", "reg_lambda",
    ):
        assert protected_key not in panel


def test_tuning_target_prefers_explicit_catalog_selection_and_status_does_not_gate_research(monkeypatch) -> None:
    monkeypatch.setattr(
        "market_cycle_trader_api.services.model_tuning.get_strategy_control",
        lambda _db: {
            "model_tuning_strategy_id": "winner-1",
            "candidate_strategy_id": "candidate-1",
            "research_strategy_id": "research-1",
        },
    )
    monkeypatch.setattr(
        "market_cycle_trader_api.services.model_tuning.get_strategy",
        lambda _db, strategy_id: {"id": strategy_id, "status": "winner", "locked": True},
    )
    monkeypatch.setattr(
        "market_cycle_trader_api.services.model_tuning.get_strategy_model_snapshot",
        lambda _db, strategy_id: {"family": "lightgbm_utility", "profile_id": strategy_id},
    )
    strategy, snapshot, source = _tuning_target_strategy(object())
    assert strategy["id"] == "winner-1"
    assert snapshot["profile_id"] == "winner-1"
    assert source == "model_tuning_selection"
    assert _tuning_target_allows_locked_strategy({"status": "promoted_candidate", "locked": True}) is True
    assert _tuning_target_allows_locked_strategy({"status": "winner", "locked": True}) is True
    assert _tuning_target_allows_locked_strategy({"status": "superseded", "locked": True}) is True


def test_caro_baseline_is_resolved_from_selected_strategy_by_technical_compatibility() -> None:
    service = (SRC / "services" / "model_tuning.py").read_text(encoding="utf-8")
    assert 'selected_strategy_id = str(control.get("model_tuning_strategy_id") or "").strip()' in service
    assert '"strategy_profile_id": str(strategy["id"])' in service
    assert '"strategy_configuration_hash": strategy.get("configuration_hash")' in service
    assert '"research_model_settings_hash": model_snapshot.get("settings_hash")' in service
    assert 'The current protected Strategy is not a Candidate tuning target.' not in service


def test_tuning_adoption_always_preserves_source_by_creating_working_strategy() -> None:
    service = (SRC / "services" / "model_tuning.py").read_text(encoding="utf-8")
    adoption = service[service.index("def adopt_model_tuning_candidate"):service.index("def _public_candidate")]
    assert "source_status" not in adoption
    assert "created = create_strategy(" in adoption
    assert "prepare_strategy_for_backtest_candidate(" in adoption
    assert "select_research_strategy(" in adoption
    assert '"source_strategy_preserved": True' in adoption


def test_tuning_workspace_is_not_rendered_inside_model_settings() -> None:
    settings_panel = (FRONT / "src" / "features" / "ModelResearchSettingsPanel.jsx").read_text(encoding="utf-8")
    backtest_page = (FRONT / "src" / "features" / "backtest" / "components" / "BacktestPage.jsx").read_text(encoding="utf-8")
    assert "ModelTuningPanel" not in settings_panel
    assert "ModelTuningPanel" in backtest_page
    assert "Simulation Backtest" in backtest_page
    assert "Research Lab" in backtest_page


def test_export_contains_caro_prior_evidence_and_anchor_metadata() -> None:
    service = (SRC / "services" / "model_tuning.py").read_text(encoding="utf-8")
    assert "model_tuning_prior_observations.csv" in service
    assert '"prior_observations"' in service
    assert '"probability_anchor"' in service
    assert '"probability_state"' in service
    assert '"probability_champion_history"' in service
    assert '"market_data_cutoff_date"' in service
    assert '"expected_market_data_signature_sha256"' in service


def test_tuning_diagnostic_log_redacts_credentials() -> None:
    raw = (
        "Authorization: Bearer top-secret "
        "MONGO_URL=mongodb://user:password@example.test/db "
        "api_key=abc123 password=hunter2"
    )
    clean = _sanitize_tuning_log_line(raw)
    assert "top-secret" not in clean
    assert "user:password" not in clean
    assert "abc123" not in clean
    assert "hunter2" not in clean
    assert "***" in clean


def test_tuning_diagnostic_routes_and_front_actions_exist() -> None:
    router = (SRC / "api" / "routers" / "model_tuning.py").read_text(encoding="utf-8")
    service = (SRC / "services" / "model_tuning.py").read_text(encoding="utf-8")
    panel = (FRONT / "src" / "features" / "ModelTuningPanel.jsx").read_text(encoding="utf-8")

    assert '@router.get("/{run_id}/log")' in router
    assert '@router.get("/{run_id}/candidates/{candidate_id}/log")' in router
    assert "get_model_tuning_campaign_log" in service
    assert "get_model_tuning_candidate_log" in service
    assert '"event_log"' in service
    assert '"diagnostic_log"' in service
    assert "_TUNING_LOG_MAX_EVENTS" in service
    assert "openCampaignLog" in panel
    assert "openCandidateLog" in panel
    assert "Copy log" in panel
    assert "Download .txt" in panel


def test_candidate_log_reads_retained_failed_job_and_redacts_it() -> None:
    class Collection:
        def __init__(self, rows):
            self.rows = rows
        def find_one(self, query, *args, **kwargs):
            return next((row for row in self.rows if all(row.get(k) == v for k, v in query.items())), None)

    class FakeDb(dict):
        def __getitem__(self, key):
            return super().__getitem__(key)

    from market_cycle_trader_api.infrastructure.persistence.mongo_repository import JOBS_COLLECTION, MODEL_TUNING_RUNS_COLLECTION

    db = FakeDb({
        MODEL_TUNING_RUNS_COLLECTION: Collection([{
            "id": "run-1",
            "status": "failed",
            "phase": "running_candidate",
            "method": "champion_probability",
            "event_log": [{"at": "2026-08-12T12:00:00Z", "level": "error", "stage": "candidate_failed", "candidate_id": 0, "job_id": "job-1", "message": "Control failed"}],
            "candidates": [{"candidate_id": 0, "kind": "control", "is_control": True, "status": "failed", "job_id": "job-1", "error": "The candidate backtest did not complete successfully."}],
        }]),
        JOBS_COLLECTION: Collection([{
            "id": "job-1",
            "status": "failed",
            "stage": "Backtest failed",
            "return_code": 1,
            "logs": [
                "Starting engine",
                "Authorization: Bearer secret-token",
                "ValueError: frozen snapshot mismatch",
            ],
        }]),
    })

    candidate = get_model_tuning_candidate_log(db, "run-1", 0)
    assert candidate["failure_type"] == "ValueError"
    assert "secret-token" not in candidate["log_text"]
    assert "frozen snapshot mismatch" in candidate["log_text"]
    campaign = get_model_tuning_campaign_log(db, "run-1")
    assert "Control failed" in campaign["log_text"]


def test_v204_tuning_reproducibility_contract_is_control_frozen_and_preflighted() -> None:
    service = (SRC / "services" / "model_tuning.py").read_text(encoding="utf-8")
    engine = (SRC / "engine" / "compound_rotation_backtest.py").read_text(encoding="utf-8")

    assert 'expected_market_data_signature: str | None = None' in service
    assert 'market_data_signature_source = "frozen_campaign_snapshot"' in service
    assert 'candidate_request["expected_market_data_signature_sha256"] = expected_signature or None' in service
    assert '"market_data_signature_established_by_candidate_id": candidate_id' in service
    assert '"source": "fresh_control"' in service
    assert 'active_backtest = db[JOBS_COLLECTION].find_one' in service
    assert 'missing_assets = [symbol for symbol in config.assets if symbol not in bars_by_symbol]' in engine
    assert engine.index('expected_market_data_signature_sha256') < engine.index('results = run_rotation_models(')


def test_derived_caro_uses_source_campaign_frozen_context() -> None:
    from market_cycle_trader_api.services.model_tuning import _frozen_execution_context_from_campaign

    digest = "a" * 64
    source = {
        "schema_version": 3,
        "execution_request_snapshot": {
            "end_date": "2026-08-11",
            "analysis_end_date": "2026-08-11",
            "research_market_data_mode": "database_only",
        },
        "execution_context_hash": "ctx-1",
        "expected_market_data_signature_sha256": digest,
        "market_data_cutoff_date": "2026-08-11",
        "strategy_profile_id": "strategy-1",
        "strategy_profile_name": "Strategy 1",
        "strategy_profile_revision": 7,
        "strategy_configuration_hash": "strategy-hash",
        "model_family": "lightgbm_utility",
        "model_label": "LightGBM Utility",
        "baseline_execution": {"job_id": "baseline-should-not-be-read"},
    }

    class FailDb(dict):
        def __getitem__(self, key):
            raise AssertionError(f"Source campaign context must not read baseline collection: {key}")

    context = _frozen_execution_context_from_campaign(FailDb(), source)
    assert context["market_data_signature_sha256"] == digest
    assert context["market_data_cutoff_date"] == "2026-08-11"
    assert context["context_hash"] == "ctx-1"
    assert context["request"]["research_market_data_mode"] == "database_only"


def test_v206_model_tuning_stop_cancels_active_candidate_and_front_is_explicit() -> None:
    service = (SRC / "services" / "model_tuning.py").read_text(encoding="utf-8")
    jobs = (SRC / "services" / "jobs.py").read_text(encoding="utf-8")
    panel = (FRONT / "src" / "features" / "ModelTuningPanel.jsx").read_text(encoding="utf-8")

    assert "request_job_cancel(current_job_id" in service
    assert '"cancelling_active_candidate"' in service
    assert '"cancelling_temporal_model_candidate"' in service
    assert '"candidates.$.status": "cancelled"' in service
    assert '"cancelled_candidates": int(document.get("cancelled_candidates") or 0)' in service
    assert "_ACTIVE_JOB_PROCESSES" in jobs
    assert '"cancel_requested": True' in jobs
    assert '"status": "cancelled"' in jobs
    assert "Start Latin Hypercube" in panel
    assert "Start Unified CARO" in panel
    assert "Stopping…" in panel
    assert "The active tuning candidate is being cancelled" in panel
    assert "Active candidates will finish safely" not in panel


def test_v206_stop_endpoint_requests_immediate_current_job_cancellation(monkeypatch) -> None:
    from copy import deepcopy
    from market_cycle_trader_api.services import model_tuning as tuning_service
    from market_cycle_trader_api.infrastructure.persistence.mongo_repository import MODEL_TUNING_RUNS_COLLECTION

    class Collection:
        def __init__(self, document):
            self.document = deepcopy(document)

        def find_one(self, query, *args, **kwargs):
            if query.get("id") == self.document.get("id"):
                return deepcopy(self.document)
            return None

        def find_one_and_update(self, query, update, **kwargs):
            assert query.get("id") == self.document.get("id")
            self.document.update(deepcopy(update.get("$set") or {}))
            return deepcopy(self.document)

    run = {
        "id": "run-stop-1",
        "status": "running",
        "phase": "running_candidate",
        "current_candidate_id": 11,
        "current_job_id": "job-11",
        "candidates": [{"candidate_id": 11, "status": "running", "job_id": "job-11"}],
    }
    collection = Collection(run)
    db = {MODEL_TUNING_RUNS_COLLECTION: collection}
    cancelled = []

    monkeypatch.setattr(tuning_service, "request_job_cancel", lambda job_id, *, reason: cancelled.append((job_id, reason)) or True)
    monkeypatch.setattr(tuning_service, "_append_campaign_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(tuning_service, "public_model_tuning_run", lambda _db, document: document)

    result = tuning_service.request_model_tuning_stop(db, "run-stop-1")

    assert result["status"] == "stop_requested"
    assert result["phase"] == "cancelling_active_candidate"
    assert result["stop_requested"] is True
    assert cancelled and cancelled[0][0] == "job-11"


def test_v206_job_cancel_sets_authoritative_flag_and_terminates_local_process(monkeypatch) -> None:
    from copy import deepcopy
    from market_cycle_trader_api.services import jobs as jobs_service
    from market_cycle_trader_api.infrastructure.persistence.mongo_repository import JOBS_COLLECTION

    class JobCollection:
        def __init__(self):
            self.document = {"id": "job-1", "status": "running"}

        def find_one(self, query, *args, **kwargs):
            return deepcopy(self.document) if query.get("id") == "job-1" else None

        def update_one(self, query, update, *args, **kwargs):
            assert query.get("id") == "job-1"
            self.document.update(deepcopy(update.get("$set") or {}))

    class Process:
        def __init__(self):
            self.terminated = False

        def poll(self):
            return None if not self.terminated else -15

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return -15

    collection = JobCollection()
    db = {JOBS_COLLECTION: collection}
    process = Process()
    monkeypatch.setattr(jobs_service, "database", lambda: db)
    with jobs_service._ACTIVE_JOB_PROCESSES_LOCK:
        jobs_service._ACTIVE_JOB_PROCESSES["job-1"] = process
    try:
        assert jobs_service.request_job_cancel("job-1", reason="Stop tuning") is True
        assert collection.document["cancel_requested"] is True
        assert collection.document["cancel_reason"] == "Stop tuning"
        assert process.terminated is True
    finally:
        with jobs_service._ACTIVE_JOB_PROCESSES_LOCK:
            jobs_service._ACTIVE_JOB_PROCESSES.pop("job-1", None)


def test_tuning_uses_physical_content_addressed_market_snapshot() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "market_cycle_trader_api"
    service = (root / "services" / "model_tuning.py").read_text(encoding="utf-8")
    market = (root / "engine" / "market_data.py").read_text(encoding="utf-8")
    snapshot = (root / "services" / "model_tuning_market_snapshot.py").read_text(encoding="utf-8")

    assert "_ensure_campaign_market_snapshot" in service
    assert 'candidate_request["research_market_data_snapshot_id"] = snapshot_id or None' in service
    assert '"frozen_tuning_snapshot"' in market
    assert "freeze_tuning_market_snapshot" in snapshot
    assert "SourceMarketDataSnapshotMismatch" in snapshot


def test_v209_caro_source_requires_pristine_completed_latin_campaign() -> None:
    from market_cycle_trader_api.services.model_tuning import _is_pristine_completed_latin_campaign

    completed = generate_latin_hypercube_candidates(BASE_LIGHTGBM, candidate_count=4, seed=42)
    for candidate in completed:
        candidate["status"] = "completed"
        candidate["metrics"] = {"eligible": True}

    pristine = {
        "status": "completed",
        "method": "latin_hypercube",
        "total_candidates": len(completed),
        "completed_candidates": len(completed),
        "failed_candidates": 0,
        "cancelled_candidates": 0,
        "candidates": completed,
    }
    assert _is_pristine_completed_latin_campaign(pristine) is True

    failed = dict(pristine)
    failed["failed_candidates"] = 1
    failed["completed_candidates"] = len(completed) - 1
    failed["candidates"] = [dict(item) for item in completed]
    failed["candidates"][-1]["status"] = "failed"
    assert _is_pristine_completed_latin_campaign(failed) is False

    cancelled = dict(pristine)
    cancelled["cancelled_candidates"] = 1
    cancelled["completed_candidates"] = len(completed) - 1
    cancelled["candidates"] = [dict(item) for item in completed]
    cancelled["candidates"][-1]["status"] = "cancelled"
    assert _is_pristine_completed_latin_campaign(cancelled) is False

    stopped = dict(pristine)
    stopped["status"] = "stopped"
    assert _is_pristine_completed_latin_campaign(stopped) is False


def test_v209_restart_invalidates_unfinished_campaign_instead_of_resuming(monkeypatch) -> None:
    from copy import deepcopy
    from market_cycle_trader_api.services import model_tuning as tuning_service
    from market_cycle_trader_api.infrastructure.persistence.mongo_repository import (
        JOBS_COLLECTION,
        MODEL_TUNING_RUNS_COLLECTION,
    )

    class RunCollection:
        def __init__(self, document):
            self.document = deepcopy(document)

        def find(self, query, *args, **kwargs):
            return [deepcopy(self.document)]

        def update_one(self, query, update, *args, **kwargs):
            self.document.update(deepcopy(update.get("$set") or {}))
            for key, value in (update.get("$inc") or {}).items():
                self.document[key] = int(self.document.get(key) or 0) + int(value)

    class JobCollection:
        def __init__(self, document):
            self.document = deepcopy(document)

        def update_one(self, query, update, *args, **kwargs):
            self.document.update(deepcopy(update.get("$set") or {}))

    run_collection = RunCollection({
        "id": "run-restart-1",
        "status": "running",
        "phase": "running_candidate",
        "execution_mode": "integrated_api_worker",
        "current_candidate_id": 2,
        "current_job_id": "job-2",
        "completed_candidates": 2,
        "failed_candidates": 0,
        "cancelled_candidates": 0,
        "candidates": [
            {"candidate_id": 0, "status": "completed"},
            {"candidate_id": 1, "status": "completed"},
            {"candidate_id": 2, "status": "running", "job_id": "job-2"},
            {"candidate_id": 3, "status": "pending"},
        ],
    })
    job_collection = JobCollection({"id": "job-2", "status": "running"})
    db = {
        MODEL_TUNING_RUNS_COLLECTION: run_collection,
        JOBS_COLLECTION: job_collection,
    }

    monkeypatch.setattr(tuning_service, "_cleanup_job_artifacts", lambda _db, _job_id: None)
    monkeypatch.setattr(tuning_service, "_append_campaign_event", lambda *args, **kwargs: None)

    recovered = tuning_service.recover_integrated_model_tuning_runs(db)

    assert recovered == 1
    assert run_collection.document["status"] == "stopped"
    assert run_collection.document["phase"] == "invalidated_after_restart"
    assert run_collection.document["current_candidate_id"] is None
    assert run_collection.document["current_job_id"] is None
    statuses = {item["candidate_id"]: item["status"] for item in run_collection.document["candidates"]}
    assert statuses == {0: "completed", 1: "completed", 2: "cancelled", 3: "cancelled"}
    assert run_collection.document["cancelled_candidates"] == 2
    assert job_collection.document["status"] == "cancelled"


def test_v209_failure_is_terminal_and_incomplete_campaign_cannot_complete() -> None:
    service = (SRC / "services" / "model_tuning.py").read_text(encoding="utf-8")
    assert 'else "CandidateFailed"' in service
    assert '"candidate_failed_terminal"' in service
    assert '"IncompleteTuningCampaign"' in service
    assert 'Campaign completed with every candidate successful.' in service
    assert 'threading.Thread(target=run_model_tuning, args=(run_id,), daemon=True).start()' in service
    recovery = service[service.index("def recover_integrated_model_tuning_runs"):service.index("def request_model_tuning_stop")]
    assert "threading.Thread(target=run_model_tuning" not in recovery
    assert '"invalidated_after_restart"' in recovery


def test_tuning_completed_candidate_can_be_used_without_research_gates() -> None:
    from market_cycle_trader_api.schemas.model_tuning import ModelTuningAdoptRequest

    request = ModelTuningAdoptRequest()
    assert request.reason is None

    service = (SRC / "services" / "model_tuning.py").read_text(encoding="utf-8")
    adoption = service[service.index("def adopt_model_tuning_candidate"):service.index("def _public_candidate")]
    assert "Only a completed tuning candidate can be adopted." in adoption
    assert "positive-fold robustness gate" not in adoption
    assert "Champion robustness gate" not in adoption
    assert "Wait for the tuning campaign to finish before adopting" not in adoption
    assert '"ready_for_backtest": True' in adoption
    assert '"adoption_requires_final_backtest": False' in service

    panel = (FRONT / "src" / "features" / "ModelTuningPanel.jsx").read_text(encoding="utf-8")
    assert "Promote to Backtest" in panel
    assert "ModelTuningHistory" not in panel
    assert "/admin/model-tuning/history?limit=100" not in panel
    assert "const [reason, setReason]" not in panel
    assert "metrics.eligible &&" not in panel
    assert "candidate.champion_gate_passed === true" not in panel
    assert "Champion Gate and fold eligibility are informational" not in panel


def test_adopted_strategy_description_uses_tuning_metrics_without_user_text() -> None:
    description = _format_adopted_strategy_description(
        {
            "id": "20260813T143142-tune-56941ff2",
            "method": "champion_probability",
            "model_label": "LightGBM Utility",
            "market_data_cutoff_date": "2026-08-11",
        },
        {
            "candidate_id": 20,
            "metrics": {
                "ending_capital": 757587.1075783046,
                "strategy_return": 74.75871075783046,
                "cagr": 1.046691911885945,
                "sharpe": 1.828050331421643,
                "maximum_drawdown": -0.33834821699534,
                "worst_fold_return": 1.0360870631508035,
            },
        },
        {"name": "Current Champion", "revision": 4},
    )
    assert "candidate #20" in description
    assert "capital $757,587.11" in description
    assert "CAGR +104.67%" in description
    assert "Sharpe 1.828" in description
    assert "Max DD -33.83%" in description
    assert "Worst Fold +103.61%" in description
    assert "2026-08-11" in description
    assert len(description) <= 500


def test_use_in_backtest_always_creates_a_new_strategy_catalog_entry() -> None:
    service = (SRC / "services" / "model_tuning.py").read_text(encoding="utf-8")
    adoption = service[service.index("def adopt_model_tuning_candidate"):service.index("def _public_candidate")]
    assert "source_status" not in adoption
    assert "create_strategy(" in adoption
    assert "prepare_strategy_for_backtest_candidate(" in adoption
    assert '"derived_strategy_created": True' in adoption
    assert '"source_strategy_preserved": True' in adoption
    assert '"auto_candidate_after_backtest": True' in adoption

    panel = (FRONT / "src" / "features" / "ModelTuningPanel.jsx").read_text(encoding="utf-8")
    assert "selected as BACKTEST" in panel
    assert "becomes the active CANDIDATE automatically" in panel


def test_model_tuning_history_returns_previous_campaigns_with_best_candidate_and_promotions() -> None:
    from copy import deepcopy
    from market_cycle_trader_api.infrastructure.persistence.mongo_repository import MODEL_TUNING_RUNS_COLLECTION

    class Cursor:
        def __init__(self, rows):
            self.rows = list(rows)
        def sort(self, key, direction):
            reverse = int(direction) < 0
            self.rows.sort(key=lambda row: row.get(key) or "", reverse=reverse)
            return self
        def limit(self, count):
            self.rows = self.rows[:count]
            return self
        def __iter__(self):
            return iter(deepcopy(self.rows))

    class Collection:
        def __init__(self, rows):
            self.rows = rows
        def find(self, query, *args, **kwargs):
            assert query == {}
            return Cursor(self.rows)

    rows = [
        {
            "id": "run-old",
            "status": "completed",
            "method": "latin_hypercube",
            "strategy_profile_name": "Older Strategy",
            "created_at": "2026-08-12T10:00:00Z",
            "completed_candidates": 2,
            "total_candidates": 2,
            "best_candidate_id": 1,
            "candidates": [
                {"candidate_id": 0, "status": "completed", "rank": 2, "metrics": {"ending_capital": 200_000.0}},
                {"candidate_id": 1, "status": "completed", "rank": 1, "metrics": {"ending_capital": 250_000.0}},
            ],
        },
        {
            "id": "run-new",
            "status": "completed",
            "method": "champion_probability",
            "strategy_profile_name": "New Strategy",
            "created_at": "2026-08-13T10:00:00Z",
            "completed_candidates": 3,
            "total_candidates": 3,
            "best_candidate_id": 2,
            "adopted_candidate_id": 2,
            "adopted_strategy_id": "strategy-backtest-2",
            "adoption_history": [
                {"candidate_id": 2, "strategy_id": "strategy-backtest-2", "at": "2026-08-13T11:00:00Z"}
            ],
            "candidates": [
                {"candidate_id": 1, "status": "completed", "rank": 2, "metrics": {"ending_capital": 700_000.0}},
                {"candidate_id": 2, "status": "completed", "rank": 1, "metrics": {"ending_capital": 1_100_000.0, "sharpe": 1.96}},
            ],
        },
    ]
    db = {MODEL_TUNING_RUNS_COLLECTION: Collection(rows)}

    history = list_model_tuning_history(db, limit=50)
    assert [item["id"] for item in history] == ["run-new", "run-old"]
    assert history[0]["best_candidate"]["candidate_id"] == 2
    assert history[0]["best_candidate"]["metrics"]["ending_capital"] == 1_100_000.0
    assert history[0]["adopted_strategy_id"] == "strategy-backtest-2"
    assert len(history[0]["adoption_history"]) == 1


def test_model_tuning_history_route_is_declared_before_dynamic_run_route() -> None:
    router = (SRC / "api" / "routers" / "model_tuning.py").read_text(encoding="utf-8")
    history_index = router.index('@router.get("/history")')
    dynamic_index = router.index('@router.get("/{run_id}")')
    assert history_index < dynamic_index
    assert "list_model_tuning_history" in router


def test_historical_tuning_candidate_can_restore_frozen_strategy_after_source_revision_changed(monkeypatch) -> None:
    import json
    from copy import deepcopy
    from market_cycle_trader_api.infrastructure.persistence.mongo_repository import MODEL_TUNING_RUNS_COLLECTION
    from market_cycle_trader_api.services import model_tuning as tuning_service

    winner_path = SRC / "parameterizations" / "winner-v1.13.2.json"
    frozen_request = json.loads(winner_path.read_text(encoding="utf-8"))
    document = {
        "id": "historical-run-1",
        "method": "champion_probability",
        "strategy_profile_id": "source-strategy",
        "strategy_profile_name": "Frozen Source",
        "strategy_profile_revision": 4,
        "execution_request_snapshot": frozen_request,
        "market_data_cutoff_date": "2026-08-11",
        "candidates": [{
            "candidate_id": 18,
            "status": "completed",
            "settings": deepcopy(BASE_LIGHTGBM),
            "metrics": {"ending_capital": 1_109_354.0, "sharpe": 1.966, "maximum_drawdown": -0.3145, "worst_fold_return": 0.9806},
        }],
    }

    class Collection:
        def __init__(self, row):
            self.row = row
            self.updates = []
        def find_one(self, query, *args, **kwargs):
            return deepcopy(self.row) if query.get("id") == self.row["id"] else None
        def update_one(self, query, update, *args, **kwargs):
            self.updates.append((deepcopy(query), deepcopy(update)))

    collection = Collection(document)
    db = {MODEL_TUNING_RUNS_COLLECTION: collection}
    restored = {}

    def fake_get_strategy(_db, strategy_id):
        if strategy_id == "source-strategy":
            return {"id": "source-strategy", "name": "Current Changed Source", "revision": 99, "status": "candidate"}
        return {"id": strategy_id, "name": "Derived", "revision": 4, "status": "backtest"}

    def fake_create_strategy(_db, **kwargs):
        assert kwargs["clone_from_strategy_id"] == "source-strategy"
        return {"id": "derived-strategy", "name": kwargs["name"], "revision": 1, "status": "draft"}

    def fake_update_strategy(_db, strategy_id, *, configuration, expected_revision, **kwargs):
        assert strategy_id == "derived-strategy"
        assert expected_revision == 1
        restored["configuration"] = configuration
        return {"id": strategy_id, "name": kwargs["name"], "revision": 2, "status": "draft"}

    def fake_update_strategy_model(_db, strategy_id, **kwargs):
        assert strategy_id == "derived-strategy"
        assert kwargs["expected_strategy_revision"] == 2
        assert kwargs["values"] == BASE_LIGHTGBM
        return {"id": strategy_id, "name": "Derived", "revision": 3, "status": "draft"}

    def fake_prepare(_db, strategy_id, **kwargs):
        assert kwargs["tuning_run_id"] == "historical-run-1"
        assert kwargs["tuning_candidate_id"] == 18
        return {"id": strategy_id, "name": "Derived", "revision": 4, "status": "backtest"}

    monkeypatch.setattr(tuning_service, "get_strategy", fake_get_strategy)
    monkeypatch.setattr(tuning_service, "create_strategy", fake_create_strategy)
    monkeypatch.setattr(tuning_service, "update_strategy", fake_update_strategy)
    monkeypatch.setattr(tuning_service, "update_strategy_model", fake_update_strategy_model)
    monkeypatch.setattr(tuning_service, "prepare_strategy_for_backtest_candidate", fake_prepare)
    monkeypatch.setattr(tuning_service, "get_strategy_control", lambda _db: {"revision": 7})
    monkeypatch.setattr(tuning_service, "select_research_strategy", lambda *args, **kwargs: None)

    result = tuning_service.adopt_model_tuning_candidate(
        db,
        "historical-run-1",
        18,
        reason=None,
        actor_email="admin@example.com",
    )

    assert restored["configuration"].assets == frozen_request["assets"]
    assert result["strategy"]["id"] == "derived-strategy"
    assert result["ready_for_backtest"] is True
    assert result["auto_candidate_after_backtest"] is True
    assert collection.updates
    pushed = collection.updates[-1][1]["$push"]["adoption_history"]
    assert pushed["candidate_id"] == 18
    assert pushed["strategy_id"] == "derived-strategy"


def test_absolute_utility_gate_jointly_tunes_model_and_cash_gate_with_valid_lhs() -> None:
    strategy = {
        "configuration": {
            "strategy_mode": "COMPOUND_ROTATION_SWING_ABSOLUTE_UTILITY_CASH_GATE",
            "opportunity_utility_entry_threshold": 0.28,
            "opportunity_utility_exit_threshold": 0.27,
        }
    }
    model_snapshot = {
        "family": "lightgbm_utility",
        "settings_snapshot": {"lightgbm": BASE_LIGHTGBM},
    }
    plan = _tuning_plan(strategy, model_snapshot)
    assert plan["scope"] == "joint_model_absolute_utility_cash_gate"
    names = [item["name"] for item in plan["search_space"]]
    assert names[:8] == [
        "n_estimators", "learning_rate", "max_depth", "num_leaves",
        "min_child_samples", "colsample_bytree", "reg_alpha", "reg_lambda",
    ]
    assert names[-2:] == [
        "opportunity_utility_entry_threshold",
        "opportunity_utility_exit_threshold",
    ]
    assert plan["tuned_model_parameters"] == names[:8]
    assert plan["tuned_strategy_parameters"] == names[-2:]
    assert plan["base_model_values"] == BASE_LIGHTGBM
    assert plan["base_values"]["opportunity_utility_entry_threshold"] == 0.28
    assert plan["base_values"]["opportunity_utility_exit_threshold"] == 0.27

    candidates = generate_latin_hypercube_candidates(
        plan["base_values"],
        candidate_count=12,
        seed=42,
        search_space=plan["search_space"],
    )
    assert len(candidates) == 13
    assert candidates[0]["settings"]["n_estimators"] == BASE_LIGHTGBM["n_estimators"]
    assert candidates[0]["settings"]["opportunity_utility_entry_threshold"] == 0.28
    assert candidates[0]["settings"]["opportunity_utility_exit_threshold"] == 0.27
    assert len({item["settings_hash"] for item in candidates}) == 13
    for candidate in candidates:
        values = candidate["settings"]
        assert 220 <= values["n_estimators"] <= 380
        assert 0.020 <= values["learning_rate"] <= 0.050
        assert 2 <= values["max_depth"] <= 4
        assert 4 <= values["num_leaves"] <= min(12, 2 ** values["max_depth"])
        assert 15 <= values["min_child_samples"] <= 30
        assert 0.75 <= values["colsample_bytree"] <= 0.95
        assert 0.0 <= values["reg_alpha"] <= 0.50
        assert 1.0 <= values["reg_lambda"] <= 4.0
        assert 0.22 <= values["opportunity_utility_entry_threshold"] <= 0.38
        assert 0.20 <= values["opportunity_utility_exit_threshold"] <= 0.36
        assert values["opportunity_utility_exit_threshold"] <= values["opportunity_utility_entry_threshold"]
        for frozen in (
            "min_child_weight", "subsample", "subsample_freq", "max_bin",
            "n_jobs", "repetitions", "seed_step", "random_state",
        ):
            assert values[frozen] == BASE_LIGHTGBM[frozen]



def _completed_candidate(candidate: dict, *, capital: float = 300_000.0, score: float = 1.0) -> dict:
    row = dict(candidate)
    row["status"] = "completed"
    row["metrics"] = {
        "ending_capital": capital,
        "sharpe": 1.6,
        "maximum_drawdown": -0.35,
        "worst_fold_return": 0.20,
        "risk_adjusted_compound_score": score,
    }
    return row


def test_unified_caro_starts_with_internal_space_filling_then_switches_to_probability() -> None:
    catalog = tuning_catalog()
    control = generate_latin_hypercube_candidates(BASE_LIGHTGBM, candidate_count=1, seed=42)[0]
    document = {
        "seed": 42,
        "candidate_count": 20,
        "total_candidates": 21,
        "search_space": catalog["search_space"],
        "base_tuning_values": BASE_LIGHTGBM,
        "base_model_values": BASE_LIGHTGBM,
        "prior_observations": [],
        "candidates": [_completed_candidate(control)],
        "probability_anchor": {"settings": BASE_LIGHTGBM, "metrics": _completed_candidate(control)["metrics"]},
        "baseline_execution": {"metrics": _completed_candidate(control)["metrics"]},
        "probability_state": initial_probability_state(),
        "probability_config": {
            "minimum_exploration_trials": 5,
            "initial_exploration_fraction": 0.45,
            "minimum_exploration_fraction": 0.20,
            "stagnation_recovery_trials": 4,
            "candidate_pool_size": 512,
            "exploration_weight": 0.15,
        },
    }
    for index in range(5):
        policy = unified_caro_next_mode(document)
        assert policy["mode"] == "space_filling"
        candidate = propose_unified_space_filling_candidate(document)
        assert candidate["kind"] == "unified_exploration"
        assert candidate["proposal"]["proposal_mode"] == "space_filling"
        candidate = _completed_candidate(candidate, capital=301_000.0 + index, score=1.01 + index / 100)
        evolution = evolve_probability_search(document, candidate, candidate["metrics"], {"passed": False})
        document["probability_state"] = evolution["state"]
        document["candidates"].append(candidate)

    policy = unified_caro_next_mode(document)
    assert policy["mode"] == "adaptive_probability"
    adaptive = propose_champion_probability_candidate(document)
    assert adaptive["kind"] == "champion_probability"
    assert adaptive["proposal"]["proposal_mode"] == "adaptive_probability"


def test_unified_caro_reopens_space_filling_after_adaptive_stagnation() -> None:
    catalog = tuning_catalog()
    initial = generate_latin_hypercube_candidates(BASE_LIGHTGBM, candidate_count=5, seed=19)
    candidates = [_completed_candidate(initial[0])]
    for item in initial[1:]:
        row = _completed_candidate(item)
        row["kind"] = "unified_exploration"
        candidates.append(row)
    document = {
        "seed": 42,
        "candidate_count": 20,
        "total_candidates": 21,
        "search_space": catalog["search_space"],
        "base_tuning_values": BASE_LIGHTGBM,
        "base_model_values": BASE_LIGHTGBM,
        "prior_observations": [],
        "candidates": candidates,
        "probability_anchor": {"settings": BASE_LIGHTGBM, "metrics": candidates[0]["metrics"]},
        "baseline_execution": {"metrics": candidates[0]["metrics"]},
        "probability_state": {
            **initial_probability_state(),
            "exploration_trials_completed": 5,
            "adaptive_trials_completed": 4,
            "no_improvement_streak": 4,
            "last_proposal_mode": "adaptive_probability",
        },
        "probability_config": {
            "minimum_exploration_trials": 5,
            "initial_exploration_fraction": 0.45,
            "minimum_exploration_fraction": 0.20,
            "stagnation_recovery_trials": 4,
        },
    }
    policy = unified_caro_next_mode(document)
    assert policy["mode"] == "space_filling"
    assert policy["reason"] == "stagnation_recovery"


def test_absolute_utility_catalog_declares_joint_10d_caro_and_larger_warmup(monkeypatch) -> None:
    plan = _tuning_plan(
        {"configuration": {
            "strategy_mode": "COMPOUND_ROTATION_SWING_ABSOLUTE_UTILITY_CASH_GATE",
            "opportunity_utility_entry_threshold": 0.28,
            "opportunity_utility_exit_threshold": 0.27,
        }},
        {"family": "lightgbm_utility", "settings_snapshot": {"lightgbm": BASE_LIGHTGBM}},
    )
    monkeypatch.setattr(
        "market_cycle_trader_api.services.model_tuning._current_tuning_plan",
        lambda _db: plan,
    )
    catalog = tuning_catalog(object())
    assert catalog["baseline_policy"] == "active_candidate_certified_backtest"
    assert catalog["joint_optimization"] is True
    assert len(catalog["search_space"]) == 10
    assert len(catalog["tuned_model_parameters"]) == 8
    assert len(catalog["tuned_strategy_parameters"]) == 2
    assert catalog["probability"]["default_minimum_exploration_trials"] == 12


def test_probability_candidate_supports_joint_model_and_strategy_scope() -> None:
    plan = _tuning_plan(
        {"configuration": {
            "strategy_mode": "COMPOUND_ROTATION_SWING_ABSOLUTE_UTILITY_CASH_GATE",
            "opportunity_utility_entry_threshold": 0.28,
            "opportunity_utility_exit_threshold": 0.27,
        }},
        {"family": "lightgbm_utility", "settings_snapshot": {"lightgbm": BASE_LIGHTGBM}},
    )
    search_space = plan["search_space"]
    initial = generate_latin_hypercube_candidates(
        plan["base_values"],
        candidate_count=7,
        seed=9,
        search_space=search_space,
    )
    observations = []
    for index, candidate in enumerate(initial):
        row = dict(candidate)
        row["status"] = "completed"
        row["metrics"] = {
            "ending_capital": 1_000_000.0 + index * 10_000.0,
            "sharpe": 1.8 + index * 0.01,
            "maximum_drawdown": -0.35 + index * 0.002,
            "worst_fold_return": 0.20 + index * 0.01,
            "risk_adjusted_compound_score": 1.0 + index * 0.01,
        }
        observations.append(row)
    anchor = observations[-1]
    document = {
        "seed": 42,
        "search_space": search_space,
        "base_tuning_values": plan["base_values"],
        "prior_observations": observations,
        "candidates": [],
        "probability_anchor": {"candidate_id": anchor["candidate_id"], "settings": anchor["settings"], "metrics": anchor["metrics"]},
        "probability_config": {
            "min_capital_improvement": 0.0,
            "sharpe_tolerance": 0.1,
            "drawdown_tolerance": 0.05,
            "min_worst_fold_return": 0.0,
            "candidate_pool_size": 256,
            "exploration_weight": 0.15,
        },
    }
    proposed = propose_champion_probability_candidate(document)
    values = proposed["settings"]
    assert 220 <= values["n_estimators"] <= 380
    assert 0.020 <= values["learning_rate"] <= 0.050
    assert values["opportunity_utility_exit_threshold"] <= values["opportunity_utility_entry_threshold"]
    assert set(proposed["proposal"]["promising_region"]) == {item["name"] for item in search_space}


def test_unified_ranking_keeps_control_above_non_beating_economically_weaker_candidate() -> None:
    control = {
        "candidate_id": 0, "is_control": True, "status": "completed", "champion_gate_passed": None,
        "metrics": {
            "eligible": True, "ending_capital": 1_142_410.0, "sharpe": 2.015,
            "maximum_drawdown": -0.3145, "worst_fold_return": 0.749,
            "risk_adjusted_compound_score": -1.368,
        },
    }
    weaker = {
        "candidate_id": 15, "is_control": False, "status": "completed", "champion_gate_passed": False,
        "metrics": {
            "eligible": True, "ending_capital": 218_426.0, "sharpe": 1.571,
            "maximum_drawdown": -0.5654, "worst_fold_return": 0.427,
            "risk_adjusted_compound_score": -0.4747,
        },
    }
    ranked = _rank_candidates([weaker, control])
    by_id = {item["candidate_id"]: item for item in ranked}
    assert by_id[0]["rank"] == 1
    assert by_id[15]["rank"] == 2


def test_joint_lhs_samples_cash_gate_directly_inside_valid_domain_without_diagonal_pileup() -> None:
    plan = _tuning_plan(
        {"configuration": {
            "strategy_mode": "COMPOUND_ROTATION_SWING_ABSOLUTE_UTILITY_CASH_GATE",
            "opportunity_utility_entry_threshold": 0.28,
            "opportunity_utility_exit_threshold": 0.27,
        }},
        {"family": "lightgbm_utility", "settings_snapshot": {"lightgbm": BASE_LIGHTGBM}},
    )
    candidates = generate_latin_hypercube_candidates(
        plan["base_values"], candidate_count=80, seed=42, search_space=plan["search_space"],
    )
    sampled = [item["settings"] for item in candidates if not item["is_control"]]
    assert all(row["opportunity_utility_exit_threshold"] <= row["opportunity_utility_entry_threshold"] for row in sampled)
    assert sum(
        row["opportunity_utility_exit_threshold"] == row["opportunity_utility_entry_threshold"]
        for row in sampled
    ) == 0


def test_unified_default_10d_readiness_requires_twelve_space_filling_observations() -> None:
    plan = _tuning_plan(
        {"configuration": {
            "strategy_mode": "COMPOUND_ROTATION_SWING_ABSOLUTE_UTILITY_CASH_GATE",
            "opportunity_utility_entry_threshold": 0.28,
            "opportunity_utility_exit_threshold": 0.27,
        }},
        {"family": "lightgbm_utility", "settings_snapshot": {"lightgbm": BASE_LIGHTGBM}},
    )
    initial = generate_latin_hypercube_candidates(plan["base_values"], candidate_count=11, seed=7, search_space=plan["search_space"])
    candidates = [_completed_candidate(initial[0])]
    for item in initial[1:]:
        row = _completed_candidate(item)
        row["kind"] = "unified_exploration"
        candidates.append(row)
    document = {
        "seed": 42, "candidate_count": 20, "total_candidates": 21,
        "search_space": plan["search_space"], "base_tuning_values": plan["base_values"],
        "prior_observations": [], "candidates": candidates,
        "probability_state": initial_probability_state(),
        "probability_config": {
            "initial_exploration_fraction": 0.45, "minimum_exploration_fraction": 0.20,
            "stagnation_recovery_trials": 4,
        },
    }
    policy = unified_caro_next_mode(document)
    assert policy["minimum_exploration_trials"] == 12
    assert policy["mode"] == "space_filling"
    assert policy["surrogate_readiness"]["observation_count"] == 11


def test_stagnation_recovery_resets_streak_and_enforces_adaptive_cooldown() -> None:
    document = {
        "probability_state": {
            **initial_probability_state(),
            "no_improvement_streak": 4,
            "last_proposal_mode": "adaptive_probability",
        },
        "probability_config": {"stagnation_recovery_trials": 4},
    }
    recovery = {
        "candidate_id": 99, "kind": "unified_exploration",
        "proposal": {"selection_reason": "stagnation_recovery"},
    }
    evolution = evolve_probability_search(document, recovery, {}, {"passed": False})
    state = evolution["state"]
    assert state["no_improvement_streak"] == 0
    assert state["recovery_cooldown_remaining"] == 2
    assert state["stagnation_recoveries"] == 1


def test_control_champion_gate_is_rendered_as_control() -> None:
    panel = (FRONT / "src" / "features" / "ModelTuningPanel.jsx").read_text(encoding="utf-8")
    assert "candidate.is_control\n                ? tr('Control')" in panel
    assert "candidate.is_control ? 'Baseline'" not in panel
    assert "Control stays fixed as the first reference card" not in panel


def test_tuning_control_reuses_certified_candidate_backtest_without_rerun(monkeypatch) -> None:
    from market_cycle_trader_api.services import model_tuning as tuning_service

    monkeypatch.setattr(tuning_service, "_candidate_equity_preview", lambda _db, job_id: [{"job": job_id}])
    baseline = {
        "job_id": "certified-job-7",
        "started_at": "2026-08-14T10:00:00Z",
        "finished_at": "2026-08-14T11:00:00Z",
        "strategy_configuration_hash": "cfg-7",
        "metrics": {
            "eligible": True,
            "ending_capital": 1_142_410.36,
            "sharpe": 2.0149,
            "maximum_drawdown": -0.3145,
            "worst_fold_return": 0.749,
        },
    }
    control = tuning_service._reused_baseline_control_candidate(
        object(), baseline=baseline, settings=BASE_LIGHTGBM,
    )

    assert control["candidate_id"] == 0
    assert control["is_control"] is True
    assert control["status"] == "completed"
    assert control["baseline_reused"] is True
    assert control["job_id"] == "certified-job-7"
    assert control["source_job_id"] == "certified-job-7"
    assert control["metrics"]["ending_capital"] == 1_142_410.36
    assert control["equity_preview"] == [{"job": "certified-job-7"}]

    service = (SRC / "services" / "model_tuning.py").read_text(encoding="utf-8")
    assert '"reuse_certified_candidate_backtest"' in service
    assert 'initial_completed_candidates = sum(1 for item in candidates if item.get("status") == "completed")' in service
    assert 'expected_signature=expected or None' in service
