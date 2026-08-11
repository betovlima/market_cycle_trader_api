from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from pydantic import ValidationError

from market_cycle_trader_api.engine.research_challengers import _IQNNetwork
from market_cycle_trader_api.schemas.model_research import ModelResearchJobRequest
from market_cycle_trader_api.schemas.requests import BacktestRequest
from market_cycle_trader_api.services.jobs import public_job

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "market_cycle_trader_api"


def test_model_research_request_accepts_all_supported_models() -> None:
    assert ModelResearchJobRequest(model_family="xgboost_utility").model_family == "xgboost_utility"
    assert ModelResearchJobRequest(model_family="lightgbm_utility").model_family == "lightgbm_utility"
    assert ModelResearchJobRequest(model_family="iqn").model_family == "iqn"
    with pytest.raises(ValidationError):
        ModelResearchJobRequest(model_family="unsupported")


def test_persisted_strategy_contract_remains_xgboost_only() -> None:
    source = (SRC / "schemas" / "requests.py").read_text(encoding="utf-8")
    assert 'RotationModel = Literal["xgboost_utility"]' in source
    assert 'StrategyMode = Literal["COMPOUND_ROTATION_SWING_XGBOOST"]' in source


def test_model_jobs_are_admin_only_and_only_live_capable_models_certify_strategy() -> None:
    main = (SRC / "main.py").read_text(encoding="utf-8")
    jobs = (SRC / "services" / "jobs.py").read_text(encoding="utf-8")
    assert "application.include_router(model_research.router, dependencies=admin_required)" in main
    router = (SRC / "api" / "routers" / "model_research.py").read_text(encoding="utf-8")
    assert '@router.get("/executions")' in router
    assert 'certifies_strategy = research_model_family in {"xgboost_utility", "lightgbm_utility"}' in jobs
    assert 'research_model_family = str(' in jobs
    assert 'research_model_family in {"xgboost_utility", "lightgbm_utility"}' in jobs


def test_public_job_keeps_model_identity_inside_admin_research_boundary() -> None:
    payload = public_job({
        "id": "job-1",
        "status": "queued",
        "stage": "Queued",
        "progress": 0,
        "created_at": None,
        "updated_at": None,
        "started_at": None,
        "finished_at": None,
        "completed_runs": 0,
        "total_runs": 1,
        "strategy_profile_name": "Research A",
        "research_model_family": "iqn",
        "research_model_label": "IQN",
        "request": {"research_model_settings": {"iqn": {"hidden_dim": 128}}},
        "progress_detail": {},
        "logs": [],
    })
    assert "research_model_family" not in payload
    assert "research_model_label" not in payload
    assert "request" not in payload
    assert "research_model_settings" not in payload


def test_iqn_quantile_network_returns_distribution_per_action() -> None:
    network = _IQNNetwork(
        input_dim=12,
        action_count=4,
        hidden_dim=16,
        cosine_embedding_dim=8,
        seed=17,
        device="cpu",
    )
    states = torch.zeros((3, 12), dtype=torch.float32)
    taus = torch.tensor([[0.1, 0.5, 0.9]] * 3, dtype=torch.float32)
    values = network.quantiles(states, taus)
    assert tuple(values.shape) == (3, 3, 4)
    assert torch.isfinite(values).all()


def test_dependencies_declare_both_challengers() -> None:
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "lightgbm>=4.6,<5" in requirements
    assert "torch>=2.10,<3" in requirements


def test_lightgbm_and_iqn_have_independent_validated_settings() -> None:
    from market_cycle_trader_api.schemas.model_research import (
        IQNResearchSettings,
        LightGBMResearchSettings,
        XGBoostResearchSettings,
    )

    xgboost = XGBoostResearchSettings(
        n_estimators=300,
        learning_rate=0.035,
        max_depth=3,
        min_child_weight=5.0,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=2.0,
        n_jobs=-1,
        repetitions=1,
        seed_step=1000,
        random_state=42,
    )
    assert xgboost.n_estimators == 300

    lightgbm = LightGBMResearchSettings(
        n_estimators=300,
        learning_rate=0.035,
        max_depth=3,
        num_leaves=8,
        min_child_samples=20,
        min_child_weight=5.0,
        subsample=0.85,
        subsample_freq=0,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=2.0,
        max_bin=255,
        n_jobs=-1,
        repetitions=1,
        seed_step=1000,
        random_state=42,
    )
    assert lightgbm.num_leaves == 8
    with pytest.raises(ValidationError):
        LightGBMResearchSettings(**{**lightgbm.model_dump(), "num_leaves": 9})

    iqn = IQNResearchSettings(
        training_steps=15000,
        episode_days=252,
        replay_size=30000,
        learning_starts=750,
        batch_size=128,
        learning_rate=0.0003,
        gamma=0.99,
        n_step=5,
        quantile_samples=32,
        target_quantile_samples=32,
        action_quantile_samples=32,
        evaluation_quantiles=64,
        hidden_dim=128,
        cosine_embedding_dim=64,
        target_update_steps=250,
        eval_every_steps=1000,
        epsilon_start=1.0,
        epsilon_end=0.05,
        early_stopping_enabled=False,
        early_stopping_patience=4,
        minimum_training_steps=5000,
        gradient_clip_norm=10.0,
        huber_kappa=1.0,
        repetitions=1,
        seed_step=1000,
        random_state=42,
    )
    assert iqn.training_steps == 15000


class _ModelSettingsCollection:
    def __init__(self) -> None:
        self.document = None

    def find_one(self, query):
        del query
        return None if self.document is None else dict(self.document)

    def insert_one(self, document):
        self.document = dict(document)
        return SimpleNamespace(inserted_id=document.get("_id"))

    def replace_one(self, query, document, upsert=False):
        del query, upsert
        self.document = dict(document)
        return SimpleNamespace(matched_count=1)


class _ModelSettingsDatabase:
    def __init__(self) -> None:
        self.collections = {}

    def __getitem__(self, name):
        self.collections.setdefault(name, _ModelSettingsCollection())
        return self.collections[name]


def test_execution_snapshot_freezes_model_specific_profile() -> None:
    from market_cycle_trader_api.services.model_research import execution_settings_for

    db = _ModelSettingsDatabase()
    lightgbm = execution_settings_for(db, "lightgbm_utility")
    iqn = execution_settings_for(db, "iqn")
    baseline = execution_settings_for(db, "xgboost_utility")

    assert baseline["profile_id"] == "baseline"
    assert baseline["xgboost"]["n_estimators"] == 300
    assert baseline["xgboost"]["random_state"] == 42
    assert lightgbm["profile_id"] == "baseline"
    assert lightgbm["lightgbm"]["n_estimators"] == 300
    assert lightgbm["lightgbm"]["num_leaves"] == 8
    assert iqn["profile_id"] == "baseline"
    assert iqn["iqn"]["training_steps"] == 15000


def test_lightgbm_runner_no_longer_reads_xgboost_hyperparameters() -> None:
    import inspect
    from market_cycle_trader_api.engine.research_challengers import _lightgbm_fit_models

    source = inspect.getsource(_lightgbm_fit_models)
    assert '_lightgbm_settings(config)' in source
    assert 'settings["n_estimators"]' in source
    assert 'settings["num_leaves"]' in source
    assert "config.rotation_xgb_n_estimators" not in source
    assert "config.rotation_xgb_learning_rate" not in source
    assert "config.rotation_xgb_max_depth" not in source


def test_model_settings_ui_is_api_driven_for_all_three_models() -> None:
    from market_cycle_trader_api.services.model_research import public_model_research_settings

    db = _ModelSettingsDatabase()
    payload = public_model_research_settings(db)
    models = {item["id"]: item for item in payload["models"]}
    assert models["xgboost_utility"]["editable"] is True
    assert models["xgboost_utility"]["configuration_owner"] == "model_research"
    assert models["xgboost_utility"]["fields"]
    assert models["lightgbm_utility"]["editable"] is True
    assert models["iqn"]["editable"] is True
    assert models["lightgbm_utility"]["fields"]
    assert models["iqn"]["fields"]

    frontend = (ROOT.parent / "market_cycle_trader" / "src" / "features" / "ModelResearchSettingsPanel.jsx")
    source = frontend.read_text(encoding="utf-8")
    assert "/admin/model-research/settings" in source
    for protected_key in ("n_estimators", "num_leaves", "training_steps", "replay_size", "quantile_samples"):
        assert protected_key not in source


def test_model_parameter_selector_is_embedded_in_selected_strategy_box() -> None:
    frontend_root = ROOT.parent / "market_cycle_trader" / "src" / "features"
    strategy_source = (frontend_root / "StrategySettingsPanel.jsx").read_text(encoding="utf-8")
    system_source = (frontend_root / "SystemSettingsPage.jsx").read_text(encoding="utf-8")
    panel_source = (frontend_root / "ModelResearchSettingsPanel.jsx").read_text(encoding="utf-8")
    assert "<ModelResearchSettingsPanel" in strategy_source
    assert "strategy={selected}" in strategy_source
    assert "onStrategyModelSaved={handleStrategyModelSaved}" in strategy_source
    assert "ModelResearchSettingsPanel" not in system_source
    assert "MODEL PARAMETERS" in panel_source
    assert "Model saved with this Strategy" in panel_source
    assert "Backtest uses this saved model automatically and cannot override it." in panel_source


def test_same_named_parameters_are_independent_per_model_profile() -> None:
    from market_cycle_trader_api.services.model_research import execution_settings_for

    db = _ModelSettingsDatabase()
    xgb = execution_settings_for(db, "xgboost_utility")["xgboost"]
    lgbm = execution_settings_for(db, "lightgbm_utility")["lightgbm"]
    iqn = execution_settings_for(db, "iqn")["iqn"]

    assert xgb["learning_rate"] == 0.035
    assert lgbm["learning_rate"] == 0.035
    assert iqn["learning_rate"] == 0.0003
    assert xgb is not lgbm
    for values in (xgb, lgbm, iqn):
        assert values["repetitions"] == 1
        assert values["seed_step"] == 1000
        assert values["random_state"] == 42


def test_xgboost_profile_maps_only_into_research_execution_snapshot() -> None:
    from market_cycle_trader_api.services.model_research import apply_execution_profile

    class _Config:
        def __init__(self) -> None:
            self.values = {"rotation_xgb_n_estimators": 111, "random_state": 7}

        def model_copy(self, update):
            clone = _Config()
            clone.values = {**self.values, **update}
            return clone

    original = _Config()
    snapshot = {
        "xgboost": {
            "n_estimators": 300,
            "learning_rate": 0.035,
            "max_depth": 3,
            "min_child_weight": 5.0,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.1,
            "reg_lambda": 2.0,
            "n_jobs": -1,
            "repetitions": 2,
            "seed_step": 2000,
            "random_state": 42,
        }
    }
    execution = apply_execution_profile(original, "xgboost_utility", snapshot)
    assert original.values["rotation_xgb_n_estimators"] == 111
    assert original.values["random_state"] == 7
    assert execution.values["rotation_xgb_n_estimators"] == 300
    assert execution.values["rotation_xgb_repetitions"] == 2
    assert execution.values["rotation_seed_step"] == 2000
    assert execution.values["random_state"] == 42


def test_model_execution_snapshot_binds_profile_revision_and_hash() -> None:
    from market_cycle_trader_api.services.model_research import (
        execution_settings_for,
        model_execution_snapshot,
    )

    db = _ModelSettingsDatabase()
    settings = execution_settings_for(db, "lightgbm_utility")
    first = model_execution_snapshot("lightgbm_utility", settings)
    second = model_execution_snapshot("lightgbm_utility", settings)

    assert first["family"] == "lightgbm_utility"
    assert first["profile_id"] == "baseline"
    assert first["settings_revision"] == settings["settings_revision"]
    assert len(first["settings_hash"]) == 64
    assert first["settings_hash"] == second["settings_hash"]
    assert first["settings_snapshot"]["lightgbm"]["num_leaves"] == 8


def test_live_trader_dispatch_supports_xgboost_and_lightgbm_only() -> None:
    source = (SRC / "engine" / "live_model_signal.py").read_text(encoding="utf-8")
    paper = (SRC / "services" / "paper_trading.py").read_text(encoding="utf-8")
    assert 'if model_family == "xgboost_utility"' in source
    assert 'if model_family == "lightgbm_utility"' in source
    assert "build_live_lightgbm_decision" in source
    assert "build_live_model_decision" in paper
    assert 'winner_model_settings_hash' in paper
    assert "different Winner model snapshot" in paper


def test_iqn_remains_non_promotable_until_live_engine_exists() -> None:
    strategy = (SRC / "services" / "strategy_lab.py").read_text(encoding="utf-8")
    assert 'bound_model_family == "iqn"' in strategy
    assert "IQN does not have a protected live Trader engine yet" in strategy


def test_live_utilities_do_not_require_unknown_next_session_bar() -> None:
    from market_cycle_trader_api.engine.live_policy import live_model_utilities

    class _Model:
        def predict(self, values):
            assert len(values) == 1
            return np.asarray([0.25])

    timestamp = pd.Timestamp("2026-08-10", tz="UTC")
    frame = pd.DataFrame(
        {name: [1.0] for name in __import__("market_cycle_trader_api.engine.capital_rotation", fromlist=["ROTATION_FEATURES"]).ROTATION_FEATURES},
        index=[timestamp],
    )
    utilities = live_model_utilities({"AAA": _Model()}, {"AAA": frame}, ["AAA"], timestamp)
    assert utilities.tolist() == [0.0, 0.25]


def test_backtest_model_is_read_only_and_owned_by_selected_strategy() -> None:
    frontend_root = ROOT.parent / "market_cycle_trader" / "src" / "features" / "backtest"
    backtest_source = (frontend_root / "components" / "BacktestPage.jsx").read_text(encoding="utf-8")
    workspace_source = (frontend_root / "hooks" / "useBacktestWorkspace.js").read_text(encoding="utf-8")
    jobs_source = (SRC / "api" / "routers" / "jobs.py").read_text(encoding="utf-8")

    assert "research-model-readonly" in backtest_source
    assert "Defined in Selected Strategy" in backtest_source
    assert "<select" not in backtest_source
    assert "research_model_family" not in workspace_source
    assert "model_family" not in workspace_source
    assert "`${API}/jobs`" in workspace_source
    assert "get_research_strategy_model_snapshot(db)" in jobs_source
    assert '"research_model_family": research_model_family' in jobs_source
