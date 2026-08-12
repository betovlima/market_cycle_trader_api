from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from market_cycle_trader_api.infrastructure.persistence.mongo_repository import (
    JOBS_COLLECTION,
    PAPER_MARKET_AUTOMATION_COLLECTION,
    PAPER_MARKET_RUNS_COLLECTION,
    PAPER_TRADE_PLANS_COLLECTION,
    PAPER_TRADING_STATE_COLLECTION,
    SETTINGS_COLLECTION,
    STRATEGY_CONTROL_COLLECTION,
    STRATEGY_PROFILES_COLLECTION,
    STRATEGY_PROMOTION_HISTORY_COLLECTION,
)
from market_cycle_trader_api.services import strategy_lab as strategy_lab_service
from market_cycle_trader_api.services.strategy_configuration import (
    install_winner_strategy_configuration,
)
from market_cycle_trader_api.services.strategy_lab import (
    create_strategy,
    ensure_strategy_catalog,
    get_research_reference_context,
    get_research_strategy_context,
    list_strategies,
    get_trader_winner_context,
    mark_strategy_as_candidate,
    mark_strategy_backtest,
    promote_strategy_to_trader,
    select_research_strategy,
    update_strategy,
    update_strategy_model,
)

# Promotion tests exercise lifecycle semantics, not the wall-clock. Production
# still enforces the XNYS regular-session boundary without calling Alpaca.
strategy_lab_service._regular_market_is_open = lambda: False


class _Result:
    def __init__(self, *, matched_count: int = 0, deleted_count: int = 0, inserted_id: str | None = None):
        self.matched_count = matched_count
        self.modified_count = matched_count
        self.deleted_count = deleted_count
        self.inserted_id = inserted_id
        self.upserted_id = inserted_id


class _Cursor(list):
    def sort(self, *args, **kwargs):
        del args, kwargs
        return self

    def limit(self, value: int):
        return _Cursor(self[:value])


class _Collection:
    def __init__(self) -> None:
        self.documents: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _matches(document: dict[str, Any], query: dict[str, Any]) -> bool:
        for key, expected in query.items():
            actual = document.get(key)
            if isinstance(expected, dict):
                if "$in" in expected and actual not in expected["$in"]:
                    return False
                if "$ne" in expected and actual == expected["$ne"]:
                    return False
                if "$exists" in expected and (key in document) != bool(expected["$exists"]):
                    return False
            elif actual != expected:
                return False
        return True

    def find_one(self, query: dict[str, Any], *args, **kwargs):
        del args, kwargs
        for document in self.documents.values():
            if self._matches(document, query):
                return copy.deepcopy(document)
        return None

    def find(self, query: dict[str, Any], *args, **kwargs):
        del args, kwargs
        return _Cursor(
            copy.deepcopy(document)
            for document in self.documents.values()
            if self._matches(document, query)
        )

    def replace_one(self, query: dict[str, Any], document: dict[str, Any], *, upsert: bool):
        document_id = str(query["_id"])
        existed = document_id in self.documents
        if existed or upsert:
            self.documents[document_id] = copy.deepcopy(document)
        return _Result(matched_count=1 if existed else 0, inserted_id=None if existed else document_id)

    def insert_one(self, document: dict[str, Any]):
        document_id = str(document.get("_id") or f"generated-{len(self.documents) + 1}")
        stored = copy.deepcopy(document)
        stored["_id"] = document_id
        self.documents[document_id] = stored
        return _Result(inserted_id=document_id)

    def find_one_and_update(self, query: dict[str, Any], update: dict[str, Any], **kwargs):
        del kwargs
        for document_id, document in self.documents.items():
            if not self._matches(document, query):
                continue
            next_document = copy.deepcopy(document)
            next_document.update(copy.deepcopy(update.get("$set", {})))
            for key, amount in update.get("$inc", {}).items():
                next_document[key] = int(next_document.get(key) or 0) + int(amount)
            self.documents[document_id] = next_document
            return copy.deepcopy(next_document)
        return None

    def update_one(self, query: dict[str, Any], update: dict[str, Any], **kwargs):
        del kwargs
        result = self.find_one_and_update(query, update)
        return _Result(matched_count=1 if result is not None else 0)

    def update_many(self, query: dict[str, Any], update: dict[str, Any], **kwargs):
        del kwargs
        matched = 0
        for document_id, document in list(self.documents.items()):
            if not self._matches(document, query):
                continue
            next_document = copy.deepcopy(document)
            next_document.update(copy.deepcopy(update.get("$set", {})))
            for key, amount in update.get("$inc", {}).items():
                next_document[key] = int(next_document.get(key) or 0) + int(amount)
            self.documents[document_id] = next_document
            matched += 1
        return _Result(matched_count=matched)

    def delete_one(self, query: dict[str, Any]):
        for document_id, document in list(self.documents.items()):
            if self._matches(document, query):
                self.documents.pop(document_id)
                return _Result(deleted_count=1)
        return _Result(deleted_count=0)

    def delete_many(self, query: dict[str, Any]):
        deleted = 0
        for document_id, document in list(self.documents.items()):
            if self._matches(document, query):
                self.documents.pop(document_id)
                deleted += 1
        return _Result(deleted_count=deleted)


class _Database:
    def __init__(self) -> None:
        self.collections: dict[str, _Collection] = {}

    def __getitem__(self, name: str) -> _Collection:
        return self.collections.setdefault(name, _Collection())


def test_research_strategy_changes_do_not_change_trader_winner() -> None:
    db = _Database()
    install_winner_strategy_configuration(
        db,
        note="Install protected winner.",
        source="test",
    )

    control = db[STRATEGY_CONTROL_COLLECTION].documents["default"]
    assert control["research_strategy_id"] == "winner-v1-13-2"
    assert control["trader_winner_strategy_id"] == "winner-v1-13-2"

    draft = create_strategy(
        db,
        name="Higher estimator test",
        description="Research only.",
        clone_from_strategy_id="winner-v1-13-2",
        actor_email="admin@example.com",
    )
    changed = dict(draft["configuration"])
    changed["rotation_xgb_n_estimators"] = 350
    updated = update_strategy(
        db,
        draft["id"],
        configuration=type(get_research_strategy_context(db)[0]).model_validate(changed),
        name=draft["name"],
        description=draft["description"],
        note="Test more estimators.",
        expected_revision=1,
        actor_email="admin@example.com",
    )
    selection = select_research_strategy(
        db,
        draft["id"],
        expected_control_revision=1,
        note="Use the draft for backtest.",
        actor_email="admin@example.com",
    )

    research, _ = get_research_strategy_context(db)
    trader, winner_profile = get_trader_winner_context(db)
    assert selection["research_strategy_id"] == draft["id"]
    assert selection["trader_winner_strategy_id"] == "winner-v1-13-2"
    assert research.rotation_xgb_n_estimators == 350
    assert trader.rotation_xgb_n_estimators == 300
    assert winner_profile["locked"] is True
    # The legacy production winner document remains untouched. Research profiles
    # live only in the additive strategy catalog.
    assert db[SETTINGS_COLLECTION].documents["default"]["rotation_xgb_n_estimators"] == 300
    assert updated["last_backtest_status"] is None


def test_research_reference_is_snapshotted_from_selected_strategy_and_does_not_follow_selection() -> None:
    db = _Database()
    install_winner_strategy_configuration(db, note="Install winner.", source="test")
    baseline = create_strategy(
        db,
        name="Research baseline",
        description="Reference universe.",
        clone_from_strategy_id="winner-v1-13-2",
        actor_email="admin@example.com",
    )
    baseline_configuration = dict(baseline["configuration"])
    baseline_configuration["assets"] = list(baseline_configuration["assets"]) + ["TESTREF"]
    baseline = update_strategy(
        db,
        baseline["id"],
        configuration=type(get_research_strategy_context(db)[0]).model_validate(baseline_configuration),
        name=baseline["name"],
        description=baseline["description"],
        note="Expand reference universe.",
        expected_revision=1,
        actor_email="admin@example.com",
    )
    select_research_strategy(
        db,
        baseline["id"],
        expected_control_revision=1,
        note="Select reference before migration.",
        actor_email="admin@example.com",
    )

    # Simulate a catalog created before v1.13.38, which did not persist a
    # separate research-reference snapshot in strategy_control/default.
    control = db[STRATEGY_CONTROL_COLLECTION].documents["default"]
    control.pop("research_reference_strategy_id", None)
    control.pop("research_reference_configuration_hash", None)
    control.pop("research_reference_assets", None)

    normalized = ensure_strategy_catalog(db)
    reference_assets, reference = get_research_reference_context(db)
    assert normalized["research_reference_strategy_id"] == baseline["id"]
    assert reference["id"] == baseline["id"]
    assert reference_assets == baseline["configuration"]["assets"]

    challenger = create_strategy(
        db,
        name="Challenger",
        description="Must not move the research reference.",
        clone_from_strategy_id=baseline["id"],
        actor_email="admin@example.com",
    )
    select_research_strategy(
        db,
        challenger["id"],
        expected_control_revision=int(normalized["revision"]),
        note="Select challenger.",
        actor_email="admin@example.com",
    )
    reference_assets_after, reference_after = get_research_reference_context(db)
    assert reference_after["id"] == baseline["id"]
    assert reference_assets_after == reference_assets


def test_promotion_creates_locked_snapshot_and_keeps_research_profile() -> None:
    db = _Database()
    install_winner_strategy_configuration(db, note="Install winner.", source="test")
    draft = create_strategy(
        db,
        name=f"Winner v{strategy_lab_service.API_VERSION}",
        description="Validated candidate with the future Winner display name.",
        clone_from_strategy_id="winner-v1-13-2",
        actor_email="admin@example.com",
    )
    select_research_strategy(
        db,
        draft["id"],
        expected_control_revision=1,
        note="Backtest candidate.",
        actor_email="admin@example.com",
    )
    db[JOBS_COLLECTION].documents["job-1"] = {
        "_id": "job-1",
        "id": "job-1",
        "status": "completed",
        "strategy_profile_id": draft["id"],
        "strategy_profile_revision": 1,
    }
    mark_strategy_backtest(
        db,
        strategy_id=draft["id"],
        strategy_revision=1,
        job_id="job-1",
        status="completed",
    )
    candidate = mark_strategy_as_candidate(
        db,
        draft["id"],
        expected_strategy_revision=1,
        note="Candidate after exact completed backtest.",
        actor_email="admin@example.com",
    )
    assert candidate["status"] == "candidate"
    assert candidate["candidate_backtest_id"] == "job-1"

    db[PAPER_MARKET_AUTOMATION_COLLECTION].documents["default"] = {
        "_id": "default",
        "enabled": True,
        "control_mode": "active",
        "phase": "waiting_for_premarket_analysis",
    }
    db[PAPER_TRADING_STATE_COLLECTION].documents["default"] = {
        "_id": "default",
        "initial_capital": 10000.0,
        "strategy_cash": 124.5,
        "managed_symbol": "NVDA",
        "managed_quantity": 12.345,
        "average_entry_price": 101.25,
        "holding_sessions": 4,
        "realized_pnl": 875.0,
        "last_decision_date": "2026-08-05",
        "last_execution_session": "2026-08-06",
    }
    db[PAPER_MARKET_RUNS_COLLECTION].documents["paper-next"] = {
        "_id": "paper-next",
        "run_id": "paper-next",
        "active_key": "alpaca-paper-next-session",
        "status": "armed",
        "phase": "waiting_for_premarket_analysis",
        "execution_session": "2026-08-07",
        "premarket_analysis_at": "2026-08-07T12:00:00+00:00",
        "plan_id": None,
    }
    state_before = copy.deepcopy(db[PAPER_TRADING_STATE_COLLECTION].documents["default"])
    automation_before = copy.deepcopy(db[PAPER_MARKET_AUTOMATION_COLLECTION].documents["default"])
    run_before = copy.deepcopy(db[PAPER_MARKET_RUNS_COLLECTION].documents["paper-next"])

    result = promote_strategy_to_trader(
        db,
        draft["id"],
        expected_control_revision=3,
        expected_strategy_revision=1,
        note="Promote after validation.",
        actor_email="admin@example.com",
    )

    assert result["status"] == "promoted"
    assert result["winner"]["name"] == "Winner #2"
    assert result["winner"]["winner_sequence"] == 2
    assert result["winner"]["locked"] is True
    assert result["winner"]["source_strategy_id"] == draft["id"]
    assert result["promotion"]["broker_interaction_performed"] is False
    assert result["promotion"]["operational_state_preserved"] is True
    assert result["promotion"]["next_scheduled_evaluation_uses_new_winner"] is True
    assert result["promotion"]["managed_symbol"] == "NVDA"
    assert result["control"]["research_strategy_id"] == draft["id"]
    assert result["control"]["trader_winner_strategy_id"] == result["winner"]["id"]
    assert result["control"]["research_reference_strategy_id"] == result["winner"]["id"]
    assert result["control"]["research_reference_assets"] == result["winner"]["research_reference_assets"]
    assert db[STRATEGY_PROFILES_COLLECTION].documents["winner-v1-13-2"]["status"] == "former_winner"
    assert db[STRATEGY_PROFILES_COLLECTION].documents[draft["id"]]["status"] == "promoted_candidate"
    assert db[STRATEGY_PROFILES_COLLECTION].documents[draft["id"]]["locked"] is True
    assert result["control"]["candidate_strategy_id"] is None
    assert result["control"]["promoted_candidate_strategy_id"] == draft["id"]
    assert result["control"]["promoted_candidate_strategy"]["id"] == draft["id"]
    assert result["control"]["paper_state_reinitialization_required"] is False
    assert db[PAPER_TRADING_STATE_COLLECTION].documents["default"] == state_before
    assert db[PAPER_MARKET_AUTOMATION_COLLECTION].documents["default"] == automation_before
    assert db[PAPER_MARKET_RUNS_COLLECTION].documents["paper-next"] == run_before
    history = list(db[STRATEGY_PROMOTION_HISTORY_COLLECTION].documents.values())
    promotion = [item for item in history if item.get("action") == "winner_promoted_preserving_operational_state"]
    assert len(promotion) == 1
    assert promotion[0]["operational_snapshot"]["managed_symbol"] == "NVDA"


def test_promotion_allows_multiple_historical_winners_from_same_api_release() -> None:
    db = _Database()
    install_winner_strategy_configuration(db, note="Install winner.", source="test")
    draft = create_strategy(
        db,
        name=f"Winner v{strategy_lab_service.API_VERSION}",
        description="Winner identity must be independent from the API release.",
        clone_from_strategy_id="winner-v1-13-2",
        actor_email="admin@example.com",
    )
    db[JOBS_COLLECTION].documents["job-release-independent"] = {
        "_id": "job-release-independent",
        "id": "job-release-independent",
        "status": "completed",
        "strategy_profile_id": draft["id"],
        "strategy_profile_revision": 1,
    }
    mark_strategy_backtest(
        db,
        strategy_id=draft["id"],
        strategy_revision=1,
        job_id="job-release-independent",
        status="completed",
    )
    mark_strategy_as_candidate(
        db,
        draft["id"],
        expected_strategy_revision=1,
        note="Validated candidate.",
        actor_email="admin@example.com",
    )

    db[STRATEGY_PROFILES_COLLECTION].documents["former-release-winner"] = {
        "_id": "former-release-winner",
        "name": f"Legacy winner from API {strategy_lab_service.API_VERSION}",
        "status": "former_winner",
        "locked": True,
        "revision": 1,
        "winner_api_version": strategy_lab_service.API_VERSION,
    }

    control_revision = int(db[STRATEGY_CONTROL_COLLECTION].documents["default"]["revision"])
    result = promote_strategy_to_trader(
        db,
        draft["id"],
        expected_control_revision=control_revision,
        expected_strategy_revision=1,
        note="Promote even though this API release already has historical Winner snapshots.",
        actor_email="admin@example.com",
    )

    assert result["status"] == "promoted"
    assert result["winner"]["winner_api_version"] == strategy_lab_service.API_VERSION
    assert result["winner"]["name"].startswith("Winner #")
    active_winners = [
        item for item in db[STRATEGY_PROFILES_COLLECTION].documents.values()
        if item.get("status") == "winner"
    ]
    assert len(active_winners) == 1
    assert active_winners[0]["_id"] == result["winner"]["id"]
    assert db[STRATEGY_PROFILES_COLLECTION].documents["former-release-winner"]["status"] == "former_winner"


def test_promotion_blocks_after_premarket_plan_exists_without_changing_state() -> None:
    db = _Database()
    install_winner_strategy_configuration(db, note="Install winner.", source="test")
    draft = create_strategy(
        db,
        name="Prepared plan candidate",
        description="Promotion must happen before model preparation.",
        clone_from_strategy_id="winner-v1-13-2",
        actor_email="admin@example.com",
    )
    db[JOBS_COLLECTION].documents["job-prepared"] = {
        "_id": "job-prepared",
        "id": "job-prepared",
        "status": "completed",
        "strategy_profile_id": draft["id"],
        "strategy_profile_revision": 1,
    }
    mark_strategy_backtest(
        db,
        strategy_id=draft["id"],
        strategy_revision=1,
        job_id="job-prepared",
        status="completed",
    )
    mark_strategy_as_candidate(
        db,
        draft["id"],
        expected_strategy_revision=1,
        note="Validated candidate.",
        actor_email="admin@example.com",
    )
    db[PAPER_TRADING_STATE_COLLECTION].documents["default"] = {
        "_id": "default",
        "managed_symbol": "NVDA",
        "managed_quantity": 1.0,
    }
    db[PAPER_MARKET_RUNS_COLLECTION].documents["paper-prepared"] = {
        "_id": "paper-prepared",
        "run_id": "paper-prepared",
        "active_key": "alpaca-paper-next-session",
        "status": "prepared",
        "phase": "waiting_for_next_market_open",
        "plan_id": "plan-old-winner",
    }
    state_before = copy.deepcopy(db[PAPER_TRADING_STATE_COLLECTION].documents["default"])
    try:
        promote_strategy_to_trader(
            db,
            draft["id"],
            expected_control_revision=2,
            expected_strategy_revision=1,
            note="Must be blocked after predictions exist.",
            actor_email="admin@example.com",
        )
    except Exception as exc:
        assert "before calibration" in str(exc).lower() or "current run status" in str(exc).lower()
    else:
        raise AssertionError("Promotion must be blocked after pre-market preparation starts.")
    assert db[PAPER_TRADING_STATE_COLLECTION].documents["default"] == state_before
    assert db[STRATEGY_CONTROL_COLLECTION].documents["default"]["trader_winner_strategy_id"] == "winner-v1-13-2"
    assert db[STRATEGY_CONTROL_COLLECTION].documents["default"].get("winner_promotion_in_progress") is False


def test_promotion_blocks_when_current_position_is_outside_candidate_universe() -> None:
    db = _Database()
    install_winner_strategy_configuration(db, note="Install winner.", source="test")
    draft = create_strategy(
        db,
        name="Universe compatibility candidate",
        description="Current position compatibility.",
        clone_from_strategy_id="winner-v1-13-2",
        actor_email="admin@example.com",
    )
    db[JOBS_COLLECTION].documents["job-universe"] = {
        "_id": "job-universe",
        "id": "job-universe",
        "status": "completed",
        "strategy_profile_id": draft["id"],
        "strategy_profile_revision": 1,
    }
    mark_strategy_backtest(
        db,
        strategy_id=draft["id"],
        strategy_revision=1,
        job_id="job-universe",
        status="completed",
    )
    mark_strategy_as_candidate(
        db,
        draft["id"],
        expected_strategy_revision=1,
        note="Validated candidate.",
        actor_email="admin@example.com",
    )
    db[PAPER_TRADING_STATE_COLLECTION].documents["default"] = {
        "_id": "default",
        "managed_symbol": "ZZZ",
        "managed_quantity": 2.0,
    }
    try:
        promote_strategy_to_trader(
            db,
            draft["id"],
            expected_control_revision=2,
            expected_strategy_revision=1,
            note="Must preserve incompatible position.",
            actor_email="admin@example.com",
        )
    except Exception as exc:
        assert "not part of the candidate asset universe" in str(exc).lower()
    else:
        raise AssertionError("Promotion must block an incompatible managed symbol without liquidating it.")
    assert db[PAPER_TRADING_STATE_COLLECTION].documents["default"]["managed_symbol"] == "ZZZ"


def test_catalog_migration_preserves_production_winner_identity_and_document() -> None:
    from market_cycle_trader_api.services.strategy_lab import ensure_strategy_catalog
    from market_cycle_trader_api.schemas.requests import BacktestRequest

    db = _Database()
    packaged = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "market_cycle_trader_api"
        / "parameterizations"
        / "winner-v1.13.2.json"
    )
    configuration = BacktestRequest.model_validate_json(packaged.read_text(encoding="utf-8"))
    legacy = {
        "_id": "default",
        **configuration.model_dump(mode="python"),
        "revision": 1,
        "schema_version": 16,
        "configuration_name": "winner-v1.13.1",
        "winner_source_file": "winner-v1.13.1.json",
        "winner_configuration_hash": "22a4193fbb30de33d75864fc28c3b1923e4dedd4970b14f9537f793bccf18953",
        "bootstrap_source": "winner-v1.13.1-install-api",
    }
    db[SETTINGS_COLLECTION].documents["default"] = copy.deepcopy(legacy)

    control = ensure_strategy_catalog(db)

    assert control["research_strategy_id"] == "winner-v1-13-1"
    assert control["trader_winner_strategy_id"] == "winner-v1-13-1"
    profile = db[STRATEGY_PROFILES_COLLECTION].documents["winner-v1-13-1"]
    assert profile["name"] == "Winner v1.13.1"
    assert profile["locked"] is True
    assert profile["origin_winner_source_file"] == "winner-v1.13.1.json"
    assert db[SETTINGS_COLLECTION].documents["default"] == legacy


def test_draft_can_be_edited_during_active_backtest_without_certifying_new_revision() -> None:
    db = _Database()
    install_winner_strategy_configuration(db, note="Install winner.", source="test")
    draft = create_strategy(
        db,
        name="Editable during run",
        description="Snapshot safety test.",
        clone_from_strategy_id="winner-v1-13-2",
        actor_email="admin@example.com",
    )
    db[JOBS_COLLECTION].documents["job-active"] = {
        "_id": "job-active",
        "id": "job-active",
        "status": "running",
        "strategy_profile_id": draft["id"],
        "strategy_profile_revision": 1,
    }
    changed = dict(draft["configuration"])
    changed["rotation_xgb_n_estimators"] = 325
    updated = update_strategy(
        db,
        draft["id"],
        configuration=type(get_research_strategy_context(db)[0]).model_validate(changed),
        name=draft["name"],
        description=draft["description"],
        note="Prepare the next revision while the current snapshot runs.",
        expected_revision=1,
        actor_email="admin@example.com",
    )
    assert updated["revision"] == 2

    mark_strategy_backtest(
        db,
        strategy_id=draft["id"],
        strategy_revision=1,
        job_id="job-active",
        status="completed",
    )
    current = db[STRATEGY_PROFILES_COLLECTION].documents[draft["id"]]
    assert current["revision"] == 2
    assert current.get("last_backtest_status") is None


def test_candidate_requires_exact_completed_revision_and_editing_returns_to_draft() -> None:
    db = _Database()
    install_winner_strategy_configuration(db, note="Install winner.", source="test")
    draft = create_strategy(
        db,
        name="Candidate lifecycle",
        description="Lifecycle test.",
        clone_from_strategy_id="winner-v1-13-2",
        actor_email="admin@example.com",
    )

    try:
        mark_strategy_as_candidate(
            db,
            draft["id"],
            expected_strategy_revision=1,
            note="Premature candidate.",
            actor_email="admin@example.com",
        )
    except Exception as exc:
        assert "complete a backtest" in str(exc).lower()
    else:
        raise AssertionError("Candidate status must require an exact completed backtest.")

    db[JOBS_COLLECTION].documents["job-candidate"] = {
        "_id": "job-candidate",
        "id": "job-candidate",
        "status": "completed",
        "strategy_profile_id": draft["id"],
        "strategy_profile_revision": 1,
    }
    mark_strategy_backtest(
        db,
        strategy_id=draft["id"],
        strategy_revision=1,
        job_id="job-candidate",
        status="completed",
    )
    candidate = mark_strategy_as_candidate(
        db,
        draft["id"],
        expected_strategy_revision=1,
        note="Validated candidate.",
        actor_email="admin@example.com",
    )
    assert candidate["status"] == "candidate"
    assert candidate["candidate_revision"] == 1
    assert candidate["candidate_backtest_id"] == "job-candidate"
    assert db[STRATEGY_CONTROL_COLLECTION].documents["default"]["candidate_strategy_id"] == draft["id"]

    changed = dict(candidate["configuration"])
    changed["rotation_switch_margin"] = 0.0075
    updated = update_strategy(
        db,
        draft["id"],
        configuration=type(get_research_strategy_context(db)[0]).model_validate(changed),
        name=candidate["name"],
        description=candidate["description"],
        note="Continue candidate research.",
        expected_revision=1,
        actor_email="admin@example.com",
    )
    assert updated["revision"] == 2
    assert updated["status"] == "draft"
    assert updated["candidate_revision"] is None
    assert updated["candidate_backtest_id"] is None
    assert db[STRATEGY_CONTROL_COLLECTION].documents["default"]["candidate_strategy_id"] is None


def test_v11321_catalog_migration_keeps_only_latest_active_candidate() -> None:
    db = _Database()
    install_winner_strategy_configuration(db, note="Install winner.", source="test")
    first = create_strategy(
        db,
        name="Legacy candidate one",
        description="v1.13.21 migration.",
        clone_from_strategy_id="winner-v1-13-2",
        actor_email="admin@example.com",
    )
    second = create_strategy(
        db,
        name="Legacy candidate two",
        description="v1.13.21 migration.",
        clone_from_strategy_id="winner-v1-13-2",
        actor_email="admin@example.com",
    )
    db[STRATEGY_PROFILES_COLLECTION].documents[first["id"]].update({
        "status": "candidate",
        "candidate_at": "2026-08-06T10:00:00+00:00",
    })
    db[STRATEGY_PROFILES_COLLECTION].documents[second["id"]].update({
        "status": "candidate",
        "candidate_at": "2026-08-06T11:00:00+00:00",
    })
    db[STRATEGY_CONTROL_COLLECTION].documents["default"].pop("candidate_strategy_id", None)

    catalog = list_strategies(db)

    assert catalog["control"]["candidate_strategy_id"] == second["id"]
    assert catalog["control"]["candidate_strategy"]["id"] == second["id"]
    previous = db[STRATEGY_PROFILES_COLLECTION].documents[first["id"]]
    assert previous["status"] == "superseded_candidate"
    assert previous["locked"] is True
    assert db[STRATEGY_PROFILES_COLLECTION].documents[second["id"]]["status"] == "candidate"


def test_marking_new_candidate_supersedes_and_locks_previous_candidate() -> None:
    db = _Database()
    install_winner_strategy_configuration(db, note="Install winner.", source="test")

    candidates = []
    for index in (1, 2):
        draft = create_strategy(
            db,
            name=f"Candidate {index}",
            description="Unique candidate lifecycle.",
            clone_from_strategy_id="winner-v1-13-2",
            actor_email="admin@example.com",
        )
        job_id = f"job-candidate-{index}"
        db[JOBS_COLLECTION].documents[job_id] = {
            "_id": job_id,
            "id": job_id,
            "status": "completed",
            "strategy_profile_id": draft["id"],
            "strategy_profile_revision": 1,
        }
        mark_strategy_backtest(
            db,
            strategy_id=draft["id"],
            strategy_revision=1,
            job_id=job_id,
            status="completed",
        )
        candidates.append(draft)

    first = mark_strategy_as_candidate(
        db,
        candidates[0]["id"],
        expected_strategy_revision=1,
        note="First validated candidate.",
        actor_email="admin@example.com",
    )
    assert first["status"] == "candidate"

    second = mark_strategy_as_candidate(
        db,
        candidates[1]["id"],
        expected_strategy_revision=1,
        note="Second candidate replaces the first.",
        actor_email="admin@example.com",
    )
    assert second["status"] == "candidate"
    control = db[STRATEGY_CONTROL_COLLECTION].documents["default"]
    assert control["candidate_strategy_id"] == candidates[1]["id"]
    previous = db[STRATEGY_PROFILES_COLLECTION].documents[candidates[0]["id"]]
    assert previous["status"] == "superseded_candidate"
    assert previous["locked"] is True
    active_candidates = [
        item for item in db[STRATEGY_PROFILES_COLLECTION].documents.values()
        if item.get("status") == "candidate"
    ]
    assert len(active_candidates) == 1
    assert active_candidates[0]["_id"] == candidates[1]["id"]


def test_promotion_rejects_completed_draft_until_marked_candidate() -> None:
    db = _Database()
    install_winner_strategy_configuration(db, note="Install winner.", source="test")
    draft = create_strategy(
        db,
        name="Unmarked draft",
        description="Promotion gate test.",
        clone_from_strategy_id="winner-v1-13-2",
        actor_email="admin@example.com",
    )
    db[JOBS_COLLECTION].documents["job-gate"] = {
        "_id": "job-gate",
        "id": "job-gate",
        "status": "completed",
        "strategy_profile_id": draft["id"],
        "strategy_profile_revision": 1,
    }
    mark_strategy_backtest(
        db,
        strategy_id=draft["id"],
        strategy_revision=1,
        job_id="job-gate",
        status="completed",
    )
    db[PAPER_MARKET_AUTOMATION_COLLECTION].documents["default"] = {
        "_id": "default",
        "control_mode": "stopped",
    }
    db[PAPER_TRADING_STATE_COLLECTION].documents["default"] = {
        "_id": "default",
        "managed_symbol": None,
    }
    try:
        promote_strategy_to_trader(
            db,
            draft["id"],
            expected_control_revision=1,
            expected_strategy_revision=1,
            note="Should be blocked.",
            actor_email="admin@example.com",
        )
    except Exception as exc:
        assert "candidate" in str(exc).lower()
    else:
        raise AssertionError("Promotion must require candidate status.")



def test_jobs_and_paper_use_separate_strategy_contexts() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "market_cycle_trader_api"
    jobs = (root / "api" / "routers" / "jobs.py").read_text(encoding="utf-8")
    paper = (root / "services" / "paper_trading.py").read_text(encoding="utf-8")
    router = (root / "api" / "routers" / "strategy_lab.py").read_text(encoding="utf-8")
    exports = (root / "api" / "routers" / "exports.py").read_text(encoding="utf-8")

    assert "get_research_strategy_context" in jobs
    assert "get_trader_winner_context" in paper
    assert '"/{strategy_id}/select-for-backtest"' in router
    assert '"/{strategy_id}/mark-as-candidate"' in router
    assert '"/{strategy_id}/promote-to-trader"' in router
    assert '"strategy_manifest.json"' in exports


def test_strategy_catalog_exposes_strategy_fields_and_moves_model_fields_out() -> None:
    from market_cycle_trader_api.schemas.requests import BacktestRequest
    from market_cycle_trader_api.services.strategy_lab import MODEL_OWNED_STRATEGY_FIELDS

    db = _Database()
    install_winner_strategy_configuration(db, note="Install winner.", source="test")

    catalog = list_strategies(db)
    expected = [field for field in BacktestRequest.model_fields if field not in MODEL_OWNED_STRATEGY_FIELDS]
    assert catalog["parameter_order"] == expected
    grouped = [field for group in catalog["parameter_groups"] for field in group["fields"]]
    assert len(grouped) == len(expected)
    assert set(grouped) == set(expected)
    assert not set(grouped).intersection(MODEL_OWNED_STRATEGY_FIELDS)
    assert set(catalog["parameter_schema"]["properties"]) == set(BacktestRequest.model_fields)
    descriptions = [
        catalog["parameter_schema"]["properties"][field].get("description", "").strip()
        for field in expected
    ]
    assert all(descriptions)


def test_strategy_model_binding_is_saved_without_changing_shared_strategy_revision() -> None:
    from market_cycle_trader_api.services.model_research import execution_settings_for

    db = _Database()
    install_winner_strategy_configuration(db, note="Install winner.", source="test")
    draft = create_strategy(
        db,
        name="Saved model UX",
        description="Backtest must use the model saved with this Strategy.",
        clone_from_strategy_id="winner-v1-13-2",
        actor_email="admin@example.com",
    )
    lightgbm = execution_settings_for(db, "lightgbm_utility")
    updated = update_strategy_model(
        db,
        draft["id"],
        model_family="lightgbm_utility",
        values=lightgbm["lightgbm"],
        note="Use LightGBM for this cloned Strategy.",
        expected_strategy_revision=1,
        actor_email="admin@example.com",
    )

    assert updated["revision"] == 1
    assert updated["research_model_revision"] == 2
    assert updated["research_model"]["family"] == "lightgbm_utility"
    stored = db[STRATEGY_PROFILES_COLLECTION].documents[draft["id"]]
    assert stored["revision"] == 1
    assert stored["research_model_snapshot"]["family"] == "lightgbm_utility"


def test_strategy_model_binding_adopts_matching_completed_backtest_snapshot() -> None:
    from market_cycle_trader_api.services.model_research import execution_settings_for, model_execution_snapshot

    db = _Database()
    install_winner_strategy_configuration(db, note="Install winner.", source="test")
    draft = create_strategy(
        db,
        name="Historical LightGBM",
        description="Reuse an exact completed research run.",
        clone_from_strategy_id="winner-v1-13-2",
        actor_email="admin@example.com",
    )
    settings = execution_settings_for(db, "lightgbm_utility")
    exact_snapshot = model_execution_snapshot("lightgbm_utility", settings)
    db[JOBS_COLLECTION].documents["existing-lightgbm"] = {
        "_id": "existing-lightgbm",
        "id": "existing-lightgbm",
        "status": "completed",
        "created_at": "2026-08-11T12:05:57+00:00",
        "finished_at": "2026-08-11T12:16:00+00:00",
        "strategy_profile_id": draft["id"],
        "strategy_profile_revision": 1,
        "research_model_family": "lightgbm_utility",
        "research_model_settings_hash": exact_snapshot["settings_hash"],
        "request": {
            "research_model_family": "lightgbm_utility",
            "research_model_settings": settings,
        },
    }

    updated = update_strategy_model(
        db,
        draft["id"],
        model_family="lightgbm_utility",
        values=settings["lightgbm"],
        note="Adopt exact historical LightGBM configuration.",
        expected_strategy_revision=1,
        actor_email="admin@example.com",
    )

    assert updated["revision"] == 1
    assert updated["last_backtest_id"] == "existing-lightgbm"
    assert updated["last_backtest_status"] == "completed"
    assert updated["research_model"]["settings_hash"] == exact_snapshot["settings_hash"]


def test_legacy_direct_mutation_routes_are_disabled() -> None:
    from pathlib import Path

    router = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "market_cycle_trader_api"
        / "api"
        / "routers"
        / "strategy_configuration.py"
    ).read_text(encoding="utf-8")

    assert "Direct strategy mutation is disabled" in router
    assert "through /api/admin/strategies" in router


def test_lightgbm_candidate_and_winner_freeze_exact_model_snapshot() -> None:
    from market_cycle_trader_api.services.model_research import execution_settings_for

    db = _Database()
    install_winner_strategy_configuration(db, note="Install winner.", source="test")
    draft = create_strategy(
        db,
        name="LightGBM champion",
        description="Model-aware Winner test.",
        clone_from_strategy_id="winner-v1-13-2",
        actor_email="admin@example.com",
    )
    model_settings = execution_settings_for(db, "lightgbm_utility")
    job_id = "job-lightgbm-champion"
    db[JOBS_COLLECTION].documents[job_id] = {
        "_id": job_id,
        "id": job_id,
        "status": "completed",
        "created_at": "2026-08-11T12:05:57+00:00",
        "strategy_profile_id": draft["id"],
        "strategy_profile_revision": 1,
        "strategy_configuration_hash": draft["configuration_hash"],
        "research_model_family": "lightgbm_utility",
        "request": {
            "research_model_family": "lightgbm_utility",
            "research_model_settings": model_settings,
        },
    }

    bound = update_strategy_model(
        db,
        draft["id"],
        model_family="lightgbm_utility",
        values=model_settings["lightgbm"],
        note="Bind the validated LightGBM champion to this Strategy.",
        expected_strategy_revision=1,
        actor_email="admin@example.com",
    )
    assert bound["revision"] == 1
    assert bound["research_model"]["family"] == "lightgbm_utility"
    assert bound["last_backtest_id"] == job_id

    candidate = mark_strategy_as_candidate(
        db,
        draft["id"],
        expected_strategy_revision=1,
        model_family="lightgbm_utility",
        note="Promote the validated LightGBM research champion.",
        actor_email="admin@example.com",
    )
    assert candidate["candidate_model"]["family"] == "lightgbm_utility"
    assert candidate["candidate_model"]["profile_id"] == "baseline"
    assert len(candidate["candidate_model"]["settings_hash"]) == 64

    db[PAPER_MARKET_AUTOMATION_COLLECTION].documents["default"] = {
        "_id": "default",
        "control_mode": "stopped",
    }
    db[PAPER_TRADING_STATE_COLLECTION].documents["default"] = {
        "_id": "default",
        "managed_symbol": None,
        "managed_quantity": 0.0,
        "strategy_cash": 10000.0,
        "holding_sessions": 0,
    }
    result = promote_strategy_to_trader(
        db,
        draft["id"],
        expected_control_revision=2,
        expected_strategy_revision=1,
        note="Officialize LightGBM Winner.",
        actor_email="admin@example.com",
    )
    assert result["winner"]["winner_model"]["family"] == "lightgbm_utility"
    assert result["winner"]["winner_model"]["settings_hash"] == candidate["candidate_model"]["settings_hash"]
    stored = db[STRATEGY_PROFILES_COLLECTION].documents[result["winner"]["id"]]
    assert stored["winner_model_snapshot"]["family"] == "lightgbm_utility"
    assert stored["winner_model_snapshot"]["settings_snapshot"]["lightgbm"]["num_leaves"] == 8


def test_only_one_promoted_candidate_remains_active_across_winner_promotions() -> None:
    db = _Database()
    install_winner_strategy_configuration(db, note="Install winner.", source="test")

    def validated_candidate(name: str, job_id: str) -> dict[str, Any]:
        draft = create_strategy(
            db,
            name=name,
            description="Lifecycle promotion test.",
            clone_from_strategy_id=str(db[STRATEGY_CONTROL_COLLECTION].documents["default"]["trader_winner_strategy_id"]),
            actor_email="admin@example.com",
        )
        db[JOBS_COLLECTION].documents[job_id] = {
            "_id": job_id,
            "id": job_id,
            "status": "completed",
            "strategy_profile_id": draft["id"],
            "strategy_profile_revision": 1,
        }
        mark_strategy_backtest(
            db,
            strategy_id=draft["id"],
            strategy_revision=1,
            job_id=job_id,
            status="completed",
        )
        return mark_strategy_as_candidate(
            db,
            draft["id"],
            expected_strategy_revision=1,
            note=f"Validate {name}.",
            actor_email="admin@example.com",
        )

    first = validated_candidate("First promoted candidate", "job-first-promoted")
    first_control_revision = int(db[STRATEGY_CONTROL_COLLECTION].documents["default"]["revision"])
    first_promotion = promote_strategy_to_trader(
        db,
        first["id"],
        expected_control_revision=first_control_revision,
        expected_strategy_revision=1,
        note="First promotion.",
        actor_email="admin@example.com",
    )
    assert first_promotion["control"]["promoted_candidate_strategy_id"] == first["id"]
    assert db[STRATEGY_PROFILES_COLLECTION].documents[first["id"]]["status"] == "promoted_candidate"

    second = validated_candidate("Second promoted candidate", "job-second-promoted")
    second_control_revision = int(db[STRATEGY_CONTROL_COLLECTION].documents["default"]["revision"])
    second_promotion = promote_strategy_to_trader(
        db,
        second["id"],
        expected_control_revision=second_control_revision,
        expected_strategy_revision=1,
        note="Second promotion replaces promoted-candidate role.",
        actor_email="admin@example.com",
    )

    first_profile = db[STRATEGY_PROFILES_COLLECTION].documents[first["id"]]
    second_profile = db[STRATEGY_PROFILES_COLLECTION].documents[second["id"]]
    assert first_profile["status"] == "superseded_candidate"
    assert first_profile["historical_lifecycle_status"] == "promoted_candidate"
    assert first_profile["superseded_by_strategy_id"] == second["id"]
    assert second_profile["status"] == "promoted_candidate"
    assert second_promotion["control"]["promoted_candidate_strategy_id"] == second["id"]
    active_promoted = [
        item for item in db[STRATEGY_PROFILES_COLLECTION].documents.values()
        if item.get("status") == "promoted_candidate"
    ]
    assert len(active_promoted) == 1
    assert active_promoted[0]["_id"] == second["id"]
    active_winners = [
        item for item in db[STRATEGY_PROFILES_COLLECTION].documents.values()
        if item.get("status") == "winner"
    ]
    assert len(active_winners) == 1
    former_winners = [
        item for item in db[STRATEGY_PROFILES_COLLECTION].documents.values()
        if item.get("status") == "former_winner"
    ]
    assert len(former_winners) >= 2


def test_catalog_normalization_keeps_only_active_winner_source_as_promoted_candidate() -> None:
    db = _Database()
    install_winner_strategy_configuration(db, note="Install winner.", source="test")
    control = db[STRATEGY_CONTROL_COLLECTION].documents["default"]
    winner_id = str(control["trader_winner_strategy_id"])

    db[STRATEGY_PROFILES_COLLECTION].documents["old-promoted"] = {
        "_id": "old-promoted",
        "name": "Old promoted candidate",
        "status": "promoted_candidate",
        "locked": True,
        "revision": 1,
        "last_promoted_winner_strategy_id": "old-winner",
        "last_promoted_at": "2026-08-01T10:00:00+00:00",
        "configuration": copy.deepcopy(db[STRATEGY_PROFILES_COLLECTION].documents[winner_id]["configuration"]),
        "configuration_hash": db[STRATEGY_PROFILES_COLLECTION].documents[winner_id]["configuration_hash"],
    }
    db[STRATEGY_PROFILES_COLLECTION].documents["active-source"] = {
        "_id": "active-source",
        "name": "Active winner source",
        "status": "promoted_candidate",
        "locked": True,
        "revision": 1,
        "last_promoted_winner_strategy_id": winner_id,
        "last_promoted_at": "2026-08-02T10:00:00+00:00",
        "configuration": copy.deepcopy(db[STRATEGY_PROFILES_COLLECTION].documents[winner_id]["configuration"]),
        "configuration_hash": db[STRATEGY_PROFILES_COLLECTION].documents[winner_id]["configuration_hash"],
    }
    db[STRATEGY_PROFILES_COLLECTION].documents[winner_id]["source_strategy_id"] = "active-source"
    control.pop("promoted_candidate_strategy_id", None)

    catalog = list_strategies(db)

    assert catalog["control"]["promoted_candidate_strategy_id"] == "active-source"
    assert db[STRATEGY_PROFILES_COLLECTION].documents["active-source"]["status"] == "promoted_candidate"
    assert db[STRATEGY_PROFILES_COLLECTION].documents["old-promoted"]["status"] == "superseded_candidate"
    assert db[STRATEGY_PROFILES_COLLECTION].documents["old-promoted"]["historical_lifecycle_status"] == "promoted_candidate"
    active_promoted = [
        item for item in db[STRATEGY_PROFILES_COLLECTION].documents.values()
        if item.get("status") == "promoted_candidate"
    ]
    assert len(active_promoted) == 1
