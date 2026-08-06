from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from market_cycle_trader_api.infrastructure.persistence.mongo_repository import (
    JOBS_COLLECTION,
    PAPER_MARKET_AUTOMATION_COLLECTION,
    PAPER_TRADING_STATE_COLLECTION,
    SETTINGS_COLLECTION,
    STRATEGY_CONTROL_COLLECTION,
    STRATEGY_PROFILES_COLLECTION,
)
from market_cycle_trader_api.services.strategy_configuration import (
    install_winner_strategy_configuration,
)
from market_cycle_trader_api.services.strategy_lab import (
    create_strategy,
    get_research_strategy_context,
    list_strategies,
    get_trader_winner_context,
    mark_strategy_backtest,
    promote_strategy_to_trader,
    select_research_strategy,
    update_strategy,
)


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


def test_promotion_creates_locked_snapshot_and_keeps_research_profile() -> None:
    db = _Database()
    install_winner_strategy_configuration(db, note="Install winner.", source="test")
    draft = create_strategy(
        db,
        name="Promotion candidate",
        description="Validated candidate.",
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
    db[PAPER_MARKET_AUTOMATION_COLLECTION].documents["default"] = {
        "_id": "default",
        "control_mode": "stopped",
    }
    db[PAPER_TRADING_STATE_COLLECTION].documents["default"] = {
        "_id": "default",
        "managed_symbol": None,
    }

    result = promote_strategy_to_trader(
        db,
        draft["id"],
        expected_control_revision=2,
        expected_strategy_revision=1,
        note="Promote after validation.",
        actor_email="admin@example.com",
    )

    assert result["status"] == "promoted"
    assert result["winner"]["locked"] is True
    assert result["winner"]["source_strategy_id"] == draft["id"]
    assert result["control"]["research_strategy_id"] == draft["id"]
    assert result["control"]["trader_winner_strategy_id"] == result["winner"]["id"]
    assert db[STRATEGY_PROFILES_COLLECTION].documents["winner-v1-13-2"]["status"] == "former_winner"


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
    assert '"/{strategy_id}/promote-to-trader"' in router
    assert '"strategy_manifest.json"' in exports


def test_strategy_catalog_exposes_every_validated_parameter_to_administrator() -> None:
    from market_cycle_trader_api.schemas.requests import BacktestRequest

    db = _Database()
    install_winner_strategy_configuration(db, note="Install winner.", source="test")

    catalog = list_strategies(db)
    expected = list(BacktestRequest.model_fields)
    assert catalog["parameter_order"] == expected
    grouped = [field for group in catalog["parameter_groups"] for field in group["fields"]]
    assert len(grouped) == len(expected)
    assert set(grouped) == set(expected)
    assert set(catalog["parameter_schema"]["properties"]) == set(expected)


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
