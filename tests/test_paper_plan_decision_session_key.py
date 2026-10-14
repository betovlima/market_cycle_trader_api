"""Prevent collisions between intraday manual and next-open Paper plans."""
from __future__ import annotations

from unittest.mock import MagicMock

from pymongo import ASCENDING, DESCENDING

from market_cycle_trader_api.infrastructure.persistence import mongo_repository as mongo


class MemoryPlans:
    def __init__(self):
        self.documents = []
        self.replacements = []

    def insert_one(self, document):
        self.documents.append(dict(document))

    def replace_one(self, selector, document, *, upsert=False):
        self.replacements.append(dict(selector))
        for i, existing in enumerate(self.documents):
            if all(existing.get(k) == v for k, v in selector.items()):
                self.documents[i] = dict(document)
                return
        if upsert:
            self.documents.append(dict(document))


def test_same_decision_date_can_have_different_execution_sessions():
    plans = MemoryPlans()
    db = {mongo.PAPER_TRADE_PLANS_COLLECTION: plans}
    manual = {
        "plan_id": "manual-2026-10-14",
        "status": "executed",
        "decision_date": "2026-10-14",
        "execution_session": "2026-10-14",
        "target_asset": "CAPR",
    }
    automatic = {
        "plan_id": "automatic-2026-10-15",
        "status": "prepared",
        "decision_date": "2026-10-14",
        "execution_session": "2026-10-15",
        "target_asset": "CAPR",
    }
    mongo.insert_paper_trade_plan(db, manual)
    mongo.insert_paper_trade_plan(db, automatic)
    assert len(plans.documents) == 2
    assert plans.documents[0]["status"] == "executed"
    assert plans.documents[1]["status"] == "prepared"


def test_replace_only_matches_same_decision_date_and_execution_session():
    plans = MemoryPlans()
    db = {mongo.PAPER_TRADE_PLANS_COLLECTION: plans}
    mongo.insert_paper_trade_plan(db, {
        "plan_id": "old-manual", "status": "executed",
        "decision_date": "2026-10-14", "execution_session": "2026-10-14",
    })
    mongo.insert_paper_trade_plan(db, {
        "plan_id": "updated-next-open", "status": "prepared",
        "decision_date": "2026-10-14", "execution_session": "2026-10-15",
    }, replace=True)
    assert len(plans.documents) == 2
    assert plans.documents[0]["plan_id"] == "old-manual"
    assert plans.documents[1]["plan_id"] == "updated-next-open"
    assert plans.replacements == [{
        "decision_date": "2026-10-14", "execution_session": "2026-10-15",
    }]


def test_index_migration_creates_composite_before_dropping_old_unique():
    plans = MagicMock()
    plans.index_information.return_value = {
        "uq_paper_plan_decision_date": {"unique": True},
    }
    db = MagicMock()
    db.__getitem__.side_effect = lambda key: (
        plans if key == mongo.PAPER_TRADE_PLANS_COLLECTION else MagicMock()
    )
    mongo.ensure_database(db)

    operations = [
        (i, call)
        for i, call in enumerate(plans.method_calls)
        if call[0] in {"create_index", "drop_index"}
    ]
    composite = next(i for i, call in operations
                     if call[0] == "create_index"
                     and call.kwargs.get("name") == "uq_paper_plan_decision_execution_session")
    dropped = next(i for i, call in operations
                   if call[0] == "drop_index")
    assert composite < dropped
    assert plans.drop_index.call_args.args[0] == "uq_paper_plan_decision_date"
    plans.create_index.assert_any_call(
        [("decision_date", DESCENDING), ("execution_session", ASCENDING)],
        unique=True,
        name="uq_paper_plan_decision_execution_session",
    )
    plans.update_many.assert_not_called()
    plans.delete_many.assert_not_called()


def test_index_migration_is_idempotent_after_legacy_index_disappears():
    plans = MagicMock()
    plans.index_information.return_value = {
        "uq_paper_plan_decision_execution_session": {"unique": True}
    }
    db = MagicMock()
    db.__getitem__.side_effect = lambda key: (
        plans if key == mongo.PAPER_TRADE_PLANS_COLLECTION else MagicMock()
    )
    mongo.ensure_database(db)
    plans.drop_index.assert_not_called()
