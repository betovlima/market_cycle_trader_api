"""Regression tests for conservative Paper plan status reconciliation."""
from __future__ import annotations

import unittest
from copy import deepcopy
from datetime import date
from unittest.mock import patch

from market_cycle_trader_api.services import paper_plan_reconciliation as reconcile


class Cursor(list):
    def sort(self, *args, **kwargs):
        return self

    def limit(self, n):
        return Cursor(self[:n])


class Collection:
    def __init__(self, docs=None):
        self.docs = deepcopy(docs or [])
        self.write_count = 0

    @staticmethod
    def _matches(doc, spec):
        if "$or" in spec:
            return any(Collection._matches(doc, q) for q in spec["$or"])
        for key, value in spec.items():
            if isinstance(value, dict) and "$exists" in value:
                if (key in doc) != bool(value["$exists"]):
                    return False
            elif doc.get(key) != value:
                return False
        return True

    def find(self, query, projection=None):
        return Cursor([deepcopy(d) for d in self.docs if self._matches(d, query)])

    def find_one(self, query, projection=None):
        for doc in self.docs:
            if self._matches(doc, query):
                return deepcopy(doc)
        return None

    def find_one_and_update(self, query, changes, return_document=None):
        for doc in self.docs:
            if self._matches(doc, query):
                doc.update(deepcopy(changes["$set"]))
                self.write_count += 1
                return deepcopy(doc)
        return None

    def insert_one(self, doc):
        self.docs.append(deepcopy(doc))
        self.write_count += 1


def sample_plan(plan_id, session, action="hold", current="MAN", target="MAN"):
    return {
        "_id": plan_id,
        "plan_id": plan_id,
        "status": "prepared",
        "created_at": f"{session}T13:00:00Z",
        "updated_at": f"{session}T13:00:00Z",
        "execution_session": session,
        "action": action,
        "current_asset": current,
        "target_asset": target,
    }


def make_db(plans, *, orders=None, runs=None):
    return {
        reconcile.PAPER_TRADE_PLANS_COLLECTION: Collection(plans),
        reconcile.PAPER_TRADE_ORDERS_COLLECTION: Collection(orders),
        reconcile.PAPER_MARKET_RUNS_COLLECTION: Collection(runs),
        reconcile.ADMIN_OPERATION_LOGS_COLLECTION: Collection(),
    }


class PaperPlanReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.date_patch = patch.object(reconcile, "_market_today", return_value=date(2026, 10, 15))
        self.date_patch.start()
        self.addCleanup(self.date_patch.stop)

    def test_preview_is_read_only_and_labels_two_stale_holds(self):
        ids = ["lightgbm-2026-10-12-ea070a26", "lightgbm-2026-10-13-4cdbd908"]
        db = make_db([sample_plan(ids[0], "2026-10-13"), sample_plan(ids[1], "2026-10-14")])
        result = reconcile.preview_paper_plan_reconciliation(db)

        self.assertEqual(result["market_date"], "2026-10-15")
        self.assertEqual(result["eligible_count"], 2)
        self.assertEqual([item["plan_id"] for item in result["items"]], ids)
        self.assertEqual(db[reconcile.PAPER_TRADE_PLANS_COLLECTION].write_count, 0)
        self.assertEqual(db[reconcile.ADMIN_OPERATION_LOGS_COLLECTION].write_count, 0)

    def test_explicit_close_preserves_plan_decision_and_records_evidence(self):
        plan = sample_plan("hold-13", "2026-10-13")
        db = make_db([plan])
        result = reconcile.close_obsolete_paper_plans(
            db, plan_ids=["hold-13"], actor_email="admin@example.com",
        )
        saved = db[reconcile.PAPER_TRADE_PLANS_COLLECTION].docs[0]
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["changed_count"], 1)
        self.assertEqual(saved["status"], "cancelled")
        self.assertEqual(saved["action"], "hold")
        self.assertEqual(saved["current_asset"], "MAN")
        self.assertEqual(saved["target_asset"], "MAN")
        self.assertEqual(saved["status_reconciliation"]["previous_status"], "prepared")
        self.assertEqual(saved["status_reconciliation"]["actor_email"], "admin@example.com")
        self.assertEqual(db[reconcile.PAPER_TRADE_ORDERS_COLLECTION].write_count, 0)
        self.assertEqual(db[reconcile.ADMIN_OPERATION_LOGS_COLLECTION].write_count, 1)

    def test_no_current_session_close_even_if_hold(self):
        db = make_db([sample_plan("hold-today", "2026-10-15")])
        result = reconcile.preview_paper_plan_reconciliation(db)
        self.assertFalse(result["items"][0]["eligible"])
        self.assertIn("session_not_in_past", result["items"][0]["blockers"])

    def test_any_associated_order_blocks_including_cancelled_partial_fill(self):
        db = make_db(
            [sample_plan("old-hold", "2026-10-13")],
            orders=[{
                "plan_id": "old-hold", "status": "canceled",
                "filled_quantity": 14.0,
            }],
        )
        result = reconcile.close_obsolete_paper_plans(
            db, plan_ids=["old-hold"], actor_email="admin@example.com",
        )
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["changed_count"], 0)
        self.assertEqual(db[reconcile.PAPER_TRADE_PLANS_COLLECTION].docs[0]["status"], "prepared")

    def test_active_or_review_required_run_blocks_close(self):
        for status in ("armed", "prepared", "executing", "review_required"):
            with self.subTest(status=status):
                db = make_db(
                    [sample_plan("old-hold", "2026-10-13")],
                    runs=[{
                        "plan_id": "old-hold", "status": status,
                        "run_id": "run-1",
                    }],
                )
                result = reconcile.preview_paper_plan_reconciliation(db)
                self.assertIn("linked_run_not_terminal", result["items"][0]["blockers"])

    def test_trading_plan_and_execution_evidence_block_close(self):
        for changed in (
            {"action": "rotate", "target_asset": "CAPR"},
            {"execution_started_at": "2026-10-13T13:30:00Z"},
        ):
            with self.subTest(changed=changed):
                plan = sample_plan("unsafe", "2026-10-13")
                plan.update(changed)
                db = make_db([plan])
                result = reconcile.preview_paper_plan_reconciliation(db)
                self.assertFalse(result["items"][0]["eligible"])

    def test_preflight_prevents_changes_if_any_request_is_unsafe(self):
        db = make_db([
            sample_plan("hold-13", "2026-10-13"),
            sample_plan("rotate-13", "2026-10-13", action="rotate", target="CAPR"),
        ])
        result = reconcile.close_obsolete_paper_plans(
            db, plan_ids=["hold-13", "rotate-13"], actor_email="admin@example.com",
        )
        self.assertEqual(result["changed_count"], 0)
        self.assertEqual(db[reconcile.PAPER_TRADE_PLANS_COLLECTION].write_count, 0)

    def test_compare_and_set_refuses_concurrently_changed_status(self):
        plan = sample_plan("hold-13", "2026-10-13")
        db = make_db([plan])
        col = db[reconcile.PAPER_TRADE_PLANS_COLLECTION]

        def changed_after_read(query, changes, return_document=None):
            col.docs[0]["status"] = "executed"
            return None

        col.find_one_and_update = changed_after_read
        result = reconcile.close_obsolete_paper_plans(
            db, plan_ids=["hold-13"], actor_email="admin@example.com",
        )
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["changed_count"], 0)
        self.assertEqual(col.docs[0]["status"], "executed")
        self.assertEqual(db[reconcile.ADMIN_OPERATION_LOGS_COLLECTION].write_count, 0)

    def test_duplicate_plan_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            reconcile.close_obsolete_paper_plans(
                make_db([]), plan_ids=["same", "same"], actor_email="admin@example.com",
            )


if __name__ == "__main__":
    unittest.main()
