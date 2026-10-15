"""Independent Live monitoring mode mutations; no real trading or Paper writes."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from market_cycle_trader_api.services.live_control import (
    LIVE_CONTROL_COLLECTION,
    LIVE_CONTROL_HISTORY_COLLECTION,
    LiveControlConflict,
    set_live_control_mode,
    live_control_status,
)


class FakeDB:
    def __init__(self, *, mode="stopped", revision=0, concurrent=False):
        self.read = []
        self.mode = mode
        self.revision = revision
        self.concurrent = concurrent
        self.history = []
        self.filters = []

    def __getitem__(self, name):
        self.read.append(name)
        if name == LIVE_CONTROL_COLLECTION:
            return self
        if name == LIVE_CONTROL_HISTORY_COLLECTION:
            return self
        raise AssertionError("Unexpected or Paper collection accessed: " + name)

    def update_one(self, query, changes, *, upsert=False):
        self.filters.append((query, changes, upsert))
        return MagicMock()

    def find_one(self, query):
        return {
            "_id": "default",
            "control_mode": self.mode,
            "revision": self.revision,
        }

    def find_one_and_update(self, query, changes, **kwargs):
        if self.concurrent:
            return None
        if query["revision"] != self.revision:
            return None
        self.revision += 1
        self.mode = changes["$set"]["control_mode"]
        return {"control_mode": self.mode, "revision": self.revision}

    def insert_one(self, doc):
        self.history.append(doc)
        return MagicMock()


class LiveModeTests(unittest.TestCase):
    def test_default_status_has_real_execution_disabled(self):
        db = FakeDB()
        status = live_control_status(db)
        self.assertEqual(status["control_mode"], "stopped")
        self.assertFalse(status["execution_enabled"])
        self.assertFalse(status["order_submission_enabled"])
        self.assertFalse(status["monitoring_enabled"])
        self.assertEqual(db.read, [LIVE_CONTROL_COLLECTION, LIVE_CONTROL_COLLECTION])

    def test_all_four_modes_independent_and_audited(self):
        db = FakeDB()
        sequence = ("active", "paused", "exit_only", "stopped")
        with patch("market_cycle_trader_api.services.live_control.live_credentials", return_value=("live-key", "live-secret")):
            for index, mode in enumerate(sequence):
                status = set_live_control_mode(
                    db, mode=mode, expected_revision=index,
                    actor_email="admin@example.test", reason="operator mode change",
                )
                self.assertEqual(status["control_mode"], mode)
                self.assertEqual(status["revision"], index + 1)
                self.assertFalse(status["execution_enabled"])
                self.assertFalse(status["order_submission_enabled"])
        self.assertEqual([x["new_mode"] for x in db.history], list(sequence))
        self.assertTrue(all(x["market"] == "live" for x in db.history))
        self.assertTrue(set(db.read) <= {LIVE_CONTROL_COLLECTION, LIVE_CONTROL_HISTORY_COLLECTION})

    def test_stale_revision_does_not_change_mode(self):
        db = FakeDB(mode="active", revision=2)
        with self.assertRaises(LiveControlConflict):
            set_live_control_mode(
                db, mode="paused", expected_revision=1,
                actor_email="admin@example.test", reason="pause monitoring",
            )
        self.assertEqual(db.mode, "active")
        self.assertFalse(db.history)

    def test_concurrent_update_does_not_write_history(self):
        db = FakeDB(mode="active", revision=1, concurrent=True)
        with self.assertRaises(LiveControlConflict):
            set_live_control_mode(
                db, mode="paused", expected_revision=1,
                actor_email="admin@example.test", reason="pause monitoring",
            )
        self.assertFalse(db.history)

    def test_invalid_mode_is_rejected(self):
        db = FakeDB()
        with self.assertRaises(ValueError):
            set_live_control_mode(
                db, mode="trade_now", expected_revision=0,
                actor_email="admin@example.test", reason="bad mode",
            )
        self.assertEqual(db.read, [])


if __name__ == "__main__":
    unittest.main()
