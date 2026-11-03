"""Read-only, account-scoped date-range audit export for Live Winner."""
from __future__ import annotations

import csv
import io
import json
import unittest
import zipfile
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from market_cycle_trader_api.services import live_portfolio as service
from market_cycle_trader_api.tcc_u67_v1210_reference.rotacao import ROTATION_FEATURES
from market_cycle_trader_api.api.routers import live_market as router


def _run(account, decision, execution, values, **extra):
    return {
        "live_account_id": account, "decision_date": decision,
        "execution_session": execution, "winner_signal_prepared": True,
        "utilities": values, "cash_edges": {"CAPR": 0.3},
        "current_asset": "CAPR", "target_asset": "CAPR",
        "raw_best_asset": "CAPR", "action": "hold",
        "status": "completed", "selected_utility": 0.9,
        "effective_switch_margin": 0.0005, **extra,
    }


class FakeLiveRuns:
    def __init__(self, documents):
        self.documents = documents
        self.queries = []
    def find(self, query, projection):
        self.queries.append(query)
        output = []
        for doc in self.documents:
            if doc.get("live_account_id") != query["live_account_id"]:
                continue
            if not doc.get("winner_signal_prepared"):
                continue
            if not isinstance(doc.get("utilities"), dict):
                continue
            day = str(doc.get("decision_date", ""))
            window = query["decision_date"]
            if not window["$gte"] <= day <= window["$lte"]:
                continue
            output.append({
                key: value for key, value in doc.items()
                if projection.get(key) and key != "_id"
            })
        return output


class FakeDB:
    def __init__(self, runs):
        self.runs = FakeLiveRuns(runs)
    def __getitem__(self, name):
        if name != service.LIVE_RUNS:
            raise AssertionError(f"Unexpected collection {name}")
        return self.runs


class LiveRangeExportTests(unittest.TestCase):
    def setUp(self):
        self.db = FakeDB([
            _run("live-1", "2026-10-26", "2026-10-27",
                 {"CAPR": 1.10, "DNN": 0.48, "CASH": 0.0}),
            _run("live-1", "2026-10-27", "2026-10-28",
                 {"CAPR": 0.7, "DNN": 0.9, "CASH": 0.0},
                 target_asset="DNN", raw_best_asset="DNN"),
            _run("live-1", "2026-11-02", "2026-11-03",
                 {"CAPR": 0.94, "DNN": 0.46, "CASH": 0.0}),
            _run("live-2", "2026-10-27", "2026-10-28",
                 {"CAPR": 9.99, "DNN": 0.5}),
            _run("live-1", "2026-10-25", "2026-10-26",
                 {"CAPR": 999.0}),
        ])
        self.snapshot = {
            "environment": "live", "portfolio_value": 123.45,
            "recent_orders": [{"symbol": "CAPR", "execution_origin": "manual_alpaca_fills"}],
        }
        self.mock_snapshot = patch.object(
            service, "live_portfolio_snapshot", return_value=self.snapshot
        )
        self.mock_broker = patch.object(
            service, "create_live_readonly_client",
            return_value=SimpleNamespace(
                get_account=lambda: SimpleNamespace(id="live-1")
            ),
        )
        self.mock_snapshot.start()
        self.mock_broker.start()
        self.addCleanup(self.mock_snapshot.stop)
        self.addCleanup(self.mock_broker.stop)

    def download(self, first=None, last=None):
        payload = service.build_live_portfolio_export(
            self.db, start_date=first, end_date=last,
        )
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            return {name: archive.read(name).decode("utf-8")
                    for name in archive.namelist()}

    def test_preserves_unfiltered_snapshot_export(self):
        result = self.download()
        self.assertEqual(set(result), {"live_portfolio.json", "README.txt"})
        self.assertEqual(json.loads(result["live_portfolio.json"]), self.snapshot)
        self.assertEqual(self.db.runs.queries, [])

    def test_date_range_includes_all_dates_and_does_not_mix_accounts(self):
        result = self.download(date(2026, 10, 26), date(2026, 11, 2))
        self.assertEqual(len(result), 6)
        self.assertEqual(json.loads(result["live_portfolio.json"]), self.snapshot)
        decisions = list(csv.DictReader(io.StringIO(result["live_decisions_period.csv"])))
        scores = list(csv.DictReader(io.StringIO(result["live_utility_scores_period.csv"])))
        self.assertEqual(len(decisions), 3)
        self.assertEqual(len(scores), 6)
        self.assertEqual(
            [d["decision_date"] for d in decisions],
            ["2026-10-26", "2026-10-27", "2026-11-02"],
        )
        self.assertEqual(
            [s["symbol"] for s in scores[:2]], ["CAPR", "DNN"],
        )
        self.assertEqual(
            [s["symbol"] for s in scores[2:4]], ["DNN", "CAPR"],
        )
        self.assertEqual(scores[0]["rank"], "1")
        self.assertEqual(scores[2]["rank"], "1")
        self.assertEqual(scores[2]["is_target"], "True")
        self.assertNotIn("live-1", result["live_decisions_period.csv"])
        self.assertNotIn("live-2", result["live_utility_scores_period.csv"])
        self.assertEqual(
            self.db.runs.queries[-1]["live_account_id"], "live-1",
        )
        manifest = json.loads(result["audit_period.json"])
        self.assertEqual(manifest["decision_count"], 3)
        self.assertEqual(manifest["days_with_decisions"], 3)
        self.assertEqual(manifest["utility_score_rows"], 6)
        self.assertEqual(manifest["date_basis"], "decision_date (completed market session)")
        self.assertEqual(manifest["shap_attribution_rows"], 0)
        self.assertEqual(len(list(csv.DictReader(io.StringIO(
            result["live_feature_contributions_period.csv"])))), 0)

    def test_same_day_is_inclusive_and_missing_dates_not_fabricated(self):
        result = self.download(date(2026, 10, 27), date(2026, 10, 27))
        rows = list(csv.DictReader(io.StringIO(result["live_decisions_period.csv"])))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["execution_session"], "2026-10-28")

    def test_empty_range_is_explicit_header_only_data(self):
        result = self.download(date(2026, 10, 29), date(2026, 10, 30))
        self.assertEqual(len(list(csv.DictReader(
            io.StringIO(result["live_utility_scores_period.csv"])
        ))), 0)
        self.assertEqual(json.loads(result["audit_period.json"])["decision_count"], 0)
        self.assertIn("live_portfolio.json", result)

    def test_recorded_shap_is_exported_with_52_feature_values(self):
        feature_values = {name: float(index) / 100 for index, name
                          in enumerate(ROTATION_FEATURES)}
        contributions = {name: 0.01 for name in ROTATION_FEATURES}
        sample = _run(
            "live-1", "2026-11-02", "2026-11-03",
            {"CAPR": 0.94, "DNN": 0.46, "CASH": 0.0},
            feature_attributions={
                "CAPR": {
                    "method": "lightgbm_tree_shap",
                    "base_value": 0.94 - len(ROTATION_FEATURES) * 0.01,
                    "prediction": 0.94,
                    "feature_values": feature_values,
                    "contributions": contributions,
                },
            },
        )
        self.db = FakeDB([sample])
        result = self.download(date(2026, 11, 2), date(2026, 11, 2))
        rows = list(csv.DictReader(io.StringIO(
            result["live_feature_contributions_period.csv"]
        )))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["symbol"], "CAPR")
        self.assertEqual(row["method"], "lightgbm_tree_shap")
        self.assertEqual(len(ROTATION_FEATURES), 52)
        self.assertEqual(float(row["prediction"]), 0.94)
        self.assertAlmostEqual(float(row["reconstructed_prediction"]), 0.94)
        self.assertEqual(
            float(row["shap_return_120"]), contributions["return_120"],
        )
        self.assertEqual(
            float(row["value_ema_distance_20"]),
            feature_values["ema_distance_20"],
        )
        manifest = json.loads(result["audit_period.json"])
        self.assertEqual(manifest["shap_attribution_rows"], 1)
        self.assertEqual(manifest["shap_decisions_covered"], 1)

    def test_inconsistent_shap_is_never_exported_as_factual(self):
        sample = _run(
            "live-1", "2026-11-02", "2026-11-03",
            {"CAPR": 0.94, "DNN": 0.46},
            feature_attributions={
                "CAPR": {
                    "method": "lightgbm_tree_shap",
                    "base_value": 100.0,
                    "prediction": 0.94,
                    "feature_values": {name: 0.0 for name in ROTATION_FEATURES},
                    "contributions": {name: 0.0 for name in ROTATION_FEATURES},
                },
            },
        )
        self.db = FakeDB([sample])
        result = self.download(date(2026, 11, 2), date(2026, 11, 2))
        rows = list(csv.DictReader(io.StringIO(
            result["live_feature_contributions_period.csv"]
        )))
        self.assertEqual(rows, [])
        self.assertEqual(
            json.loads(result["audit_period.json"])["shap_attribution_rows"], 0
        )

    def test_reject_invalid_or_one_sided_range(self):
        for first, last in [
            (date(2026, 11, 2), date(2026, 10, 26)),
            (date(2025, 1, 1), date(2026, 11, 2)),
            (date(2026, 10, 26), None),
            (None, date(2026, 10, 26)),
        ]:
            with self.subTest(first=first, last=last):
                with self.assertRaises(ValueError):
                    self.download(first, last)

    def test_service_export_does_not_write_to_database_or_submit_orders(self):
        self.download(date(2026, 10, 26), date(2026, 11, 2))
        self.assertTrue(self.db.runs.queries)
        # Fake DB exposes only .find; any write operation would fail immediately.

    def test_http_route_passes_dates_and_produces_period_filename(self):
        with patch.object(router, "database", return_value=self.db), patch.object(
            router, "build_live_portfolio_export",
            return_value=b"PK fake",
        ) as build:
            result = router.export_portfolio(
                identity=SimpleNamespace(subject="admin"),
                start_date=date(2026, 10, 26),
                end_date=date(2026, 11, 2),
            )
            self.assertEqual(result.body, b"PK fake")
            build.assert_called_once_with(
                self.db,
                start_date=date(2026, 10, 26),
                end_date=date(2026, 11, 2),
            )
            self.assertIn(
                "2026-10-26_2026-11-02",
                result.headers["content-disposition"],
            )

    def test_http_invalid_range_returns_422(self):
        with patch.object(router, "database", return_value=self.db):
            with self.assertRaises(HTTPException) as context:
                router.export_portfolio(
                    identity=SimpleNamespace(subject="admin"),
                    start_date=date(2026, 11, 2),
                    end_date=date(2026, 10, 26),
                )
        self.assertEqual(context.exception.status_code, 422)


if __name__ == "__main__":
    unittest.main()
