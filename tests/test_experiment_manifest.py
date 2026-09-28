from __future__ import annotations

import unittest

from market_cycle_trader_api.services.experiment_manifest import (
    build_experiment_manifest,
)


class ExperimentManifestTests(unittest.TestCase):
    def test_exports_alpaca_history_audit_aggregates(self) -> None:
        job = {
            "id": "job-1",
            "status": "completed",
            "request": {
                "calendar_anchor_assets": ["AAPL"],
            },
        }
        runs = [
            {
                "metrics": {
                    "market_data_signature_sha256": "model-signature",
                    "market_data_audit_signature_sha256": "audit-signature",
                    "market_data_signatures": {
                        "AAPL": {
                            "raw_sha256": "raw-current",
                            "previous_raw_sha256": "raw-previous",
                            "historical_data_changed": True,
                        }
                    },
                    "market_data_history_complete": True,
                    "market_data_history_compared_assets": ["AAPL"],
                    "market_data_history_compared_asset_count": 1,
                    "market_data_history_changed_assets": ["AAPL"],
                    "market_data_history_changed_asset_count": 1,
                    "market_data_history_comparison_unavailable_assets": [],
                }
            }
        ]

        manifest = build_experiment_manifest(job, runs)

        self.assertEqual(manifest["schema_version"], 3)
        self.assertEqual(
            manifest["market_data_audit_signature_sha256"],
            "audit-signature",
        )
        self.assertEqual(
            manifest["market_data_history_compared_assets"],
            ["AAPL"],
        )
        self.assertEqual(
            manifest["market_data_history_compared_asset_count"],
            1,
        )
        self.assertEqual(
            manifest["market_data_history_changed_assets"],
            ["AAPL"],
        )
        self.assertEqual(
            manifest["market_data_history_changed_asset_count"],
            1,
        )
        self.assertEqual(
            manifest["market_data_history_comparison_unavailable_assets"],
            [],
        )


if __name__ == "__main__":
    unittest.main()
