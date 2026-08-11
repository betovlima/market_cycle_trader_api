from __future__ import annotations

from market_cycle_trader_api.services.experiment_manifest import build_experiment_manifest


def test_experiment_manifest_binds_reference_candidate_runtime_and_folds() -> None:
    job = {
        "id": "job-1",
        "status": "completed",
        "strategy_profile_id": "candidate-1",
        "strategy_profile_name": "Candidate",
        "strategy_profile_revision": 4,
        "strategy_configuration_hash": "cfg",
        "winner_engine_compatibility": "api-v1.13.16",
        "trader_winner_strategy_id_at_queue": "winner-1",
        "trader_winner_strategy_name_at_queue": "Winner",
        "trader_winner_configuration_hash_at_queue": "winner-cfg",
        "trader_winner_api_version_at_queue": "1.13.36",
        "research_reference_assets": ["AAPL", "MSFT"],
        "research_candidate_assets": ["NVDA"],
        "request": {"calendar_anchor_assets": ["AAPL", "MSFT"], "assets": ["AAPL", "MSFT", "NVDA"]},
    }
    runs = [
        {
            "metrics": {
                "market_data_signature_sha256": "market",
                "runtime_fingerprint_sha256": "runtime",
                "engine_source_sha256": "engine",
                "git_commit": "commit",
                "decision_diagnostics_schema_version": 1,
                "decision_diagnostics_rows": 10,
                "walk_forward_fold_count": 3,
                "walk_forward_folds": [{"fold_id": 1}],
            }
        }
    ]

    manifest = build_experiment_manifest(job, runs)

    assert manifest["schema_version"] == 2
    assert manifest["trader_winner_configuration_hash_at_queue"] == "winner-cfg"
    assert manifest["trader_winner_api_version_at_queue"] == "1.13.36"
    assert manifest["research_reference_assets"] == ["AAPL", "MSFT"]
    assert manifest["research_candidate_assets"] == ["NVDA"]
    assert manifest["execution_calendar_anchor_assets"] == ["AAPL", "MSFT"]
    assert manifest["market_data_signature_sha256"] == "market"
    assert manifest["runtime_fingerprint_sha256"] == "runtime"
    assert manifest["engine_source_sha256"] == "engine"
    assert manifest["decision_diagnostics_rows"] == 10
    assert manifest["walk_forward_fold_count"] == 3
