from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "market_cycle_trader_api"


def test_public_job_endpoint_has_no_date_request_model() -> None:
    router = (SRC / "api" / "routers" / "jobs.py").read_text(encoding="utf-8")
    requests = (SRC / "schemas" / "requests.py").read_text(encoding="utf-8")

    assert "def create_job()" in router
    assert "PublicBacktestRequest" not in router
    assert "class PublicBacktestRequest" not in requests
    assert '"analysis_start_date": locked_configuration.start_date' in router
    assert '"analysis_end_date": locked_configuration.end_date' in router
    assert "public_date_range" not in router


def test_public_results_and_zip_do_not_export_locked_configuration() -> None:
    results = (SRC / "services" / "results.py").read_text(encoding="utf-8")
    exports = (SRC / "api" / "routers" / "exports.py").read_text(encoding="utf-8")

    assert '"effectiveConfig"' not in results
    assert '"reproducibility"' not in results
    assert '"effective_config.json"' not in exports
    assert '"reproducibility.json"' not in exports


def test_winner_v1_13_2_owns_the_execution_period() -> None:
    winner = SRC / "parameterizations" / "winner-v1.13.2.json"
    assert winner.exists()
    text = winner.read_text(encoding="utf-8")
    assert '"start_date"' in text
    assert '"end_date"' in text
