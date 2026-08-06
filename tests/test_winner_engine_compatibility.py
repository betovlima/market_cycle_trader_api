from __future__ import annotations

from pathlib import Path

from market_cycle_trader_api.services.jobs import numeric_thread_environment

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "market_cycle_trader_api"


def test_non_deterministic_winner_inherits_host_numeric_runtime() -> None:
    environment = numeric_thread_environment(
        {
            "deterministic_execution": False,
            "numeric_thread_limit": 1,
            "xgb_n_jobs": -1,
        }
    )
    assert environment == {}


def test_deterministic_strategy_applies_numeric_thread_limit() -> None:
    environment = numeric_thread_environment(
        {
            "deterministic_execution": True,
            "numeric_thread_limit": 3,
        }
    )
    assert environment == {
        "OMP_NUM_THREADS": "3",
        "OPENBLAS_NUM_THREADS": "3",
        "MKL_NUM_THREADS": "3",
        "NUMEXPR_NUM_THREADS": "3",
    }


def test_capital_rotation_preserves_api_v1_13_16_thread_semantics() -> None:
    source = (SRC / "engine" / "capital_rotation.py").read_text(encoding="utf-8")
    assert "from contextlib import nullcontext" in source
    assert "if not bool(config.deterministic_execution):" in source
    assert "return nullcontext()" in source
    assert "return threadpool_limits(limits=int(config.numeric_thread_limit))" in source


def test_jobs_record_winner_engine_compatibility() -> None:
    source = (SRC / "services" / "jobs.py").read_text(encoding="utf-8")
    router = (SRC / "api" / "routers" / "jobs.py").read_text(encoding="utf-8")
    assert 'WINNER_ENGINE_COMPATIBILITY = "api-v1.13.16"' in source
    assert '"winner_engine_compatibility": WINNER_ENGINE_COMPATIBILITY' in source
    assert '"winner_engine_compatibility": "api-v1.13.16"' in router
