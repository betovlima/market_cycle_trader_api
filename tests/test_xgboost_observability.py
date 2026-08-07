from __future__ import annotations

from pathlib import Path

from market_cycle_trader_api.services.jobs import public_job

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "market_cycle_trader_api"


def test_public_job_exposes_only_sanitized_training_progress() -> None:
    payload = public_job({
        "id": "job-1",
        "status": "running",
        "stage": "Run 1/2 — fold 2/3 — final training 8/21",
        "progress": 54.0,
        "progress_detail": {
            "run_index": 1,
            "run_count": 2,
            "fold_index": 2,
            "fold_count": 3,
            "phase": "Final training",
            "trained_models": 8,
            "total_models": 21,
            "device": "CPU",
            "asset": "NVDA",
            "seed": 42,
            "unexpected_detail": "protected",
        },
        "logs": [
            "Run 1/2 — fold 2/3 — final training 8/21",
            "XGB_TECH|asset=NVDA seed=42",
        ],
    })

    assert payload is not None
    assert payload["stage"] == "Running analysis"
    assert payload["logs"] == ["Run 1/2 — fold 2/3 — final training 8/21"]
    assert payload["progress_detail"] == {
        "run_index": 1,
        "run_count": 2,
        "fold_index": 2,
        "fold_count": 3,
        "phase": "Final training",
        "trained_models": 8,
        "total_models": 21,
        "device": "CPU",
    }


def test_public_job_rejects_unrecognized_progress_labels() -> None:
    payload = public_job({
        "id": "job-unsafe",
        "status": "running",
        "stage": "Running analysis",
        "progress_detail": {
            "run_index": 1,
            "run_count": 1,
            "fold_index": 1,
            "fold_count": 3,
            "phase": "Training NVDA with protected features",
            "trained_models": 1,
            "total_models": 21,
            "device": "CUDA — hidden device name",
        },
    })

    assert payload is not None
    assert payload["progress_detail"] == {
        "run_index": 1,
        "run_count": 1,
        "fold_index": 1,
        "fold_count": 3,
        "trained_models": 1,
        "total_models": 21,
    }


def test_terminal_job_status_overrides_stale_stage() -> None:
    payload = public_job({
        "id": "job-2",
        "status": "failed",
        "stage": "Preparing aligned panel",
        "logs": ["ERROR: protected failure"],
    })

    assert payload is not None
    assert payload["stage"] == "Backtest failed"
    assert payload["logs"] == ["Backtest failed. Check the protected server logs."]


def test_xgboost_engine_contains_console_and_model_progress_hooks() -> None:
    engine = (SRC / "engine" / "capital_rotation.py").read_text(encoding="utf-8")
    worker = (SRC / "services" / "jobs.py").read_text(encoding="utf-8")
    subprocess_engine = (SRC / "engine" / "compound_rotation_backtest.py").read_text(encoding="utf-8")

    assert "technical_log_callback" in engine
    assert "progress_detail_callback" in engine
    assert "event=model_complete" in engine
    assert "JOB_DETAIL|" in subprocess_engine
    assert "XGB_TECH|" in subprocess_engine
    assert "_write_child_line_to_console" in worker
    assert "logger.exception" in worker
