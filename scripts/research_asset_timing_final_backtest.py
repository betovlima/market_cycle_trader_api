from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
from pymongo import MongoClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import research_asset_signature_leave_one_out as common  # noqa: E402
from research_asset_signature_pipeline_backtest import (  # noqa: E402
    JOBS_COLLECTION,
    compare_results,
    exact_baseline_job,
)
from research_asset_signature_pipeline_ranking import canonical_hash  # noqa: E402

SCRIPT_VERSION = "asset-timing-final-backtest-v1.3"


def _log(message: str) -> None:
    stamp = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{stamp}] {message}", flush=True)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run exactly one full Strategy backtest after the point-in-time asset timing "
            "qualification universe has been frozen."
        )
    )
    parser.add_argument("--strategy-sequence", type=int, default=10)
    parser.add_argument("--strategy-id", default=None)
    parser.add_argument("--snapshot-end", required=True)
    parser.add_argument("--mongo-uri", default=None)
    parser.add_argument("--database", default=None)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--frozen-snapshot", default=None)
    parser.add_argument("--baseline-job-id", default=None)
    parser.add_argument("--allow-remote-mongo", action="store_true")
    return parser


def _verify_frozen_snapshot(frozen: dict[str, Any]) -> None:
    expected = str(frozen.get("decision_snapshot_sha256") or "")
    canonical = dict(frozen)
    canonical.pop("decision_snapshot_sha256", None)
    actual = _sha256_json(canonical)
    if not expected or actual != expected:
        raise RuntimeError(
            "Frozen timing snapshot hash mismatch. The qualification decision changed after freezing."
        )
    if frozen.get("full_strategy_backtest_used_for_selection") is not False:
        raise RuntimeError("Frozen snapshot does not certify Backtest-independent asset qualification.")


def _build_job(
    baseline_job: dict[str, Any],
    strategy: dict[str, Any],
    qualified_assets: list[str],
    frozen: dict[str, Any],
    snapshot_end: pd.Timestamp,
) -> tuple[dict[str, Any], dict[str, Any]]:
    request = deepcopy(baseline_job.get("request") or {})
    if not request:
        raise RuntimeError("Baseline Backtest has no immutable request payload.")

    qualified_set = set(qualified_assets)
    request["assets"] = list(qualified_assets)
    request["analysis_end_date"] = snapshot_end.date().isoformat()
    request["end_date"] = snapshot_end.date().isoformat()
    request["research_market_data_mode"] = "database_only"
    request["research_reference_assets"] = list(frozen.get("retained_existing_assets") or [])
    request["research_candidate_assets"] = list(frozen.get("added_candidate_assets") or [])

    anchors = [
        str(item).upper()
        for item in request.get("calendar_anchor_assets") or []
        if str(item).upper() in qualified_set
    ]
    if len(anchors) < 2:
        anchors = list(qualified_assets[: min(2, len(qualified_assets))])
    request["calendar_anchor_assets"] = anchors

    now = datetime.now(timezone.utc)
    job = deepcopy(baseline_job)
    job.pop("_id", None)
    for key in (
        "process_id",
        "return_code",
        "timed_out",
        "error",
        "cancel_requested",
        "cancel_reason",
    ):
        job.pop(key, None)
    job.update(
        {
            "id": now.strftime("%Y%m%dT%H%M%S") + "-timing-" + uuid4().hex[:8],
            "status": "queued",
            "stage": "Queued",
            "progress": 0,
            "completed_runs": 0,
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "finished_at": None,
            "request": request,
            "strategy_profile_name": f"{strategy.get('name') or 'Strategy'} · timing-qualified universe",
            "strategy_configuration_hash": canonical_hash(request),
            "source_strategy_configuration_hash": strategy.get("configuration_hash"),
            "research_reference_assets": list(frozen.get("retained_existing_assets") or []),
            "research_candidate_assets": list(frozen.get("added_candidate_assets") or []),
            "certifies_strategy": False,
            "internal_job": True,
            "tuning_summary_only": False,
            "tuning_run_id": None,
            "tuning_candidate_id": None,
            "logs": [
                "Final research Backtest queued only after asset timing qualification was frozen."
            ],
            "progress_detail": {},
            "timing_validation_snapshot_sha256": frozen["decision_snapshot_sha256"],
            "timing_ranking_sha256": frozen["ranking_sha256"],
            "qualified_assets_sha256": frozen["qualified_assets_sha256"],
            "experiment_kind": "post_asset_timing_qualification_full_backtest",
        }
    )
    return job, request


def _bind_explicit_runtime_mongo(mongo_uri: str, database_name: str) -> Any:
    """Bind the research process and its Backtest subprocess to the exact Mongo used above."""
    os.environ["MONGO_URL"] = mongo_uri
    os.environ["MONGO_URI"] = mongo_uri
    os.environ["MONGO_DATABASE"] = database_name

    # These modules may already be imported by the research helpers, so their
    # module-level environment snapshots must be synchronized explicitly.
    from market_cycle_trader_api.infrastructure.persistence import mongo_repository
    from market_cycle_trader_api.core import runtime

    mongo_repository.MONGO_URI = mongo_uri
    mongo_repository.MONGO_DATABASE = database_name
    runtime.MONGO_URI = mongo_uri
    runtime.MONGO_DATABASE = database_name

    runtime.initialize_mongo(role="research")
    if not bool(runtime.MONGO_STATUS.get("available")):
        raise RuntimeError(
            "Explicit local MongoDB runtime initialization failed: "
            + str(runtime.MONGO_STATUS.get("message") or "unknown MongoDB initialization error")
        )
    return runtime


def main() -> int:
    args = _parser().parse_args()
    common.load_project_environment(args.env_file)

    mongo_uri = str(
        args.mongo_uri
        or os.getenv("MONGO_URL")
        or os.getenv("MONGO_URI")
        or "mongodb://localhost:27017"
    ).strip()
    database_name = str(args.database or os.getenv("MONGO_DATABASE") or "").strip()
    if not database_name:
        raise RuntimeError("MONGO_DATABASE is required in .env or via --database.")
    common._assert_local_mongo(mongo_uri, bool(args.allow_remote_mongo))

    snapshot_end = common._normalize_date(args.snapshot_end)
    output_dir = Path(
        args.output_dir
        or PROJECT_ROOT
        / "research_output"
        / f"asset_timing_strategy_{args.strategy_sequence}_{snapshot_end.date().isoformat()}"
    ).resolve()
    frozen_path = Path(args.frozen_snapshot).resolve() if args.frozen_snapshot else output_dir / "timing_validation_snapshot_frozen.json"
    if not frozen_path.exists():
        raise RuntimeError(f"Frozen timing qualification snapshot not found: {frozen_path}")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    _verify_frozen_snapshot(frozen)

    if int(frozen.get("strategy_sequence") or 0) != int(args.strategy_sequence):
        raise RuntimeError("Frozen snapshot Strategy sequence does not match the requested Strategy.")
    if str(frozen.get("snapshot_end") or "") != snapshot_end.date().isoformat():
        raise RuntimeError("Frozen snapshot market date does not match --snapshot-end.")

    qualified_assets = [str(item).strip().upper() for item in frozen.get("qualified_assets") or [] if str(item).strip()]
    qualified_assets = list(dict.fromkeys(qualified_assets))
    if len(qualified_assets) < 2:
        raise RuntimeError("Frozen qualified universe has fewer than two assets; full rotation Backtest cannot run.")

    client = MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=3_000,
        connectTimeoutMS=3_000,
        maxPoolSize=8,
        retryWrites=False,
    )
    client.admin.command("ping")
    db = client[database_name]
    strategy = common._strategy_document(db, args.strategy_sequence, args.strategy_id)
    baseline_job = exact_baseline_job(db, strategy, snapshot_end, args.baseline_job_id)
    client.close()

    _log(
        f"Frozen universe verified: {len(qualified_assets)} assets. "
        f"Baseline job={baseline_job['id']}."
    )
    _log("Starting ONE full Strategy Backtest. Its result cannot add/remove assets from this frozen run.")

    runtime = _bind_explicit_runtime_mongo(mongo_uri, database_name)
    from market_cycle_trader_api.services.jobs import run_job
    from market_cycle_trader_api.services.results import build_results

    runtime_db = runtime.database()
    _log(f"Runtime MongoDB bound explicitly to database={database_name}.")
    job, request = _build_job(
        baseline_job,
        strategy,
        qualified_assets,
        frozen,
        snapshot_end,
    )
    _write_json(output_dir / "baseline_backtest_job.json", {key: value for key, value in baseline_job.items() if key != "_id"})
    _write_json(output_dir / "final_strategy_backtest_request.json", request)
    runtime_db[JOBS_COLLECTION].insert_one(deepcopy(job))

    started = time.perf_counter()
    try:
        baseline_results = build_results(str(baseline_job["id"]))
        _write_json(output_dir / "baseline_backtest_results.json", baseline_results)
        run_job(job["id"])
        completed = runtime_db[JOBS_COLLECTION].find_one({"id": job["id"]}) or {}
        _write_json(
            output_dir / "final_strategy_backtest_job.json",
            {key: value for key, value in completed.items() if key != "_id"},
        )
        if completed.get("status") != "completed":
            _write_json(
                output_dir / "final_experiment_summary.json",
                {
                    "status": "backtest_failed",
                    "backtest_job_id": job["id"],
                    "timing_validation_snapshot_sha256": frozen["decision_snapshot_sha256"],
                    "qualified_assets": qualified_assets,
                    "full_strategy_backtest_used_for_selection": False,
                },
            )
            raise RuntimeError(f"Final Strategy Backtest failed: {job['id']}")

        candidate_results = build_results(job["id"])
        comparison = compare_results(baseline_results, candidate_results)
        comparison.update(
            {
                "baseline_job_id": baseline_job["id"],
                "candidate_job_id": job["id"],
                "original_asset_count": len(frozen.get("original_assets") or []),
                "qualified_asset_count": len(qualified_assets),
                "retained_existing_assets": frozen.get("retained_existing_assets") or [],
                "removed_existing_assets": frozen.get("removed_existing_assets") or [],
                "added_candidate_assets": frozen.get("added_candidate_assets") or [],
                "timing_validation_snapshot_sha256": frozen["decision_snapshot_sha256"],
                "interpretation": (
                    "This one full Backtest validates the already-frozen timing-qualified universe. "
                    "It did not participate in asset qualification."
                ),
            }
        )
        _write_json(output_dir / "final_strategy_backtest_results.json", candidate_results)
        _write_json(output_dir / "backtest_comparison.json", comparison)
        _write_json(
            output_dir / "final_experiment_summary.json",
            {
                "schema_version": 2,
                "status": "completed",
                "script_version": SCRIPT_VERSION,
                "strategy_id": str(strategy.get("_id") or ""),
                "strategy_sequence": int(strategy.get("strategy_sequence") or args.strategy_sequence),
                "snapshot_end": snapshot_end.date().isoformat(),
                "original_asset_count": len(frozen.get("original_assets") or []),
                "qualified_asset_count": len(qualified_assets),
                "qualified_assets": qualified_assets,
                "retained_existing_assets": frozen.get("retained_existing_assets") or [],
                "removed_existing_assets": frozen.get("removed_existing_assets") or [],
                "added_candidate_assets": frozen.get("added_candidate_assets") or [],
                "timing_validation_snapshot_sha256": frozen["decision_snapshot_sha256"],
                "timing_ranking_sha256": frozen["ranking_sha256"],
                "qualified_assets_sha256": frozen["qualified_assets_sha256"],
                "full_strategy_backtest_used_for_selection": False,
                "full_strategy_backtest_run_after_freeze": True,
                "baseline_backtest_job_id": baseline_job["id"],
                "candidate_backtest_job_id": job["id"],
                "backtest_elapsed_seconds": float(time.perf_counter() - started),
                "backtest_comparison": comparison,
            },
        )
    finally:
        runtime.close_mongo()

    _log(f"Completed. Final comparison: {output_dir / 'backtest_comparison.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
