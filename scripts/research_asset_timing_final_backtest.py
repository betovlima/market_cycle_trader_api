from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from market_cycle_trader_api.infrastructure.persistence.mongo_repository import MongoRepository  # noqa: E402
from market_cycle_trader_api.services.strategy_configuration import get_strategy_profile  # noqa: E402
from market_cycle_trader_api.services.jobs import create_job, get_job  # noqa: E402

SCRIPT_VERSION = "asset-timing-final-backtest-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Runs one full Strategy backtest using the already-frozen asset-timing qualified universe."
    )
    parser.add_argument("--strategy-sequence", type=int, default=10)
    parser.add_argument("--snapshot-end", required=True)
    parser.add_argument("--mongo-uri", default=os.getenv("MONGO_URI") or "mongodb://localhost:27017")
    parser.add_argument("--database", default=os.getenv("MONGO_DB") or os.getenv("MONGODB_DB") or "market_cycle_trader")
    parser.add_argument("--output-root", default="research_output")
    parser.add_argument("--frozen-snapshot", default=None)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    return parser.parse_args()


def _sha256_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _is_local_mongo(uri: str) -> bool:
    raw = str(uri or "").lower()
    return any(token in raw for token in ("localhost", "127.0.0.1", "::1"))


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _strategy_config(profile: Any) -> Any:
    for name in ("config", "configuration", "strategy_config"):
        value = _get(profile, name)
        if value is not None:
            return value
    raise RuntimeError("Could not resolve Strategy configuration snapshot.")


def _request_from_config(config: Any, profile: Any, assets: list[str], snapshot_end: str) -> dict[str, Any]:
    if hasattr(config, "model_dump"):
        request = config.model_dump(mode="json")
    elif isinstance(config, dict):
        request = dict(config)
    else:
        request = dict(vars(config))

    request["assets"] = list(assets)
    request["end_date"] = snapshot_end
    request["analysis_end_date"] = snapshot_end
    request["research_market_data_mode"] = "database_only"
    request["internal_job"] = True
    request["certifies_strategy"] = False
    request["research_experiment"] = "asset_timing_vs_buyhold_final_validation"
    request["research_reference_assets"] = []
    request["research_candidate_assets"] = list(assets)

    strategy_profile_id = _get(profile, "id") or _get(profile, "_id") or _get(profile, "strategy_profile_id")
    revision = _get(profile, "revision")
    configuration_hash = _get(profile, "configuration_hash") or _get(profile, "config_hash")
    if strategy_profile_id is not None:
        request["strategy_profile_id"] = str(strategy_profile_id)
    if revision is not None:
        request["strategy_revision"] = revision
    if configuration_hash is not None:
        request["strategy_configuration_hash"] = str(configuration_hash)
    return request


def _wait_job(db: Any, job_id: str, poll_seconds: float) -> dict[str, Any]:
    import time

    terminal = {"completed", "failed", "cancelled", "canceled", "stopped"}
    while True:
        job = get_job(db, job_id)
        if job is None:
            raise RuntimeError(f"Backtest job {job_id} disappeared.")
        status = str(_get(job, "status", "")).lower()
        progress = _get(job, "progress")
        stage = _get(job, "progress_stage") or _get(job, "stage") or ""
        print(f"Backtest {job_id}: status={status} progress={progress} stage={stage}")
        if status in terminal:
            if status != "completed":
                raise RuntimeError(f"Final Strategy backtest ended with status={status}: {_get(job, 'error')}")
            if hasattr(job, "model_dump"):
                return job.model_dump(mode="json")
            if isinstance(job, dict):
                return job
            return dict(vars(job))
        time.sleep(max(1.0, float(poll_seconds)))


def _extract_metrics(job: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        job.get("metrics"),
        job.get("result", {}).get("metrics") if isinstance(job.get("result"), dict) else None,
        job.get("results", {}).get("metrics") if isinstance(job.get("results"), dict) else None,
    ]
    for value in candidates:
        if isinstance(value, dict) and value:
            return value
    return {}


def main() -> int:
    args = parse_args()
    if not _is_local_mongo(args.mongo_uri):
        raise RuntimeError("This research final-backtest script only accepts a local MongoDB connection.")

    output = Path(args.output_root) / f"asset_timing_strategy_{args.strategy_sequence}_{str(args.snapshot_end)[:10]}"
    frozen_path = Path(args.frozen_snapshot) if args.frozen_snapshot else output / "timing_validation_snapshot_frozen.json"
    if not frozen_path.exists():
        raise RuntimeError(f"Frozen timing snapshot not found: {frozen_path}")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))

    expected_hash = frozen.get("decision_snapshot_sha256")
    verify = dict(frozen)
    verify.pop("decision_snapshot_sha256", None)
    actual_hash = _sha256_json(verify)
    if expected_hash != actual_hash:
        raise RuntimeError(
            "Frozen timing snapshot hash mismatch. The qualification decision was changed after freezing."
        )
    if frozen.get("full_strategy_backtest_used_for_selection") is not False:
        raise RuntimeError("Frozen snapshot does not certify backtest-independent selection.")

    assets = [str(item).upper() for item in frozen.get("qualified_assets") or []]
    if len(assets) < 2:
        raise RuntimeError("Frozen qualified universe must contain at least two assets for Strategy rotation.")

    repository = MongoRepository(args.mongo_uri, args.database)
    db = repository.db
    profile = get_strategy_profile(db, sequence=int(args.strategy_sequence))
    if profile is None:
        raise RuntimeError(f"Strategy #{args.strategy_sequence} was not found.")
    config = _strategy_config(profile)

    request = _request_from_config(config, profile, assets, str(args.snapshot_end)[:10])
    request["research_timing_snapshot_hash"] = expected_hash
    request["research_timing_script_version"] = frozen.get("script_version")
    request_path = output / "final_strategy_backtest_request.json"
    request_path.write_text(json.dumps(request, indent=2, default=str), encoding="utf-8")

    print(f"Frozen universe: {len(assets)} assets")
    print("Starting ONE full Strategy backtest. This backtest does not alter the frozen qualification.")
    job = create_job(db, request)
    job_id = str(_get(job, "job_id") or _get(job, "id") or _get(job, "_id") or "")
    if not job_id:
        raise RuntimeError("Backtest service did not return a job id.")

    final_job = _wait_job(db, job_id, args.poll_seconds)
    (output / "final_strategy_backtest_job.json").write_text(json.dumps(final_job, indent=2, default=str), encoding="utf-8")
    metrics = _extract_metrics(final_job)
    (output / "final_strategy_backtest_metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    summary = {
        "script_version": SCRIPT_VERSION,
        "strategy_sequence": int(args.strategy_sequence),
        "snapshot_end": str(args.snapshot_end)[:10],
        "qualified_assets": assets,
        "qualified_asset_count": len(assets),
        "timing_decision_snapshot_sha256": expected_hash,
        "full_strategy_backtest_used_for_selection": False,
        "full_strategy_backtest_run_after_freeze": True,
        "job_id": job_id,
        "metrics": metrics,
    }
    (output / "final_experiment_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("Final Strategy backtest completed.")
    print(f"Summary: {output / 'final_experiment_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
