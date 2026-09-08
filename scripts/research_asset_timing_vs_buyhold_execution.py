from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from pymongo import MongoClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import research_asset_signature_leave_one_out as common  # noqa: E402
import research_asset_timing_vs_buyhold as timing  # noqa: E402
from market_cycle_trader_api.schemas.requests import (  # noqa: E402
    BacktestExecutionRequest,
    BacktestRequest,
)
from market_cycle_trader_api.services.model_research import (  # noqa: E402
    apply_execution_profile,
    model_execution_snapshot,
)

SCRIPT_VERSION = "asset-timing-vs-buyhold-v1.2"


class _ExecutionRequestFactory:
    snapshot_end: str = ""
    model_family: str = ""
    model_settings: dict[str, Any] = {}

    @classmethod
    def model_validate(cls, configuration: Any) -> BacktestExecutionRequest:
        base = BacktestRequest.model_validate(configuration)
        if not cls.snapshot_end:
            raise RuntimeError("Asset timing execution requires a frozen --snapshot-end.")
        if cls.model_family != "lightgbm_utility":
            raise RuntimeError(
                "Asset timing research requires the Strategy immutable LightGBM Utility snapshot; "
                f"received {cls.model_family or 'missing'}."
            )

        locked = base.model_copy(
            update={
                "end_date": cls.snapshot_end,
            }
        )
        locked = apply_execution_profile(locked, cls.model_family, cls.model_settings)
        return BacktestExecutionRequest.model_validate(
            {
                **locked.model_dump(mode="python"),
                "analysis_start_date": locked.start_date,
                "analysis_end_date": cls.snapshot_end,
                "calendar_anchor_assets": list(locked.assets),
                "research_reference_assets": list(locked.assets),
                "research_candidate_assets": [],
                "research_model_family": cls.model_family,
                "research_model_settings": cls.model_settings,
                "research_market_data_mode": "database_only",
            }
        )


def _bootstrap_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--strategy-sequence", type=int, default=10)
    parser.add_argument("--strategy-id", default=None)
    parser.add_argument("--snapshot-end", required=True)
    parser.add_argument("--mongo-uri", default=None)
    parser.add_argument("--database", default=None)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--allow-remote-mongo", action="store_true")
    args, _ = parser.parse_known_args()
    return args


def _immutable_model_snapshot(strategy: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
    stored = strategy.get("research_model_snapshot")
    if not isinstance(stored, dict):
        raise RuntimeError(
            "Selected Strategy has no immutable research_model_snapshot. "
            "The experiment refuses to fall back to mutable global model settings."
        )

    family = str(stored.get("family") or "").strip().lower()
    if family != "lightgbm_utility":
        raise RuntimeError(
            "Selected Strategy is not bound to LightGBM Utility: "
            f"family={family or 'missing'}."
        )
    settings = (
        dict(stored.get("settings_snapshot") or {})
        if isinstance(stored.get("settings_snapshot"), dict)
        else {}
    )
    resolved = model_execution_snapshot(family, settings)
    stored_hash = str(stored.get("settings_hash") or "").strip().lower()
    resolved_hash = str(resolved.get("settings_hash") or "").strip().lower()
    if stored_hash and stored_hash != resolved_hash:
        raise RuntimeError(
            "Strategy immutable model snapshot hash mismatch: "
            f"stored={stored_hash}, resolved={resolved_hash}."
        )
    return family, settings, resolved_hash


def main() -> int:
    args = _bootstrap_args()
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

    client = MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=3_000,
        connectTimeoutMS=3_000,
        maxPoolSize=4,
        retryWrites=False,
    )
    try:
        client.admin.command("ping")
        db = client[database_name]
        strategy = common._strategy_document(db, args.strategy_sequence, args.strategy_id)
        family, settings, settings_hash = _immutable_model_snapshot(strategy)
    finally:
        client.close()

    _ExecutionRequestFactory.snapshot_end = str(args.snapshot_end)
    _ExecutionRequestFactory.model_family = family
    _ExecutionRequestFactory.model_settings = settings

    timing.BacktestRequest = _ExecutionRequestFactory
    timing.SCRIPT_VERSION = SCRIPT_VERSION
    print(
        "Asset Timing research model binding: "
        f"family={family}, settings_hash={settings_hash}, source=Strategy immutable snapshot.",
        flush=True,
    )
    return timing.main()


if __name__ == "__main__":
    raise SystemExit(main())
