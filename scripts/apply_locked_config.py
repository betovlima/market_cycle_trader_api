from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from market_cycle_trader_api.core.environment import load_project_environment

load_project_environment()

from market_cycle_trader_api.infrastructure.persistence.mongo_repository import (
    SETTINGS_COLLECTION,
    create_client,
    ensure_database,
    get_database,
    mongo_database_name,
)
from market_cycle_trader_api.schemas.requests import (
    BacktestRequest,
    LOCKED_CONFIGURATION_FIELDS,
)
from market_cycle_trader_api.services.strategy_configuration import (
    replace_strategy_configuration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and apply a complete private strategy configuration to MongoDB."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--name", default=None)
    parser.add_argument("--note", default="manual promotion")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit("Configuration JSON must contain one object.")

    unknown = sorted(set(raw) - LOCKED_CONFIGURATION_FIELDS)
    missing = sorted(LOCKED_CONFIGURATION_FIELDS - set(raw))
    if unknown:
        raise SystemExit("Unsupported configuration keys: " + ", ".join(unknown))
    if missing:
        raise SystemExit("Full configuration required. Missing keys: " + ", ".join(missing))

    validated = BacktestRequest.model_validate(raw)
    canonical = json.dumps(
        validated.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    fingerprint = hashlib.sha256(canonical).hexdigest()
    if args.dry_run:
        print(f"Configuration is valid. SHA-256: {fingerprint}")
        print("MongoDB was not changed.")
        return 0

    client = create_client()
    try:
        db = get_database(client)
        ensure_database(db)
        existing = db[SETTINGS_COLLECTION].find_one({"_id": "default"})
        result = replace_strategy_configuration(
            db,
            validated,
            note=args.note,
            source="apply_locked_config.py",
            expected_revision=(
                int(existing.get("revision") or 1)
                if existing is not None
                else None
            ),
            allow_create=True,
        )
        if args.name:
            db[SETTINGS_COLLECTION].update_one(
                {"_id": "default"},
                {"$set": {"configuration_name": str(args.name)}},
            )
        print(f"Applied configuration to MongoDB database: {mongo_database_name()}")
        print(f"Revision: {result['metadata']['revision']}")
        print(f"SHA-256: {result['configuration_hash']}")
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
