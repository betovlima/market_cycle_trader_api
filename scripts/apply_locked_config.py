from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from market_cycle_trader_api.core.environment import load_project_environment  # noqa: E402

load_project_environment()

from market_cycle_trader_api.infrastructure.persistence.mongo_repository import (  # noqa: E402
    MONGO_DATABASE,
    SETTINGS_COLLECTION,
    SETTINGS_HISTORY_COLLECTION,
    SETTINGS_SCHEMA_VERSION,
    bson_value,
    create_client,
    ensure_database,
    get_database,
)
from market_cycle_trader_api.schemas.requests import (  # noqa: E402
    BacktestRequest,
    LOCKED_CONFIGURATION_FIELDS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and apply a complete locked strategy configuration directly to MongoDB."
    )
    parser.add_argument("config", type=Path, help="Path to the full JSON configuration file.")
    parser.add_argument("--name", default=None, help="Optional configuration name stored as document metadata.")
    parser.add_argument("--note", default="manual promotion", help="Audit note stored with the history record.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the effective configuration without writing MongoDB.",
    )
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
        raise SystemExit("Full locked configuration required. Missing keys: " + ", ".join(missing))

    validated = BacktestRequest.model_validate(raw).model_dump(mode="json")
    effective = {key: bson_value(value) for key, value in validated.items()}

    print(json.dumps(validated, indent=2, default=str))
    if args.dry_run:
        print("\nDRY RUN: configuration is valid; MongoDB was not changed.")
        return 0

    client = create_client()
    try:
        db = get_database(client)
        ensure_database(db)
        collection = db[SETTINGS_COLLECTION]
        previous = collection.find_one({"_id": "default"})
        now = datetime.now(timezone.utc)
        if previous:
            history = deepcopy(previous)
            history.pop("_id", None)
            history.update(
                {
                    "captured_at": now,
                    "note": args.note,
                    "source": "apply_locked_config.py",
                }
            )
            db[SETTINGS_HISTORY_COLLECTION].insert_one(history)

        created_at = previous.get("created_at", now) if previous else now
        document = {
            "_id": "default",
            **effective,
            "created_at": created_at,
            "updated_at": now,
            "schema_version": SETTINGS_SCHEMA_VERSION,
            "configuration_name": args.name or args.config.stem,
            "configuration_note": args.note,
        }
        collection.replace_one({"_id": "default"}, document, upsert=True)
        print(f"\nApplied locked configuration to MongoDB database: {MONGO_DATABASE}")
        print(f"History snapshot collection: {SETTINGS_HISTORY_COLLECTION}")
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
