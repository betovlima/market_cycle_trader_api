from __future__ import annotations

import argparse
import json
import sys
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
    PAPER_TRADING_SETTINGS_COLLECTION,
    PAPER_TRADING_SETTINGS_HISTORY_COLLECTION,
    bson_value,
    create_client,
    ensure_database,
    get_database,
)
from market_cycle_trader_api.schemas.paper_trading import PaperTradingSettings  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and store administrative Alpaca paper-trading settings in MongoDB."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    validated = PaperTradingSettings.model_validate(raw).model_dump(mode="json")
    print(json.dumps(validated, indent=2))
    if args.dry_run:
        print("\nDRY RUN: paper-trading settings are valid; MongoDB was not changed.")
        return 0

    client = create_client()
    try:
        db = get_database(client)
        ensure_database(db)
        now = datetime.now(timezone.utc)
        previous = db[PAPER_TRADING_SETTINGS_COLLECTION].find_one({"_id": "default"})
        if previous:
            history = dict(previous)
            history.pop("_id", None)
            history.update({
                "captured_at": now,
                "source": "apply_paper_trading_config.py",
            })
            db[PAPER_TRADING_SETTINGS_HISTORY_COLLECTION].insert_one(history)
        db[PAPER_TRADING_SETTINGS_COLLECTION].replace_one(
            {"_id": "default"},
            {
                "_id": "default",
                **bson_value(validated),
                "schema_version": 1,
                "created_at": previous.get("created_at", now) if previous else now,
                "updated_at": now,
            },
            upsert=True,
        )
        print(f"\nApplied paper-trading settings to MongoDB database: {MONGO_DATABASE}")
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
