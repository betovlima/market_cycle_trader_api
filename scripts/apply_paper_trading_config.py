from __future__ import annotations

import argparse
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
    create_client,
    ensure_database,
    get_database,
    mongo_database_name,
)
from market_cycle_trader_api.schemas.paper_trading import PaperTradingSettings
from market_cycle_trader_api.services.parameter_bootstrap import apply_parameter_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--note", default="manual paper configuration promotion")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    validated = PaperTradingSettings.model_validate(raw)
    if args.dry_run:
        print("Paper-trading configuration is valid.")
        print("MongoDB was not changed.")
        return 0

    client = create_client()
    try:
        db = get_database(client)
        ensure_database(db)
        result = apply_parameter_documents(
            db,
            strategy_configuration=None,
            paper_trading_configuration=validated,
            replace_existing=args.replace,
            note=args.note,
            source="apply_paper_trading_config.py",
        )
        print(f"Applied paper settings to MongoDB database: {mongo_database_name()}")
        print(json.dumps(result["results"], indent=2, default=str))
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
