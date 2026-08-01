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
)
from market_cycle_trader_api.services.paper_trading import prepare_next_paper_plan  


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a next-session paper order plan."
    )
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = create_client()
    try:
        db = get_database(client)
        ensure_database(db)
        result = prepare_next_paper_plan(db, replace=args.replace)
        print(json.dumps(result, indent=2, default=str))
        print("\nNo order was submitted. Execute this plan only during its market session.")
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
