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
from market_cycle_trader_api.services.paper_trading import initialize_paper_state  


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize the isolated $10k strategy sleeve in an empty Alpaca paper account."
    )
    parser.add_argument("--confirm-paper", action="store_true")
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.confirm_paper:
        raise SystemExit("Refusing to initialize. Re-run with --confirm-paper.")
    client = create_client()
    try:
        db = get_database(client)
        ensure_database(db)
        result = initialize_paper_state(db, replace=args.replace)
        print(json.dumps(result, indent=2, default=str))
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
