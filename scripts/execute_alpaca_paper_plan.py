from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from market_cycle_trader_api.core.environment import load_project_environment  # noqa: E402

load_project_environment()

from market_cycle_trader_api.infrastructure.persistence.mongo_repository import (  # noqa: E402
    create_client,
    ensure_database,
    get_database,
)
from market_cycle_trader_api.services.paper_trading import execute_prepared_paper_plan  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute a prepared XGBoost plan against Alpaca paper trading only."
    )
    parser.add_argument("--plan-id", default=None)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.execute:
        raise SystemExit("No paper order was submitted. Re-run with --execute.")
    client = create_client()
    try:
        db = get_database(client)
        ensure_database(db)
        result = execute_prepared_paper_plan(db, plan_id=args.plan_id)
        print(json.dumps(result, indent=2, default=str))
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
