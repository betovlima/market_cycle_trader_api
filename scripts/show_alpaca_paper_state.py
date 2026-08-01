from __future__ import annotations

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
    PAPER_TRADE_PLANS_COLLECTION,
    create_client,
    get_database,
    get_paper_trading_settings,
    get_paper_trading_state,
)


def main() -> int:
    client = create_client()
    try:
        db = get_database(client)
        latest_plan = db[PAPER_TRADE_PLANS_COLLECTION].find_one(
            {}, {"_id": 0}, sort=[("created_at", -1)]
        )
        payload = {
            "settings": get_paper_trading_settings(db),
            "state": get_paper_trading_state(db),
            "latest_plan": latest_plan,
        }
        print(json.dumps(payload, indent=2, default=str))
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
