from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from market_cycle_trader_api.core.environment import load_project_environment  # noqa: E402

load_project_environment()

from market_cycle_trader_api.infrastructure.persistence.mongo_repository import (  # noqa: E402
    ALPACA_INTEGRATION_ID,
    INTEGRATIONS_COLLECTION,
    create_client,
    get_database,
)


def env(name: str) -> str:
    return str(os.getenv(name) or "").strip()


def main() -> int:
    api_key = env("ALPACA_API_KEY_ID") or env("APCA_API_KEY_ID")
    secret = env("ALPACA_SECRET_KEY") or env("ALPACA_API_SECRET_KEY") or env("APCA_API_SECRET_KEY")
    if not api_key or not secret:
        raise SystemExit(
            "Refusing to purge legacy MongoDB Alpaca credentials until ALPACA_API_KEY_ID "
            "and ALPACA_SECRET_KEY are configured in the environment."
        )

    client = create_client()
    try:
        result = get_database(client)[INTEGRATIONS_COLLECTION].delete_one({"_id": ALPACA_INTEGRATION_ID})
        print(f"Removed {result.deleted_count} legacy Alpaca credential document(s) from MongoDB.")
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
