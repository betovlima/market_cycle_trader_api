from __future__ import annotations

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
    get_database,
    mongo_connection_status,
)


def main() -> int:
    status = mongo_connection_status()
    print(f"MONGO_URL configured: {bool(status['configured'])}")
    print(f"MongoDB database: {status['database']}")
    client = create_client()
    try:
        get_database(client)
        print("MongoDB ping: OK")
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
