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

from market_cycle_trader_api.infrastructure.market_data.alpaca import (  # noqa: E402
    test_connection as verify_alpaca_connection,
)
from market_cycle_trader_api.infrastructure.persistence.mongo_repository import (  # noqa: E402
    create_client,
    get_alpaca_credentials,
    get_database,
    get_settings,
)
from market_cycle_trader_api.schemas.requests import BacktestRequest  # noqa: E402


def main() -> int:
    credentials = get_alpaca_credentials()
    client = create_client()
    try:
        config = BacktestRequest.model_validate(get_settings(get_database(client)))
    finally:
        client.close()

    result = verify_alpaca_connection(
        api_key_id=credentials["api_key_id"],
        secret_key=credentials["secret_key"],
        feed=config.alpaca_feed,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
