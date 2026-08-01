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

from market_cycle_trader_api.infrastructure.trading.alpaca_paper import (  # noqa: E402
    account_snapshot,
    clock_snapshot,
    create_unverified_paper_trading_client,
)


def main() -> int:
    client = create_unverified_paper_trading_client()
    account = account_snapshot(client)
    number = account.get("account_number") or ""
    account["account_number"] = (
        f"***{number[-4:]}" if len(number) >= 4 else "***"
    )
    print(json.dumps({"account": account, "clock": clock_snapshot(client)}, indent=2, default=str))
    print("\nSet this exact value in .env / Railway:")
    print(f"ALPACA_PAPER_ACCOUNT_ID={account['id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
