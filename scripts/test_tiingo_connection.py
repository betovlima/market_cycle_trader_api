from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from market_cycle_trader_api.core.environment import load_project_environment
from market_cycle_trader_api.infrastructure.market_data.tiingo import test_connection


def main() -> int:
    loaded = load_project_environment()
    if not loaded:
        raise RuntimeError(
            "No .env file was found. Copy .env.example to .env and set TIINGO_API_KEY."
        )

    result = test_connection("SPY")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
