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
    get_database,
    get_settings,
)
from market_cycle_trader_api.schemas.requests import BacktestRequest              


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the active locked MongoDB configuration to JSON.")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    client = create_client()
    try:
        raw = get_settings(get_database(client))
        payload = BacktestRequest.model_validate(raw).model_dump(mode="json")
    finally:
        client.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Exported: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
