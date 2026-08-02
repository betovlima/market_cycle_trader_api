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
    mongo_database_name,
)
from market_cycle_trader_api.schemas.paper_trading import PaperTradingSettings
from market_cycle_trader_api.schemas.requests import BacktestRequest
from market_cycle_trader_api.services.parameter_bootstrap import (
    apply_parameter_documents,
    parameter_status,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--strategy-config", type=Path)
    parser.add_argument("--paper-config", type=Path)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--note", default="private parameter bootstrap")
    return parser.parse_args()


def load_json(path: Path | None) -> dict | None:
    if path is None:
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"{path} must contain one JSON object.")
    return value


def main() -> int:
    args = parse_args()
    client = create_client()
    try:
        db = get_database(client)
        ensure_database(db)
        if args.status:
            print(f"MongoDB database: {mongo_database_name()}")
            print(json.dumps(parameter_status(db), indent=2, default=str))
            return 0

        strategy_raw = load_json(args.strategy_config)
        paper_raw = load_json(args.paper_config)
        if strategy_raw is None and paper_raw is None:
            raise SystemExit(
                "Use --status or supply --strategy-config and/or --paper-config."
            )

        result = apply_parameter_documents(
            db,
            strategy_configuration=(
                BacktestRequest.model_validate(strategy_raw)
                if strategy_raw is not None
                else None
            ),
            paper_trading_configuration=(
                PaperTradingSettings.model_validate(paper_raw)
                if paper_raw is not None
                else None
            ),
            replace_existing=args.replace,
            note=args.note,
            source="bootstrap_parameters.py",
        )
        print(json.dumps(result, indent=2, default=str))
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
