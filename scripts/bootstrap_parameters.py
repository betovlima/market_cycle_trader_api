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
    get_database,
)
from market_cycle_trader_api.services.parameter_bootstrap import (  # noqa: E402
    bootstrap_missing_parameterizations,
    parameterization_status,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Insert missing parameter documents, repair invalid strategy schemas, "
            "and preserve valid API-managed strategy parameters."
        )
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show existence and validation status without writing MongoDB.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = create_client()
    try:
        db = get_database(client)
        if args.status:
            result = {
                "mode": "status_only",
                "items": parameterization_status(db),
            }
        else:
            result = bootstrap_missing_parameterizations(
                db,
                source="bootstrap_parameters.py",
            )
        print(json.dumps(result, indent=2, default=str))
        if not args.status:
            invalid = [item for item in result.get("results", []) if not item.get("valid")]
            if invalid:
                print(
                    "Parameter bootstrap failed because one or more non-strategy documents are invalid.",
                    file=sys.stderr,
                )
                return 1
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
