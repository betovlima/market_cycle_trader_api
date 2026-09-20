from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from market_cycle_trader_api.core.environment import load_project_environment

load_project_environment()

from market_cycle_trader_api.infrastructure.persistence.mongo_repository import (
    JOBS_COLLECTION,
    create_client,
    get_database,
)
from market_cycle_trader_api.schemas.requests import BacktestExecutionRequest


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export only the immutable Backtest request/configuration to JSON. "
            "This helper does not export candles, corporate actions, predictions, "
            "or any other research data."
        )
    )
    parser.add_argument("--job-id", required=True)
    parser.add_argument(
        "--output",
        default="research/final_research_request.json",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
    )
    args = parser.parse_args()

    output = (ROOT / args.output).resolve()
    if output.exists() and not args.replace:
        raise FileExistsError(
            f"{output} already exists. Use --replace only if you intentionally "
            "want to replace the frozen research request."
        )

    client = create_client()
    try:
        db = get_database(client)
        job = db[JOBS_COLLECTION].find_one(
            {"id": str(args.job_id)},
            {"_id": 0, "id": 1, "request": 1},
        )
        if job is None or not isinstance(job.get("request"), dict):
            raise RuntimeError(
                f"Backtest job {args.job_id!r} with an immutable request was not found."
            )

        request = BacktestExecutionRequest.model_validate(job["request"])
        payload = {
            "schema_version": 1,
            "source_job_id": str(job.get("id") or args.job_id),
            "purpose": "final_research_configuration_only",
            "market_data_embedded": False,
            "request": request.model_dump(mode="json"),
        }

        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"[request-export] written={output}")
        print("[request-export] market_data_embedded=false")
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
