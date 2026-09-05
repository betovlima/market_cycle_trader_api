from __future__ import annotations

import sys

from ..core.runtime import close_mongo, database, initialize_mongo
from . import asset_discovery as discovery


def main() -> int:
    if len(sys.argv) < 2:
        return 2
    run_id = str(sys.argv[1] or "").strip()
    if not run_id:
        return 2

    initialize_mongo(role="worker")
    try:
        discovery._run_existing_marginal_worker(database(), run_id)
        return 0
    finally:
        close_mongo()


if __name__ == "__main__":
    raise SystemExit(main())
