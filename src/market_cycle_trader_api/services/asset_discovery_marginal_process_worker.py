from __future__ import annotations

import sys
import traceback

from ..core.runtime import close_mongo, database, initialize_mongo
from . import asset_discovery as discovery


def _record_worker_failure(run_id: str, error: Exception) -> None:
    try:
        db = database()
    except Exception:
        return
    try:
        now = discovery.utc_now()
        db[discovery.COLLECTION].update_one(
            {"_id": discovery.CURRENT_ID, "run_id": run_id},
            {"$set": {
                "marginal_replay.worker_child_error_type": error.__class__.__name__,
                "marginal_replay.worker_child_error": str(error)[:1000],
                "updated_at": now,
            }},
        )
    except Exception:
        return


def main() -> int:
    if len(sys.argv) < 2:
        return 2
    run_id = str(sys.argv[1] or "").strip()
    if not run_id:
        return 2

    try:
        initialize_mongo(role="worker")
        discovery._run_existing_marginal_worker(database(), run_id)
        return 0
    except Exception as exc:
        _record_worker_failure(run_id, exc)
        traceback.print_exc(file=sys.stderr)
        return 1
    finally:
        close_mongo()


if __name__ == "__main__":
    raise SystemExit(main())
