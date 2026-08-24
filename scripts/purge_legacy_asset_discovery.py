from __future__ import annotations

import argparse

from market_cycle_trader_api.core.environment import load_project_environment


LEGACY_COLLECTIONS = (
    "asset_discovery_settings",
    "asset_discovery_settings_history",
    "asset_discovery_candidates",
    "asset_discovery_runs",
    "asset_discovery_state",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    if not args.confirm:
        raise SystemExit("Use --confirm to drop the retired Asset Discovery collections.")

    load_project_environment()
    from market_cycle_trader_api.infrastructure.persistence.mongo_repository import create_client, get_database

    client = create_client()
    try:
        db = get_database(client)
        existing = set(db.list_collection_names())
        for collection_name in LEGACY_COLLECTIONS:
            if collection_name not in existing:
                print(f"{collection_name}: already absent")
                continue
            db.drop_collection(collection_name)
            print(f"{collection_name}: dropped")
    finally:
        client.close()


if __name__ == "__main__":
    main()
