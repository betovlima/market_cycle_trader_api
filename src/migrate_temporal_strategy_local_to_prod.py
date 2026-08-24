#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable

try:
    from pymongo import MongoClient, ReturnDocument
    from pymongo.database import Database
except ImportError:
    print("ERROR: pymongo is required. Install it with: python -m pip install pymongo", file=sys.stderr)
    raise SystemExit(2)

STRATEGY_PROFILES = "strategy_profiles"
STRATEGY_CONTROL = "strategy_control"
TEMPORAL_RUNS = "temporal_intelligence_runs"
TEMPORAL_OBSERVATIONS = "temporal_intelligence_observations"
TEMPORAL_ARTIFACTS = "temporal_intelligence_artifacts"
MARKET_SNAPSHOTS = "model_tuning_market_snapshots"
CONTROL_ID = "default"

DEFAULT_LOCAL_URI = "mongodb://localhost:27017/"
DEFAULT_TARGET_CAPITAL = 1_386_476.42
SUPPORTED_VARIANTS = {"winner_anchored_timing", ""}

LIFECYCLE_FIELDS_TO_CLEAR = {
    "promoted_at", "promoted_by", "promotion_note", "winner_api_version",
    "winner_front_version", "winner_promoted_from_strategy_id",
    "candidate_at", "candidate_by", "candidate_note", "candidate_revision",
    "candidate_backtest_id", "certified_backtest_cutoff", "source_candidate_backtest_id",
    "last_backtest_id", "last_backtest_status", "last_backtest_revision",
    "last_backtest_model_snapshot", "last_backtest_configuration_hash",
    "last_backtest_started_at", "last_backtest_finished_at",
    "tuning_source_run_id", "tuning_source_candidate_id", "tuning_result_metrics",
    "temporal_validation_status", "temporal_validation_id", "temporal_validation_at",
    "temporal_validation_by", "temporal_trader_eligible", "temporal_trader_block_reason",
    "superseded_at", "superseded_by_strategy_id", "supersession_note",
    "last_promoted_winner_strategy_id", "last_promoted_at",
    "auto_candidate_after_backtest",
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def nested_get(document: dict[str, Any] | None, *path: str) -> Any:
    current: Any = document
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def sanitize_uri(uri: str) -> str:
    if "@" not in uri or "://" not in uri:
        return uri
    scheme, rest = uri.split("://", 1)
    auth_host = rest.split("@", 1)
    if len(auth_host) != 2:
        return uri
    auth, host = auth_host
    username = auth.split(":", 1)[0] if ":" in auth else auth
    return f"{scheme}://{username}:***@{host}"


def make_client(uri: str, label: str) -> MongoClient:
    client = MongoClient(
        uri,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000,
        socketTimeoutMS=30000,
        retryWrites=True,
    )
    try:
        client.admin.command("ping")
    except Exception as exc:
        raise RuntimeError(f"Unable to connect to {label} MongoDB ({sanitize_uri(uri)}): {exc}") from exc
    return client


def detect_database(client: MongoClient, explicit: str | None, label: str) -> Database:
    if explicit:
        db = client[explicit]
        try:
            collections = db.list_collection_names()
        except Exception as exc:
            raise RuntimeError(f"Unable to inspect {label} database '{explicit}': {exc}") from exc
        if STRATEGY_PROFILES not in collections:
            raise RuntimeError(
                f"Database '{explicit}' does not contain '{STRATEGY_PROFILES}'. "
                f"Set the correct MCT_{label.upper()}_MONGO_DATABASE value."
            )
        return db

    try:
        names = [name for name in client.list_database_names() if name not in {"admin", "config", "local"}]
    except Exception as exc:
        raise RuntimeError(
            f"Mongo user cannot list {label} databases. Set MCT_{label.upper()}_MONGO_DATABASE explicitly."
        ) from exc

    candidates: list[str] = []
    for name in names:
        try:
            if STRATEGY_PROFILES in client[name].list_collection_names():
                candidates.append(name)
        except Exception:
            continue
    if len(candidates) != 1:
        raise RuntimeError(
            f"Could not safely auto-detect the {label} database. Candidates containing '{STRATEGY_PROFILES}': "
            f"{candidates or 'none'}. Set MCT_{label.upper()}_MONGO_DATABASE explicitly."
        )
    return client[candidates[0]]


def strategy_capital(profile: dict[str, Any], run: dict[str, Any] | None) -> float | None:
    values = [
        nested_get(profile, "temporal_policy_snapshot", "validation", "ending_capital"),
        nested_get(run, "result", "multi_horizon_metrics", "shadow_capital", "ending_capital"),
        nested_get(run, "result", "shadow_capital", "ending_capital"),
        nested_get(run, "result", "ending_capital"),
    ]
    for value in values:
        parsed = finite_float(value)
        if parsed is not None and parsed > 0:
            return parsed
    return None


def temporal_candidates(db: Database) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cursor = db[STRATEGY_PROFILES].find({"strategy_kind": "temporal_intelligence"})
    for profile in cursor:
        run_id = str(profile.get("source_temporal_run_id") or nested_get(profile, "temporal_policy_snapshot", "source_run_id") or "").strip()
        run = db[TEMPORAL_RUNS].find_one({"id": run_id}) if run_id else None
        rows.append({
            "profile": profile,
            "run": run,
            "capital": strategy_capital(profile, run),
            "run_id": run_id,
            "variant": str(profile.get("temporal_strategy_variant") or ""),
        })
    rows.sort(key=lambda item: (int(item["profile"].get("strategy_sequence") or 0), str(item["profile"].get("_id") or "")))
    return rows


def print_candidates(candidates: list[dict[str, Any]], target_capital: float) -> None:
    print("\nTemporal Strategies found in LOCAL MongoDB")
    print("=" * 118)
    print(f"{'#':>3} {'Seq':>4} {'Status':<12} {'Variant':<25} {'Policy capital':>16}  {'Strategy ID'}")
    print("-" * 118)
    for index, item in enumerate(candidates, start=1):
        profile = item["profile"]
        capital = item["capital"]
        cap_text = f"${capital:,.2f}" if capital is not None else "n/a"
        marker = "*" if capital is not None and abs(capital - target_capital) == min(
            [abs(other["capital"] - target_capital) for other in candidates if other["capital"] is not None] or [float("inf")]
        ) else " "
        print(
            f"{index:>2}{marker} {int(profile.get('strategy_sequence') or 0):>4} "
            f"{str(profile.get('status') or 'draft'):<12.12} {item['variant']:<25.25} {cap_text:>16}  {profile.get('_id')}"
        )
    print("=" * 118)
    print("* = closest stored Temporal validation capital to the target; selection is still confirmed before writing.\n")


def select_candidate(
    candidates: list[dict[str, Any]],
    strategy_id: str | None,
    target_capital: float,
    non_interactive: bool,
) -> dict[str, Any]:
    if not candidates:
        raise RuntimeError("No temporal_intelligence Strategy exists in the local strategy_profiles collection.")
    if strategy_id:
        matches = [item for item in candidates if str(item["profile"].get("_id")) == strategy_id]
        if len(matches) != 1:
            raise RuntimeError(f"Local temporal Strategy '{strategy_id}' was not found.")
        return matches[0]

    with_capital = [item for item in candidates if item["capital"] is not None]
    default = min(with_capital, key=lambda item: abs(item["capital"] - target_capital)) if with_capital else candidates[0]
    default_index = candidates.index(default) + 1

    if non_interactive or not sys.stdin.isatty():
        raise RuntimeError(
            "Multiple/local temporal Strategies require explicit selection in non-interactive mode. "
            "Run again with --strategy-id <local_strategy_id>."
        )

    answer = input(f"Select the LOCAL Strategy to migrate [default {default_index}]: ").strip()
    if not answer:
        return default
    try:
        index = int(answer)
    except ValueError as exc:
        raise RuntimeError("Selection must be a numeric row from the list.") from exc
    if index < 1 or index > len(candidates):
        raise RuntimeError("Selection is outside the candidate list.")
    return candidates[index - 1]


def source_configuration_hash(profile: dict[str, Any], run: dict[str, Any]) -> str:
    return str(
        nested_get(profile, "temporal_policy_snapshot", "source_strategy_configuration_hash")
        or run.get("strategy_configuration_hash")
        or profile.get("configuration_hash")
        or ""
    ).strip()


def map_production_base_strategy(prod_db: Database, local_profile: dict[str, Any], local_run: dict[str, Any]) -> dict[str, Any]:
    config_hash = source_configuration_hash(local_profile, local_run)
    if not config_hash:
        raise RuntimeError("The local Temporal Strategy does not contain a source configuration hash.")

    control = prod_db[STRATEGY_CONTROL].find_one({"_id": CONTROL_ID}) or {}
    winner_id = str(control.get("trader_winner_strategy_id") or "").strip()
    if winner_id:
        winner = prod_db[STRATEGY_PROFILES].find_one({"_id": winner_id})
        if winner and str(winner.get("configuration_hash") or "") == config_hash:
            return winner

    matches = list(prod_db[STRATEGY_PROFILES].find({"configuration_hash": config_hash}))
    if not matches:
        raise RuntimeError(
            "Production has no Strategy with the same source configuration_hash as the local Temporal Strategy. "
            "Migration stopped to avoid creating a dangling source_strategy_id."
        )
    if len(matches) == 1:
        return matches[0]

    matches.sort(
        key=lambda row: (
            0 if str(row.get("status") or "") == "winner" else 1,
            0 if str(row.get("catalog_status") or "") == "winner" else 1,
            -int(row.get("strategy_sequence") or 0),
        )
    )
    chosen = matches[0]
    print(
        f"WARNING: production has {len(matches)} Strategies with configuration_hash {config_hash[:12]}...; "
        f"using {chosen.get('_id')} (Strategy #{chosen.get('strategy_sequence')})."
    )
    return chosen


def count_dependencies(db: Database, run_id: str, snapshot_id: str) -> dict[str, int]:
    return {
        TEMPORAL_OBSERVATIONS: db[TEMPORAL_OBSERVATIONS].count_documents({"run_id": run_id}),
        TEMPORAL_ARTIFACTS: db[TEMPORAL_ARTIFACTS].count_documents({"run_id": run_id}),
        MARKET_SNAPSHOTS: db[MARKET_SNAPSHOTS].count_documents({"snapshot_id": snapshot_id}) if snapshot_id else 0,
    }


def exact_replace(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: exact_replace(item, mapping) for key, item in value.items()}
    if isinstance(value, list):
        return [exact_replace(item, mapping) for item in value]
    if isinstance(value, tuple):
        return [exact_replace(item, mapping) for item in value]
    if isinstance(value, str) and value in mapping:
        return mapping[value]
    return value


def sanitize_profile(
    local_profile: dict[str, Any],
    new_strategy_id: str,
    new_sequence: int,
    prod_base: dict[str, Any],
) -> dict[str, Any]:
    profile = copy.deepcopy(local_profile)
    old_name = str(profile.get("name") or "").strip()
    profile["_id"] = new_strategy_id
    profile["strategy_sequence"] = new_sequence
    profile["name"] = f"Strategy #{new_sequence}"
    if old_name and old_name != profile["name"]:
        profile["legacy_display_name"] = old_name
    profile["status"] = "draft"
    profile["catalog_status"] = "saved"
    profile["locked"] = False
    profile["source_strategy_id"] = str(prod_base.get("_id"))
    profile["source_strategy_revision"] = int(prod_base.get("revision") or 1)
    profile["created_at"] = utc_now()
    profile["updated_at"] = utc_now()
    profile["created_by"] = "migration-local-to-production"
    profile["updated_by"] = "migration-local-to-production"
    profile["auto_candidate_after_backtest"] = False

    for field in LIFECYCLE_FIELDS_TO_CLEAR:
        if field != "auto_candidate_after_backtest":
            profile.pop(field, None)

    policy = profile.get("temporal_policy_snapshot")
    if isinstance(policy, dict):
        policy = copy.deepcopy(policy)
        policy["source_strategy_id"] = str(prod_base.get("_id"))
        policy["source_strategy_revision"] = int(prod_base.get("revision") or 1)
        profile["temporal_policy_snapshot"] = policy

    return profile


def reserve_sequence(prod_db: Database) -> int:
    control = prod_db[STRATEGY_CONTROL].find_one_and_update(
        {"_id": CONTROL_ID},
        {"$inc": {"strategy_sequence": 1}, "$set": {"updated_at": utc_now()}},
        return_document=ReturnDocument.AFTER,
    )
    if not control:
        raise RuntimeError("Production strategy_control/default is unavailable; refusing to create catalog identity.")
    sequence = int(control.get("strategy_sequence") or 0)
    if sequence <= 0:
        raise RuntimeError("Production returned an invalid Strategy sequence.")
    return sequence


def insert_stream(source_cursor: Iterable[dict[str, Any]], target_collection: Any, *, strip_id: bool, batch_size: int = 500) -> int:
    batch: list[dict[str, Any]] = []
    inserted = 0
    for item in source_cursor:
        doc = copy.deepcopy(item)
        if strip_id:
            doc.pop("_id", None)
        batch.append(doc)
        if len(batch) >= batch_size:
            target_collection.insert_many(batch, ordered=True)
            inserted += len(batch)
            batch.clear()
    if batch:
        target_collection.insert_many(batch, ordered=True)
        inserted += len(batch)
    return inserted


def snapshot_ready(db: Database, snapshot_id: str) -> bool:
    if not snapshot_id:
        return False
    return db[MARKET_SNAPSHOTS].find_one({"snapshot_id": snapshot_id, "kind": "manifest", "ready": True}) is not None


def migrate(args: argparse.Namespace) -> int:
    local_uri = args.local_uri or os.getenv("MCT_LOCAL_MONGO_URI") or DEFAULT_LOCAL_URI
    prod_uri = args.prod_uri or os.getenv("MCT_PROD_MONGO_URI") or ""
    if not prod_uri:
        raise RuntimeError(
            "MCT_PROD_MONGO_URI is required. Set it in your shell; do not put the production token inside this script."
        )

    local_db_name = args.local_db or os.getenv("MCT_LOCAL_MONGO_DATABASE")
    prod_db_name = args.prod_db or os.getenv("MCT_PROD_MONGO_DATABASE")

    print(f"LOCAL Mongo: {sanitize_uri(local_uri)}")
    print(f"PROD  Mongo: {sanitize_uri(prod_uri)}")

    local_client = make_client(local_uri, "local")
    prod_client = make_client(prod_uri, "production")
    try:
        local_db = detect_database(local_client, local_db_name, "local")
        prod_db = detect_database(prod_client, prod_db_name, "prod")
        print(f"LOCAL database: {local_db.name}")
        print(f"PROD  database: {prod_db.name}")

        candidates = temporal_candidates(local_db)
        print_candidates(candidates, args.target_capital)
        selected = select_candidate(candidates, args.strategy_id, args.target_capital, args.non_interactive)
        local_profile = selected["profile"]
        local_run = selected["run"]
        run_id = selected["run_id"]
        variant = selected["variant"]

        if variant not in SUPPORTED_VARIANTS:
            raise RuntimeError(
                f"Selected variant '{variant}' is not supported by this one-off migration script. "
                "Stateful/MILP overlays require additional dependent research collections and are intentionally refused."
            )
        if not run_id or local_run is None:
            raise RuntimeError("Selected Temporal Strategy does not have its source Temporal Intelligence run in local MongoDB.")
        if str(local_run.get("status") or "") != "completed":
            raise RuntimeError(f"Source Temporal run {run_id} is not completed.")

        capital = selected["capital"]
        prod_base = map_production_base_strategy(prod_db, local_profile, local_run)
        control_before = prod_db[STRATEGY_CONTROL].find_one({"_id": CONTROL_ID}) or {}
        winner_before = str(control_before.get("trader_winner_strategy_id") or "")
        snapshot_id = str(
            local_run.get("market_data_snapshot_id")
            or nested_get(local_profile, "temporal_policy_snapshot", "market_data_snapshot_id")
            or ""
        ).strip().lower()
        local_counts = count_dependencies(local_db, run_id, snapshot_id)

        existing_strategy = prod_db[STRATEGY_PROFILES].find_one({
            "strategy_kind": "temporal_intelligence",
            "source_temporal_run_id": run_id,
            "temporal_strategy_variant": variant,
        })
        existing_run = prod_db[TEMPORAL_RUNS].find_one({"id": run_id})

        print("\nSelected migration")
        print("=" * 90)
        print(f"LOCAL Strategy ID       : {local_profile.get('_id')}")
        print(f"LOCAL Strategy sequence : {local_profile.get('strategy_sequence')}")
        print(f"Variant                 : {variant or 'winner_anchored_timing/legacy'}")
        print(f"Source Temporal run     : {run_id}")
        print(f"Stored ending capital   : {('$' + format(capital, ',.2f')) if capital else 'n/a'}")
        print(f"Configuration hash      : {str(local_profile.get('configuration_hash') or '')}")
        print(f"PROD base Strategy      : {prod_base.get('_id')} (Strategy #{prod_base.get('strategy_sequence')})")
        print(f"Market snapshot         : {snapshot_id or 'none'}")
        print(f"Observations            : {local_counts[TEMPORAL_OBSERVATIONS]}")
        print(f"Artifacts               : {local_counts[TEMPORAL_ARTIFACTS]}")
        print(f"Snapshot documents      : {local_counts[MARKET_SNAPSHOTS]}")
        print(f"Already in PROD Strategy: {existing_strategy.get('_id') if existing_strategy else 'no'}")
        print(f"Already in PROD run     : {'yes' if existing_run else 'no'}")
        print("Production Winner       : WILL NOT BE CHANGED")
        print("Imported catalog status : draft / saved")
        print("=" * 90)

        if existing_strategy:
            raise RuntimeError(
                f"A production Temporal Strategy already uses source run {run_id}: {existing_strategy.get('_id')}. "
                "No duplicate was created."
            )
        if existing_run:
            raise RuntimeError(
                f"Production already contains Temporal run {run_id}, but no matching Strategy. "
                "Migration stopped to avoid merging potentially different run data. Inspect this run first."
            )
        if local_counts[TEMPORAL_ARTIFACTS] <= 0:
            raise RuntimeError("Source run has no temporal_intelligence_artifacts; exact economic replay cannot be migrated safely.")
        if snapshot_id and local_counts[MARKET_SNAPSHOTS] <= 0 and not snapshot_ready(prod_db, snapshot_id):
            raise RuntimeError(
                f"Source run references market snapshot {snapshot_id}, but it is absent both locally and in production."
            )

        if not args.apply:
            print("\nDRY-RUN ONLY: no production documents were changed.")
            print("If this is the correct Strategy, run the same command with --apply.")
            return 0

        if not args.yes:
            if not sys.stdin.isatty():
                raise RuntimeError("--apply in non-interactive mode also requires --yes.")
            confirmation = input("\nType IMPORT to copy this Strategy and its immutable Temporal dependencies to production: ").strip()
            if confirmation != "IMPORT":
                print("Cancelled. No production documents were changed.")
                return 1

        # Generate the new Strategy id before copying the run so any exact local strategy id
        # inside run metadata can be remapped consistently.
        new_strategy_id = f"strategy-{uuid.uuid4().hex}"
        local_base_id = str(local_profile.get("source_strategy_id") or nested_get(local_profile, "temporal_policy_snapshot", "source_strategy_id") or local_run.get("strategy_profile_id") or "")
        id_mapping = {
            str(local_profile.get("_id")): new_strategy_id,
            local_base_id: str(prod_base.get("_id")),
        }
        id_mapping = {key: value for key, value in id_mapping.items() if key and value}

        copied_snapshot = False
        copied_run = False
        copied_observations = False
        copied_artifacts = False
        inserted_strategy = False

        try:
            # Snapshot is content-addressed. If production already has a ready identical snapshot,
            # reuse it instead of copying it again.
            if snapshot_id and not snapshot_ready(prod_db, snapshot_id):
                partial = prod_db[MARKET_SNAPSHOTS].count_documents({"snapshot_id": snapshot_id})
                if partial:
                    raise RuntimeError(
                        f"Production contains {partial} partial documents for snapshot {snapshot_id} without a ready manifest. "
                        "Refusing to overwrite them."
                    )
                snapshot_docs = list(local_db[MARKET_SNAPSHOTS].find({"snapshot_id": snapshot_id}).sort("kind", 1))
                if snapshot_docs:
                    prod_db[MARKET_SNAPSHOTS].insert_many(copy.deepcopy(snapshot_docs), ordered=True)
                    copied_snapshot = True

            run_copy = exact_replace(copy.deepcopy(local_run), id_mapping)
            run_copy.pop("_id", None)
            run_copy["strategy_profile_id"] = str(prod_base.get("_id"))
            run_copy["strategy_profile_revision"] = int(prod_base.get("revision") or 1)
            run_copy["materialized_strategy_id"] = new_strategy_id
            run_copy["materialized_strategy_name"] = None
            run_copy["materialized_strategy_at"] = utc_now()
            prod_db[TEMPORAL_RUNS].insert_one(run_copy)
            copied_run = True

            obs_cursor = local_db[TEMPORAL_OBSERVATIONS].find({"run_id": run_id}).sort("timestamp", 1)
            inserted_obs = insert_stream(obs_cursor, prod_db[TEMPORAL_OBSERVATIONS], strip_id=True)
            copied_observations = inserted_obs > 0

            artifact_cursor = local_db[TEMPORAL_ARTIFACTS].find({"run_id": run_id}).sort([("kind", 1), ("sequence", 1)])
            inserted_artifacts = insert_stream(artifact_cursor, prod_db[TEMPORAL_ARTIFACTS], strip_id=True)
            copied_artifacts = inserted_artifacts > 0

            new_sequence = reserve_sequence(prod_db)
            profile_copy = sanitize_profile(local_profile, new_strategy_id, new_sequence, prod_base)
            profile_copy = exact_replace(profile_copy, id_mapping)
            profile_copy["source_temporal_run_id"] = run_id
            if isinstance(profile_copy.get("temporal_policy_snapshot"), dict):
                profile_copy["temporal_policy_snapshot"]["source_run_id"] = run_id
            prod_db[STRATEGY_PROFILES].insert_one(profile_copy)
            inserted_strategy = True

            prod_db[TEMPORAL_RUNS].update_one(
                {"id": run_id},
                {"$set": {
                    "materialized_strategy_id": new_strategy_id,
                    "materialized_strategy_name": profile_copy["name"],
                    "materialized_strategy_at": utc_now(),
                    "updated_at": utc_now(),
                }},
            )

            # Verification
            prod_profile = prod_db[STRATEGY_PROFILES].find_one({"_id": new_strategy_id})
            prod_run = prod_db[TEMPORAL_RUNS].find_one({"id": run_id})
            prod_counts = count_dependencies(prod_db, run_id, snapshot_id)
            if not prod_profile or str(prod_profile.get("status")) != "draft":
                raise RuntimeError("Verification failed: imported Strategy is missing or is not draft.")
            if str(prod_profile.get("strategy_kind")) != "temporal_intelligence":
                raise RuntimeError("Verification failed: imported Strategy lost strategy_kind=temporal_intelligence.")
            if not prod_run or str(prod_run.get("status")) != "completed":
                raise RuntimeError("Verification failed: source Temporal run is not available as completed in production.")
            if prod_counts[TEMPORAL_OBSERVATIONS] != local_counts[TEMPORAL_OBSERVATIONS]:
                raise RuntimeError(
                    f"Verification failed: observations count local={local_counts[TEMPORAL_OBSERVATIONS]} "
                    f"prod={prod_counts[TEMPORAL_OBSERVATIONS]}."
                )
            if prod_counts[TEMPORAL_ARTIFACTS] != local_counts[TEMPORAL_ARTIFACTS]:
                raise RuntimeError(
                    f"Verification failed: artifacts count local={local_counts[TEMPORAL_ARTIFACTS]} "
                    f"prod={prod_counts[TEMPORAL_ARTIFACTS]}."
                )
            if snapshot_id and not snapshot_ready(prod_db, snapshot_id):
                raise RuntimeError("Verification failed: referenced frozen market snapshot is not ready in production.")

            control_after = prod_db[STRATEGY_CONTROL].find_one({"_id": CONTROL_ID}) or {}
            winner_after = str(control_after.get("trader_winner_strategy_id") or "")
            if winner_after != winner_before:
                raise RuntimeError("Verification failed: production Winner identity changed unexpectedly.")

            print("\nIMPORT COMPLETED")
            print("=" * 90)
            print(f"New production Strategy ID : {new_strategy_id}")
            print(f"New catalog sequence       : Strategy #{new_sequence}")
            print(f"Status                     : draft / saved")
            print(f"Temporal source run        : {run_id}")
            print(f"Observations copied        : {prod_counts[TEMPORAL_OBSERVATIONS]}")
            print(f"Artifacts copied           : {prod_counts[TEMPORAL_ARTIFACTS]}")
            print(f"Frozen snapshot ready      : {'yes' if (not snapshot_id or snapshot_ready(prod_db, snapshot_id)) else 'no'}")
            print("Production Winner           : unchanged")
            print("=" * 90)
            print("Next step: open Strategy Catalog in production and run a validation/backtest before any promotion.")
            return 0

        except Exception:
            # Roll back only documents introduced by this invocation. The sequence counter may
            # retain a harmless gap if the failure occurred after identity reservation.
            if inserted_strategy:
                prod_db[STRATEGY_PROFILES].delete_one({"_id": new_strategy_id})
            if copied_artifacts:
                prod_db[TEMPORAL_ARTIFACTS].delete_many({"run_id": run_id})
            if copied_observations:
                prod_db[TEMPORAL_OBSERVATIONS].delete_many({"run_id": run_id})
            if copied_run:
                prod_db[TEMPORAL_RUNS].delete_one({"id": run_id})
            if copied_snapshot:
                prod_db[MARKET_SNAPSHOTS].delete_many({"snapshot_id": snapshot_id})
            raise
    finally:
        local_client.close()
        prod_client.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Safely migrate one materialized winner-anchored Temporal Intelligence Strategy "
            "from local MongoDB to production together with its immutable Temporal run, "
            "observations, artifacts and frozen market snapshot. Production Winner is not changed."
        )
    )
    parser.add_argument("--local-uri", help="Local MongoDB URI. Default: MCT_LOCAL_MONGO_URI or mongodb://localhost:27017/")
    parser.add_argument("--prod-uri", help="Production MongoDB URI. Prefer MCT_PROD_MONGO_URI instead of this argument.")
    parser.add_argument("--local-db", help="Local database name. Default: MCT_LOCAL_MONGO_DATABASE or auto-detect.")
    parser.add_argument("--prod-db", help="Production database name. Default: MCT_PROD_MONGO_DATABASE or auto-detect.")
    parser.add_argument("--strategy-id", help="Exact LOCAL temporal Strategy _id to migrate.")
    parser.add_argument("--target-capital", type=float, default=DEFAULT_TARGET_CAPITAL, help="Capital hint used only to preselect a candidate in interactive mode.")
    parser.add_argument("--apply", action="store_true", help="Actually write to production. Without this flag the script is dry-run only.")
    parser.add_argument("--yes", action="store_true", help="Skip the IMPORT confirmation prompt. Intended for controlled non-interactive execution.")
    parser.add_argument("--non-interactive", action="store_true", help="Require --strategy-id and never prompt for Strategy selection.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return migrate(args)
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
