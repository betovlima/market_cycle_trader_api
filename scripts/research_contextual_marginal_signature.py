from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import traceback
import uuid
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymongo import MongoClient

import research_contextual_marginal_signature_v1144 as frozen
import research_contextual_marginal_signature_v116 as paired

base = frozen.base

SCRIPT_VERSION = "contextual-marginal-signature-v1.0.16.1"
EXPERIMENT_NAME = "contextual_marginal_signature_paired_action_advantage"
HISTORY_START = "2016-01-01"
SNAPSHOT_END = "2026-09-04"
HORIZON_SESSIONS = 40
EXPORT_FOLDER_NAME = "contextual_marginal_signature"
EXPORT_ZIP_NAME = "contextual_marginal_signature.zip"

RUNS_COLLECTION = "research_contextual_signature_runs"
OBSERVATIONS_COLLECTION = "research_contextual_signature_observations"
TRACE_RUNS_COLLECTION = "research_contextual_signature_trace_runs"
TRACE_ROWS_COLLECTION = "research_contextual_signature_trace_rows"

ORIGINAL25 = (
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AMD", "JPM", "SPY",
    "AVGO", "NFLX", "CRM", "ORCL", "COST", "LLY", "XOM", "CAT", "WMT", "V",
    "HD", "ADC", "ADEA", "ADI", "ADM",
)
UNIVERSES = (
    {"name": "Original25", "assets": ORIGINAL25},
    {"name": "Original24_MinusADM", "assets": tuple(x for x in ORIGINAL25 if x != "ADM")},
)
CANDIDATES = ("XSD", "MKSI", "GKOS", "CLMT", "CORT", "APD", "CCK")
DECISION_DATES = (
    "2024-01-02", "2024-07-01", "2025-01-02",
    "2025-07-01", "2026-01-02", "2026-07-01",
)
TOLERANCE = 1e-12


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Single-entry contextual marginal signature research runner. "
            "MongoDB is the persistent source of truth; filesystem output is export-only."
        )
    )
    parser.add_argument("--strategy-sequence", type=int, default=10)
    parser.add_argument("--strategy-id", default=None)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--mongo-uri", default=None)
    parser.add_argument("--database", default=None)
    return parser


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mongo_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _mongo_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_mongo_value(v) for v in value]
    if isinstance(value, pd.Timestamp):
        stamp = value
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("UTC")
        else:
            stamp = stamp.tz_convert("UTC")
        return stamp.to_pydatetime()
    if isinstance(value, np.datetime64):
        return _mongo_value(pd.Timestamp(value))
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _runtime_mongo_settings(args: argparse.Namespace, mongo_repository: Any) -> tuple[str, str]:
    mongo_uri = str(
        args.mongo_uri
        or os.getenv("MONGO_URL")
        or os.getenv("MONGO_URI")
        or ""
    ).strip()
    database_name = str(args.database or os.getenv("MONGO_DATABASE") or "").strip()
    if not mongo_uri:
        raise RuntimeError("MONGO_URL/MONGO_URI is required in .env or --mongo-uri.")
    if not database_name:
        raise RuntimeError("MONGO_DATABASE is required in .env or --database.")
    base._assert_local_mongo(mongo_uri, allow_remote=False)
    mongo_repository.MONGO_URI = mongo_uri
    mongo_repository.MONGO_DATABASE = database_name
    return mongo_uri, database_name


def _campaign_symbols() -> list[str]:
    values = [symbol for universe in UNIVERSES for symbol in universe["assets"]]
    values.extend(CANDIDATES)
    return base._normalize_symbols(values)


def _load_market_frames(config: Any, market_data: Any) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    symbols = _campaign_symbols()
    history_start = base._normalize_date(HISTORY_START)
    snapshot_end = base._normalize_date(SNAPSHOT_END)
    expected = base._expected_sessions(history_start, snapshot_end)
    expected = pd.DatetimeIndex(pd.to_datetime(expected, utc=True)).normalize()

    frames: dict[str, pd.DataFrame] = {}
    provenance: list[dict[str, Any]] = []
    for index, symbol in enumerate(symbols, start=1):
        base._log(f"[market {index}/{len(symbols)}] MongoDB Alpaca bars: {symbol}")
        frame = market_data.load_market_bars(symbol, config)
        frame = market_data.validate_and_clean_bars(frame, config)
        observed = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True)).normalize().unique()
        missing = expected.difference(observed)
        if len(missing):
            sample = ", ".join(ts.date().isoformat() for ts in missing[:5])
            raise RuntimeError(
                f"MongoDB market history is incomplete for {symbol}: "
                f"missing={len(missing)} sample={sample}"
            )
        frames[symbol] = frame
        detail = dict(frame.attrs.get("market_data_provenance") or {})
        provenance.append(
            {
                "symbol": symbol,
                "rows": int(len(frame)),
                "first": pd.Timestamp(frame.index.min()).isoformat(),
                "last": pd.Timestamp(frame.index.max()).isoformat(),
                "source": detail.get("source") or "mongo_cache",
                "access": detail.get("research_access_path") or "mongodb_only",
            }
        )
    return frames, provenance


def _ending_capital(metrics: dict[str, Any], label: str) -> float:
    value = base.discovery._finite_number(metrics.get("ending_capital"))
    if value is None or value <= 0:
        raise RuntimeError(f"Invalid ending capital for {label}: {value}")
    return float(value)


def _first_selected_asset(captured: list[Any]) -> tuple[str, int]:
    for result in captured:
        predictions = getattr(result, "predictions", None)
        if isinstance(predictions, pd.DataFrame) and not predictions.empty:
            ordered = predictions.sort_index()
            selected = str(ordered.iloc[0].get("selected_asset") or "CASH").strip().upper()
            return selected, int(len(ordered))
    raise RuntimeError("Captured replay has no prediction rows.")


def _persist_capture(
    *,
    run_id: str,
    decision_date: str,
    universe_name: str,
    candidate: str | None,
    arm: str,
    captured: list[Any],
    trace_runs: Any,
    trace_rows: Any,
) -> None:
    for result_index, result in enumerate(captured, start=1):
        trace_id = f"{run_id}:{decision_date}:{universe_name}:{candidate or 'BASELINE'}:{arm}:{result_index}"
        metrics = dict(getattr(result, "metrics", {}) or {})
        trace_runs.insert_one(
            {
                "_id": trace_id,
                "run_id": run_id,
                "decision_date": decision_date,
                "universe_name": universe_name,
                "candidate": candidate,
                "arm": arm,
                "result_index": result_index,
                "backend": str(getattr(result, "backend", "")),
                "metrics": _mongo_value(metrics),
                "summary": str(getattr(result, "summary", "") or ""),
            }
        )

        rows: list[dict[str, Any]] = []
        predictions = getattr(result, "predictions", None)
        if isinstance(predictions, pd.DataFrame) and not predictions.empty:
            frame = predictions.reset_index()
            for row_index, record in enumerate(frame.to_dict(orient="records")):
                rows.append(
                    {
                        "run_id": run_id,
                        "trace_id": trace_id,
                        "row_type": "prediction",
                        "row_index": row_index,
                        "payload": _mongo_value(record),
                    }
                )
        trades = getattr(result, "trades", None)
        if isinstance(trades, pd.DataFrame) and not trades.empty:
            for row_index, record in enumerate(trades.to_dict(orient="records")):
                rows.append(
                    {
                        "run_id": run_id,
                        "trace_id": trace_id,
                        "row_type": "trade",
                        "row_index": row_index,
                        "payload": _mongo_value(record),
                    }
                )
        if rows:
            trace_rows.insert_many(rows, ordered=True)


def _run_replay(
    *,
    db: Any,
    config: Any,
    strategy_id: str,
    reference_assets: list[str],
    candidate: str | None,
    frames: dict[str, pd.DataFrame],
    decision: pd.Timestamp,
    horizon_end: pd.Timestamp,
    forced: bool,
) -> tuple[dict[str, Any], Any, list[Any]]:
    candidate_assets = [candidate] if candidate else []
    assets = [*reference_assets, *candidate_assets]
    request = frozen._window_request_frozen(
        db=db,
        config=config,
        strategy_id=strategy_id,
        assets=assets,
        reference_assets=reference_assets,
        candidate_assets=candidate_assets,
        decision_session=decision,
        horizon_end=horizon_end,
    )
    original_simulator = paired.research_challengers._simulate_exact
    try:
        paired.research_challengers._simulate_exact = (
            paired._forced_first_action_simulator if forced else paired._ORIGINAL_SIMULATE_EXACT
        )
        return frozen._run_with_frozen_margin(frames, request)
    finally:
        paired.research_challengers._simulate_exact = original_simulator


def _readiness(dataset: pd.DataFrame) -> dict[str, Any]:
    nonzero = dataset.loc[dataset["action_advantage_log"].abs() > TOLERANCE].copy()
    positive = int((dataset["action_advantage_log"] > TOLERANCE).sum())
    negative = int((dataset["action_advantage_log"] < -TOLERANCE).sum())
    checks = {
        "nonzero_rows": int(len(nonzero)) >= 12,
        "candidate_diversity": int(nonzero["candidate"].nunique()) >= 3,
        "date_diversity": int(nonzero["decision_date"].nunique()) >= 3,
        "both_signs": positive > 0 and negative > 0,
        "validation_2026": bool(
            pd.to_datetime(nonzero["decision_date"], errors="coerce").dt.year.eq(2026).any()
        ),
    }
    return {
        **checks,
        "ready": bool(all(checks.values())),
        "nonzero_action_advantage_rows": int(len(nonzero)),
        "positive_action_advantage_rows": positive,
        "negative_action_advantage_rows": negative,
        "nonzero_candidates": int(nonzero["candidate"].nunique()),
        "nonzero_dates": int(nonzero["decision_date"].nunique()),
    }


def _export(
    *,
    run_id: str,
    dataset: pd.DataFrame,
    summary: dict[str, Any],
) -> tuple[Path, Path]:
    research_root = (base.PROJECT_ROOT / "research_output").resolve()
    research_root.mkdir(parents=True, exist_ok=True)
    export_dir = research_root / EXPORT_FOLDER_NAME
    zip_path = research_root / EXPORT_ZIP_NAME
    temp_dir = research_root / f".{EXPORT_FOLDER_NAME}.tmp"

    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    dataset.to_csv(temp_dir / "dataset.csv", index=False)
    (temp_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    (temp_dir / "README.txt").write_text(
        "\n".join(
            [
                "Market Cycle Trader - Contextual Marginal Signature",
                "",
                f"Mongo run id: {run_id}",
                f"Internal script version: {SCRIPT_VERSION}",
                "",
                "MongoDB is the source of truth for analysis.",
                "This folder and ZIP are export-only and are never required as input.",
                "",
                f"Collections: {RUNS_COLLECTION}, {OBSERVATIONS_COLLECTION}, "
                f"{TRACE_RUNS_COLLECTION}, {TRACE_ROWS_COLLECTION}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    if export_dir.exists():
        shutil.rmtree(export_dir)
    temp_dir.replace(export_dir)

    temporary_zip = research_root / f".{EXPORT_ZIP_NAME}.tmp"
    if temporary_zip.exists():
        temporary_zip.unlink()
    with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(export_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(research_root))
    if zip_path.exists():
        zip_path.unlink()
    temporary_zip.replace(zip_path)
    return export_dir, zip_path


def main() -> int:
    args = _parser().parse_args()
    base.load_project_environment(args.env_file)

    from market_cycle_trader_api.engine import market_data
    from market_cycle_trader_api.infrastructure.persistence import mongo_repository
    from market_cycle_trader_api.schemas.requests import BacktestRequest

    mongo_uri, database_name = _runtime_mongo_settings(args, mongo_repository)
    client = MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=3000,
        connectTimeoutMS=3000,
        retryWrites=False,
    )
    run_id = uuid.uuid4().hex
    started = pd.Timestamp.now(tz="UTC")

    try:
        client.admin.command("ping")
        db = client[database_name]
        runs = db[RUNS_COLLECTION]
        observations = db[OBSERVATIONS_COLLECTION]
        trace_runs = db[TRACE_RUNS_COLLECTION]
        trace_rows = db[TRACE_ROWS_COLLECTION]

        strategy = base._strategy_document(db, args.strategy_sequence, args.strategy_id)
        stored = base._configuration(strategy)
        strategy_id = str(strategy.get("_id") or "").strip()
        strategy_sequence = int(strategy.get("strategy_sequence") or args.strategy_sequence)
        if not strategy_id:
            raise RuntimeError("Selected Strategy has no _id.")
        if base._normalize_date(stored.get("start_date")) != base._normalize_date(HISTORY_START):
            raise RuntimeError(
                f"Strategy start_date must be {HISTORY_START} for this frozen campaign."
            )

        required_symbols = set(_campaign_symbols())
        strategy_symbols = set(base._normalize_symbols(list(stored.get("assets") or [])))
        missing = sorted(required_symbols.difference(strategy_symbols))
        if missing:
            raise RuntimeError(f"Strategy #{strategy_sequence} is missing campaign symbols: {missing}")

        config_base = BacktestRequest.model_validate(stored).model_copy(
            update={
                "end_date": SNAPSHOT_END,
                "research_market_data_mode": "database_only",
                "mongo_cache_enabled": True,
                "market_data_require_complete_history": True,
            }
        )
        config_snapshot = config_base.model_dump(mode="json")
        strategy_configuration_hash = _canonical_hash(config_snapshot)

        runs.insert_one(
            {
                "_id": run_id,
                "schema_version": 1,
                "script_version": SCRIPT_VERSION,
                "experiment": EXPERIMENT_NAME,
                "status": "running",
                "started_utc": started.to_pydatetime(),
                "updated_utc": started.to_pydatetime(),
                "strategy_id": strategy_id,
                "strategy_sequence": strategy_sequence,
                "strategy_configuration_hash": strategy_configuration_hash,
                "history_start": HISTORY_START,
                "snapshot_end": SNAPSHOT_END,
                "horizon_sessions": HORIZON_SESSIONS,
                "persistent_source": "mongodb_local",
                "filesystem_role": "export_only",
                "universes": _mongo_value(UNIVERSES),
                "candidates": list(CANDIDATES),
                "decision_dates": list(DECISION_DATES),
            }
        )

        frames, market_provenance = _load_market_frames(config_base, market_data)
        snapshot = base._snapshot_table(frames)
        market_snapshot_hash = base._snapshot_hash(snapshot)
        runs.update_one(
            {"_id": run_id},
            {
                "$set": {
                    "market_snapshot_hash": market_snapshot_hash,
                    "market_provenance": _mongo_value(market_provenance),
                    "updated_utc": pd.Timestamp.now(tz="UTC").to_pydatetime(),
                }
            },
        )

        expected_sessions = base._expected_sessions(
            base._normalize_date(HISTORY_START),
            base._normalize_date(SNAPSHOT_END),
        )
        frozen._BASELINE_SCHEDULE_BY_KEY.clear()

        aggregate_rows: list[dict[str, Any]] = []
        total_contexts = len(DECISION_DATES) * len(UNIVERSES) * len(CANDIDATES)
        completed_contexts = 0

        for decision_text in DECISION_DATES:
            decision = base._utc_timestamp(decision_text)
            horizon_end = base._horizon_end(expected_sessions, decision, HORIZON_SESSIONS)
            base._log(f"Decision {decision.date()} -> {horizon_end.date()}")

            for universe in UNIVERSES:
                universe_name = str(universe["name"])
                reference_assets = list(universe["assets"])
                config = config_base.model_copy(
                    update={"assets": reference_assets, "end_date": SNAPSHOT_END}
                )
                seed_frames = {symbol: frames[symbol] for symbol in reference_assets}

                base._log(f"  {universe_name}: baseline")
                baseline_metrics, baseline_sessions, baseline_captured = _run_replay(
                    db=db,
                    config=config,
                    strategy_id=strategy_id,
                    reference_assets=reference_assets,
                    candidate=None,
                    frames=seed_frames,
                    decision=decision,
                    horizon_end=horizon_end,
                    forced=False,
                )
                baseline_capital = _ending_capital(
                    baseline_metrics, f"{decision_text}/{universe_name}/baseline"
                )
                _persist_capture(
                    run_id=run_id,
                    decision_date=decision_text,
                    universe_name=universe_name,
                    candidate=None,
                    arm="baseline",
                    captured=baseline_captured,
                    trace_runs=trace_runs,
                    trace_rows=trace_rows,
                )

                for candidate in CANDIDATES:
                    challenger_frames = dict(seed_frames)
                    challenger_frames[candidate] = frames[candidate]

                    base._log(f"  {universe_name}/{candidate}: normal-policy arm")
                    policy_metrics, policy_sessions, policy_captured = _run_replay(
                        db=db,
                        config=config,
                        strategy_id=strategy_id,
                        reference_assets=reference_assets,
                        candidate=candidate,
                        frames=challenger_frames,
                        decision=decision,
                        horizon_end=horizon_end,
                        forced=False,
                    )
                    compatibility = base.discovery._research_context_compatibility(
                        baseline_sessions, policy_sessions
                    )
                    if not bool(compatibility.get("research_context_compatible")):
                        raise RuntimeError(
                            f"Policy context mismatch for {decision_text}/{universe_name}/{candidate}"
                        )
                    policy_capital = _ending_capital(
                        policy_metrics, f"{decision_text}/{universe_name}/{candidate}/policy"
                    )
                    policy_first, policy_transitions = _first_selected_asset(policy_captured)
                    _persist_capture(
                        run_id=run_id,
                        decision_date=decision_text,
                        universe_name=universe_name,
                        candidate=candidate,
                        arm="policy",
                        captured=policy_captured,
                        trace_runs=trace_runs,
                        trace_rows=trace_rows,
                    )

                    base._log(f"  {universe_name}/{candidate}: forced-first-action arm")
                    forced_metrics, forced_sessions, forced_captured = _run_replay(
                        db=db,
                        config=config,
                        strategy_id=strategy_id,
                        reference_assets=reference_assets,
                        candidate=candidate,
                        frames=challenger_frames,
                        decision=decision,
                        horizon_end=horizon_end,
                        forced=True,
                    )
                    forced_compatibility = base.discovery._research_context_compatibility(
                        policy_sessions, forced_sessions
                    )
                    if not bool(forced_compatibility.get("research_context_compatible")):
                        raise RuntimeError(
                            f"Forced context mismatch for {decision_text}/{universe_name}/{candidate}"
                        )
                    forced_capital = _ending_capital(
                        forced_metrics, f"{decision_text}/{universe_name}/{candidate}/forced"
                    )
                    forced_first, forced_transitions = _first_selected_asset(forced_captured)
                    if forced_first != candidate:
                        raise RuntimeError(
                            f"Forced-action invariant failed for {decision_text}/{universe_name}/{candidate}: "
                            f"first_selected={forced_first}"
                        )
                    if forced_transitions != policy_transitions:
                        raise RuntimeError(
                            f"Paired transition mismatch for {decision_text}/{universe_name}/{candidate}: "
                            f"policy={policy_transitions}, forced={forced_transitions}"
                        )
                    _persist_capture(
                        run_id=run_id,
                        decision_date=decision_text,
                        universe_name=universe_name,
                        candidate=candidate,
                        arm="forced",
                        captured=forced_captured,
                        trace_runs=trace_runs,
                        trace_rows=trace_rows,
                    )

                    row = {
                        "run_id": run_id,
                        "decision_date": decision_text,
                        "horizon_end": pd.Timestamp(horizon_end).date().isoformat(),
                        "universe_name": universe_name,
                        "candidate": candidate,
                        "baseline_ending_capital": baseline_capital,
                        "policy_ending_capital": policy_capital,
                        "forced_action_ending_capital": forced_capital,
                        "source_direct_delta_log_capital": float(
                            math.log(policy_capital / baseline_capital)
                        ),
                        "action_advantage_log": float(
                            math.log(forced_capital / policy_capital)
                        ),
                        "action_advantage_rate": float(
                            forced_capital / policy_capital - 1.0
                        ),
                        "policy_first_selected_asset": policy_first,
                        "policy_first_selected_candidate": bool(policy_first == candidate),
                        "forced_first_selected_asset": forced_first,
                        "execution_transitions": int(forced_transitions),
                        "decision_points": int(forced_transitions + 1),
                        "market_snapshot_hash": market_snapshot_hash,
                        "strategy_configuration_hash": strategy_configuration_hash,
                    }
                    aggregate_rows.append(row)
                    observations.insert_one(_mongo_value(row))
                    completed_contexts += 1
                    runs.update_one(
                        {"_id": run_id},
                        {
                            "$set": {
                                "updated_utc": pd.Timestamp.now(tz="UTC").to_pydatetime(),
                                "current_context": (
                                    f"{decision_text}/{universe_name}/{candidate}"
                                ),
                                "completed_contexts": completed_contexts,
                                "total_contexts": total_contexts,
                            }
                        },
                    )
                    base._log(
                        f"    [paired] Y={row['action_advantage_log']:+.8f}; "
                        f"policy={policy_capital:.2f}; forced={forced_capital:.2f}; "
                        f"progress={completed_contexts}/{total_contexts}"
                    )

        dataset = pd.DataFrame(aggregate_rows)
        readiness = _readiness(dataset)
        ended = pd.Timestamp.now(tz="UTC")
        summary = {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "experiment": EXPERIMENT_NAME,
            "status": "completed",
            "run_id": run_id,
            "strategy_id": strategy_id,
            "strategy_sequence": strategy_sequence,
            "strategy_configuration_hash": strategy_configuration_hash,
            "market_snapshot_hash": market_snapshot_hash,
            "history_start": HISTORY_START,
            "snapshot_end": SNAPSHOT_END,
            "horizon_sessions": HORIZON_SESSIONS,
            "rows": int(len(dataset)),
            "dates": int(dataset["decision_date"].nunique()),
            "universes": int(dataset["universe_name"].nunique()),
            "candidates": int(dataset["candidate"].nunique()),
            "readiness": readiness,
            "model_capacity_benchmark_ready": bool(readiness["ready"]),
            "execution_transitions_values": sorted(
                int(x) for x in dataset["execution_transitions"].unique()
            ),
            "decision_points_values": sorted(
                int(x) for x in dataset["decision_points"].unique()
            ),
            "target_definition": (
                "action_advantage_log = log(W_forced_candidate_then_same_policy / "
                "W_same_policy_without_force). Both arms are generated in the same run, "
                "with the same MongoDB market data and fold-aware frozen switch-margin schedule."
            ),
            "storage": {
                "source_of_truth": "MongoDB local",
                "runs_collection": RUNS_COLLECTION,
                "observations_collection": OBSERVATIONS_COLLECTION,
                "trace_runs_collection": TRACE_RUNS_COLLECTION,
                "trace_rows_collection": TRACE_ROWS_COLLECTION,
                "filesystem": "export only; never required as analysis input",
            },
            "started_utc": started.isoformat(),
            "completed_utc": ended.isoformat(),
            "elapsed_seconds": float((ended - started).total_seconds()),
        }

        runs.update_many(
            {"experiment": EXPERIMENT_NAME, "is_latest": True},
            {"$set": {"is_latest": False}},
        )
        runs.update_one(
            {"_id": run_id},
            {
                "$set": {
                    **_mongo_value(summary),
                    "status": "completed",
                    "is_latest": True,
                    "updated_utc": ended.to_pydatetime(),
                },
                "$unset": {"current_context": ""},
            },
        )

        export_dir, zip_path = _export(run_id=run_id, dataset=dataset, summary=summary)
        base._log(f"[complete] Mongo run_id={run_id}")
        base._log(f"[complete] Export folder={export_dir}")
        base._log(f"[complete] ZIP={zip_path}")
        sound_mode = frozen.sound._play_completion_sound()
        base._log(f"[complete] Completion sound mode={sound_mode}")
        return 0

    except Exception as exc:
        try:
            db = client[database_name]
            db[RUNS_COLLECTION].update_one(
                {"_id": run_id},
                {
                    "$set": {
                        "status": "failed",
                        "updated_utc": pd.Timestamp.now(tz="UTC").to_pydatetime(),
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                },
            )
        except Exception:
            pass
        raise
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
