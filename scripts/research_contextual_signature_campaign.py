from __future__ import annotations

import math
import time
import traceback
import uuid
from typing import Any

import pandas as pd
from pymongo import MongoClient

import research_contextual_signature_protocol as prev

SCRIPT_VERSION = "contextual-marginal-signature-v1.0.17.4"
EXPERIMENT_NAME = prev.EXPERIMENT_NAME
HISTORY_START = prev.HISTORY_START
SNAPSHOT_END = prev.SNAPSHOT_END
HORIZON_SESSIONS = prev.HORIZON_SESSIONS
EXPORT_FOLDER_NAME = prev.EXPORT_FOLDER_NAME
EXPORT_ZIP_NAME = prev.EXPORT_ZIP_NAME
RUNS_COLLECTION = prev.RUNS_COLLECTION
OBSERVATIONS_COLLECTION = prev.OBSERVATIONS_COLLECTION
TRACE_RUNS_COLLECTION = prev.TRACE_RUNS_COLLECTION
TRACE_ROWS_COLLECTION = prev.TRACE_ROWS_COLLECTION
ORIGINAL25 = prev.ORIGINAL25
UNIVERSES = prev.UNIVERSES
CANDIDATES = prev.CANDIDATES
DECISION_DATES = prev.DECISION_DATES

prev.SCRIPT_VERSION = SCRIPT_VERSION

base = prev.base
frozen = prev.frozen
paired = prev.paired
live = prev.live
storage = prev.storage


def _parser():
    return prev._parser()


def _campaign_symbols() -> list[str]:
    return prev._campaign_symbols()


def _observation_key(value: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(value.get("decision_date") or ""),
        str(value.get("universe_name") or ""),
        str(value.get("candidate") or ""),
    )


def _clean_observation(document: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in document.items() if key != "_id"}


def _campaign_hash(
    *,
    strategy_id: str,
    configuration_hash: str,
    snapshot_hash: str,
) -> str:
    return storage.canonical_hash(
        {
            "experiment": EXPERIMENT_NAME,
            "strategy_id": strategy_id,
            "strategy_configuration_hash": configuration_hash,
            "market_snapshot_hash": snapshot_hash,
            "history_start": HISTORY_START,
            "snapshot_end": SNAPSHOT_END,
            "horizon_sessions": HORIZON_SESSIONS,
            "decision_dates": list(DECISION_DATES),
            "universes": UNIVERSES,
            "candidates": CANDIDATES,
        }
    )


def _preflight_executable_states(
    *,
    frames: dict[str, pd.DataFrame],
    config_base: Any,
    capital_rotation: Any,
) -> None:
    expected_sessions = base._expected_sessions(
        base._normalize_date(HISTORY_START),
        base._normalize_date(SNAPSHOT_END),
    )
    invalid: list[str] = []
    for universe in UNIVERSES:
        universe_name = str(universe["name"])
        reference_assets = list(universe["assets"])
        config = config_base.model_copy(
            update={"assets": reference_assets, "end_date": SNAPSHOT_END}
        )
        seed_frames = {symbol: frames[symbol] for symbol in reference_assets}
        _, common_dates = capital_rotation.prepare_rotation_panel(seed_frames, config)
        folds = capital_rotation._build_walk_forward_folds(common_dates, config)
        if not folds:
            invalid.append(f"{universe_name}: no walk-forward folds")
            continue

        oos_start = int(folds[0]["test_start_index"])
        oos_end = int(folds[-1]["test_end_index"])
        for decision_text in DECISION_DATES:
            decision = base._utc_timestamp(decision_text)
            horizon_end = base._horizon_end(
                expected_sessions,
                decision,
                HORIZON_SESSIONS,
            )
            requested_start = int(common_dates.searchsorted(decision, side="left"))
            requested_end = int(common_dates.searchsorted(horizon_end, side="right"))
            execution_start = max(oos_start, requested_start)
            execution_end = min(oos_end, requested_end)
            if execution_start >= execution_end:
                invalid.append(
                    f"{universe_name}/{live.display_date(decision_text)}"
                )

    if invalid:
        rendered = ", ".join(invalid[:12])
        suffix = " ..." if len(invalid) > 12 else ""
        raise RuntimeError(
            "Campaign preflight found non-executable temporal states before any replay: "
            f"{rendered}{suffix}"
        )

    live.console_log(
        f"[preflight] OK | states={len(DECISION_DATES)} | "
        f"universes={len(UNIVERSES)} | "
        "all requested windows intersect executable OOS folds"
    )


def _find_resumable_run(
    *,
    runs: Any,
    strategy_id: str,
    config_hash: str,
    snapshot_hash: str,
    campaign_hash: str,
) -> dict[str, Any] | None:
    exact = runs.find_one(
        {
            "experiment": EXPERIMENT_NAME,
            "campaign_hash": campaign_hash,
            "status": {"$in": ["running", "failed"]},
        },
        sort=[("started_utc", -1)],
    )
    if exact is not None:
        return exact

    return runs.find_one(
        {
            "experiment": EXPERIMENT_NAME,
            "strategy_id": strategy_id,
            "strategy_configuration_hash": config_hash,
            "market_snapshot_hash": snapshot_hash,
            "horizon_sessions": HORIZON_SESSIONS,
            "decision_dates": list(DECISION_DATES),
            "status": {"$in": ["running", "failed"]},
        },
        sort=[("started_utc", -1)],
    )


def _load_observations(
    collection: Any,
    run_id: str,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    output: dict[tuple[str, str, str], dict[str, Any]] = {}
    for document in collection.find({"run_id": run_id}):
        clean = _clean_observation(dict(document))
        key = _observation_key(clean)
        if all(key):
            output[key] = clean
    return output


def _ordered_dataset(
    observation_map: dict[tuple[str, str, str], dict[str, Any]],
) -> pd.DataFrame:
    if not observation_map:
        return pd.DataFrame()

    date_order = {value: index for index, value in enumerate(DECISION_DATES)}
    universe_order = {
        str(item["name"]): index for index, item in enumerate(UNIVERSES)
    }
    candidate_order = {
        value: index for index, value in enumerate(CANDIDATES)
    }
    rows = list(observation_map.values())
    rows.sort(
        key=lambda row: (
            date_order.get(str(row.get("decision_date")), 9999),
            universe_order.get(str(row.get("universe_name")), 9999),
            candidate_order.get(str(row.get("candidate")), 9999),
        )
    )
    return pd.DataFrame(rows)


def _run_replay(**kwargs: Any):
    return prev._run_replay(**kwargs)


def main() -> int:
    prev._install_console_logging()
    args = _parser().parse_args()
    base.load_project_environment(args.env_file)

    from market_cycle_trader_api.engine import capital_rotation, market_data
    from market_cycle_trader_api.infrastructure.persistence import mongo_repository
    from market_cycle_trader_api.schemas.requests import BacktestRequest

    mongo_uri, database_name = storage.runtime_mongo_settings(
        args,
        mongo_repository,
        base,
    )
    client = MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=3000,
        connectTimeoutMS=3000,
        retryWrites=False,
    )

    process_started = pd.Timestamp.now(tz="UTC")
    process_started_monotonic = time.monotonic()
    total_observations = (
        len(DECISION_DATES) * len(UNIVERSES) * len(CANDIDATES)
    )
    expected_replays = (
        len(DECISION_DATES) * len(UNIVERSES) * (1 + 2 * len(CANDIDATES))
    )
    run_id: str | None = None

    live.console_log(
        f"[study] temporal action-advantage expansion | "
        f"states={len(DECISION_DATES)} | universes={len(UNIVERSES)} | "
        f"candidates={len(CANDIDATES)} | observations={total_observations} | "
        f"expected replays={expected_replays}"
    )
    live.console_log(
        "[study] MongoDB local is the source of truth; "
        "execution supports automatic resume."
    )

    try:
        storage.mongo_retry(lambda: client.admin.command("ping"))
        db = client[database_name]
        runs = db[RUNS_COLLECTION]
        observations = db[OBSERVATIONS_COLLECTION]
        trace_runs = db[TRACE_RUNS_COLLECTION]
        trace_rows = db[TRACE_ROWS_COLLECTION]

        strategy = base._strategy_document(
            db,
            args.strategy_sequence,
            args.strategy_id,
        )
        stored = base._configuration(strategy)
        strategy_id = str(strategy.get("_id") or "").strip()
        strategy_sequence = int(
            strategy.get("strategy_sequence") or args.strategy_sequence
        )
        if not strategy_id:
            raise RuntimeError("Selected Strategy has no _id.")

        if base._normalize_date(stored.get("start_date")) != base._normalize_date(
            HISTORY_START
        ):
            raise RuntimeError(
                f"Strategy start_date must be "
                f"{live.display_date(HISTORY_START)} for this campaign."
            )

        missing = sorted(
            set(_campaign_symbols()).difference(
                base._normalize_symbols(list(stored.get("assets") or []))
            )
        )
        if missing:
            raise RuntimeError(
                f"Strategy #{strategy_sequence} is missing campaign symbols: "
                f"{missing}"
            )

        config_base = BacktestRequest.model_validate(stored).model_copy(
            update={
                "end_date": SNAPSHOT_END,
                "research_market_data_mode": "database_only",
                "mongo_cache_enabled": True,
                "market_data_require_complete_history": True,
            }
        )
        config_snapshot = config_base.model_dump(mode="json")
        config_hash = storage.canonical_hash(config_snapshot)

        frames, provenance = prev._load_market_frames(config_base, market_data)
        snapshot_hash = base._snapshot_hash(base._snapshot_table(frames))
        live.console_log(
            f"[market] complete | symbols={len(frames)} | "
            f"snapshot hash={snapshot_hash[:12]}…"
        )
        _preflight_executable_states(
            frames=frames,
            config_base=config_base,
            capital_rotation=capital_rotation,
        )

        campaign_hash = _campaign_hash(
            strategy_id=strategy_id,
            configuration_hash=config_hash,
            snapshot_hash=snapshot_hash,
        )
        resumable = _find_resumable_run(
            runs=runs,
            strategy_id=strategy_id,
            config_hash=config_hash,
            snapshot_hash=snapshot_hash,
            campaign_hash=campaign_hash,
        )

        if resumable is None:
            run_id = uuid.uuid4().hex
            run_started = process_started
            storage.mongo_retry(
                lambda: runs.insert_one(
                    {
                        "_id": run_id,
                        "schema_version": 2,
                        "script_version": SCRIPT_VERSION,
                        "experiment": EXPERIMENT_NAME,
                        "campaign_hash": campaign_hash,
                        "status": "running",
                        "started_utc": run_started.to_pydatetime(),
                        "updated_utc": run_started.to_pydatetime(),
                        "strategy_id": strategy_id,
                        "strategy_sequence": strategy_sequence,
                        "strategy_configuration_hash": config_hash,
                        "market_snapshot_hash": snapshot_hash,
                        "market_provenance": storage.mongo_value(provenance),
                        "history_start": HISTORY_START,
                        "snapshot_end": SNAPSHOT_END,
                        "horizon_sessions": HORIZON_SESSIONS,
                        "persistent_source": "mongodb_local",
                        "filesystem_role": "export_only",
                        "resume_supported": True,
                        "universes": storage.mongo_value(UNIVERSES),
                        "candidates": list(CANDIDATES),
                        "decision_dates": list(DECISION_DATES),
                        "planned_observations": total_observations,
                        "expected_replays": expected_replays,
                        "resume_count": 0,
                    }
                )
            )
            live.console_log(f"[checkpoint] new Mongo run={run_id}")
        else:
            run_id = str(resumable["_id"])
            run_started = pd.Timestamp(
                resumable.get("started_utc") or process_started
            )
            if run_started.tzinfo is None:
                run_started = run_started.tz_localize("UTC")
            else:
                run_started = run_started.tz_convert("UTC")

            previous_error = resumable.get("error")
            storage.mongo_retry(
                lambda: runs.update_one(
                    {"_id": run_id},
                    {
                        "$set": {
                            "schema_version": 2,
                            "script_version": SCRIPT_VERSION,
                            "campaign_hash": campaign_hash,
                            "status": "running",
                            "resume_supported": True,
                            "resumed_utc": process_started.to_pydatetime(),
                            "updated_utc": process_started.to_pydatetime(),
                            "previous_error": previous_error,
                            "market_snapshot_hash": snapshot_hash,
                            "market_provenance": storage.mongo_value(provenance),
                        },
                        "$inc": {"resume_count": 1},
                        "$unset": {"error": "", "traceback": ""},
                    },
                )
            )

        observation_map = _load_observations(observations, run_id)
        completed = len(observation_map)
        if completed:
            live.console_log(
                f"[resume] recovered {completed}/{total_observations} "
                "completed observations from MongoDB; "
                "completed pairs will not be replayed"
            )
        else:
            live.console_log(
                "[checkpoint] no completed observations recovered; "
                "starting campaign"
            )

        expected_sessions = base._expected_sessions(
            base._normalize_date(HISTORY_START),
            base._normalize_date(SNAPSHOT_END),
        )
        frozen._BASELINE_SCHEDULE_BY_KEY.clear()

        for state_index, decision_text in enumerate(
            DECISION_DATES,
            start=1,
        ):
            decision = base._utc_timestamp(decision_text)
            horizon_end = base._horizon_end(
                expected_sessions,
                decision,
                HORIZON_SESSIONS,
            )
            live.console_log(
                f"[state {state_index}/{len(DECISION_DATES)}] "
                f"{live.display_date(decision)} → "
                f"{live.display_date(horizon_end)}"
            )

            state_has_pending = False
            for universe in UNIVERSES:
                universe_name = str(universe["name"])
                pending = [
                    candidate
                    for candidate in CANDIDATES
                    if (
                        decision_text,
                        universe_name,
                        candidate,
                    )
                    not in observation_map
                ]

                if not pending:
                    live.console_log(
                        f"  [resume] {universe_name} complete | "
                        "skipping 7/7 candidate pairs"
                    )
                    context_rows = [
                        observation_map[
                            (decision_text, universe_name, candidate)
                        ]
                        for candidate in CANDIDATES
                    ]
                    prev._log_context_summary(
                        context_rows,
                        decision_text,
                        universe_name,
                    )
                    continue

                state_has_pending = True
                reference_assets = list(universe["assets"])
                config = config_base.model_copy(
                    update={
                        "assets": reference_assets,
                        "end_date": SNAPSHOT_END,
                    }
                )
                seed_frames = {
                    symbol: frames[symbol]
                    for symbol in reference_assets
                }

                live.console_log(
                    f"  [baseline] {universe_name} | "
                    f"assets={len(reference_assets)} | "
                    f"pending candidates={len(pending)}"
                )
                (
                    baseline_metrics,
                    baseline_sessions,
                    baseline_captured,
                ) = _run_replay(
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
                baseline_capital = prev._ending_capital(
                    baseline_metrics,
                    f"{live.display_date(decision_text)}/"
                    f"{universe_name}/baseline",
                )
                storage.persist_capture(
                    run_id=run_id,
                    decision_date=decision_text,
                    universe_name=universe_name,
                    candidate=None,
                    arm="baseline",
                    captured=baseline_captured,
                    trace_runs=trace_runs,
                    trace_rows=trace_rows,
                )
                live.console_log(
                    f"  [baseline] {universe_name} complete | "
                    f"capital={baseline_capital:,.2f}"
                )

                for candidate in pending:
                    pair_number = len(observation_map) + 1
                    challenger_frames = dict(seed_frames)
                    challenger_frames[candidate] = frames[candidate]

                    live.console_log(
                        f"  [pair {pair_number}/{total_observations}] "
                        f"{universe_name}/{candidate} | normal policy"
                    )
                    (
                        policy_metrics,
                        policy_sessions,
                        policy_captured,
                    ) = _run_replay(
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

                    compatibility = (
                        base.discovery._research_context_compatibility(
                            baseline_sessions,
                            policy_sessions,
                        )
                    )
                    if not bool(
                        compatibility.get("research_context_compatible")
                    ):
                        raise RuntimeError(
                            "Policy context mismatch for "
                            f"{live.display_date(decision_text)}/"
                            f"{universe_name}/{candidate}"
                        )

                    policy_capital = prev._ending_capital(
                        policy_metrics,
                        f"{live.display_date(decision_text)}/"
                        f"{universe_name}/{candidate}/policy",
                    )
                    (
                        policy_first,
                        policy_transitions,
                    ) = prev._first_selected_asset(policy_captured)
                    storage.persist_capture(
                        run_id=run_id,
                        decision_date=decision_text,
                        universe_name=universe_name,
                        candidate=candidate,
                        arm="policy",
                        captured=policy_captured,
                        trace_runs=trace_runs,
                        trace_rows=trace_rows,
                    )

                    live.console_log(
                        f"  [pair {pair_number}/{total_observations}] "
                        f"{universe_name}/{candidate} | "
                        "force candidate once"
                    )
                    (
                        forced_metrics,
                        forced_sessions,
                        forced_captured,
                    ) = _run_replay(
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

                    forced_compatibility = (
                        base.discovery._research_context_compatibility(
                            policy_sessions,
                            forced_sessions,
                        )
                    )
                    if not bool(
                        forced_compatibility.get(
                            "research_context_compatible"
                        )
                    ):
                        raise RuntimeError(
                            "Forced context mismatch for "
                            f"{live.display_date(decision_text)}/"
                            f"{universe_name}/{candidate}"
                        )

                    forced_capital = prev._ending_capital(
                        forced_metrics,
                        f"{live.display_date(decision_text)}/"
                        f"{universe_name}/{candidate}/forced",
                    )
                    (
                        forced_first,
                        forced_transitions,
                    ) = prev._first_selected_asset(forced_captured)

                    if forced_first != candidate:
                        raise RuntimeError(
                            "Forced-action invariant failed for "
                            f"{live.display_date(decision_text)}/"
                            f"{universe_name}/{candidate}: "
                            f"first={forced_first}"
                        )
                    if forced_transitions != policy_transitions:
                        raise RuntimeError(
                            "Paired transition mismatch for "
                            f"{live.display_date(decision_text)}/"
                            f"{universe_name}/{candidate}: "
                            f"policy={policy_transitions}, "
                            f"forced={forced_transitions}"
                        )

                    storage.persist_capture(
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
                        "horizon_end": pd.Timestamp(
                            horizon_end
                        ).date().isoformat(),
                        "universe_name": universe_name,
                        "candidate": candidate,
                        "baseline_ending_capital": baseline_capital,
                        "policy_ending_capital": policy_capital,
                        "forced_action_ending_capital": forced_capital,
                        "source_direct_delta_log_capital": float(
                            math.log(
                                policy_capital / baseline_capital
                            )
                        ),
                        "action_advantage_log": float(
                            math.log(
                                forced_capital / policy_capital
                            )
                        ),
                        "action_advantage_rate": float(
                            forced_capital / policy_capital - 1.0
                        ),
                        "policy_first_selected_asset": policy_first,
                        "policy_first_selected_candidate": bool(
                            policy_first == candidate
                        ),
                        "forced_first_selected_asset": forced_first,
                        "execution_transitions": int(
                            forced_transitions
                        ),
                        "decision_points": int(
                            forced_transitions + 1
                        ),
                        "market_snapshot_hash": snapshot_hash,
                        "strategy_configuration_hash": config_hash,
                    }
                    storage.upsert_observation(observations, row)
                    observation_map[_observation_key(row)] = row

                    completed = len(observation_map)
                    elapsed = (
                        time.monotonic()
                        - process_started_monotonic
                    )
                    progress = completed / total_observations
                    eta = (
                        elapsed * (1.0 - progress) / progress
                        if progress > 0
                        else None
                    )

                    storage.mongo_retry(
                        lambda: runs.update_one(
                            {"_id": run_id},
                            {
                                "$set": {
                                    "updated_utc": pd.Timestamp.now(
                                        tz="UTC"
                                    ).to_pydatetime(),
                                    "current_context": (
                                        f"{decision_text}/"
                                        f"{universe_name}/{candidate}"
                                    ),
                                    "completed_observations": completed,
                                    "total_observations": total_observations,
                                    "elapsed_seconds_current_process": float(
                                        elapsed
                                    ),
                                    "eta_seconds_live": (
                                        None
                                        if eta is None
                                        else float(eta)
                                    ),
                                }
                            },
                        )
                    )
                    live.console_log(
                        f"    [result] "
                        f"Y={row['action_advantage_log']:+.6f} "
                        f"({row['action_advantage_rate']:+.2%}) | "
                        f"first policy={policy_first} | "
                        f"policy={policy_capital:,.2f} | "
                        f"forced={forced_capital:,.2f} | "
                        f"checkpoint={completed}/"
                        f"{total_observations} | "
                        f"ETA≈{live.format_duration(eta)}"
                    )

                context_rows = [
                    observation_map[
                        (decision_text, universe_name, candidate)
                    ]
                    for candidate in CANDIDATES
                    if (
                        decision_text,
                        universe_name,
                        candidate,
                    )
                    in observation_map
                ]
                if len(context_rows) == len(CANDIDATES):
                    prev._log_context_summary(
                        context_rows,
                        decision_text,
                        universe_name,
                    )

            dataset_so_far = _ordered_dataset(
                observation_map
            )
            if not dataset_so_far.empty:
                elapsed = (
                    time.monotonic()
                    - process_started_monotonic
                )
                partial = prev._log_partial(
                    dataset_so_far.to_dict(orient="records"),
                    state_index,
                    elapsed,
                )
                storage.mongo_retry(
                    lambda: runs.update_one(
                        {"_id": run_id},
                        {
                            "$set": {
                                "partial_analysis": (
                                    storage.mongo_value(partial)
                                ),
                                "completed_temporal_states": int(
                                    dataset_so_far.groupby(
                                        "decision_date"
                                    )["candidate"]
                                    .count()
                                    .ge(
                                        len(UNIVERSES)
                                        * len(CANDIDATES)
                                    )
                                    .sum()
                                ),
                                "updated_utc": pd.Timestamp.now(
                                    tz="UTC"
                                ).to_pydatetime(),
                            }
                        },
                    )
                )

            if not state_has_pending:
                live.console_log(
                    "[resume] state already complete; "
                    "no replay executed"
                )

        dataset = _ordered_dataset(observation_map)
        if len(dataset) != total_observations:
            raise RuntimeError(
                "Campaign ended with incomplete checkpoint set: "
                f"{len(dataset)}/{total_observations} observations."
            )

        readiness = live.readiness(
            dataset,
            len(CANDIDATES),
            len(UNIVERSES),
        )
        final_analysis = live.partial_analysis(
            dataset,
            len(UNIVERSES),
        )
        ended = pd.Timestamp.now(tz="UTC")

        summary = {
            "schema_version": 2,
            "script_version": SCRIPT_VERSION,
            "experiment": EXPERIMENT_NAME,
            "campaign_hash": campaign_hash,
            "status": "completed",
            "run_id": run_id,
            "strategy_id": strategy_id,
            "strategy_sequence": strategy_sequence,
            "strategy_configuration_hash": config_hash,
            "market_snapshot_hash": snapshot_hash,
            "history_start": HISTORY_START,
            "snapshot_end": SNAPSHOT_END,
            "horizon_sessions": HORIZON_SESSIONS,
            "rows": int(len(dataset)),
            "dates": int(dataset["decision_date"].nunique()),
            "years": int(
                pd.to_datetime(
                    dataset["decision_date"]
                ).dt.year.nunique()
            ),
            "universes": int(
                dataset["universe_name"].nunique()
            ),
            "candidates": int(
                dataset["candidate"].nunique()
            ),
            "final_analysis": final_analysis,
            "readiness": readiness,
            "temporal_model_benchmark_ready": bool(
                readiness["ready"]
            ),
            "resume_supported": True,
            "target_definition": (
                "action_advantage_log = "
                "log(W_forced_candidate_then_same_policy / "
                "W_same_policy_without_force). "
                "Both arms use the same MongoDB market data and "
                "fold-aware frozen switch-margin schedule."
            ),
            "next_step_if_ready": (
                "Run the integrated research tournament over "
                "linear, LightGBM, MLP and temporal neural models "
                "using the same frozen MongoDB observations and "
                "chronological validation."
            ),
            "storage": {
                "source_of_truth": "MongoDB local",
                "runs_collection": RUNS_COLLECTION,
                "observations_collection": (
                    OBSERVATIONS_COLLECTION
                ),
                "trace_runs_collection": (
                    TRACE_RUNS_COLLECTION
                ),
                "trace_rows_collection": (
                    TRACE_ROWS_COLLECTION
                ),
                "filesystem": (
                    "export only; never required as analysis input"
                ),
            },
            "started_utc": run_started.isoformat(),
            "completed_utc": ended.isoformat(),
            "elapsed_seconds_wall": float(
                (ended - run_started).total_seconds()
            ),
            "elapsed_seconds_current_process": float(
                time.monotonic()
                - process_started_monotonic
            ),
        }

        storage.mongo_retry(
            lambda: runs.update_many(
                {
                    "experiment": EXPERIMENT_NAME,
                    "is_latest": True,
                },
                {"$set": {"is_latest": False}},
            )
        )
        storage.mongo_retry(
            lambda: runs.update_one(
                {"_id": run_id},
                {
                    "$set": {
                        **storage.mongo_value(summary),
                        "status": "completed",
                        "is_latest": True,
                        "updated_utc": ended.to_pydatetime(),
                        "completed_observations": (
                            total_observations
                        ),
                    },
                    "$unset": {
                        "current_context": "",
                        "eta_seconds_live": "",
                        "error": "",
                        "traceback": "",
                    },
                },
            )
        )

        export_dir, zip_path = storage.export_result(
            project_root=base.PROJECT_ROOT,
            run_id=run_id,
            script_version=SCRIPT_VERSION,
            dataset=dataset,
            summary=summary,
            export_folder_name=EXPORT_FOLDER_NAME,
            export_zip_name=EXPORT_ZIP_NAME,
            collections=(
                RUNS_COLLECTION,
                OBSERVATIONS_COLLECTION,
                TRACE_RUNS_COLLECTION,
                TRACE_ROWS_COLLECTION,
            ),
        )

        live.console_log(
            f"[complete] Mongo run_id={run_id}"
        )
        live.console_log(
            f"[complete] Export folder={export_dir}"
        )
        live.console_log(
            f"[complete] ZIP={zip_path}"
        )
        live.console_log(
            "[complete] temporal_model_benchmark_ready="
            f"{readiness['ready']}"
        )
        sound_mode = frozen.sound._play_completion_sound()
        live.console_log(
            f"[complete] Completion sound mode={sound_mode}"
        )
        return 0

    except Exception as exc:
        if run_id is not None:
            try:
                db = client[database_name]
                storage.mongo_retry(
                    lambda: db[RUNS_COLLECTION].update_one(
                        {"_id": run_id},
                        {
                            "$set": {
                                "status": "failed",
                                "updated_utc": (
                                    pd.Timestamp.now(
                                        tz="UTC"
                                    ).to_pydatetime()
                                ),
                                "error": (
                                    f"{type(exc).__name__}: "
                                    f"{exc}"
                                ),
                                "traceback": traceback.format_exc(),
                                "resume_supported": True,
                            }
                        },
                    )
                )
            except Exception:
                pass
        raise
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
