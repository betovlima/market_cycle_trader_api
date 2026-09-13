from __future__ import annotations

import argparse
import math
import time
import traceback
import uuid
from typing import Any

import pandas as pd
from pymongo import MongoClient

import research_contextual_marginal_signature_v1144 as frozen
import research_contextual_marginal_signature_v116 as paired
import research_contextual_signature_analysis as live
import research_contextual_signature_storage as storage

base = frozen.base

SCRIPT_VERSION = "contextual-marginal-signature-v1.0.17"
EXPERIMENT_NAME = "contextual_marginal_signature_temporal_state_expansion"
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
    "2019-01-02", "2019-05-01", "2019-09-03",
    "2020-01-02", "2020-03-02", "2020-06-01", "2020-11-02",
    "2021-01-04", "2021-05-03", "2021-09-01",
    "2022-01-03", "2022-04-01", "2022-07-01", "2022-10-03",
    "2023-01-03", "2023-05-01", "2023-09-01",
    "2024-01-02", "2024-07-01",
    "2025-01-02", "2025-07-01",
    "2026-01-02", "2026-07-01",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Single-entry temporal expansion of paired action advantage. "
            "MongoDB local is the persistent source of truth; filesystem output is export-only."
        )
    )
    parser.add_argument("--strategy-sequence", type=int, default=10)
    parser.add_argument("--strategy-id", default=None)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--mongo-uri", default=None)
    parser.add_argument("--database", default=None)
    return parser


def _install_console_logging() -> None:
    base._log = live.console_log
    frozen.diagnostics._ORIGINAL_LOG = live.console_log


def _campaign_symbols() -> list[str]:
    values = [symbol for universe in UNIVERSES for symbol in universe["assets"]]
    values.extend(CANDIDATES)
    return base._normalize_symbols(values)


def _load_market_frames(config: Any, market_data: Any) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    symbols = _campaign_symbols()
    expected = base._expected_sessions(base._normalize_date(HISTORY_START), base._normalize_date(SNAPSHOT_END))
    expected = pd.DatetimeIndex(pd.to_datetime(expected, utc=True)).normalize()
    frames: dict[str, pd.DataFrame] = {}
    provenance: list[dict[str, Any]] = []
    for index, symbol in enumerate(symbols, start=1):
        live.console_log(f"[market {index}/{len(symbols)}] MongoDB Alpaca bars: {symbol}")
        frame = market_data.validate_and_clean_bars(market_data.load_market_bars(symbol, config), config)
        observed = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True)).normalize().unique()
        missing = expected.difference(observed)
        if len(missing):
            sample = ", ".join(live.display_date(ts) for ts in missing[:5])
            raise RuntimeError(f"MongoDB history incomplete for {symbol}: missing={len(missing)} sample={sample}")
        frames[symbol] = frame
        detail = dict(frame.attrs.get("market_data_provenance") or {})
        provenance.append({
            "symbol": symbol,
            "rows": int(len(frame)),
            "first": pd.Timestamp(frame.index.min()).isoformat(),
            "last": pd.Timestamp(frame.index.max()).isoformat(),
            "source": detail.get("source") or "mongo_cache",
            "access": detail.get("research_access_path") or "mongodb_only",
        })
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
            return str(ordered.iloc[0].get("selected_asset") or "CASH").strip().upper(), int(len(ordered))
    raise RuntimeError("Captured replay has no prediction rows.")


def _run_replay(
    *, db: Any, config: Any, strategy_id: str, reference_assets: list[str], candidate: str | None,
    frames: dict[str, pd.DataFrame], decision: pd.Timestamp, horizon_end: pd.Timestamp, forced: bool,
) -> tuple[dict[str, Any], Any, list[Any]]:
    candidate_assets = [candidate] if candidate else []
    request = frozen._window_request_frozen(
        db=db,
        config=config,
        strategy_id=strategy_id,
        assets=[*reference_assets, *candidate_assets],
        reference_assets=reference_assets,
        candidate_assets=candidate_assets,
        decision_session=decision,
        horizon_end=horizon_end,
    )
    original = paired.research_challengers._simulate_exact
    try:
        paired.research_challengers._simulate_exact = paired._forced_first_action_simulator if forced else paired._ORIGINAL_SIMULATE_EXACT
        return frozen._run_with_frozen_margin(frames, request)
    finally:
        paired.research_challengers._simulate_exact = original


def _log_context_summary(rows: list[dict[str, Any]], decision_text: str, universe_name: str) -> None:
    frame = pd.DataFrame(rows)
    best = frame.loc[frame["action_advantage_log"].astype(float).idxmax()]
    best_y = float(best["action_advantage_log"])
    choice = str(best["candidate"]) if best_y > live.TOLERANCE else "NORMAL POLICY"
    pos = int((frame["action_advantage_log"] > live.TOLERANCE).sum())
    neg = int((frame["action_advantage_log"] < -live.TOLERANCE).sum())
    zero = int(len(frame) - pos - neg)
    live.console_log(
        f"[context] {live.display_date(decision_text)} | {universe_name} | best={choice} | "
        f"best Y={max(0.0, best_y):+.5f} | candidates +{pos}/-{neg}/0={zero}"
    )


def _log_partial(rows: list[dict[str, Any]], state_index: int, elapsed: float) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    stats = live.partial_analysis(frame, len(UNIVERSES))
    progress = state_index / len(DECISION_DATES)
    eta = elapsed * (1.0 - progress) / progress if progress > 0 else None
    rho = stats["cross_universe_rank_spearman_mean"]
    rho_text = "n/a" if rho is None else f"{rho:+.3f}"
    live.console_log(
        f"[partial] states={state_index}/{len(DECISION_DATES)} | obs={stats['rows']} | "
        f"+{stats['positive']} -{stats['negative']} 0={stats['zero']} | "
        f"mean Y={stats['mean_action_advantage_log']:+.5f} | median Y={stats['median_action_advantage_log']:+.5f}"
    )
    live.console_log(
        f"[partial] intervene={stats['intervention_preferred_contexts']} | "
        f"normal-policy={stats['normal_policy_preferred_contexts']} | "
        f"oracle mean={stats['oracle_mean_context_advantage_log']:+.5f} | "
        f"candidate mean leader={stats['candidate_mean_leader']} ({stats['candidate_mean_leader_value']:+.5f}) | "
        f"cross-universe rho={rho_text}"
    )
    live.console_log(f"[progress] elapsed={live.format_duration(elapsed)} | ETA≈{live.format_duration(eta)}")
    return stats


def main() -> int:
    _install_console_logging()
    args = _parser().parse_args()
    base.load_project_environment(args.env_file)

    from market_cycle_trader_api.engine import market_data
    from market_cycle_trader_api.infrastructure.persistence import mongo_repository
    from market_cycle_trader_api.schemas.requests import BacktestRequest

    mongo_uri, database_name = storage.runtime_mongo_settings(args, mongo_repository, base)
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000, connectTimeoutMS=3000, retryWrites=False)
    run_id = uuid.uuid4().hex
    started = pd.Timestamp.now(tz="UTC")
    started_monotonic = time.monotonic()
    total_observations = len(DECISION_DATES) * len(UNIVERSES) * len(CANDIDATES)
    expected_replays = len(DECISION_DATES) * len(UNIVERSES) * (1 + 2 * len(CANDIDATES))

    live.console_log(
        f"[study] temporal action-advantage expansion | states={len(DECISION_DATES)} | "
        f"universes={len(UNIVERSES)} | candidates={len(CANDIDATES)} | "
        f"observations={total_observations} | expected replays={expected_replays}"
    )
    live.console_log("[study] MongoDB local is the source of truth; research_output is export-only.")

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
            raise RuntimeError(f"Strategy start_date must be {live.display_date(HISTORY_START)} for this campaign.")

        missing = sorted(set(_campaign_symbols()).difference(base._normalize_symbols(list(stored.get("assets") or []))))
        if missing:
            raise RuntimeError(f"Strategy #{strategy_sequence} is missing campaign symbols: {missing}")

        config_base = BacktestRequest.model_validate(stored).model_copy(update={
            "end_date": SNAPSHOT_END,
            "research_market_data_mode": "database_only",
            "mongo_cache_enabled": True,
            "market_data_require_complete_history": True,
        })
        config_snapshot = config_base.model_dump(mode="json")
        config_hash = storage.canonical_hash(config_snapshot)
        runs.insert_one({
            "_id": run_id,
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "experiment": EXPERIMENT_NAME,
            "status": "running",
            "started_utc": started.to_pydatetime(),
            "updated_utc": started.to_pydatetime(),
            "strategy_id": strategy_id,
            "strategy_sequence": strategy_sequence,
            "strategy_configuration_hash": config_hash,
            "history_start": HISTORY_START,
            "snapshot_end": SNAPSHOT_END,
            "horizon_sessions": HORIZON_SESSIONS,
            "persistent_source": "mongodb_local",
            "filesystem_role": "export_only",
            "universes": storage.mongo_value(UNIVERSES),
            "candidates": list(CANDIDATES),
            "decision_dates": list(DECISION_DATES),
            "planned_observations": total_observations,
            "expected_replays": expected_replays,
        })

        frames, provenance = _load_market_frames(config_base, market_data)
        snapshot_hash = base._snapshot_hash(base._snapshot_table(frames))
        live.console_log(f"[market] complete | symbols={len(frames)} | snapshot hash={snapshot_hash[:12]}…")
        runs.update_one({"_id": run_id}, {"$set": {
            "market_snapshot_hash": snapshot_hash,
            "market_provenance": storage.mongo_value(provenance),
            "updated_utc": pd.Timestamp.now(tz="UTC").to_pydatetime(),
        }})

        expected_sessions = base._expected_sessions(base._normalize_date(HISTORY_START), base._normalize_date(SNAPSHOT_END))
        frozen._BASELINE_SCHEDULE_BY_KEY.clear()
        aggregate_rows: list[dict[str, Any]] = []
        completed = 0

        for state_index, decision_text in enumerate(DECISION_DATES, start=1):
            decision = base._utc_timestamp(decision_text)
            horizon_end = base._horizon_end(expected_sessions, decision, HORIZON_SESSIONS)
            live.console_log(
                f"[state {state_index}/{len(DECISION_DATES)}] {live.display_date(decision)} → {live.display_date(horizon_end)}"
            )
            for universe in UNIVERSES:
                universe_name = str(universe["name"])
                reference_assets = list(universe["assets"])
                config = config_base.model_copy(update={"assets": reference_assets, "end_date": SNAPSHOT_END})
                seed_frames = {symbol: frames[symbol] for symbol in reference_assets}

                live.console_log(f"  [baseline] {universe_name} | assets={len(reference_assets)}")
                baseline_metrics, baseline_sessions, baseline_captured = _run_replay(
                    db=db, config=config, strategy_id=strategy_id, reference_assets=reference_assets,
                    candidate=None, frames=seed_frames, decision=decision, horizon_end=horizon_end, forced=False,
                )
                baseline_capital = _ending_capital(baseline_metrics, f"{decision_text}/{universe_name}/baseline")
                storage.persist_capture(
                    run_id=run_id, decision_date=decision_text, universe_name=universe_name,
                    candidate=None, arm="baseline", captured=baseline_captured,
                    trace_runs=trace_runs, trace_rows=trace_rows,
                )
                live.console_log(f"  [baseline] {universe_name} complete | capital={baseline_capital:,.2f}")

                context_rows: list[dict[str, Any]] = []
                for candidate in CANDIDATES:
                    pair_number = completed + 1
                    challenger_frames = dict(seed_frames)
                    challenger_frames[candidate] = frames[candidate]
                    live.console_log(f"  [pair {pair_number}/{total_observations}] {universe_name}/{candidate} | normal policy")
                    policy_metrics, policy_sessions, policy_captured = _run_replay(
                        db=db, config=config, strategy_id=strategy_id, reference_assets=reference_assets,
                        candidate=candidate, frames=challenger_frames, decision=decision, horizon_end=horizon_end, forced=False,
                    )
                    compatibility = base.discovery._research_context_compatibility(baseline_sessions, policy_sessions)
                    if not bool(compatibility.get("research_context_compatible")):
                        raise RuntimeError(f"Policy context mismatch for {live.display_date(decision_text)}/{universe_name}/{candidate}")
                    policy_capital = _ending_capital(policy_metrics, f"{decision_text}/{universe_name}/{candidate}/policy")
                    policy_first, policy_transitions = _first_selected_asset(policy_captured)
                    storage.persist_capture(
                        run_id=run_id, decision_date=decision_text, universe_name=universe_name,
                        candidate=candidate, arm="policy", captured=policy_captured,
                        trace_runs=trace_runs, trace_rows=trace_rows,
                    )

                    live.console_log(f"  [pair {pair_number}/{total_observations}] {universe_name}/{candidate} | force candidate once")
                    forced_metrics, forced_sessions, forced_captured = _run_replay(
                        db=db, config=config, strategy_id=strategy_id, reference_assets=reference_assets,
                        candidate=candidate, frames=challenger_frames, decision=decision, horizon_end=horizon_end, forced=True,
                    )
                    forced_compatibility = base.discovery._research_context_compatibility(policy_sessions, forced_sessions)
                    if not bool(forced_compatibility.get("research_context_compatible")):
                        raise RuntimeError(f"Forced context mismatch for {live.display_date(decision_text)}/{universe_name}/{candidate}")
                    forced_capital = _ending_capital(forced_metrics, f"{decision_text}/{universe_name}/{candidate}/forced")
                    forced_first, forced_transitions = _first_selected_asset(forced_captured)
                    if forced_first != candidate:
                        raise RuntimeError(
                            f"Forced-action invariant failed for {live.display_date(decision_text)}/{universe_name}/{candidate}: first={forced_first}"
                        )
                    if forced_transitions != policy_transitions:
                        raise RuntimeError(
                            f"Paired transition mismatch for {live.display_date(decision_text)}/{universe_name}/{candidate}: "
                            f"policy={policy_transitions}, forced={forced_transitions}"
                        )
                    storage.persist_capture(
                        run_id=run_id, decision_date=decision_text, universe_name=universe_name,
                        candidate=candidate, arm="forced", captured=forced_captured,
                        trace_runs=trace_runs, trace_rows=trace_rows,
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
                        "source_direct_delta_log_capital": float(math.log(policy_capital / baseline_capital)),
                        "action_advantage_log": float(math.log(forced_capital / policy_capital)),
                        "action_advantage_rate": float(forced_capital / policy_capital - 1.0),
                        "policy_first_selected_asset": policy_first,
                        "policy_first_selected_candidate": bool(policy_first == candidate),
                        "forced_first_selected_asset": forced_first,
                        "execution_transitions": int(forced_transitions),
                        "decision_points": int(forced_transitions + 1),
                        "market_snapshot_hash": snapshot_hash,
                        "strategy_configuration_hash": config_hash,
                    }
                    aggregate_rows.append(row)
                    context_rows.append(row)
                    observations.insert_one(storage.mongo_value(row))
                    completed += 1
                    elapsed = time.monotonic() - started_monotonic
                    progress = completed / total_observations
                    eta = elapsed * (1.0 - progress) / progress if progress > 0 else None
                    runs.update_one({"_id": run_id}, {"$set": {
                        "updated_utc": pd.Timestamp.now(tz="UTC").to_pydatetime(),
                        "current_context": f"{decision_text}/{universe_name}/{candidate}",
                        "completed_observations": completed,
                        "total_observations": total_observations,
                        "elapsed_seconds_live": float(elapsed),
                        "eta_seconds_live": None if eta is None else float(eta),
                    }})
                    live.console_log(
                        f"    [result] Y={row['action_advantage_log']:+.6f} ({row['action_advantage_rate']:+.2%}) | "
                        f"first policy={policy_first} | policy={policy_capital:,.2f} | forced={forced_capital:,.2f} | "
                        f"progress={completed}/{total_observations} | ETA≈{live.format_duration(eta)}"
                    )

                _log_context_summary(context_rows, decision_text, universe_name)

            elapsed = time.monotonic() - started_monotonic
            partial = _log_partial(aggregate_rows, state_index, elapsed)
            runs.update_one({"_id": run_id}, {"$set": {
                "partial_analysis": storage.mongo_value(partial),
                "completed_temporal_states": state_index,
                "updated_utc": pd.Timestamp.now(tz="UTC").to_pydatetime(),
            }})

        dataset = pd.DataFrame(aggregate_rows)
        readiness = live.readiness(dataset, len(CANDIDATES), len(UNIVERSES))
        final_analysis = live.partial_analysis(dataset, len(UNIVERSES))
        ended = pd.Timestamp.now(tz="UTC")
        summary = {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "experiment": EXPERIMENT_NAME,
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
            "years": int(pd.to_datetime(dataset["decision_date"]).dt.year.nunique()),
            "universes": int(dataset["universe_name"].nunique()),
            "candidates": int(dataset["candidate"].nunique()),
            "final_analysis": final_analysis,
            "readiness": readiness,
            "temporal_model_benchmark_ready": bool(readiness["ready"]),
            "target_definition": (
                "action_advantage_log = log(W_forced_candidate_then_same_policy / W_same_policy_without_force). "
                "Both arms use the same MongoDB market data and fold-aware frozen switch-margin schedule."
            ),
            "next_step_if_ready": (
                "Benchmark regularized linear, LightGBM, MLP and temporal neural models with and without "
                "temporal representation under identical chronological validation."
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

        runs.update_many({"experiment": EXPERIMENT_NAME, "is_latest": True}, {"$set": {"is_latest": False}})
        runs.update_one({"_id": run_id}, {"$set": {
            **storage.mongo_value(summary),
            "status": "completed",
            "is_latest": True,
            "updated_utc": ended.to_pydatetime(),
        }, "$unset": {"current_context": "", "eta_seconds_live": ""}})

        export_dir, zip_path = storage.export_result(
            project_root=base.PROJECT_ROOT,
            run_id=run_id,
            script_version=SCRIPT_VERSION,
            dataset=dataset,
            summary=summary,
            export_folder_name=EXPORT_FOLDER_NAME,
            export_zip_name=EXPORT_ZIP_NAME,
            collections=(RUNS_COLLECTION, OBSERVATIONS_COLLECTION, TRACE_RUNS_COLLECTION, TRACE_ROWS_COLLECTION),
        )
        live.console_log(f"[complete] Mongo run_id={run_id}")
        live.console_log(f"[complete] Export folder={export_dir}")
        live.console_log(f"[complete] ZIP={zip_path}")
        live.console_log(f"[complete] temporal_model_benchmark_ready={readiness['ready']}")
        sound_mode = frozen.sound._play_completion_sound()
        live.console_log(f"[complete] Completion sound mode={sound_mode}")
        return 0

    except Exception as exc:
        try:
            db = client[database_name]
            db[RUNS_COLLECTION].update_one({"_id": run_id}, {"$set": {
                "status": "failed",
                "updated_utc": pd.Timestamp.now(tz="UTC").to_pydatetime(),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }})
        except Exception:
            pass
        raise
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
