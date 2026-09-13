from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from pymongo.uri_parser import parse_uri

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from market_cycle_trader_api.core.environment import load_project_environment  # noqa: E402
from market_cycle_trader_api.engine import research_challengers  # noqa: E402
from market_cycle_trader_api.services import asset_discovery as discovery  # noqa: E402

import research_contextual_signature_analysis as live  # noqa: E402
import research_contextual_signature_storage as storage  # noqa: E402

SCRIPT_VERSION = "contextual-marginal-signature-v1.0.17.4"
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

STRATEGY_PROFILES_COLLECTION = "strategy_profiles"
LOCAL_MONGO_HOSTS = {"localhost", "127.0.0.1", "::1", "mongo", "host.docker.internal"}
TOLERANCE = 1e-12
HEARTBEAT_SECONDS = 30.0
LEGACY_MODE = "COMPOUND_ROTATION_SWING_XGBOOST"

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
    "2020-08-03", "2020-11-02",
    "2021-02-01", "2021-05-03", "2021-08-02", "2021-11-01",
    "2022-02-01", "2022-05-02", "2022-08-01", "2022-11-01",
    "2023-02-01", "2023-05-01", "2023-08-01", "2023-11-01",
    "2024-02-01", "2024-05-01", "2024-08-01", "2024-11-01",
    "2025-02-03", "2025-05-01", "2025-08-01",
    "2026-02-02", "2026-06-01",
)

_BASELINE_SCHEDULE_BY_KEY: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
_CURRENT_CONTEXT: dict[str, Any] | None = None
_ORIGINAL_SIMULATE_EXACT = research_challengers._simulate_exact


def _log(message: str) -> None:
    live.console_log(message)


def _install_console_logging() -> None:
    return None


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


def _normalize_date(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError(f"Invalid date: {value}")
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp.normalize()


def _utc_timestamp(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _utc_index(values: Any) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(pd.DatetimeIndex(values), utc=True)).sort_values()


def _assert_local_mongo(uri: str, allow_remote: bool = False) -> None:
    if allow_remote:
        return
    parsed = parse_uri(uri)
    hosts = {str(host).strip().lower() for host, _ in parsed.get("nodelist") or []}
    if hosts and hosts.issubset(LOCAL_MONGO_HOSTS):
        return
    raise RuntimeError(
        "This research requires local MongoDB. "
        f"Resolved hosts={sorted(hosts) or ['unknown']}."
    )


def _strategy_document(db: Any, sequence: int, strategy_id: str | None) -> dict[str, Any]:
    query = {"_id": strategy_id} if strategy_id else {"strategy_sequence": int(sequence)}
    document = db[STRATEGY_PROFILES_COLLECTION].find_one(query)
    if document is None:
        raise RuntimeError(f"Strategy not found: {strategy_id or f'Strategy #{sequence}'}")
    return document


def _configuration(document: dict[str, Any]) -> dict[str, Any]:
    value = document.get("configuration")
    return dict(value) if isinstance(value, dict) else dict(document)


def _normalize_symbols(values: list[str] | tuple[str, ...] | None) -> list[str]:
    output: list[str] = []
    for value in values or []:
        symbol = str(value).strip().upper()
        if symbol and symbol not in output:
            output.append(symbol)
    return output


def _expected_sessions(start_date: pd.Timestamp, snapshot_end: pd.Timestamp) -> pd.DatetimeIndex:
    calendar = xcals.get_calendar("XNYS")
    first = pd.Timestamp(calendar.date_to_session(start_date, direction="next"))
    last = pd.Timestamp(calendar.date_to_session(snapshot_end, direction="previous"))
    return _utc_index(calendar.sessions_in_range(first, last)).normalize()


def _horizon_end(
    sessions: pd.DatetimeIndex,
    decision: pd.Timestamp,
    horizon_sessions: int,
) -> pd.Timestamp:
    ordered = _utc_index(sessions)
    decision_utc = _utc_timestamp(decision)
    position = int(ordered.searchsorted(decision_utc, side="left"))
    if position >= len(ordered) or pd.Timestamp(ordered[position]) != decision_utc:
        raise RuntimeError(f"Decision date is not an XNYS session: {decision_utc.date()}")
    end_position = position + max(1, int(horizon_sessions)) - 1
    if end_position >= len(ordered):
        raise RuntimeError(f"Horizon exceeds snapshot for {decision_utc.date()}")
    return pd.Timestamp(ordered[end_position])


def _snapshot_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for symbol in sorted(frames):
        frame = frames[symbol].copy().sort_index()
        frame = frame.loc[
            :,
            [column for column in ["open", "high", "low", "close", "volume"] if column in frame.columns],
        ]
        frame = frame.reset_index().rename(columns={frame.index.name or "index": "timestamp"})
        if "timestamp" not in frame.columns:
            frame = frame.rename(columns={frame.columns[0]: "timestamp"})
        frame.insert(0, "symbol", symbol)
        rows.append(frame)
    snapshot = pd.concat(rows, ignore_index=True)
    snapshot["timestamp"] = pd.to_datetime(snapshot["timestamp"], utc=True)
    return snapshot.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def _snapshot_hash(snapshot: pd.DataFrame) -> str:
    normalized = snapshot.copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True).astype(str)
    for column in ["open", "high", "low", "close", "volume"]:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce").round(12)
    payload = normalized.to_csv(
        index=False,
        na_rep="NaN",
        float_format="%.12g",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _campaign_symbols() -> list[str]:
    values = [symbol for universe in UNIVERSES for symbol in universe["assets"]]
    values.extend(CANDIDATES)
    return _normalize_symbols(values)


def _load_market_frames(
    config: Any,
    market_data: Any,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    symbols = _campaign_symbols()
    expected = _expected_sessions(_normalize_date(HISTORY_START), _normalize_date(SNAPSHOT_END))
    expected = pd.DatetimeIndex(pd.to_datetime(expected, utc=True)).normalize()
    frames: dict[str, pd.DataFrame] = {}
    provenance: list[dict[str, Any]] = []

    for index, symbol in enumerate(symbols, start=1):
        _log(f"[market {index}/{len(symbols)}] MongoDB Alpaca bars: {symbol}")
        frame = market_data.validate_and_clean_bars(
            market_data.load_market_bars(symbol, config),
            config,
        )
        observed = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True)).normalize().unique()
        missing = expected.difference(observed)
        if len(missing):
            sample = ", ".join(live.display_date(ts) for ts in missing[:5])
            raise RuntimeError(
                f"MongoDB history incomplete for {symbol}: missing={len(missing)} sample={sample}"
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


def _window_request(
    *,
    db: Any,
    config: Any,
    strategy_id: str,
    assets: list[str],
    reference_assets: list[str],
    candidate_assets: list[str],
    decision_session: pd.Timestamp,
    horizon_end: pd.Timestamp,
):
    return discovery._marginal_execution_request(
        db,
        config,
        {"id": strategy_id},
        config,
        horizon_end.date().isoformat(),
        assets=assets,
        reference_assets=reference_assets,
        candidate_assets=candidate_assets,
        analysis_start_date=decision_session.date().isoformat(),
        analysis_end_date=horizon_end.date().isoformat(),
    )


def _run_with_capture(
    frames: dict[str, pd.DataFrame],
    request: Any,
) -> tuple[dict[str, Any], pd.DatetimeIndex, list[Any]]:
    captured: list[Any] = []
    original = discovery.run_rotation_models

    def wrapper(*args: Any, **kwargs: Any):
        results = original(*args, **kwargs)
        captured.extend(list(results or []))
        return results

    discovery.run_rotation_models = wrapper
    try:
        metrics, sessions = discovery._run_rotation_replay(frames, request)
    finally:
        discovery.run_rotation_models = original

    if not captured:
        raise RuntimeError("Replay completed without captured RotationRunResult objects.")
    return metrics, sessions, captured


def _run_with_diagnostics(
    frames: dict[str, Any],
    request: Any,
) -> tuple[dict[str, Any], Any, list[Any]]:
    context = dict(_CURRENT_CONTEXT or {})
    candidate = str(context.get("candidate") or "").strip().upper()
    label = "baseline" if bool(context.get("is_baseline")) else f"candidate={candidate or 'unknown'}"
    started = time.monotonic()
    stop_event = threading.Event()

    def heartbeat() -> None:
        while not stop_event.wait(HEARTBEAT_SECONDS):
            _log(f"    [heartbeat] {label} still running; elapsed={time.monotonic() - started:.0f}s")

    thread = threading.Thread(
        target=heartbeat,
        name="contextual-signature-heartbeat",
        daemon=True,
    )
    thread.start()
    try:
        metrics, sessions, captured = _run_with_capture(frames, request)
    finally:
        stop_event.set()
        thread.join(timeout=1.0)

    elapsed = time.monotonic() - started
    ending_capital = discovery._finite_number(metrics.get("ending_capital")) if isinstance(metrics, dict) else None
    capital_text = f"; ending_capital={ending_capital:.6f}" if ending_capital is not None else ""
    _log(
        f"    [done] {label}; elapsed={elapsed:.1f}s; sessions={len(sessions) if sessions is not None else 0}; "
        f"backends={len(captured)}{capital_text}"
    )
    return metrics, sessions, captured


def _context_key(
    decision_session: Any,
    reference_assets: list[str],
) -> tuple[str, tuple[str, ...]]:
    decision = pd.Timestamp(decision_session)
    if decision.tzinfo is not None:
        decision = decision.tz_convert("UTC").tz_localize(None)
    return (
        decision.normalize().date().isoformat(),
        tuple(str(item).strip().upper() for item in reference_assets),
    )


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _extract_margin_schedule(results: list[Any]) -> dict[str, Any]:
    repetitions: list[dict[str, Any]] = []
    relevant: dict[tuple[int, int], dict[str, float]] = {}

    for result_index, result in enumerate(results, start=1):
        metrics = dict(getattr(result, "metrics", {}) or {})
        repetition_index = int(metrics.get("repetition_index") or result_index)
        fold_count = int(metrics.get("walk_forward_fold_count") or 0)
        if fold_count <= 0:
            raise RuntimeError(
                "Fold-aware frozen-margin protocol requires walk_forward_fold_count in replay metrics."
            )

        fold_rows = list(metrics.get("walk_forward_folds") or [])
        observed_by_fold: dict[int, dict[str, float]] = {}
        for row in fold_rows:
            if not isinstance(row, dict):
                continue
            fold_id_raw = row.get("fold_id")
            calibrated = _finite(row.get("calibrated_candidate_margin"))
            effective = _finite(row.get("effective_switch_margin"))
            if fold_id_raw is None or calibrated is None or effective is None:
                continue
            fold_id = int(fold_id_raw)
            if fold_id < 1 or fold_id > fold_count:
                raise RuntimeError(
                    f"Invalid fold id {fold_id} for walk_forward_fold_count={fold_count}."
                )
            previous = observed_by_fold.get(fold_id)
            current = {
                "calibrated_switch_margin": float(calibrated),
                "effective_switch_margin": float(effective),
            }
            if previous is not None and (
                abs(previous["calibrated_switch_margin"] - current["calibrated_switch_margin"]) > TOLERANCE
                or abs(previous["effective_switch_margin"] - current["effective_switch_margin"]) > TOLERANCE
            ):
                raise RuntimeError(
                    f"Conflicting baseline switch margins for repetition={repetition_index}, fold={fold_id}."
                )
            observed_by_fold[fold_id] = current
            relevant[(repetition_index, fold_id)] = current

        if not observed_by_fold:
            raise RuntimeError(
                "Fold-aware frozen-margin protocol found no fold-specific margin diagnostics."
            )

        repetitions.append(
            {
                "repetition_index": repetition_index,
                "fold_count": fold_count,
                "relevant_folds": observed_by_fold,
            }
        )

    repetitions.sort(key=lambda item: int(item["repetition_index"]))
    return {"repetitions": repetitions, "relevant": relevant}


class _FoldScheduledMarginCandidates(list[float]):
    def __init__(self, original: Iterable[float], schedule: dict[str, Any]) -> None:
        original_values = [float(value) for value in original]
        if not original_values:
            raise RuntimeError("rotation_switch_margin_candidates cannot be empty.")
        super().__init__(original_values)
        self._original = tuple(original_values)
        self._steps: list[dict[str, Any]] = []
        self._cursor = 0

        for repetition in list(schedule.get("repetitions") or []):
            repetition_index = int(repetition["repetition_index"])
            fold_count = int(repetition["fold_count"])
            relevant_folds = dict(repetition.get("relevant_folds") or {})
            for fold_id in range(1, fold_count + 1):
                observed = relevant_folds.get(fold_id)
                self._steps.append(
                    {
                        "repetition_index": repetition_index,
                        "fold_id": fold_id,
                        "forced_margin": (
                            float(observed["calibrated_switch_margin"])
                            if isinstance(observed, dict)
                            else None
                        ),
                    }
                )

    def __iter__(self):
        if self._cursor >= len(self._steps):
            raise RuntimeError(
                "Fold-aware frozen-margin schedule was exhausted before replay calibration completed."
            )
        step = self._steps[self._cursor]
        self._cursor += 1
        forced = step.get("forced_margin")
        values = (float(forced),) if forced is not None else self._original
        return iter(values)

    @property
    def consumed_steps(self) -> int:
        return int(self._cursor)

    @property
    def expected_steps(self) -> int:
        return int(len(self._steps))

    @property
    def steps(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._steps]


def _window_request_frozen(
    *,
    db: Any,
    config: Any,
    strategy_id: str,
    assets: list[str],
    reference_assets: list[str],
    candidate_assets: list[str],
    decision_session: Any,
    horizon_end: Any,
):
    global _CURRENT_CONTEXT
    key = _context_key(decision_session, reference_assets)
    candidate = str(candidate_assets[0]).strip().upper() if candidate_assets else None

    request = _window_request(
        db=db,
        config=config,
        strategy_id=strategy_id,
        assets=assets,
        reference_assets=reference_assets,
        candidate_assets=candidate_assets,
        decision_session=decision_session,
        horizon_end=horizon_end,
    )

    schedule = None
    if candidate is not None:
        schedule = _BASELINE_SCHEDULE_BY_KEY.get(key)
        if schedule is None:
            raise RuntimeError(
                "Fold-aware frozen-margin challenger requested before baseline schedule was captured: "
                f"decision={key[0]}, reference_assets={len(key[1])}, candidate={candidate}."
            )
        _log(
            f"    [freeze-folds] candidate={candidate}; baseline_relevant_folds="
            f"{len(schedule.get('relevant') or {})}"
        )

    _CURRENT_CONTEXT = {
        "key": key,
        "candidate": candidate,
        "is_baseline": candidate is None,
        "schedule": schedule,
    }
    return request


def _compare_relevant_schedules(
    expected: dict[str, Any],
    observed: dict[str, Any],
    candidate: str,
) -> None:
    expected_map = dict(expected.get("relevant") or {})
    observed_map = dict(observed.get("relevant") or {})
    missing = sorted(set(expected_map).difference(observed_map))
    if missing:
        raise RuntimeError(
            f"Frozen-margin challenger {candidate} is missing baseline OOS folds: {missing}."
        )

    for key, baseline_margin in expected_map.items():
        challenger = observed_map[key]
        if (
            abs(float(challenger["calibrated_switch_margin"]) - float(baseline_margin["calibrated_switch_margin"])) > TOLERANCE
            or abs(float(challenger["effective_switch_margin"]) - float(baseline_margin["effective_switch_margin"])) > TOLERANCE
        ):
            raise RuntimeError(
                "Fold-aware frozen switch-margin invariant failed for "
                f"candidate={candidate}, repetition={key[0]}, fold={key[1]}."
            )


def _run_with_frozen_margin(
    frames: dict[str, Any],
    request: Any,
) -> tuple[dict[str, Any], Any, list[Any]]:
    context = dict(_CURRENT_CONTEXT or {})
    scheduled_candidates: _FoldScheduledMarginCandidates | None = None

    if not bool(context.get("is_baseline")):
        schedule = context.get("schedule")
        if not isinstance(schedule, dict):
            raise RuntimeError("Fold-aware frozen-margin challenger has no baseline schedule.")
        scheduled_candidates = _FoldScheduledMarginCandidates(
            list(request.rotation_switch_margin_candidates),
            schedule,
        )
        request = request.model_copy(
            update={"rotation_switch_margin_candidates": scheduled_candidates}
        )
        forced_steps = [step for step in scheduled_candidates.steps if step.get("forced_margin") is not None]
        rendered = ", ".join(
            f"r{step['repetition_index']}/f{step['fold_id']}={float(step['forced_margin']):.6f}"
            for step in forced_steps
        )
        _log(f"    [freeze-fold-plan] {rendered}")

    metrics, sessions, captured = _run_with_diagnostics(frames, request)

    if scheduled_candidates is not None and (
        scheduled_candidates.consumed_steps != scheduled_candidates.expected_steps
    ):
        raise RuntimeError(
            "Fold-aware frozen-margin calibration count mismatch: "
            f"consumed={scheduled_candidates.consumed_steps}, "
            f"expected={scheduled_candidates.expected_steps}."
        )

    observed = _extract_margin_schedule(captured)
    key = context.get("key")
    if not isinstance(key, tuple):
        raise RuntimeError("Fold-aware frozen-margin replay completed without a request context key.")

    if bool(context.get("is_baseline")):
        _BASELINE_SCHEDULE_BY_KEY[key] = observed
        rendered = ", ".join(
            f"r{rep}/f{fold}={values['calibrated_switch_margin']:.6f}"
            for (rep, fold), values in sorted(dict(observed.get("relevant") or {}).items())
        )
        _log(f"    [freeze-folds] baseline captured; decision={key[0]}; {rendered}")
    else:
        candidate = str(context.get("candidate") or "UNKNOWN")
        expected = _BASELINE_SCHEDULE_BY_KEY[key]
        _compare_relevant_schedules(expected, observed, candidate)
        _log(
            f"    [freeze-folds-ok] candidate={candidate}; "
            f"matched_relevant_folds={len(expected.get('relevant') or {})}"
        )

    return metrics, sessions, captured


def _utc(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _forced_first_action_simulator(
    backend: str,
    policy: Callable[[pd.Timestamp, int, int], tuple[int, float]],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    decision_dates: pd.DatetimeIndex,
    config: Any,
    fee_calculator: Callable,
    slippage: Callable,
    decision_metadata: dict[pd.Timestamp, dict[str, Any]] | None = None,
    policy_decision_diagnostics: dict[pd.Timestamp, dict[str, Any]] | None = None,
    trade_callback: Callable[[dict[str, Any]], None] | None = None,
    *,
    model_label: str = "LightGBM Utility",
    method_line: str | None = None,
):
    context = dict(_CURRENT_CONTEXT or {})
    candidate = str(context.get("candidate") or "").strip().upper()
    if not candidate:
        return _ORIGINAL_SIMULATE_EXACT(
            backend,
            policy,
            frames,
            symbols,
            decision_dates,
            config,
            fee_calculator,
            slippage,
            decision_metadata=decision_metadata,
            policy_decision_diagnostics=policy_decision_diagnostics,
            trade_callback=trade_callback,
            model_label=model_label,
            method_line=method_line,
        )

    if str(config.strategy_mode) != LEGACY_MODE:
        raise RuntimeError(
            "Paired rollout is intentionally restricted to the stateless legacy rotation policy; "
            f"observed strategy_mode={config.strategy_mode}."
        )
    if candidate not in symbols:
        raise RuntimeError(f"Forced candidate {candidate} is not in simulator symbols.")
    if len(decision_dates) < 2:
        raise RuntimeError("Paired action rollout requires at least one decision-to-execution transition.")

    candidate_position = int(symbols.index(candidate) + 1)
    first_decision = _utc(decision_dates[0])
    forced = False
    policy_action_before_force: str | None = None

    def forced_policy(
        timestamp: pd.Timestamp,
        current_position: int,
        holding_days: int,
    ) -> tuple[int, float]:
        nonlocal forced, policy_action_before_force
        ts = _utc(timestamp)
        if not forced and ts == first_decision:
            normal_target, _normal_score = policy(timestamp, current_position, holding_days)
            policy_action_before_force = (
                "CASH"
                if int(normal_target) <= 0
                else str(symbols[int(normal_target) - 1])
            )
            forced = True

            diag = (policy_decision_diagnostics or {}).get(pd.Timestamp(timestamp))
            if isinstance(diag, dict):
                diag["research_forced_first_action"] = True
                diag["research_policy_action_before_force"] = policy_action_before_force
                diag["research_forced_action_asset"] = candidate
                diag["final_action_asset"] = candidate
                diag["final_action_score"] = None
                diag["decision_reason"] = "RESEARCH_FORCE_CANDIDATE_FIRST_ACTION"

            return candidate_position, 0.0
        return policy(timestamp, current_position, holding_days)

    result = _ORIGINAL_SIMULATE_EXACT(
        backend,
        forced_policy,
        frames,
        symbols,
        decision_dates,
        config,
        fee_calculator,
        slippage,
        decision_metadata=decision_metadata,
        policy_decision_diagnostics=policy_decision_diagnostics,
        trade_callback=trade_callback,
        model_label=model_label,
        method_line=method_line,
    )
    if not forced:
        raise RuntimeError(f"Forced first action was never applied for candidate={candidate}.")

    result.metrics["research_forced_first_action"] = True
    result.metrics["research_forced_action_asset"] = candidate
    result.metrics["research_policy_action_before_force"] = policy_action_before_force
    result.metrics["research_action_advantage_protocol"] = (
        "force candidate on the first decision, then resume the same frozen-margin legacy policy "
        "from the resulting simulator state"
    )
    return result


def _ending_capital(metrics: dict[str, Any], label: str) -> float:
    value = discovery._finite_number(metrics.get("ending_capital"))
    if value is None or value <= 0:
        raise RuntimeError(f"Invalid ending capital for {label}: {value}")
    return float(value)


def _first_selected_asset(captured: list[Any]) -> tuple[str, int]:
    for result in captured:
        predictions = getattr(result, "predictions", None)
        if isinstance(predictions, pd.DataFrame) and not predictions.empty:
            ordered = predictions.sort_index()
            return (
                str(ordered.iloc[0].get("selected_asset") or "CASH").strip().upper(),
                int(len(ordered)),
            )
    raise RuntimeError("Captured replay has no prediction rows.")


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
    request = _window_request_frozen(
        db=db,
        config=config,
        strategy_id=strategy_id,
        assets=[*reference_assets, *candidate_assets],
        reference_assets=reference_assets,
        candidate_assets=candidate_assets,
        decision_session=decision,
        horizon_end=horizon_end,
    )
    original = research_challengers._simulate_exact
    try:
        research_challengers._simulate_exact = (
            _forced_first_action_simulator if forced else _ORIGINAL_SIMULATE_EXACT
        )
        return _run_with_frozen_margin(frames, request)
    finally:
        research_challengers._simulate_exact = original


def _log_context_summary(
    rows: list[dict[str, Any]],
    decision_text: str,
    universe_name: str,
) -> None:
    frame = pd.DataFrame(rows)
    best = frame.loc[frame["action_advantage_log"].astype(float).idxmax()]
    best_y = float(best["action_advantage_log"])
    choice = str(best["candidate"]) if best_y > live.TOLERANCE else "NORMAL POLICY"
    pos = int((frame["action_advantage_log"] > live.TOLERANCE).sum())
    neg = int((frame["action_advantage_log"] < -live.TOLERANCE).sum())
    zero = int(len(frame) - pos - neg)
    _log(
        f"[context] {live.display_date(decision_text)} | {universe_name} | best={choice} | "
        f"best Y={max(0.0, best_y):+.5f} | candidates +{pos}/-{neg}/0={zero}"
    )


def _log_partial(
    rows: list[dict[str, Any]],
    state_index: int,
    elapsed: float,
) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    stats = live.partial_analysis(frame, len(UNIVERSES))
    progress = state_index / len(DECISION_DATES)
    eta = elapsed * (1.0 - progress) / progress if progress > 0 else None
    rho = stats["cross_universe_rank_spearman_mean"]
    rho_text = "n/a" if rho is None else f"{rho:+.3f}"
    _log(
        f"[partial] states={state_index}/{len(DECISION_DATES)} | obs={stats['rows']} | "
        f"+{stats['positive']} -{stats['negative']} 0={stats['zero']} | "
        f"mean Y={stats['mean_action_advantage_log']:+.5f} | "
        f"median Y={stats['median_action_advantage_log']:+.5f}"
    )
    _log(
        f"[partial] intervene={stats['intervention_preferred_contexts']} | "
        f"normal-policy={stats['normal_policy_preferred_contexts']} | "
        f"oracle mean={stats['oracle_mean_context_advantage_log']:+.5f} | "
        f"candidate mean leader={stats['candidate_mean_leader']} "
        f"({stats['candidate_mean_leader_value']:+.5f}) | cross-universe rho={rho_text}"
    )
    _log(
        f"[progress] elapsed={live.format_duration(elapsed)} | "
        f"ETA≈{live.format_duration(eta)}"
    )
    return stats


def _play_completion_sound(
    *,
    platform_name: str | None = None,
    beep: Callable[[int, int], None] | None = None,
    terminal_write: Callable[[str], object] | None = None,
) -> str:
    platform_name = platform_name or os.name
    if platform_name == "nt":
        try:
            if beep is None:
                import winsound
                beep = winsound.Beep
            for frequency, duration_ms in ((880, 160), (1175, 160), (1568, 280)):
                beep(frequency, duration_ms)
                time.sleep(0.04)
            return "windows_beep"
        except Exception:
            pass

    writer = terminal_write or sys.stdout.write
    try:
        writer("\a")
        if hasattr(sys.stdout, "flush"):
            sys.stdout.flush()
        return "terminal_bell"
    except Exception:
        return "silent"


class _SoundProxy:
    @staticmethod
    def _play_completion_sound() -> str:
        return _play_completion_sound()


_this_module = sys.modules[__name__]
base = _this_module
frozen = _this_module
paired = _this_module
sound = _SoundProxy()
