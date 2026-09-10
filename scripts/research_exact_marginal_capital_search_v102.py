from __future__ import annotations

from pathlib import Path
import sys
import threading
from typing import Any

import exchange_calendars as xcals
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import research_exact_marginal_capital_search as base  # noqa: E402
from market_cycle_trader_api.engine import market_data as market_data_engine  # noqa: E402
from market_cycle_trader_api.services import asset_discovery as discovery  # noqa: E402

SCRIPT_VERSION = "exact-marginal-capital-search-v1.0.2"
_ORIGINAL_LOAD_CANDIDATE_FRAME = base._load_candidate_frame
_ORIGINAL_CANDIDATE_EVALUATION = base._candidate_evaluation
_ORIGINAL_MANIFEST_CONTRACT = base._manifest_contract
_STATE = threading.local()


def _transient_full_history(
    db: Any,
    symbol: str,
    history_start: pd.Timestamp,
    snapshot_end: pd.Timestamp,
    config: Any,
    required_sessions: pd.DatetimeIndex,
):
    """Download the candidate history explicitly with validated credentials.

    v1.0.1 delegated this fallback to Asset Discovery's private helper.  In the
    standalone benchmark that produced an opaque ``NoneType`` failure before any
    network request on some local environments.  v1.0.2 keeps the same legacy
    Discovery semantics but makes every dependency explicit: credentials, chunked
    Alpaca download, cleaning and baseline-calendar coverage validation.
    """
    credentials = discovery.get_alpaca_credentials(db)
    if not isinstance(credentials, dict):
        raise RuntimeError(
            "ExactSearchCredentialContractError: get_alpaca_credentials returned no mapping."
        )
    api_key_id = str(credentials.get("api_key_id") or "").strip()
    secret_key = str(credentials.get("secret_key") or "").strip()
    if not api_key_id or not secret_key:
        raise RuntimeError(
            "ExactSearchCredentialContractError: Alpaca credentials are incomplete."
        )

    candidate_config = config.model_copy(
        update={
            "end_date": pd.Timestamp(snapshot_end).date().isoformat(),
            "mongo_cache_enabled": False,
            "market_data_history_backfill_enabled": False,
        }
    )
    requested_start = market_data_engine._utc_timestamp(history_start)
    normalized_end = market_data_engine.normalize_end_date(candidate_config.end_date)
    if normalized_end is None:
        raise RuntimeError("ExactSearchCandidateEndDateMissing")

    calendar = xcals.get_calendar("XNYS")
    end_session = pd.Timestamp(
        calendar.date_to_session(pd.Timestamp(normalized_end), direction="previous")
    )
    requested_end = pd.Timestamp(calendar.session_close(end_session)).tz_convert("UTC")
    if requested_end <= requested_start:
        raise RuntimeError("insufficient_history")

    chunk_days = (
        market_data_engine.ALPACA_DAILY_HISTORY_CHUNK_DAYS
        if candidate_config.timeframe == "1Day"
        else market_data_engine.ALPACA_INTRADAY_HISTORY_CHUNK_DAYS
    )
    frames: list[pd.DataFrame] = []
    cursor = requested_start
    while cursor < requested_end:
        chunk_end = min(cursor + pd.Timedelta(days=chunk_days), requested_end)
        frame = discovery.download_stock_bars(
            api_key_id=api_key_id,
            secret_key=secret_key,
            symbol=symbol,
            timeframe=candidate_config.timeframe,
            start=cursor.to_pydatetime(),
            end=chunk_end.to_pydatetime(),
            feed=candidate_config.alpaca_historical_feed,
            adjustment=candidate_config.alpaca_adjustment,
        )
        if frame is not None and not frame.empty:
            frames.append(frame)
        cursor = chunk_end

    if not frames:
        raise RuntimeError("insufficient_history")

    combined = pd.concat(frames).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = market_data_engine.trim_downloaded_range(
        combined,
        candidate_config.start_date,
        candidate_config.end_date,
        candidate_config.timeframe,
    )
    try:
        cleaned = discovery.validate_and_clean_bars(combined, candidate_config)
    except ValueError as exc:
        raise RuntimeError("insufficient_history") from exc
    coverage = discovery._history_coverage_against_baseline(
        symbol,
        cleaned,
        candidate_config,
        required_sessions,
    )
    cleaned.attrs["exact_candidate_history_source"] = "alpaca_transient_full_history_v102"
    return cleaned, coverage


def _load_candidate_frame_v102(
    collection: Any,
    symbol: str,
    identity: dict[str, str],
    history_start: pd.Timestamp,
    snapshot_end: pd.Timestamp,
    config: Any,
    required_sessions: pd.DatetimeIndex,
):
    try:
        frame, coverage = _ORIGINAL_LOAD_CANDIDATE_FRAME(
            collection,
            symbol,
            identity,
            history_start,
            snapshot_end,
            config,
            required_sessions,
        )
        frame.attrs["exact_candidate_history_source"] = "local_mongodb_cache"
        return frame, coverage
    except (RuntimeError, ValueError):
        db = getattr(_STATE, "db", None)
        if db is None:
            raise RuntimeError(
                "ExactSearchInternalContractError: candidate fallback has no active database context."
            )
        return _transient_full_history(
            db,
            symbol,
            history_start,
            snapshot_end,
            config,
            required_sessions,
        )


def _candidate_evaluation_v102(**kwargs: Any) -> dict[str, Any]:
    previous = getattr(_STATE, "db", None)
    _STATE.db = kwargs.get("db")
    try:
        return _ORIGINAL_CANDIDATE_EVALUATION(**kwargs)
    finally:
        _STATE.db = previous


def _manifest_contract_v102(**kwargs: Any) -> dict[str, Any]:
    payload = dict(_ORIGINAL_MANIFEST_CONTRACT(**kwargs))
    payload.update(
        {
            "schema_version": 3,
            "script_version": SCRIPT_VERSION,
            "candidate_source": "explicit_symbols_or_local_mongodb_external_symbols",
            "market_data_source": (
                "baseline_local_mongodb; candidate_local_cache_then_explicit_chunked_alpaca_fallback"
            ),
            "alpaca_network_used": "only_for_candidate_history_missing_from_local_cache",
            "candidate_history_persistence": "none",
            "legacy_discovery_history_parity": True,
            "candidate_fallback_implementation": "explicit_credentials_chunked_alpaca_v102",
            "note": (
                "Exact judge benchmark. The preselector never filters evaluation. "
                "v1.0.2 removes the opaque delegated fallback used by v1.0.1 and downloads "
                "missing candidate history explicitly with validated credentials before exact replay."
            ),
        }
    )
    return payload


def install_v102() -> None:
    base.SCRIPT_VERSION = SCRIPT_VERSION
    base._load_candidate_frame = _load_candidate_frame_v102
    base._candidate_evaluation = _candidate_evaluation_v102
    base._manifest_contract = _manifest_contract_v102


if __name__ == "__main__":
    install_v102()
    raise SystemExit(base.main())
