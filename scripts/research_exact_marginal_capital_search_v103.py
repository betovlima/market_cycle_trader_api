from __future__ import annotations

import json
import threading
import time
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import research_exact_marginal_capital_search as base  # noqa: E402
import research_exact_marginal_capital_search_v102 as previous  # noqa: E402
from market_cycle_trader_api.services import asset_discovery as discovery  # noqa: E402

SCRIPT_VERSION = "exact-marginal-capital-search-v1.0.3"
_ORIGINAL_CANDIDATE_SYMBOLS = base._candidate_symbols
_ORIGINAL_LOG = base._log
_IDENTITY_LOCK = threading.Lock()
_SELECTED_CANDIDATES: list[str] = []
_IDENTITY_CACHE: dict[str, dict[str, Any]] | None = None
_IDENTITY_ERROR: str | None = None


def _candidate_symbols_v103(
    collection: Any,
    identity: dict[str, str],
    baseline_assets: list[str],
    explicit: list[str] | None,
) -> list[str]:
    global _SELECTED_CANDIDATES, _IDENTITY_CACHE, _IDENTITY_ERROR
    values = _ORIGINAL_CANDIDATE_SYMBOLS(collection, identity, baseline_assets, explicit)
    with _IDENTITY_LOCK:
        _SELECTED_CANDIDATES = list(values)
        _IDENTITY_CACHE = None
        _IDENTITY_ERROR = None
    return values


def _identity_integrity_snapshot(
    db: Any,
    history_start: pd.Timestamp,
    snapshot_end: pd.Timestamp,
) -> dict[str, dict[str, Any]]:
    global _IDENTITY_CACHE, _IDENTITY_ERROR
    with _IDENTITY_LOCK:
        if _IDENTITY_CACHE is not None:
            return _IDENTITY_CACHE
        if _IDENTITY_ERROR is not None:
            raise RuntimeError(_IDENTITY_ERROR)
        try:
            metadata = discovery._discover_asset_metadata(db)
            _IDENTITY_CACHE = discovery._identity_integrity_for_symbols(
                db,
                list(_SELECTED_CANDIDATES),
                start_date=pd.Timestamp(history_start).date().isoformat(),
                end_date=pd.Timestamp(snapshot_end).date().isoformat(),
                asset_metadata=metadata,
            )
            return _IDENTITY_CACHE
        except Exception as exc:
            _IDENTITY_ERROR = f"ExactSearchIdentityIntegrityFailed: {str(exc)[:700]}"
            raise RuntimeError(_IDENTITY_ERROR) from exc


def _identity_fields(integrity: dict[str, Any]) -> dict[str, Any]:
    breaks = list(integrity.get("comparability_breaks") or [])
    return {
        "identity_integrity_status": str(integrity.get("status") or "unknown"),
        "identity_integrity_checked": bool(integrity.get("checked")),
        "identity_integrity_source": integrity.get("source"),
        "identity_integrity_reason": integrity.get("reason"),
        "identity_integrity_event_count": int(integrity.get("event_count") or 0),
        "identity_integrity_break_count": int(integrity.get("comparability_break_count") or 0),
        "identity_integrity_breaks_json": json.dumps(breaks, ensure_ascii=False, default=str),
    }


def _candidate_evaluation_v103(**kwargs: Any) -> dict[str, Any]:
    started = time.perf_counter()
    symbol = str(kwargs.get("symbol") or "").strip().upper()
    try:
        integrity_map = _identity_integrity_snapshot(
            kwargs.get("db"),
            pd.Timestamp(kwargs.get("history_start")),
            pd.Timestamp(kwargs.get("snapshot_end")),
        )
    except Exception as exc:
        return {
            "symbol": symbol,
            "evaluation_status": "failed",
            "economic_outcome": "failed",
            "preselector_raw_score": None,
            "preselector_error": None,
            "history_window_complete": False,
            "history_load_seconds": None,
            "preselector_seconds": None,
            "exact_replay_seconds": None,
            "total_seconds": float(time.perf_counter() - started),
            "error": str(exc)[:700],
        }

    integrity = dict(integrity_map.get(symbol) or {"status": "unknown", "checked": False})
    fields = _identity_fields(integrity)
    if str(integrity.get("status") or "").lower() != "passed":
        return {
            "symbol": symbol,
            "evaluation_status": "context_rejected",
            "economic_outcome": "context_rejected",
            "preselector_raw_score": None,
            "preselector_error": None,
            "history_window_complete": False,
            "history_load_seconds": None,
            "preselector_seconds": None,
            "exact_replay_seconds": None,
            "total_seconds": float(time.perf_counter() - started),
            "rejection_reason": "economic_identity_discontinuity",
            **fields,
        }

    result = previous._candidate_evaluation_v102(**kwargs)
    result.update(fields)
    return result


def _manifest_contract_v103(**kwargs: Any) -> dict[str, Any]:
    payload = dict(previous._manifest_contract_v102(**kwargs))
    payload.update(
        {
            "schema_version": 4,
            "script_version": SCRIPT_VERSION,
            "identity_integrity_precheck": True,
            "identity_integrity_policy": "same_alpaca_corporate_actions_guard_as_asset_discovery",
            "identity_integrity_before_candidate_history": True,
            "identity_integrity_before_exact_replay": True,
            "note": (
                "Exact judge benchmark. v1.0.3 restores the Asset Discovery economic-identity "
                "guard before any expensive candidate history/replay. Ticker reuse, symbol changes "
                "and hard corporate-action identity breaks are rejected before capital evaluation."
            ),
        }
    )
    return payload


def _log_v103(message: str) -> None:
    if message == "MongoDB read-only. No Alpaca request. Preselector never filters the exact judge.":
        _ORIGINAL_LOG(
            "MongoDB remains read-only; missing candidate history and identity checks may use Alpaca transiently. "
            "Preselector never filters the exact judge."
        )
        return
    _ORIGINAL_LOG(message)


def install_v103() -> None:
    previous.install_v102()
    base.SCRIPT_VERSION = SCRIPT_VERSION
    base._candidate_symbols = _candidate_symbols_v103
    base._candidate_evaluation = _candidate_evaluation_v103
    base._manifest_contract = _manifest_contract_v103
    base._log = _log_v103


if __name__ == "__main__":
    install_v103()
    raise SystemExit(base.main())
