from __future__ import annotations

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
from market_cycle_trader_api.services import asset_discovery as discovery  # noqa: E402

SCRIPT_VERSION = "exact-marginal-capital-search-v1.0.1"
_ORIGINAL_LOAD_CANDIDATE_FRAME = base._load_candidate_frame
_ORIGINAL_MANIFEST_CONTRACT = base._manifest_contract


def _load_candidate_frame_with_transient_fallback(
    collection: Any,
    symbol: str,
    identity: dict[str, str],
    history_start: pd.Timestamp,
    snapshot_end: pd.Timestamp,
    config: Any,
    required_sessions: pd.DatetimeIndex,
):
    """Prefer the local cache, but reproduce legacy Discovery when the asset is absent.

    Legacy Asset Discovery did not require rejected/external candidates to already exist
    in MongoDB.  It downloaded the candidate's complete historical window transiently,
    validated it against the baseline calendar, evaluated the Strategy, and discarded
    the market frame unless the user later selected the asset for persistence.
    """
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
        frame, coverage = discovery._candidate_history_coverage(
            None,
            symbol,
            config,
            snapshot_end,
            required_sessions,
        )
        frame.attrs["exact_candidate_history_source"] = "alpaca_transient_full_history"
        return frame, coverage


def _manifest_contract_v101(**kwargs: Any) -> dict[str, Any]:
    payload = dict(_ORIGINAL_MANIFEST_CONTRACT(**kwargs))
    payload.update(
        {
            "schema_version": 2,
            "script_version": SCRIPT_VERSION,
            "candidate_source": (
                "explicit_symbols_or_local_mongodb_external_symbols"
            ),
            "market_data_source": (
                "baseline_local_mongodb; candidate_local_cache_then_transient_alpaca_fallback"
            ),
            "alpaca_network_used": "only_when_candidate_full_history_is_not_available_in_local_cache",
            "candidate_history_persistence": "none",
            "legacy_discovery_history_parity": True,
            "note": (
                "Exact judge benchmark. The preselector never filters evaluation. "
                "External candidates absent from MongoDB are fetched transiently exactly so "
                "historical Discovery controls can be reproduced. Candidate frames are not persisted."
            ),
        }
    )
    return payload


def install_v101() -> None:
    base.SCRIPT_VERSION = SCRIPT_VERSION
    base._load_candidate_frame = _load_candidate_frame_with_transient_fallback
    base._manifest_contract = _manifest_contract_v101


if __name__ == "__main__":
    install_v101()
    raise SystemExit(base.main())
