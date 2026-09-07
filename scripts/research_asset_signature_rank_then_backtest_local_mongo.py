from __future__ import annotations

"""Local-Mongo launcher for the rank-then-backtest research experiment.

This launcher preserves the complete experiment implemented by
``research_asset_signature_rank_then_backtest.py`` but makes candidate discovery
robust for local research databases.  The primary source remains the latest
Asset Discovery result.  If that result contains no symbol outside the selected
Strategy, the launcher falls back to every symbol currently cached in
``alpaca_market_bars`` and lets the main pipeline apply the Full Strategy History
integrity gate before ranking.

No Backtest result participates in candidate ranking.
"""

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
SRC_ROOT = PROJECT_ROOT / "src"
for root in (SCRIPTS_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import research_asset_signature_rank_then_backtest as pipeline
import research_asset_signature_leave_one_out as calibration
from research_asset_signature_pipeline_ranking import canonical_hash


def _candidate_source_with_local_mongo_fallback(
    db: Any,
    baseline_assets: list[str],
    explicit: str | None,
) -> tuple[list[str], dict[str, Any]]:
    """Resolve external candidates without requiring a surviving Discovery result.

    Resolution order:
    1. Explicit ``--candidate-symbols`` supplied by the user.
    2. Latest Asset Discovery result, exactly as the original pipeline does.
    3. Local MongoDB market-bar cache, excluding only the selected Strategy assets.

    The third mode does *not* declare every cached symbol eligible.  The main
    pipeline immediately validates each symbol against the complete XNYS Strategy
    window and only complete histories proceed to the mathematical ranking.
    """

    if explicit:
        return pipeline.candidate_source(db, baseline_assets, explicit)

    try:
        return pipeline.candidate_source(db, baseline_assets, None)
    except RuntimeError as exc:
        if "No new candidate symbols were found outside the selected Strategy" not in str(exc):
            raise

    baseline = {str(symbol).strip().upper() for symbol in baseline_assets if str(symbol).strip()}
    collection = db[calibration.ALPACA_MARKET_BARS_COLLECTION]
    raw = [
        str(symbol).strip().upper()
        for symbol in collection.distinct("symbol")
        if str(symbol).strip()
    ]
    unique = sorted(set(raw))
    external = [symbol for symbol in unique if symbol not in baseline]
    if not external:
        raise RuntimeError(
            "No external candidate symbols exist in the local MongoDB market-data cache. "
            "The cache currently contains only assets already present in the selected Strategy."
        )

    source = {
        "type": "local_mongodb_market_cache_fallback",
        "run_id": None,
        "source_result_count": len(unique),
        "candidate_symbols": external,
        "candidate_symbols_sha256": canonical_hash(external),
        "excluded_existing_strategy_symbols": [symbol for symbol in unique if symbol in baseline],
        "candidate_eligibility_note": (
            "These are cached external symbols only. Full Strategy History eligibility is "
            "evaluated immediately afterward by candidate_history_integrity.csv."
        ),
    }
    calibration._log(
        "Asset Discovery result has no external symbols; using local MongoDB market cache "
        f"as candidate source: cached={len(unique)}, external={len(external)}."
    )
    return external, source


def main() -> int:
    pipeline.candidate_source = _candidate_source_with_local_mongo_fallback
    return pipeline.main()


if __name__ == "__main__":
    raise SystemExit(main())
