"""Live Winner intraday data follows the Trading (Paper) scoring contract.

The only permitted distinction is the broker order/account boundary. Market
data use the SAME credentials, routine, feed fallback, as-of cutoff and OHLCV
aggregation as Trading. No Paper position, plan or order is ever referenced.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from ..engine.operational_market_snapshot import append_current_session_intraday

# Keep prior safe error taxonomy compatible with Live operations logs.
_PAPER_COVERAGE_ERRORS = (
    "CurrentSessionIntradayDataMissing:",
    "CurrentSessionIntradayCoverageUnavailable:",
    "CurrentSessionIntradayDelayedSIPNotReady:",
    "CurrentSessionIntradayNotStarted:",
)


def load_live_intraday_session(
    historical: dict[str, pd.DataFrame],
    config: Any,
    *,
    session: str,
    now: datetime | pd.Timestamp,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Score the current session with the verified Trading pipeline.

    Calls the SAME engine function the Trading workspace uses (including
    its uniform IEX/SIP/20-minute delayed SIP policies and market-data
    credentials). Live account identity and real orders remain independent
    of Trading; nothing here can submit orders.

    Preserve the Live analysis_mode discriminator required by the real-order
    executor without changing the actual bars or decision inputs.
    """
    if not historical:
        raise RuntimeError("LiveIntradayCoverageNotReady: no historical universe.")
    try:
        result, info = append_current_session_intraday(
            historical, config, session=session, now=now,
        )
    except RuntimeError as exc:
        # Normalize *only* the known Trading coverage errors. Every other
        # exception preserves its type and exact origin for diagnosis.
        if str(exc).startswith(_PAPER_COVERAGE_ERRORS):
            raise RuntimeError(
                f"LiveIntradayCoverageNotReady: {exc}"
            ) from exc
        raise
    return result, {
        **info,
        "analysis_mode": "live_current_session_intraday",
        "data_pipeline": "trading_intraday_parity",
    }
