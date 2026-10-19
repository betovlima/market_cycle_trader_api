"""Retired Live market-open recalibration entry point.

The Live Winner is trained ONLY with fully completed daily bars after close
(in live_winner_signal.prepare_live_cash_winner_signal). The next opening
performs broker/account validations and executes that persisted decision, never
training a second model or requesting one-minute candles.

This stub intentionally retains the legacy import path so old integrations
fail explicitly, without creating orders, modifying session state or
silently recalibrating the production strategy.
"""
from __future__ import annotations
from typing import Any


def recalibrate_live_cash_winner_signal(db: Any, session: str) -> dict[str, Any]:
    raise RuntimeError(
        "LiveIntradayRecalibrationDisabled: Winner is trained once after market close; "
        "only a failed post-close preparation may be retried before next market open."
    )
