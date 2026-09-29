"""Read-only data/calendar contract for a future TCC Control live runtime.

This module intentionally does not select a Strategy, fit a live model, prepare
an Alpaca order, or change the current Winner. It uses the frozen scientific
v1.0.6 panel builder, with an explicit completed-session boundary, to make
parity requirements testable before connecting the operational executor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..tcc_v106_reference.capital_rotation import (
    ROTATION_FEATURES as SCIENTIFIC_FEATURES,
    prepare_rotation_panel as prepare_scientific_panel,
)
from ..tcc_v106_reference.config import (
    CONFIG as FROZEN_CONFIG,
    REFERENCE_ASSETS,
    build_control_config,
)
from .capital_rotation import ROTATION_FEATURES as LIVE_FEATURES


@dataclass(frozen=True)
class ControlPanelAudit:
    requested_assets: int
    available_assets: int
    required_reference_assets: tuple[str, ...]
    calendar_sessions: int
    first_session: str
    last_session: str


def prepare_operational_control_panel(
    bars_by_symbol: dict[str, pd.DataFrame],
    *,
    completed_session: str,
    config: Any | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex, ControlPanelAudit]:
    """Build an isolated Control panel without silently changing its calendar.

    `completed_session` is a New York trading-session DATE (YYYY-MM-DD);
    data must contain completed bars only. This function does not fetch or
    normalize Alpaca data. It must only receive upstream audited split-adjusted
    OHLCV, and is not by itself proof of end-to-end data/model parity.
    """
    if list(LIVE_FEATURES) != list(SCIENTIFIC_FEATURES):
        raise ValueError(
            "The scientific and live LightGBM feature lists differ. "
            "Do not enable Control operational execution."
        )
    try:
        cutoff = pd.Timestamp(completed_session)
    except (TypeError, ValueError) as exc:
        raise ValueError("completed_session must be a YYYY-MM-DD trading-session date.") from exc
    if cutoff.tzinfo is not None or cutoff.strftime("%Y-%m-%d") != completed_session:
        raise ValueError("completed_session must be a YYYY-MM-DD trading-session date.")

    selected_config = build_control_config(config or FROZEN_CONFIG)
    configured_assets = tuple(str(symbol) for symbol in selected_config.assets)
    anchors = tuple(str(symbol) for symbol in REFERENCE_ASSETS)
    missing_anchors = [symbol for symbol in anchors if symbol not in bars_by_symbol]
    if missing_anchors:
        raise ValueError(
            "Frozen Control reference calendar is incomplete; missing assets: "
            + ", ".join(missing_anchors)
        )
    if not bars_by_symbol:
        raise ValueError("Control panel needs completed OHLCV input.")

    for symbol, frame in bars_by_symbol.items():
        if frame is None or frame.empty:
            raise ValueError(f"Missing completed OHLCV rows for {symbol}.")
        dates = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True))
        if dates.has_duplicates or not dates.is_monotonic_increasing:
            raise ValueError(f"Invalid session ordering or duplicates for {symbol}.")
        # Alpaca daily candles are stamped at session start in UTC, which
        # can be 04:00/05:00Z, not a date-only midnight value.
        if (dates.date > cutoff.date()).any():
            raise ValueError(f"Future daily bars supplied for {symbol} after {completed_session}.")

    aligned, common = prepare_scientific_panel(bars_by_symbol, selected_config)
    if common.empty or common[-1].date() != cutoff.date():
        raise ValueError(
            "The frozen anchor calendar is not current through the requested "
            f"completed session {completed_session}."
        )
    audit = ControlPanelAudit(
        requested_assets=len(configured_assets),
        available_assets=len(aligned),
        required_reference_assets=anchors,
        calendar_sessions=len(common),
        first_session=common[0].date().isoformat(),
        last_session=common[-1].date().isoformat(),
    )
    return aligned, common, audit
