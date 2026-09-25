"""Pure execution-cost helpers used by the standalone TCC engine."""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def round_fee_to_cent(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        return 0.0
    return math.ceil((value - 1e-12) * 100.0) / 100.0


def calculate_reference_fees(
    side: str,
    quantity: float,
    price: float,
    config: Any,
) -> dict[str, float]:
    if quantity <= 0 or price <= 0:
        return {
            "commission_fee": 0.0,
            "sec_fee": 0.0,
            "taf_fee": 0.0,
            "cat_fee": 0.0,
            "total_fee": 0.0,
        }
    normalized_side = side.upper()
    trade_value = quantity * price
    commission = round_fee_to_cent(trade_value * config.commission_rate)
    cat = round_fee_to_cent(quantity * config.cat_fee_per_share)
    sec = 0.0
    taf = 0.0
    if normalized_side == "SELL":
        sec = round_fee_to_cent(trade_value * config.sec_fee_rate)
        taf = round_fee_to_cent(
            min(quantity * config.taf_fee_per_share, config.taf_fee_cap)
        )
    elif normalized_side != "BUY":
        raise ValueError(f"Unsupported side: {side}")
    return {
        "commission_fee": commission,
        "sec_fee": sec,
        "taf_fee": taf,
        "cat_fee": cat,
        "total_fee": commission + sec + taf + cat,
    }


def apply_slippage(price: float, side: str, config: Any) -> float:
    adjustment = config.slippage_bps / 10_000
    return price * (1 + adjustment if side == "BUY" else 1 - adjustment)
