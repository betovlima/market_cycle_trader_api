from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ABSOLUTE_UTILITY_CASH_GATE_MODE = "COMPOUND_ROTATION_SWING_ABSOLUTE_UTILITY_CASH_GATE"


@dataclass(frozen=True)
class AbsoluteUtilityCashGateEvaluation:
    best_score: float
    active_threshold: float
    accepted: bool
    hysteresis_market_hold: bool
    hysteresis_cash_block: bool


def absolute_utility_cash_gate_enabled(config: Any) -> bool:
    return str(getattr(config, "strategy_mode", "")) == ABSOLUTE_UTILITY_CASH_GATE_MODE


def evaluate_absolute_utility_cash_gate(
    config: Any,
    *,
    best_score: float,
    current_position: int,
) -> AbsoluteUtilityCashGateEvaluation:
    """Decide MARKET vs CASH directly from the Champion Top-1 utility.

    This gate intentionally does not fit a second predictive model.  It treats
    the Champion's absolute Top-1 utility as the opportunity signal and applies
    a two-threshold hysteresis rule:

    * while in CASH, enter only at/above the entry threshold;
    * while invested, remain invested until utility falls below the exit threshold.

    The thresholds are Strategy parameters and can therefore be explored by the
    existing probabilistic Model Tuning campaign without changing the protected
    LightGBM snapshot.
    """
    entry = float(getattr(config, "opportunity_utility_entry_threshold"))
    exit_ = float(getattr(config, "opportunity_utility_exit_threshold"))
    active = exit_ if int(current_position) > 0 else entry
    accepted = float(best_score) >= active
    inside_band = exit_ <= float(best_score) < entry
    return AbsoluteUtilityCashGateEvaluation(
        best_score=float(best_score),
        active_threshold=float(active),
        accepted=bool(accepted),
        hysteresis_market_hold=bool(int(current_position) > 0 and inside_band),
        hysteresis_cash_block=bool(int(current_position) <= 0 and inside_band),
    )
