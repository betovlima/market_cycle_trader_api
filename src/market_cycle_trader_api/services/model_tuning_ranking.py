from __future__ import annotations

import math
from typing import Any


def _finite(value: Any, default: float = -math.inf) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def candidate_economic_sort_key(candidate: dict[str, Any], *, champion_aware: bool) -> tuple[float, ...]:
    """Return an economically coherent descending ranking key.

    For Unified CARO campaigns, observed Champion-Gate passers come first, the
    Control is the reference tier, and non-beating challengers remain below the
    Control. Inside each tier we prioritize realized capital, then risk quality.
    The legacy compound score remains a final diagnostic tie-breaker only.
    """
    metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    if champion_aware:
        if candidate.get("champion_gate_passed") is True:
            tier = 2.0
        elif bool(candidate.get("is_control")):
            tier = 1.0
        else:
            tier = 0.0
    else:
        tier = 0.0

    return (
        tier,
        _finite(metrics.get("ending_capital")),
        _finite(metrics.get("sharpe")),
        _finite(metrics.get("maximum_drawdown")),  # less negative is better
        _finite(metrics.get("worst_fold_return")),
        _finite(metrics.get("risk_adjusted_compound_score")),
    )
