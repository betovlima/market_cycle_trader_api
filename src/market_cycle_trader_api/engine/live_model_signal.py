from __future__ import annotations

from typing import Any

import pandas as pd

from .live_lightgbm_signal import build_live_lightgbm_decision
from .live_xgboost_signal import build_live_xgboost_decision


def build_live_model_decision(
    bars_by_symbol: dict[str, pd.DataFrame],
    config: Any,
    *,
    model_family: str,
    current_asset: str | None,
    holding_sessions: int,
) -> Any:
    if model_family == "xgboost_utility":
        return build_live_xgboost_decision(
            bars_by_symbol,
            config,
            current_asset=current_asset,
            holding_sessions=holding_sessions,
        )
    if model_family == "lightgbm_utility":
        return build_live_lightgbm_decision(
            bars_by_symbol,
            config,
            current_asset=current_asset,
            holding_sessions=holding_sessions,
        )
    raise ValueError(
        f"Trader Winner model {model_family!r} does not have a protected live execution engine."
    )
