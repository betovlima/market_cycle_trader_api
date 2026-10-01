from __future__ import annotations

from typing import Any

import pandas as pd

from ..core.config import TCC_CONTROL_OPERATIONAL_MODE
from .live_lightgbm_signal import build_live_lightgbm_decision
from .tcc_control_operational_runtime import build_live_tcc_control_decision


def build_live_model_decision(
    bars_by_symbol: dict[str, pd.DataFrame],
    config: Any,
    *,
    model_family: str,
    current_asset: str | None,
    holding_sessions: int,
) -> Any:
    if model_family == "xgboost_utility":
        raise ValueError("XGBoost Utility was retired in API v8.0.0. The live Trader uses LightGBM Utility.")
    if (
        model_family == "lightgbm_utility"
        and str(getattr(config, "strategy_mode", "")) == TCC_CONTROL_OPERATIONAL_MODE
    ):
        return build_live_tcc_control_decision(
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
