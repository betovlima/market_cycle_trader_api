from __future__ import annotations

import os
from pathlib import Path
from typing import Any

API_VERSION = "6.10.2"
ENGINE_MODULE = "market_cycle_trader_api.engine.compound_rotation_backtest"
PACKAGE_DIR = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_DIR.parent
ENGINE_PATH = PACKAGE_DIR / "engine" / "compound_rotation_backtest.py"

STRATEGY_LABELS = {
    "COMPOUND_ROTATION_SWING_XGBOOST": "Compound Capital Rotation — XGBoost",
    "COMPOUND_ROTATION_SWING_RISK_OFF": "Compound Capital Rotation — Explicit Risk-Off",
    "COMPOUND_ROTATION_SWING_SELECTIVE": "Compound Capital Rotation — Selective Opportunity",
    "COMPOUND_ROTATION_SWING_OPPORTUNITY_CASH_GATE": "Compound Capital Rotation — Opportunity Cash Gate v2",
    "COMPOUND_ROTATION_SWING_ABSOLUTE_UTILITY_CASH_GATE": "Compound Capital Rotation — Absolute Utility Cash Gate",
    "COMPOUND_ROTATION_SWING_OPTIMIZED_ALLOCATION": "Compound Capital Rotation — Optimized Allocation",
    "COMPOUND_ROTATION_SWING_CONCENTRATED_ALLOCATION": "Compound Capital Rotation — Concentrated Optimal Allocation",
    "COMPOUND_ROTATION_SWING_COMPOUND_RISK_OVERLAY": "Compound Capital Rotation — Compound Risk Overlay",
}
SWING_STRATEGY_MODES = frozenset(STRATEGY_LABELS)
RESEARCH_ONLY_SWING_STRATEGY_MODES = frozenset({
    "COMPOUND_ROTATION_SWING_OPPORTUNITY_CASH_GATE",
    "COMPOUND_ROTATION_SWING_ABSOLUTE_UTILITY_CASH_GATE",
    "COMPOUND_ROTATION_SWING_OPTIMIZED_ALLOCATION",
    "COMPOUND_ROTATION_SWING_CONCENTRATED_ALLOCATION",
    "COMPOUND_ROTATION_SWING_COMPOUND_RISK_OVERLAY",
})


def strategy_lifecycle(mode: str) -> dict[str, Any]:
    normalized = str(mode)
    return {
        "mode": normalized,
        "label": STRATEGY_LABELS.get(normalized, normalized),
    }


def cors_origins() -> list[str]:
    raw = str(os.getenv("CORS_ORIGINS") or "").strip()
    if raw:
        return list(
            dict.fromkeys(
                item.strip().rstrip("/")
                for item in raw.split(",")
                if item.strip()
            )
        )
    return [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]
