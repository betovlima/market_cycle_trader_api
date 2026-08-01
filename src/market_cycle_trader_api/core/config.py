from __future__ import annotations

import os
from pathlib import Path
from typing import Any

API_VERSION = "1.10.3"
ACTIVE_STRATEGY_MODE = "COMPOUND_ROTATION_SWING_XGBOOST"
ENGINE_MODULE = "market_cycle_trader_api.engine.compound_rotation_backtest"
PACKAGE_DIR = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_DIR.parent
ENGINE_PATH = PACKAGE_DIR / "engine" / "compound_rotation_backtest.py"

STRATEGY_CATALOG: dict[str, dict[str, Any]] = {
    "COMPOUND_ROTATION_SWING_XGBOOST": {
        "mode": "COMPOUND_ROTATION_SWING_XGBOOST",
        "label": "Compound Capital Rotation — XGBoost",
        "status": "official",
        "executable": True,
        "reason": "Official Swing strategy with XGBoost Utility and the validated H40 configuration.",
    },
    "COMPOUND_ROTATION_SWING_QRDQN": {
        "mode": "COMPOUND_ROTATION_SWING_QRDQN",
        "label": "Compound Capital Rotation — QR-DQN",
        "status": "research",
        "executable": True,
        "reason": "QR-DQN research strategy using QR1 five-step returns as the current controlled challenger.",
    },
    "COMPOUND_ROTATION_DAY_TRADE_OPEN_CLOSE": {
        "mode": "COMPOUND_ROTATION_DAY_TRADE_OPEN_CLOSE",
        "label": "Compound Capital Rotation — Day Trade Open→Close",
        "status": "evaluation",
        "executable": True,
        "reason": "One pre-open decision per session, open execution and mandatory same-session close exit.",
    },
}

SWING_STRATEGY_MODES = {"COMPOUND_ROTATION_SWING_XGBOOST", "COMPOUND_ROTATION_SWING_QRDQN"}


def strategy_lifecycle(mode: str) -> dict[str, Any]:
    metadata = STRATEGY_CATALOG.get(str(mode))
    if metadata is not None:
        return dict(metadata)
    return {"mode": str(mode), "label": str(mode), "status": "unknown", "executable": False, "reason": "No lifecycle metadata is registered."}


def cors_origins() -> list[str]:
    defaults = ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8000", "http://127.0.0.1:8000"]
    raw = os.getenv("CORS_ORIGINS", "")
    extras = [item.strip().rstrip("/") for item in raw.split(",") if item.strip()]
    return list(dict.fromkeys(defaults + extras))
