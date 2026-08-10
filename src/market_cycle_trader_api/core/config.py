from __future__ import annotations

import os
from pathlib import Path
from typing import Any

API_VERSION = "1.13.36"
ENGINE_MODULE = "market_cycle_trader_api.engine.compound_rotation_backtest"
PACKAGE_DIR = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_DIR.parent
ENGINE_PATH = PACKAGE_DIR / "engine" / "compound_rotation_backtest.py"

STRATEGY_LABELS = {
    "COMPOUND_ROTATION_SWING_XGBOOST": "Compound Capital Rotation — XGBoost",
}
SWING_STRATEGY_MODES = frozenset(STRATEGY_LABELS)


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
