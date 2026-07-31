from __future__ import annotations

import os
from pathlib import Path
from typing import Any

API_VERSION = "1.9.20"
ACTIVE_STRATEGY_MODE = "COMPOUND_ROTATION_SWING_1W"

PACKAGE_DIR = Path(__file__).resolve().parents[1]
ENGINE_PATH = PACKAGE_DIR / "engine" / "multi_asset_extrema_backtest.py"

STRATEGY_CATALOG: dict[str, dict[str, Any]] = {
    "COMPOUND_ROTATION_SWING_1W": {
        "mode": "COMPOUND_ROTATION_SWING_1W",
        "label": "Compound Capital Rotation — Swing",
        "status": "evaluation",
        "executable": True,
        "reason": (
            "Official shared-capital Swing configuration using XGBoost Utility H40. "
            "QR-DQN remains available as an experimental challenger."
        ),
    },
    "COMPOUND_ROTATION_DAY_TRADE_OPEN_CLOSE": {
        "mode": "COMPOUND_ROTATION_DAY_TRADE_OPEN_CLOSE",
        "label": "Compound Capital Rotation — Day Trade Open→Close",
        "status": "evaluation",
        "executable": True,
        "reason": (
            "One decision per trading session: select CASH or one asset before the "
            "regular-session open using completed prior-session data, enter at the open, "
            "and liquidate the position at the same-session close."
        ),
    },
}


def strategy_lifecycle(mode: str) -> dict[str, Any]:
    metadata = STRATEGY_CATALOG.get(str(mode))
    if metadata is not None:
        return dict(metadata)
    return {
        "mode": str(mode),
        "label": str(mode),
        "status": "unknown",
        "executable": False,
        "reason": "No lifecycle metadata is registered.",
    }


def cors_origins() -> list[str]:
    """Return explicit browser origins allowed to call the public API.

    Railway frontend URLs belong here because browser requests cannot use
    Railway's private ``*.railway.internal`` network.
    """
    defaults = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]
    raw = os.getenv("CORS_ORIGINS", "")
    extras = [item.strip().rstrip("/") for item in raw.split(",") if item.strip()]
    return list(dict.fromkeys(defaults + extras))
