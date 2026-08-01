from __future__ import annotations

import os
from pathlib import Path

API_VERSION = "1.12.17"
ENGINE_MODULE = "market_cycle_trader_api.engine.compound_rotation_backtest"
PACKAGE_DIR = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_DIR.parent
ENGINE_PATH = PACKAGE_DIR / "engine" / "compound_rotation_backtest.py"


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
