from __future__ import annotations

import os
from pathlib import Path
from typing import Any

API_VERSION = "10.8.15"
ENGINE_MODULE = "market_cycle_trader_api.engine.compound_rotation_backtest"
PACKAGE_DIR = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_DIR.parent
ENGINE_PATH = PACKAGE_DIR / "engine" / "compound_rotation_backtest.py"
