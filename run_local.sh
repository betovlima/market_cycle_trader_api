#!/usr/bin/env sh
set -eu
python -m uvicorn market_cycle_trader_api.main:app --app-dir src --host 127.0.0.1 --port 8000 --reload
