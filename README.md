# market_cycle_trader_api

FastAPI backend for Market Cycle Trader v1.10.3.

## Local

```powershell
python -m pip install -r requirements.txt
.\run_local.ps1
```

Alternative:

```powershell
python -m uvicorn market_cycle_trader_api.main:app --app-dir src --host 127.0.0.1 --port 8000 --reload
```

Swagger: `http://127.0.0.1:8000/docs`

## Packages

- `api/routers` — HTTP endpoints.
- `schemas` — Pydantic contracts and validation.
- `services` — application orchestration and result construction.
- `core` — application/runtime configuration.
- `infrastructure` — MongoDB and market-data adapters.
- `engine/compound_rotation_backtest.py` — focused backtest entry point.
- `engine/capital_rotation.py` — Swing XGBoost/QR-DQN engine.
- `engine/day_trade_open_close.py` — Day Trade Open→Close engine.
- `engine/market_data.py` — active market-data loading and cache path.

The API contains no legacy extrema/Fibonacci strategy implementation.

## Railway

Use root directory `/market_cycle_trader_api` and config path `/market_cycle_trader_api/railway.toml`.

Required production variables:

```text
MONGO_URL
MONGO_DATABASE
CORS_ORIGINS
```
