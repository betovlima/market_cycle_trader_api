# market_cycle_trader_api

FastAPI backend for Market Cycle Trader v1.9.20.

## Local

```powershell
python -m pip install -r requirements.txt
.\run_local.ps1
```

or:

```powershell
python -m uvicorn market_cycle_trader_api.main:app --app-dir src --host 127.0.0.1 --port 8000 --reload
```

Swagger: `http://127.0.0.1:8000/docs`

## Package responsibilities

- `api/routers`: HTTP transport only.
- `schemas`: Pydantic contracts and validation.
- `services`: application use-cases/orchestration.
- `core`: runtime and application configuration.
- `infrastructure`: MongoDB and market-data adapters.
- `engine`: isolated quantitative/ML core.

## Railway

Use root directory `/market_cycle_trader_api` and config path `/market_cycle_trader_api/railway.toml`.

Required production variables:

```text
MONGO_URL
MONGO_DATABASE
CORS_ORIGINS
```
