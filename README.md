# Market Cycle Trader API v1.13.4

MongoDB-backed multi-horizon XGBoost rotation backtest and Alpaca Paper API.

## Active strategy engine

- Strategy mode: `COMPOUND_ROTATION_SWING_XGBOOST`
- Engine module: `market_cycle_trader_api.engine.compound_rotation_backtest`
- Daily decisions with next-session execution
- Multi-horizon targets: 5, 10, 20, 40, and 60 trading sessions
- Protected MongoDB strategy administration
- Alpaca credentials read only from server environment variables

## Main routes

- `/api/jobs` — queue, inspect, and read backtests.
- `/api/paper-market` — inspect the paper portfolio and arm/cancel next-session runs.
- `/api/admin/parameters` — inspect and install the initial MongoDB parameter documents.
- `/api/admin/setup` — bind the Alpaca Paper account and initialize paper state.
- `/api/admin/strategy-configuration` — read, patch, replace, reset, restore, and install the protected winner file.
- `/api/health/live` and `/api/health/ready` — liveness and strict readiness.

## Required server variables

```text
MONGO_URL
MONGO_DATABASE
ALPACA_API_KEY_ID
ALPACA_SECRET_KEY
PARAMETER_BOOTSTRAP_API_TOKEN
PAPER_MARKET_API_TOKEN
CORS_ORIGINS
```

The MongoDB connection string, database name, Alpaca credentials, and API tokens are never strategy parameters and are never written by the application to MongoDB.

## Fresh database

A new or cleared database is intentionally empty. Install the canonical parameter documents through the protected API before starting a backtest:

1. `POST /api/admin/strategy-configuration/winner/install`
2. `GET /api/admin/parameters/status`
3. `POST /api/admin/parameters/bootstrap` when non-strategy documents are missing
4. `GET /api/admin/strategy-configuration`
5. `POST /api/admin/setup/initialize`
6. `POST /api/jobs`

The exact order and payload file names are documented in `script/EXECUTION.md`.

## Locked execution period

`POST /api/jobs` accepts no body. The active `winner-v1.13.2.json` document stored in MongoDB supplies the historical start and end dates.

## Additive dashboard API

The redesigned frontend uses two read-only endpoints:

```http
GET /api/dashboard/summary?limit=12
GET /api/dashboard/jobs/{job_id}
```

They expose only strategy-neutral operational metrics. Existing API endpoints and engine behavior remain unchanged.
