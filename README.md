# Market Cycle Trader API v1.12.26

Complete MongoDB-backed backtest and Alpaca Paper API.

## Main routes

- `/api/jobs` — complete backtest execution and results.
- `/api/paper-market` — Alpaca Paper portfolio and next-session execution.
- `/api/admin/parameters` — protected initial parameter installation and status.
- `/api/admin/setup` — protected account binding and paper-state initialization.
- `/api/admin/strategy-configuration` — protected strategy configuration and history.
- `/api/health/live` and `/api/health/ready` — liveness and readiness.

## Administration contract

MongoDB configuration is changed only through protected administrative endpoints. Railway starts the API directly and has no database pre-deploy command. Infrastructure credentials and administrative tokens remain server environment variables.

## v1.12.26 hotfix

The paper-settings repository now strips all administrative metadata, including `revision`, before Pydantic validation. This fixes `POST /api/admin/setup/initialize` after a clean endpoint-driven bootstrap. No strategy parameter was changed.
