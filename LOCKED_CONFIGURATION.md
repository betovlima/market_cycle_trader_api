# Locked production configuration

The public API does not expose or mutate strategy configuration.

- `POST /api/jobs` accepts only `start_date` and optional `end_date`.
- The complete execution snapshot is read from MongoDB.
- The public job response does not include the internal snapshot or hyperparameters.
- Public configuration GET/PUT/PATCH/reset endpoints do not exist.
- Alpaca credentials are read only from server environment variables.
- Alpaca credentials are never stored in MongoDB by the application.

## Required server variables

```text
MONGO_URL=...
MONGO_DATABASE=extrema_backtest
ALPACA_API_KEY_ID=...
ALPACA_SECRET_KEY=...
CORS_ORIGINS=...
```

The aliases `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY` are also accepted for Alpaca compatibility.

## Promote a validated configuration

The JSON must contain every field defined by `BacktestRequest`. Python has no
operational defaults and MongoDB startup never fills missing fields.

Validate without changing MongoDB:

```bash
python scripts/apply_locked_config.py configs/xgboost_champion.json --dry-run
```

Apply it directly to the MongoDB selected by `MONGO_URL` and `MONGO_DATABASE`:

```bash
python scripts/apply_locked_config.py configs/xgboost_champion.json \
  --name xgboost-champion-cpu-v1 \
  --note "promote XGBoost champion"
```

Before every replacement, the prior document is copied to
`backtest_settings_history` for audit and rollback.

Export the currently active document:

```bash
python scripts/export_locked_config.py configs/current_locked_config.json
```

## Missing or invalid configuration

The API starts with MongoDB connected but reports degraded health. New jobs are
rejected until a complete valid document is promoted with the admin script.
The application never creates or repairs the configuration automatically.

## Remove legacy Alpaca secrets from MongoDB

After the environment variables are configured:

```bash
python scripts/purge_legacy_alpaca_credentials.py
```

## Local `.env` loading

The API loads `market_cycle_trader_api/.env` before modules that read MongoDB or
Alpaca variables are imported. Copy `.env.example` to `.env`, fill in the
values, and fully restart Uvicorn. Values injected by Railway or the operating
system are never overwritten by the local file.

## Test Alpaca without a public endpoint

```bash
python scripts/test_alpaca_connection.py
```

The script reads credentials from the environment and `alpaca_feed` from the
locked MongoDB document. The public API does not expose a credential status or
connection-test route.
