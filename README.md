# Market Cycle Trader API v1.12.13

FastAPI backend for the XGBoost-only Compound Capital Rotation strategy.

## Market data

All historical, backtest, diagnostic, and paper-trading market data is obtained through the Alpaca API. The application does not use Yahoo Finance.

The Alpaca cache is stored in MongoDB collection `alpaca_market_bars` using the configured feed and adjustment policy.

## Automatic Railway deployment

The Railway pre-deploy phase automatically performs these steps:

1. creates missing database indexes and parameter documents;
2. migrates `backtest_settings/default` to the current schema;
3. removes retired provider settings from the active document;
4. downloads missing historical bars from Alpaca in bounded date ranges;
5. validates that every configured asset reaches the locked historical start;
6. starts the API only after the database and market history are ready.

No manual MongoDB migration is required for this release.

## Runtime modules

- `engine/capital_rotation.py`: XGBoost walk-forward training and simulation.
- `engine/live_xgboost_signal.py`: next-session paper-trading decision.
- `engine/market_data.py`: Alpaca-only historical loading and MongoDB cache.
- `services/paper_trading.py`: isolated Alpaca Paper portfolio workflow.

## Active configuration

MongoDB collection: `backtest_settings`

Active document: `_id = "default"`

Strategy, XGBoost, market-data, cost, and scheduling parameters are stored in MongoDB. Railway variables remain limited to connection values, credentials, and tokens.
