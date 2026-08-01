# Market Cycle Trader API v1.12.15

FastAPI backend for the XGBoost-only Compound Capital Rotation strategy.

## Market data

The application uses the Alpaca API exclusively. The active MongoDB strategy document selects the historical and live feeds used by new backtests and paper plans.

The historical cache is stored in MongoDB collection `alpaca_market_bars`. Its unique key includes symbol, timeframe, feed, adjustment, and timestamp, so records from different Alpaca feeds are not mixed.

## Automatic Railway deployment

The Railway pre-deploy phase executes `scripts/bootstrap_parameters.py` automatically. It:

1. creates storage indexes;
2. inserts the initial strategy when `backtest_settings/default` is missing;
3. preserves every valid strategy configuration changed through the API;
4. archives and repairs the strategy automatically only when the stored schema is invalid;
5. removes extra strategy documents while preserving the valid `default` document;
6. inserts missing paper-trading settings.

Historical data is loaded and refreshed on demand by the backtest and signal workflows. A long market-data download no longer blocks deployment.

## Strategy configuration API

All routes use the `X-Parameter-Bootstrap-Token` header.

- `GET /api/admin/strategy-configuration`: read the active configuration and revision.
- `PATCH /api/admin/strategy-configuration`: update selected parameters.
- `PUT /api/admin/strategy-configuration`: replace the complete validated configuration.
- `POST /api/admin/strategy-configuration/reset`: restore the bundled initial configuration.
- `GET /api/admin/strategy-configuration/history`: list archived configurations.
- `POST /api/admin/strategy-configuration/history/{history_id}/restore`: restore an archived configuration.

Each update is validated with `BacktestRequest`, archived before replacement, assigned a new revision, and made available immediately to new backtests. Updates are rejected while a backtest is queued or running.

## Runtime modules

- `engine/capital_rotation.py`: XGBoost walk-forward training and simulation.
- `engine/live_xgboost_signal.py`: next-session paper-trading decision.
- `engine/market_data.py`: Alpaca-only historical loading and MongoDB cache.
- `services/paper_trading.py`: isolated Alpaca Paper portfolio workflow.

## Active configuration

MongoDB collection: `backtest_settings`

Active document: `_id = "default"`

Strategy, XGBoost, market-data, cost, and scheduling parameters are stored in MongoDB. Railway variables remain limited to connection values, credentials, and tokens.
