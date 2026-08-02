# Market Cycle Trader API v1.12.14

FastAPI backend for the XGBoost-only Compound Capital Rotation strategy.

## Market data

The application uses the Alpaca API exclusively:

- `sip` for historical training, backtests, diagnostics, and daily signal preparation;
- `iex` for recent/live market-data connectivity;
- Alpaca Paper for order execution.

The historical cache is stored in MongoDB collection `alpaca_market_bars`. Its unique key includes symbol, timeframe, feed, adjustment, and timestamp, so SIP and IEX records cannot be mixed.

## Automatic Railway deployment

The Railway pre-deploy phase automatically:

1. validates the canonical strategy bundled with the release;
2. archives every existing document from `backtest_settings`;
3. clears that collection and inserts one canonical `_id="default"` document;
4. preserves valid paper-trading settings or inserts them when missing;
5. downloads missing historical bars from Alpaca SIP in bounded date ranges;
6. validates that every configured asset reaches the locked historical start;
7. starts the API only after the database and historical cache are ready.

No manual MongoDB migration is required.

## Runtime modules

- `engine/capital_rotation.py`: XGBoost walk-forward training and simulation.
- `engine/live_xgboost_signal.py`: next-session paper-trading decision.
- `engine/market_data.py`: Alpaca-only historical loading and MongoDB cache.
- `services/paper_trading.py`: isolated Alpaca Paper portfolio workflow.

## Active configuration

MongoDB collection: `backtest_settings`

Active document: `_id = "default"`

Strategy, XGBoost, market-data, cost, and scheduling parameters are stored in MongoDB. Railway variables remain limited to connection values, credentials, and tokens.
