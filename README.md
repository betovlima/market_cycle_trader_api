# Market Cycle Trader API v1.12.16

FastAPI backend for the XGBoost Compound Capital Rotation strategy and continuous Alpaca Paper execution.

## Fixed production research rules

The following values are code-level rules and cannot be changed by the frontend, MongoDB configuration API, or Railway variables:

- training history starts at `2016-01-01`;
- market data provider is Alpaca;
- historical training/backtest feed is Alpaca SIP;
- live connectivity feed is Alpaca IEX;
- decisions are generated only after a completed regular-session daily candle;
- orders are intended for the next regular-session open;
- look-ahead is forbidden;
- the model is retrained after every completed regular session.

The public frontend `start_date` and `end_date` fields select only the simulated analysis/report window. They never move the fixed training-history start.

## Robust seed ensemble

Every study trains the configured seed sequence, normally:

- 42
- 1042
- 2042
- 3042
- 4042

The backtest stores the five individual seed studies and a sixth production result named `xgboost_utility_ensemble`. The ensemble uses majority voting and stays in the current position when agreement is below the configured minimum. The paper trader uses the same ensemble instead of selecting the seed with the highest observed historical return.

Each result records fold-level robustness measurements, including robust score, positive-fold ratio, folds above the benchmark, worst-fold return, median fold return, median excess return, and fold-return dispersion.

## Continuous Alpaca Paper cycle

A run is initially armed through the existing Paper Market API. After a completed paper run, the scheduler automatically arms the next Alpaca regular session when `automatic_continuation_enabled=true` in `paper_trading_settings/default`.

For each session, the service:

1. waits for the regular-session daily candle to complete;
2. refreshes Alpaca SIP history in MongoDB;
3. retrains all ensemble seeds using data available through that completed session;
4. prepares the next-session target after market close;
5. submits the isolated Paper order after the configured opening delay;
6. reconciles the strategy state with the Alpaca Paper account;
7. arms the following regular session automatically.

Prepared plans contain the strategy configuration SHA-256. A deployment or strategy change invalidates an older prepared plan and forces a fresh ensemble decision before order submission.

## Automatic Railway migration

The Railway pre-deploy phase runs `scripts/bootstrap_parameters.py`. It automatically:

- migrates `backtest_settings/default` to schema 16;
- removes legacy editable training-start/provider/feed fields;
- archives the previous configuration in `backtest_settings_history`;
- adds the seed-ensemble parameters without manual MongoDB editing;
- migrates `paper_trading_settings/default` with continuous-session scheduling fields;
- preserves valid user-managed strategy parameters whenever they remain compatible.

Scheduler timing values are stored in MongoDB, not Railway variables. Railway remains limited to connection strings, Alpaca credentials, and protected API tokens.

## Strategy configuration API

All routes use the `X-Parameter-Bootstrap-Token` header.

- `GET /api/admin/strategy-configuration`
- `PATCH /api/admin/strategy-configuration`
- `PUT /api/admin/strategy-configuration`
- `POST /api/admin/strategy-configuration/reset`
- `GET /api/admin/strategy-configuration/history`
- `POST /api/admin/strategy-configuration/history/{history_id}/restore`

The GET response includes a `system_rules` object. Attempts to change fixed rules such as `start_date`, `training_start_date`, `market_data_provider`, `alpaca_historical_feed`, or `alpaca_live_feed` are rejected.
