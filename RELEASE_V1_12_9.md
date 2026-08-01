# Market Cycle Trader API v1.12.9

## Fixed

- Prevented `GET /api/jobs/{job_id}/results` from failing when an exit has no future market bars available.
- Empty market-price series now always use a UTC `DatetimeIndex` instead of the default `RangeIndex`.
- Exit diagnostics now normalize timestamps and safely skip missing or malformed future-price series.
- Duplicate market-bar timestamps are deduplicated before diagnostic calculations.
- Added `scikit-learn` to `requirements.txt`, which is required by `xgboost.XGBRegressor` on clean Railway builds.

## Preserved

- XGBoost high-performance configuration with seed `3042`.
- MongoDB bootstrap before Railway startup.
- Railway liveness healthcheck at `/api/health/live`.
- Alpaca Paper execution and isolated strategy budget behavior.
- Existing valid MongoDB parameter documents remain unchanged during bootstrap.

## Expected behavior

A completed backtest can now return its result payload even when a SELL operation occurs near the latest available market session and there are no later cached bars for the 1, 5, 10, or 20-session exit diagnostics.
