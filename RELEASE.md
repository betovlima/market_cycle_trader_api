# Releases

## 1.13.1 — Multi-horizon migration hardening

- Preserves the multi-horizon XGBoost strategy and the dedicated `compound_rotation_backtest` engine.
- Sets the bundled fresh-database strategy to the validated CPU configuration.
- Bumps the strategy document schema to 16.
- Keeps the protected administration endpoints for parameter bootstrap, setup, strategy revision management, and paper execution.
- Loads `.env` before infrastructure imports and fills variables that Windows launchers created as empty.
- Refreshes the environment immediately before spawning the backtest worker.
- Passes the complete refreshed environment explicitly to the worker.
- Reads Alpaca credentials exclusively from environment variables.
- Rejects a job before queueing when Alpaca credentials are unavailable.
- Records the effective engine module, engine path, and Python executable on the job.
- Uses `alpaca_market_bars/1Day` for exit diagnostics with a legacy `market_bars/1d` fallback.
- Guarantees a UTC `DatetimeIndex` for empty and normalized diagnostic price series.
- Prevents optional performance diagnostics from hiding an otherwise completed backtest.
- Removes unused legacy public configuration/credential routers and legacy strategy engines from this package.
- Consolidates all release notes in `RELEASE.md` and validation notes in `VALIDATION.md`.
- Adds endpoint-named JSON payloads and a complete execution sequence under `script/`.

## 1.13.0 — Multi-horizon series movements

- Added weighted forward targets for 5, 10, 20, 40, and 60 trading sessions.
- Added movement-capture and trend-persistence target components.
- Enforced a purge length no shorter than the maximum target horizon.
- Added the multi-horizon configuration fields to the protected strategy schema.
- Preserved next-session execution and expanding walk-forward validation.
