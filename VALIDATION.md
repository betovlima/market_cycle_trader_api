# Validation

## 1.13.2

Validated in the packaging environment:

- Python source compilation with `compileall`.
- Full pytest suite: 35 tests passed using temporary import stubs for unavailable `pymongo`/`bson`; service behavior was exercised with an in-memory fake database.
- Bundled winner file validation against `BacktestRequest`.
- Exact winner configuration SHA-256 verification.
- Replacement of an old default strategy document.
- Deletion of extra strategy documents and all strategy history documents.
- Revision reset to 1 and winner source/hash metadata persistence.
- Rejection while a backtest is active.
- Rejection while an Alpaca Paper run is active.
- Source contract and generated OpenAPI validation for `POST /api/admin/strategy-configuration/winner/install` and the packaged winner file.
- JSON syntax validation for all bundled parameterizations and `script/` payloads.

Deletion scope of the new endpoint:

- Deleted or overwritten: `backtest_settings` strategy documents and `backtest_settings_history`.
- Preserved: jobs, runs, predictions, trades, comparisons, failures, market bars, Paper settings, Paper runs, Paper plans, Paper orders, and portfolio snapshots.

## 1.13.1

Validated in the packaging environment:

- Python source compilation with `compileall`.
- JSON syntax for every bundled parameterization and every file under `script/`.
- Multi-horizon champion configuration validation against `BacktestRequest` using dependency stubs for unavailable external packages.
- Source contract: API version 1.13.1 and engine module `market_cycle_trader_api.engine.compound_rotation_backtest`.
- Source contract: protected strategy, parameter-bootstrap, setup, and paper-market routers remain composed in `main.py`.
- Source contract: legacy public config mutation router, credential mutation router, and legacy multi-strategy engine are not packaged.
- Environment loader tests for empty Windows variables, non-empty system-variable priority, and child-process propagation.
- Diagnostic helper tests for typed empty series and safe future-price filtering.
- Static import and syntax checks for modified modules.

Not executed in the packaging environment:

- Real MongoDB integration because the packaging environment does not provide the project database.
- Real Alpaca market-data and Paper requests because credentials are not available.
- Complete dependency-backed pytest execution because `pymongo` and `alpaca-py` are not installed in the packaging runtime.
- Full champion backtest, which must be reproduced locally after the empty database is bootstrapped.

Required local acceptance sequence:

1. Bootstrap the cleared database.
2. Confirm the active configuration is revision 1, CPU, seed 42, and multi-horizon.
3. Run the full-history backtest from 2016-01-01.
4. Confirm the effective engine is `compound_rotation_backtest`.
5. Compare the result with the validated champion artifact before arming Paper.
