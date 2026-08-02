# Validation

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
