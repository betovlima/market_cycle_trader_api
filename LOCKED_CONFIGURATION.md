# MongoDB configuration administration

All MongoDB configuration changes are performed through protected HTTP endpoints.
The API startup creates storage indexes and validates state, but does not silently replace strategy documents.

Administrative authentication uses the `X-Parameter-Bootstrap-Token` header and the server-side `PARAMETER_BOOTSTRAP_API_TOKEN` environment variable.

## Winner installation

The validated strategy source is packaged as:

```text
src/market_cycle_trader_api/parameterizations/winner-v1.13.2.json
```

Install it through:

```http
POST /api/admin/strategy-configuration/winner/install
```

The endpoint replaces the active strategy document, removes extra strategy documents, clears strategy-configuration history, and installs the winner as revision 1. It does not delete jobs, backtest results, market bars, or Paper execution data.

After installation:

1. `GET /api/admin/parameters/status`
2. `POST /api/admin/parameters/bootstrap` when non-strategy documents are missing
3. `GET /api/admin/strategy-configuration`
4. `POST /api/admin/setup/initialize`
5. `GET /api/admin/setup/status`

Other strategy changes continue to use the protected `/api/admin/strategy-configuration` endpoints.
No MongoDB configuration script or Railway pre-deploy database command is required.


Paper automation v1.13.12 uses `premarket_analysis_minutes` from `paper_trading_settings/_id=default` (default: 90).
