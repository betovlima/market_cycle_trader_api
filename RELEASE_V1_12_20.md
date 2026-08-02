# Market Cycle Trader API v1.12.20

## MongoDB runtime correction

- `MONGO_URL` is the only required MongoDB connection variable.
- `MONGO_URI` remains accepted as a compatibility alias.
- `MONGO_DATABASE` is optional.
- When `MONGO_DATABASE` is absent, the API uses `extrema_backtest`, matching the previously validated deployment contract.
- MongoDB values are resolved from the process environment when a connection is created.
- A request retries initialization when startup encountered a temporary connection failure.
- MongoDB connection values are never read from or written to MongoDB collections.

## Private administration restored

The protected administration routes were restored and are excluded from the public OpenAPI document:

- `GET /api/admin/parameters/status`
- `POST /api/admin/parameters/bootstrap`
- `GET /api/admin/strategy-configuration`
- `PATCH /api/admin/strategy-configuration`
- `PUT /api/admin/strategy-configuration`
- `GET /api/admin/strategy-configuration/history`
- `POST /api/admin/strategy-configuration/history/{history_id}/restore`

They require `X-Parameter-Bootstrap-Token`, matched against `PARAMETER_BOOTSTRAP_API_TOKEN` in the API environment.

No strategic parameterization is bundled with this source package. Private JSON files are supplied separately to the administrative script or protected endpoint.

## Administrative scripts

- `scripts/bootstrap_parameters.py`
- `scripts/apply_locked_config.py`
- `scripts/export_locked_config.py`
- `scripts/apply_paper_trading_config.py`
- `scripts/check_mongo_connection.py`

This release does not include `.env.example` or `.gitignore`.
