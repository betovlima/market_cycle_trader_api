## v1.13.9 Administrator rotation reporting

- Administrator-only route composed with `require_admin_session`.
- Rotation response excludes private strategy and model fields.
- Viewer Dashboard and Backtest permissions remain unchanged.

# Validation

## 1.13.4

Validated contracts:

- Existing API routers remain composed without changes to their route handlers.
- The dashboard router is additive and read-only.
- Dashboard payloads exclude private configuration and internal execution identifiers.
- Dashboard job detail returns only sanitized metrics and downsampled public equity series.
- Python source compilation passes.
- Existing tests plus the new dashboard contract tests are included.


## v1.13.8 Viewer permission boundary

- Dashboard, jobs and exports are composed with authenticated Viewer-or-Administrator sessions.
- `POST /api/jobs` is available to Viewer sessions and still loads all strategy configuration exclusively from MongoDB.
- The Portfolio snapshot and all Paper Market, setup, strategy and access administration routes remain Administrator-only.
- Focused route-composition and source-contract tests pass without changing the trading engine.
