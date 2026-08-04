# Market Cycle Trader API v1.13.12

FastAPI and MongoDB backend for protected historical simulations, sanitized analytics and Alpaca Paper portfolio monitoring.

## Access roles

| Capability | Viewer | Trader | Administrator |
|---|---:|---:|---:|
| Dashboard, Backtest and Run Backtest | Yes | Yes | Yes |
| Backtest Analytics | Yes | Yes | Yes |
| Paper Portfolio and Portfolio Analytics | No | Yes | Yes |
| Access, setup and strategy administration | No | No | Yes |

Existing invitation documents without `role` are interpreted as `viewer`; no MongoDB migration is required.

## Analytical routes

```http
GET /api/analytics/backtests
GET /api/analytics/backtests/{job_id}
GET /api/analytics/portfolio
```

Backtest analytics return only strategy-neutral execution outputs and aggregates:

- equity and drawdown series;
- monthly returns and consistency;
- asset attribution;
- rotation transition matrix;
- holding-period buckets;
- drawdown episodes;
- trade-result concentration;
- sanitized capital rotations.

Portfolio analytics return only account and execution observations:

- stored portfolio snapshots;
- current Paper account projection;
- drawdown and period returns;
- order status, fill and rejection statistics;
- current position and recent sanitized orders;
- controlled connection health.

The analytics layer rejects protected output keys such as backend identifiers, seeds, model scores, effective configuration and decision values.

## Main routes

- `/api/jobs` — queue, inspect and read backtests.
- `/api/dashboard` — strategy-neutral operational dashboard.
- `/api/analytics` — sanitized analytical dashboards.
- `/api/paper-market/public-portfolio` — read-only portfolio for Trader or Administrator.
- `/api/paper-market` — administrator-only Paper operations.
- `/api/admin/invitations` — administrator-only temporary access management.
- `/api/admin/strategy-configuration` — administrator-only protected configuration administration.
- `/api/health/live` and `/api/health/ready` — liveness and readiness.

## Required server variables

```text
MONGO_URL
MONGO_DATABASE
ALPACA_API_KEY_ID
ALPACA_SECRET_KEY
CORS_ORIGINS
TRADER_ADMIN_PASSWORD
TRADER_SESSION_SECRET
TRADER_FRONTEND_BASE_URL
TRADER_AUTH_STORAGE
TRADER_SESSION_MAX_AGE_SECONDS
TRADER_COOKIE_SECURE
TRADER_COOKIE_SAMESITE
```

## v1.13.12

- Adds the `trader` temporary-access role while preserving existing Viewer sessions.
- Adds Backtest Analytics for Viewer, Trader and Administrator sessions.
- Adds Portfolio Analytics for Trader and Administrator sessions.
- Allows Trader sessions to read the sanitized Paper portfolio.
- Keeps Paper operations, setup, strategy configuration and access administration Administrator-only.
- Keeps the legacy administrator rotation endpoint for compatibility while exposing the same sanitized rotation data through Backtest Analytics.
- Returns historical Portfolio analytics even when a live Alpaca refresh is temporarily unavailable.


## v1.13.12 — continuous regular-session Paper robot

- `POST /api/paper-market/start-next-session` now enables a persistent continuous robot.
- The controller automatically arms the following Alpaca regular session after each terminal execution.
- The enabled state survives API restarts and adopts an already active v1.13.10 run during upgrade.
- `GET /api/paper-market/robot/status` exposes administrator diagnostics and scheduler heartbeat.
- `POST /api/paper-market/robot/stop` disables future sessions and safely cancels a pending run.
- `GET /api/paper-market/public-robot-status` exposes only sanitized status to Trader and Administrator portfolios.
- Execution remains restricted to the regular market open returned by Alpaca, plus the configured safety delay.


## v1.13.12 — mandatory pre-market refresh

- Arms the following regular session immediately after the previous run completes.
- Waits until the configured pre-market window instead of training immediately after the close.
- Refreshes completed Alpaca daily bars, reconciles account state and retrains XGBoost before every open.
- Defaults to 90 minutes before Alpaca `next_open` through `premarket_analysis_minutes` in MongoDB.
- Rejects or replaces a legacy prepared plan that did not complete the mandatory pre-market validation.
- Preserves open Alpaca Paper positions and continuous-controller state across API deployments.
