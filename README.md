# Market Cycle Trader API v1.13.44

FastAPI and MongoDB backend for protected historical simulations, sanitized analytics and Alpaca Paper portfolio monitoring.

## Access roles

| Capability | Viewer | Trader | Administrator |
|---|---:|---:|---:|
| Dashboard, Backtest and Run Backtest | Yes | Yes | Yes |
| Backtest Analytics | Yes | Yes | Yes |
| Paper Portfolio and Portfolio Analytics | No | Yes | Yes |
| Access, setup and strategy administration | No | No | Yes |

At v1.13.13 startup, pre-identity invitation documents are marked `legacy_unverified` and their temporary sessions are revoked. Administrators must create new invitations with an authorized Google email.

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
- `/api/auth/access/preview` — validates an invitation locator without exposing the complete email.
- `/api/auth/access` — verifies the Google ID token and creates an identity-bound session.
- `/api/admin/invitations` — administrator-only identity-bound access management.
- `/api/admin/strategies` — administrator-only research catalog, backtest selection, and explicit Trader-winner promotion.
- `/api/admin/strategy-configuration/winner/install` — emergency installation of the bundled protected winner.
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
GOOGLE_CLIENT_ID
TRADER_ADMIN_GOOGLE_EMAIL
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


## v1.13.13 — Google identity-bound Viewer and Trader access

- Requires an administrator-approved Google email for every new Viewer or Trader invitation.
- Verifies Google Identity Services ID tokens server-side against `GOOGLE_CLIENT_ID`.
- Requires Google `email_verified` and exact normalized-email matching before a first claim.
- Binds each token digest to its invitation UUID, consumes it atomically and binds the authorization to the Google `sub` identifier.
- Allows returning access only for the same Google subject and email.
- Defaults Viewer invitations to two active sessions and Trader invitations to one.
- Revokes the oldest session when the configured session limit would be exceeded.
- Regenerating a claim link terminates sessions, rotates the token and requires a new identity claim.
- Marks legacy token-only invitations as `legacy_unverified` and revokes their sessions at startup.
- Keeps Administrator password authentication unchanged.
- Stores no raw invitation token or Google ID token in MongoDB.


## v1.13.14 — unified Google identity access

- Supports Viewer, Trader and Administrator as Google identity-bound roles.
- Adds direct Google sign-in for accounts that have already claimed active access.
- Bootstraps the primary Google Administrator from `TRADER_ADMIN_GOOGLE_EMAIL` on the first verified login.
- Keeps the password administrator endpoint as a server-side recovery path, but the frontend no longer exposes a password tab.
- Protects the primary Google Administrator from revocation, deletion and claim-link regeneration.
- Keeps one active session by default for Trader and Administrator and two for Viewer.


## v1.13.15 — role-based session expiration

Session limits are enforced by the API with absolute and inactivity expiration per role.

```env
TRADER_VIEWER_SESSION_MAX_AGE_SECONDS=43200
TRADER_VIEWER_SESSION_IDLE_SECONDS=7200
TRADER_TRADER_SESSION_MAX_AGE_SECONDS=28800
TRADER_TRADER_SESSION_IDLE_SECONDS=3600
TRADER_ADMIN_SESSION_MAX_AGE_SECONDS=7200
TRADER_ADMIN_SESSION_IDLE_SECONDS=1800
```


## v1.13.16 — administrative Trader control

- Adds Administrator-only `active`, `paused`, `exit_only`, and `stopped` Trader modes.
- Persists operational state and audit history in MongoDB.
- Keeps strategy parameters and model rules outside public responses.

## v1.13.17 — Administrator System Settings

- Adds Administrator-only, revision-controlled runtime settings in MongoDB.
- Adds training enablement, automatic pre-market training, thread preferences, concurrent-job limit, and backtest timeout.
- Adds optimistic revision checks and a settings history audit trail.
- Keeps Railway plan, replica, CPU, and memory allocation outside application control.

## v1.13.18 — Winner execution lock and Administrator exports

- Preserves the installed winner execution configuration exactly in manual backtests and Paper preparation.
- Stops System Settings from replacing model and numeric execution fields.
- Keeps thread-preference fields only for backward compatibility; they no longer affect the winner.
- Restricts all backtest export endpoints to Administrator sessions.

Protected routes:

```http
GET   /api/admin/system-settings
PATCH /api/admin/system-settings
GET   /api/admin/system-settings/history
```


## v1.13.21 — Candidate strategy lifecycle

The Administrator strategy catalog now has an explicit lifecycle:

- `draft`: editable research revision;
- `candidate`: exact revision certified by a completed backtest;
- `winner`: immutable snapshot used by Trader;
- `former_winner`: preserved immutable historical winner.

Marking a candidate never changes Trader. Editing a candidate returns it to draft. Promotion requires candidate status, the exact certified backtest revision, and the existing safe Trader/Paper state.

Administrator endpoint:

```http
POST /api/admin/strategies/{strategy_id}/mark-as-candidate
```


## v1.13.20 — Protected research catalog with v1.13.16 winner compatibility

- Migrates the existing Railway production winner into an additive strategy catalog without rewriting `backtest_settings/default`.
- Preserves the historical winner identity, including `winner-v1.13.1`, revision, schema version, source file, bootstrap source and configuration hash.
- Restores the exact numerical execution semantics used by API v1.13.16: when `deterministic_execution=false`, no global numeric-thread limit or OMP/BLAS/MKL/NumExpr environment override is applied.
- Applies `numeric_thread_limit` only when `deterministic_execution=true`.
- Allows Administrators to clone and edit draft strategies while a backtest snapshot is running.
- Keeps strategy selection, deletion, promotion and starting another backtest locked until the active job finishes.
- Serializes backtests to one active job at a time.
- Updates a draft's backtest certification only when the completed job used that exact strategy revision.
- Backtests use the selected research snapshot; Paper Trader continues using only the immutable promoted winner snapshot.
- Promotion remains explicit and requires an exact completed Candidate revision, no active backtest, and the Paper pipeline to be idle before calibration, prediction or order execution. The current position and operational state are preserved; no broker call is made.

### Administrator strategy endpoints

```http
GET    /api/admin/strategies
POST   /api/admin/strategies
GET    /api/admin/strategies/{strategy_id}
PUT    /api/admin/strategies/{strategy_id}
DELETE /api/admin/strategies/{strategy_id}
POST   /api/admin/strategies/{strategy_id}/select-for-backtest
POST   /api/admin/strategies/{strategy_id}/promote-to-trader
```

Direct PATCH, PUT, reset and restore operations on `/api/admin/strategy-configuration` remain disabled. `POST /api/admin/strategy-configuration/winner/install` remains an explicit recovery operation only and must not be used during the normal production migration.


## v1.13.23 — Metadata-only Winner promotion

- Promotes the single active Candidate to `Winner v1.13.23` without requiring cash or changing the Paper sleeve.
- Preserves `managed_symbol`, quantity, entry price, holding sessions, strategy cash, realized P&L, trade history, controller mode and an armed next-session run.
- Makes no Alpaca request and does not run calibration, prediction, training or order execution during promotion.
- Verifies locally that the XNYS regular session is closed before changing the Winner pointer.
- Allows an armed run that is still waiting for pre-market analysis; blocks `preparing`, `prepared`, `executing` and plans already pending execution.
- The next scheduled pre-market cycle loads the new immutable Winner and evaluates its complete asset universe.
- Newly prepared plans carry the Winner id, revision, configuration hash and asset list and cannot execute under a different Winner.
