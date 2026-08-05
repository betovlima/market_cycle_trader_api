## 1.13.9

- Adds sanitized administrator-only capital-rotation reporting for completed backtests.
- Keeps detailed rotation output behind the administrator session boundary.
- Does not expose strategy parameters, Q-values, seeds or backend identifiers.

# Market Cycle Trader API

## 1.13.4 — Additive dashboard read model

This version adds two read-only, strategy-neutral endpoints for the redesigned frontend:

- `GET /api/dashboard/summary`
- `GET /api/dashboard/jobs/{job_id}`

Compatibility guarantees:

- No existing route was removed, renamed, or changed.
- No existing request or response payload was modified.
- No backtest, Paper, MongoDB, configuration, export, scheduler, or engine behavior was modified.
- The new endpoints do not mutate MongoDB.
- Private configuration, model identifiers, seeds, assets, horizons, hashes, internal backends, and effective configuration are excluded from the new payloads.


## 1.13.8

Viewer sessions can now use Dashboard, Backtest, job results and exports, including starting a protected backtest. Portfolio and all administration surfaces remain restricted to Administrator sessions.

## v1.13.10 — Analytical dashboards and Trader access

- Backtest Analytics for all authenticated roles.
- Portfolio and Portfolio Analytics for Trader and Administrator.
- Temporary role selection for Viewer or Trader invitations.
- Strategy-neutral analytics payload enforcement.
- No MongoDB migration and no new environment variables.


## v1.13.12 — Continuous regular-session Paper robot

- Converts the protected next-session activation into a persistent continuous controller.
- Automatically schedules the following Alpaca regular session after every completed or failed run.
- Preserves the enabled state across API restarts.
- Adopts an existing active v1.13.10 run without replacing its prepared plan.
- Adds protected robot heartbeat/status and stop endpoints.
- Adds a sanitized robot-status endpoint for Trader and Administrator Portfolio screens.
- Stops automatic scheduling when an execution enters `review_required`.


## v1.13.12 — Mandatory pre-market analysis

- Moves the definitive daily data refresh, reconciliation, calibration and XGBoost training to the pre-market window.
- Uses Alpaca `next_open` minus the MongoDB setting `premarket_analysis_minutes` (default 90).
- Keeps the API restart-safe and backward-compatible with active v1.13.11 runs.
- Discards pre-v1.13.12 prepared plans before execution and rebuilds them during the mandatory pre-market window.
- Does not liquidate or modify an existing Alpaca Paper position during deployment.


## v1.13.13 — Identity-bound temporary access

- Replaces bearer-only Viewer/Trader access with an authorized-email Google identity claim.
- Adds server-side Google ID-token verification and immutable `sub` binding.
- Adds invitation-UUID-bound token digests, atomic token consumption and invitation-specific returning login.
- Adds per-invitation active-session limits, with Viewer default 2 and Trader default 1.
- Adds idempotent migration of old invitations to `legacy_unverified` and revokes their sessions.
- Adds identity mismatch, claim, login and session replacement audit events.
- Preserves all strategy, backtest, analytics, Portfolio and Paper automation behavior.

## v1.13.14 — Unified Google authentication

- Adds Google identity-bound Administrator access alongside Viewer and Trader.
- Adds direct login for previously claimed identities.
- Creates and protects the primary Google Administrator configured by `TRADER_ADMIN_GOOGLE_EMAIL`.
- Keeps the password login endpoint as a recovery mechanism while removing it from the frontend.
