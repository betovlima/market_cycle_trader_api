## v1.13.23 — Metadata-only Winner promotion

- Removes the obsolete requirement that the Paper sleeve be in cash before a Candidate can become Winner.
- Promotes only the immutable Winner pointer and lifecycle metadata; no Alpaca request, order, liquidation, calibration, prediction or Paper-state reset is executed.
- Preserves the current managed position, quantities, entry price, holding sessions, strategy cash, realized P&L, scheduler control mode and an armed next-session run.
- Blocks promotion after pre-market preparation begins or while a Paper plan is pending/executing.
- Enforces that the XNYS regular session is closed using the local exchange calendar, without contacting Alpaca.
- Validates that the current managed symbol belongs to the Candidate asset universe without contacting Alpaca.
- Names the promoted snapshot `Winner v1.13.23`.
- Binds every newly prepared Paper plan to Winner id, revision, configuration hash and full asset universe.
- Ensures the next scheduled pre-market cycle dynamically loads the promoted Winner and evaluates all of its assets.

## v1.13.22 — Single Candidate and Winner Lifecycle

- Enforces one active Candidate and one active Trader Winner.
- Replacing a Candidate changes the former Candidate to a locked `superseded_candidate` snapshot.
- Promoting the active Candidate changes its research profile to locked `promoted_candidate` and creates the immutable Winner snapshot.
- The former Trader Winner remains a locked `former_winner`.
- Adds `candidate_strategy_id` to the strategy-control document and migrates v1.13.21 catalogs without rewriting the protected winner configuration.
- Candidate edits return the profile to Draft and clear the active Candidate pointer.
- Only Draft strategies can be deleted; lifecycle history remains auditable.

## v1.13.21 — Explicit Candidate Strategy Lifecycle

- Adds `candidate` as an audited status between editable drafts and immutable Trader winners.
- Adds `POST /api/admin/strategies/{strategy_id}/mark-as-candidate`.
- Candidate status requires a completed backtest for the exact current strategy revision.
- Candidate certification stores the exact backtest id, revision, reason, actor and timestamp.
- Editing a candidate creates a new draft revision and clears candidate certification.
- Promotion now requires an explicitly marked candidate revision in addition to the existing safe Trader/Paper state checks.
- Research assets remain editable only in Administrator strategy profiles; the protected Trader winner is unchanged.
- Preserves API v1.13.16 numerical execution compatibility and all v1.13.20 winner-isolation boundaries.

## v1.13.20 — Winner-Compatible Strategy Research Workspace

- Preserves the existing API v1.13.16 production winner as an immutable Trader snapshot.
- Migrates `winner-v1.13.1` metadata without renaming or rewriting the production source document.
- Restores API v1.13.16 numerical execution semantics for non-deterministic XGBoost runs.
- Adds editable research clones with every validated `BacktestRequest` parameter available to Administrators.
- Allows cloning and editing while another immutable backtest snapshot runs.
- Keeps one active backtest globally; selection, deletion and promotion wait for completion.
- Prevents an old job from certifying a newer edited revision.
- Keeps Trader isolated from research until explicit Administrator promotion.
- Preserves former winners as locked snapshots and requires Paper-state reinitialization after promotion.
- Adds winner-engine compatibility metadata to Administrator exports.

## v1.13.18 — Winner Execution Lock and Administrator Exports

- Preserves the installed winner execution configuration exactly during manual and automatic runs.
- System Settings no longer overrides `xgb_n_jobs` or `numeric_thread_limit`.
- Runtime thread preferences remain stored only for backward compatibility and are not applied to the winner.
- Backtest exports are restricted to Administrator sessions.

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


## v1.13.15 — Role-based session expiration

- Enforces absolute and inactivity timeouts by Viewer, Trader, and Administrator role.

## v1.13.16 — Administrative Trader controls

- Adds persisted `active`, `paused`, `exit_only`, and `stopped` operation modes.
- Adds Administrator operational audit history.

## v1.13.17 — Administrator System Settings

- Adds revision-controlled MongoDB runtime settings for model execution.
- Adds XGBoost and numeric thread controls, concurrent-job limits, backtest timeout, and automatic pre-market training control.
- Applies runtime settings to manual backtests and Paper model training while preserving the locked winner strategy.
- Adds protected settings history and conflict-safe updates.
