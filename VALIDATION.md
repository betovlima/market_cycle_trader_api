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


Paper automation v1.13.12 uses `premarket_analysis_minutes` from `paper_trading_settings/_id=default` (default: 90).


## v1.13.13 identity-bound access

Validated contracts:

- A valid invitation token cannot be claimed by a different authorized email.
- The token digest is invitation-UUID-bound; the first successful Google claim replaces it and stores Google `sub`.
- A different Google subject is rejected after the invitation is claimed, even with the same email.
- Returning access works for the original Google subject without reusing the claim token.
- Trader session limit 1 replaces the older active session.
- Viewer default session limit 2 preserves only the two newest active sessions.
- Legacy invitations are marked `legacy_unverified` and cannot create new sessions.
- OpenAPI requires `credential` for `POST /api/auth/access`.
- Python compilation, unresolved-global scan and the complete automated API test suite pass.


## v1.13.20 winner-compatible strategy-boundary validation

- Python compilation passed for API source and tests.
- Complete automated API suite passed: 89 tests.
- Research profile edits and selections do not change the Trader winner pointer.
- Backtest jobs store profile id, revision, hash, and an immutable execution request.
- Paper preparation reads only the immutable Trader winner context.
- Promotion requires a completed job for the exact candidate revision and safe Trader/Paper state.
- Promotion creates a locked snapshot and preserves the former winner.
- Administrator catalog metadata exposes every `BacktestRequest` field; non-Administrator payloads remain sanitized.
- Legacy direct strategy mutation routes are disabled.
- Administrator ZIP exports include `strategy_manifest.json`.

- API v1.13.16 non-deterministic numerical-thread semantics are restored and contract-tested.
- The initial catalog migration preserves the Railway production winner identity and does not rewrite `backtest_settings/default`.
- Drafts may be edited while an immutable job snapshot runs; old jobs cannot certify newer revisions.
- Backtests remain serialized to one active job.
