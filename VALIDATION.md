## v1.13.41 independent model profiles validation

- XGBoost, LightGBM and IQN each own an editable independent model-research profile.
- Repeated fields such as repetitions, seed step and random state are isolated per model.
- XGBoost profile values are mapped only into immutable research job snapshots; persisted Strategy and Trader Winner documents are unchanged.
- Model-owned legacy XGBoost fields are hidden from the visible Strategy parameter groups.
- Model Research executions do not automatically certify the legacy Strategy lifecycle.
- Frontend model parameter selection is embedded inside `SELECTED STRATEGY` and the separate settings box is removed.

## v1.13.40 model-specific research settings validation

- Python compileall: PASS.
- Complete automated API suite: PASS (142 tests).
- Unresolved-global scan: PASS.
- LightGBM fit reads only its immutable model-research snapshot for tree hyperparameters.
- IQN keeps its independent validated training profile.
- XGBoost remains Strategy-owned and unchanged by model-research settings.
- Model settings are Administrator-only, revisioned, audited and stored in MongoDB.
- Frontend receives field metadata/current values from the API and contains no model hyperparameter keys/defaults.

## v1.13.39 LightGBM + IQN research challenger validation

- Python compileall: passed.
- Unresolved-global scan: passed.
- Complete repository test suite: 138 passed with temporary `pymongo`/`bson` import shims because those runtime packages are unavailable in this execution container; shims are not packaged.
- LightGBM smoke training: passed for two synthetic assets using the production feature/utility interface.
- IQN smoke training: passed through replay, n-step targets, quantile loss, target-network updates and validation with a finite score.
- Persisted Strategy and Trader Winner contracts remain XGBoost-only; challenger jobs are Administrator-only and cannot certify a Strategy revision for promotion.
- Viewer/Trader job and dashboard payloads remain model-neutral; model identity is returned only by the Administrator model-research endpoint. Protected model settings remain inside the immutable execution snapshot.
- Frontend JSX/JavaScript parse check passes with TypeScript `--allowJs --jsx react-jsx --noEmit --noResolve`. Full Vite build could not run in this container because frontend node_modules are not present and package installation has no network access.

## v1.13.38 position risk diagnostics validation

- Python compileall: passed.
- Unresolved-global scan: passed.
- Complete repository test suite: 131 passed in the build container. `pymongo`/`bson` import shims were used only because those runtime packages are unavailable in this container; the shims are not packaged.
- Policy-equivalence regression confirms diagnostic collection does not change cash, minimum-hold, expected-edge or switch-margin decisions.
- Position-path tests verify point-in-time return, MFE, MAE, drawdown-from-peak, entry-score deterioration and Top-1 persistence tracking without adding or removing trades.
- Market-regime tests verify SPY and breadth values are computed from data available on or before each decision timestamp.
- Research-reference lifecycle tests verify v1.13.38 snapshots the selected reference independently from later research selection and moves the reference only when a new Winner is promoted.
- Reproducibility tests verify research-reference/candidate metadata is independent from calendar anchors and does not change the strategy-configuration fingerprint.
- Administrator export boundary remains protected; the new model/risk diagnostics are stripped from non-Administrator trade payloads.

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


## v1.13.21 candidate lifecycle validation

- Python compilation passed.
- Complete automated API suite passed: 92 tests.
- Candidate status requires a completed job for the exact current revision.
- Candidate metadata records revision, job id, reason, actor and timestamp.
- Editing a candidate returns it to draft and clears candidate certification.
- Promotion rejects completed drafts until they are explicitly marked as candidates.
- Candidate marking does not change the research selection or immutable Trader winner pointer.
- Winner configuration and numerical execution semantics are unchanged.


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


## v1.13.22 single-candidate lifecycle validation

- Exactly one active `candidate` is represented by `strategy_control/default.candidate_strategy_id`.
- A replaced Candidate becomes locked `superseded_candidate`.
- A promoted Candidate becomes locked `promoted_candidate`.
- Exactly one active `winner` remains selected by `trader_winner_strategy_id`; the previous winner becomes `former_winner`.
- Candidate edits clear the active Candidate pointer and create a Draft revision.
- Production winner configuration and execution semantics are unchanged.


## v1.13.23 metadata-only promotion validation

- Python source and test compilation passed.
- Complete automated API suite passed: 97 tests.
- Promotion enforces the XNYS closed-session boundary through `exchange-calendars`, without a broker request.
- Promotion succeeds with an existing managed position when its symbol belongs to the Candidate universe.
- The Paper state, controller document and an armed next-session run remain byte-for-byte unchanged in the promotion contract test.
- Promotion is blocked after pre-market preparation starts or a plan exists.
- Promotion is blocked for an incompatible managed symbol without liquidation or broker interaction.
- The promoted snapshot is named `Winner v1.13.23` and leaves `paper_state_reinitialization_required=false`.
- Paper plans are bound to Winner id, revision, configuration hash and full asset list before order execution.
- The numerical engine, bundled recovery winner and production backtest semantics are unchanged.
