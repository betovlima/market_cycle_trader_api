## v1.13.44 — Model estimator step metadata fix

- Makes the model saved on the selected Strategy the only source of truth for Backtest execution.
- `POST /api/jobs` no longer accepts or resolves a UI-selected algorithm; it loads the immutable model snapshot from the selected Strategy.
- Strategy clones retain their saved model snapshot, while model changes remain independent from shared Strategy parameter revisions.
- Saving a model invalidates Candidate certification unless an already completed Backtest exists for the same Strategy revision, model family and exact parameter values.
- When an exact historical run exists, the Strategy adopts that immutable job model snapshot so validated LightGBM research can remain eligible without rerunning.
- Candidate and Winner remain bound to the certified job model snapshot; IQN remains Research-only.
- Legacy XGBoost jobs remain compatible through explicit Strategy-owned XGBoost snapshot reconstruction.

## v1.13.42 — Model-aware Winner Lifecycle and LightGBM Live Trader

- Candidate certification now binds the exact completed backtest, Strategy revision and immutable model snapshot.
- Existing completed LightGBM jobs can be selected as Candidate without rerunning the backtest.
- Winner snapshots now include model family, profile id, settings revision, settings hash and the protected settings snapshot.
- Legacy Winners without model metadata remain backward-compatible as Strategy-owned XGBoost Winners.
- Paper Trader dispatches live decisions by the model frozen in Winner: XGBoost or LightGBM.
- IQN remains research-only until a protected live Trader engine exists.
- Prepared Paper plans are bound to both Winner Strategy hash and Winner model-settings hash, preventing stale plans from executing after a model promotion.
- Fixes live utility inference so next-open decisions do not require the unknown next-session OHLC row; backtest tradability checks remain unchanged.
- Promotion still requires the regular market to be closed and preserves the existing operational safety gates.

## v1.13.41 — Independent Model Profiles and Selected Strategy Integration

- XGBoost, LightGBM and IQN now each own independent versioned research parameter profiles.
- Repeated fields such as learning rate, repetitions, seed step and random state belong to the selected model profile rather than being shared implicitly.
- XGBoost research jobs freeze the XGBoost model profile and map it onto legacy engine fields only inside the immutable execution snapshot.
- Legacy Strategy documents and the Trader Winner remain unchanged for backward compatibility.
- Model-owned legacy XGBoost fields are no longer shown in the Strategy parameter groups.
- Research jobs using an independent model profile do not certify the legacy Strategy lifecycle automatically.

## v1.13.40 — Model-specific Research Settings

- Adds Administrator-only, MongoDB-backed model profiles for LightGBM and IQN.
- LightGBM no longer reuses XGBoost tree hyperparameters at execution time; its baseline profile starts from the v1.13.39 challenger values and evolves independently.
- IQN settings migrate from the v1.13.39 document shape into the same revisioned profile structure without changing the stored baseline values.
- XGBoost remains Strategy-owned and read-only in Model Research so the baseline continues to represent the promotable Winner contract exactly.
- Every challenger job freezes model profile id, settings revision and values into its immutable execution snapshot. Running jobs are unaffected by later edits.
- Adds optimistic revision control and audited model-settings history.
- Keeps Strategy fingerprint, Candidate certification, Winner promotion, Viewer/Trader payloads and the single-backtest queue unchanged.

## v1.13.39 — LightGBM and IQN Research Challengers

- Adds Administrator-only LightGBM and IQN challenger executions.
- Reuses the selected Strategy snapshot, asset universe, dates, folds, transaction costs and repetition seed schedule.
- Keeps the persisted Strategy and Trader Winner contract XGBoost-only. Challenger jobs never certify or promote a Strategy revision.
- LightGBM uses the same engineered features, risk-adjusted utility target and policy calibration boundary as the XGBoost baseline.
- IQN uses the existing rotation state/reward/cost environment with distributional value learning and the same walk-forward folds.
- Stores model family/settings only inside the immutable execution request and protected reproducibility metadata.
- Exposes model identity only through the Administrator model-research boundary; Viewer/Trader job and dashboard payloads remain model-neutral.

## v1.13.38 — Position Risk Diagnostics

- Preserves the v1.13.37 XGBoost policy, training targets, calibrated switch margins and execution decisions; this release adds observation only.
- Separates the research-reference universe from `calendar_anchor_assets` and from the mutable research selection. Existing catalogs snapshot the currently selected research strategy once, while Winner promotion advances the reference to the newly promoted immutable Winner.
- Records point-in-time position state on every decision: return since entry, running MFE/MAE, peak return, drawdown from peak, entry/current score change, current-asset rank and days outside Top 1.
- Records cross-sectional score context: universe mean/std, positive-score count, best/current z-scores and Top1-vs-Top2 standardized gap.
- Adds point-in-time market context using only information available by the decision close: SPY 5/20-session returns, 20-session realized volatility, and 5/20-session universe breadth.
- Extends Administrator `decision_diagnostics.csv` and `experiment_manifest.json`; protected model/risk fields remain excluded from non-Administrator trade payloads.
- Keeps the strategy configuration hash independent from research-reference metadata so research bookkeeping cannot silently change the strategy identity.

## v1.13.28 — Asset Discovery

- Adds Administrator-only candidate-asset discovery with manual start/stop, bounded automatic batches and persisted execution history.
- Applies a cheap recent-market prefilter before loading the fixed historical window; qualifying assets reuse the existing MongoDB market-data cache and refresh incrementally.
- Candidate discovery never changes the Winner universe or promotes assets automatically.
- Automatic discovery is disabled by default until enabled by an Administrator.

## v1.13.26 — Administrator parameter descriptions

- The protected strategy catalog now enriches every editable strategy parameter schema with an administrator-facing description.
- Existing validation bounds, parameter names, strategy values and execution behavior are unchanged.
- Descriptions are served only through the existing authenticated Administrator strategy-catalog endpoint; no strategy documentation is hardcoded into the public frontend bundle.
- No MongoDB migration or environment-variable change is required.

## v1.13.25 — XGBoost observability and fold progress

- Streams the backtest subprocess output into the protected API console.
- Adds technical XGBoost start/completion logs per fitted asset, phase, fold and run.
- Emits full Python tracebacks to the local PyCharm/Railway console when the engine fails.
- Persists sanitized run/fold/training progress for the frontend without exposing assets, scores, features or parameters.
- Preserves the validated Winner execution semantics, XGBoost configuration and non-deterministic thread behavior.
