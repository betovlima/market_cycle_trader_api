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
