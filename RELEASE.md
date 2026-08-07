## v1.13.25 — XGBoost observability and fold progress

- Streams the backtest subprocess output into the protected API console.
- Adds technical XGBoost start/completion logs per fitted asset, phase, fold and run.
- Emits full Python tracebacks to the local PyCharm/Railway console when the engine fails.
- Persists sanitized run/fold/training progress for the frontend without exposing assets, scores, features or parameters.
- Preserves the validated Winner execution semantics, XGBoost configuration and non-deterministic thread behavior.
