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
