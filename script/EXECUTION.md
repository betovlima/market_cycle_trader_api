# Production migration and strategy-research sequence

All changes must use the protected API or Administrator interface. Do not write strategy documents directly in MongoDB.

## 1. Preserve the current Railway winner

Before deployment, save:

- the current API v1.13.16 deployment/commit;
- the current `backtest_settings/default` document;
- the latest winning backtest export;
- the current Paper/Trader status.

Pause Trader before replacing the API. Pausing must not liquidate an existing position.

## 2. Deploy API v1.13.20

```http
GET /api/health/live
```

Expected API version: `1.13.20`.

```http
GET /api/health/ready
```

On first startup, the API creates the additive strategy catalog from the existing production winner. It does not rewrite `backtest_settings/default` and does not install another winner.

## 3. Do not reinstall the bundled winner during migration

Do **not** call:

```http
POST /api/admin/strategy-configuration/winner/install
```

That endpoint remains available only for an explicit disaster-recovery operation. Normal migration must preserve the production `winner-v1.13.1` provenance.

## 4. Verify the imported production winner

Using an Administrator session:

```http
GET /api/admin/strategies
GET /api/admin/strategies/control
```

Expected initial state:

```text
research strategy: imported production winner
Trader winner: the same imported immutable snapshot
winner source file: winner-v1.13.1.json
winner configuration hash: 22a4193fbb30de33d75864fc28c3b1923e4dedd4970b14f9537f793bccf18953
locked: true
paper_state_reinitialization_required: false
```

The original `backtest_settings/default` document must remain unchanged.

## 5. Validate winner-engine compatibility

Start one full-history backtest:

```http
POST /api/jobs
```

Request body: none. The job uses the selected immutable strategy snapshot.

Poll:

```http
GET /api/jobs/{job_id}
```

After completion:

```http
GET /api/jobs/{job_id}/results
GET /api/jobs/{job_id}/export.zip
```

The Administrator export must include:

```text
strategy_manifest.json
winner_engine_compatibility: api-v1.13.16
numeric_thread_environment_applied: false
```

For the historical production winner, `deterministic_execution=false`; therefore no global OMP/BLAS/MKL/NumExpr thread override is applied.

Do not resume Trader when the reproduced result is materially inconsistent with the preserved production result.

## 6. Resume the unchanged Trader winner

After validation, resume Trader through Administrator controls. No Paper-state reinitialization is required because the winner pointer did not change.

## 7. Research workflow

1. Clone the current winner or another strategy.
2. Edit any validated strategy parameter in the draft.
3. A draft may be cloned or edited while another immutable backtest snapshot runs.
4. Wait for the active backtest to finish before selecting another strategy or starting another backtest.
5. Select the draft for future backtests. Trader remains on its current winner.
6. Run and export the candidate backtest.
7. Keep the current winner, or explicitly promote the exact tested revision.

Promotion requires:

- the candidate's current revision has a completed backtest;
- no queued/running backtest;
- Trader paused or stopped;
- no active Paper run;
- no pending Paper plan;
- the controlled Paper sleeve in cash.

Promotion creates a new immutable winner snapshot and preserves the former winner. Only after promotion must Paper state be reinitialized before Trader is restarted.

## 8. Single-backtest rule

Only one backtest may be queued or running. Cloning and editing drafts remain available during the run, but selection, deletion, promotion and `Start New Backtest` remain locked until completion.
