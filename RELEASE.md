## v2.0.0 — Winner Lifecycle Governance

- Decouples Winner identity from the API release version. Multiple immutable historical Winner snapshots may be created by the same API release.
- Assigns each new Winner an independent monotonic `winner_sequence` and keeps `winner_api_version`/`source_api_version` only as audit metadata.
- Enforces the lifecycle roles `Candidate`, `Promoted Candidate` and `Winner` independently: at most one active profile may hold each role.
- Adds `promoted_candidate_strategy_id` to Strategy control and exposes the corresponding profile in the admin catalog contract.
- When a new Candidate is promoted, the previous Winner becomes `former_winner`, the previous Promoted Candidate becomes immutable `superseded_candidate` history, and the new validated Candidate becomes the single `promoted_candidate`.
- Migrates legacy catalogs that contain multiple `promoted_candidate` profiles, keeping the source Strategy of the active Winner as the current Promoted Candidate whenever it can be identified.
- Preserves promotion lock, market-closed checks, exact Candidate/backtest/model validation, operational-state preservation and compensating rollback behavior.
- Keeps API version metadata for reproducibility without using it as a uniqueness constraint for Winner promotion.

## v1.13.50 — Inclusive Frozen Cutoff and CARO Prior Replay

- Treats a historical `end_date` as inclusive while keeping Mongo/Alpaca right-hand boundaries exclusive internally.
- Replays frozen tuning snapshots through the exact final completed market session used by the source experiment.
- Preserves the market-data signature guard; signature mismatches still fail the campaign instead of being ignored.
- Adds regression coverage proving that a requested final daily session is retained.
- Adds regression coverage proving that a completed Control + 20-candidate Latin Hypercube campaign imports all 21 observations into CARO prior evidence.

## v1.13.49 — Persistent Model Tuning Diagnostics

- Adds admin-only campaign and candidate diagnostic-log endpoints for Model Tuning.
- Preserves candidate failure type/message and bounded tuning-worker traceback context.
- Exposes the retained internal backtest job log after secret redaction, including failed control executions.
- Adds bounded campaign orchestration events without persisting failed candidates as analytical results.

## v1.13.48 — CARO Prior Campaign Reuse + Gaussian-Process Champion Search

- CARO Probability may run independently or reuse a completed Latin Hypercube campaign as prior observations.
- A source candidate can be selected as the Champion anchor; imported observations are not rerun.
- Reused CARO trials start after the imported candidate IDs and are proposed sequentially from all prior + newly observed results.
- CARO now uses Gaussian-process surrogates with constrained expected improvement. This allows non-zero probability mass above the current best observed capital, unlike a purely interpolative tree ensemble.
- Candidate proposals use a hybrid pool: global exploration, local exploration around the Champion, and local exploration around the strongest observed regions.
- Actual completed CARO candidates are checked against the configured Champion gate before they can be adopted.
- Every campaign executes from a frozen Backtest request snapshot and fixed market-data cutoff; a market-data signature mismatch fails the campaign.
- Integrated tuning restarts safely after an API/container restart by rerunning only the interrupted candidate from the frozen snapshot.
- The tuning process runs inside the main API research worker; no separate Railway service is required.
- CARO exports include prior observations, Champion anchor, actual Champion-gate result, frozen data boundary, proposal region and reproducibility metadata.
