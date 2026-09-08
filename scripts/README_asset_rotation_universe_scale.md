# Asset Rotation Universe Scale

Research-only experiment. Current script version: `asset-rotation-universe-scale-v1.1.1`.

It measures the effect of expanding the opportunity set available to the Strategy's existing intelligent-rotation engine without changing the immutable LightGBM model settings.

## Architecture

The Git history carries versions; filenames remain stable.

- `scripts/research_asset_rotation_universe_scale.py`: command-line entry point and experiment orchestration.
- `src/market_cycle_trader_api/services/asset_universe_scale_candidates.py`: external-universe discovery, full-history validation, objective quality checks and RAM-only candidate loading.
- Existing engine/service modules continue to own LightGBM rotation, transaction costs, Strategy snapshots and market-data access.

No `_v101.py`, `_v102.py`, etc. file naming is used for this experiment.

## Execution sequence

1. Load Strategy #10, requested dates and immutable LightGBM snapshot.
2. Load the 56 Strategy assets from MongoDB and validate Full Strategy History.
3. If the largest requested series exceeds 56, discover and prepare enough external assets. Existing complete Mongo history is reused; missing histories are downloaded to RAM only. Previous candidate rankings are not read.
4. Close MongoDB and freeze the nested universes in memory.
5. Execute 56 -> 82 -> 250 -> 500 with identical LightGBM settings, costs and economic window. Each series is intelligent rotation versus equal-weight Buy & Hold across the same series universe.
6. Write the compact cross-series comparison and local diagnostics, then create a ZIP archive of the complete analysis directory next to that directory.

For `--series 56`, step 3 is skipped entirely, including the Alpaca universe lookup.

## Question

With the same Strategy, model settings, costs and economic window, does intelligent rotation improve or degrade when the eligible US-equity universe grows from the Strategy baseline to larger nested universes?

Default series:

- 56 assets: exact Strategy #10 control universe.
- 82 assets: 56 baseline + 26 prior full-history external candidates when the prior history-integrity artifact is available; otherwise the missing slots are filled deterministically.
- 250 assets: prior universe + additional eligible US equities.
- 500 assets: prior universe + additional eligible US equities.

Each larger universe contains every asset from the smaller universe.

## Candidate eligibility

External assets are not selected from full Strategy Backtest results. The research loader applies:

1. Alpaca active/tradable `us_equity` universe on supported US exchanges.
2. Full Strategy History coverage from `--history-start` through `--snapshot-end`.
3. Asset Discovery price, dollar-volume and volume-quality settings read from MongoDB when available; research defaults are used only if the settings document is absent.
4. Existing Asset Discovery behavior-risk checks.
5. Deterministic ordering using Strategy configuration hash + snapshot date + symbol.

This is a scale experiment, not a new independent final validation. Current-eligibility information is used to construct the larger historical universes, so the result must not be described as an untouched final validation period.

## Persistence contract

The experiment does not call MongoDB insert/update/delete operations.

- Strategy baseline history already in MongoDB is reused.
- Candidate history already complete in MongoDB is read.
- Missing candidate history is downloaded from Alpaca directly into RAM.
- Downloaded candidate history is never upserted to MongoDB.
- Rejected candidate history is never persisted.
- Backtest predictions/trades/results are written only to local `research_output` files.
- The experiment reports `persistent_candidate_history_added = 0`.

Only after research identifies which assets should be retained should a separate explicit persistence step be implemented.

## Run

```bash
git fetch origin
git checkout research/asset-rotation-universe-scale-v1
git pull --ff-only

python scripts/research_asset_rotation_universe_scale.py \
  --strategy-sequence 10 \
  --history-start 2016-01-01 \
  --snapshot-end 2026-09-04 \
  --analysis-start 2020-07-22 \
  --analysis-end 2026-09-03 \
  --series 56 82 250 500 \
  --workers 4
```

Use `--max-candidate-scans` if more than the default 2500 Alpaca symbols must be inspected to obtain 500 eligible assets.

## Main artifacts

- `universe_scale_manifest.json`: experiment description and persistence contract.
- `universe_candidate_history_integrity.csv`: accepted/rejected eligibility diagnostics.
- `universe_data_inventory.csv`: raw-frame row counts and RAM footprint estimates.
- `universe_<N>_assets.json`: exact nested asset list for each series.
- `universe_<N>_result.json`: full metrics for each universe.
- `universe_<N>_predictions.csv`: portfolio path.
- `universe_<N>_trades.csv`: executed rotations.
- `universe_scale_comparison.csv`: compact 56/82/250/500 comparison.
- `universe_scale_summary.json`: machine-readable final summary.
- `<analysis-directory>.zip`: compressed copy of the complete analysis directory, created only after all final artifacts have been written.

The ZIP is created as a sibling of the analysis directory, so it never includes itself. If a ZIP with the same name already exists, it is replaced atomically after the new archive is complete.

The primary comparison within each series is intelligent rotation versus equal-weight Buy & Hold over the same universe. The secondary comparison is rotation performance across universe sizes.