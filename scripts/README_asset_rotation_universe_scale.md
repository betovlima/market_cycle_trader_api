# Asset Rotation Universe Scale v1

Research-only experiment. It measures the effect of expanding the opportunity set available to the Strategy's existing intelligent-rotation engine without changing the immutable LightGBM model settings.

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
3. Existing Asset Discovery price, dollar-volume and volume-quality settings.
4. Existing Asset Discovery behavior-risk checks.
5. Deterministic ordering using Strategy configuration hash + snapshot date + symbol.

This is a scale experiment, not a new independent final validation. Current-eligibility information is used to construct the larger historical universes, so the result must not be described as an untouched holdout result.

## Persistence contract

The script opens MongoDB read-only at the application level. It does not call insert/update/delete operations.

- Strategy baseline history already in MongoDB is reused.
- Candidate history already complete in MongoDB is read.
- Missing candidate history is downloaded from Alpaca directly into RAM.
- Downloaded candidate history is never upserted to MongoDB.
- Rejected candidate history is never persisted.
- Backtest predictions/trades/results are written only to local `research_output` files.
- The experiment reports `persistent_candidate_history_added = 0`.

Only after the research identifies the universe/assets we actually want to keep should a separate explicit promotion/persistence step be implemented.

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

- `universe_scale_manifest.json`: immutable experiment description and persistence contract.
- `universe_candidate_history_integrity.csv`: accepted/rejected eligibility diagnostics.
- `universe_data_inventory.csv`: raw-frame row counts and RAM footprint estimates.
- `universe_<N>_assets.json`: exact nested asset list for each series.
- `universe_<N>_result.json`: full metrics for each universe.
- `universe_<N>_predictions.csv`: portfolio path.
- `universe_<N>_trades.csv`: executed rotations.
- `universe_scale_comparison.csv`: compact 56/82/250/500 comparison.
- `universe_scale_summary.json`: machine-readable final summary.

The primary comparison within each series is intelligent rotation versus equal-weight Buy & Hold over the same universe. The secondary comparison is rotation performance across universe sizes.
