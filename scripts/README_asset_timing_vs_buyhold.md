# Asset Timing vs Buy & Hold — Research v1

This experiment revalidates **all existing Strategy assets** together with all complete local-Mongo candidate histories.

The qualification question is intentionally independent from the existing 56-asset composition:

> Can the Strategy's own LightGBM timing mechanism trade this asset out-of-sample better than buying the same asset at the beginning of each test fold and holding it to the end?

## Scientific separation

1. `research_asset_timing_vs_buyhold.py`
   - local MongoDB only;
   - no Alpaca network;
   - no Mongo writes;
   - validates full Strategy history;
   - revalidates current Strategy assets and external cached candidates under the same rule;
   - runs point-in-time walk-forward LightGBM timing per asset;
   - compares every fold against same-asset buy-and-hold;
   - freezes the qualified universe before any full Strategy backtest.

2. `research_asset_timing_final_backtest.py`
   - refuses to run if the frozen snapshot hash was changed;
   - uses only `qualified_assets` from the frozen snapshot;
   - runs one complete Strategy backtest after qualification is frozen;
   - the backtest result cannot add/remove assets from that frozen run.

## Default qualification rule

`positive-median-and-majority`:

- median fold excess return over buy-and-hold > 0;
- beats buy-and-hold in at least 3 of 4 folds;
- asset compound return while the model is invested > compound return while the model is in CASH.

A stricter `all-folds-positive` rule is available for research comparison, but must be chosen before inspecting the full Strategy backtest result.

## Run qualification

```bash
python scripts/research_asset_timing_vs_buyhold.py \
  --strategy-sequence 10 \
  --snapshot-end 2026-09-04 \
  --workers 4
```

## Review frozen decision

Main artifacts:

- `asset_history_integrity.csv`
- `asset_timing_folds.csv`
- `asset_timing_summary.csv`
- `timing_validation_snapshot_frozen.json`
- `experiment_manifest.json`

## Run exactly one full Strategy backtest

Only after reviewing the frozen universe:

```bash
python scripts/research_asset_timing_final_backtest.py \
  --strategy-sequence 10 \
  --snapshot-end 2026-09-04
```

The frozen snapshot records `full_strategy_backtest_used_for_selection=false`.
