# Asset Timing vs Buy & Hold — Research v1

This experiment revalidates **all existing Strategy assets** together with all complete local-Mongo candidate histories.

The qualification question is intentionally independent from the existing 56-asset composition:

> Can the Strategy's own LightGBM timing mechanism trade this asset out-of-sample better than buying the same asset at the beginning of the same OOS window and holding it to the end?

## Scientific separation

1. `research_asset_timing_vs_buyhold.py`
   - local MongoDB only;
   - no Alpaca network;
   - no Mongo writes;
   - validates Full Strategy History;
   - revalidates current Strategy assets and external cached candidates under the same rule;
   - uses the Strategy's own LightGBM Utility features, target, parameters and walk-forward protocol;
   - calibrates the switch margin inside each fold using only pre-test calibration data;
   - decisions are made at the close and executed at the next session open;
   - compares every OOS fold against same-asset buy-and-hold;
   - freezes the qualified universe before any full Strategy Backtest.

2. `research_asset_timing_final_backtest.py`
   - refuses to run if the frozen snapshot hash was changed;
   - uses only `qualified_assets` from the frozen snapshot;
   - starts from the exact completed baseline Backtest request for the Strategy revision/hash/snapshot;
   - replaces its asset universe by the frozen qualified universe;
   - runs one complete Strategy Backtest after qualification is frozen;
   - the Backtest result cannot add/remove assets from that frozen run.

3. `research_asset_timing_then_backtest.py`
   - launcher for the complete experiment;
   - runs qualification first;
   - only after the frozen snapshot exists does it run the one final Strategy Backtest.

## Default qualification rule

`positive-median-and-majority`:

- majority of OOS folds beat the same asset's buy-and-hold;
- median fold excess return over buy-and-hold is positive;
- compounded OOS timing return exceeds compounded OOS buy-and-hold return;
- periods in which the timing policy is invested have better compound asset return than periods in which it elects to remain in CASH.

The last item is a timing-quality diagnostic/gate, not a comparison with the original 56 assets. No correlation, state similarity, Pareto layer or full Strategy Backtest metric participates in qualification.

A stricter `all-folds-positive` rule is available, but it must be chosen before inspecting the final Strategy Backtest result.

## Run the complete experiment

```bash
python scripts/research_asset_timing_then_backtest.py \
  --strategy-sequence 10 \
  --snapshot-end 2026-09-04 \
  --workers 4
```

This command runs:

```text
56 current Strategy assets + complete cached candidates
→ point-in-time per-asset timing validation vs same-asset Buy & Hold
→ requalification of every current asset
→ qualification of new candidates
→ frozen qualified universe + hashes
→ exactly ONE full Strategy Backtest with that frozen universe
→ comparison against the exact certified Strategy baseline Backtest
```

## Qualification only

Use this when the frozen universe must be inspected before the full Strategy Backtest:

```bash
python scripts/research_asset_timing_then_backtest.py \
  --strategy-sequence 10 \
  --snapshot-end 2026-09-04 \
  --workers 4 \
  --qualification-only
```

## Main pre-Backtest artifacts

- `asset_history_integrity.csv`
- `asset_timing_folds.csv`
- `asset_timing_summary.csv`
- `timing_validation_snapshot_frozen.json`
- `experiment_manifest.json`

## Main final-Backtest artifacts

- `baseline_backtest_job.json`
- `baseline_backtest_results.json`
- `final_strategy_backtest_request.json`
- `final_strategy_backtest_job.json`
- `final_strategy_backtest_results.json`
- `backtest_comparison.json`
- `final_experiment_summary.json`

The frozen snapshot records `full_strategy_backtest_used_for_selection=false` and contains hashes for the ranking and qualified universe.