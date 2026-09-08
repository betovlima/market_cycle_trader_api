# Asset Rotation — Independent Final-Period Validation v1

This research test answers one question:

> Using only information available before a final reserved period, can causal intelligent rotation across a frozen asset universe beat buying and holding the same frozen universe?

## Scientific design

The test intentionally separates time into two parts.

1. **Selection period** — used to evaluate rotation opportunity and freeze the assets that are allowed into the final test.
2. **Independent final validation period** — never used for asset selection, LightGBM training, or switch-margin calibration.

The default final validation window is the last **252 XNYS trading sessions** ending at `--snapshot-end`.

For `--snapshot-end 2026-09-04`, the runner resolves the exact XNYS sessions automatically and prints the resulting dates before any model is run.

## Candidate-universe freeze

The runner reads only these columns from the previous completed `asset_history_integrity.csv`:

- `symbol`
- `source`
- `history_complete`

It deliberately does **not** read any previous qualification result, ranking, realized Utility, or Backtest result. The previous file is used only to keep the same candidate universe and the Full Strategy History rule.

## Phase 1 — selection using only the past

`research_asset_rotation_leadership_v13.py` is run with `snapshot_end` changed to the last market session before the independent validation period.

The final selection rule remains rotation-opportunity only. Same-asset Timing vs Buy & Hold is diagnostic and does not decide participation.

The resulting `rotation_leadership_snapshot_frozen.json` is immutable input to Phase 2.

## Phase 2 — independent final period

`research_asset_rotation_independent_validation.py`:

- loads the frozen qualified universe;
- applies a strict purge before the final period so no forward Utility label can cross into validation;
- calibrates the switch margin only before validation;
- fits the final LightGBM models only on labels that fully mature before validation;
- runs the production single-position rotation policy on the final period;
- executes decisions at the next daily open;
- uses the production transaction-fee and slippage functions;
- compares against **equal-weight Buy & Hold across the exact same frozen assets and execution dates**.

The final period never adds or removes an asset and never changes a model parameter.

## Run

```bash
python scripts/research_asset_rotation_independent_then_validate.py \
  --strategy-sequence 10 \
  --snapshot-end 2026-09-04 \
  --validation-sessions 252 \
  --workers 4
```

By default, the candidate universe is read from:

```text
research_output/asset_rotation_leadership_strategy_10_2026-09-04/asset_history_integrity.csv
```

An explicit file can be supplied with `--universe-file`.

## Outputs

The root experiment directory contains `independent_validation_design.json`.

The selection subdirectory contains the pre-validation rotation-opportunity qualification and frozen snapshot.

The independent validation subdirectory contains:

- `independent_validation_result.json`
- `independent_validation_manifest.json`
- `independent_validation_predictions.csv`
- `independent_validation_trades.csv`
- `independent_validation_history_integrity.csv`

The console ends with an explicit result:

```text
RESULT: PASS - intelligent rotation beat Buy & Hold on the untouched final period.
```

or

```text
RESULT: FAIL - intelligent rotation did not beat Buy & Hold on the untouched final period.
```

No production API or Front version is changed by this research branch.
