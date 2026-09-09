# Counterfactual Rotation Advantage v5.0.0

## Research question

Does an expanding memory of prior out-of-sample calibration blocks reduce the
regime imprinting observed in v4 while keeping the same persistence target and
the same HOLD-versus-ROTATE decision?

## Why v5 exists

v4 improved final capital from about US$646k to about US$920k with nearly the
same number of rotations. The 5-session persistence target therefore improved
trade quality versus the 1-session target.

However, the second outer fold remained strongly biased by its short calibration
window. That fold's calibration target mean was negative, and the learned policy
then rejected most Top-1 changes for the following two-year test interval.

The v5 hypothesis is deliberately narrow:

> the action model should not forget earlier calibration regimes when a new outer
> fold begins.

## What remains frozen

- frozen 56-asset universe;
- LightGBM opportunity target;
- LightGBM hyperparameters;
- state/action features;
- exact execution simulator and transaction costs;
- outer walk-forward train/calibration/purge/test boundaries;
- v4 persistent relative label horizon: `min(rotation_target_horizons)`;
- CASH excluded from the decision layer;
- decision boundary remains economic zero: predicted advantage > 0 means ROTATE.

There is no new switch margin, minimum holding period, CASH threshold, market
regime rule, or manually tuned confidence gate.

## What changes

v4 trained each outer-fold rotation model using only that fold's 126-session
calibration block.

v5 keeps each calibration block as an out-of-sample memory block and expands the
rotation training set chronologically:

```text
Outer fold 1 action model:
  calibration block 1

Outer fold 2 action model:
  calibration block 1
  + calibration block 2

Outer fold 3 action model:
  calibration block 1
  + calibration block 2
  + calibration block 3
```

Each block is generated with the LightGBM opportunity models trained before that
block. Therefore its opportunity scores are out-of-sample relative to the model
that produced them.

No outer test labels are ever replayed into later folds. Only calibration blocks
are retained.

## Weighting

The counterfactual generator may create several incumbent/challenger states for a
single decision date. As in v4, all rows from the same date share total weight 1.
When blocks are concatenated, every historical decision date therefore carries
the same total weight regardless of how many hypothetical states were generated.

There is no recency multiplier or manually chosen regime weighting in v5.

## Target

For each calibration state where the incumbent differs from the LightGBM Top-1:

```text
persistent_rotation_advantage
=
net_log_return(ROTATE to Top-1 over shortest Strategy horizon)
-
net_log_return(HOLD incumbent over same horizon)
```

The switching cost is paid only on the first transition of the ROTATE path.

The live policy is still evaluated every session. The target horizon is not a
minimum holding rule.

## Decision rule

```text
predicted advantage > 0  -> ROTATE Top-1
predicted advantage <= 0 -> HOLD incumbent
```

Zero is the economic indifference point, not an arbitrary threshold.

## Recommended test

Run only the frozen 56-asset series first:

```bash
git fetch origin
git switch research/counterfactual-rotation-advantage-v5
git pull --ff-only origin research/counterfactual-rotation-advantage-v5

python scripts/research_counterfactual_rotation_advantage_v5.py \
  --strategy-sequence 10 \
  --history-start 2016-01-01 \
  --snapshot-end 2026-09-04 \
  --analysis-start 2020-07-22 \
  --analysis-end 2026-09-03 \
  --series 56 \
  --workers 4
```

If the baseline archive is already under `research_output`, no
`--baseline-output-dir` argument is required.

## Expected log markers

```text
LightGBM opportunity model + Counterfactual Rotation Advantage v5
building OOS temporal-memory block
fitting expanding temporal rotation memory (N block(s))
simulating OOS expanding-memory rotation portfolio
```

## Output

```text
research_output/
  counterfactual_rotation_advantage_v5_strategy_10_2020-07-22_to_2026-09-03/

counterfactual_rotation_advantage_v5_strategy_10_2020-07-22_to_2026-09-03.zip
```

Important diagnostics include:

```text
cra5_rotation_advantage
cra5_positive_advantage
cra5_training_memory_blocks
cra5_training_memory_decision_dates
cra5_label_horizon_sessions
```

The fitted model metadata also records every calibration source block, its dates,
sample count, independent decision-date count, target mean, target dispersion and
positive-target fraction.

## Interpretation

The main v5 question is not whether rotations fall. It is whether the expanding
memory prevents one short calibration regime from dominating the following outer
test fold.

The most important comparisons are:

- final compounded capital versus the US$23.52M baseline;
- fold 2 capital multiple versus v4;
- predicted rotation-advantage distribution by fold;
- fraction of changed Top-1 decisions accepted/rejected by fold;
- maximum consecutive non-Top-1 HOLD sequence;
- geometric trade return and win rate;
- maximum drawdown.

If v5 materially repairs fold 2 while preserving v4 improvements in folds 1 and
3, the regime-imprinting hypothesis is supported.

If v5 remains far below the baseline, the next hypothesis should not add another
horizon or threshold. The next investigation should address the distribution of
counterfactual incumbent states versus states actually visited by a realizable
policy.
