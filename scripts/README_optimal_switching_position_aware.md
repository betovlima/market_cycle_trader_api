# Position-Aware Optimal Switching v1.1.0

## Research question

Does explicit causal information about the currently held position improve the Fitted Q Iteration decision between HOLD, ROTATE and CASH without changing the LightGBM opportunity model?

This experiment follows the rejected Optimal Switching FQI v1.0.0. The v1.0.0 result is preserved in Git history and is not used as training data.

## What changes

Only the FQI state construction changes. The LightGBM target, features, hyperparameters, folds, dates, costs, universe and Buy & Hold benchmark remain unchanged.

The decision state now includes causal position-history fields reconstructed only from information available at the decision date:

- holding days;
- return since entry;
- peak return since entry;
- drawdown from the position peak;
- maximum favorable excursion so far;
- maximum adverse excursion so far;
- LightGBM score change since entry;
- days outside Top-1;
- consecutive days outside Top-1;
- fraction of holding days outside Top-1.

The action space remains:

```text
CASH + HOLD + Top-5 LightGBM challengers
```

The Bellman iteration count remains 20 and the discount factor continues to be derived from the Strategy target horizons.

## Calibration-state construction

v1.0.0 expanded many arbitrary current-position/action combinations on each calendar date, creating a large apparent sample count from a much smaller number of independent market dates.

v1.1.0 constructs causal position states from deterministic baseline-policy trajectories over the calibration window. The configured switch-margin candidates provide multiple behavior trajectories. Counterfactual one-session rewards are still available because the complete historical prices of each eligible action are known.

This keeps the state tied to a plausible position path and allows the FQI to observe actual holding history instead of treating two economically different positions as the same state.

## Recommended experiment

Use the same frozen Universe Scale baseline ZIP and compare only the control universe and the largest universe first:

```bash
git fetch origin
git checkout research/optimal-switching-position-aware-v1
git pull --ff-only

python scripts/research_optimal_switching_position_aware.py \
  --strategy-sequence 10 \
  --history-start 2016-01-01 \
  --snapshot-end 2026-09-04 \
  --analysis-start 2020-07-22 \
  --analysis-end 2026-09-03 \
  --series 56 500 \
  --workers 4
```

The default output folder is:

```text
research_output/optimal_switching_position_aware_strategy_10_2020-07-22_to_2026-09-03
```

The full folder is archived automatically as a ZIP.

## Interpretation

The main comparison is against the immutable Universe Scale baseline:

- 56 assets: approximately US$ 23.52M baseline capital;
- 500 assets: approximately US$ 31.66K baseline capital.

The experiment should be judged by final capital, CAGR, Sharpe, maximum drawdown, fold consistency, rotations, CASH usage and the distribution of Q(HOLD), Q(ROTATE) and Q(CASH).

A result that improves only one fold but fails the others is evidence of regime instability, not a successful policy.

## Reporting fixes

The position-aware entrypoint also corrects two reporting-only issues from v1.0.0:

- FQI action counts are read from the emitted `fqi_selected_action` diagnostic;
- maximum drawdown comparison accepts the engine field `strategy_maximum_drawdown`.

These fixes do not alter the trading policy or economic simulation.
