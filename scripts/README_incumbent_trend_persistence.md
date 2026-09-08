# Incumbent Trend Persistence v1

## Question

Can the same LightGBM rotation engine avoid unnecessary rotations when the currently held asset still shows causal evidence of an intact upward trend?

This study does **not** try to predict the exact market top. It tests whether the incumbent should receive temporary protection when its trend remains healthy and the challenger is not overwhelmingly better.

## What changes

Only the final out-of-sample decision policy receives an incumbent-persistence guard. The following remain unchanged from the baseline universe-scale experiment:

- LightGBM target and features
- LightGBM hyperparameters and random seed
- walk-forward folds and purge
- switch-margin calibration
- dates, capital, costs and execution at the next open
- frozen asset universe
- Buy & Hold benchmark

The guard can block an otherwise valid `ROTATE_TO_BEST_ASSET` decision when all conditions hold:

1. the incumbent is still ranked in the Top 5 of the same-date universe;
2. the challenger advantage is at most 1.0 same-date universe standard deviation above the incumbent;
3. at least 4 of 5 causal continuation signals are positive:
   - `return_5 > 0`
   - `return_20 > 0`
   - `ema_5_vs_20 > 0`
   - `ema_slope_20_5 > 0`
   - `channel_position_20 >= 0.5`

No forward target or future price is read by the guard.

## Frozen universe

The experiment reuses `universe_candidate_history_integrity.csv` from the completed baseline universe-scale run. Only rows that were both `history_complete=true` and `accepted=true` are loaded, in their original accepted order.

No candidate discovery or substitution is allowed. If one frozen external asset cannot be reproduced, the experiment stops instead of silently changing the comparison universe.

## Recommended first run

Run only the control and the largest universe first. This answers the main question with substantially less compute than repeating every intermediate series:

```bash
git fetch origin
git checkout research/incumbent-trend-persistence-v1
git pull --ff-only

python scripts/research_incumbent_trend_persistence.py \
  --strategy-sequence 10 \
  --history-start 2016-01-01 \
  --snapshot-end 2026-09-04 \
  --analysis-start 2020-07-22 \
  --analysis-end 2026-09-03 \
  --series 56 500 \
  --workers 4
```

The default baseline folder is:

```text
research_output/asset_rotation_universe_scale_strategy_10_2020-07-22_to_2026-09-03
```

Use `--baseline-output-dir` only if the completed baseline artifacts are elsewhere.

## Outputs

The experiment writes to:

```text
research_output/incumbent_trend_persistence_strategy_10_2020-07-22_to_2026-09-03
```

In addition to the normal universe artifacts, it writes:

- `incumbent_persistence_comparison.csv`
- `incumbent_persistence_comparison.json`

The comparison includes baseline versus persistence capital, CAGR, Sharpe, maximum drawdown, rotations and the number of rotation decisions blocked by the persistence guard.

At completion the full output folder is also archived automatically as a ZIP.

## Interpretation

The first decision rule is deliberately simple:

- if 56 deteriorates materially, the guard is probably overprotecting incumbents;
- if 56 stays close to baseline and 500 improves materially, the hypothesis gains support;
- if 500 does not improve, the false-leader problem is probably not solved by incumbent persistence alone.

Do not tune these thresholds on the already observed final validation period and then call that period independent validation.
