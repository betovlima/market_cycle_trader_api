# Optimal Switching FQI v1.0.0

## Research question

Can a sequential action-value policy choose among HOLD, ROTATE and CASH more effectively than the instantaneous utility-ranking policy, while keeping the same LightGBM opportunity model and the same frozen asset universe?

This experiment is the next step after the rejected Incumbent Trend Persistence v1. The failed persistence guard showed that simply holding a position longer when recent trend signals remain positive can reduce useful rotations and destroy compound growth.

## Method

The decision layer uses **Fitted Q Iteration (FQI)**, a standard offline reinforcement-learning / approximate dynamic-programming method.

The LightGBM opportunity models remain unchanged. They continue to estimate the cross-asset risk-adjusted utility used to identify plausible challengers.

For every fold:

1. LightGBM opportunity models are trained only on the chronological training period.
2. The existing calibration period is used to build state/action transitions.
3. FQI learns an approximation of `Q(state, action)` from those pre-test transitions.
4. Final LightGBM opportunity models are trained on the normal final-fit period.
5. The untouched fold test period is simulated using the learned FQI switching policy.

The FQI test period is never used for Q-function training.

## State

The state contains only information available at the decision date:

- current position / CASH status;
- current and target LightGBM utility;
- relative rank and cross-sectional utility statistics;
- 20-session market breadth;
- recent return, volatility, EMA, ATR, trend-efficiency and channel-position features for the current and target assets;
- explicit switching cost.

No forward target or future price is a state feature.

## Actions

The action set is:

- CASH;
- HOLD the current asset;
- ROTATE / ENTER one of the current Top-5 LightGBM challengers.

The Top-5 shortlist is fixed for this v1 and is not tuned on the test period. CASH and the incumbent are always included even when they are not in the Top-5.

## Reward

The immediate reward is the engine's existing exact next-session **net log return**, including the same switching transaction cost used by the research engine.

Therefore the decision layer does not invent a hand-written trend score or persistence bonus.

## Bellman update

FQI applies the standard Bellman backup:

```text
Q(s, a) = r(s, a) + gamma * max_a' Q(s', a')
```

The discount factor is not independently tuned. Its half-life is derived from the median target horizon already frozen in the Strategy:

```text
gamma = 0.5 ** (1 / median_target_horizon)
```

The first experiment performs 20 Bellman backup iterations.

## Frozen universe

The experiment reuses the completed Universe Scale baseline:

```text
asset_rotation_universe_scale_strategy_10_2020-07-22_to_2026-09-03
```

It accepts either the extracted folder or its ZIP archive.

`universe_candidate_history_integrity.csv` is used only to recover the exact previously accepted external symbols and their order. No candidate discovery or substitution is allowed.

## Recommended first run

Run only the control and largest universe:

```bash
git fetch origin
git checkout research/optimal-switching-fqi-v1
git pull --ff-only

python scripts/research_optimal_switching_fqi.py \
  --strategy-sequence 10 \
  --history-start 2016-01-01 \
  --snapshot-end 2026-09-04 \
  --analysis-start 2020-07-22 \
  --analysis-end 2026-09-03 \
  --series 56 500 \
  --workers 4
```

If the baseline ZIP is outside the default `research_output` location, add:

```text
--baseline-output-dir "/path/to/asset_rotation_universe_scale_strategy_10_2020-07-22_to_2026-09-03.zip"
```

## Recovery / finalize only

The entrypoint has a post-processing-only mode so a completed expensive run never has to be repeated merely because final comparison or ZIP generation failed:

```bash
python scripts/research_optimal_switching_fqi.py \
  --strategy-sequence 10 \
  --analysis-start 2020-07-22 \
  --analysis-end 2026-09-03 \
  --baseline-output-dir "./research_output/asset_rotation_universe_scale_strategy_10_2020-07-22_to_2026-09-03.zip" \
  --finalize-only
```

## Outputs

Default output folder:

```text
research_output/optimal_switching_fqi_strategy_10_2020-07-22_to_2026-09-03
```

Additional comparison artifacts:

```text
optimal_switching_fqi_comparison.csv
optimal_switching_fqi_comparison.json
```

At completion the entire output folder is archived automatically as a ZIP.

## Initial interpretation

The first test is intentionally not tuned.

Useful evidence would be:

- Universe 56 remains reasonably close to its baseline;
- Universe 500 materially improves versus its US$31.7K baseline;
- CASH appears in genuinely weak periods rather than remaining at zero days;
- rotations decrease only when the learned continuation value supports HOLD;
- Sharpe / MaxDD do not deteriorate enough to offset the capital improvement.

If the result fails, do not immediately add new manual rules. Inspect the learned Q decisions and the offline state/action coverage first.

## Research script cleanup

The existing safe cleanup checker remains:

```bash
bash scripts/cleanup_research_wrappers.sh
```

Do not run `--apply` until the old versioned research wrappers have been consolidated and the checker reports them as `READY`.
