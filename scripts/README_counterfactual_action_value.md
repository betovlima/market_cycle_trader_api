# Counterfactual Action Value v1.0.0

## Research question

Can the existing LightGBM opportunity ranker keep identifying the asset with the
highest predicted future growth while a separate supervised model learns whether
portfolio capital should **HOLD the incumbent, ROTATE/ENTER the current Top-1, or
remain/go to CASH**?

The experiment intentionally adds **no hand-written switch margin, CASH threshold,
or minimum-holding rule**. The learned layer compares predicted economic value.

## Why this experiment exists

The production baseline compounds strongly with the original 56-asset universe,
but the portfolio can rotate too aggressively in two opposite situations:

- broad market strength: it repeatedly chases the newest "best of the best";
- broad market weakness: a relative ranking still produces a Top-1 even when all
  available assets may have unattractive forward value.

Fitted Q Iteration v1 tried to solve the full sequential problem at once and was
rejected after its learned Q-values failed to separate actions reliably. This
experiment keeps the useful state/action representation but removes Bellman
backups and learns the directly observable counterfactual economic value.

## What remains frozen

- asset discovery / frozen universe;
- LightGBM opportunity target and hyperparameters;
- walk-forward fold boundaries;
- costs and exact execution simulator;
- Strategy target horizons and horizon weights;
- market snapshot / history used by the Universe Scale baseline.

## Learned action space

For each decision state the supervised layer evaluates only the decisions needed
to test the current failure mode:

1. `CASH`;
2. `HOLD` the current position, when invested;
3. `ROTATE/ENTER` the **current LightGBM Top-1**.

There is no arbitrary Top-K decision shortlist in v1. LightGBM remains responsible
for identifying the best growth candidate; the new model decides whether acting
on that ranking adds value relative to HOLD or CASH.

## Counterfactual target

For every calibration decision and feasible current position, the research engine
computes the exact net log return that would have resulted from each action over
the Strategy's already-frozen target horizons.

The target is the Strategy-weighted average of cumulative net log returns across
those horizons. Switching costs are naturally present in the first transition;
continuing to hold does not pay a synthetic switching penalty; CASH has its actual
exit/entry economics.

No test-period observation is used to create labels. Because the label needs the
largest Strategy horizon, the last `max(rotation_target_horizons)` calibration
sessions are purged before fitting the action-value model.

## Decision rule

At test time the model receives only information available on the decision date
and predicts the economic value of each feasible action.

```text
action* = argmax predicted_action_value(state, action)
```

No additional margin or threshold is applied. Transaction costs are already part
of the learned target.

## Recommended first run

Run the 56-asset control first. It is the fastest and strongest falsification test
because the frozen baseline is approximately US$23.5M.

```bash
git fetch origin
git switch research/counterfactual-action-value-v1
git pull --ff-only

python scripts/research_counterfactual_action_value.py \
  --strategy-sequence 10 \
  --history-start 2016-01-01 \
  --snapshot-end 2026-09-04 \
  --analysis-start 2020-07-22 \
  --analysis-end 2026-09-03 \
  --series 56 \
  --workers 4
```

Then test scale only if the 56-asset result is informative:

```bash
python scripts/research_counterfactual_action_value.py \
  --strategy-sequence 10 \
  --history-start 2016-01-01 \
  --snapshot-end 2026-09-04 \
  --analysis-start 2020-07-22 \
  --analysis-end 2026-09-03 \
  --series 56 500 \
  --workers 4
```

If the baseline ZIP is outside the default `research_output` location, add:

```bash
--baseline-output-dir "/path/to/asset_rotation_universe_scale_strategy_10_2020-07-22_to_2026-09-03.zip"
```

## Outputs

Default output folder:

```text
research_output/counterfactual_action_value_strategy_10_2020-07-22_to_2026-09-03
```

Additional artifacts:

```text
counterfactual_action_value_comparison.csv
counterfactual_action_value_comparison.json
```

The decision diagnostics also expose:

```text
cav_selected_value
cav_hold_value
cav_cash_value
cav_best_candidate_value
cav_selected_edge_vs_hold
cav_selected_edge_vs_cash
```

The comparison report counts HOLD / ROTATE / ENTER / CASH decisions and also
measures how often the learned policy refuses to chase a changed Top-1.

## Interpretation

The experiment is successful only if OOS evidence supports it. Do not introduce a
manual pass/fail business gate. Compare at minimum:

- final compounded capital;
- fold-by-fold capital and consistency;
- CAGR, Sharpe and maximum drawdown;
- rotation count;
- CASH sessions;
- HOLD decisions when Top-1 changes;
- periods where CASH beats acting on a weak Top-1.

A lower rotation count is not itself a goal. The goal is to remove rotations whose
learned counterfactual value is inferior while preserving rotations that drive
compound growth.
