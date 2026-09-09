# Pooled Candidate Episode Advantage v2

Research-only experiment. Version: `pooled-candidate-episode-advantage-v2.0.0`.

Base branch: `research/pooled-candidate-marginal-advantage-v1`, itself based on `research/asset-marginal-rotation-contribution-v2`.

## Why this version exists

PCMA v1 tested a pooled, ticker-agnostic model with a one-session marginal target:

`candidate next-session net log return - baseline-action next-session net log return`

The untouched result falsified that target/model combination. In the observed run, predicted advantage was effectively anti-informative: the highest predicted region had worse realized marginal return, while the posterior predictive uncertainty was much larger than the mean predicted advantage.

This version isolates one hypothesis only:

> the one-session target is too noisy and too short relative to the multi-horizon LightGBM Utility policy; a candidate should instead be judged over the complete stateful interval during which adding it actually changes the protected 56-asset policy state.

The pooled estimator, candidate-identity exclusion, features, LightGBM snapshot, transaction costs, baseline policy and zero economic indifference point remain unchanged.

## New target: stateful divergence episode

For each fold and each trainable external candidate:

1. Fit the same pre-period LightGBM models used by PCMA v1.
2. Build the same decision-time pooled features.
3. Export a cached OOS score/execution tape.
4. Replay the immutable 56-asset universe with exact policy state, costs, quantity rounding and next-open execution.
5. Replay `56 + candidate` on the identical calendar and settings.
6. Start an episode when the expanded replay directly selects the candidate while the protected baseline selects something else.
7. Continue the episode until both replays reconverge to the same observable policy state: selected asset and holding-days state.
8. The target is the cumulative account-level difference:

`sum(expanded daily net log return - baseline daily net log return)`

over that complete divergence episode.

The reconvergence session itself is included, so switch-back costs are part of the target.

Right-censored episodes that have not reconverged before the end of a fold/holdout are excluded rather than assigned an incomplete future label.

## Why this avoids the PCMA v1 problem

PCMA v1 asked whether a candidate beats the baseline on only the next session, even though the production Utility model is trained on 5/10/20/40/60-session opportunity information and the policy has state through minimum holding and prior position.

PCEA v2 labels the actual stateful economic consequence of a candidate intervention, including the path created by that intervention, until the policy state reconverges.

This also avoids summing overlapping forward labels: each candidate replay is chronological and each evaluation chooses non-overlapping episode starts.

## Model

Unchanged from PCMA v1:

- pooled model across all candidates;
- ticker identity is not a feature;
- `StandardScaler + BayesianRidge`;
- decision-time feature vector remains the PCMA v1 vector;
- zero is the only acceptance boundary: predicted episode advantage must be positive.

No confidence threshold, z-score cutoff, minimum event count, stop-loss, minimum profit or hand-tuned acceptance margin is added.

## Chronological validation

Pre-validation:

- train on fold 1 episode samples -> evaluate fold 2;
- train on folds 1+2 -> evaluate fold 3.

At evaluation time, if several candidate episodes start on the same session, the model chooses the largest positive predicted episode advantage. Once one episode is chosen, later episode starts are ignored until its replay-defined stateful episode ends. This prevents overlapping realized targets from being added as if they were independent capital paths.

Final holdout:

- fit the pooled episode model on all pre-validation episode samples;
- evaluate on the untouched final 252 XNYS sessions;
- no holdout target is used in model fitting, candidate discovery or margin calibration.

This v2 is still a signal/target validation. It does not yet claim an exact combined overlay portfolio result. If cross-fit and holdout are both positive, the next version should integrate the episode-admission signal into one exact stateful overlay account and compare final capital against the immutable 56-asset baseline.

## Run from zero

```bash
git fetch origin
git switch research/pooled-candidate-episode-advantage-v2
git pull --ff-only origin research/pooled-candidate-episode-advantage-v2

python scripts/research_pooled_candidate_episode_advantage.py \
  --strategy-sequence 10 \
  --history-start 2016-01-01 \
  --snapshot-end 2026-09-04 \
  --validation-sessions 252 \
  --fresh-run
```

No prior PCMA or marginal research output is reused. The only persisted source is the local MongoDB market cache and Strategy definition.

## Main outputs

- `pcea_selection_history_integrity.csv`
- `pcea_prevalidation_episode_samples.csv`
- `pcea_prevalidation_all_episodes.csv`
- `pcea_fold_diagnostics.csv`
- `pcea_crossfit_scored_episodes.csv`
- `pcea_crossfit_decisions.csv`
- `pcea_crossfit_summary.json`
- `pcea_holdout_episode_samples.csv`
- `pcea_holdout_all_episodes.csv`
- `pcea_holdout_scored_episodes.csv`
- `pcea_holdout_decisions.csv`
- `pcea_candidate_summary.csv`
- `pcea_result.json`
- `experiment_manifest.json`

Primary values to inspect:

- `prevalidation_crossfit_realized_marginal_log_sum`
- `prevalidation_crossfit_incremental_factor`
- `holdout_summary.realized_marginal_log_sum`
- `holdout_incremental_factor`
- `untouched_holdout_signal_positive`
