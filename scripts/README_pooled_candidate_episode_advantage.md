# Pooled Candidate Episode Advantage v2

Research-only experiment. Current patch: `pooled-candidate-episode-advantage-v2.0.4`.

Base branch: `research/pooled-candidate-marginal-advantage-v1`, itself based on `research/asset-marginal-rotation-contribution-v2`.

## Why this version exists

PCMA v1 tested a pooled, ticker-agnostic model with a one-session marginal target:

`candidate next-session net log return - baseline-action next-session net log return`

That target/model combination failed to generalize. PCEA v2 isolates one hypothesis:

> the one-session target is too noisy and too short relative to the multi-horizon LightGBM Utility policy; a candidate should instead be judged over the complete stateful interval during which adding it changes the protected 56-asset policy state.

The pooled estimator, candidate-identity exclusion, features, LightGBM snapshot, transaction costs, baseline policy and zero economic indifference point remain unchanged.

## v2.0.4 correctness patch — policy-sufficient reconvergence

The first PCEA v2 implementation compared the **exact** holding-day counters of the baseline and expanded paths when deciding whether they had reconverged. That was stricter than the actual Utility policy.

The policy only tests whether `holding_days < rotation_min_holding_days`. Once the minimum holding period has been reached, exact ages such as 2, 3 or 20 sessions are behaviorally equivalent for future decisions. Requiring exact equality could therefore keep an episode open long after both paths had already returned to the same decision state, contaminating the episode target with unrelated future returns.

Patch v2.0.4 fixes this without changing the research hypothesis:

- cached-score replay exports the configured `rotation_min_holding_days` as replay metadata;
- episode extraction requires that metadata and refuses older replay objects without it;
- holding age is saturated at the configured minimum when state equivalence is tested;
- the selected asset must still be identical;
- the reconvergence session remains included, so switch-back costs remain in the target;
- no manual economic threshold, stop-loss, confidence gate or profit target is introduced.

Because this changes episode boundaries and labels, **all PCEA artifacts produced before v2.0.4 are invalid for this experiment and must not be reused**. Run with `--fresh-run`.

## Target: stateful divergence episode

For each fold and each trainable external candidate:

1. Fit the same pre-period LightGBM models used by PCMA v1.
2. Build the same decision-time pooled features.
3. Export a cached OOS score/execution tape.
4. Replay the immutable 56-asset universe with exact policy state, costs, quantity rounding and next-open execution.
5. Replay `56 + candidate` on the identical calendar and settings.
6. Start an episode when the expanded replay directly selects the candidate while the protected baseline selects something else.
7. Continue the episode until both replays reconverge to the same **policy-sufficient** state: same selected asset and same minimum-holding state after saturation at `rotation_min_holding_days`.
8. The target is the cumulative account-level difference:

`sum(expanded daily net log return - baseline daily net log return)`

over that complete divergence episode.

Right-censored episodes that have not reconverged before the end of a fold/holdout are excluded rather than assigned an incomplete future label.

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
- evaluate on the final 252 XNYS sessions used by the experiment;
- no holdout target is used in model fitting, candidate discovery or margin calibration.

The 2025-09-05 to 2026-09-04 holdout has already been observed in prior research iterations, so a rerun after v2.0.4 is a retrospective correctness comparison, not a new untouched performance claim.

## Run from zero

```bash
git fetch origin
git switch research/pooled-candidate-episode-advantage-v2
git pull --ff-only origin research/pooled-candidate-episode-advantage-v2

python -m unittest discover -s tests -p "test_pooled_candidate_episode_advantage.py" -v

python scripts/research_pooled_candidate_episode_advantage.py \
  --strategy-sequence 10 \
  --history-start 2016-01-01 \
  --snapshot-end 2026-09-04 \
  --validation-sessions 252 \
  --fresh-run
```

No prior PCMA, PCEA or marginal research output is reused. The only persisted source is the local MongoDB market cache and Strategy definition.

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

Each newly extracted episode also carries `policy_min_holding_days` and `episode_definition_version=pcea-2.0.4-policy-sufficient-holding-state` so exported results can be distinguished from the invalid pre-patch episode definition.
