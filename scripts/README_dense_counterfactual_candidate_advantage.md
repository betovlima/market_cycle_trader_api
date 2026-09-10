# Dense Counterfactual Candidate Advantage v1

Research-only experiment. Version: `dense-counterfactual-candidate-advantage-v1.0.0`.

Base: PCEA v2 with policy-sufficient holding-state reconvergence, itself based on
`research/asset-marginal-rotation-contribution-v2`.

## Why this research exists

The corrected PCEA run still failed to generalize, but it exposed a more basic
sampling problem. PCEA can only learn from dates on which an external candidate
already wins the raw cross-symbol LightGBM competition strongly enough to alter
the `56 + candidate` policy naturally.

In the observed corrected run, only 160 usable natural episodes were produced
from 32,454 pre-validation candidate-date opportunities (about 0.49%). The
validation period had 31 usable natural episodes from 6,552 candidate-date
opportunities (about 0.47%).

That is both sparse and selection-biased: the training set contains almost only
the same extreme raw-score events whose comparability across independently
trained per-symbol LightGBM models is under investigation.

DCCA v1 changes one failure family only:

> replace natural candidate-led episode sampling with dense counterfactual
> candidate-date interventions from the exact immutable baseline prefix state.

The pooled estimator, ticker exclusion, decision-time features, LightGBM models,
transaction-cost model, minimum-holding guard and zero economic indifference
boundary remain unchanged.

## Counterfactual question

For each legal candidate-date pair:

1. Replay the immutable 56-asset policy up to date `t`.
2. Do not allow the candidate to affect any decision before `t`.
3. If the production minimum-holding guard is active at `t`, no intervention
   sample is created.
4. Otherwise force the external candidate at `t`, regardless of whether its raw
   LightGBM score is Top-1.
5. From `t+1` onward, let the normal Utility policy run on `56 + candidate`.
6. End the label when the forced path reconverges with the immutable baseline on
   the policy-sufficient state: same selected asset and holding age saturated at
   `rotation_min_holding_days`.
7. The target is cumulative account-level net-log-growth difference over that
   complete episode, including transaction costs and the reconvergence session.
8. Right-censored episodes are exported but excluded from training.

This asks directly:

`What would the economic consequence have been if candidate X had been chosen now?`

It no longer requires the raw candidate score to prove itself before the sample
is allowed to exist.

## Model and features

Unchanged from PCEA/PCMA:

- one pooled model across all candidates;
- ticker identity is not a feature;
- `StandardScaler + BayesianRidge`;
- the 64 decision-time features are unchanged;
- candidate raw score, train-relative z-score and percentile remain diagnostic
  inputs rather than an admission rule;
- zero is the only learned-decision boundary:
  `predicted marginal episode advantage > 0`.

No stop-loss, profit target, confidence cutoff, z-score threshold, minimum event
count or manually tuned candidate margin is introduced.

## Validation semantics

Pre-validation remains chronological:

- train fold 1 -> evaluate fold 2;
- train folds 1+2 -> evaluate fold 3.

When several candidate interventions are predicted positive on the same date,
the largest predicted advantage is chosen. Its counterfactual path is followed
until state reconvergence before another intervention may start. The realized
episode end is used only to replay the already chosen intervention; it does not
participate in the decision at its start.

The 2025-09-05 to 2026-09-04 period has already been observed in earlier
research. It is therefore reported as **retrospective validation**, not as a new
untouched holdout claim. It is still excluded from model fitting.

The oracle summaries use realized future labels only as a diagnostic upper
bound. They are never used for training or operational decisions.

## Fresh run

```bash
git fetch origin
git switch research/dense-counterfactual-candidate-advantage-v1
git pull --ff-only origin research/dense-counterfactual-candidate-advantage-v1

python -m unittest discover \
  -s tests \
  -p "test_dense_counterfactual_candidate_advantage.py" \
  -v

python scripts/research_dense_counterfactual_candidate_advantage.py \
  --strategy-sequence 10 \
  --history-start 2016-01-01 \
  --snapshot-end 2026-09-04 \
  --validation-sessions 252 \
  --fresh-run
```

No prior PCMA, PCEA or marginal research output is read. The persisted inputs are
the local MongoDB market cache and Strategy definition.

## Main outputs

- `dcca_selection_history_integrity.csv`
- `dcca_prevalidation_episode_samples.csv`
- `dcca_prevalidation_all_forced_episodes.csv`
- `dcca_fold_diagnostics.csv`
- `dcca_crossfit_scored_episodes.csv`
- `dcca_crossfit_decisions.csv`
- `dcca_crossfit_summary.json`
- `dcca_validation_episode_samples.csv`
- `dcca_validation_all_forced_episodes.csv`
- `dcca_validation_scored_episodes.csv`
- `dcca_validation_decisions.csv`
- `dcca_candidate_summary.csv`
- `dcca_result.json`
- `experiment_manifest.json`

The first decision is whether dense counterfactual sampling produces a stable
out-of-sample relationship between predicted and realized episode advantage.
Only if that signal generalizes should a later research version integrate it
into one exact continuously stateful overlay account and compare final capital
against the immutable 56-asset baseline.
