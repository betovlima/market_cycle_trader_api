# Dense Counterfactual Candidate Advantage v1.1

Research-only experiment.

- Research version: `dense-counterfactual-candidate-advantage-v1.1.0`
- API/package: `10.8.44`
- Branch: `research/dense-counterfactual-candidate-advantage-v1-1`
- Direct base: `research/pooled-candidate-episode-advantage-v2-1` at PCEA v2.1.1

## Why this branch exists

PCEA v2.1.1 corrected censoring, label maturity and inference contracts, but the observed run still showed no useful generalization from the naturally occurring candidate-led episodes. The natural episode sampler only observes a candidate after its raw per-asset LightGBM score has already won enough of the protected 56-asset policy to create a divergence. That leaves very few labels and conditions the learning set on the same cross-asset score competition we are trying to improve.

DCCA changes one research family only: **sampling density / selection bias**.

For each policy-legal `(decision date, external candidate)` pair, DCCA asks:

> What would have happened if this candidate were forced now from the exact protected-baseline prefix state, and the normal Utility policy then continued until the two paths reconverged?

The target is the cumulative account-level net-log-growth difference over that forced divergence episode. The candidate does not need to win the raw score ranking before the counterfactual is generated.

## What remains frozen

- immutable 56-asset baseline;
- same LightGBM Utility models and model settings;
- same 64 pooled decision-time features;
- candidate ticker identity is not a feature;
- same costs, next-open execution, quantity rules and minimum-holding policy;
- same `StandardScaler + BayesianRidge` pooled estimator;
- zero remains the only economic indifference boundary;
- no confidence threshold, z-score cutoff, stop-loss, minimum profit or hand-tuned margin is added.

## v1.1 censor-aware contract

This branch is based directly on PCEA v2.1.1 and deliberately carries its correctness rules into dense sampling:

- decision-time features are built with `include_future_targets=False`;
- every legal forced episode start remains available for inference, including right-censored episodes;
- right-censored episodes keep a missing completed target and a separate observed prefix;
- pooled training uses only finite completed episode targets;
- open episodes are never silently converted to zero return;
- if a selected episode remains open, the complete aggregate/factor is exported as `null` and completed-only plus observed-prefix diagnostics are reported separately;
- no prior PCEA, PCMA, marginal or DCCA result artifact is read by a fresh run.

The implementation reuses the original DCCA forced-replay engine as an internal base and applies the PCEA v2.1.1 inference/censoring contract in the public `research_dense_counterfactual_candidate_advantage.py` runner.

## Run from zero

```bash
git fetch origin
git switch research/dense-counterfactual-candidate-advantage-v1-1
git pull --ff-only origin research/dense-counterfactual-candidate-advantage-v1-1

python -m unittest discover -s tests -p "test_pooled_candidate_episode_advantage.py" -v
python -m unittest discover -s tests -p "test_dense_counterfactual_candidate_advantage.py" -v
python -m unittest discover -s tests -p "test_dense_counterfactual_candidate_advantage_v11.py" -v

python scripts/research_dense_counterfactual_candidate_advantage.py \
  --strategy-sequence 10 \
  --history-start 2016-01-01 \
  --snapshot-end 2026-09-04 \
  --validation-sessions 252 \
  --fresh-run
```

`--fresh-run` deletes only this DCCA experiment output directory and rebuilds the research from the local MongoDB market cache and Strategy definition.

## Main outputs

- `dcca_selection_history_integrity.csv`
- `dcca_fold_diagnostics.csv`
- `dcca_prevalidation_episode_samples.csv`
- `dcca_prevalidation_all_forced_episodes.csv`
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

## Decision rule for the research

The first question is not whether the portfolio beats the historical US$23.5M trajectory. DCCA v1.1 first tests whether dense counterfactual labels produce a generalizable candidate signature:

- cross-fit ranking/association between predicted and realized episode advantage improves materially versus PCEA;
- candidate overrides have positive completed economic contribution across chronological evaluation stages rather than from one isolated outlier;
- retrospective validation does not contradict the cross-fit direction.

If that signal generalizes, the next separate research family is one exact stateful overlay account that applies the learned admission decision and compares final capital directly against the immutable 56-asset baseline.
