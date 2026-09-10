# Cross-Asset Utility Signature v1

Research-only experiment. Version: `cross-asset-utility-signature-v1.0.0`.
API/package: `10.8.45`.

Branch:

`research/cross-asset-utility-signature-v1`

Base:

`research/dense-counterfactual-candidate-advantage-v1-1`
at `8e2a4caf103867c56226008a9fc8be8ba3955e30`.

## Why this experiment exists

DCCA v1.1 removed the natural-episode selection bias and increased the candidate
supervision from a few hundred naturally occurring episodes to tens of thousands
of forced candidate-date counterfactuals. The supplied DCCA result still showed
near-zero out-of-sample association between the 64 admission features and the
full stateful episode advantage.

This version does not add another threshold or another episode model. It tests a
different failure family:

> independently trained per-asset LightGBMs may produce Utility scores that are
> useful inside each asset but are not economically comparable across assets.

The production target is already multi-horizon and economic:
`forward_risk_adjusted_utility`. This experiment asks whether one pooled model,
trained only on the protected 56 assets and without ticker identity, can learn a
transferable technical-state signature on one common score scale.

## What stays protected

- the Strategy's 56 baseline assets;
- the original independent LightGBM policy used to determine the reference action;
- the protected LightGBM hyperparameter snapshot;
- the 5/10/20/40/60 target construction;
- transaction-cost/downside/drawdown/movement/persistence target definition;
- minimum holding and calibrated switch-margin logic of the reference action;
- no candidate ticker/identity feature;
- zero economic indifference as the only candidate-vs-reference boundary.

## What changes

A second research-only LightGBM is fitted on stacked historical rows from the
protected 56 assets:

`ROTATION_FEATURES -> forward_risk_adjusted_utility`

No external candidate target is used to fit this pooled model.

At each OOS decision:

1. the original protected per-asset policy chooses its reference action;
2. the pooled model scores that reference asset on the common scale;
3. the same pooled model scores every eligible external candidate;
4. the best external candidate is admitted by the signal only when its common-scale
   predicted Utility is greater than the common-scale predicted Utility of the
   protected reference action.

This experiment evaluates the realized difference in the same
`forward_risk_adjusted_utility` units. It is not a portfolio backtest and does not
compound those differences into capital.

## Why this is a signature test

The pooled model never receives the ticker. An external asset that the model has
never trained on can score well only if its technical state resembles cross-asset
patterns learned from the protected universe.

Therefore this version directly tests whether the states that made the protected
56 useful contain a transferable signature.

## Leakage controls

- candidate pool is frozen from the local MongoDB cache before validation is loaded;
- pooled model training uses protected baseline assets only;
- external candidate targets are evaluation-only;
- walk-forward final-fit labels must mature before the first OOS decision;
- final validation uses the existing strict 60-session purge;
- no prior PCMA/PCEA/DCCA/marginal result artifact is read;
- `--fresh-run` deletes only this experiment directory;
- no Alpaca network access or Mongo writes.

The 2025-09-05 to 2026-09-04 validation period has already been inspected in prior
research and is therefore retrospective, not untouched.

## Run

```bash
git fetch origin
git switch research/cross-asset-utility-signature-v1
git pull --ff-only origin research/cross-asset-utility-signature-v1

python -m unittest discover \
  -s tests \
  -p "test_cross_asset_utility_signature.py" \
  -v

python scripts/research_cross_asset_utility_signature.py \
  --strategy-sequence 10 \
  --history-start 2016-01-01 \
  --snapshot-end 2026-09-04 \
  --validation-sessions 252 \
  --fresh-run
```

## Main outputs

- `caus_selection_history_integrity.csv`
- `caus_prevalidation_baseline_common_scores.csv`
- `caus_prevalidation_candidate_common_scores.csv`
- `caus_prevalidation_decisions.csv`
- `caus_prevalidation_fold_summary.json`
- `caus_validation_baseline_common_scores.csv`
- `caus_validation_candidate_common_scores.csv`
- `caus_validation_decisions.csv`
- `caus_result.json`
- `experiment_manifest.json`

## Decision criteria

Do not promote or integrate this research because one aggregate is positive.

The useful evidence is:

- positive/consistent daily cross-sectional rank association on external candidates;
- pooled Top-1 candidate realized Utility better than the external cross-sectional mean;
- positive realized common-scale candidate advantage in multiple prevalidation folds;
- the same direction in the retrospective validation period.

If the common-scale signature generalizes, the next experiment is one exact
stateful overlay account. Only that later experiment can answer whether final
capital exceeds the protected 56-asset trajectory.
