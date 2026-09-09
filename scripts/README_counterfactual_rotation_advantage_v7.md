# Counterfactual Rotation Advantage v7.0.0

## Research question

CRA v6 tested whether the persistent ROTATE-versus-HOLD model generalizes better
when it is trained only on states visited by one feasible calibration trajectory.

The result exposed an estimator-capacity failure before that hypothesis could be
evaluated cleanly:

- fold 1 final reachable dataset: 28 samples
- fold 1 state features: 37
- fold 1 target standard deviation: about 12.56%
- fold 1 in-sample RMSE: about 12.56%
- fold 1 OOS rotation-advantage prediction collapsed to one negative constant
- fold 1 OOS rotations: 0

The v6 final model reused the LightGBM hyperparameters protected for the much larger
Opportunity Model dataset. With a reachable dataset smaller than the feature count,
that tree model can become degenerate.

v7 changes only the final decision estimator.

## Frozen components

The following remain unchanged from CRA v6:

- frozen 56-asset universe
- LightGBM Opportunity Model
- Opportunity Model target and hyperparameters
- outer walk-forward folds
- transaction costs
- state features
- policy-reachable calibration trajectory
- v4-equivalent LightGBM seed model used only to collect the feasible trajectory
- persistent ROTATE-versus-HOLD target
- target horizon = `min(rotation_target_horizons)` (currently 5 sessions)
- CASH excluded
- 100% market exposure after initial entry
- decision boundary = economic indifference at zero advantage
- no switch margin
- no minimum-holding rule
- no manual confidence threshold

## Isolated change

v6 final estimator:

```text
LightGBM using Opportunity Model hyperparameters
```

v7 final estimator:

```text
StandardScaler
    ↓
BayesianRidge
```

Bayesian Ridge is a regularized probabilistic linear regression. Its coefficient
and noise precisions are inferred from the observed calibration data. The research
does not introduce a hand-picked regularization coefficient or a new trading
threshold.

The decision remains:

```text
posterior mean rotation advantage > 0
    ROTATE current incumbent → current LightGBM Top-1

otherwise
    HOLD incumbent
```

## Why this experiment is necessary

In v6, fold 1 had only 28 policy-reachable rotation states for 37 features. Its
LightGBM final fit did not reduce training RMSE below the target standard deviation,
and every OOS challenger received the same negative value. That is estimator
degeneration, not evidence that every rotation in the two-year fold was economically
bad.

v7 tests the original v6 state-support hypothesis with an estimator designed to
remain identifiable in small-sample/high-dimensional conditions.

## New diagnostics

Each fold records:

- `sample_count`
- `decision_date_count`
- `seed_sample_count`
- `seed_decision_date_count`
- `feature_count`
- `sample_feature_ratio`
- `posterior_noise_precision`
- `posterior_weight_precision`
- `coefficient_l2_norm`
- `fit_mae`
- `fit_rmse`

Daily diagnostics include:

- `cra7_rotation_advantage`
- `cra7_positive_advantage`
- `cra7_training_decision_dates`
- `cra7_seed_decision_dates`
- `cra7_sample_feature_ratio`

## First falsification test

Run only the frozen 56-asset series:

```bash
python scripts/research_counterfactual_rotation_advantage_v7.py \
  --strategy-sequence 10 \
  --history-start 2016-01-01 \
  --snapshot-end 2026-09-04 \
  --analysis-start 2020-07-22 \
  --analysis-end 2026-09-03 \
  --series 56 \
  --workers 4
```

Do not run 500 assets until the 56-asset test is informative.

## What to inspect first

The first check is fold 1. Unlike v6, the predicted rotation advantage must not
collapse to a single negative constant merely because the final reachable dataset
contains fewer samples than features.

Then compare:

1. ending capital and fold capital path
2. number of rotations
3. fraction of Top-1 changes accepted
4. distribution of predicted advantages
5. maximum consecutive non-Top-1 HOLD sequence
6. trade geometric return and cycles per year

A higher rotation count is not itself success. The objective is to recover valuable
compound cycles while avoiding rotations whose expected incremental value is negative.
