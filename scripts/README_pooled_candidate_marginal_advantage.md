# Pooled Candidate Marginal Advantage v1

Research-only experiment. Version: `pooled-candidate-marginal-advantage-v1.0.0`.

## Hypothesis

The 56-asset Strategy baseline remains immutable. A useful external asset is not globally good or bad; its value depends on the market state and on the action that the protected 56-asset baseline would take at that exact decision.

This v1 therefore stops asking whether a candidate deserves permanent membership. It tests whether one pooled model can learn an **asset-agnostic signature** of when a candidate has positive marginal economic value versus the baseline action.

## Frozen behavior

- The Strategy #10 baseline assets remain unchanged.
- Baseline LightGBM Utility, production rotation features, transaction costs, walk-forward folds, switch-margin calibration and baseline policy are unchanged.
- Candidate ticker identity is not a model feature.
- No production API mode is changed.
- No frontend change is included.

## Candidate pool

The run reads no previous research artifact. Before the untouched validation period is loaded, it freezes all external symbols already cached in the local MongoDB that have Full Strategy History through `selection_end`.

Candidates that cannot fit the required LightGBM model in one fold simply do not contribute samples in that fold. The immutable 56-asset baseline must remain fully trainable.

## Exact target

For every baseline decision date and every trainable external candidate, the research computes two counterfactual next-session actions from the **same baseline current position**:

1. the action chosen by the immutable 56-asset baseline;
2. switch to the external candidate.

Both use the production transition-cost function. The supervised target is:

`candidate next-session net log return - baseline-action next-session net log return`

This is intentionally a one-decision target so it matches the frequency at which a future overlay could reconsider its action. It does not reuse the failed fixed multi-session action-value target from the earlier CAV experiments.

## Features

The pooled model sees only information available at decision time:

- candidate LightGBM score;
- candidate score normalized against its own pre-test fit distribution;
- protected baseline target score and its cross-sectional z-score;
- baseline current score and target-vs-current gap;
- baseline score mean, dispersion and positive-score breadth;
- baseline holding days;
- candidate-minus-baseline-target differences for every production `ROTATION_FEATURES` field.

Ticker identity is excluded, so the model cannot memorize UTI, LAND, PB or any other symbol. The purpose is to learn a reusable relative market-state signature.

## Estimator

`StandardScaler + BayesianRidge` (Bayesian regularized linear regression).

Each decision date receives total sample weight 1.0, divided equally among the available candidates on that date. Dates with a larger candidate universe therefore do not automatically dominate training.

There is no hand-tuned confidence gate. Candidate override is considered only when:

`predicted marginal advantage > 0`

Zero is economic indifference. Posterior predictive standard deviation is persisted for diagnostics but is not used as an arbitrary penalty in v1.

## Chronological validation

The pooled model is evaluated before the final holdout:

- train on fold 1 -> evaluate fold 2;
- train on folds 1+2 -> evaluate fold 3.

Then one model is fit on all pre-validation samples and evaluated on the untouched final 252 XNYS sessions by default.

At each evaluation date, the candidate with the highest positive predicted marginal advantage is selected counterfactually; otherwise the protected baseline action remains unchanged.

## Scope boundary

v1 deliberately does **not** claim portfolio-capital improvement. Candidate overrides would alter the actual incumbent state, so a stateful exact simulation is required before comparing final capital.

If the chronological cross-fit and untouched holdout show a positive generalizing signal, the next isolated version will integrate this model as a research-only stateful overlay and run an exact capital backtest versus the immutable 56-asset baseline.

## Run from zero

```bash
git fetch origin
git switch research/pooled-candidate-marginal-advantage-v1
git pull --ff-only origin research/pooled-candidate-marginal-advantage-v1

python scripts/research_pooled_candidate_marginal_advantage.py \
  --strategy-sequence 10 \
  --history-start 2016-01-01 \
  --snapshot-end 2026-09-04 \
  --validation-sessions 252 \
  --fresh-run
```

Main outputs:

- `pcma_selection_history_integrity.csv`
- `pcma_prevalidation_samples.csv`
- `pcma_fold_model_diagnostics.csv`
- `pcma_crossfit_scored_samples.csv`
- `pcma_crossfit_decisions.csv`
- `pcma_crossfit_summary.json`
- `pcma_holdout_samples.csv`
- `pcma_holdout_scored_samples.csv`
- `pcma_holdout_decisions.csv`
- `pcma_candidate_summary.csv`
- `pcma_result.json`
- `experiment_manifest.json`
