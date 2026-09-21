# Alpaca × Tiingo Equivalence Audit

API v10.8.62 provides the isolated provider-audit workflow. It does not change the normal
backtest engine or production market-data routing.

## Goal

Answer two questions before any new tuning:

1. How different are the cached Alpaca and Tiingo OHLCV histories when the
   symbol, date, interval and adjustment contract are held constant?
2. When the exact same MCT model and parameters are replayed on the largest
   common modelable universe and common cutoff, when do the decisions first
   diverge?

## Data contract

The audit aligns daily bars and model decisions by trading-session date rather than by the provider-specific UTC timestamp. Alpaca daily bars may be stored at 04:00/05:00 UTC while Tiingo EOD bars are stored at 00:00 UTC for the same market session.

The audit reads only from MongoDB:

- Alpaca: `alpaca_market_bars`
- Tiingo: `tiingo_market_bars`

It makes no market-data API calls.

The most recent completed Tiingo Backtest job is used as the immutable source
of strategy/model/runtime settings. A specific job can be supplied with
`--job-id`.

The script automatically chooses the common modelable universe. An asset is
excluded when either provider has no data, starts outside the configured
history tolerance, or has fewer rows than the locked minimum training +
horizon + purge requirement. This is expected to exclude CLMT in the current
Tiingo snapshot rather than filling its history from Alpaca.

The comparison cutoff is the latest date that is simultaneously available for
every retained asset in both providers. Therefore the two controlled replays
use the same assets and same time window.

## Run

From the API project root:

```bash
python scripts/audit_alpaca_tiingo_equivalence.py
```

To audit only candles without running the two LightGBM replays:

```bash
python scripts/audit_alpaca_tiingo_equivalence.py --skip-model-replay
```

To bind the audit to one completed Tiingo job:

```bash
python scripts/audit_alpaca_tiingo_equivalence.py --job-id <JOB_ID>
```

## Outputs

Default directory:

`output/alpaca_tiingo_equivalence_audit`

Files:

- `summary.json`: common-universe contract, candle totals, controlled replay
  metrics and first decision divergence.
- `coverage.csv`: row counts and coverage status for every configured asset.
- `per_asset_equivalence.csv`: per-asset OHLCV difference statistics.
- `candle_differences.csv`: non-identical/materially different candles.
- `missing_dates.csv`: dates present in only one provider.
- `alpaca_predictions.csv` / `tiingo_predictions.csv`: controlled model outputs.
- `decision_divergences.csv`: dates where the selected/final action differs.
- `model_input_equivalence.csv`: per-asset/per-feature and target differences after the exact model preprocessing.
- `first_divergence_model_inputs.csv`: Alpaca vs Tiingo model inputs for the assets involved in the first divergent decision.
- `fold1_pretest_model_input_equivalence.csv`: feature/target equivalence restricted to sessions before the first out-of-sample test session.
- `fold1_pretest_model_input_anomalies.csv`: largest pre-test feature/target deltas by asset and column.
- `fold1_initial_training_input_equivalence.csv`: exact first-fold calibration-model training window.
- `fold1_calibration_input_equivalence.csv`: exact first-fold policy calibration window.
- `fold1_final_fit_input_equivalence.csv`: exact first-fold final LightGBM fit window used for OOS scores.
- `fold1_phase_input_anomalies.csv`: top feature/target deltas tagged by training phase.
- `alpaca_trades.csv` / `tiingo_trades.csv`: controlled trade sequences.

Material-difference flags are descriptive only: price differences above 1 bp
(0.01%) and volume differences above 1%. They do not alter the model or filter
the replay.

No parameter optimization is performed by this audit.


API v10.8.58 note: fold-phase timestamps are normalized to UTC session dates before intersecting Alpaca and Tiingo model-input panels. This prevents 04:00/05:00 Alpaca daily timestamps from producing empty phase-equivalence outputs against 00:00 Tiingo EOD timestamps.


## Fresh Alpaca snapshot verification (API v10.8.60)

The preserved collection `alpaca_market_bars` is read-only for this experiment.

Download a new snapshot directly from the Alpaca API into a separate collection:

```bash
python scripts/download_fresh_alpaca_snapshot.py \
  --job-id 20260918T234903-52bd06f3
```

Default destination:

```text
alpaca_market_bars_fresh_20260919
```

The downloader refuses to use `alpaca_market_bars` as a target. It stores a snapshot manifest and SHA-256 in `market_data_snapshot_manifests`.

Compare the preserved Alpaca cache against the fresh API snapshot without running the model:

```bash
python scripts/compare_alpaca_snapshots.py \
  --job-id 20260918T234903-52bd06f3
```

The equivalence audit can also use the fresh snapshot explicitly:

```bash
python scripts/audit_alpaca_tiingo_equivalence.py \
  --job-id 20260918T234903-52bd06f3 \
  --alpaca-collection alpaca_market_bars_fresh_20260919
```


## Fresh Alpaca split-only snapshot (API v10.8.61)

To isolate dividend adjustments while preserving split continuity, download a new Alpaca snapshot with `adjustment=split`:

```bash
python scripts/download_fresh_alpaca_snapshot.py \
  --job-id 20260918T234903-52bd06f3 \
  --adjustment split
```

Default destination:

```text
alpaca_market_bars_split_20260919
```

The legacy `alpaca_market_bars` collection remains protected and is never overwritten.

Run the controlled replay using the split-only Alpaca snapshot:

```bash
python scripts/audit_alpaca_tiingo_equivalence.py \
  --job-id 20260918T234903-52bd06f3 \
  --alpaca-collection alpaca_market_bars_split_20260919 \
  --alpaca-adjustment split \
  --output-dir output/alpaca_split_tiingo_equivalence_audit
```

The Alpaca replay metadata records `split`; the Tiingo control keeps the adjustment stored by the original Tiingo job. Raw candle-equivalence statistics between these two legs should therefore be treated as diagnostic only because their adjustment semantics differ. The main objective of this run is the controlled Alpaca split-only model result.


## Point-in-time corporate actions experiment (API v10.8.62)

This experiment separates observed market prices from corporate-action context.

Architecture:

```text
Alpaca RAW OHLCV (immutable)
        +
Alpaca corporate actions snapshot
        |
        +-- forward/reverse split -> local continuity normalization
        |
        +-- cash dividend -> point-in-time LightGBM features
```

The first version deliberately changes only one research family: dividend context in the model inputs. The target stays price-based and dividends are not yet credited to portfolio cash.

### 1. Download a fresh RAW snapshot

```bash
python scripts/download_fresh_alpaca_snapshot.py \
  --job-id 20260918T234903-52bd06f3 \
  --adjustment raw
```

Default collection:

```text
alpaca_market_bars_raw_20260919
```

### 2. Freeze corporate actions

```bash
python scripts/download_alpaca_corporate_actions_snapshot.py \
  --job-id 20260918T234903-52bd06f3
```

Default collection:

```text
alpaca_corporate_actions_20260919
```

The REST snapshot includes forward/reverse/unit splits, cash/stock dividends and spin-offs. The experiment currently applies forward/reverse splits and cash-dividend features. Alpaca `process_date` is the point-in-time availability proxy; an event can only enter model features when `process_date <= decision session`.

### 3. Run the controlled experiment

```bash
python scripts/research_point_in_time_corporate_actions.py \
  --job-id 20260918T234903-52bd06f3
```

Outputs:

```text
output/point_in_time_corporate_actions/
  summary.json
  data_diagnostics.csv
  baseline_predictions.csv
  baseline_trades.csv
  experiment_predictions.csv
  experiment_trades.csv
```

The script runs two replays on the same locally reconstructed RAW->split price series:

1. `RAW_SPLIT_PRICE_ONLY`: existing MCT feature family.
2. `RAW_SPLIT_PIT_DIVIDEND_FEATURES`: same prices, same target, same LightGBM hyperparameters, plus dividend features known by `process_date`.

Added features:

```text
ca_ex_dividend_yield_today
ca_known_dividend_yield_next_5
ca_known_dividend_yield_next_20
ca_known_dividend_yield_next_60
ca_dividend_yield_trailing_60
```

The script also compares the locally reconstructed split series with the previously downloaded Alpaca `adjustment=split` snapshot when that collection is available. This validates the split reconstruction independently of the model result.

This version intentionally does not add dividend cash to simulated portfolio equity and does not change the forward target. Those are separate hypotheses and should only be tested after this input-context experiment is evaluated.


## RAW + split Unified CARO recalibration (API v10.8.63)

This campaign recalibrates LightGBM hyperparameters after the market-data representation changed from provider-adjusted history to immutable RAW bars with local split normalization.

A full corporate-action snapshot is required because structural identity events (for example mergers and ticker lineage changes) must be detected before tuning. The DOC/PEAK 2024 merger is the first observed case motivating this guard.

### 1. Refresh the full corporate-action snapshot

```bash
python scripts/download_alpaca_corporate_actions_snapshot.py \
  --job-id 20260918T234903-52bd06f3
```

Default destination:

```text
alpaca_corporate_actions_full_20260919
```

### 2. Run Unified CARO on RAW + local split normalization

```bash
python scripts/research_raw_split_unified_caro.py \
  --job-id 20260918T234903-52bd06f3 \
  --candidate-count 20
```

The campaign uses the existing Unified CARO implementation:
- initial space-filling exploration;
- Gaussian-process probabilistic refinement;
- trust-region adaptation;
- stagnation recovery;
- champion gate against the current RAW+split Control.

This campaign deliberately excludes dividend features. It recalibrates only the canonical price architecture established by v10.8.62.

Outputs:

```text
output/raw_split_unified_caro/
  summary.json
  campaign_checkpoint.json
  candidates.csv
  data_diagnostics.csv
  excluded_assets.csv
  control_predictions.csv
  control_trades.csv
  champion_predictions.csv
  champion_trades.csv
```

Structural lineage guard:
- forward/reverse splits are normalized locally;
- a symbol acting as the acquiree in a merger into a different ticker is excluded from this tuning campaign until point-in-time lineage reconstruction is implemented;
- this is deterministic corporate-action handling, not a performance heuristic.


## Unified CARO control-anchor correction (API v10.8.64)

The v10.8.63 RAW+split campaign revealed that the control execution was used as the champion threshold but was not included in the surrogate training observations. As a result, the Gaussian-process model only learned from weak candidates while trying to beat a much stronger external threshold.

API v10.8.64 corrects this by adding the RAW+split Control as:

- a completed `is_control=true` prior observation;
- candidate id `0`;
- the initial probability anchor;
- an explicit normalized point in space-filling distance calculations;
- a training observation for Gaussian-process outcome modeling.

The control does not count toward the required exploration-trial floor.

Run:

```bash
python scripts/research_raw_split_unified_caro.py \
  --job-id 20260918T234903-52bd06f3 \
  --candidate-count 20
```

Expected checkpoint fields:

```json
{
  "api_version": "10.8.64",
  "control_observation_in_surrogate": true,
  "probability_anchor": {
    "source": "control",
    "candidate_id": 0
  }
}
```

The v10.8.63 campaign should not be interpreted as evidence that the old hyperparameters are globally optimal because its adaptive surrogate did not include the control outcome.


## LightGBM GPU accelerator support (API v10.8.65)

LightGBM now consumes the existing rotation accelerator configuration instead of always forcing CPU.

Configuration precedence:

```text
persisted Backtest/MongoDB rotation_accelerator
        ↓ if absent
MCT_ROTATION_ACCELERATOR from .env
        ↓ if absent
auto
```

The same fallback rule applies to `rotation_allow_cpu_fallback` using `MCT_ROTATION_ALLOW_CPU_FALLBACK`.

Supported values remain:

```text
auto
cpu
cuda
```

LightGBM backend mapping:
- Windows: GPU acceleration uses `device_type=gpu` (OpenCL).
- Linux: `cuda` is tried first, then `gpu`.
- `auto` probes GPU support and falls back to CPU.
- explicit `cuda` follows `rotation_allow_cpu_fallback` before falling back to CPU.

The campaign output records:
- `requested_compute_device`
- `effective_compute_device`
- `compute_device_probe_errors`

This allows every CARO result to prove whether training actually used GPU or CPU.

For a local fallback when the persisted request does not contain accelerator fields:

```env
MCT_ROTATION_ACCELERATOR=cuda
MCT_ROTATION_ALLOW_CPU_FALLBACK=false
```

If the persisted MongoDB/job request already contains `rotation_accelerator`, that value remains authoritative.


## LightGBM GPU cache-signature compatibility (API v10.8.66)

API v10.8.65 added LightGBM GPU selection, but the accelerated research path can replace `_lightgbm_fit_models` at runtime with a cached wrapper that still exposes the older call signature.

The v10.8.65 caller passed `device_type=...` explicitly, causing:

```text
TypeError: _cached_lightgbm_fit_models() got an unexpected keyword argument 'device_type'
```

API v10.8.66 keeps the cached wrapper contract unchanged. GPU selection is resolved inside the underlying LightGBM fit function from the existing request / environment configuration, so callers no longer pass a new keyword argument through the cache boundary.

The precedence remains unchanged:

```text
MongoDB/job rotation_accelerator
  -> .env MCT_ROTATION_ACCELERATOR if absent
  -> auto if both are absent
```

GPU diagnostics remain persisted in campaign results.


## Temporal early stopping + expanded LightGBM CARO (API v10.8.67)

This version incorporates the overfitting/generalization controls studied in the MBA material on tree ensembles and boosting while preserving the point-in-time RAW + local split architecture.

### LightGBM temporal early stopping

Early stopping is enabled as a fixed methodology safeguard, not a CARO search dimension.

Default protocol:

```text
chronological training sample
    |
    +-- first ~85% -> LightGBM fit
    |
    +-- final ~15% -> internal temporal validation
                          |
                          +-- stop after 30 non-improving boosting rounds
```

Guardrails:
- validation is always the chronological tail of the training sample;
- OOS is never used for early stopping;
- validation uses at least 40 and at most 126 sessions when enough rows exist;
- at least 80% of the configured minimum training rows are retained for fitting;
- `n_estimators` remains the maximum tree budget while `best_iteration` records the effective tree count.

### Expanded Unified CARO search space

The previous eight dimensions remain at their prior bounds. Three variance-control dimensions were added:

```text
min_child_weight  0.0001 .. 0.0500
subsample         0.65   .. 1.00
subsample_freq    1      .. 5
```

The full search space is now:

```text
n_estimators
learning_rate
max_depth
num_leaves
min_child_samples
min_child_weight
subsample
subsample_freq
colsample_bytree
reg_alpha
reg_lambda
```

The default campaign size is 24 candidates to reflect the larger search dimension.

### Predictive diagnostics

Each final fold now records:
- training MAE;
- training RMSE;
- temporal-validation MAE;
- temporal-validation RMSE;
- RMSE generalization gap;
- configured estimator count;
- mean/median effective `best_iteration`;
- fraction of models using early stopping;
- LightGBM gain-based feature importance.

These diagnostics are informative only. They do not replace the existing economic Champion gate.

### RAW + split research outputs

```text
output/raw_split_unified_caro/
  summary.json
  campaign_checkpoint.json
  candidates.csv
  data_diagnostics.csv
  excluded_assets.csv
  control_predictions.csv
  control_trades.csv
  champion_predictions.csv
  champion_trades.csv
  control_model_diagnostics.json
  champion_model_diagnostics.json
  feature_importance_gain.csv
```

Run:

```bash
python scripts/research_raw_split_unified_caro.py \
  --job-id 20260918T234903-52bd06f3 \
  --candidate-count 24
```

GPU behavior from API v10.8.66 is preserved. The result continues to report `requested_compute_device`, `effective_compute_device`, and GPU probe errors.

The course material motivates early stopping, MAE/RMSE regression diagnostics, regularization/tuning sensitivity, and feature importance. Random Forest and stacking remain separate future research hypotheses and are intentionally not mixed into this calibration campaign.


## Diagnostic-only validation + log-scaled child weight (API v10.8.68)

The v10.8.67 campaign showed that per-asset RMSE early stopping was not aligned with the portfolio's economic cross-asset ranking objective. It also exposed a search-space defect: the existing Control used `min_child_weight=5.0`, while the v10.8.67 CARO domain stopped at `0.05`.

API v10.8.68 corrects both issues.

### Full LightGBM fit restored

LightGBM once again trains on the complete chronological training window using the configured `n_estimators`.

```text
chronological TRAIN
        |
        +-- full LightGBM fit
        |
        +-- existing CALIBRATION window
                 |
                 +-- MAE/RMSE diagnostics only
                 +-- switch-margin calibration
                 +-- never truncates tree construction
```

There is no RMSE-based early stopping in the economic ranking model. The OOS region remains untouched.

Predictive diagnostics remain available:
- training MAE/RMSE from the calibration-training models;
- validation MAE/RMSE on the already existing chronological calibration window;
- RMSE generalization gap;
- gain-based feature importance;
- configured/effective estimator count.

These metrics are informative only and do not enter the Champion gate.

### Corrected CARO domain

`min_child_weight` now uses a logarithmic domain:

```text
0.001 .. 10.0  (log scale)
```

This contains the existing Control value `5.0` and gives the Gaussian-process surrogate a meaningful distance around it instead of clipping the Control onto an artificial boundary.

`subsample_freq` now spans:

```text
0 .. 5
```

so a historical Control with bagging disabled (`0`) is represented exactly. The RAW+split campaign no longer forces `subsample_freq=1`.

The tuning-space mapper and Unified CARO normalization both understand `scale="log"`, so Latin-Hypercube / space-filling proposals and Gaussian-process observations use the same geometry.

### Run

```bash
python scripts/research_raw_split_unified_caro.py \
  --job-id 20260918T234903-52bd06f3 \
  --candidate-count 24
```

The expected Control should again reflect full-fit LightGBM behavior rather than the v10.8.67 RMSE-truncated trees. The exact capital must be observed from the frozen execution rather than hard-coded.


## Hybrid discontinuity-aware CARO surrogate (API v10.8.69)

The v10.8.68 campaign restored the RAW+split full-fit Control but showed that the Gaussian-process surrogate was poorly calibrated for the discrete rotation objective. Small hyperparameter changes can change asset ordering, which can change the entire portfolio path and produce a non-smooth economic response surface.

API v10.8.69 replaces the GP-only adaptive proposal with a hybrid surrogate:

```text
completed CARO observations
          |
          +-- Gaussian Process
          |      smooth response component
          |
          +-- Extra Trees
                 discontinuity-aware component
          |
          +-- cross-validated reliability
                 Spearman rank correlation
                 normalized RMSE
          |
          +-- reliability-weighted blend
          |
          +-- calibrated P(beat) + expected improvement
```

The Gaussian Process is retained because it is useful when the local response is smooth. Extra Trees is added because tree ensembles can represent abrupt changes and piecewise response surfaces without imposing the same smoothness assumption.

### Data-driven family weights

Each adaptive iteration performs deterministic K-fold out-of-sample surrogate diagnostics on the completed campaign observations.

For each economic metric:

- GP out-of-fold Spearman correlation;
- Extra Trees out-of-fold Spearman correlation;
- GP normalized RMSE;
- Extra Trees normalized RMSE.

These diagnostics produce separate GP / Extra Trees weights for:
- ending capital;
- Sharpe;
- maximum drawdown;
- worst-fold return.

The blend is selected from out-of-fold predictions, which makes the surrogate ensemble a small stacking problem inside CARO only. It does not change the trading model.

Because the observed economic response is discontinuous, Extra Trees has an 80% minimum central-estimate weight unless the GP demonstrates material out-of-fold superiority: positive rank skill, normalized RMSE at or below 1.0, and at least 25% more rank/error skill than Extra Trees. This prevents small-sample cross-validation noise from returning control to a GP-dominated smoothness assumption.

### Small-sample confidence correction

Cross-validation can appear overconfident when the number of completed observations is small relative to the 11-dimensional tuning space. The hybrid reliability is therefore multiplied by:

```text
observation_support =
    observations / (observations + 2 * search_dimensions)
```

This deliberately keeps the first adaptive proposals conservative.

### Empirical Champion-pass prior

The model-based Champion probability is not used directly. Completed non-control candidates with an evaluated Champion gate define a Beta(1,1)-smoothed empirical prior:

```text
P_empirical =
    (champion_passes + 1)
    / (evaluated_candidates + 2)
```

The displayed / acquisition `P(beat)` becomes:

```text
P_adjusted =
    reliability * P_model
    + (1 - reliability) * P_empirical
```

When the surrogate has weak evidence, confidence is pulled toward the actual campaign success rate instead of producing a misleading large probability.

Expected improvement is also damped under weak surrogate reliability, while the exploration term is increased. Therefore low-confidence periods trigger more discovery rather than aggressive exploitation.

### Persisted diagnostics

Adaptive candidate proposals now persist:

- raw model P(beat);
- reliability-adjusted P(beat);
- empirical Champion-pass prior;
- raw and effective surrogate reliability;
- observation-support factor;
- GP / Extra Trees cross-validation weights;
- GP / Extra Trees Spearman correlations;
- GP / Extra Trees normalized RMSE;
- hybrid, GP-only and Extra-Trees-only capital estimates;
- base and effective exploration weights.

The RAW+split research `candidates.csv` also exports the principal hybrid-surrogate fields for direct analysis.

### Retrospective check on the v10.8.68 campaign

Using the already observed v10.8.68 candidate settings/outcomes as an offline diagnostic, the previous GP-only capital estimates had approximately 0.12 Spearman rank correlation across the nine adaptive candidates, while Extra Trees reached approximately 0.67. The discontinuity-guarded hybrid was therefore designed to be tree-dominant unless GP earns more influence out-of-fold. This retrospective check is diagnostic only; it is not reused as candidate evidence in the new campaign.

The actual v10.8.69 candidate search must still run prospectively on the frozen snapshot.

Run:

```bash
python scripts/research_raw_split_unified_caro.py \
  --job-id 20260918T234903-52bd06f3 \
  --candidate-count 24
```


## Optuna TPE library baseline (API v10.8.70)

API v10.8.70 adds a deliberately simpler optimizer baseline for comparison with the custom Unified CARO implementation.

The experiment keeps the MCT research methodology fixed:

```text
same frozen RAW snapshot
same local split reconstruction
same structural identity exclusions
same LightGBM full-fit behavior
same 3 research folds
same Control
same Champion gate
same 11-parameter search domain
same seed
same 24-candidate budget
```

Only the hyperparameter proposal engine changes.

### Optimizer

The baseline uses the pinned Optuna 5.0.0 `TPESampler` for the reproducible research run:

```text
seed = fixed
direction = maximize ending_capital
multivariate = true
group = true
constant_liar = false
execution = sequential ask/tell
```

Sequential ask/tell is intentional. Candidate training can still use the configured LightGBM GPU, but optimizer proposals are generated one at a time so that the seeded experiment remains reproducible.

The Control is inserted into the Optuna study as completed trial 0 with its actual ending capital. Therefore TPE starts with the same strong reference that Unified CARO receives.

### Dynamic structural search domain

The Optuna adapter preserves the LightGBM structural rule:

```text
num_leaves <= 2 ** max_depth
```

This is represented as a dynamic Optuna search space rather than by sampling an invalid point and modifying it afterward.

`TPESampler(multivariate=True, group=True)` is used specifically so Optuna can model this decomposed dynamic space.

`min_child_weight` keeps the corrected v10.8.68 domain:

```text
0.001 .. 10.0
log scale
```

### Objective and constraints

Optuna receives one optimization objective:

```text
maximize ending_capital
```

Robustness is represented through fixed Control-relative feasibility constraints:

```text
Sharpe >= Control Sharpe - 0.05
MaxDD  >= Control MaxDD  - 0.03
Worst Fold Return > 0
```

These constraints guide TPE search only. The worst-fold constraint uses a tiny positive epsilon so an exact zero remains infeasible, matching the strict MCT gate. They do not replace MCT promotion governance.

The existing dynamic MCT Champion gate remains authoritative. When a candidate passes the gate, it becomes the new MCT anchor and subsequent promotions must beat that Champion.

This separation avoids implementing a custom scalar penalty function merely to make Optuna understand risk.

### Startup phase

By default the TPE startup count is:

```text
max(10, search_dimensions + 1)
```

For the current 11-dimensional LightGBM search this is 12 completed study trials. Because the Control is preloaded as trial 0, the fresh campaign normally evaluates 11 startup candidates before TPE begins adaptive proposals.

### Dependency

```text
optuna==5.0.0
```

The v10.8.70 research environment pins Optuna 5.0.0 so the sampler implementation is part of the frozen experiment. The adapter still detects the constraint API at runtime: Optuna 5.0.0 uses `Trial.set_constraint()`, while a 4.8 development environment can fall back to the legacy `constraints_func` path. Cross-version sampling sequences are not assumed to be identical.

### Run after v10.8.69 completes

```bash
python scripts/research_raw_split_optuna_tpe.py \
  --job-id 20260918T234903-52bd06f3 \
  --candidate-count 24
```

Outputs are isolated from CARO:

```text
output/raw_split_optuna_tpe/
  summary.json
  campaign_checkpoint.json
  candidates.csv
  data_diagnostics.csv
  excluded_assets.csv
  control_predictions.csv
  control_trades.csv
  champion_predictions.csv
  champion_trades.csv
  control_model_diagnostics.json
  champion_model_diagnostics.json
  feature_importance_gain.csv
```

The comparison question is intentionally narrow:

> With the same frozen MCT experiment and the same candidate budget, can a specialized optimization library find a Control-beating or similarly strong region with less custom optimizer logic than Unified CARO?


## Control-centered Optuna + OOS profiling + buy-and-hold reporting (API v10.8.71)

API v10.8.71 supersedes the interrupted v10.8.70 run. It addresses three issues observed during that campaign.

### 1. Control-centered Optuna warm start

The certified Control remains trial 0. Instead of spending the initial budget on globally scattered startup trials, the next six trials are deterministic Latin-Hypercube perturbations around the Control in normalized parameter space.

Defaults:

```text
warm_start_count = 6
warm_start_radius = 0.06
seed = 42
```

The radius applies after each parameter is mapped to [0,1], including logarithmic dimensions such as `min_child_weight`. Structural LightGBM constraints such as `num_leaves <= 2^max_depth` are preserved by the existing tuning-space mapper.

After the Control plus the six local warm-start trials are complete, TPE becomes adaptive. This keeps the optimizer anchored around the already strong incumbent while still allowing later exploration.

The goal is not to forbid global search; it is to avoid spending most of a 24-candidate budget proving again that distant regions are poor.

### 2. Granular OOS simulation progress and profiling

The old progress gap around 87.7% was caused by a silent full out-of-sample portfolio replay after final training.

The simulators now emit granular progress through:
- buy-and-hold benchmark construction;
- market-regime diagnostics;
- chronological OOS replay;
- completion.

They also persist:

```text
simulation_benchmark_seconds
simulation_market_regime_seconds
simulation_policy_seconds
simulation_accounting_seconds
simulation_total_seconds
simulation_session_count
```

This allows later performance optimization to target the measured hotspot rather than changing deterministic calculations speculatively.

### 3. Buy-and-hold remains a first-class benchmark

The benchmark already implemented by MCT is a true equal-weight buy-and-hold across assets with complete prices for the OOS execution window:

```text
initial capital
    -> one purchase at first execution open
    -> fixed quantities, no periodic rebalance
    -> final liquidation at the last close
```

The same fee and slippage functions are applied to the initial purchases and final liquidation.

The research outputs now surface, for Control, every candidate, and Champion:

- buy-and-hold ending capital;
- buy-and-hold total return;
- buy-and-hold CAGR;
- buy-and-hold Sharpe;
- buy-and-hold maximum drawdown;
- strategy / buy-and-hold capital ratio;
- excess capital;
- excess return;
- CAGR spread;
- Sharpe spread;
- drawdown spread.

Per-fold research already records `benchmark_return` and `excess_return`; v10.8.71 keeps these and adds the full-study comparison explicitly.

A dedicated artifact is written:

```text
output/raw_split_optuna_tpe_control_warm_start/buy_hold_comparison.csv
```

### Run

Stop the older campaign before switching branches. Then run:

```bash
python scripts/research_raw_split_optuna_tpe.py \
  --job-id 20260918T234903-52bd06f3 \
  --candidate-count 24 \
  --warm-start-count 6 \
  --warm-start-radius 0.06
```

The previous v10.8.70 output directory is intentionally not reused.


## OOS technical logger callback fix (API v10.8.72)

API v10.8.71 added OOS simulation profiling. The post-simulation timing log accidentally called a local helper named `technical()` that exists in the LightGBM fit helper but not in `_run_lightgbm()`.

The failure occurred only after the CONTROL completed its OOS replay:

```text
NameError: name 'technical' is not defined
```

API v10.8.72 calls the existing optional `technical_log_callback` directly under a null guard. No model, market-data, optimizer, benchmark, or simulation calculation changes.

The v10.8.71 warm-start, buy-and-hold reporting, OOS granular progress, and profiling behavior are preserved unchanged.


## Weighted multi-horizon voting consensus (API v10.8.73)

API v10.8.73 adds a focused A/B experiment for an ensemble-voting hypothesis without changing the certified Control.

The existing MCT model remains the Control:

```text
5d/10d/20d/40d/60d
        |
weighted target
        |
LightGBM
        |
base asset/CASH policy
```

The challenger trains an additional LightGBM target for each configured horizon using the same features and the same LightGBM hyperparameters.

Each horizon predicts its own forward risk-adjusted utility. Its vote is:

```text
argmax(CASH=0, utility(asset 1), ..., utility(asset N))
```

Therefore a horizon votes CASH when every finite asset utility for that horizon is non-positive.

The configured target-horizon weights are reused as vote weights. With the current profile:

```text
5d  = 0.10
10d = 0.15
20d = 0.20
40d = 0.30
60d = 0.25
```

### Conservative v1 decision rule

Voting v1 is a consensus guard around the existing policy, not a replacement policy.

A weighted consensus can:
- confirm the base model's selected asset;
- veto a proposed rotation and keep the current position;
- override an asset proposal to CASH when CASH has sufficient weighted consensus.

It cannot jump directly to another asset that the base policy did not select. This keeps the A/B experiment focused on whether independent horizon agreement reduces unstable switches.

The default minimum consensus is:

```text
0.50
```

The existing base CASH decision is always preserved; voting is never allowed to block a protective CASH action.

### Horizon-specific labels

The canonical frame now persists:

```text
forward_horizon_utility_5
forward_horizon_utility_10
forward_horizon_utility_20
forward_horizon_utility_40
forward_horizon_utility_60
```

and the corresponding horizon net-log-return targets.

These labels contain only information from their own forward horizon. The existing weighted production target remains unchanged.

### Experimental isolation

The first experiment does not tune voting parameters and does not run Optuna.

It runs only:

```text
A: CONTROL
B: CONTROL + HORIZON_VOTING
```

with identical:
- RAW + local split data;
- eligible assets;
- folds and purge;
- LightGBM hyperparameters;
- transaction costs and slippage;
- buy-and-hold benchmark;
- initial capital.

This intentionally costs more model training for the challenger because five independent horizon model sets are fitted. The experiment is limited to the canonical single-position rotation policy; optimized-allocation and compound-risk-overlay modes are rejected in v1.

### Outputs

```text
output/raw_split_horizon_voting_consensus/
  summary.json
  strategy_comparison.csv
  horizon_voting_decisions.csv
  control_predictions.csv
  control_trades.csv
  horizon_voting_predictions.csv
  horizon_voting_trades.csv
  data_diagnostics.csv
  excluded_assets.csv
```

Decision diagnostics include:
- winner asset and weighted consensus;
- CASH vote weight;
- base-target vote weight;
- per-horizon winner, score and weight;
- whether voting changed the base action;
- accept / CASH override / blocked-switch reason.

### Run

```bash
python scripts/research_raw_split_horizon_voting.py \
  --job-id 20260918T234903-52bd06f3 \
  --minimum-consensus-weight 0.50
```

The first question is deliberately narrow:

> Does independent agreement across forecast horizons improve OOS capital and/or robustness relative to the exact current Control?


## Soft multi-horizon rank consensus + batched OOS inference (API v10.8.74)

API v10.8.74 follows the negative v10.8.73 hard-voting result.

The hard-voting experiment showed that exact ticker agreement across horizons is too restrictive for a 55-asset universe. It changed approximately 36.6% of base actions, reduced trading activity by roughly 69%, and materially reduced ending capital. It also showed that CASH never won the horizon vote, because choosing the maximum predicted utility across many assets makes a simple CASH=0 comparison unsuitable.

v10.8.74 therefore changes both the statistical representation of consensus and the OOS inference implementation.

### Soft rank consensus

Each horizon still has an independent LightGBM model trained on:

```text
forward_horizon_utility_5
forward_horizon_utility_10
forward_horizon_utility_20
forward_horizon_utility_40
forward_horizon_utility_60
```

But horizons no longer emit a one-hot ticker vote.

For every decision date and horizon:

1. all finite assets are ranked by predicted horizon utility;
2. rank is converted to a percentile-like score in [0,1];
3. top asset receives 1.0 and the last ranked asset receives 0.0;
4. scores are aggregated with the existing horizon weights.

With the canonical profile:

```text
5d  = 0.10
10d = 0.15
20d = 0.20
40d = 0.30
60d = 0.25
```

This preserves information such as an asset being consistently second or third across horizons even when it rarely wins an exact ticker vote.

### CASH is not part of the consensus vote

The existing base MCT policy remains solely responsible for CASH.

If the base policy chooses CASH, the soft-consensus layer always preserves CASH.

The horizon rank layer cannot:
- force CASH;
- block a base CASH decision;
- choose a different asset.

This removes the multiple-comparison problem observed in v10.8.73.

### Continuous switch-margin modifier

The base weighted-utility model remains the only layer that selects the candidate asset.

For a proposed asset switch, the soft layer computes:

```text
base_target_rank_score
current_asset_rank_score
relative_rank_component
soft_support
```

The existing calibrated switch margin is then multiplied by:

```text
margin_multiplier =
    1 + penalty_strength * (1 - soft_support)
```

Default:

```text
penalty_strength = 1.0
```

Therefore:
- support = 1.0 -> existing margin unchanged;
- support = 0.75 -> margin becomes 1.25x;
- support = 0.50 -> margin becomes 1.50x;
- support = 0.00 -> margin becomes 2.00x.

There is no exact majority threshold.

A strong base-model utility gap can still execute even when horizons disagree. Only marginal rotations are increasingly filtered as consensus weakens.

### Batched deterministic-equivalent OOS inference

Profiling in v10.8.73 showed that approximately 98-99% of the silent OOS replay time was policy/model inference.

The old replay performed approximately:

```text
decision date
    x asset
    x model set
    -> model.predict(one row)
```

v10.8.74 precomputes predictions by fold and model set:

```text
one model
    -> predict(all valid OOS rows for that fold)
    -> cache by timestamp
```

The policy then performs only cache lookups during replay.

The cache preserves the same validity rules as scalar inference:
- feature row must be complete;
- next session must exist;
- next open and close must be finite and positive.

No future information is added. The cache contains predictions only from the fold-specific model already trained before that fold.

The implementation persists:
- cache build seconds;
- cached session count;
- symbol count;
- number of LightGBM predict calls;
- predicted row count;
- model role and fold.

Tests compare cached and scalar utility vectors at zero relative tolerance with a tiny floating absolute tolerance.

### Focused A/B

The experiment remains intentionally small:

```text
A: CONTROL
B: CONTROL + SOFT_HORIZON_CONSENSUS
```

No Optuna tuning is used.

Both variants use:
- the same frozen RAW + locally split-normalized data;
- the same eligible universe;
- the same folds and purge;
- the same LightGBM settings;
- the same transaction costs;
- the same buy-and-hold benchmark.

The Control also uses batched OOS inference, so the economic comparison is not confounded by different replay implementations.

### Outputs

```text
output/raw_split_soft_horizon_consensus/
  summary.json
  strategy_comparison.csv
  soft_horizon_consensus_decisions.csv
  control_predictions.csv
  control_trades.csv
  soft_horizon_consensus_predictions.csv
  soft_horizon_consensus_trades.csv
  data_diagnostics.csv
  excluded_assets.csv
```

### Run

```bash
python scripts/research_raw_split_soft_horizon_consensus.py \
  --job-id 20260918T234903-52bd06f3 \
  --penalty-strength 1.0
```

Primary questions:

1. Does the exact Control remain economically identical under batched inference?
2. How much does OOS replay time fall relative to v10.8.73?
3. Does soft rank support filter only marginal rotations rather than suppressing rotation broadly?
4. Does the challenger improve capital and/or robustness while retaining the strategy's rotation edge?


## Final standalone fresh-Alpaca research (API v10.8.75)

API v10.8.75 creates the final reproducible research path without using MongoDB as a source of market data, corporate actions, job configuration, or experiment state.

The frozen experiment configuration is committed to:

```text
research/final_research_v10_8_75.json
```

It contains the complete 56-asset universe, study window, transaction-cost assumptions, walk-forward configuration, LightGBM hyperparameters, and the already-selected soft-consensus setting:

```text
penalty_strength = 1.0
```

The final runner is:

```text
scripts/research_final_fresh_alpaca_standalone.py
```

### Data acquisition

By default every execution downloads again:

1. RAW daily SIP stock bars directly from Alpaca;
2. corporate actions directly from Alpaca's corporate-actions endpoint.

Credentials are read exclusively from environment variables:

```text
ALPACA_API_KEY_ID
ALPACA_SECRET_KEY
```

No MongoDB client is created.

The requested study window remains frozen:

```text
2016-01-01 through 2026-09-17
```

This means a new execution tests whether the same frozen historical experiment can be reconstructed from a fresh Alpaca retrieval without relying on the project's historical database snapshot.

### Local immutable research snapshot

Fresh inputs are written under:

```text
output/final_fresh_alpaca_standalone/input/
  raw/
  split/
  corporate_actions.json
  snapshot_manifest.json
```

RAW data is never overwritten by local adjustments. Split-normalized series are written separately.

The manifest stores:
- frozen configuration SHA-256;
- canonical per-symbol RAW data SHA-256;
- file SHA-256;
- corporate-action artifact hash;
- combined snapshot SHA-256;
- coverage and row counts.

Canonical data hashes are calculated from deterministic uncompressed CSV serialization rather than gzip bytes, so gzip metadata cannot change the scientific data identity.

### Corporate-action policy

The same point-in-time architecture is preserved:

```text
fresh Alpaca RAW bars
        +
fresh Alpaca corporate-action ledger
        ↓
local split/reverse-split reconstruction
        ↓
structural identity guard
        ↓
MCT features / LightGBM / walk-forward
```

Mergers that make a ticker structurally ambiguous remain excluded. In the established dataset this affects DOC.

Dividend back-adjustment is not applied.

### Models evaluated

The final run is intentionally not a new tuning campaign.

It evaluates the already-defined hypotheses:

```text
A: CONTROL
B: CONTROL + SOFT_HORIZON_CONSENSUS
```

The LightGBM parameters and soft-consensus penalty are frozen before the fresh download.

This separation is important for the TCC: the final fresh-data run is a reproducibility/confirmation experiment, not another search for parameters that maximize the same sample.

### Buy-and-hold

The equal-weight buy-and-hold benchmark remains included with the same initial capital, transaction-cost functions, and OOS window.

The final summary reports:
- ending capital;
- total return;
- CAGR;
- Sharpe;
- maximum drawdown;
- worst fold;
- strategy / buy-and-hold ratio.

### No-Mongo guarantee

A dedicated test verifies that the final runner does not import or call:

```text
create_client
get_database
mongo_repository
JOBS_COLLECTION
_latest_job
```

Run:

```bash
pytest -q tests/test_final_fresh_alpaca_standalone.py
```

### Final execution

```bash
python scripts/research_final_fresh_alpaca_standalone.py
```

The normal mode always downloads fresh data from Alpaca.

Only for exact replay/debugging of the just-downloaded local artifact:

```bash
python scripts/research_final_fresh_alpaca_standalone.py \
  --reuse-local-snapshot
```

That replay still does not access MongoDB.
