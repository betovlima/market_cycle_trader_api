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
