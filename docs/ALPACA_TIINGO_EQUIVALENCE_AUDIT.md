# Alpaca × Tiingo Equivalence Audit

API v10.8.60 provides the isolated provider-audit workflow. It does not change the normal
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
