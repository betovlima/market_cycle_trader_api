# Alpaca × Tiingo Equivalence Audit

API v10.8.56 provides an isolated research script. It does not change the normal
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
- `alpaca_trades.csv` / `tiingo_trades.csv`: controlled trade sequences.

Material-difference flags are descriptive only: price differences above 1 bp
(0.01%) and volume differences above 1%. They do not alter the model or filter
the replay.

No parameter optimization is performed by this audit.
