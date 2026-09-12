# Contextual Marginal Signature v1.0.3

Branch: `research/contextual-marginal-signature-v1`

Research version: `contextual-marginal-signature-v1.0.3`

API/package remains **10.8.48**. Frontend is unchanged.

## What this fixes

The v1.0.2 failure `The requested analysis interval contains no executable session` was not an Alpaca outage. The Strategy walk-forward engine has a protected minimum training/calibration/purge history before its first executable out-of-sample session. The previous sampler could choose an early monthly date whose 40-session analysis window ended before any protected OOS policy existed.

v1.0.3 builds the seed rotation panel first and selects only decision dates for which the exact Strategy engine can create a valid walk-forward execution window. Early dates are skipped deterministically instead of failing at runtime.

## Data-source independence

The research processor is standalone and supports:

- `--data-source mongo`: frozen local MongoDB bars already cached by Market Cycle Trader. This mode makes no live Alpaca request, so an Alpaca maintenance window does not affect the experiment.
- `--data-source yahoo`: Yahoo Finance via `yfinance`, downloaded directly for this experiment with `auto_adjust=True`. Install once with `pip install yfinance`.

MongoDB is still used for the Strategy configuration/model snapshot even when Yahoo supplies OHLCV bars.

Do not compare absolute ending-capital levels across Mongo/Alpaca-derived bars and Yahoo bars as if they were the same market snapshot. Adjustments and vendor candles can differ. Within one run, however, baseline and candidates use the same source, so the marginal `DeltaCapital` experiment remains internally consistent.

## Recommended run while Alpaca is unavailable

Git Bash:

```bash
git fetch origin
git switch research/contextual-marginal-signature-v1
git pull --ff-only origin research/contextual-marginal-signature-v1
git log -1 --oneline
```

Windows CMD:

```cmd
pip install yfinance
python -m unittest discover -s tests -p "test_contextual_marginal_signature_v103.py" -v
python scripts/research_contextual_marginal_signature_v103.py --strategy-sequence 10 --history-start 2016-01-01 --snapshot-end 2026-09-04 --seed-assets NVDA AAPL MSFT AMZN GOOGL META TSLA AMD JPM SPY AVGO NFLX CRM ORCL COST LLY XOM CAT WMT V HD ADC ADEA ADI ADM --candidate-symbols GKOS DNN CORT MKSI VNCE UNFI APD YANG CCK CXW CLMT XSD --decision-start 2019-01-01 --decision-end 2025-12-01 --decision-count 20 --horizon-sessions 40 --validation-start 2024-01-01 --market-proxy SPY --data-source yahoo --workers 4 --output-dir research_output/contextual_marginal_signature_micro_v103_yahoo --fresh-run
```

If you prefer to keep the exact frozen local dataset already used by MCT, omit the Yahoo dependency and run the same command with `--data-source mongo`.

## Scientific rule

The source choice is not part of the hypothesis being tested. The experiment still asks whether point-in-time contextual features known at `t` contain stable information about the exact future marginal capital contribution of candidate `a` to universe `U`. The chronological holdout remains the judge; in-sample correlation alone is not evidence.
