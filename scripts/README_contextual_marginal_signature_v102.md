# Contextual Marginal Signature v1.0.2

Branch: `research/contextual-marginal-signature-v1`

Research version: `contextual-marginal-signature-v1.0.2`

API/package remains **10.8.48**. Frontend is unchanged.

## Important processing rule

This version is a **standalone research processor**. It does not import or execute previous research processor files such as `research_contextual_marginal_signature_v1.py`, `research_contextual_marginal_signature_v101.py`, `research_asset_signature_leave_one_out.py`, or `research_sequential_exact_marginal_search*.py`.

Shared production engine/service code is still used, because that is the Strategy implementation being measured. The exact replay prediction cache used by the experiment is embedded directly in this versioned processor.

The timezone correction is also embedded directly: every calendar used for horizon lookup is normalized to a UTC-aware `DatetimeIndex` before `searchsorted`.

## Why the previous command failed

The failing command was pasted into Git Bash and used Windows backslashes:

`python scripts\research_contextual_marginal_signature_v101.py ...`

In Bash, the backslash is an escape character, so the shell turned the path into `scriptsresearch_contextual_marginal_signature_v101.py`.

Python research commands should be run from **Windows Command Prompt**. To make copy/paste safer even if a command accidentally lands in Git Bash, the commands below use forward slashes, which Python on Windows accepts.

## Git Bash

```bash
git fetch origin
git switch research/contextual-marginal-signature-v1
git pull --ff-only origin research/contextual-marginal-signature-v1
git log -1 --oneline
```

## Windows Command Prompt

Run the v1.0.2 tests:

```cmd
python -m unittest discover -s tests -p "test_contextual_marginal_signature_v102.py" -v
```

Then run the standalone experiment:

```cmd
python scripts/research_contextual_marginal_signature_v102.py --strategy-sequence 10 --history-start 2016-01-01 --snapshot-end 2026-09-04 --seed-assets NVDA AAPL MSFT AMZN GOOGL META TSLA AMD JPM SPY AVGO NFLX CRM ORCL COST LLY XOM CAT WMT V HD ADC ADEA ADI ADM --candidate-symbols GKOS DNN CORT MKSI VNCE UNFI APD YANG CCK CXW CLMT XSD --decision-start 2019-01-01 --decision-end 2025-12-01 --decision-count 20 --horizon-sessions 40 --validation-start 2024-01-01 --market-proxy SPY --workers 4 --output-dir research_output/contextual_marginal_signature_micro_v102 --fresh-run
```

## Scientific definition

For each decision date `t` and candidate `a`, the label remains:

`Delta(a | U, t, H) = log C(U + a, t:t+H) - log C(U, t:t+H)`

The feature set remains restricted to backward-looking Strategy inputs and contextual/relative quantities known at `t`. The chronological validation boundary remains 2024-01-01.

This patch changes processing isolation and timezone handling only. It does not intentionally change the economic hypothesis or selection target.
