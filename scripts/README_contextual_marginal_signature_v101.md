# Contextual Marginal Signature v1.0.1

Branch: `research/contextual-marginal-signature-v1`

Research version: `contextual-marginal-signature-v1.0.1`

API/package remains **10.8.48**. Frontend is unchanged.

## Fix

The first local execution exposed a timezone compatibility defect in the calendar horizon resolver. `_expected_sessions(...)` can return a tz-naive `DatetimeIndex`, while the selected decision sessions are normalized to UTC-aware timestamps. Pandas 3 rejects `searchsorted()` between those two representations with:

`TypeError: Cannot compare tz-naive and tz-aware datetime-like objects.`

v1.0.1 normalizes the calendar index to UTC before searching for the decision date and future horizon endpoint.

This is an implementation correction only. It does **not** change:

- the 12 candidates;
- the Original25 base universe;
- the 20 deterministic decision dates;
- the 40-session future horizon;
- the exact Strategy replay judge;
- the feature representation;
- the chronological validation boundary;
- the economic interpretation of the experiment.

A dedicated regression test now covers both tz-naive and tz-aware calendars.

## Git Bash

```bash
git fetch origin
git switch research/contextual-marginal-signature-v1
git pull --ff-only origin research/contextual-marginal-signature-v1
git log -1 --oneline
```

## Windows Command Prompt

Run the v1.0.1 timezone regression tests:

```cmd
python -m unittest discover -s tests -p "test_contextual_marginal_signature_v101.py" -v
```

You can also rerun the original research tests:

```cmd
python -m unittest discover -s tests -p "test_contextual_marginal_signature_v1.py" -v
```

Then run the corrected microexperiment through the v1.0.1 wrapper:

```cmd
python scripts\research_contextual_marginal_signature_v101.py --strategy-sequence 10 --history-start 2016-01-01 --snapshot-end 2026-09-04 --seed-assets NVDA AAPL MSFT AMZN GOOGL META TSLA AMD JPM SPY AVGO NFLX CRM ORCL COST LLY XOM CAT WMT V HD ADC ADEA ADI ADM --candidate-symbols GKOS DNN CORT MKSI VNCE UNFI APD YANG CCK CXW CLMT XSD --decision-start 2019-01-01 --decision-end 2025-12-01 --decision-count 20 --horizon-sessions 40 --validation-start 2024-01-01 --market-proxy SPY --workers 4 --output-dir research_output\contextual_marginal_signature_micro_v101 --fresh-run
```

The wrapper installs only the corrected horizon resolver and then executes the original v1 research pipeline.
