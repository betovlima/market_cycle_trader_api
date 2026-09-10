# Exact Marginal Capital Search v1.0.2

Research branch: `research/exact-marginal-capital-search-v1`.

This patch fixes the candidate-history fallback discovered by the v1.0.1 smoke run. The benchmark baseline and exact economic judge are unchanged.

## Failure reproduced

The v1.0.1 smoke run completed the 56-asset baseline, but all explicit external controls (`LB`, `VSAT`, `RARE`, `CCS`) failed before any candidate replay with:

```text
'NoneType' object is not subscriptable
```

The failure occurred in the delegated transient-history path before any meaningful candidate evaluation.

## v1.0.2 change

The benchmark no longer delegates the missing-candidate download to the opaque helper path. It now performs the legacy Asset Discovery behavior explicitly:

1. prefer complete local MongoDB candidate history;
2. when absent, read and validate Alpaca credentials explicitly;
3. download the full candidate window in the same bounded chunks used by the market-data engine;
4. clean and validate the frame against the baseline XNYS research calendar;
5. keep the candidate frame in memory only;
6. run the unchanged exact Strategy judge.

No candidate history is persisted. The preselector still never filters the exact economic evaluation.

The patch also passes the active database context into the worker thread using thread-local state, so credential/configuration helpers have the same context as the exact-search worker even when candidate evaluations are parallelized later.

## Version scope

Research script: `exact-marginal-capital-search-v1.0.2`.

API package remains `10.8.47`: this patch changes only the standalone research harness and tests, not the production API runtime.

## Validation

Run:

```bash
python -m unittest discover \
  -s tests \
  -p "test_exact_marginal_capital_search*.py" \
  -v
```

Then repeat the control smoke test from a clean output directory:

```bash
python scripts/research_exact_marginal_capital_search_v102.py \
  --strategy-sequence 10 \
  --history-start 2016-01-01 \
  --snapshot-end 2026-09-04 \
  --candidate-symbols LB VSAT RARE CCS \
  --workers 1 \
  --output-dir research_output/exact_marginal_capital_search_smoke \
  --fresh-run
```

Do not interpret the benchmark until at least one control reaches the exact replay. Do not merge or tag production from this research branch.
