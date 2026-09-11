# Exact Bidirectional Universe Search v1.0.0

Branch: `research/exact-bidirectional-universe-search-v1`.

Research version: `exact-bidirectional-universe-search-v1.0.0`.

API/package remains **10.8.48** and Front is unchanged. This is a research-only continuation of the exact forward search and Exact Universe Pruning work.

## Why this branch exists

The forward exact search reached 42 assets and approximately **US$ 33.44M**, then Exact Universe Pruning removed GOOGL, AAPL, AMZN, CRM and MAN and reached approximately **US$ 43.76M** with 37 assets.

That result proved that asset usefulness is state-dependent and path-dependent: an asset can be useful when admitted and become redundant or harmful after later changes. It also proved that a one-way admission process is insufficient.

This branch therefore evaluates **both directions in the same round**.

For the current universe `U` and fixed known pool `P`:

- every asset `a` outside `U` is tested as `C(U + a)`;
- every asset `b` inside `U` is tested as `C(U - b)`;
- all moves use the same exact Strategy capital judge and the same round baseline;
- the single move with maximum exact positive capital delta is applied;
- all additions and removals are recalculated from scratch against the changed universe.

The search stops only after a **complete round** in which no single addition or removal has `DeltaCapital > 0`.

## Complete-round guard

A round cannot select a winner unless every expected move finishes with `evaluation_status=completed`.

If even one add/remove evaluation fails, is rejected, or is missing, the script:

1. saves the partial CSV/JSON outputs;
2. records `incomplete_round`;
3. selects no move;
4. exits with an error.

This prevents an operational failure from silently changing the economic search path.

## Objective

The decision objective remains **exact ending capital only**.

Sharpe, maximum drawdown, worst fold, switches and other metrics remain diagnostic. No new risk gate or predictive preselector is introduced in this research family.

The v1.1.x batched prediction cache remains an acceleration layer only. Model fitting, policy logic, fees, slippage, folds and capital accounting are unchanged.

## Starting point

Run from the validated 42-asset forward-selected checkpoint:

`NVDA AAPL MSFT AMZN GOOGL META TSLA AMD JPM SPY AVGO NFLX CRM ORCL COST LLY XOM CAT WMT V HD ADC ADEA ADI ADM GKOS VNCE CORT UNFI DNN MKSI APD DDS RACE UNF TX CEF YANG KKR BXMT SCSC MAN`

If `--pool-symbols` is omitted, the fixed candidate pool is the complete asset list stored in Strategy #10, currently the known 56-asset research pool.

## Git Bash

```bash
git fetch origin
git switch research/exact-bidirectional-universe-search-v1
git pull --ff-only origin research/exact-bidirectional-universe-search-v1
git log -1 --oneline
```

## Windows Command Prompt

Run the tests:

```cmd
python -m unittest discover -s tests -p "test_exact_bidirectional_universe_search_v1.py" -v
```

Run the research in Windows `cmd.exe`:

```cmd
python scripts\research_exact_bidirectional_universe_search_v1.py --strategy-sequence 10 --history-start 2016-01-01 --snapshot-end 2026-09-04 --seed-assets NVDA AAPL MSFT AMZN GOOGL META TSLA AMD JPM SPY AVGO NFLX CRM ORCL COST LLY XOM CAT WMT V HD ADC ADEA ADI ADM GKOS VNCE CORT UNFI DNN MKSI APD DDS RACE UNF TX CEF YANG KKR BXMT SCSC MAN --workers 4 --max-rounds 64 --output-dir research_output\exact_bidirectional_universe_search_forward42_v100 --fresh-run
```

## Outputs

- `bidirectional_manifest.json`
- `bidirectional_round_evaluations.csv`
- `bidirectional_path.csv`
- `bidirectional_summary.json`
- `accelerator_replay_index.csv`

Each evaluation row contains normalized fields:

- `move_type`: `add` or `remove`
- `move_symbol`
- `move_key`
- `move_delta_rate`
- `resulting_ending_capital`

The original addition/removal-specific fields are preserved as well.

## Valid stopping claim

A natural stop establishes only a local **1-opt** result inside the fixed known pool:

`max(Delta_add, Delta_remove) <= 0`

This means no single asset can improve ending capital by entering or leaving from that exact state.

It does **not** prove the global best subset. A two-asset swap or larger combination may still improve capital, and the historical 2016-2026 snapshot has been repeatedly used for research. This experiment therefore remains in-sample and must not be interpreted as a guarantee of future performance or merged/tagged as production strategy evidence.
