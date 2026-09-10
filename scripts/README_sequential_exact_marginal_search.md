# Sequential Exact Marginal Search v1.0.0

Branch: `research/sequential-exact-marginal-search-v1`.
API/package: **10.8.48**.
Base: Exact Marginal Capital Search v1.0.3 (`a3dab1d7cc8b793ae1789193ed815612224ca433`).

## Why this branch exists

The Exact Marginal Capital Search smoke against the final 56-asset universe produced no positive candidates among VSAT, RARE, HL and CCS. This does not contradict older Asset Discovery runs in which some of those symbols were positive against smaller universes. It reveals that marginal contribution is conditional on the current asset set.

The relevant objective is a set function:

`Delta(a | U) = Capital(U + a) / Capital(U) - 1`

There is no requirement that `Delta(a | U1)` has the same sign as `Delta(a | U2)`. Assets can substitute for each other's opportunities, alter rotation paths and make a previously useful candidate redundant or harmful.

This branch therefore stops treating an asset as universally good or bad. It reconstructs the greedy mechanism used by brute-force Discovery:

1. run the exact Strategy on the current universe;
2. run the exact Strategy once for every remaining `current universe + candidate`;
3. choose the candidate with the largest exact `ending_capital_delta_rate > 0`;
4. add only that candidate;
5. re-run the remaining candidates against the changed universe;
6. stop when no exact positive candidate remains or `--max-rounds` is reached.

There is no ML preselector in this experiment. Zero is the only economic boundary.

## Default seed

The default seed is the historical Original25 universe:

`NVDA,AAPL,MSFT,AMZN,GOOGL,META,TSLA,AMD,JPM,SPY,AVGO,NFLX,CRM,ORCL,COST,LLY,XOM,CAT,WMT,V,HD,ADC,ADEA,ADI,ADM`

If `--candidate-symbols` is omitted, the candidate pool is the selected Strategy's current asset set minus the seed. Explicit candidates can also be supplied for a small path-dependence probe.

## First probe

Do not start with the full 31-asset complement. First run a small controlled experiment:

```bash
git fetch origin
git switch research/sequential-exact-marginal-search-v1
git pull --ff-only origin research/sequential-exact-marginal-search-v1

python -m unittest discover -s tests -p "test_sequential_exact_marginal_search.py" -v

python scripts/research_sequential_exact_marginal_search.py \
  --strategy-sequence 10 \
  --history-start 2016-01-01 \
  --snapshot-end 2026-09-04 \
  --candidate-symbols HL CCS VSAT RARE \
  --max-rounds 2 \
  --workers 1 \
  --output-dir research_output/sequential_exact_marginal_search_probe \
  --fresh-run
```

Round 1 evaluates the four candidates against Original25. If one is positive, Round 2 adds the best one and re-evaluates the remaining three. A sign or rank change across rounds is direct evidence of state-dependent marginal contribution.

## Outputs

- `sequential_exact_manifest.json`: reproducibility contract.
- `sequential_round_evaluations.csv`: every candidate evaluation by round and current baseline hash.
- `sequential_path.csv`: candidate selected in each round and exact capital lift.
- `sequential_exact_summary.json`: final path, selected assets and stop reason.

Candidate bars are read from local MongoDB when available and otherwise downloaded transiently from Alpaca without persistence. Corporate-action identity checks are reused from Asset Discovery, but exact decision-context compatibility remains mandatory because the corporate-action endpoint is not sufficient to catch every ticker-reuse case.

## Interpretation

This is not an OOS/generalization claim. It answers a narrower question first: **did brute-force succeed because candidate value changes as the baseline universe changes?**

If the probe confirms that effect, the next benchmark can reconstruct the path through the final Strategy's 31 additions. Only after reconstructing that mechanism should we optimize runtime or place an outer walk-forward around selection.

Do not merge or tag this research branch for production.
