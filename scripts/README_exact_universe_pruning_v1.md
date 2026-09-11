# Exact Universe Pruning v1.0.0

Branch: `research/exact-universe-pruning-v1`.

Research version: `exact-universe-pruning-v1.0.0`.

API/package remains **10.8.48** and Front is unchanged. This is a research-only audit built on top of the exact replay accelerator already parity-validated in Sequential Exact Marginal Search v1.1.0.

## Why this branch exists

The forward greedy search reached a 42-asset universe with ending capital about **US$ 33.44M** and stopped naturally because none of the 14 remaining known candidates had positive exact marginal contribution.

That proves a local optimum under **single additions**, but not under **single removals**. An asset that was positive when it entered can become redundant or harmful after later additions change the policy path.

This branch tests the complementary question:

`Delta_remove(a | U) = C(U - {a}) / C(U) - 1`

For every asset currently in the universe it executes the exact Strategy without that asset. If any removal improves ending capital, the best positive removal is applied and every survivor is re-evaluated against the changed universe. The search stops only when no single removal has positive exact capital delta.

The economic judge, model fitting, policies, fees, slippage, fold logic and capital accounting are unchanged. The v1.1.0 batched prediction cache is used only as an exact accelerator.

## Run against the 42-asset forward-selected universe

```bash
git fetch origin
git switch research/exact-universe-pruning-v1
git pull --ff-only origin research/exact-universe-pruning-v1

git log -1 --oneline

python -m unittest discover \
  -s tests \
  -p "test_exact_universe_pruning_v1.py" \
  -v

python scripts/research_exact_universe_pruning_v1.py \
  --strategy-sequence 10 \
  --history-start 2016-01-01 \
  --snapshot-end 2026-09-04 \
  --seed-assets NVDA AAPL MSFT AMZN GOOGL META TSLA AMD JPM SPY AVGO NFLX CRM ORCL COST LLY XOM CAT WMT V HD ADC ADEA ADI ADM GKOS VNCE CORT UNFI DNN MKSI APD DDS RACE UNF TX CEF YANG KKR BXMT SCSC MAN \
  --workers 4 \
  --max-rounds 42 \
  --output-dir research_output/exact_universe_pruning_forward42_v100 \
  --fresh-run
```

## Outputs

- `exact_pruning_manifest.json`
- `pruning_round_evaluations.csv`
- `pruning_path.csv`
- `exact_pruning_summary.json`
- `accelerator_replay_index.csv`

A positive `removal_capital_delta_rate` means the universe performs better **without** that asset.

## Interpretation

If the first pruning round has no positive removal, the 42-asset result is a local optimum under both one-asset additions and one-asset removals within the known 56-asset pool.

If one or more assets are removed, the forward path contained assets that became redundant/harmful after later additions. Re-run until the branch stops naturally; that final universe is stronger than the pure forward-greedy result under the exact same historical judge.

This remains an in-sample reconstruction experiment. It does not establish out-of-sample generalization and must not be merged/tagged for production.
