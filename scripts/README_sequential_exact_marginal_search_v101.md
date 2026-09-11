# Sequential Exact Marginal Search v1.0.1 — Replay Profiler

Branch: `research/sequential-exact-marginal-search-v1`.
Research version: `sequential-exact-marginal-search-v1.0.1`.
API/package remains **10.8.48** because this patch changes only the standalone research harness/tests; the production engine contract is not changed.

## Why this patch exists

Round 1 against the Original25 proved that many assets have positive exact marginal capital contribution, but the 31-candidate round required hours even with four workers. Before changing the exact judge, this patch measures where the wall-clock time is actually spent.

The profiler wraps the existing `asset_discovery._run_rotation_replay` with Python `cProfile`. It does **not** change Strategy parameters, candidate selection, model fitting, decision logic, fees, slippage, folds, capital accounting or the `DeltaCapital > 0` rule.

Each exact replay writes:

- a binary `.prof` file for detailed inspection;
- a text file with the 50 functions with highest cumulative time;
- `replay_profile_index.csv` with asset count, elapsed seconds and artifact names.

The patch also removes the standalone runner's `Timestamp.utcnow()` deprecation warning by using `Timestamp.now(tz="UTC")`; this does not affect research results.

## Minimal profiling run

Do not run another 31-candidate round. Profile only the Original25 baseline and one known high-impact candidate, GKOS:

```bash
git fetch origin
git switch research/sequential-exact-marginal-search-v1
git pull --ff-only origin research/sequential-exact-marginal-search-v1

git log -1 --oneline

python -m unittest discover \
  -s tests \
  -p "test_sequential_exact_marginal_search*.py" \
  -v

python scripts/research_sequential_exact_marginal_search_v101.py \
  --strategy-sequence 10 \
  --history-start 2016-01-01 \
  --snapshot-end 2026-09-04 \
  --candidate-symbols GKOS \
  --max-rounds 1 \
  --workers 1 \
  --output-dir research_output/sequential_exact_marginal_search_profile_v101 \
  --fresh-run
```

This performs only two exact replays: the Original25 baseline and `Original25 + GKOS`. The purpose is to identify the dominant cumulative-time functions before implementing any cache or incremental execution path.

## What to send back

Zip `research_output/sequential_exact_marginal_search_profile_v101`. The key files are:

- `replay_profiles/replay_profile_index.csv`;
- `replay_profiles/replay_*.txt`;
- `sequential_round_evaluations.csv`;
- `sequential_path.csv`;
- `sequential_exact_summary.json`.

The next optimization will be chosen only from measured hotspots. Any accelerator must pass parity testing against the unmodified exact judge for both positive and negative candidates before it can be used for multi-round reconstruction.

Do not merge or tag this research branch for production.
