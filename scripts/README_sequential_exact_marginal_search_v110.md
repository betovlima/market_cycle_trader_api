# Sequential Exact Marginal Search v1.1.0 — Exact Prediction-Cache Accelerator

Branch: `research/sequential-exact-marginal-search-v1`.

Research version: `sequential-exact-marginal-search-v1.1.0`.

API/package remains **10.8.48**. This is a research-only acceleration wrapper around the existing exact judge; production engine source is unchanged.

## Measured hotspot

The v1.0.1 profile measured two exact replays:

- Original25: 389.3 s wall time.
- Original25 + GKOS: 448.4 s wall time.

The dominant path was not feature generation or model fitting. It was repeated one-row inference inside `capital_rotation._model_utilities`:

- Original25: `_model_utilities` cumulative 337.9 s, 75,950 LightGBM `predict` calls.
- Original25 + GKOS: `_model_utilities` cumulative 420.5 s, 78,988 LightGBM `predict` calls.
- `_simple_policy_growth` alone consumed 125.0 s / 172.7 s because the same fitted models were queried repeatedly while comparing switch-margin policies.
- LightGBM's sklearn wrapper repeatedly rebuilt pandas conversion metadata and CPU-count state for each one-row prediction.

The expensive repetition is mathematically unnecessary: for a fixed fitted model and fixed aligned frame, the predicted utility for `(symbol, timestamp)` is immutable during that replay.

## Accelerator

v1.1.0 keeps model fitting, folds, calibration, decision policies, fees, slippage, execution and capital accounting unchanged.

During one exact replay only, it:

1. detects a fitted model set the first time `_model_utilities` is called;
2. performs one **batched `model.predict` per asset** over every valid timestamp;
3. stores those predictions in a thread-local in-memory cache;
4. serves all later policy/margin-calibration utility requests from the immutable cache;
5. discards the cache at the end of that replay.

The cache is isolated per worker thread and per exact replay. It is not persisted and cannot leak across candidate universes.

Uncommon ambiguous frames (for example duplicate indexes) fall back to the original per-row semantics.

## Mandatory parity run

Before using the accelerator for a multi-candidate round, run it with `--parity-check`.

Parity mode executes every replay twice:

1. original `_model_utilities`;
2. accelerated batched cache.

It requires identical decision sessions and recursively compares every aggregate replay metric with absolute tolerance `1e-12`. A mismatch aborts the run.

Use only one worker in parity mode.

```bash
git fetch origin
git switch research/sequential-exact-marginal-search-v1
git pull --ff-only origin research/sequential-exact-marginal-search-v1

python -m unittest discover \
  -s tests \
  -p "test_sequential_exact_marginal_search*.py" \
  -v

python scripts/research_sequential_exact_marginal_search_v110.py \
  --parity-check \
  --strategy-sequence 10 \
  --history-start 2016-01-01 \
  --snapshot-end 2026-09-04 \
  --candidate-symbols GKOS \
  --max-rounds 1 \
  --workers 1 \
  --output-dir research_output/sequential_exact_marginal_search_accel_parity_v110 \
  --fresh-run
```

The run writes `accelerator_replay_index.csv` with original time, accelerated time, speedup, parity status, prediction-cache build counts and batch-prediction statistics.

## After parity passes

Do **not** immediately run 31 sequential rounds. First use one accelerated Round 2 against `Original25 + GKOS` and measure the new wall time. The branch must continue using the exact economic judge; the accelerator is valid only while parity remains green.

Do not merge or tag this research branch for production.
