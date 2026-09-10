# Exact Marginal Capital Search v1.0.3

Branch: `research/exact-marginal-capital-search-v1`.
Research version: `exact-marginal-capital-search-v1.0.3`.
API/package remains `10.8.47`; this patch changes only the standalone research harness and tests.

## Why v1.0.3 exists

The v1.0.2 smoke run proved the exact judge can now evaluate external assets, but it also exposed a historical ticker-identity problem. `LB` produced a complete raw OHLCV window yet the exact model context covered only 817 of the baseline's 1,538 decision sessions and was rejected after an expensive replay. Public issuer history confirms that old `LB` (L Brands) changed to `BBWI` in 2021, while a different issuer, LandBridge, later began trading as `LB` in 2024.

The production Asset Discovery already contains an economic-identity guard using Alpaca corporate actions before the fast scan. v1.0.3 restores that same guard to Exact Marginal Capital Search before candidate history download and before exact replay.

## Contract

For the full candidate set, the first candidate evaluation performs one batched identity-integrity lookup using the existing Asset Discovery implementation. Results are cached in-memory for the process.

Candidates with an identity break are exported with:

- `evaluation_status=context_rejected`;
- `rejection_reason=economic_identity_discontinuity`;
- identity source/reason/event counts and serialized break details;
- no candidate-history replay and no exact replay time.

Candidates that pass identity integrity continue unchanged through v1.0.2: local candidate history first, then transient chunked Alpaca fallback if missing, preselector diagnostics, and the exact `_run_rotation_replay -> run_rotation_models` economic judge.

The preselector still never accepts or rejects a candidate.

## Run

```bash
git fetch origin
git switch research/exact-marginal-capital-search-v1
git pull --ff-only origin research/exact-marginal-capital-search-v1

python -m unittest discover \
  -s tests \
  -p "test_exact_marginal_capital_search*.py" \
  -v

python scripts/research_exact_marginal_capital_search_v103.py \
  --strategy-sequence 10 \
  --history-start 2016-01-01 \
  --snapshot-end 2026-09-04 \
  --candidate-symbols HL VSAT RARE CCS LB \
  --workers 1 \
  --output-dir research_output/exact_marginal_capital_search_smoke_v103 \
  --fresh-run
```

`LB` is deliberately retained in this smoke test as a negative control for the identity-integrity guard. `HL` is included because an older Asset Discovery campaign recorded it as a strong positive marginal-capital candidate, while VSAT and RARE are known negative controls. The absolute results can differ when the source Strategy/revision/universe differs; this smoke test is primarily a correctness and identity-parity check.

Do not merge or tag this research branch for production.
