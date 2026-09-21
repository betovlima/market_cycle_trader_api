# TCC — CPU isolation on exact v10.8.83 snapshot (v10.8.84)

## Objective

Determine whether the remaining divergence from the recovered 10.8.74 result is
caused by the LightGBM compute backend.

v10.8.84 changes exactly one experimental factor relative to v10.8.83:

```text
v10.8.83: rotation_accelerator = cuda
v10.8.84: rotation_accelerator = cpu
```

Everything else is preserved.

## Exact snapshot reuse

v10.8.84 must reuse the immutable v10.8.83 snapshot:

```text
snapshot_sha256 =
5b4a2dac1ed9a6128c504d3cb12048726bda3f879f7448317a578776ac3d3dde
```

The runner validates:

- manifest snapshot SHA-256;
- SHA-256 of every RAW bar file;
- SHA-256 of corporate_actions.jsonl.

No Alpaca download is performed for this test.

## Preserved experiment contract

- 56 requested assets
- 55 eligible assets
- DOC excluded by structural identity guard
- 148,060 eligible RAW bars
- 17 applied splits
- 1,546 simulation sessions
- last execution session: 2026-09-16
- original recovered request semantics
- same LightGBM settings
- same folds
- same seeds
- same Soft Horizon Consensus penalty strength
- deterministic_execution remains false
- rotation_allow_cpu_fallback remains false

Keeping deterministic_execution unchanged is intentional: this test isolates the
compute device itself instead of changing both device and determinism.

## GPU reference (v10.8.83)

```text
Control = 4,150,383.1310602436
Soft    = 4,489,769.453271521
Soft changed base actions = 16
requested device = cuda
effective device = gpu
```

v10.8.84 writes automatic deltas against those values into summary.json.

## Version

- API/package: `10.8.84`
- branch: `research/api-v10.8.84-soft-horizon-7m-cpu-isolation`
- config: `research/soft_horizon_7m_direct_alpaca_v10_8_84.json`
- runner: `scripts/research_soft_horizon_7m_direct_alpaca.py`
- results: `output/soft_horizon_7m_direct_alpaca_v10884/results/`

## Run

The v10.8.83 snapshot must still exist locally at:

`output/soft_horizon_7m_direct_alpaca_v10883/snapshot`

Run:

```bash
git fetch origin
git switch research/api-v10.8.84-soft-horizon-7m-cpu-isolation
git pull --ff-only origin research/api-v10.8.84-soft-horizon-7m-cpu-isolation

python -m pytest tests/test_soft_horizon_7m_direct_alpaca.py -q

python scripts/research_soft_horizon_7m_direct_alpaca.py \
  --snapshot-source-dir output/soft_horizon_7m_direct_alpaca_v10883/snapshot
```

Do not pass `--replace-snapshot`. This CPU test must use the exact bytes already
used by the GPU run.

## Interpretation

If CPU returns close to the recovered 10.8.74 values:

```text
Control = 5,551,143.96
Soft    = 7,376,955.56
Soft changed base actions = 11
```

while using the exact same v10.8.83 snapshot, then the compute backend is a
material source of the remaining divergence.

If CPU remains close to the v10.8.83 GPU values, then the cause is elsewhere in
model training/inference and the next step is a fold/model-level prediction audit.
