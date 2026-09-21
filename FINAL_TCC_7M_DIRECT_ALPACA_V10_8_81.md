# TCC — Direct Alpaca RAW replay v10.8.81

## Purpose

Re-run the recovered 10.8.74 Soft Horizon Consensus experiment using a fresh
direct Alpaca load with the same historical bar semantics that created
`alpaca_market_bars_raw_20260919`.

Reference result:

- Control: US$ 5,551,143.96
- Soft Horizon Consensus: US$ 7,376,955.56
- Soft changed base actions: 11
- Base commit: `05b765df0496c905a4195948027a9b3b7adf2bce`

## Correct Alpaca bar configuration

The 10.8.74 RAW collection was created by
`scripts/download_fresh_alpaca_snapshot.py` through
`market_cycle_trader_api.infrastructure.market_data.alpaca.download_stock_bars`.

v10.8.81 reuses that exact downloader and semantics:

```text
feed       = sip
timeframe  = 1Day
adjustment = raw
start      = 2016-01-01
end        = 2026-09-18
API end    = 2026-09-19  (end + 1 calendar day so Sep 18 is included)
```

Bars are downloaded one symbol at a time, just as in the original snapshot
builder. No `adjustment=all` is used anywhere in this experiment.

The raw bars are saved to immutable local CSV files with 17 significant digits.

## Corporate actions

The corporate-actions request matches
`download_alpaca_corporate_actions_snapshot.py`:

```text
query start = research_start - 366 days
query end   = 2026-09-18
region      = us
data_quality= complete
limit       = 1000
sort        = asc
```

The same action types are requested and the same local split normalization is
applied after loading RAW bars.

## GPU

This run uses GPU as requested:

```text
rotation_accelerator       = cuda
rotation_allow_cpu_fallback= false
deterministic_execution    = false
```

CPU/GPU equivalence was studied separately. This run does not fall back silently
to CPU.

## Reference data audit

The reference 10.8.74 result had:

```text
requested assets          = 56
eligible assets           = 55
eligible RAW rows         = 148,060
RAW rows per eligible asset = 2,692
splits applied            = 17
excluded                  = DOC -> PEAK
```

The frozen audit source is:

`research/reference_10_8_74_raw_snapshot_diagnostics.json`

Before model execution, v10.8.81 writes:

- `results/reference_10_8_74_data_audit.csv`
- `results/reference_10_8_74_data_audit.json`

The audit compares, per asset:

- RAW row count
- corporate-action count
- split count

Differences are printed in the log before training. This audit is informational
only. It is not model input and is not a tuning gate.

## Version

- API/package: `10.8.81`
- branch: `research/api-v10.8.81-soft-horizon-7m-direct-alpaca-raw-gpu`
- config: `research/soft_horizon_7m_direct_alpaca_v10_8_81.json`
- runner: `scripts/research_soft_horizon_7m_direct_alpaca.py`
- runner version: `soft-horizon-7m-direct-alpaca-v1.1.0`

## Run

```bash
git fetch origin
git switch research/api-v10.8.81-soft-horizon-7m-direct-alpaca-raw-gpu
git pull --ff-only origin research/api-v10.8.81-soft-horizon-7m-direct-alpaca-raw-gpu

python -m pytest tests/test_soft_horizon_7m_direct_alpaca.py -q

python scripts/research_soft_horizon_7m_direct_alpaca.py --replace-snapshot
```

Output:

`output/soft_horizon_7m_direct_alpaca_v10881/`

If the download completes and a later processing step fails, preserve the
downloaded snapshot and resume with:

```bash
python scripts/research_soft_horizon_7m_direct_alpaca.py --reuse-snapshot
```

Do not tune parameters to recover the reference capital. First inspect the data
audit to determine whether the fresh Alpaca RAW/corporate-action snapshot differs
from the frozen 10.8.74 snapshot.
