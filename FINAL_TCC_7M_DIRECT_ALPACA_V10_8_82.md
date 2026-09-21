# TCC — Alpaca RAW as-of reproduction v10.8.82

## Finding from v10.8.81

The v10.8.81 direct Alpaca snapshot was valid RAW/SIP data and GPU was
effectively used, but the bar set was not identical to the snapshot that
produced the 10.8.74 ~US$7.38M result.

Reference 10.8.74:

- requested assets: 56
- bars per asset: 2,692
- total requested RAW bars: 150,752
- eligible assets: 55
- eligible RAW bars: 148,060
- splits applied: 17
- structural exclusion: DOC -> PEAK

v10.8.81:

- requested assets: 56
- bars per asset: 2,693
- total requested RAW bars: 150,808
- eligible assets: 55
- eligible RAW bars: 148,115
- splits applied: 17
- structural exclusion: DOC -> PEAK

The difference is exactly one additional daily bar per asset. The new direct
snapshot includes the 2026-09-18 session. The reference snapshot did not.

This is also visible in the prediction calendar:

- reference 10.8.74 final decision date: 2026-09-16
- v10.8.81 final decision date: 2026-09-17

Corporate-action counts and applied split counts matched the reference.

## v10.8.81 result

GPU execution was confirmed:

- requested device: cuda
- effective device: gpu

Result:

- Control: US$ 4,250,935.99
- Soft Horizon Consensus: US$ 4,484,480.81
- Soft vs Control: +US$ 233,544.82 (+5.49%)
- Soft changed base actions: 16

This is not treated as a direct reproduction of the 10.8.74 snapshot because the
bar set contains an additional session.

## v10.8.82 correction

The scientific research window remains:

`2016-01-01 -> 2026-09-18`

Corporate actions also remain queried through:

`2026-09-18`

Only the historical bar snapshot as-of cutoff is recovered from the reference
10.8.74 data:

`bar_snapshot_as_of_end = 2026-09-17`

The Alpaca bar request therefore uses an API upper bound of 2026-09-18 so the
last included daily session is 2026-09-17.

Expected bar audit after the fresh download:

```text
eligible_rows=148060
reference_rows=148060
row_mismatches=0
split_mismatches=0
```

GPU remains mandatory:

```text
rotation_accelerator=cuda
rotation_allow_cpu_fallback=false
deterministic_execution=false
```

## Version

- API: `10.8.82`
- branch: `research/api-v10.8.82-soft-horizon-7m-alpaca-asof-cutoff`
- config: `research/soft_horizon_7m_direct_alpaca_v10_8_82.json`
- runner: `scripts/research_soft_horizon_7m_direct_alpaca.py`
- output: `output/soft_horizon_7m_direct_alpaca_v10882/`

## Run

```bash
git fetch origin
git switch research/api-v10.8.82-soft-horizon-7m-alpaca-asof-cutoff
git pull --ff-only origin research/api-v10.8.82-soft-horizon-7m-alpaca-asof-cutoff

python -m pytest tests/test_soft_horizon_7m_direct_alpaca.py -q

python scripts/research_soft_horizon_7m_direct_alpaca.py --replace-snapshot
```

Do not reuse the v10.8.81 snapshot for this test because it contains the extra
2026-09-18 session.
