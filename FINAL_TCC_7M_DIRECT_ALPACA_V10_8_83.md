# TCC — original 10.8.74 request reconstruction v10.8.83

## Why v10.8.82 still diverged

v10.8.82 fixed the fresh Alpaca bar count:

- 55 eligible assets
- 148,060 eligible RAW bars
- 17 applied splits
- zero row-count mismatches
- zero corporate-action-count mismatches
- zero split-count mismatches

However it still produced 1,547 replay sessions and executed through 2026-09-17,
while the recovered 10.8.74 result contains 1,546 replay sessions and executes
through 2026-09-16.

The uploaded research artifacts contain a `controlled_request` reconstructed
from the same source job:

`20260918T234903-52bd06f3`

The audit code shows that it starts from:

```python
request = BacktestExecutionRequest.model_validate(job["request"])
```

and only replaces assets/common-end/reference lists and market-data snapshot
metadata. Therefore the remaining fields preserve the original source-job
semantics.

## Recovered source-job semantics

Relevant request fields:

```text
start_date          = 2016-01-01
end_date            = null
analysis_start_date = 2016-01-01
analysis_end_date   = 2026-09-17
```

The original source request also contained:

```text
market_data_provider = tiingo
alpaca_adjustment    = all
backfill_enabled     = true
backfill_provider    = tiingo
```

These values are part of the source-job request. They do NOT mean that the
10.8.74 RAW research used adjusted-all data.

The 10.8.74 research runner itself replaced the model-facing data semantics
after reading the frozen RAW collection:

```text
market_data_provider      = alpaca
alpaca_adjustment         = split
research_market_data_mode = database_only
```

v10.8.83 preserves the same distinction.

## Direct Alpaca transport

The standalone data transport remains:

```text
feed       = SIP
timeframe  = 1Day
download   = RAW
bar as-of  = 2026-09-17
```

No MongoDB market data is read or written.

## Recovered LightGBM settings

The original request did not pre-populate Soft Horizon Consensus settings.
It contained:

```json
{
  "schema_version": 3,
  "settings_revision": 2,
  "profile_id": "strategy",
  "lightgbm": {
    "n_estimators": 329,
    "learning_rate": 0.020731,
    "max_depth": 3,
    "num_leaves": 6,
    "min_child_samples": 18,
    "min_child_weight": 5.0,
    "subsample": 0.85,
    "subsample_freq": 0,
    "colsample_bytree": 0.88067,
    "reg_alpha": 0.050837,
    "reg_lambda": 3.596305,
    "max_bin": 255,
    "n_jobs": -1,
    "repetitions": 1,
    "seed_step": 1000,
    "random_state": 42
  }
}
```

The 10.8.74 runner then explicitly applied:

```text
early_stopping_enabled = false
horizon_voting.enabled = false
soft_horizon_consensus.enabled = false  # Control
```

and for the challenger:

```text
soft_horizon_consensus.enabled = true
penalty_strength = 1.0
```

v10.8.83 now reproduces that sequence rather than storing a later reconstructed
settings object in the frozen request.

## Other recovered top-level defaults

The source-job request also contained the original top-level XGBoost/default
fields:

```text
rotation_xgb_n_estimators = 300
rotation_xgb_learning_rate = 0.035
xgb_colsample_bytree = 0.85
xgb_reg_alpha = 0.1
xgb_reg_lambda = 2.0
xgb_n_jobs = -1
```

The actual research model remains `lightgbm_utility` with the 329-estimator
settings above.

## GPU

Per the current research decision, GPU remains mandatory:

```text
rotation_accelerator = cuda
rotation_allow_cpu_fallback = false
deterministic_execution = false
```

## Version

- API/package: `10.8.83`
- branch: `research/api-v10.8.83-soft-horizon-7m-original-request-gpu`
- config: `research/soft_horizon_7m_direct_alpaca_v10_8_83.json`
- runner: `scripts/research_soft_horizon_7m_direct_alpaca.py`
- output: `output/soft_horizon_7m_direct_alpaca_v10883/`

## Run

```bash
git fetch origin
git switch research/api-v10.8.83-soft-horizon-7m-original-request-gpu
git pull --ff-only origin research/api-v10.8.83-soft-horizon-7m-original-request-gpu

python -m pytest tests/test_soft_horizon_7m_direct_alpaca.py -q

python scripts/research_soft_horizon_7m_direct_alpaca.py --replace-snapshot
```

Do not reuse the v10.8.82 snapshot because the configuration hash and
corporate-action cutoff contract changed.

## What to verify

Before interpreting capital, verify:

```text
eligible RAW rows = 148,060
splits applied = 17
prediction/session count = 1,546
last execution session = 2026-09-16
requested device = cuda
effective device = gpu
```

Then compare:

```text
reference Control = 5,551,143.96
reference Soft    = 7,376,955.56
reference Soft changed base actions = 11
```
