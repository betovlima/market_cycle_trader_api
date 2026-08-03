# Winner installation and execution sequence

All commands are HTTP operations through Swagger or another API client. Do not write strategy documents directly in MongoDB.

## Required headers

Administrative endpoints:

```text
X-Parameter-Bootstrap-Token: <PARAMETER_BOOTSTRAP_API_TOKEN>
```

Paper endpoints:

```text
X-Paper-Market-Token: <PAPER_MARKET_API_TOKEN>
```

## 1. Start and verify the API

```http
GET /api/health/live
```

Expected API version: `1.13.4`.

```http
GET /api/health/ready
```

With an empty database, strict readiness can report that the locked configuration is unavailable. This is expected until the winner installation finishes.

## 2. Install winner-v1.13.2 from the packaged file

```http
POST /api/admin/strategy-configuration/winner/install
```

Payload file:

```text
script/post_api_admin_strategy-configuration_winner_install.json
```

The endpoint reads only the packaged file:

```text
src/market_cycle_trader_api/parameterizations/winner-v1.13.2.json
```

It performs these strategy-only changes:

1. Validates the JSON against `BacktestRequest`.
2. Verifies configuration SHA-256 `22a4193fbb30de33d75864fc28c3b1923e4dedd4970b14f9537f793bccf18953`.
3. Replaces or creates `backtest_settings/default`.
4. Deletes every extra document from `backtest_settings`.
5. Deletes every document from `backtest_settings_history`.
6. Stores the winner as revision 1.

It does not delete backtest results, Alpaca market bars, or Paper execution data.

The operation is rejected when a backtest is queued/running or a Paper run is active.

Expected response values:

```text
status: winner_installed
source_file: winner-v1.13.2.json
configuration_hash: 22a4193fbb30de33d75864fc28c3b1923e4dedd4970b14f9537f793bccf18953
metadata.revision: 1
metadata.configuration_name: winner-v1.13.2
```

## 3. Install missing non-strategy parameter documents

```http
GET /api/admin/parameters/status
```

When Paper settings are missing:

```http
POST /api/admin/parameters/bootstrap
```

Payload file:

```text
script/post_api_admin_parameters_bootstrap.json
```

Bootstrap preserves the valid winner strategy and inserts missing non-strategy documents.

## 4. Confirm the winner strategy

```http
GET /api/admin/strategy-configuration
```

Required values:

```text
strategy_mode: COMPOUND_ROTATION_SWING_XGBOOST
rotation_accelerator: cpu
random_state: 42
rotation_target_horizons: [5, 10, 20, 40, 60]
rotation_target_horizon_weights: [0.1, 0.15, 0.2, 0.3, 0.25]
rotation_movement_capture_weight: 0.35
rotation_trend_persistence_weight: 0.2
configuration_hash: 22a4193fbb30de33d75864fc28c3b1923e4dedd4970b14f9537f793bccf18953
metadata.revision: 1
metadata.winner_source_file: winner-v1.13.2.json
```

## 5. Inspect and initialize Paper state

```http
GET /api/admin/setup/status
```

Then initialize without arming a run:

```http
POST /api/admin/setup/initialize
```

Payload file:

```text
script/post_api_admin_setup_initialize.json
```

## 6. Run the full-history validation

```http
POST /api/jobs
```

Request body: none. The execution period is loaded from the installed winner configuration.

Read the returned job id, then poll:

```http
GET /api/jobs/{job_id}
```

After `status=completed`:

```http
GET /api/jobs/{job_id}/results
```

Do not arm Paper when the job fails or when the reproduced result is materially inconsistent with the champion artifact.

## 7. Arm Paper only after validation

Inspect current state:

```http
GET /api/paper-market/status
```

Cancel an obsolete active run when necessary:

```http
POST /api/paper-market/{run_id}/cancel
```

Payload file:

```text
script/post_api_paper-market_run-id_cancel.json
```

Arm the next regular session:

```http
POST /api/paper-market/start-next-session
```

Payload file:

```text
script/post_api_paper-market_start-next-session.json
```
