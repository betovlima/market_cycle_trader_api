# Fresh-database execution sequence

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

Expected API version: `1.13.1`.

```http
GET /api/health/ready
```

With an empty database, strict readiness can report that the locked configuration is unavailable. This is expected until bootstrap finishes.

## 2. Inspect the empty database

```http
GET /api/admin/parameters/status
```

No payload. A cleared database should report missing parameter documents.

## 3. Install canonical documents

```http
POST /api/admin/parameters/bootstrap
```

Payload file:

```text
script/post_api_admin_parameters_bootstrap.json
```

Run the status endpoint again. Required result:

```text
all_present: true
all_valid: true
```

## 4. Confirm the champion strategy

```http
GET /api/admin/strategy-configuration
```

No payload. A fresh bootstrap should return revision 1 with:

```text
strategy_mode: COMPOUND_ROTATION_SWING_XGBOOST
rotation_accelerator: cpu
random_state: 42
rotation_target_horizons: [5, 10, 20, 40, 60]
rotation_target_horizon_weights: [0.1, 0.15, 0.2, 0.3, 0.25]
rotation_movement_capture_weight: 0.35
rotation_trend_persistence_weight: 0.2
```

The complete replacement payload is available at:

```text
script/put_api_admin_strategy-configuration_champion.json
```

Use it only when the active configuration differs. Before using it, replace `expected_revision` with the revision returned by the latest GET.

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

Payload file:

```text
script/post_api_jobs_full_history.json
```

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
