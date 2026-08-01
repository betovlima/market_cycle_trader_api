# Market Cycle Trader API v1.12.11

FastAPI backend for the XGBoost-only Compound Capital Rotation strategy.

## Runtime modules

- `engine/capital_rotation.py`: XGBoost walk-forward training and simulation.
- `engine/live_xgboost_signal.py`: next-session paper-trading decision.
- `engine/market_data.py`: historical market-data loading and MongoDB cache.
- `services/paper_trading.py`: isolated Alpaca Paper portfolio workflow.

The API does not include a QR-DQN execution path and does not require PyTorch.

## Active configuration

MongoDB collection: `backtest_settings`

Active document: `_id = "default"`

Use `scripts/apply_locked_config.py` with the complete JSON from `configs/`.

## Next-session Alpaca Paper API

The API can arm one locked XGBoost execution for the next regular Alpaca paper session.
It uses the Alpaca clock instead of assuming weekdays and persists its state in
`paper_market_runs`.

```bash
curl -X POST "http://127.0.0.1:8000/api/paper-market/start-next-session" \
  -H "X-Paper-Market-Token: $PAPER_MARKET_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"confirm_paper":true}'
```

See `NEXT_SESSION_MARKET_API_V1_12_0.md` at the project root.

## Idempotent parameter bootstrap

Set `PARAMETER_BOOTSTRAP_API_TOKEN`, then inspect or insert missing parameter documents:

```bash
curl "http://127.0.0.1:8000/api/admin/parameters/status" \
  -H "X-Parameter-Bootstrap-Token: YOUR_TOKEN"

curl -X POST "http://127.0.0.1:8000/api/admin/parameters/bootstrap" \
  -H "X-Parameter-Bootstrap-Token: YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"confirm_insert_missing_only":true}'
```

Existing `_id=default` documents are never overwritten. The equivalent CLI is:

```bash
python scripts/bootstrap_parameters.py --status
python scripts/bootstrap_parameters.py
```
