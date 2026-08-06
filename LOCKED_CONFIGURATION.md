# Strategy catalog and protected Trader winner

All strategy changes are performed through protected HTTP endpoints. Do not write strategy documents directly in MongoDB.

## Research strategies

Administrator sessions use `/api/admin/strategies` to clone, edit, select, test, and delete draft strategies. Every configuration field is validated by `BacktestRequest`. Selecting a research strategy changes only future backtests.

```http
GET    /api/admin/strategies
POST   /api/admin/strategies
GET    /api/admin/strategies/{strategy_id}
PUT    /api/admin/strategies/{strategy_id}
DELETE /api/admin/strategies/{strategy_id}
POST   /api/admin/strategies/{strategy_id}/select-for-backtest
```

## Trader winner

Trader reads only the immutable profile referenced by `strategy_control/default.trader_winner_strategy_id`. Editing or selecting a research profile never changes this pointer.

A candidate can change Trader only through explicit Administrator promotion after a completed backtest of the same profile revision:

```http
POST /api/admin/strategies/{strategy_id}/promote-to-trader
```

Promotion creates a new locked snapshot, preserves the previous winner as a locked former winner, and requires Paper state reinitialization before Trader can restart.

## Bundled winner recovery

The packaged recovery source remains:

```text
src/market_cycle_trader_api/parameterizations/winner-v1.13.2.json
```

Install it only through:

```http
POST /api/admin/strategy-configuration/winner/install
```

This explicit recovery operation resets both the research selection and Trader winner to the bundled snapshot. Direct PATCH, PUT, reset, and history restore operations on the legacy strategy-configuration route are disabled.
