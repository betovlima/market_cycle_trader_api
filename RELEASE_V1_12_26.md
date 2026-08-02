# Market Cycle Trader API v1.12.26

- Preserves the complete backtest, exports, Alpaca Paper, portfolio, scheduler, and administrative APIs from the supplied canonical code.
- Fixes `PaperTradingSettings` validation after endpoint-driven bootstrap by excluding `revision` and all administrative metadata.
- Applies the same metadata boundary to paper-state reads.
- Removes the Railway database pre-deploy command.
- Contains no database configuration scripts.
- Does not change strategy, model, capital, asset, market-data, or Alpaca execution parameters.
