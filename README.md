# Market Cycle Trader API

Market Cycle Trader API is a FastAPI backend for backtesting and operating an XGBoost-based capital rotation strategy.

The service manages historical market data, walk-forward model execution, strategy configuration stored in MongoDB, portfolio state, scheduled next-session decisions, and Alpaca paper-trading orders using an isolated strategy budget.

## Main features

- XGBoost-only strategy runtime
- Walk-forward backtesting
- Alpaca historical market data integration
- Automated Alpaca paper-trading execution
- Next-session BUY, HOLD, SELL, and rotation decisions
- Isolated paper portfolio with a configurable strategy budget
- MongoDB configuration and execution persistence
- Automated environment initialization through administrative APIs
- Portfolio status, order history, and process logs
- Railway-ready deployment
