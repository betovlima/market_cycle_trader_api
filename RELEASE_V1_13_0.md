# Market Cycle Trader API v1.13.0

Multi-horizon series-movement research release.

- Adds weighted utility targets for 5, 10, 20, 40 and 60 sessions.
- Adds movement-capture and trend-persistence target components.
- Expands price, volatility, trend, channel, momentum and volume features.
- Keeps expanding walk-forward validation and requires purge >= maximum target horizon.
- Exposes the new fields through the existing protected strategy-configuration API.
- Does not promote a configuration automatically to Paper Trading.
