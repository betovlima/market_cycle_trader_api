## v2.0.7 — Frozen market-data snapshots for Latin Hypercube / CARO

- Fixes `MarketDataSignatureMismatch` when CARO imports observations from an earlier Latin Hypercube campaign and the shared MongoDB market cache has changed afterward.
- Every new tuning campaign is bound to a physical immutable, content-addressed candle snapshot before candidate training starts.
- Derived CARO reuses the exact physical snapshot of its source Latin Hypercube campaign.
- Candidate backtests read the frozen snapshot instead of the mutable operational market cache.
- Research signatures use only the OHLCV fields consumed by the rotation models; VWAP/trade_count remain audit-only.
- Legacy hash-only source campaigns fail before candidate training when the old candles can no longer be verified, rather than mixing incomparable datasets.
- Immediate tuning cancellation from API v2.0.6 remains unchanged.
- Risk-Off/CASH logic from API v2.0.5 remains unchanged.
- Frontend remains v2.0.2.
