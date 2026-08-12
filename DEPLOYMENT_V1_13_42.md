# API v1.13.42 deployment

## Safety order

1. Ensure no backtest is queued or running.
2. Prefer deployment while the XNYS regular market is closed.
3. Ensure no Paper plan is prepared/executing before Winner promotion.
4. Deploy API v1.13.42 before Front v1.12.111.
5. Verify health endpoints.
6. In Administrator → System Settings → SELECTED STRATEGY, select the Strategy revision used by the validated LightGBM run.
7. Select `LightGBM Utility` in Model Parameters.
8. Click `Mark as candidate`. The API locates the latest completed LightGBM job for that exact Strategy revision and freezes its model snapshot.
9. Confirm Candidate shows `LightGBM Utility`.
10. When the regular market is closed and no protected Paper plan is pending, click `Promote to Trader winner`.
11. Confirm Trader Winner shows `LightGBM Utility`.
12. The next scheduled pre-market cycle trains/calibrates the live LightGBM engine from the immutable Winner model snapshot.

## Compatibility

Legacy Winners without `winner_model_snapshot` continue as Strategy-owned XGBoost Winners. IQN remains non-promotable. No MongoDB migration command is required; model-aware fields are additive.
