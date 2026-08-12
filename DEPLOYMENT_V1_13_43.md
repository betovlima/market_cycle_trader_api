# API v1.13.43 deployment

## Safety order

1. Ensure no Backtest is queued or running.
2. Deploy API v1.13.43 before Front v1.12.112.
3. Verify `/api/health/live` and `/api/health/ready`.
4. Open Administrator → System Settings → `SELECTED STRATEGY`.
5. Confirm the desired Strategy shows its saved algorithm and parameters.
6. If changing the algorithm, save it there before opening Backtest.
7. Backtest must show the saved model as information only; start the run with `Start New Backtest`.
8. After completion, return to `SELECTED STRATEGY`; `Mark as candidate` is enabled only for the exact saved model snapshot.

No MongoDB migration command is required. Existing Strategy documents receive additive model-binding metadata. Existing exact completed model runs may be adopted automatically when the saved algorithm/parameter values match.
