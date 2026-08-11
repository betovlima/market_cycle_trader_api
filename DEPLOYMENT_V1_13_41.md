# Market Cycle Trader — API v1.13.41 + Frontend v1.12.110

## Important before deploy

Let the XGBoost backtest currently running on v1.13.40 finish and export/save its result before deploying this release. Railway will restart the API process during deployment and an active local/remote execution could be interrupted.

## What changes

- Each model owns its own research profile: XGBoost, LightGBM and IQN.
- Same-named parameters are independent between model profiles.
- Model selector and model parameter editor live inside `SELECTED STRATEGY`.
- The separate Model Research block is removed from System Settings.
- The Trader Winner and persisted Strategy snapshots are not rewritten.

## Deploy API first

```powershell
cd C:\Users\betov\des\gpt\extrema_backtest_dashboard_mongo\market_cycle_trader\market_cycle_trader_api

git status
git switch main
git pull --ff-only origin main

git switch -c feature/1.13.41-independent-model-parameterization

git add -A
git status
git commit -m "feat(research): isolate parameters for xgboost lightgbm and iqn"
git push -u origin feature/1.13.41-independent-model-parameterization

git switch main
git pull --ff-only origin main
git merge --no-ff feature/1.13.41-independent-model-parameterization -m "release: Market Cycle Trader API v1.13.41"
git push origin main

git tag -a api-v1.13.41 -m "Market Cycle Trader API v1.13.41 - independent model parameterization"
git push origin api-v1.13.41
```

Confirm `/api/health/live` and `/api/health/ready` before deploying the frontend.

## Deploy frontend

```powershell
cd C:\Users\betov\des\gpt\extrema_backtest_dashboard_mongo\market_cycle_trader\market_cycle_trader

git status
git switch main
git pull --ff-only origin main

git switch -c feature/1.12.110-selected-strategy-model-parameters

git add -A
git add -f public
git status
git commit -m "feat(settings): move model parameters into selected strategy"
git push -u origin feature/1.12.110-selected-strategy-model-parameters

git switch main
git pull --ff-only origin main
git merge --no-ff feature/1.12.110-selected-strategy-model-parameters -m "release: Market Cycle Trader frontend v1.12.110"
git push origin main

git tag -a front-v1.12.110 -m "Market Cycle Trader frontend v1.12.110 - selected strategy model parameters"
git push origin front-v1.12.110
```

## Post-deploy validation

1. Log in as Administrator.
2. Open System Settings → Strategy workspace.
3. Confirm the `SELECTED STRATEGY` box contains the Model Parameters selector.
4. Select XGBoost, LightGBM and IQN and confirm each has a separate parameter set.
5. Confirm changing one model does not change values in another model.
6. Save one harmless research-profile change with an audit reason, then restore it if desired.
7. Start a research backtest and confirm the selected model appears in Administrator execution status/history.
8. Confirm the Trader Winner remains unchanged.
