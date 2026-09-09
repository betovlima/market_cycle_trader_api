# Counterfactual Rotation Advantage v3.0.0

## Research question

Can the existing LightGBM opportunity ranker remain fully invested while a separate
supervised model learns whether a newly ranked Top-1 is economically worth
rotating into versus HOLDing the incumbent?

This version intentionally removes CASH from the learned action space. It isolates
only the "best of the best" rotation problem observed in strong markets.

## Why v3 exists

Counterfactual Action Advantage v2 improved strongly over CAV v1, but its behavior
was dominated by one-session CASH timing:

- roughly half of OOS sessions were in CASH;
- hundreds of asset->CASH and CASH->asset transitions occurred;
- most CASH episodes lasted only one session;
- when invested, the portfolio still followed the current Top-1 most of the time.

That means v2 mixed two distinct research questions:

1. should we rotate from one good asset to another?
2. should we leave the market entirely?

v3 tests only the first question.

## Frozen components

- frozen asset universe;
- LightGBM opportunity target;
- LightGBM opportunity hyperparameters;
- state/action features;
- walk-forward folds;
- transaction-cost model;
- exact execution simulator;
- analysis dates and snapshot.

## Learned decision

When invested and the LightGBM Top-1 differs from the incumbent, the target is:

```text
rotation_advantage =
    next_session_net_log_return(ROTATE to Top-1)
  - next_session_net_log_return(HOLD incumbent)
```

The common close(t)->open(t+1) incumbent movement cancels from the comparison.
Transaction costs remain in the ROTATE branch.

The economic reference is exactly zero:

```text
HOLD advantage = 0
```

The policy is therefore:

```text
if predicted rotation advantage > 0:
    ROTATE to current Top-1
else:
    HOLD incumbent
```

Zero is not a tuned threshold. It is the mathematical indifference point of the
incremental target.

## CASH behavior

CASH is not a learned alternative in this experiment. The initial portfolio enters
the current Top-1 and stays exposed thereafter. This is deliberate: v3 is a
falsification test for rotation intelligence only.

A separate research version should address "best of the worst" / CASH timing only
if v3 proves that the rotation model can preserve compound growth.

## First run

Use only the frozen 56-asset universe:

```bash
git fetch origin
git switch research/counterfactual-rotation-advantage-v3
git pull --ff-only origin research/counterfactual-rotation-advantage-v3

python scripts/research_counterfactual_rotation_advantage.py \
  --strategy-sequence 10 \
  --history-start 2016-01-01 \
  --snapshot-end 2026-09-04 \
  --analysis-start 2020-07-22 \
  --analysis-end 2026-09-03 \
  --series 56 \
  --workers 4
```

The existing universe-scale ZIP under `research_output` is resolved automatically.

## Expected artifacts

```text
research_output/counterfactual_rotation_advantage_strategy_10_2020-07-22_to_2026-09-03/

counterfactual_rotation_advantage_comparison.csv
counterfactual_rotation_advantage_comparison.json
universe_56_predictions.csv
universe_56_trades.csv
universe_56_result.json
```

Decision diagnostics include:

```text
cra_reference_asset
cra_reference_advantage
cra_rotation_candidate_asset
cra_rotation_advantage
cra_positive_advantage
cra_cash_action_enabled
```

Decision reasons:

```text
CRA_ENTER_TOP1
CRA_HOLD_CURRENT_TOP1
CRA_HOLD_NONPOSITIVE_ADVANTAGE
CRA_ROTATE_POSITIVE_ADVANTAGE
```

## What to evaluate

Do not optimize for fewer rotations by itself. Compare:

- final compounded capital;
- CAGR;
- Sharpe;
- maximum drawdown;
- fold-by-fold capital;
- rotation count;
- average holding duration;
- number of Top-1 changes ignored because predicted advantage was non-positive;
- realized behavior of accepted versus rejected rotations.

The strongest comparison remains the frozen 56-asset baseline at approximately
US$23.52M. If v3 cannot preserve a substantial part of that compound edge, the
next change should address the rotation target horizon/continuation value rather
than adding manual thresholds.
