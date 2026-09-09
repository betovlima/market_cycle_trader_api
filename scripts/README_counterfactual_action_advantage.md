# Counterfactual Action Advantage v2.0.0

## Research question

Can the existing LightGBM opportunity model keep selecting the asset with the
highest predicted future growth while a separate supervised model learns whether
changing today's portfolio action adds value before the next decision?

This experiment keeps the current opportunity model intact and replaces the
multi-horizon fixed-action target from CAV v1 with a one-session incremental
advantage target.

## Why v2 exists

Counterfactual Action Value v1 was rejected on the 56-asset control because it
implicitly evaluated HOLD / ROTATE / CASH as if the chosen action had to remain
fixed across the Strategy's 5/10/20/40/60-session horizons.

The live decision process does not work that way: it decides again on the next
session.

CAA v2 therefore asks only:

```text
If I change today's action, how much more or less net log return do I control
until the next decision versus simply staying in my current state?
```

## Frozen components

The following remain unchanged:

- frozen asset universe;
- LightGBM opportunity target;
- LightGBM opportunity hyperparameters;
- walk-forward fold boundaries;
- execution costs;
- action/state features;
- ranking logic;
- exact backtest simulator;
- production control artifacts.

Only the decision-layer target changes.

## Reference action

The reference action always has advantage exactly zero:

- when invested: `HOLD current asset`;
- when in CASH: `STAY CASH`.

Alternatives are:

- `CASH`, when currently invested;
- `ROTATE/ENTER current LightGBM Top-1`, when Top-1 differs from the current state.

There is no manual Top-K shortlist.

## Label

For decision date `t` and next decision date `t+1`:

```text
advantage(action) =
    exact_net_log_return(current -> action, t -> t+1)
  - exact_net_log_return(current -> current, t -> t+1)
```

Because both terms begin with the same current position, the incumbent's
close(t) -> open(t+1) movement cancels. The label therefore isolates the part of
the return that today's action can control.

Switching costs remain inside the alternative return.

## Decision rule

At inference time:

```text
reference advantage = 0

predicted CASH advantage
predicted Top-1 advantage

action* = argmax(all feasible advantages)
```

No switch margin, CASH threshold, minimum-hold rule, bull/bear threshold or
waiting period is added.

## Statistical benefit versus v1

CAV v1 needed the largest target horizon (60 sessions), which left roughly
`calibration_sessions - 60` usable decision dates.

CAA v2 needs only the next session, leaving:

```text
calibration_sessions - 1
```

usable decision dates.

This increases the effective temporal sample while keeping the same fold
structure.

## First validation

Run only the 56-asset control first.

```bash
git fetch origin
git switch research/counterfactual-action-advantage-v2
git pull --ff-only origin research/counterfactual-action-advantage-v2

python scripts/research_counterfactual_action_advantage.py \
  --strategy-sequence 10 \
  --history-start 2016-01-01 \
  --snapshot-end 2026-09-04 \
  --analysis-start 2020-07-22 \
  --analysis-end 2026-09-03 \
  --series 56 \
  --workers 4
```

The baseline ZIP is automatically resolved from:

```text
research_output/asset_rotation_universe_scale_strategy_10_2020-07-22_to_2026-09-03.zip
```

If needed, pass it explicitly with `--baseline-output-dir`.

## Expected log markers

```text
LightGBM opportunity model + Counterfactual Action Advantage
fitting next-decision counterfactual action-advantage model
simulating out-of-sample counterfactual action-advantage portfolio
```

## Diagnostics

Per decision:

```text
caa_reference_asset
caa_reference_advantage
caa_selected_advantage
caa_cash_advantage
caa_best_candidate_advantage
```

Decision reasons:

```text
CAA_HOLD
CAA_ROTATE
CAA_ENTER
CAA_CASH
CAA_STAY_CASH
```

## Interpretation

Do not judge the experiment by rotation count alone.

Compare at minimum:

- final compounded capital;
- CAGR;
- Sharpe;
- maximum drawdown;
- fold-by-fold capital;
- rotation count;
- CASH sessions;
- Top-1 changes that were ignored;
- predicted CASH advantage;
- predicted Top-1 advantage.

The key falsification question is whether the model can avoid economically weak
rotations while preserving the rotations responsible for compound growth.

Do not run the 500-asset experiment unless the 56-asset result is informative.
