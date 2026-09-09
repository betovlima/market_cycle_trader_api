# Counterfactual Rotation Advantage v4.0.0

## Research question

Can the existing LightGBM opportunity ranker preserve more of the production
compound if the learned ROTATE-versus-HOLD decision stops using a one-session
target and instead learns **relative persistence** over the Strategy's shortest
already-configured target horizon?

This experiment changes only the rotation-advantage target. CASH remains excluded.

## Why v4 exists

v3 isolated the rotation problem successfully:

- market exposure stayed at 100%;
- CASH was not available after the initial entry;
- the model rotated 260 times versus 317 in the frozen baseline;
- however ending capital was only about US$646k versus US$23.52M.

The v3 diagnostics also showed that the policy sometimes remained in an incumbent
for long periods after it stopped being Top-1. HOLD decisions while another asset
was Top-1 reached streaks above 30 sessions. The v3 label looked only one session
ahead, so a noisy next day could dominate a decision whose economic value actually
depends on several sessions of relative performance.

## What remains frozen

- frozen 56-asset universe;
- LightGBM opportunity target and hyperparameters;
- walk-forward folds;
- state/action features;
- costs and exact execution;
- Top-1 ranking;
- CASH exclusion;
- no manual switch margin;
- no manual minimum-hold rule.

## The only conceptual change

v3 target:

```text
advantage =
next-session return(ROTATE Top-1)
-
next-session return(HOLD incumbent)
```

v4 target:

```text
advantage =
relative cumulative net log return of ROTATE Top-1
-
relative cumulative net log return of HOLD incumbent
over the Strategy's shortest frozen target horizon
```

The horizon is **not a new hand-written parameter**. It is obtained from:

```text
min(rotation_target_horizons)
```

## Important distinction

The horizon is a **training label horizon**, not a holding rule.

The live policy still runs every session:

```text
day t:   estimate persistent ROTATE-vs-HOLD advantage
day t+1: estimate again
day t+2: estimate again
...
```

There is no code that forces a position to remain open for N days.

## Counterfactual economics

If the incumbent is A and the current Top-1 is B:

```text
HOLD path:
A -> A -> A -> ... -> A

ROTATE path:
A -> B -> B -> ... -> B
```

Both paths are evaluated over the same label window.

The ROTATE path pays the switching cost only on its first transition. Because both
paths begin from the same incumbent, the uncontrollable close(t)->open(t+1)
movement of A is common to both alternatives and cancels in the relative target.

## Decision rule

```text
predicted advantage > 0  -> ROTATE Top-1
predicted advantage <= 0 -> HOLD incumbent
```

Zero is the mathematical point of economic indifference, not a manually tuned
threshold.

## Recommended test

Run only 56 assets first:

```bash
git fetch origin
git switch research/counterfactual-rotation-advantage-v4
git pull --ff-only origin research/counterfactual-rotation-advantage-v4

python scripts/research_counterfactual_rotation_advantage_v4.py \
  --strategy-sequence 10 \
  --history-start 2016-01-01 \
  --snapshot-end 2026-09-04 \
  --analysis-start 2020-07-22 \
  --analysis-end 2026-09-03 \
  --series 56 \
  --workers 4
```

If the frozen baseline archive is in the standard `research_output` location,
`--baseline-output-dir` is not required.

## Expected log markers

```text
LightGBM opportunity model + Counterfactual Rotation Advantage v4
fitting persistent counterfactual rotation-advantage model
simulating out-of-sample persistent rotation-advantage portfolio
```

## Output

```text
research_output/counterfactual_rotation_advantage_v4_strategy_10_2020-07-22_to_2026-09-03
```

Archive:

```text
counterfactual_rotation_advantage_v4_strategy_10_2020-07-22_to_2026-09-03.zip
```

## Diagnostics

Each decision records:

```text
cra4_reference_asset
cra4_reference_advantage
cra4_rotation_candidate_asset
cra4_rotation_advantage
cra4_positive_advantage
cra4_label_horizon_sessions
```

Decision reasons:

```text
CRA4_ENTER_TOP1
CRA4_HOLD_CURRENT_TOP1
CRA4_HOLD_NONPOSITIVE_PERSISTENT_ADVANTAGE
CRA4_ROTATE_POSITIVE_PERSISTENT_ADVANTAGE
```

## Evaluation

Do not optimize for fewer rotations by itself. Compare ending compounded capital,
fold-by-fold capital, CAGR, Sharpe, maximum drawdown, rotation count, geometric
trade return, win rate, how often a changed Top-1 is rejected, and duration of
incumbent-not-Top-1 streaks.

If v4 remains far below the frozen production baseline, the next investigation
should focus on **training-sample coverage / temporal cross-fitting**, not on adding
manual thresholds.
