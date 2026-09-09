# Asset Marginal Rotation Contribution v1

Research-only experiment. Script version: `asset-marginal-rotation-contribution-v1.0.2`.

## Hypothesis

The Strategy baseline that produced the large compound result stays immutable. A new asset is admitted only when its OOS leadership displacement adds economic value relative to the leader the current universe would otherwise have selected.

The selector uses existing walk-forward OOS leadership predictions and does **not** run a full Strategy backtest for every candidate.

## Candidate source

`--universe-file` is optional.

- If explicitly supplied, that frozen history-integrity file defines the external candidate pool.
- If omitted and the default prior Leadership history-integrity file exists, it is reused.
- If neither exists, Phase 1A uses the native Leadership behavior and evaluates every external symbol already cached for the Strategy market-data identity in the local MongoDB.

Therefore a previous `asset_rotation_leadership_strategy_<N>_<date>/asset_history_integrity.csv` run is not required just to start this experiment.

`--validation-only` reuses the already frozen baseline/expanded snapshots and does not resolve the candidate universe again.

## Self-contained research helpers

v1.0.2 removes dependencies on the obsolete versioned wrapper files `research_asset_rotation_leadership_v12.py`, `research_asset_rotation_leadership_v13.py`, `research_asset_rotation_independent_validation_v101.py`, and `research_asset_rotation_independent_validation_v103.py`.

The experiment now owns two stable helpers:

- `research_asset_marginal_rotation_leadership.py`: candidate failure isolation plus the stable Leadership engine;
- `research_asset_marginal_rotation_validation.py`: strict history-start/session validation, frozen-universe anchors, and Windows long-path-safe writes plus the stable independent validator.

The marginal candidate pool explicitly prefers `leadership_qualified`; intrinsic-timing-only qualification cannot admit a candidate into this experiment.

## Selection

1. Run the pre-validation rotation-leadership study.
2. Keep only external candidates that passed leadership qualification.
3. Start from the Strategy's original universe unchanged.
4. For each remaining candidate, detect OOS event starts where its predicted Utility newly exceeds the current-universe leader.
5. Measure candidate minus current-leader realized `forward_net_log_return` at those event starts.
6. Sum those deltas; log return is additive and therefore matches the compound objective better than an arithmetic-return score.
7. Add the candidate with the largest positive marginal sum, rebuild the current universe, and repeat.
8. Stop when the best remaining marginal sum is non-positive. Zero is the economic indifference point; no manual acceptance margin is added.

Because the current universe is rebuilt after each addition, redundant candidates naturally lose marginal value once another asset already covers the same leadership windows.

## Independent validation

`research_asset_marginal_rotation_independent_then_validate.py` reserves the last 252 XNYS sessions by default. Selection ends before that period. The strict independent validator then runs twice on the same untouched period:

- immutable Strategy baseline;
- marginally expanded universe.

The primary PASS condition is `expanded ending capital > baseline ending capital`.

## Run

```bash
git fetch origin
git switch research/asset-marginal-rotation-contribution-v1
git pull --ff-only origin research/asset-marginal-rotation-contribution-v1

python scripts/research_asset_marginal_rotation_independent_then_validate.py \
  --strategy-sequence 10 \
  --history-start 2016-01-01 \
  --snapshot-end 2026-09-04 \
  --validation-sessions 252 \
  --workers 4
```

Use `--selection-only` to stop after the frozen snapshots, or `--validation-only` to reuse a completed selection.

Main outputs include `marginal_rotation_selection_steps.csv`, `marginal_rotation_candidate_evaluations.csv`, `marginal_rotation_selected_events.csv`, both frozen snapshots, and `marginal_rotation_independent_comparison.json`.
