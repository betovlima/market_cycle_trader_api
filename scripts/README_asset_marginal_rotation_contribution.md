# Asset Marginal Rotation Contribution v1

Research-only experiment. Script version: `asset-marginal-rotation-contribution-v1.0.5`.

## Hypothesis

The Strategy baseline that produced the large compound result stays immutable. A new asset is admitted only when its OOS leadership displacement adds economic value relative to the leader the current universe would otherwise have selected.

The selector uses walk-forward OOS leadership predictions and does **not** run a full Strategy backtest for every candidate.

## Candidate source

Normal mode may reuse an explicitly supplied `--universe-file` or an existing default Leadership history-integrity artifact. If neither exists, Phase 1A evaluates every external symbol already cached for the Strategy market-data identity in the local MongoDB.

`--validation-only` reuses already frozen baseline/expanded snapshots and does not resolve the candidate universe again.

## Strict fresh run — v1.0.5

Use `--fresh-run` when no prior experiment file may influence the new execution.

Fresh mode:

- deletes only this experiment's existing output directory under `PROJECT_ROOT/research_output`;
- ignores every prior candidate-universe / Leadership artifact;
- discovers the external candidate pool directly from the local MongoDB cache;
- automatically disables Leadership resume;
- recomputes Phase 1A, Phase 1B, baseline validation and expanded validation from zero.

`--fresh-run` cannot be combined with `--validation-only` or `--universe-file`.

The safety guard refuses to recursively delete arbitrary directories outside the expected `research_output/asset_marginal_rotation_strategy_*` tree.

## Self-contained helpers

The experiment owns stable helpers and does not depend on obsolete versioned wrappers:

- `research_asset_marginal_rotation_leadership.py`
- `research_asset_marginal_rotation_validation.py`
- `research_asset_marginal_rotation_contribution.py`

The marginal candidate pool explicitly prefers `leadership_qualified`; intrinsic-timing-only qualification cannot admit a candidate into this experiment.

## Windows long-path safety — v1.0.3

The project checkout can already consume most of the legacy Windows path budget. Research artifact I/O is long-path safe through `research_windows_file_io.py` and uses short Phase 1B / validation subdirectory names.

## Mixed ISO timestamp normalization — v1.0.4

Leadership CSVs may legitimately contain UTC timestamps serialized in more than one ISO representation, for example `2020-07-20 04:00:00+00:00` and `2020-07-21T04:00:00+00:00`. The service parses the leadership timestamp column with explicit mixed-ISO handling, always normalizes to UTC, and raises a targeted error only if a timestamp is genuinely invalid after parsing.

## Selection

1. Run pre-validation rotation leadership.
2. Keep only external candidates that passed leadership qualification.
3. Start from the Strategy's original universe unchanged.
4. For each remaining candidate, detect OOS event starts where its predicted Utility newly exceeds the current-universe leader.
5. Measure candidate minus current-leader realized `forward_net_log_return` at those event starts.
6. Sum those deltas; log return is additive and therefore aligns with the compound objective.
7. Add the candidate with the largest positive marginal sum, rebuild the current universe, and repeat.
8. Stop when the best remaining marginal sum is non-positive. Zero is the economic indifference point; no manual acceptance margin is added.

Because the current universe is rebuilt after each addition, redundant candidates naturally lose marginal value once another asset already covers the same leadership windows.

## Independent validation

The runner reserves the last 252 XNYS sessions by default. Selection ends before that period. The strict independent validator then runs twice on the same untouched period:

- immutable Strategy baseline;
- marginally expanded universe.

The primary PASS condition is `expanded ending capital > baseline ending capital`.

## Run from zero

```bash
git fetch origin
git switch research/asset-marginal-rotation-contribution-v1
git pull --ff-only origin research/asset-marginal-rotation-contribution-v1

python scripts/research_asset_marginal_rotation_independent_then_validate.py \
  --strategy-sequence 10 \
  --history-start 2016-01-01 \
  --snapshot-end 2026-09-04 \
  --validation-sessions 252 \
  --workers 4 \
  --fresh-run
```

Main outputs include `marginal_rotation_selection_steps.csv`, `marginal_rotation_candidate_evaluations.csv`, `marginal_rotation_selected_events.csv`, both frozen snapshots, and `marginal_rotation_independent_comparison.json`.
