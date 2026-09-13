# Contextual Marginal Signature — living research log

Branch: `research/contextual-marginal-signature-v1`

Current research runtime: `contextual-marginal-signature-v1.0.17.4`

This is the **single living document** for the Contextual Marginal Signature research line. Do not create a new README for each research version. Git history preserves the old documents and implementations when historical reconstruction is needed.

## Current command

Windows CMD:

```cmd
python scripts\research_contextual_marginal_signature.py ^
  --strategy-sequence 10 ^
  --env-file .env
```

The public entrypoint, export folder and ZIP name are stable:

- `scripts/research_contextual_marginal_signature.py`
- `research_output/contextual_marginal_signature/`
- `research_output/contextual_marginal_signature.zip`

MongoDB local is the persistent source of truth. Filesystem output is export-only and must never be required as input by a later analysis.

## Research question

The research evolved from asking whether an asset has a fixed marginal signature to asking a more causal decision question:

> If we choose candidate `a` now, is that intervention better than allowing the normal policy to decide by itself?

The current target is paired action advantage:

`Y(t,a,H) = log(W_forced_candidate_then_policy / W_normal_policy)`

Both arms contain the candidate. The forced arm changes only the first causal action, then returns to the same policy from the resulting simulator state.

The future decision problem includes abstention:

`max(0, Y_candidate_1, ..., Y_candidate_k)`

so `NORMAL POLICY` is a valid action when all candidate interventions are predicted to be harmful.

## Current campaign

The current temporal expansion uses:

- 23 executable XNYS temporal states from 2020 through 2026;
- 2 universes: `Original25` and `Original24_MinusADM`;
- 7 candidates: `XSD`, `MKSI`, `GKOS`, `CLMT`, `CORT`, `APD`, `CCK`;
- 40 decision points / 39 execution transitions per paired rollout;
- 322 paired action-advantage observations;
- local MongoDB market bars only;
- fold-aware frozen switch-margin calibration;
- automatic Mongo checkpoint/resume.

The model-readiness gate requires temporal diversity, both positive and negative targets, and completed contexts where both intervention and abstention are optimal.

## Important research evolution

### v1.0.0–v1.0.5 — first contextual signature attempts

The first experiments used exact marginal capital as the target and small backward-looking feature sets. Linear and shallow-tree screens showed weak and unstable out-of-time ranking. This established that a fixed standalone asset signature was not enough.

### v1.0.6–v1.0.9 — context and universe effects

Leave-one-out and reduced-universe experiments showed that apparent candidate contribution could change when the surrounding universe changed. ADM and ADI became useful diagnostic cases. A major correction was introduced:

`m = log(candidate_capital / baseline_capital)`

so changes in baseline capital could no longer masquerade as candidate-specific effects.

### v1.0.11–v1.0.14.4 — path attribution and frozen policy calibration

Full replay traces exposed an important mechanism: candidate presence could change switch-margin calibration even when the candidate was never traded. The research separated direct participation from policy recalibration and then froze the baseline-selected switch margin fold-by-fold.

The direct-effect prevalence campaign produced 27 non-zero direct effects out of 84 observations and confirmed that non-zero direct effects coincided with candidate participation.

### v1.0.13–v1.0.15 — path representation

A level-2 Log-Signature representation confirmed that candidate × universe path interactions can be represented without simple cancellation. However, the first Ridge benchmark using all LogSig terms failed the scientific screen: pooled correlation improved while within-context ranking collapsed and validation error expanded sharply.

Conclusion: representational capacity exists, but predictability had not yet been demonstrated.

### v1.0.16 — paired action advantage

The target was changed to the cleaner counterfactual decision advantage:

`log(W_force_candidate_once_then_policy / W_policy)`

The target became dense: 82/84 observations were non-zero. Date/context explained far more variation than candidate identity. This was strong evidence that the problem is temporal and contextual rather than a fixed ranking of assets.

### v1.0.17–v1.0.17.3 — temporal expansion and resilient execution

The campaign expanded from 6 to 23 independent temporal states. MongoDB became the persistent source of truth. The runner gained:

- preflight validation of executable walk-forward windows;
- NaT-safe BSON serialization;
- idempotent trace writes;
- automatic resume from Mongo checkpoints;
- informative partial-analysis logs;
- pair-scoped RAM reuse of prepared execution context and fitted LightGBM models.

### v1.0.17.4 — guarded RAM acceleration

This version extends the optimization without changing the scientific protocol.

RAM now reuses only invariant computation:

- market-derived rotation feature frames across the process;
- prepared execution context inside a candidate pair;
- fitted LightGBM models inside the policy/forced pair;
- raw model utility predictions inside the same policy/forced pair.

The following are deliberately **never cached**:

- portfolio position;
- holding days;
- trades;
- simulator state;
- decision diagnostics;
- ending capital/results.

Before trusting the optimization, the first eligible policy replay is executed once without cache and once with RAM acceleration. The long campaign continues only if ending capital, executable sessions, prediction trace and trade trace are equivalent at tolerance `1e-12`.

Expected console evidence:

```text
[cache-validation] ...
[cache-validation] PASS ...
[cache] pair reuse | ... feature frames=... | utility predictions=...
[partial] ...
[ranking] ...
```

This is a technical/runtime version. It does not change the target, temporal states, candidates, universes, frozen-margin protocol or readiness gate.

## Research finish line

The program now has a fixed three-stage finish line:

1. finish the 23-state paired counterfactual dataset;
2. run one integrated Research Tournament over the frozen Mongo dataset;
3. freeze the winner and perform final chronological confirmation.

The Research Tournament should compare, under identical chronological splits and decision metrics:

- regularized linear models;
- LightGBM;
- MLP neural network;
- compact temporal neural models;
- representations with and without path/Log-Signature information.

Primary judgement must include within-context ranking, economic Top-1 utility, abstention quality, robustness across periods and cost sensitivity. Failure to find robust predictability is also a valid terminal scientific result.

## Source-code organization policy

Do not create new user-facing scripts such as `research_contextual_marginal_signature_vXYZ.py` for future runtime patches. Continue evolving the stable entrypoint and non-versioned helpers.

Current non-versioned runtime helpers are:

- `research_contextual_marginal_signature.py` — stable entrypoint;
- `research_contextual_signature_runtime.py` — RAM cache, runtime observability and equivalence guard;
- `research_contextual_signature_storage.py` — Mongo persistence, retry, resume/export helpers;
- `research_contextual_signature_analysis.py` — partial/final analysis and readiness.

Some older versioned Python modules are still imported by the current scientific protocol chain. They must not be deleted blindly until that protocol is flattened into non-versioned modules and deterministic equivalence has been demonstrated. Old versioned README files, however, are documentation-only and are obsolete after this consolidation.

## Safety rules for future runtime changes

A runtime optimization is accepted only when it preserves the paired scientific result. Long-running campaigns must remain resumable from MongoDB. A technical failure may stop the process, but must not destroy completed observations. New logging should continue exposing enough information for partial scientific analysis while the campaign is still running.
