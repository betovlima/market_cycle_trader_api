from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for _path in (SRC_ROOT, SCRIPT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import research_dense_counterfactual_candidate_advantage_base as base  # noqa: E402
import research_windows_file_io as file_io  # noqa: E402
from market_cycle_trader_api.services import dense_counterfactual_candidate_advantage as dcca_service  # noqa: E402
from market_cycle_trader_api.services.pooled_candidate_episode_advantage import (  # noqa: E402
    LABEL_AVAILABLE_COLUMN,
    OBSERVED_TARGET_COLUMN,
    TARGET_COLUMN,
    build_episode_samples,
    summarize_episode_decisions as strict_episode_summary,
)
from market_cycle_trader_api.services.pooled_candidate_marginal_advantage import (  # noqa: E402
    fit_pooled_candidate_marginal_model,
)

SCRIPT_VERSION = "dense-counterfactual-candidate-advantage-v1.1.0"
EXPERIMENT = "dense_forced_candidate_stateful_episode_advantage_censor_aware"
EPISODE_DEFINITION_VERSION = "dcca-1.1.0-censor-aware-forced-candidate"

_ORIGINAL_DENSE_BUILD = dcca_service.build_dense_counterfactual_episode_samples
_ORIGINAL_PCMA_BUILD = base.pcma._build_samples


def _decision_features_without_future_targets(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """Build admission-time features without reading next-session outcomes."""
    kwargs["include_future_targets"] = False
    return _ORIGINAL_PCMA_BUILD(*args, **kwargs)


def _censor_aware_dense_samples(*args: Any, **kwargs: Any):
    """Keep every legal forced episode start available for inference.

    The original DCCA v1 sampler generated the forced replay correctly but only
    returned completed episodes in its scoring frame. That makes inference depend
    on whether the future episode happened to finish before the evaluation boundary.
    Rebuild the scoring frame from all forced episodes with the PCEA v2.1 contract:
    completed targets are finite, right-censored targets remain NaN, and their
    observed prefix is retained only for audit/evaluation reporting.
    """
    _completed_only, episodes, diagnostics = _ORIGINAL_DENSE_BUILD(*args, **kwargs)
    daily_features = kwargs.get("daily_features")
    if daily_features is None:
        raise RuntimeError("DCCA v1.1 requires daily_features for censor-aware scoring.")

    samples = build_episode_samples(
        daily_features=daily_features,
        episodes=episodes,
    )
    diagnostics = dict(diagnostics)
    diagnostics["scoring_episode_rows"] = int(len(samples))
    if samples.empty:
        diagnostics["completed_training_episode_rows"] = 0
        diagnostics["right_censored_episode_rows"] = 0
    else:
        censored = samples["right_censored"].astype(str).str.lower().isin(
            ["true", "1"]
        )
        diagnostics["completed_training_episode_rows"] = int((~censored).sum())
        diagnostics["right_censored_episode_rows"] = int(censored.sum())
    diagnostics["inference_requires_episode_completion"] = False
    diagnostics["future_outcome_used_to_filter_scoring_rows"] = False
    diagnostics["episode_definition_version"] = EPISODE_DEFINITION_VERSION
    return samples, episodes, diagnostics


def _fit_completed_dense_labels(
    samples: pd.DataFrame,
    feature_names: Iterable[str],
):
    """Fit only finite completed dense labels.

    DCCA cross-fit trains on entire earlier folds. A completed forced episode has
    label_available_at inside the fold that produced it; right-censored episodes
    carry a missing target and are removed by the pooled finite-frame contract.
    This keeps prediction rows censor-aware without using open prefixes as labels.
    """
    if LABEL_AVAILABLE_COLUMN not in samples.columns:
        raise RuntimeError(
            "DCCA v1.1 samples are missing label availability metadata; recompute from scratch."
        )
    return fit_pooled_candidate_marginal_model(
        samples,
        feature_names,
        target_column=TARGET_COLUMN,
    )


def _numeric_summary_for_base_runner(decisions: pd.DataFrame) -> dict[str, Any]:
    """Preserve strict censoring semantics while keeping the v1 base runner numeric.

    The inherited v1 runner formats and sums `realized_marginal_log_sum` as a float.
    PCEA v2.1 correctly returns None whenever a chosen episode is still open. For
    internal compatibility only, feed the completed-only subtotal to that legacy
    arithmetic and retain markers so `_repair_result_contract` restores the strict
    public result afterwards.
    """
    summary = dict(strict_episode_summary(decisions))
    strict_total = summary.get("realized_marginal_log_sum")
    pending = strict_total is None
    summary["_strict_realized_marginal_log_sum"] = strict_total
    summary["_has_pending_override"] = bool(pending)
    if pending:
        summary["realized_marginal_log_sum"] = float(
            summary.get("completed_override_marginal_log_sum") or 0.0
        )
    return summary


def _result_directory(args: Any) -> Path:
    if args.output_dir:
        return Path(args.output_dir).resolve()
    history_start = base.common._normalize_date(args.history_start)
    _, validation_start_text, validation_end_text = base.independent._split(
        history_start.date().isoformat(),
        args.snapshot_end,
        int(args.validation_sessions),
    )
    validation_start = base.common._normalize_date(validation_start_text)
    validation_end = base.common._normalize_date(validation_end_text)
    return (
        PROJECT_ROOT
        / "research_output"
        / (
            f"dense_counterfactual_candidate_advantage_strategy_{args.strategy_sequence}_"
            f"{validation_start.date().isoformat()}_to_{validation_end.date().isoformat()}"
        )
    ).resolve()


def _strictify_summary(summary: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    value = dict(summary)
    pending = bool(value.pop("_has_pending_override", False))
    strict_total = value.pop("_strict_realized_marginal_log_sum", None)
    if pending:
        value["realized_marginal_log_sum"] = None
    elif strict_total is not None:
        value["realized_marginal_log_sum"] = float(strict_total)
    return value, pending


def _repair_result_contract(output_dir: Path) -> None:
    result_path = output_dir / "dcca_result.json"
    if not file_io.exists(result_path):
        raise RuntimeError(f"DCCA result was not produced: {result_path}")
    result = file_io.read_json(result_path)

    repaired_stages: list[dict[str, Any]] = []
    stage_pending = False
    completed_crossfit_total = 0.0
    observed_crossfit_total = 0.0
    observed_crossfit_complete = True
    for raw_stage in result.get("prevalidation_crossfit_stages", []):
        stage, pending = _strictify_summary(raw_stage)
        stage_pending = stage_pending or pending
        completed_crossfit_total += float(
            stage.get("completed_override_marginal_log_sum") or 0.0
        )
        observed = stage.get("observed_marginal_log_sum")
        if observed is None:
            observed_crossfit_complete = False
        else:
            observed_crossfit_total += float(observed)
        repaired_stages.append(stage)

    validation_summary, validation_pending = _strictify_summary(
        result.get("validation_summary", {})
    )
    validation_completed = float(
        validation_summary.get("completed_override_marginal_log_sum") or 0.0
    )
    validation_observed = validation_summary.get("observed_marginal_log_sum")

    result["schema_version"] = 2
    result["script_version"] = SCRIPT_VERSION
    result["experiment"] = EXPERIMENT
    result["base_research"] = "PCEA v2.1.1 censor-aware episode contract"
    result["episode_definition_version"] = EPISODE_DEFINITION_VERSION
    result["sampling_change_from_pcea_v2"] = (
        "dense forced candidate-date interventions from the exact protected baseline prefix; "
        "raw candidate score is not required to win before a label is generated"
    )
    result["right_censored_episodes_used_for_training"] = False
    result["right_censored_episodes_excluded_from_evaluation"] = False
    result["inference_requires_future_labels"] = False
    result["decision_time_features_use_next_session_targets"] = False
    result["prevalidation_crossfit_stages"] = repaired_stages
    result["prevalidation_crossfit_completed_override_marginal_log_sum"] = (
        completed_crossfit_total
    )
    result["prevalidation_crossfit_observed_marginal_log_sum"] = (
        observed_crossfit_total if observed_crossfit_complete else None
    )
    result["prevalidation_crossfit_realized_marginal_log_sum"] = (
        None if stage_pending else completed_crossfit_total
    )
    result["prevalidation_crossfit_incremental_factor"] = (
        None if stage_pending else float(math.expm1(completed_crossfit_total))
    )
    result["validation_summary"] = validation_summary
    result["validation_completed_override_marginal_log_sum"] = validation_completed
    result["validation_observed_marginal_log_sum"] = validation_observed
    result["validation_incremental_factor"] = (
        None if validation_pending else float(math.expm1(validation_completed))
    )
    result["validation_signal_positive"] = (
        None if validation_pending else bool(validation_completed > 0.0)
    )
    result["episode_factors_are_portfolio_capital_returns"] = False
    file_io.write_json(result_path, result)

    crossfit_path = output_dir / "dcca_crossfit_summary.json"
    if file_io.exists(crossfit_path):
        file_io.write_json(crossfit_path, repaired_stages)

    manifest_path = output_dir / "experiment_manifest.json"
    if file_io.exists(manifest_path):
        manifest = file_io.read_json(manifest_path)
        manifest["schema_version"] = 2
        manifest["script_version"] = SCRIPT_VERSION
        manifest["experiment"] = EXPERIMENT
        manifest["base_research"] = "PCEA v2.1.1"
        manifest["episode_definition_version"] = EPISODE_DEFINITION_VERSION
        manifest["prior_research_artifacts_read"] = False
        manifest["right_censored_episode_exclusion_from_training"] = True
        manifest["right_censored_episode_exclusion_from_evaluation"] = False
        manifest["inference_requires_future_labels"] = False
        manifest["decision_time_features_use_next_session_targets"] = False
        file_io.write_json(manifest_path, manifest)


def _install_v11_contract() -> None:
    dcca_service.DCCA_EPISODE_DEFINITION_VERSION = EPISODE_DEFINITION_VERSION
    base.DCCA_EPISODE_DEFINITION_VERSION = EPISODE_DEFINITION_VERSION
    base.SCRIPT_VERSION = SCRIPT_VERSION
    base.EXPERIMENT = EXPERIMENT
    base.pcma._build_samples = _decision_features_without_future_targets
    base.build_dense_counterfactual_episode_samples = _censor_aware_dense_samples
    base.fit_episode_model = _fit_completed_dense_labels
    base.summarize_episode_decisions = _numeric_summary_for_base_runner


def main() -> int:
    args = base._parser().parse_args()
    output_dir = _result_directory(args)
    _install_v11_contract()
    code = int(base.main())
    if code != 0:
        return code
    _repair_result_contract(output_dir)
    print(
        "DCCA v1.1 audit contract applied: dense inference kept right-censored starts, "
        "training used completed labels only, and result aggregates preserve pending outcomes.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
