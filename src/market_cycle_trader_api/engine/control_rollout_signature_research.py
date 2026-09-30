"""Orchestrator for v10.8.49 rollout decision signature diagnostics."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ..tcc_v106_reference.config import CONFIG, build_control_config
from .control_snapshot_validation import read_verified_control_snapshot
from .control_rollout_decision_signature import (
    build_rollout_decision_dataset,
    feature_diagnostics,
    fold_feature_stability,
    temporal_model_evaluation,
    summarize_signature,
    write_signature_artifacts,
)


def run_rollout_decision_signature_research(
    *,
    source_job_id: str,
    rollout_job_id: str,
    signature_job_id: str,
    expected_sha256: str,
    rollout_result: dict[str, Any],
    progress: Callable[[str, int, int], None] | None = None,
    snapshot_root: Path | None = None,
) -> dict[str, Any]:
    if not re.fullmatch(r"control-signature-[a-f0-9]{16}", signature_job_id):
        raise ValueError("Expected server-generated v10.8.49 signature job ID.")
    if (
        rollout_result.get("research_kind")
        != "control_policy_rollout_advantage_meta_veto"
        or rollout_result.get("rollout_job_id") != rollout_job_id
        or rollout_result.get("source_job_id") != source_job_id
        or rollout_result.get("source_snapshot_sha256") != expected_sha256
        or rollout_result.get("source_unchanged") is not True
        or rollout_result.get("order_submission") != "never"
        or (rollout_result.get("numeric_input_integrity") or {}).get("status")
        != "verified"
        or (rollout_result.get("v1044_parity") or {}).get("status")
        != "verified"
    ):
        raise ValueError("v10.8.49 requires a verified completed v10.8.48 result.")

    bars, manifest, directory = read_verified_control_snapshot(
        source_job_id,
        snapshot_root=snapshot_root,
        expected_sha256=expected_sha256,
    )
    source_dir = Path(str(rollout_result.get("report_directory") or ""))
    labels_path = source_dir / "paired_rollout_labels.csv"
    curve_path = source_dir / "liquidity_baseline_capital_curve.csv"
    if not labels_path.is_file() or not curve_path.is_file():
        raise FileNotFoundError("v10.8.48 rollout artifacts are missing.")

    labels = pd.read_csv(labels_path)
    curve = pd.read_csv(curve_path, index_col=0, parse_dates=True)
    if len(labels) != int(rollout_result.get("paired_rollout_count") or -1):
        raise ValueError("v10.8.48 rollout label count does not match summary.")

    cutoff = str(manifest["completed_session"])
    inclusive = (
        pd.Timestamp(cutoff, tz="UTC") + pd.Timedelta(days=1)
        - pd.Timedelta(seconds=1)
    ).isoformat()
    config = build_control_config(CONFIG).model_copy(update={
        "analysis_end_date": inclusive,
        "end_date": cutoff,
    })

    if progress:
        progress("building causal decision dataset", 20, 100)
    dataset = build_rollout_decision_dataset(
        bars=bars,
        completed_session=cutoff,
        labels=labels,
        baseline_curve=curve,
        config=config,
    )
    if progress:
        progress("feature diagnostics", 45, 100)
    correlations, classes, quantiles = feature_diagnostics(dataset)
    stability = fold_feature_stability(dataset)
    if progress:
        progress("chronological fixed-model evaluation", 70, 100)
    model_results, importance = temporal_model_evaluation(dataset)
    summary_core = summarize_signature(
        dataset, correlations, stability, model_results,
    )

    output = directory / "validation" / "v10.8.49" / signature_job_id
    summary = {
        "schema_version": 1,
        "research_kind": "control_rollout_decision_signature_diagnostic",
        "signature_job_id": signature_job_id,
        "source_rollout_job_id": rollout_job_id,
        "source_job_id": source_job_id,
        "source_snapshot_sha256": expected_sha256,
        "source_unchanged": True,
        "numeric_input_integrity": {"status": "verified", "assets": 55},
        "completed_session": cutoff,
        "source_rollout_rows": int(len(labels)),
        "source_rollout_positive_rate": float(labels["rotate_better"].astype(bool).mean()),
        "protocol": {
            "policy_creation": False,
            "hyperparameter_search": False,
            "fold_2_train_source": "fold_1_only",
            "fold_3_train_source": "fold_1_and_fold_2_only",
            "models": [
                "logistic_regression",
                "decision_tree",
                "random_forest",
                "lightgbm_small",
            ],
            "decision_threshold": 0.5,
            "descriptive_correlations_are_not_predictive_evidence": True,
        },
        "diagnostic_summary": summary_core,
        "report_directory": str(output),
        "order_eligible": False,
        "order_submission": "never",
        "source_download": "never",
    }
    artifacts = write_signature_artifacts(
        output=output,
        dataset=dataset,
        correlations=correlations,
        class_comparison=classes,
        quantiles=quantiles,
        stability=stability,
        model_results=model_results,
        feature_importance=importance,
        summary=summary,
    )
    summary["artifacts"] = artifacts
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2, default=str)
        + "\n",
        encoding="utf-8",
    )
    if progress:
        progress("completed", 100, 100)
    return summary
