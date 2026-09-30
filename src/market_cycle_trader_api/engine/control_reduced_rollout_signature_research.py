"""Orchestrator for v10.8.50 reduced rollout signature confirmation."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .control_snapshot_validation import read_verified_control_snapshot
from .control_reduced_rollout_signature import (
    REDUCED_FEATURES,
    coefficient_stability,
    evaluate_reduced_signature,
    write_reduced_signature_artifacts,
)


def run_reduced_rollout_signature_research(
    *,
    source_job_id: str,
    rollout_job_id: str,
    signature_job_id: str,
    reduced_job_id: str,
    expected_sha256: str,
    signature_result: dict[str, Any],
    progress: Callable[[str, int, int], None] | None = None,
    snapshot_root: Path | None = None,
) -> dict[str, Any]:
    if not re.fullmatch(r"control-reduced-[a-f0-9]{16}", reduced_job_id):
        raise ValueError("Expected server-generated v10.8.50 reduced-signature job ID.")
    if (
        signature_result.get("research_kind")
        != "control_rollout_decision_signature_diagnostic"
        or signature_result.get("signature_job_id") != signature_job_id
        or signature_result.get("source_rollout_job_id") != rollout_job_id
        or signature_result.get("source_job_id") != source_job_id
        or signature_result.get("source_snapshot_sha256") != expected_sha256
        or signature_result.get("source_unchanged") is not True
        or signature_result.get("order_submission") != "never"
        or (signature_result.get("numeric_input_integrity") or {}).get("status")
        != "verified"
        or not (signature_result.get("diagnostic_summary") or {}).get(
            "predictive_signal_detected"
        )
    ):
        raise ValueError("v10.8.50 requires the verified signal-positive v10.8.49 result.")

    _bars, _manifest, directory = read_verified_control_snapshot(
        source_job_id,
        snapshot_root=snapshot_root,
        expected_sha256=expected_sha256,
    )
    source_dir = Path(str(signature_result.get("report_directory") or ""))
    dataset_path = source_dir / "rollout_decision_dataset.csv"
    model_path = source_dir / "model_results.csv"
    if not dataset_path.is_file() or not model_path.is_file():
        raise FileNotFoundError("v10.8.49 diagnostic artifacts are missing.")

    dataset = pd.read_csv(dataset_path)
    prior_models = pd.read_csv(model_path)
    if len(dataset) != 321:
        raise ValueError("v10.8.49 source dataset must contain exactly 321 events.")
    missing = [feature for feature in REDUCED_FEATURES if feature not in dataset.columns]
    if missing:
        raise ValueError(f"Reduced signature features missing from source dataset: {missing}")

    if progress:
        progress("loading verified v10.8.49 diagnostic dataset", 20, 100)
    results, coefficients, reduced_summary = evaluate_reduced_signature(dataset)
    if progress:
        progress("checking reduced coefficient stability", 65, 100)
    stability = coefficient_stability(coefficients)

    prior_logistic = prior_models.loc[
        prior_models["model"] == "logistic_regression"
    ].copy()
    prior_reference = []
    for row in prior_logistic.to_dict("records"):
        prior_reference.append({
            "test_fold": int(row["test_fold"]),
            "balanced_accuracy": float(row["balanced_accuracy"]),
            "roc_auc": (
                float(row["roc_auc"])
                if pd.notna(row.get("roc_auc")) else None
            ),
            "brier": float(row["brier"]),
        })

    output = directory / "validation" / "v10.8.50" / reduced_job_id
    report = {
        "schema_version": 1,
        "research_kind": "control_reduced_rollout_signature_confirmation",
        "reduced_job_id": reduced_job_id,
        "source_signature_job_id": signature_job_id,
        "source_rollout_job_id": rollout_job_id,
        "source_job_id": source_job_id,
        "source_snapshot_sha256": expected_sha256,
        "source_unchanged": True,
        "numeric_input_integrity": {"status": "verified", "assets": 55},
        "source_rows": int(len(dataset)),
        "source_v1049_logistic_reference": prior_reference,
        "reduced_signature": reduced_summary,
        "protocol": {
            "policy_creation": False,
            "feature_selection_after_execution": False,
            "hyperparameter_search": False,
            "decision_threshold": 0.50,
            "fold_2_train_source": "fold_1_only",
            "fold_3_train_source": "fold_1_and_fold_2_only",
            "confirmation_requires_balanced_accuracy_and_auc_above_0_50_both_folds": True,
        },
        "report_directory": str(output),
        "order_eligible": False,
        "order_submission": "never",
        "source_download": "never",
    }
    artifacts = write_reduced_signature_artifacts(
        output=output,
        results=results,
        coefficients=coefficients,
        stability=stability,
        summary=report,
    )
    report["artifacts"] = artifacts
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2, default=str)
        + "\n",
        encoding="utf-8",
    )
    if progress:
        progress("completed", 100, 100)
    return report
