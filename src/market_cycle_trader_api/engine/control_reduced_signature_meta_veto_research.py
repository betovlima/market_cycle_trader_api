"""SHA-pinned v10.8.51 reduced-signature meta-veto capital research."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ..tcc_v106_reference.config import CONFIG, build_control_config
from ..tcc_v106_reference.execution import calculate_reference_fees, apply_slippage
from .control_snapshot_validation import read_verified_control_snapshot
from .control_execution_sensitivity import _validate_folds
from .operational_control_contract import prepare_operational_control_panel
from .control_policy_rollout_research import _metrics
from .control_reduced_signature_meta_veto import (
    MODE,
    MODEL_C,
    MIN_CALIBRATION_SAMPLES,
    MIN_CALIBRATION_BALANCED_ACCURACY,
    MIN_CALIBRATION_ROC_AUC,
    VETO_PROBABILITY_MAX,
    run_reduced_signature_meta_veto_pair,
)

EXPECTED_V1044_CAPITAL = 1078635.4115518222


def run_reduced_signature_meta_veto_research(
    *,
    source_job_id: str,
    rollout_job_id: str,
    signature_job_id: str,
    reduced_job_id: str,
    meta_job_id: str,
    expected_sha256: str,
    rollout_result: dict[str, Any],
    signature_result: dict[str, Any],
    reduced_result: dict[str, Any],
    progress: Callable[[str, int, int], None] | None = None,
    snapshot_root: Path | None = None,
) -> dict[str, Any]:
    if not re.fullmatch(r"control-meta-[a-f0-9]{16}", meta_job_id):
        raise ValueError("Expected server-generated v10.8.51 meta-veto job ID.")

    if (
        rollout_result.get("research_kind")
        != "control_policy_rollout_advantage_meta_veto"
        or rollout_result.get("rollout_job_id") != rollout_job_id
        or rollout_result.get("source_job_id") != source_job_id
        or rollout_result.get("source_snapshot_sha256") != expected_sha256
        or (rollout_result.get("v1044_parity") or {}).get("status") != "verified"
        or signature_result.get("research_kind")
        != "control_rollout_decision_signature_diagnostic"
        or signature_result.get("signature_job_id") != signature_job_id
        or signature_result.get("source_rollout_job_id") != rollout_job_id
        or signature_result.get("source_job_id") != source_job_id
        or signature_result.get("source_snapshot_sha256") != expected_sha256
        or reduced_result.get("research_kind")
        != "control_reduced_rollout_signature_confirmation"
        or reduced_result.get("reduced_job_id") != reduced_job_id
        or reduced_result.get("source_signature_job_id") != signature_job_id
        or reduced_result.get("source_rollout_job_id") != rollout_job_id
        or reduced_result.get("source_job_id") != source_job_id
        or reduced_result.get("source_snapshot_sha256") != expected_sha256
        or not (reduced_result.get("reduced_signature") or {}).get(
            "reduced_signature_confirmed"
        )
        or any(
            item.get("source_unchanged") is not True
            or item.get("order_submission") != "never"
            or (item.get("numeric_input_integrity") or {}).get("status")
            != "verified"
            for item in (rollout_result, signature_result, reduced_result)
        )
    ):
        raise ValueError("v10.8.51 requires exact verified v10.8.48/49/50 chain.")

    bars, manifest, directory = read_verified_control_snapshot(
        source_job_id,
        snapshot_root=snapshot_root,
        expected_sha256=expected_sha256,
    )
    source_dir = Path(str(signature_result.get("report_directory") or ""))
    dataset_path = source_dir / "rollout_decision_dataset.csv"
    if not dataset_path.is_file():
        raise FileNotFoundError("v10.8.49 rollout decision dataset is missing.")
    dataset = pd.read_csv(dataset_path)
    if len(dataset) != 321:
        raise ValueError("v10.8.51 requires exact 321-event source dataset.")

    cutoff = str(manifest["completed_session"])
    inclusive = (
        pd.Timestamp(cutoff, tz="UTC")
        + pd.Timedelta(days=1)
        - pd.Timedelta(seconds=1)
    ).isoformat()
    config = build_control_config(CONFIG).model_copy(update={
        "analysis_end_date": inclusive,
        "end_date": cutoff,
    })
    _, _, calendar_audit = prepare_operational_control_panel(
        bars, completed_session=cutoff, config=config,
    )
    if calendar_audit.available_assets != 55:
        raise ValueError("v10.8.51 requires the frozen 55-asset panel.")

    if progress:
        progress("training prior-fold reduced logistic gates", 5, 100)

    def on_progress(pct: float, stage: str, completed: int) -> None:
        if progress:
            progress(str(stage)[:150], max(5, min(95, int(pct))), 100)

    paths, training_folds, decisions = run_reduced_signature_meta_veto_pair(
        bars,
        config,
        calculate_reference_fees,
        apply_slippage,
        source_dataset=dataset,
        progress_callback=on_progress,
    )
    baseline = paths["liquidity_baseline"]
    meta = paths["meta_veto"]
    if len(baseline.predictions) != 1554 or len(meta.predictions) != 1554:
        raise ValueError("v10.8.51 replay must contain exactly 1,554 OOS sessions.")

    reproduced = float(baseline.metrics["strategy_ending_capital"])
    source_v1044 = float(
        (rollout_result.get("liquidity_baseline") or {}).get(
            "strategy_ending_capital"
        )
    )
    if (
        not math.isclose(source_v1044, EXPECTED_V1044_CAPITAL, rel_tol=0, abs_tol=1e-6)
        or not math.isclose(reproduced, EXPECTED_V1044_CAPITAL, rel_tol=0, abs_tol=1e-6)
    ):
        raise ValueError(
            "v10.8.44 parity failed before v10.8.51 capital comparison: "
            f"expected {EXPECTED_V1044_CAPITAL:.12f}, source {source_v1044:.12f}, "
            f"reproduced {reproduced:.12f}."
        )
    if (
        (baseline.predictions["cash"] < -1e-8).any()
        or (meta.predictions["cash"] < -1e-8).any()
    ):
        raise ValueError("v10.8.51 generated negative CASH.")

    original_folds = (
        (rollout_result.get("liquidity_baseline") or {}).get("walk_forward_folds")
        or []
    )
    # v10.8.48 metric summary may not carry folds; use source validation shape
    # only when available. Capital comparison itself remains exact.
    baseline_folds = []
    meta_folds = []
    if original_folds:
        baseline_folds = _validate_folds(
            baseline.predictions, original_folds, float(config.initial_capital)
        )
        meta_folds = _validate_folds(
            meta.predictions, original_folds, float(config.initial_capital)
        )

    decision_frame = pd.DataFrame(decisions)
    enabled_folds = int(sum(
        bool(item.get("model_enabled")) for item in training_folds
    ))
    veto_count = int(
        decision_frame["veto_applied"].sum()
        if not decision_frame.empty else 0
    )
    eligible = decision_frame.loc[
        decision_frame["probability_rotate_better"].notna()
    ] if not decision_frame.empty else decision_frame

    output = directory / "validation" / "v10.8.51" / meta_job_id
    if output.exists():
        raise FileExistsError("Never overwrite v10.8.51 research artifacts.")
    output.mkdir(parents=True)

    for name, run in (
        ("liquidity_baseline", baseline),
        ("reduced_signature_meta_veto", meta),
    ):
        curve = run.predictions.copy()
        curve["drawdown"] = (
            curve["strategy_equity"]
            / curve["strategy_equity"].cummax().clip(lower=1e-12)
            - 1.0
        )
        curve.to_csv(output / f"{name}_capital_curve.csv", float_format="%.17g")
        run.trades.to_csv(
            output / f"{name}_fills.csv", index=False, float_format="%.17g"
        )

    pd.DataFrame(training_folds).to_csv(
        output / "meta_training_folds.csv", index=False, float_format="%.17g"
    )
    decision_frame.to_csv(
        output / "meta_veto_decisions.csv", index=False, float_format="%.17g"
    )
    if baseline_folds:
        pd.DataFrame(baseline_folds).assign(path="liquidity_baseline").to_csv(
            output / "liquidity_baseline_folds.csv", index=False, float_format="%.17g"
        )
        pd.DataFrame(meta_folds).assign(path="reduced_signature_meta_veto").to_csv(
            output / "meta_veto_capital_folds.csv", index=False, float_format="%.17g"
        )

    aligned = pd.concat([
        baseline.predictions["strategy_equity"].rename("liquidity_baseline"),
        meta.predictions["strategy_equity"].rename("reduced_signature_meta_veto"),
    ], axis=1)
    aligned.to_csv(output / "aligned_capital_curves.csv", float_format="%.17g")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(aligned.index, aligned["liquidity_baseline"], label="v10.8.44 Liquidity-Aware")
    ax.plot(
        aligned.index,
        aligned["reduced_signature_meta_veto"],
        label="v10.8.51 Reduced Signature Meta-Veto",
    )
    ax.set_yscale("log")
    ax.set_title("Reduced Signature Meta-Veto · research only")
    ax.set_ylabel("USD, log scale")
    ax.grid(alpha=.2)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output / "paired_capital.png", dpi=140)
    plt.close(fig)

    meta_capital = float(meta.metrics["strategy_ending_capital"])
    report = {
        "schema_version": 1,
        "research_kind": "control_reduced_signature_meta_veto",
        "strategy_mode": MODE,
        "meta_job_id": meta_job_id,
        "source_reduced_job_id": reduced_job_id,
        "source_signature_job_id": signature_job_id,
        "source_rollout_job_id": rollout_job_id,
        "source_job_id": source_job_id,
        "source_snapshot_sha256": expected_sha256,
        "source_unchanged": True,
        "numeric_input_integrity": {"status": "verified", "assets": 55},
        "completed_session": cutoff,
        "v1044_parity": {
            "status": "verified",
            "expected_ending_capital": EXPECTED_V1044_CAPITAL,
            "reproduced_ending_capital": reproduced,
            "absolute_difference": abs(reproduced - EXPECTED_V1044_CAPITAL),
        },
        "meta_protocol": {
            "model": "balanced_l2_logistic_regression",
            "C": MODEL_C,
            "feature_count": 9,
            "fold_1_model": "disabled",
            "fold_2_training_source": "fold_1_only",
            "fold_3_training_source": "fold_1_and_fold_2_only",
            "chronological_train_fraction": 0.70,
            "minimum_calibration_samples": MIN_CALIBRATION_SAMPLES,
            "minimum_calibration_balanced_accuracy":
                MIN_CALIBRATION_BALANCED_ACCURACY,
            "minimum_calibration_roc_auc": MIN_CALIBRATION_ROC_AUC,
            "veto_probability_max": VETO_PROBABILITY_MAX,
            "veto_action": "hold incumbent for one decision",
            "cash_transitions_modified": False,
            "dynamic_features_use_actual_meta_state": True,
            "training_labels_always_from_baseline_v1049": True,
            "oos_tuning": False,
        },
        "training_folds": training_folds,
        "enabled_fold_count": enabled_folds,
        "eligible_model_decisions": int(len(eligible)),
        "veto_count": veto_count,
        "liquidity_baseline": _metrics(baseline),
        "reduced_signature_meta_veto_portfolio": _metrics(meta),
        "ending_capital_delta_usd": meta_capital - reproduced,
        "ending_capital_delta_pct": (
            meta_capital / reproduced - 1.0 if reproduced else None
        ),
        "report_directory": str(output),
        "artifacts": [
            "summary.json",
            "liquidity_baseline_capital_curve.csv",
            "liquidity_baseline_fills.csv",
            "reduced_signature_meta_veto_capital_curve.csv",
            "reduced_signature_meta_veto_fills.csv",
            "meta_training_folds.csv",
            "meta_veto_decisions.csv",
            "aligned_capital_curves.csv",
            "paired_capital.png",
        ],
        "order_eligible": False,
        "order_submission": "never",
        "source_download": "never",
    }
    if baseline_folds:
        report["artifacts"] += [
            "liquidity_baseline_folds.csv",
            "meta_veto_capital_folds.csv",
        ]
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2, default=str)
        + "\n",
        encoding="utf-8",
    )
    if progress:
        progress("completed", 100, 100)
    return report
