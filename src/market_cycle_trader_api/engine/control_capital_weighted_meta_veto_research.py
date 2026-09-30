"""v10.8.54 capital-weighted one-shot Meta-Veto research.

Runs the frozen v10.8.53 reference and one candidate that differs only in
training sample weights.  The candidate gives more influence to prior rollout
labels with larger absolute capital impact.  No threshold, feature, gate,
execution or one-shot semantics are changed.
"""
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
    MODEL_C,
    MIN_CALIBRATION_SAMPLES,
    MIN_CALIBRATION_BALANCED_ACCURACY,
    MIN_CALIBRATION_ROC_AUC,
    VETO_PROBABILITY_MAX,
    run_reduced_signature_meta_veto_pair,
)

EXPECTED_V1044_CAPITAL = 1078635.4115518222
EXPECTED_V1053_CAPITAL = 1891417.6670329159
MODE = "MCT_RESEARCH_CAPITAL_WEIGHTED_META_VETO_V1"


def _attach_maturity(
    dataset: pd.DataFrame,
    rollout_result: dict[str, Any],
) -> pd.DataFrame:
    rollout_dir = Path(str(rollout_result.get("report_directory") or ""))
    labels_path = rollout_dir / "paired_rollout_labels.csv"
    if not labels_path.is_file():
        raise FileNotFoundError("v10.8.48 paired rollout labels are missing.")
    labels = pd.read_csv(
        labels_path,
        usecols=[
            "decision_date",
            "incumbent_asset",
            "control_target_asset",
            "rollout_end_date",
        ],
    )
    labels["decision_date"] = pd.to_datetime(labels["decision_date"], utc=True)
    labels["rollout_end_date"] = pd.to_datetime(
        labels["rollout_end_date"], utc=True,
    )
    source = dataset.copy()
    source["decision_date"] = pd.to_datetime(source["decision_date"], utc=True)
    source = source.merge(
        labels.rename(columns={"control_target_asset": "candidate_asset"}),
        on=["decision_date", "incumbent_asset", "candidate_asset"],
        how="left",
        validate="one_to_one",
    )
    if source["rollout_end_date"].isna().any() or len(source) != 321:
        raise ValueError("v10.8.54 could not attach exact rollout maturity dates.")
    return source


def _write_run(
    *,
    output: Path,
    name: str,
    run: Any,
) -> None:
    curve = run.predictions.copy()
    curve["drawdown"] = (
        curve["strategy_equity"]
        / curve["strategy_equity"].cummax().clip(lower=1e-12)
        - 1.0
    )
    curve.to_csv(output / f"{name}_capital_curve.csv", float_format="%.17g")
    run.trades.to_csv(
        output / f"{name}_fills.csv", index=False, float_format="%.17g",
    )


def run_capital_weighted_meta_veto_research(
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
        raise ValueError("Expected server-generated v10.8.54 meta-veto job ID.")

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
        raise ValueError("v10.8.54 requires exact verified v10.8.48/49/50 chain.")

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
        raise ValueError("v10.8.54 requires exact 321-event source dataset.")
    dataset = _attach_maturity(dataset, rollout_result)

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
        raise ValueError("v10.8.54 requires the frozen 55-asset panel.")

    def emit(offset: int, width: int):
        def callback(pct: float, stage: str, completed: int) -> None:
            if progress:
                scaled = offset + int(width * max(0.0, min(100.0, pct)) / 100.0)
                progress(str(stage)[:150], max(1, min(99, scaled)), 100)
        return callback

    if progress:
        progress("replaying frozen v10.8.53 reference", 2, 100)
    ref_paths, ref_training, ref_decisions = run_reduced_signature_meta_veto_pair(
        bars,
        config,
        calculate_reference_fees,
        apply_slippage,
        source_dataset=dataset,
        capital_weighted=False,
        progress_callback=emit(2, 45),
    )

    if progress:
        progress("replaying v10.8.54 capital-weighted candidate", 50, 100)
    candidate_paths, candidate_training, candidate_decisions = (
        run_reduced_signature_meta_veto_pair(
            bars,
            config,
            calculate_reference_fees,
            apply_slippage,
            source_dataset=dataset,
            capital_weighted=True,
            progress_callback=emit(50, 45),
        )
    )

    ref_baseline = ref_paths["liquidity_baseline"]
    ref_meta = ref_paths["meta_veto"]
    candidate_baseline = candidate_paths["liquidity_baseline"]
    candidate_meta = candidate_paths["meta_veto"]

    for run in (ref_baseline, ref_meta, candidate_baseline, candidate_meta):
        if len(run.predictions) != 1554:
            raise ValueError("v10.8.54 replay must contain exactly 1,554 OOS sessions.")
        if (run.predictions["cash"] < -1e-8).any():
            raise ValueError("v10.8.54 generated negative CASH.")

    source_v1044 = float(
        (rollout_result.get("liquidity_baseline") or {}).get(
            "strategy_ending_capital"
        )
    )
    ref_baseline_cap = float(
        ref_baseline.metrics["strategy_ending_capital"]
    )
    candidate_baseline_cap = float(
        candidate_baseline.metrics["strategy_ending_capital"]
    )
    if not all(
        math.isclose(value, EXPECTED_V1044_CAPITAL, rel_tol=0, abs_tol=1e-6)
        for value in (
            source_v1044, ref_baseline_cap, candidate_baseline_cap,
        )
    ):
        raise ValueError(
            "v10.8.44 parity failed before v10.8.54 comparison: "
            f"expected {EXPECTED_V1044_CAPITAL:.12f}, source {source_v1044:.12f}, "
            f"reference {ref_baseline_cap:.12f}, candidate {candidate_baseline_cap:.12f}."
        )

    ref_capital = float(ref_meta.metrics["strategy_ending_capital"])
    if not math.isclose(
        ref_capital, EXPECTED_V1053_CAPITAL, rel_tol=0, abs_tol=1e-6,
    ):
        raise ValueError(
            "v10.8.53 reference parity failed before v10.8.54 comparison: "
            f"expected {EXPECTED_V1053_CAPITAL:.12f}, got {ref_capital:.12f}."
        )

    candidate_capital = float(
        candidate_meta.metrics["strategy_ending_capital"]
    )
    ref_decision_frame = pd.DataFrame(ref_decisions)
    candidate_decision_frame = pd.DataFrame(candidate_decisions)

    output = directory / "validation" / "v10.8.54" / meta_job_id
    if output.exists():
        raise FileExistsError("Never overwrite v10.8.54 research artifacts.")
    output.mkdir(parents=True)

    _write_run(output=output, name="liquidity_baseline", run=ref_baseline)
    _write_run(output=output, name="v1053_reference_meta_veto", run=ref_meta)
    _write_run(
        output=output,
        name="capital_weighted_meta_veto",
        run=candidate_meta,
    )
    pd.DataFrame(ref_training).to_csv(
        output / "v1053_reference_training_folds.csv",
        index=False,
        float_format="%.17g",
    )
    pd.DataFrame(candidate_training).to_csv(
        output / "capital_weighted_training_folds.csv",
        index=False,
        float_format="%.17g",
    )
    ref_decision_frame.to_csv(
        output / "v1053_reference_decisions.csv",
        index=False,
        float_format="%.17g",
    )
    candidate_decision_frame.to_csv(
        output / "capital_weighted_decisions.csv",
        index=False,
        float_format="%.17g",
    )

    aligned = pd.concat([
        ref_baseline.predictions["strategy_equity"].rename("liquidity_baseline"),
        ref_meta.predictions["strategy_equity"].rename("v1053_reference"),
        candidate_meta.predictions["strategy_equity"].rename("capital_weighted"),
    ], axis=1)
    aligned.to_csv(output / "aligned_capital_curves.csv", float_format="%.17g")

    original_folds = (
        (rollout_result.get("liquidity_baseline") or {}).get("walk_forward_folds")
        or []
    )
    ref_folds: list[dict[str, Any]] = []
    candidate_folds: list[dict[str, Any]] = []
    if original_folds:
        ref_folds = _validate_folds(
            ref_meta.predictions,
            original_folds,
            float(config.initial_capital),
        )
        candidate_folds = _validate_folds(
            candidate_meta.predictions,
            original_folds,
            float(config.initial_capital),
        )
        pd.DataFrame(ref_folds).assign(path="v1053_reference").to_csv(
            output / "v1053_reference_capital_folds.csv",
            index=False,
            float_format="%.17g",
        )
        pd.DataFrame(candidate_folds).assign(path="capital_weighted").to_csv(
            output / "capital_weighted_capital_folds.csv",
            index=False,
            float_format="%.17g",
        )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        aligned.index, aligned["liquidity_baseline"],
        label="v10.8.44 Liquidity-Aware",
    )
    ax.plot(
        aligned.index, aligned["v1053_reference"],
        label="v10.8.53 Reference",
    )
    ax.plot(
        aligned.index, aligned["capital_weighted"],
        label="v10.8.54 Capital-Weighted",
    )
    ax.set_yscale("log")
    ax.set_title("Capital-Weighted Meta-Veto · research only")
    ax.set_ylabel("USD, log scale")
    ax.grid(alpha=.2)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output / "capital_weighted_comparison.png", dpi=140)
    plt.close(fig)

    ref_veto_count = int(
        ref_decision_frame["veto_applied"].sum()
        if not ref_decision_frame.empty else 0
    )
    candidate_veto_count = int(
        candidate_decision_frame["veto_applied"].sum()
        if not candidate_decision_frame.empty else 0
    )

    report = {
        "schema_version": 1,
        "research_kind": "control_capital_weighted_meta_veto",
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
            "reference_reproduced": ref_baseline_cap,
            "candidate_reproduced": candidate_baseline_cap,
        },
        "v1053_reference_parity": {
            "status": "verified",
            "expected_ending_capital": EXPECTED_V1053_CAPITAL,
            "reproduced_ending_capital": ref_capital,
            "absolute_difference": abs(ref_capital - EXPECTED_V1053_CAPITAL),
        },
        "candidate_protocol": {
            "model": "balanced_l2_logistic_regression",
            "C": MODEL_C,
            "feature_count": 9,
            "sample_weight": (
                "max(abs(delta_capital_fraction),1e-6)"
                "/mean(max(abs(delta_capital_fraction),1e-6))"
            ),
            "calibration_weighted": False,
            "minimum_calibration_samples": MIN_CALIBRATION_SAMPLES,
            "minimum_calibration_balanced_accuracy":
                MIN_CALIBRATION_BALANCED_ACCURACY,
            "minimum_calibration_roc_auc": MIN_CALIBRATION_ROC_AUC,
            "veto_probability_max": VETO_PROBABILITY_MAX,
            "training_requires_rollout_end_before_test_start": True,
            "one_shot_veto": True,
            "oos_threshold_tuning": False,
        },
        "reference_training_folds": ref_training,
        "candidate_training_folds": candidate_training,
        "reference_veto_count": ref_veto_count,
        "candidate_veto_count": candidate_veto_count,
        "v1053_reference_portfolio": _metrics(ref_meta),
        "capital_weighted_portfolio": _metrics(candidate_meta),
        "candidate_vs_reference_delta_usd": candidate_capital - ref_capital,
        "candidate_vs_reference_delta_pct": (
            candidate_capital / ref_capital - 1.0 if ref_capital else None
        ),
        "candidate_improves_reference": bool(candidate_capital > ref_capital),
        "report_directory": str(output),
        "artifacts": [
            "summary.json",
            "liquidity_baseline_capital_curve.csv",
            "liquidity_baseline_fills.csv",
            "v1053_reference_meta_veto_capital_curve.csv",
            "v1053_reference_meta_veto_fills.csv",
            "capital_weighted_meta_veto_capital_curve.csv",
            "capital_weighted_meta_veto_fills.csv",
            "v1053_reference_training_folds.csv",
            "capital_weighted_training_folds.csv",
            "v1053_reference_decisions.csv",
            "capital_weighted_decisions.csv",
            "aligned_capital_curves.csv",
            "capital_weighted_comparison.png",
        ],
        "order_eligible": False,
        "order_submission": "never",
        "source_download": "never",
    }
    if ref_folds:
        report["artifacts"] += [
            "v1053_reference_capital_folds.csv",
            "capital_weighted_capital_folds.csv",
        ]
    (output / "summary.json").write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            default=str,
        ) + "\n",
        encoding="utf-8",
    )
    if progress:
        progress("completed", 100, 100)
    return report
