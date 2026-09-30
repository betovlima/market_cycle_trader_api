"""v10.8.55 expected-advantage regression Meta-Veto comparison."""
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
from .operational_control_contract import prepare_operational_control_panel
from .control_policy_rollout_research import _metrics
from .control_reduced_signature_meta_veto import run_reduced_signature_meta_veto_pair
from .control_expected_advantage_meta_veto import (
    RIDGE_ALPHA,
    MIN_CALIBRATION_SAMPLES,
    MIN_CALIBRATION_SIGN_BALANCED_ACCURACY,
    MIN_CALIBRATION_SPEARMAN,
    run_expected_advantage_meta_veto_pair,
)

EXPECTED_V1044_CAPITAL = 1078635.4115518222
EXPECTED_V1053_CAPITAL = 1891417.6670329159
MODE = "MCT_RESEARCH_EXPECTED_ADVANTAGE_REGRESSION_META_VETO_V1"


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
        raise ValueError("v10.8.55 could not attach exact rollout maturity dates.")
    return source


def _write_run(output: Path, name: str, run: Any) -> None:
    curve = run.predictions.copy()
    curve["drawdown"] = (
        curve["strategy_equity"]
        / curve["strategy_equity"].cummax().clip(lower=1e-12)
        - 1.0
    )
    curve.to_csv(output / f"{name}_capital_curve.csv", float_format="%.17g")
    run.trades.to_csv(
        output / f"{name}_fills.csv",
        index=False,
        float_format="%.17g",
    )


def run_expected_advantage_regression_research(
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
        raise ValueError("Expected server-generated v10.8.55 meta-veto job ID.")

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
        raise ValueError("v10.8.55 requires exact verified v10.8.48/49/50 chain.")

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
        raise ValueError("v10.8.55 requires exact 321-event source dataset.")
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
        bars,
        completed_session=cutoff,
        config=config,
    )
    if calendar_audit.available_assets != 55:
        raise ValueError("v10.8.55 requires the frozen 55-asset panel.")

    def scaled_callback(offset: int, width: int):
        def callback(pct: float, stage: str, completed: int) -> None:
            if progress:
                scaled = offset + int(width * max(0.0, min(100.0, pct)) / 100.0)
                progress(str(stage)[:150], max(1, min(99, scaled)), 100)
        return callback

    if progress:
        progress("replaying frozen v10.8.53 reference", 2, 100)
    reference_paths, reference_training, reference_decisions = (
        run_reduced_signature_meta_veto_pair(
            bars,
            config,
            calculate_reference_fees,
            apply_slippage,
            source_dataset=dataset,
            progress_callback=scaled_callback(2, 45),
        )
    )

    if progress:
        progress("running v10.8.55 expected-advantage regression", 50, 100)
    candidate_paths, candidate_training, candidate_decisions = (
        run_expected_advantage_meta_veto_pair(
            bars,
            config,
            calculate_reference_fees,
            apply_slippage,
            source_dataset=dataset,
            progress_callback=scaled_callback(50, 45),
        )
    )

    reference_baseline = reference_paths["liquidity_baseline"]
    reference_meta = reference_paths["meta_veto"]
    candidate_baseline = candidate_paths["liquidity_baseline"]
    candidate_meta = candidate_paths["meta_veto"]

    for run in (
        reference_baseline,
        reference_meta,
        candidate_baseline,
        candidate_meta,
    ):
        if len(run.predictions) != 1554:
            raise ValueError("v10.8.55 replay must contain exactly 1,554 OOS sessions.")
        if (run.predictions["cash"] < -1e-8).any():
            raise ValueError("v10.8.55 generated negative CASH.")

    source_v1044 = float(
        (rollout_result.get("liquidity_baseline") or {}).get(
            "strategy_ending_capital"
        )
    )
    reference_baseline_capital = float(
        reference_baseline.metrics["strategy_ending_capital"]
    )
    candidate_baseline_capital = float(
        candidate_baseline.metrics["strategy_ending_capital"]
    )
    if not all(
        math.isclose(value, EXPECTED_V1044_CAPITAL, rel_tol=0, abs_tol=1e-6)
        for value in (
            source_v1044,
            reference_baseline_capital,
            candidate_baseline_capital,
        )
    ):
        raise ValueError(
            "v10.8.44 parity failed before v10.8.55 comparison."
        )

    reference_capital = float(
        reference_meta.metrics["strategy_ending_capital"]
    )
    if not math.isclose(
        reference_capital,
        EXPECTED_V1053_CAPITAL,
        rel_tol=0,
        abs_tol=1e-6,
    ):
        raise ValueError(
            "v10.8.53 reference parity failed before v10.8.55 comparison: "
            f"expected {EXPECTED_V1053_CAPITAL:.12f}, got {reference_capital:.12f}."
        )

    candidate_capital = float(
        candidate_meta.metrics["strategy_ending_capital"]
    )
    reference_decision_frame = pd.DataFrame(reference_decisions)
    candidate_decision_frame = pd.DataFrame(candidate_decisions)

    output = directory / "validation" / "v10.8.55" / meta_job_id
    if output.exists():
        raise FileExistsError("Never overwrite v10.8.55 research artifacts.")
    output.mkdir(parents=True)

    _write_run(output, "liquidity_baseline", reference_baseline)
    _write_run(output, "v1053_reference_meta_veto", reference_meta)
    _write_run(output, "expected_advantage_meta_veto", candidate_meta)

    pd.DataFrame(reference_training).to_csv(
        output / "v1053_reference_training_folds.csv",
        index=False,
        float_format="%.17g",
    )
    pd.DataFrame(candidate_training).to_csv(
        output / "expected_advantage_training_folds.csv",
        index=False,
        float_format="%.17g",
    )
    reference_decision_frame.to_csv(
        output / "v1053_reference_decisions.csv",
        index=False,
        float_format="%.17g",
    )
    candidate_decision_frame.to_csv(
        output / "expected_advantage_decisions.csv",
        index=False,
        float_format="%.17g",
    )

    aligned = pd.concat([
        reference_baseline.predictions["strategy_equity"].rename(
            "liquidity_baseline"
        ),
        reference_meta.predictions["strategy_equity"].rename(
            "v1053_reference"
        ),
        candidate_meta.predictions["strategy_equity"].rename(
            "expected_advantage"
        ),
    ], axis=1)
    aligned.to_csv(
        output / "aligned_capital_curves.csv",
        float_format="%.17g",
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        aligned.index,
        aligned["liquidity_baseline"],
        label="v10.8.44 Liquidity-Aware",
    )
    ax.plot(
        aligned.index,
        aligned["v1053_reference"],
        label="v10.8.53 Reference",
    )
    ax.plot(
        aligned.index,
        aligned["expected_advantage"],
        label="v10.8.55 Expected Advantage Regression",
    )
    ax.set_yscale("log")
    ax.set_title("Expected Advantage Regression Meta-Veto · research only")
    ax.set_ylabel("USD, log scale")
    ax.grid(alpha=.2)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output / "expected_advantage_comparison.png", dpi=140)
    plt.close(fig)

    reference_veto_count = int(
        reference_decision_frame["veto_applied"].sum()
        if not reference_decision_frame.empty else 0
    )
    candidate_veto_count = int(
        candidate_decision_frame["veto_applied"].sum()
        if not candidate_decision_frame.empty else 0
    )

    report = {
        "schema_version": 1,
        "research_kind": "control_expected_advantage_regression_meta_veto",
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
            "reference_reproduced": reference_baseline_capital,
            "candidate_reproduced": candidate_baseline_capital,
        },
        "v1053_reference_parity": {
            "status": "verified",
            "expected_ending_capital": EXPECTED_V1053_CAPITAL,
            "reproduced_ending_capital": reference_capital,
            "absolute_difference": abs(
                reference_capital - EXPECTED_V1053_CAPITAL
            ),
        },
        "candidate_protocol": {
            "model": "ridge_regression",
            "alpha": RIDGE_ALPHA,
            "target": "delta_capital_fraction",
            "feature_count": 9,
            "minimum_calibration_samples": MIN_CALIBRATION_SAMPLES,
            "minimum_calibration_sign_balanced_accuracy":
                MIN_CALIBRATION_SIGN_BALANCED_ACCURACY,
            "minimum_calibration_spearman": MIN_CALIBRATION_SPEARMAN,
            "veto_rule": "predicted_delta_capital_fraction <= 0.0",
            "training_requires_rollout_end_before_test_start": True,
            "one_shot_veto": True,
            "threshold_tuning": False,
            "alpha_tuning": False,
        },
        "reference_training_folds": reference_training,
        "candidate_training_folds": candidate_training,
        "reference_veto_count": reference_veto_count,
        "candidate_veto_count": candidate_veto_count,
        "v1053_reference_portfolio": _metrics(reference_meta),
        "expected_advantage_portfolio": _metrics(candidate_meta),
        "candidate_vs_reference_delta_usd":
            candidate_capital - reference_capital,
        "candidate_vs_reference_delta_pct": (
            candidate_capital / reference_capital - 1.0
            if reference_capital else None
        ),
        "candidate_improves_reference":
            bool(candidate_capital > reference_capital),
        "report_directory": str(output),
        "artifacts": [
            "summary.json",
            "liquidity_baseline_capital_curve.csv",
            "liquidity_baseline_fills.csv",
            "v1053_reference_meta_veto_capital_curve.csv",
            "v1053_reference_meta_veto_fills.csv",
            "expected_advantage_meta_veto_capital_curve.csv",
            "expected_advantage_meta_veto_fills.csv",
            "v1053_reference_training_folds.csv",
            "expected_advantage_training_folds.csv",
            "v1053_reference_decisions.csv",
            "expected_advantage_decisions.csv",
            "aligned_capital_curves.csv",
            "expected_advantage_comparison.png",
        ],
        "order_eligible": False,
        "order_submission": "never",
        "source_download": "never",
    }
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
