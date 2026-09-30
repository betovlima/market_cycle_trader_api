"""SHA-pinned Control-first meta-veto research report."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..tcc_v106_reference.config import CONFIG, build_control_config
from ..tcc_v106_reference.execution import calculate_reference_fees, apply_slippage
from .control_execution_sensitivity import _validate_folds
from .control_meta_veto import (
    MODE, TARGET, HORIZON, MAX_EPOCHS, PATIENCE,
    VETO_THRESHOLDS, MIN_VETO_PRECISION, MIN_VETO_VALIDATION_SAMPLES,
    run_meta_veto_pair,
)
from .control_snapshot_validation import read_verified_control_snapshot
from .operational_control_contract import prepare_operational_control_panel

Progress = Callable[[str, int, int], None]
METRIC_KEYS = (
    "initial_capital", "strategy_ending_capital", "strategy_return",
    "strategy_cagr", "strategy_sharpe", "strategy_maximum_drawdown",
    "risk_adjusted_compound_score", "market_exposure", "cash_weight_mean",
    "cash_days", "capital_rotations", "policy_target_changes",
    "simulated_buys", "simulated_sells", "blocked_sessions",
    "partial_sessions", "zero_volume_block_sessions",
    "unfilled_requested_shares", "modeled_price_cost_usd",
    "total_transaction_fees", "terminal_holdings_shares", "terminal_cash",
    "stale_mark_sessions", "session_count",
)


def _num(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _metrics(run: Any) -> dict[str, Any]:
    result = {key: _num(run.metrics.get(key)) for key in METRIC_KEYS}
    result["terminal_holdings_asset"] = str(run.metrics["terminal_holdings_asset"])
    return result


def run_control_meta_veto_research(
    *,
    source_job_id: str,
    validation_job_id: str,
    execution_job_id: str,
    liquidity_job_id: str,
    tcn_job_id: str,
    ranking_job_id: str,
    meta_job_id: str,
    expected_sha256: str,
    baseline41: dict[str, Any],
    baseline42: dict[str, Any],
    baseline44: dict[str, Any],
    baseline45: dict[str, Any],
    baseline46: dict[str, Any],
    progress: Progress | None = None,
    snapshot_root: Path | None = None,
) -> dict[str, Any]:
    if not re.fullmatch(r"control-veto-[a-f0-9]{16}", meta_job_id):
        raise ValueError("Expected server-generated meta-veto job ID.")
    bars, manifest, directory = read_verified_control_snapshot(
        source_job_id,
        snapshot_root=snapshot_root,
        expected_sha256=expected_sha256,
    )
    sources = (baseline41, baseline42, baseline44, baseline45, baseline46)
    if (
        len(bars) != 55
        or any(
            item.get("source_snapshot_sha256") != expected_sha256
            or item.get("source_job_id") != source_job_id
            or item.get("source_unchanged") is not True
            or item.get("order_submission") != "never"
            or (item.get("numeric_input_integrity") or {}).get("status") != "verified"
            for item in sources
        )
        or not (baseline41.get("original_shadow") or {}).get("reproduced")
        or baseline42.get("source_validation_job_id") != validation_job_id
        or baseline44.get("source_execution_job_id") != execution_job_id
        or baseline45.get("source_liquidity_job_id") != liquidity_job_id
        or baseline46.get("source_tcn_job_id") != tcn_job_id
        or baseline46.get("ranking_job_id") != ranking_job_id
        or baseline44.get("control_reference_parity") != "verified"
        or baseline45.get("research_kind") != "control_tcn_fixed_temporal_baseline"
        or baseline46.get("research_kind") != "control_deep_pairwise_rank_hybrid"
    ):
        raise ValueError("Meta-veto requires exact v10.8.41/42/44/45/46 source chain.")

    scientific_cap = _num((baseline41.get("oos") or {}).get("strategy_ending_capital"))
    execution_cap = _num((baseline42.get("feasible_oos") or {}).get("strategy_ending_capital"))
    liquidity_cap = _num((baseline44.get("liquidity_aware") or {}).get("strategy_ending_capital"))
    tcn_cap = _num((baseline45.get("tcn_portfolio") or {}).get("strategy_ending_capital"))
    rank_cap = _num((baseline46.get("deep_rank_portfolio") or {}).get("strategy_ending_capital"))
    if None in (scientific_cap, execution_cap, liquidity_cap, tcn_cap, rank_cap):
        raise ValueError("Benchmark hierarchy is incomplete.")

    session = str(manifest["completed_session"])
    inclusive = (
        pd.Timestamp(session, tz="UTC")
        + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    ).isoformat()
    config = build_control_config(CONFIG).model_copy(update={
        "analysis_end_date": inclusive,
        "end_date": session,
    })
    _, _, calendar_audit = prepare_operational_control_panel(
        bars, completed_session=session, config=config,
    )
    if int(baseline41.get("calendar_sessions", -1)) != calendar_audit.calendar_sessions:
        raise ValueError("Meta-veto canonical calendar differs from scientific Control.")

    def on_progress(pct: float, stage: str, completed: int) -> None:
        if progress is not None:
            progress(str(stage)[:160], max(0, min(100, int(pct))), 100)

    captured, training_reports, veto_audit = run_meta_veto_pair(
        bars, config, calculate_reference_fees, apply_slippage,
        progress_callback=on_progress,
    )
    base = captured["base_liquidity"]
    veto = captured["meta_veto"]
    if not math.isclose(
        float(base.metrics["strategy_ending_capital"]),
        float(liquidity_cap),
        rel_tol=0.0, abs_tol=1e-6,
    ):
        raise ValueError(
            "Paired base path did not reproduce v10.8.44; meta-veto result rejected."
        )
    if (
        len(base.predictions) != 1554
        or len(veto.predictions) != 1554
        or len(veto_audit) != 1554
        or not base.predictions.index.equals(veto.predictions.index)
    ):
        raise ValueError("Meta-veto paired OOS calendar/audit is incomplete.")

    original_folds = (baseline41.get("oos") or {}).get("walk_forward_folds") or []
    base_folds = _validate_folds(
        base.predictions, original_folds, float(config.initial_capital),
    )
    veto_folds = _validate_folds(
        veto.predictions, original_folds, float(config.initial_capital),
    )
    if not math.isclose(
        float(veto_folds[-1]["strategy_ending_capital"]),
        float(veto.metrics["strategy_ending_capital"]),
        rel_tol=0.0, abs_tol=1e-6,
    ):
        raise ValueError("Meta-veto fold reconciliation failed.")

    audit_frame = pd.DataFrame.from_records([
        {"decision_date": date, **row}
        for date, row in sorted(veto_audit.items())
    ])
    veto_count = int(audit_frame["veto_applied"].fillna(False).sum())
    eligible = audit_frame["probability_candidate_better"].notna()
    eligible_count = int(eligible.sum())
    per_fold = []
    for fold_id in (1, 2, 3):
        subset = audit_frame.loc[audit_frame["fold_id"] == fold_id]
        per_fold.append({
            "fold_id": fold_id,
            "decisions": len(subset),
            "eligible_rotation_evaluations": int(
                subset["probability_candidate_better"].notna().sum()
            ),
            "vetoes": int(subset["veto_applied"].fillna(False).sum()),
            "intervention_enabled": bool(
                next(x for x in training_reports if x["fold_id"] == fold_id)[
                    "intervention_enabled"
                ]
            ),
            "chosen_veto_threshold": next(
                x for x in training_reports if x["fold_id"] == fold_id
            )["chosen_veto_threshold"],
        })

    output = directory / "validation" / "v10.8.47" / meta_job_id
    if output.exists():
        raise FileExistsError("Never overwrite completed meta-veto artifacts.")
    output.mkdir(parents=True)

    base_curve = base.predictions.copy()
    veto_curve = veto.predictions.copy()
    base_curve["drawdown"] = (
        base_curve["strategy_equity"]
        / base_curve["strategy_equity"].cummax().clip(lower=1e-12) - 1
    )
    veto_curve["drawdown"] = (
        veto_curve["strategy_equity"]
        / veto_curve["strategy_equity"].cummax().clip(lower=1e-12) - 1
    )
    base_curve.to_csv(output / "base_liquidity_curve.csv", float_format="%.17g")
    veto_curve.to_csv(output / "meta_veto_curve.csv", float_format="%.17g")
    base.trades.to_csv(output / "base_liquidity_fills.csv", index=False, float_format="%.17g")
    veto.trades.to_csv(output / "meta_veto_fills.csv", index=False, float_format="%.17g")
    audit_frame.to_csv(output / "meta_veto_decisions.csv", index=False, float_format="%.17g")
    pd.DataFrame(training_reports).to_csv(output / "meta_veto_training_folds.csv", index=False)
    pd.DataFrame(veto_folds).to_csv(output / "meta_veto_capital_folds.csv", index=False, float_format="%.17g")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(base_curve.index, base_curve["strategy_equity"], label="v10.8.44 base")
    ax.plot(veto_curve.index, veto_curve["strategy_equity"], label="v10.8.47 meta-veto")
    ax.set_yscale("log")
    ax.set_title("Control-first meta-veto · paired constrained OOS")
    ax.set_ylabel("USD, log scale")
    ax.grid(alpha=.2)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output / "meta_veto_capital.png", dpi=140)
    plt.close(fig)

    base_metrics = _metrics(base)
    veto_metrics = _metrics(veto)
    report = {
        "schema_version": 1,
        "research_kind": "control_first_neural_meta_veto",
        "strategy_mode": MODE,
        "source_job_id": source_job_id,
        "source_validation_job_id": validation_job_id,
        "source_execution_job_id": execution_job_id,
        "source_liquidity_job_id": liquidity_job_id,
        "source_tcn_job_id": tcn_job_id,
        "source_ranking_job_id": ranking_job_id,
        "meta_job_id": meta_job_id,
        "source_snapshot_sha256": expected_sha256,
        "source_unchanged": True,
        "numeric_input_integrity": {"status": "verified", "assets": 55},
        "completed_session": session,
        "benchmark_hierarchy": {
            "scientific_control_usd": scientific_cap,
            "execution_constrained_control_usd": execution_cap,
            "liquidity_aware_v1044_usd": liquidity_cap,
            "failed_tiny_tcn_v1045_usd": tcn_cap,
            "failed_deep_rank_v1046_usd": rank_cap,
        },
        "architecture": {
            "role": "veto_only_never_select_asset",
            "temporal_encoder": "shared causal TinyTCN encoder",
            "meta_target": TARGET,
            "meta_label": "accept if candidate realized utility > incumbent realized utility",
            "label_horizon_sessions": HORIZON,
            "epochs_max": MAX_EPOCHS,
            "patience": PATIENCE,
            "fixed_threshold_candidates": list(VETO_THRESHOLDS),
            "minimum_bad_rotation_precision": MIN_VETO_PRECISION,
            "minimum_validation_veto_samples": MIN_VETO_VALIDATION_SAMPLES,
            "fallback": "exact v10.8.44 base when intervention right is not earned",
        },
        "training_fold_reports": training_reports,
        "base_reference_parity": "verified",
        "base_liquidity": {
            **base_metrics,
            "walk_forward_fold_count": 3,
            "walk_forward_folds": base_folds,
        },
        "meta_veto": {
            **veto_metrics,
            "walk_forward_fold_count": 3,
            "walk_forward_folds": veto_folds,
            "eligible_rotation_evaluations": eligible_count,
            "veto_count": veto_count,
            "fold_interventions": per_fold,
        },
        "capital_delta_vs_v1044_usd": (
            veto_metrics["strategy_ending_capital"] - liquidity_cap
        ),
        "capital_delta_vs_scientific_control_usd": (
            veto_metrics["strategy_ending_capital"] - scientific_cap
        ),
        "caveats": [
            "Historical OOS is exploratory because v10.8.45/v10.8.46 failures motivated this Control-first hypothesis.",
            "The neural model never chooses an asset; it can only veto an asset-to-asset rotation already proposed by the base liquidity-aware Control.",
            "A fold defaults exactly to the base policy unless pre-OOS chronological validation earns intervention rights at >=60% bad-rotation precision with >=50 veto examples.",
            "The first meta-label is relative forward scientific utility, not a direct paired capital rollout. A direct state-aware capital-advantage label is a separate future hypothesis.",
            "Same v10.8.42/v10.8.44 execution assumptions; daily bars do not prove broker fills.",
            "No source download, TCC edit, production strategy registration, Winner promotion or real order.",
        ],
        "report_directory": str(output),
        "artifacts": [
            "summary.json", "base_liquidity_curve.csv", "meta_veto_curve.csv",
            "base_liquidity_fills.csv", "meta_veto_fills.csv",
            "meta_veto_decisions.csv", "meta_veto_training_folds.csv",
            "meta_veto_capital_folds.csv", "meta_veto_capital.png",
        ],
        "source_download": "never",
        "order_submission": "never",
        "order_eligible": False,
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    if progress:
        progress("completed", 1, 1)
    return report
