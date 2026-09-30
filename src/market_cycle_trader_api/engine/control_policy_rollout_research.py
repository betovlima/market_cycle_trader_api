"""SHA-pinned v10.8.48 Control policy-rollout advantage research."""
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
from .control_policy_rollout_advantage import (
    MODE,
    ROLLOUT_HORIZON_SESSIONS,
    run_policy_rollout_meta_veto_pair,
)

PORTFOLIO_KEYS = (
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


def _number(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _metrics(run: Any) -> dict[str, Any]:
    result = {key: _number(run.metrics.get(key)) for key in PORTFOLIO_KEYS}
    result["terminal_holdings_asset"] = str(
        run.metrics.get("terminal_holdings_asset")
    )
    return result


def run_policy_rollout_advantage_research(
    *,
    source_job_id: str,
    validation_job_id: str,
    execution_job_id: str,
    liquidity_job_id: str,
    tcn_job_id: str,
    ranking_job_id: str,
    advantage_job_id: str,
    rollout_job_id: str,
    expected_sha256: str,
    baseline41: dict[str, Any],
    baseline42: dict[str, Any],
    baseline44: dict[str, Any],
    baseline45: dict[str, Any],
    baseline46: dict[str, Any],
    baseline47: dict[str, Any],
    progress: Callable[[str, int, int], None] | None = None,
    snapshot_root: Path | None = None,
) -> dict[str, Any]:
    if not re.fullmatch(r"control-rollout-[a-f0-9]{16}", rollout_job_id):
        raise ValueError("Expected server-generated v10.8.48 rollout job ID.")

    bars, manifest, directory = read_verified_control_snapshot(
        source_job_id,
        snapshot_root=snapshot_root,
        expected_sha256=expected_sha256,
    )
    sources = (
        baseline41, baseline42, baseline44,
        baseline45, baseline46, baseline47,
    )
    if (
        len(bars) != 55
        or any(
            item.get("source_snapshot_sha256") != expected_sha256
            or item.get("source_job_id") != source_job_id
            or item.get("source_unchanged") is not True
            or item.get("order_submission") != "never"
            or (item.get("numeric_input_integrity") or {}).get("status")
            != "verified"
            for item in sources
        )
        or not (baseline41.get("original_shadow") or {}).get("reproduced")
        or baseline42.get("source_validation_job_id") != validation_job_id
        or baseline44.get("source_validation_job_id") != validation_job_id
        or baseline44.get("source_execution_job_id") != execution_job_id
        or baseline44.get("control_reference_parity") != "verified"
        or baseline45.get("source_liquidity_job_id") != liquidity_job_id
        or baseline46.get("source_liquidity_job_id") != liquidity_job_id
        or baseline46.get("source_tcn_job_id") != tcn_job_id
        or baseline47.get("source_liquidity_job_id") != liquidity_job_id
        or baseline47.get("source_tcn_job_id") != tcn_job_id
        or baseline47.get("source_ranking_job_id") != ranking_job_id
        or baseline47.get("research_kind")
        != "control_counterfactual_advantage_meta_veto"
    ):
        raise ValueError(
            "v10.8.48 requires exact SHA-matched v10.8.41/42/44/45/46/47 chain."
        )

    scientific_cap = _number(
        (baseline41.get("oos") or {}).get("strategy_ending_capital")
    )
    executable_cap = _number(
        (baseline42.get("feasible_oos") or {}).get("strategy_ending_capital")
    )
    liquidity_cap = _number(
        (baseline44.get("liquidity_aware") or {}).get("strategy_ending_capital")
    )
    tcn_cap = _number(
        (baseline45.get("tcn_portfolio") or {}).get("strategy_ending_capital")
    )
    rank_cap = _number(
        (baseline46.get("deep_rank_portfolio") or {}).get("strategy_ending_capital")
    )
    v47_cap = _number(
        (baseline47.get("meta_veto_portfolio") or {}).get(
            "strategy_ending_capital"
        )
    )
    if None in (
        scientific_cap, executable_cap, liquidity_cap,
        tcn_cap, rank_cap, v47_cap,
    ):
        raise ValueError("Missing benchmark hierarchy capital for v10.8.48.")

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
    if (
        int(baseline41.get("calendar_sessions", -1))
        != calendar_audit.calendar_sessions
    ):
        raise ValueError("v10.8.48 canonical calendar differs from Control.")

    def on_progress(pct: float, stage: str, completed: int) -> None:
        if progress:
            progress(
                str(stage)[:150],
                max(0, min(100, int(pct))),
                100,
            )

    paths, rollout_rows, training_folds, decisions = (
        run_policy_rollout_meta_veto_pair(
            bars,
            config,
            calculate_reference_fees,
            apply_slippage,
            progress_callback=on_progress,
        )
    )
    baseline = paths["liquidity_baseline"]
    meta = paths["meta_veto"]
    if len(baseline.predictions) != 1554 or len(meta.predictions) != 1554:
        raise ValueError(
            "v10.8.48 paired replay must contain exactly 1,554 OOS sessions."
        )

    reproduced = float(baseline.metrics["strategy_ending_capital"])
    if not math.isclose(
        reproduced, float(liquidity_cap), rel_tol=0, abs_tol=1e-6,
    ):
        raise ValueError(
            "v10.8.44 parity failed before v10.8.48 meta-veto: "
            f"expected {liquidity_cap:.12f}, got {reproduced:.12f}."
        )
    if (
        (baseline.predictions["cash"] < -1e-8).any()
        or (meta.predictions["cash"] < -1e-8).any()
    ):
        raise ValueError("v10.8.48 generated negative CASH.")

    original_folds = (
        (baseline41.get("oos") or {}).get("walk_forward_folds") or []
    )
    baseline_folds = _validate_folds(
        baseline.predictions,
        original_folds,
        float(config.initial_capital),
    )
    meta_folds = _validate_folds(
        meta.predictions,
        original_folds,
        float(config.initial_capital),
    )

    rollout_frame = pd.DataFrame(rollout_rows)
    decision_frame = pd.DataFrame(decisions)
    enabled_folds = int(sum(
        bool(item.get("model_enabled")) for item in training_folds
    ))
    veto_count = int(
        decision_frame["veto_applied"].sum()
        if not decision_frame.empty else 0
    )
    positive_rollouts = int(
        rollout_frame["rotate_better"].sum()
        if not rollout_frame.empty else 0
    )

    output = (
        directory / "validation" / "v10.8.48" / rollout_job_id
    )
    if output.exists():
        raise FileExistsError("Never overwrite v10.8.48 research artifacts.")
    output.mkdir(parents=True)

    for name, run in (
        ("liquidity_baseline", baseline),
        ("rollout_meta_veto", meta),
    ):
        curve = run.predictions.copy()
        curve["drawdown"] = (
            curve["strategy_equity"]
            / curve["strategy_equity"].cummax().clip(lower=1e-12)
            - 1.0
        )
        curve.to_csv(
            output / f"{name}_capital_curve.csv",
            float_format="%.17g",
        )
        run.trades.to_csv(
            output / f"{name}_fills.csv",
            index=False,
            float_format="%.17g",
        )

    rollout_frame.to_csv(
        output / "paired_rollout_labels.csv",
        index=False,
        float_format="%.17g",
    )
    pd.DataFrame(training_folds).to_csv(
        output / "rollout_training_folds.csv",
        index=False,
        float_format="%.17g",
    )
    decision_frame.to_csv(
        output / "rollout_meta_veto_decisions.csv",
        index=False,
        float_format="%.17g",
    )
    pd.DataFrame(baseline_folds).assign(
        path="liquidity_baseline"
    ).to_csv(
        output / "liquidity_baseline_folds.csv",
        index=False,
        float_format="%.17g",
    )
    pd.DataFrame(meta_folds).assign(
        path="rollout_meta_veto"
    ).to_csv(
        output / "rollout_meta_veto_capital_folds.csv",
        index=False,
        float_format="%.17g",
    )
    aligned = pd.concat([
        baseline.predictions["strategy_equity"].rename(
            "liquidity_baseline"
        ),
        meta.predictions["strategy_equity"].rename(
            "rollout_meta_veto"
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
        aligned["rollout_meta_veto"],
        label="v10.8.48 Policy-Rollout Meta-Veto",
    )
    ax.set_yscale("log")
    ax.set_title(
        "Control Policy-Rollout Advantage Meta-Veto · research only"
    )
    ax.set_ylabel("USD, log scale")
    ax.grid(alpha=.2)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output / "paired_capital.png", dpi=140)
    plt.close(fig)

    report = {
        "schema_version": 1,
        "research_kind": "control_policy_rollout_advantage_meta_veto",
        "strategy_mode": MODE,
        "source_job_id": source_job_id,
        "source_validation_job_id": validation_job_id,
        "source_execution_job_id": execution_job_id,
        "source_liquidity_job_id": liquidity_job_id,
        "source_tcn_job_id": tcn_job_id,
        "source_ranking_job_id": ranking_job_id,
        "source_advantage_job_id": advantage_job_id,
        "rollout_job_id": rollout_job_id,
        "source_snapshot_sha256": expected_sha256,
        "source_unchanged": True,
        "numeric_input_integrity": {"status": "verified", "assets": 55},
        "completed_session": cutoff,
        "benchmark_hierarchy": {
            "scientific_control_usd": scientific_cap,
            "execution_constrained_control_usd": executable_cap,
            "liquidity_aware_v1044_usd": liquidity_cap,
            "failed_tiny_tcn_v1045_usd": tcn_cap,
            "failed_deep_rank_v1046_usd": rank_cap,
            "safe_fallback_v1047_usd": v47_cap,
        },
        "v1044_parity": {
            "status": "verified",
            "expected_ending_capital": liquidity_cap,
            "reproduced_ending_capital": reproduced,
            "absolute_difference": abs(
                reproduced - float(liquidity_cap)
            ),
        },
        "rollout_protocol": {
            "horizon_sessions": ROLLOUT_HORIZON_SESSIONS,
            "paired_same_initial_state": True,
            "rotate_branch": "Control proposed rotation once, then same Control policy",
            "hold_branch": "HOLD incumbent once, then same Control policy",
            "target": "rotate_ending_equity - hold_ending_equity",
            "execution_model": "unchanged v10.8.42 constrained execution",
            "policy_after_first_action": "same v10.8.44 Liquidity-Aware Control",
            "fold_1_meta_model": "disabled",
            "fold_2_training_source": "mature prior fold-1 OOS rollouts only",
            "fold_3_training_source": "mature prior fold-1/2 OOS rollouts only",
            "oos_tuning": False,
        },
        "paired_rollout_count": int(len(rollout_frame)),
        "paired_rollout_rotate_better_count": positive_rollouts,
        "training_folds": training_folds,
        "enabled_fold_count": enabled_folds,
        "veto_count": veto_count,
        "liquidity_baseline": _metrics(baseline),
        "rollout_meta_veto_portfolio": _metrics(meta),
        "report_directory": str(output),
        "artifacts": [
            "summary.json",
            "liquidity_baseline_capital_curve.csv",
            "liquidity_baseline_fills.csv",
            "rollout_meta_veto_capital_curve.csv",
            "rollout_meta_veto_fills.csv",
            "paired_rollout_labels.csv",
            "rollout_training_folds.csv",
            "rollout_meta_veto_decisions.csv",
            "liquidity_baseline_folds.csv",
            "rollout_meta_veto_capital_folds.csv",
            "aligned_capital_curves.csv",
            "paired_capital.png",
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
        progress("completed", 1, 1)
    return report
