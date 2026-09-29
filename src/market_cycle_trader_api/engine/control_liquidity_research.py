"""Predeclared historical experiment: liquidity-aware candidate selection.

Read-only snapshot, original verified Control policy and v10.8.42 execution.
The research-specific mode is NOT registered in protected operational
strategy selectors. No orders, Alpaca calls, tuning, promotion or TCC edits.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..tcc_v106_reference.config import CONFIG, build_control_config
from ..tcc_v106_reference.execution import apply_slippage, calculate_reference_fees
from .control_execution_sensitivity import PARITY_METRICS, _validate_folds
from .control_liquidity_policy import (
    POLICY_SPEC, RESEARCH_STRATEGY_MODE, run_control_liquidity_pair,
)
from .control_snapshot_validation import read_verified_control_snapshot
from .operational_control_contract import prepare_operational_control_panel

Progress = Callable[[str, int, int], None]
REPORT_METRICS = (
    "initial_capital", "strategy_ending_capital", "strategy_return",
    "strategy_cagr", "strategy_sharpe", "strategy_maximum_drawdown",
    "risk_adjusted_compound_score", "market_exposure", "cash_weight_mean",
    "cash_days", "capital_rotations", "policy_target_changes",
    "simulated_buys", "simulated_sells", "blocked_sessions",
    "partial_sessions", "zero_volume_block_sessions",
    "unfilled_requested_shares", "modeled_price_cost_usd",
    "total_transaction_fees", "terminal_holdings_shares",
    "terminal_cash", "stale_mark_sessions", "session_count",
)


def _numeric(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def run_control_liquidity_research(
    *,
    source_job_id: str,
    validation_job_id: str,
    execution_job_id: str,
    sensitivity_job_id: str,
    research_job_id: str,
    expected_sha256: str,
    baseline41: dict[str, Any],
    baseline42: dict[str, Any],
    baseline43: dict[str, Any],
    progress: Progress | None = None,
    snapshot_root: Path | None = None,
) -> dict[str, Any]:
    if not re.fullmatch(r"control-liquidity-[a-f0-9]{16}", research_job_id):
        raise ValueError("Expected server-generated research job identifier.")
    if progress:
        progress("verify_immutable_source", 0, 1)
    bars, manifest, directory = read_verified_control_snapshot(
        source_job_id, snapshot_root=snapshot_root, expected_sha256=expected_sha256,
    )
    oos41 = baseline41.get("oos") or {}
    oos42 = baseline42.get("feasible_oos") or {}
    scenario43 = baseline43.get("scenario_comparison") or []
    reference43 = next(
        (row for row in scenario43 if row.get("scenario") == "cap10_cost"),
        None,
    )
    if (
        len(bars) != 55
        or not all(
            source.get("source_snapshot_sha256") == expected_sha256
            and source.get("source_job_id") == source_job_id
            and source.get("source_unchanged") is True
            and source.get("order_submission") == "never"
            for source in (baseline41, baseline42, baseline43)
        )
        or (baseline41.get("original_shadow") or {}).get("reproduced") is not True
        or baseline42.get("source_validation_job_id") != validation_job_id
        or baseline43.get("source_validation_job_id") != validation_job_id
        or baseline43.get("source_execution_job_id") != execution_job_id
        or baseline43.get("v1042_scenario_regression") != "verified"
        or len(scenario43) != 5
        or int((baseline41.get("numeric_input_integrity") or {}).get("checked_assets", -1)) != 55
        or int((baseline42.get("numeric_input_integrity") or {}).get("assets", -1)) != 55
        or int((baseline43.get("numeric_input_integrity") or {}).get("assets", -1)) != 55
        or any(
            (source.get("numeric_input_integrity") or {}).get("status") != "verified"
            for source in (baseline41, baseline42, baseline43)
        )
        or int(oos41.get("walk_forward_fold_count", -1)) != 3
        or int(oos42.get("walk_forward_fold_count", -1)) != 3
        or reference43 is None
        or not math.isclose(
            float(reference43.get("strategy_ending_capital", float("nan"))),
            float(oos42.get("strategy_ending_capital", float("nan"))),
            rel_tol=0, abs_tol=1e-6,
        )
    ):
        raise ValueError("Expected exact matched and verified v10.8.41–v10.8.43 Control sources.")
    if (baseline42.get("execution_scenario") or {}).get("participation_rate") != .10:
        raise ValueError("Reference execution scenario must retain 10% participation.")
    cutoff = str(manifest["completed_session"])
    inclusive_utc = (
        pd.Timestamp(cutoff, tz="UTC") + pd.Timedelta(days=1)
        - pd.Timedelta(seconds=1)
    ).isoformat()
    config = build_control_config(CONFIG).model_copy(update={
        "analysis_end_date": inclusive_utc, "end_date": cutoff,
    })
    _, _, audit = prepare_operational_control_panel(
        bars, completed_session=cutoff, config=config,
    )
    if audit.calendar_sessions != int(baseline41.get("calendar_sessions", -1)):
        raise ValueError("Canonical calendar differs from verified scientific baseline.")

    def progress_adapter(pct: float, stage: str, _completed: int) -> None:
        if progress:
            progress(str(stage)[:160], max(0, min(100, int(pct))), 100)

    runs, reference, detail = run_control_liquidity_pair(
        bars, config, calculate_reference_fees, apply_slippage,
        progress_callback=progress_adapter,
    )
    if set(runs) != {"control_reference", "liquidity_aware"}:
        raise ValueError("Experiment missing a predeclared paired path.")
    for key in PARITY_METRICS:
        old, new = _numeric(oos42.get(key)), _numeric(reference.metrics.get(key))
        if old is None or new is None or not math.isclose(old, new, rel_tol=0, abs_tol=1e-6):
            raise ValueError(
                f"Control reference parity FAILED: {key}: expected={old}, actual={new}. "
                "Do not publish liquidity-aware outcome without the exact original replay."
            )
    if reference.metrics["terminal_holdings_asset"] != oos42.get("terminal_holdings_asset"):
        raise ValueError("Control reference ending held asset changed.")
    original_folds = oos41["walk_forward_folds"]
    initial = float(config.initial_capital)
    original_dates = reference.predictions.index
    if len(original_dates) != 1554 or len(detail) != 1554:
        raise ValueError("Expected 1,554 audited chronological decision dates.")
    decision_keys = pd.DatetimeIndex(
        pd.to_datetime(runs["liquidity_aware"].predictions["decision_date"], utc=True)
    )
    if len(decision_keys) != len(original_dates) or set(decision_keys) != set(detail):
        raise ValueError("Decision audit dates do not align with prior-close context.")
    fold_comparison = []
    summary = []
    for name in ("control_reference", "liquidity_aware"):
        run = runs[name]
        if not run.predictions.index.equals(original_dates):
            raise ValueError(f"Research mode {name} has a different chronological OOS calendar.")
        folds = _validate_folds(run.predictions, original_folds, initial)
        for row in folds:
            fold_comparison.append({"policy": name, **row})
        ending = float(run.metrics["strategy_ending_capital"])
        if not math.isclose(ending, folds[-1]["strategy_ending_capital"], rel_tol=0, abs_tol=1e-7):
            raise AssertionError("Ending capital differs from final chronological fold.")
        summary.append({
            "policy": name,
            "strategy_mode": (
                RESEARCH_STRATEGY_MODE if name == "liquidity_aware"
                else "FROZEN_TCC_CONTROL_BASE_REFERENCE"
            ),
            **{metric: _numeric(run.metrics.get(metric)) for metric in REPORT_METRICS},
            "terminal_holdings_asset": str(run.metrics["terminal_holdings_asset"]),
        })
    output = directory / "validation" / "v10.8.44" / research_job_id
    if output.exists():
        raise FileExistsError("Never overwrite an existing research result.")
    output.mkdir(parents=True)
    filenames = []
    for name, run in runs.items():
        curve = run.predictions.copy()
        curve["drawdown"] = (
            curve["strategy_equity"] / curve["strategy_equity"].cummax() - 1.0
        )
        curve.to_csv(output / f"{name}_curve.csv", float_format="%.17g")
        run.trades.to_csv(
            output / f"{name}_fills.csv", index=False, float_format="%.17g",
        )
        filenames.extend([f"{name}_curve.csv", f"{name}_fills.csv"])
    pd.DataFrame(summary).to_csv(
        output / "policy_comparison.csv", index=False, float_format="%.17g",
    )
    pd.DataFrame(fold_comparison).to_csv(
        output / "fold_comparison.csv", index=False, float_format="%.17g",
    )
    aligned = pd.concat(
        [runs[name].predictions["strategy_equity"].rename(name)
         for name in ("control_reference", "liquidity_aware")],
        axis=1,
    )
    aligned.to_csv(output / "aligned_capital_curves.csv", float_format="%.17g")
    records = []
    for date in original_dates:
        decision_key = pd.Timestamp(
            runs["liquidity_aware"].predictions.loc[date, "decision_date"]
        )
        item = detail[decision_key]
        picked = str(runs["liquidity_aware"].predictions.loc[date, "signal_asset"])
        raw = str(runs["control_reference"].predictions.loc[date, "signal_asset"])
        details = item["candidate_liquidity_detail"]
        records.append({
            "timestamp": date.isoformat(),
            "decision_timestamp": decision_key.isoformat(),
            "original_policy_target": raw,
            "liquidity_policy_target": picked,
            "raw_best_asset": item["raw_best_asset"],
            "adjusted_best_asset": item["adjusted_best_asset"],
            "account_equity_at_decision": item["equity_at_decision"],
            "account_cash_at_decision": item["cash_at_decision"],
            "held_asset_at_decision": item["held_asset_at_decision"],
            "incumbent_exit_fraction": item["incumbent_exit_fraction"],
            "selected_asset_capacity_fraction": (
                details.get(picked, {}).get("capacity_fraction")
                if picked != "CASH" else 0.0
            ),
            "selected_asset_raw_utility": (
                details.get(picked, {}).get("raw_utility")
                if picked != "CASH" else 0.0
            ),
            "selected_asset_effective_utility": (
                details.get(picked, {}).get("effective_utility")
                if picked != "CASH" else 0.0
            ),
            "target_differs_from_control": raw != picked,
            "raw_rank_differs_from_liquidity_rank": (
                item["raw_best_asset"] != item["adjusted_best_asset"]
            ),
        })
    pd.DataFrame(records).to_csv(
        output / "liquidity_decision_audit.csv",
        index=False, float_format="%.17g",
    )
    candidate_rows = []
    for timestamp in original_dates:
        decision_key = pd.Timestamp(
            runs["liquidity_aware"].predictions.loc[timestamp, "decision_date"]
        )
        audit_item = detail[decision_key]
        for symbol, d in audit_item["candidate_liquidity_detail"].items():
            candidate_rows.append({
                "timestamp": timestamp.isoformat(),
                "decision_timestamp": decision_key.isoformat(),
                "asset": symbol,
                "known_capacity_dollars": d["capacity_dollars"],
                "capacity_fraction": d["capacity_fraction"],
                "raw_utility": d["raw_utility"],
                "effective_utility": d["effective_utility"],
            })
    pd.DataFrame(candidate_rows).to_csv(
        output / "liquidity_candidate_scores.csv",
        index=False, float_format="%.17g",
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 6))
    for column in aligned.columns:
        ax.plot(aligned.index, aligned[column], label=column)
    ax.set_yscale("log")
    ax.set_title("Original Control vs liquidity-aware Control · research only")
    ax.set_ylabel("Marked equity (USD) · log scale")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output / "paired_capital.png", dpi=140)
    plt.close(fig)

    original = summary[0]
    challenger = summary[1]
    report = {
        "schema_version": 1,
        "research_kind": "control_liquidity_aware_fixed_hypothesis",
        "source_job_id": source_job_id,
        "source_validation_job_id": validation_job_id,
        "source_execution_job_id": execution_job_id,
        "source_sensitivity_job_id": sensitivity_job_id,
        "research_job_id": research_job_id,
        "source_snapshot_sha256": expected_sha256,
        "source_unchanged": True,
        "completed_session": cutoff,
        "numeric_input_integrity": {"status": "verified", "assets": len(bars)},
        "snapshot_assets": len(bars),
        "calendar_sessions": audit.calendar_sessions,
        "model_fit_count": 1,
        "policy_spec": dict(POLICY_SPEC),
        "execution_scenario": dict(reference.metrics["execution_scenario"]),
        "control_reference_parity": "verified",
        "control_reference": original,
        "liquidity_aware": challenger,
        "fold_comparison": fold_comparison,
        "diagnostics": {
            "policy_target_disagreement_sessions": int(
                sum(row["target_differs_from_control"] for row in records)
            ),
            "raw_vs_adjusted_rank_disagreement_sessions": int(
                sum(row["raw_rank_differs_from_liquidity_rank"] for row in records)
            ),
            "candidate_score_rows": len(candidate_rows),
        },
        "ending_capital_difference_usd": (
            challenger["strategy_ending_capital"] - original["strategy_ending_capital"]
        ),
        "caveats": [
            "New conceptual research mode only: never register in production strategy selectors or replace the frozen Control automatically.",
            "The transformation is a prespecified hypothesis, not trained, calibrated or OOS-tuned to these 1554 sessions.",
            "The effective utility uses only decision-date completed close, trailing 20 completed daily volumes and actual simulated current capital/holding.",
            "Both paired policies use exactly the same 10% ex-post realized daily volume cap and hypothetical adverse spread/impact execution scenario.",
            "No current-session execution bar is read for prior-close candidate selection. The next session's volume is visible only to the execution simulator ex post.",
            "The liquidity-aware policy may still fail to exit an illiquid incumbent, and assumed fill prices have not been verified against quotes/order books.",
            "A single historical OOS test is exploratory, not prospective evidence or an instruction to promote policy. No Alpaca, TCC, Winner, production or orders.",
        ],
        "report_directory": str(output),
        "artifacts": [
            "summary.json", "policy_comparison.csv", "fold_comparison.csv",
            "aligned_capital_curves.csv", "liquidity_decision_audit.csv",
            "liquidity_candidate_scores.csv", "paired_capital.png",
            *filenames,
        ],
        "order_eligible": False, "order_submission": "never",
        "source_download": "never",
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    if progress:
        progress("completed", 1, 1)
    return report
