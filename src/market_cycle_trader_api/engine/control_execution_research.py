"""Recompute historical Control OOS with state-aware execution constraints.

Runs against independently SHA-pinned v10.8.41 verified MCT snapshot, not
the protected Trader scheduler. Uses the SAME trained chronological policies
but substitutes the portfolio-accounting simulator, so partial fills alter
subsequent policy state. The liquidity model is a fixed research scenario.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..tcc_v106_reference.config import CONFIG, build_control_config
from ..tcc_v106_reference.execution import apply_slippage, calculate_reference_fees
from .control_execution_feasibility import SCENARIO
from .control_execution_adapter import run_feasible_lightgbm
from .control_snapshot_validation import read_verified_control_snapshot
from .operational_control_contract import prepare_operational_control_panel

Progress = Callable[[str, int, int], None]


def _number(x: Any) -> float | None:
    if x is None:
        return None
    n = float(x)
    return n if math.isfinite(n) else None


def run_control_execution_feasibility(
    *,
    source_job_id: str,
    validation_job_id: str,
    execution_job_id: str,
    expected_sha256: str,
    baseline: dict[str, Any],
    progress: Progress | None = None,
    snapshot_root: Path | None = None,
) -> dict[str, Any]:
    if not execution_job_id.startswith("control-execution-") or len(execution_job_id) != 34:
        raise ValueError("Expected server-generated Control execution job ID.")
    if progress is not None:
        progress("verify_immutable_snapshot", 0, 1)
    bars, manifest, directory = read_verified_control_snapshot(
        source_job_id, snapshot_root=snapshot_root,
        expected_sha256=expected_sha256,
    )
    session = str(manifest["completed_session"])
    inclusive_utc = (
        pd.Timestamp(session, tz="UTC")
        + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    ).isoformat()
    config = build_control_config(CONFIG).model_copy(update={
        "analysis_end_date": inclusive_utc,
        "end_date": session,
    })
    _, _, audit = prepare_operational_control_panel(
        bars, completed_session=session, config=config,
    )
    if len(bars) != int(baseline.get("snapshot_assets", -1)):
        raise ValueError("Baseline's eligible asset count differs from frozen snapshot.")
    if str(baseline.get("source_snapshot_sha256")) != expected_sha256:
        raise ValueError("Baseline Control digest differs from frozen snapshot.")
    if not bool((baseline.get("original_shadow") or {}).get("reproduced")):
        raise ValueError("Baseline calibration has not been reproduced; execution scenario rejected.")
    integrity = baseline.get("numeric_input_integrity") or {}
    if integrity.get("status") != "verified" or int(integrity.get("checked_assets", -1)) != len(bars):
        raise ValueError("Exact original numeric Control input integrity is not verified.")

    def on_progress(percentage: float, stage: str, _completed: int) -> None:
        if progress is not None:
            progress(str(stage)[:160], max(0, min(100, int(percentage))), 100)
    results = run_feasible_lightgbm(
        bars, config, calculate_reference_fees, apply_slippage,
        progress_callback=on_progress,
    )
    if len(results) != 1 or results[0].predictions.empty:
        raise ValueError("Control feasibility OOS did not yield a complete single-run replay.")
    result = results[0]
    original = baseline.get("oos") or {}
    if int(result.metrics.get("walk_forward_fold_count", 0)) != int(original.get("walk_forward_fold_count", -1)):
        raise ValueError("Feasibility walk-forward fold count differs from verified Control reference.")
    if int(result.metrics.get("session_count", -1)) != sum(
        int(fold.get("sessions", 0)) for fold in original.get("walk_forward_folds", [])
    ):
        raise ValueError("Feasibility OOS session count differs from verified reference.")
    for old, new in zip(original.get("walk_forward_folds", []), result.metrics["walk_forward_folds"]):
        if (pd.Timestamp(old["test_start"]).date() != pd.Timestamp(new["test_start"]).date()
                or pd.Timestamp(old["test_end"]).date() != pd.Timestamp(new["test_end"]).date()):
            raise ValueError("Feasibility OOS fold dates differ from verified reference.")

    output = directory / "validation" / "v10.8.42" / execution_job_id
    if output.exists():
        raise FileExistsError("Execution report directory already exists; never overwrite a run.")
    output.mkdir(parents=True)
    curve = result.predictions.copy()
    curve["drawdown"] = curve["strategy_equity"] / curve["strategy_equity"].cummax().clip(lower=1e-12) - 1.0
    curve.to_csv(output / "feasible_capital_curve.csv", index=True, float_format="%.17g")
    trades = result.trades.copy()
    trades.to_csv(output / "feasible_fills.csv", index=False, float_format="%.17g")
    curve[[
        "decision_date", "signal_asset", "previous_asset", "selected_asset",
        "cash", "shares", "cash_weight", "requested_quantity",
        "executed_quantity", "unfilled_quantity", "trade_action",
        "trade_reason", "walk_forward_fold", "actual_position_used_in_policy",
        "mark_stale",
    ]].to_csv(output / "feasible_decisions.csv", index=True, float_format="%.17g")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(curve.index, curve["strategy_equity"], label="Control feasibility scenario")
    ax.set_title("Control — capacity-constrained capital (not live fills)")
    ax.set_ylabel("USD · marked-to-market, unliquidated at end")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output / "feasible_capital.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(curve.index, 100 * curve["drawdown"])
    ax.set_title("Control — feasibility scenario drawdown")
    ax.set_ylabel("%")
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output / "feasible_drawdown.png", dpi=140)
    plt.close(fig)

    keys = (
        "initial_capital", "strategy_ending_capital", "strategy_return",
        "strategy_cagr", "strategy_sharpe", "strategy_maximum_drawdown",
        "risk_adjusted_compound_score", "buy_hold_ending_capital",
        "market_exposure", "cash_weight_mean", "cash_days",
        "capital_rotations", "simulated_buys", "simulated_sells",
        "blocked_sessions", "partial_sessions", "zero_volume_block_sessions",
        "unfilled_requested_shares", "modeled_price_cost_usd",
        "total_transaction_fees", "terminal_holdings_shares",
        "terminal_cash", "stale_mark_sessions", "session_count",
    )
    metrics = {
        key: _number(result.metrics[key]) for key in keys
    }
    metrics["terminal_holdings_asset"] = result.metrics["terminal_holdings_asset"]
    fold_rows = []
    for item in result.metrics["walk_forward_folds"]:
        fold_rows.append({
            key: value.isoformat() if isinstance(value, pd.Timestamp) else (
                _number(value) if isinstance(value, (float, np.floating))
                else int(value) if isinstance(value, (int, np.integer))
                else value
            )
            for key, value in item.items()
            if key not in {"model_test_start", "model_test_end"} or value is not None
        })
    verified_original_capital = _number(original.get("strategy_ending_capital"))
    report = {
        "schema_version": 1,
        "research_kind": "control_execution_feasibility_scenario",
        "source_policy": "tcc_v1.0.6_control",
        "source_job_id": source_job_id,
        "source_validation_job_id": validation_job_id,
        "execution_job_id": execution_job_id,
        "source_snapshot_sha256": expected_sha256,
        "completed_session": session,
        "source_unchanged": True,
        "numeric_input_integrity": {
            "status": "verified", "assets": len(bars),
            "method": "manifest SHA and original per-asset numeric OHLCV SHA",
        },
        "calendar_sessions": audit.calendar_sessions,
        "execution_scenario": dict(SCENARIO),
        "source_verified_control": {
            "strategy_ending_capital": verified_original_capital,
            "strategy_cagr": _number(original.get("strategy_cagr")),
            "strategy_sharpe": _number(original.get("strategy_sharpe")),
            "strategy_maximum_drawdown": _number(original.get("strategy_maximum_drawdown")),
            "capital_rotations": original.get("capital_rotations"),
            "walk_forward_fold_count": original.get("walk_forward_fold_count"),
            "source_original_calibration_reproduced": True,
        },
        "feasible_oos": {
            **metrics,
            "walk_forward_fold_count": len(fold_rows),
            "walk_forward_folds": fold_rows,
        },
        "capital_delta_vs_unconstrained_usd": (
            metrics["strategy_ending_capital"] - verified_original_capital
            if verified_original_capital is not None else None
        ),
        "caveats": [
            "Next-session original Control models recalculated by fold; policy is queried at each close using actual constrained position, not the unconstrained decision CSV.",
            "Historical completed-day volume is used only as an ex-post execution-capacity constraint. It is not used in signal selection or pre-session target sizing.",
            "A daily bar does not disclose intraday order book liquidity, spread, fill sequence or the opening auction. Modeled price/impact are fixed assumptions, not verified executions.",
            "The equal-weight reference benchmark remains unconstrained and cannot be compared as an executable portfolio.",
            "Terminal stock is valued at last close without assuming a liquidity-free forced sale. Marked equity may not be liquidatable.",
            "Prices use split-normalized historical bars and actual final valuation is a scenario, not a live trading recommendation.",
            "No Alpaca API calls, live orders, production Strategy selection, Winner promotion or TCC repository writes.",
        ],
        "report_directory": str(output),
        "artifacts": [
            "summary.json", "feasible_capital_curve.csv", "feasible_decisions.csv",
            "feasible_fills.csv", "feasible_capital.png", "feasible_drawdown.png",
        ],
        "order_eligible": False,
        "order_submission": "never",
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    if progress is not None:
        progress("completed", 1, 1)
    return report
