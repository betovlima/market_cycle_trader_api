"""Fixed-design Control execution-sensitivity research on one immutable snapshot.

One frozen chronological LightGBM fit; five state-aware accounting replays.
No TCC edits, Alpaca refresh, orders, Winner selection or OOS-driven tuning.
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
from .control_execution_adapter import SENSITIVITY_SCENARIOS, run_sensitivity_lightgbm
from .control_snapshot_validation import read_verified_control_snapshot
from .operational_control_contract import prepare_operational_control_panel

Progress = Callable[[str, int, int], None]
METRICS = (
    "strategy_ending_capital", "strategy_return", "strategy_cagr",
    "strategy_sharpe", "strategy_maximum_drawdown",
    "risk_adjusted_compound_score", "market_exposure", "cash_weight_mean",
    "cash_days", "capital_rotations", "policy_target_changes",
    "simulated_buys", "simulated_sells", "blocked_sessions",
    "partial_sessions", "zero_volume_block_sessions",
    "unfilled_requested_shares", "modeled_price_cost_usd",
    "total_transaction_fees", "terminal_holdings_shares", "terminal_cash",
    "stale_mark_sessions", "session_count",
)
PARITY_METRICS = (
    "strategy_ending_capital", "strategy_cagr", "strategy_sharpe",
    "strategy_maximum_drawdown", "capital_rotations",
    "simulated_buys", "simulated_sells", "blocked_sessions",
    "partial_sessions", "modeled_price_cost_usd", "total_transaction_fees",
    "terminal_holdings_shares", "terminal_cash", "session_count",
)


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    x = float(value)
    return x if math.isfinite(x) else None


def _validate_folds(predictions: pd.DataFrame, original_folds: list[dict[str, Any]],
                    initial: float) -> list[dict[str, Any]]:
    if predictions.empty or len(original_folds) != 3:
        raise ValueError("Expected three nonempty chronological OOS folds.")
    if len(predictions) != sum(int(item["sessions"]) for item in original_folds):
        raise ValueError("OOS session count diverges from source reference.")
    output = []
    preceding_capital = initial
    for original in original_folds:
        fold_id = int(original["fold_id"])
        subset = predictions.loc[predictions["walk_forward_fold"] == fold_id]
        if (
            len(subset) != int(original["sessions"])
            or subset.index[0].date() != pd.Timestamp(original["test_start"]).date()
            or subset.index[-1].date() != pd.Timestamp(original["test_end"]).date()
        ):
            raise ValueError(f"Scenario fold {fold_id} deviates from source window.")
        end = float(subset["strategy_equity"].iloc[-1])
        extended = pd.concat([
            pd.Series([preceding_capital], dtype=float),
            subset["strategy_equity"].reset_index(drop=True).astype(float),
        ], ignore_index=True)
        maximum_drawdown = float((extended / extended.cummax() - 1).min())
        output.append({
            "fold_id": fold_id,
            "test_start": pd.Timestamp(subset.index[0]).isoformat(),
            "test_end": pd.Timestamp(subset.index[-1]).isoformat(),
            "sessions": len(subset),
            "strategy_starting_capital": preceding_capital,
            "strategy_ending_capital": end,
            "strategy_return": end / preceding_capital - 1.0,
            "maximum_drawdown": maximum_drawdown,
            "mean_cash_weight": float(subset["cash_weight"].mean()),
        })
        preceding_capital = end
    if sum(row["sessions"] for row in output) != len(predictions):
        raise AssertionError("Fold accounting failed to partition the OOS replay.")
    return output


def run_control_execution_sensitivity(
    *,
    source_job_id: str,
    validation_job_id: str,
    feasibility_job_id: str,
    sensitivity_job_id: str,
    expected_sha256: str,
    baseline41: dict[str, Any],
    baseline42: dict[str, Any],
    progress: Progress | None = None,
    snapshot_root: Path | None = None,
) -> dict[str, Any]:
    if not re.fullmatch(r"control-sensitivity-[a-f0-9]{16}", sensitivity_job_id):
        raise ValueError("Expected server-generated Control sensitivity job ID.")
    if progress:
        progress("verify_immutable_snapshot", 0, 1)
    bars, manifest, directory = read_verified_control_snapshot(
        source_job_id, snapshot_root=snapshot_root, expected_sha256=expected_sha256,
    )
    cutoff = str(manifest["completed_session"])
    base41_oos = baseline41.get("oos") or {}
    base42_oos = baseline42.get("feasible_oos") or {}
    if (
        baseline41.get("source_snapshot_sha256") != expected_sha256
        or baseline42.get("source_snapshot_sha256") != expected_sha256
        or baseline41.get("source_job_id") != source_job_id
        or baseline42.get("source_job_id") != source_job_id
        or baseline42.get("source_validation_job_id") != validation_job_id
        or not baseline41.get("source_unchanged")
        or not baseline42.get("source_unchanged")
        or not (baseline41.get("original_shadow") or {}).get("reproduced")
        or (baseline41.get("numeric_input_integrity") or {}).get("status") != "verified"
        or (baseline42.get("numeric_input_integrity") or {}).get("status") != "verified"
        or len(bars) != 55
        or int(baseline41.get("snapshot_assets", -1)) != len(bars)
        or int((baseline41.get("numeric_input_integrity") or {}).get("checked_assets", -1)) != len(bars)
        or int((baseline42.get("numeric_input_integrity") or {}).get("assets", -1)) != len(bars)
        or int(base41_oos.get("walk_forward_fold_count", -1)) != 3
        or int(base42_oos.get("walk_forward_fold_count", -1)) != 3
        or baseline41.get("order_submission") != "never"
        or baseline42.get("order_submission") != "never"
    ):
        raise ValueError("Sensitivity requires both exact reproduced and SHA-pinned Control research results.")
    original_capital = _finite(base41_oos.get("strategy_ending_capital"))
    old_feasible_capital = _finite(base42_oos.get("strategy_ending_capital"))
    if original_capital is None or old_feasible_capital is None:
        raise ValueError("Missing source capital reference.")
    if not math.isclose(
        float((baseline42.get("source_verified_control") or {}).get("strategy_ending_capital", float("nan"))),
        original_capital, rel_tol=0, abs_tol=1e-6,
    ):
        raise ValueError("Source feasibility baseline has different scientific Control capital.")
    old_scenario = baseline42.get("execution_scenario") or {}
    if (old_scenario.get("participation_rate") != .10
            or old_scenario.get("assumed_full_spread_bps") != 15
            or old_scenario.get("assumed_impact_coefficient_bps") != 20):
        raise ValueError("Expected the exact v10.8.42 10% reference scenario.")

    inclusive_utc = (
        pd.Timestamp(cutoff, tz="UTC")
        + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    ).isoformat()
    config = build_control_config(CONFIG).model_copy(update={
        "analysis_end_date": inclusive_utc, "end_date": cutoff,
    })
    _, _, audit = prepare_operational_control_panel(
        bars, completed_session=cutoff, config=config,
    )
    if audit.calendar_sessions != int(baseline41["calendar_sessions"]):
        raise ValueError("Calendar drift versus original scientific Control.")

    def progress_adapter(pct: float, stage: str, _completed: int) -> None:
        if progress:
            progress(str(stage)[:160], max(0, min(100, int(pct))), 100)

    runs, primary = run_sensitivity_lightgbm(
        bars, config, calculate_reference_fees, apply_slippage,
        progress_callback=progress_adapter,
    )
    names = [name for name, _ in SENSITIVITY_SCENARIOS]
    if list(runs) != names or runs["cap10_cost"] is not primary:
        raise ValueError("Missing or reordered sensitivity scenario.")
    for key in PARITY_METRICS:
        old = _finite(base42_oos.get(key))
        new = _finite(primary.metrics.get(key))
        if old is None or new is None or not math.isclose(old, new, rel_tol=0, abs_tol=1e-6):
            raise ValueError(
                f"v10.8.42 regression mismatch: {key} ({old} vs {new}). "
                "Do not compare sensitivity scenarios with a changed Control baseline."
            )
    if str(primary.metrics["terminal_holdings_asset"]) != str(base42_oos["terminal_holdings_asset"]):
        raise ValueError("v10.8.42 ending asset regression mismatch.")

    output = directory / "validation" / "v10.8.43" / sensitivity_job_id
    if output.exists():
        raise FileExistsError("Sensitivity report already exists; source remains immutable.")
    output.mkdir(parents=True)
    scenario_rows = []
    fold_rows = []
    curve_parts = []
    file_names = []
    original_folds = base41_oos["walk_forward_folds"]
    initial = float(config.initial_capital)
    reference_index = primary.predictions.index
    for name, override in SENSITIVITY_SCENARIOS:
        run = runs[name]
        data = run.predictions.copy()
        if not data.index.equals(reference_index) or len(data) != len(reference_index):
            raise ValueError(f"Scenario {name} has a different OOS trading calendar.")
        scenario_config = run.metrics.get("execution_scenario") or {}
        if any(scenario_config.get(key) != value for key, value in override.items()):
            raise ValueError(f"Scenario {name} was not executed with its predeclared assumptions.")
        folds = _validate_folds(data, original_folds, initial)
        for fold in folds:
            fold_rows.append({"scenario": name, **fold})
        cap = float(run.metrics["strategy_ending_capital"])
        if not math.isclose(cap, folds[-1]["strategy_ending_capital"], rel_tol=0, abs_tol=1e-7):
            raise AssertionError("Scenario fold and portfolio accounting differ.")
        summary = {
            "scenario": name,
            **{key: _finite(run.metrics.get(key)) for key in METRICS},
            "terminal_holdings_asset": run.metrics["terminal_holdings_asset"],
            "participation_rate": float(scenario_config["participation_rate"]),
            "unlimited_capacity": bool(scenario_config["unlimited_capacity"]),
            "assumed_full_spread_bps": float(scenario_config["assumed_full_spread_bps"]),
            "assumed_impact_coefficient_bps": float(scenario_config["assumed_impact_coefficient_bps"]),
            "delta_vs_v1042_usd": cap - old_feasible_capital,
            "delta_vs_scientific_usd": cap - original_capital,
        }
        scenario_rows.append(summary)
        data["drawdown"] = data["strategy_equity"] / data["strategy_equity"].cummax() - 1
        data.to_csv(output / f"{name}_curve.csv", index=True, float_format="%.17g")
        run.trades.to_csv(output / f"{name}_fills.csv", index=False, float_format="%.17g")
        file_names.extend([f"{name}_curve.csv", f"{name}_fills.csv"])
        curve_parts.append(data["strategy_equity"].rename(name))
    pd.DataFrame(scenario_rows).to_csv(
        output / "scenario_comparison.csv", index=False, float_format="%.17g",
    )
    pd.DataFrame(fold_rows).to_csv(
        output / "fold_comparison.csv", index=False, float_format="%.17g",
    )
    pd.concat(curve_parts, axis=1).to_csv(
        output / "aligned_capital_curves.csv", index=True, float_format="%.17g",
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 6))
    for curve in curve_parts:
        ax.plot(curve.index, curve.values, label=curve.name)
    ax.set_yscale("log")
    ax.set_title("Control OOS — fixed execution sensitivity scenarios (marked equity)")
    ax.set_ylabel("USD · log scale · no guaranteed broker fills")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output / "scenario_capital.png", dpi=140)
    plt.close(fig)
    report = {
        "schema_version": 1,
        "research_kind": "control_execution_fixed_sensitivity",
        "source_job_id": source_job_id,
        "source_validation_job_id": validation_job_id,
        "source_execution_job_id": feasibility_job_id,
        "sensitivity_job_id": sensitivity_job_id,
        "source_snapshot_sha256": expected_sha256,
        "source_unchanged": True,
        "completed_session": cutoff,
        "snapshot_assets": len(bars),
        "calendar_sessions": audit.calendar_sessions,
        "numeric_input_integrity": {"status": "verified", "assets": len(bars)},
        "method": "One original chronological LightGBM fit and independent state-aware OOS accounting per fixed case.",
        "original_scientific_capital_usd": original_capital,
        "verified_execution_v1042_capital_usd": old_feasible_capital,
        "v1042_scenario_regression": "verified",
        "predeclared_scenarios": [name for name, _ in SENSITIVITY_SCENARIOS],
        "scenario_comparison": scenario_rows,
        "fold_comparison": fold_rows,
        "comparisons_are_not_causal_attributions": True,
        "caveats": [
            "No OOS scenario is used to select a winner, retune a margin, change universe or promote a trading policy.",
            "The no-cap comparator is intentionally idealized: it can fill above actual daily volume and even when daily volume is zero. It is NOT executable.",
            "Each scenario recomputes decisions with its own actual held position, cash and holding history. Pairwise equity differences combine execution, timing and policy-state effects; they do not isolate a causal fee or capacity contribution.",
            "Realized next-session volume is used ex post only to constrain fills, never to create a signal or determine a prior-close action.",
            "Spread and square-root impact are declared hypothetical assumptions. Daily OHLCV does not establish intraday fill prices or opening-auction capacity.",
            "All scenarios use whole shares, original Control fees, next-open reference price and close mark; no forced final sell. The old v10.8.41 fractional-share/terminal-sell science replay is not an accounting-identical no-cap comparator.",
            "The original equal-weight benchmark is not constrained for liquidity.",
            "No Alpaca API calls, real orders, production Strategy changes, TCC writes or Winner promotion.",
        ],
        "report_directory": str(output),
        "artifacts": [
            "summary.json", "scenario_comparison.csv", "fold_comparison.csv",
            "aligned_capital_curves.csv", "scenario_capital.png", *file_names,
        ],
        "order_eligible": False, "order_submission": "never",
        "source_download": "never",
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    if progress:
        progress("completed", 1, 1)
    return report
