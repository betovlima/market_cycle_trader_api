"""Integrated v10.8.47 replay: exact Liquidity-Aware Control vs meta-veto.

One frozen LightGBM training/calibration chain builds the original Control
policies.  The same fold models, utility caches and execution assumptions are
then replayed twice:
  1) v10.8.44 Liquidity-Aware baseline;
  2) v10.8.47 Liquidity-Aware + counterfactual advantage meta-veto.

No operational registration and no order path.
"""
from __future__ import annotations

from types import FunctionType
from typing import Any, Callable
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..tcc_v106_reference import research_challengers as scientific
from ..tcc_v106_reference.config import CONFIG, build_control_config
from ..tcc_v106_reference.execution import calculate_reference_fees, apply_slippage
from ..tcc_v106_reference.capital_rotation import _utility_policy as frozen_utility_policy
from .control_execution_feasibility import simulate_feasible_control
from .control_snapshot_validation import read_verified_control_snapshot
from .control_execution_sensitivity import _validate_folds
from .operational_control_contract import prepare_operational_control_panel
from .control_liquidity_policy import _CapitalAwareUtilityCache
from .control_counterfactual_advantage import (
    MODE,
    RANDOM_SEED,
    apply_meta_veto,
    build_cross_sectional_pair_samples,
    fit_meta_veto,
    predict_rotation_advantage_probability,
    refit_meta_veto,
)


def run_counterfactual_meta_veto_pair(
    bars: dict[str, pd.DataFrame],
    config: Any,
    fee_calculator: Callable,
    slippage: Callable,
    *,
    progress_callback: Callable[[float, str, int], None] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    (
        frames, common_dates, symbols, folds, _all_decision_dates,
        _decision_to_fold, _decision_metadata,
    ) = scientific._build_execution_context(bars, config)
    if len(symbols) != 55 or len(folds) != 3:
        raise ValueError("v10.8.47 expects frozen 55-asset / 3-fold Control context.")

    meta_by_fold: dict[int, dict[str, Any]] = {}
    fold_reports: list[dict[str, Any]] = []
    for fold in folds:
        fid = int(fold["fold_id"])
        train_dates = common_dates[:int(fold["train_end_index"])]
        calibration_dates = common_dates[
            int(fold["calibration_start_index"]):int(fold["calibration_end_index"])
        ]
        final_dates = common_dates[:int(fold["final_fit_end_index"])]

        training = build_cross_sectional_pair_samples(
            frames, symbols, train_dates,
            maturity_before=pd.Timestamp(fold["calibration_start"]),
        )
        validation = build_cross_sectional_pair_samples(
            frames, symbols, calibration_dates,
            maturity_before=pd.Timestamp(fold["test_start"]),
            date_stride=1,
        )
        if progress_callback:
            progress_callback(
                2.0 + 5.0 * (fid - 1),
                f"Meta-veto fold {fid}: chronological calibration",
                0,
            )
        _cal_model, _cal_scale, epochs, report = fit_meta_veto(
            frames, training, validation, seed=RANDOM_SEED + fid,
        )
        final_samples = build_cross_sectional_pair_samples(
            frames, symbols, final_dates,
            maturity_before=pd.Timestamp(fold["test_start"]),
        )
        final_model, final_scale = refit_meta_veto(
            frames, final_samples,
            seed=RANDOM_SEED + 100 + fid,
            epochs=epochs,
        )
        meta_by_fold[fid] = {
            "model": final_model,
            "scale": final_scale,
            "enabled": bool(report["model_enabled"]),
        }
        fold_reports.append({
            "fold_id": fid,
            "training_pairs": len(training),
            "calibration_pairs": len(validation),
            "final_training_pairs": len(final_samples),
            "test_start": pd.Timestamp(fold["test_start"]).isoformat(),
            "test_end": pd.Timestamp(fold["test_end"]).isoformat(),
            **report,
        })

    original_runner = scientific._run_lightgbm
    original_policy = original_runner.__globals__.get("_utility_policy")
    original_simulator = original_runner.__globals__.get("_simulate_exact")
    if original_policy is not frozen_utility_policy or original_simulator is not scientific._simulate_exact:
        raise RuntimeError("Frozen Control runner bindings changed unexpectedly.")

    account: dict[str, Any] = {"enabled": True, "mode": "baseline", "audit": {}}
    meta_audit: list[dict[str, Any]] = []

    def policy_wrapper(models, panel, labels, settings, switch_margin, **kwargs):
        utility_cache = kwargs.get("utility_cache")
        fold_id = kwargs.get("fold_id")
        if utility_cache is None or fold_id is None:
            return original_policy(
                models, panel, labels, settings, switch_margin, **kwargs,
            )
        fid = int(fold_id)
        liquidity_cache = _CapitalAwareUtilityCache(
            utility_cache, panel, labels, account,
        )
        base_kwargs = dict(kwargs)
        base_kwargs["utility_cache"] = liquidity_cache
        base_policy = original_policy(
            models, panel, labels, settings, switch_margin, **base_kwargs,
        )
        meta = meta_by_fold[fid]

        def policy(timestamp: pd.Timestamp, current_position: int, holding_days: int):
            control_target, control_score = base_policy(
                timestamp, current_position, holding_days,
            )
            if account.get("mode") != "meta":
                return control_target, control_score
            current_asset = labels[current_position-1] if current_position > 0 else "CASH"
            candidate_asset = labels[control_target-1] if control_target > 0 else "CASH"
            probability = predict_rotation_advantage_probability(
                meta["model"], meta["scale"], panel,
                date=pd.Timestamp(timestamp),
                incumbent=current_asset,
                candidate=candidate_asset,
            )
            final_target, reason = apply_meta_veto(
                current_position=int(current_position),
                control_target=int(control_target),
                probability_positive_advantage=probability,
                model_enabled=bool(meta["enabled"]),
            )
            meta_audit.append({
                "decision_date": pd.Timestamp(timestamp),
                "fold_id": fid,
                "incumbent_asset": current_asset,
                "control_target_asset": candidate_asset,
                "final_target_asset": (
                    labels[final_target-1] if final_target > 0 else "CASH"
                ),
                "probability_positive_advantage": probability,
                "model_enabled": bool(meta["enabled"]),
                "veto_applied": int(final_target) != int(control_target),
                "reason": reason,
            })
            return int(final_target), float(control_score)

        return policy

    captured: dict[str, Any] = {}
    def simulator_wrapper(*args, **kwargs):
        if captured:
            raise ValueError("v10.8.47 expected exactly one scheduled OOS simulation call.")

        def replay(mode: str):
            account.update({"mode": mode, "enabled": True, "audit": {}})
            def prepare(date, position, holding, cash, shares, equity):
                account.update({
                    "decision_timestamp": pd.Timestamp(date),
                    "position": int(position),
                    "holding_days": int(holding),
                    "cash": float(cash),
                    "shares": int(shares),
                    "equity": float(equity),
                })
            return simulate_feasible_control(
                *args,
                **{**kwargs, "decision_prepare": prepare},
            )

        captured["liquidity_baseline"] = replay("baseline")
        captured["meta_veto"] = replay("meta")
        return captured["liquidity_baseline"]

    isolated_globals = dict(original_runner.__globals__)
    isolated_globals["_utility_policy"] = policy_wrapper
    isolated_globals["_simulate_exact"] = simulator_wrapper
    isolated = FunctionType(
        original_runner.__code__, isolated_globals, original_runner.__name__,
        original_runner.__defaults__, original_runner.__closure__,
    )
    isolated.__kwdefaults__ = dict(original_runner.__kwdefaults__ or {})
    result = isolated(
        bars, config, fee_calculator, slippage,
        progress_callback=progress_callback,
        trade_callback=None,
        progress_detail_callback=None,
        technical_log_callback=None,
    )
    if (
        len(result) != 1
        or result[0] is not captured.get("liquidity_baseline")
        or set(captured) != {"liquidity_baseline", "meta_veto"}
    ):
        raise ValueError("v10.8.47 paired replay did not return the expected two paths.")
    if (
        original_runner.__globals__.get("_utility_policy") is not original_policy
        or original_runner.__globals__.get("_simulate_exact") is not original_simulator
    ):
        raise RuntimeError("Frozen TCC module was mutated by v10.8.47.")

    return captured, fold_reports, meta_audit


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


def run_counterfactual_advantage_research(
    *,
    source_job_id: str,
    validation_job_id: str,
    execution_job_id: str,
    liquidity_job_id: str,
    tcn_job_id: str,
    ranking_job_id: str,
    advantage_job_id: str,
    expected_sha256: str,
    baseline41: dict[str, Any],
    baseline42: dict[str, Any],
    baseline44: dict[str, Any],
    baseline45: dict[str, Any],
    baseline46: dict[str, Any],
    progress: Callable[[str, int, int], None] | None = None,
    snapshot_root: Path | None = None,
) -> dict[str, Any]:
    if not re.fullmatch(r"control-advantage-[a-f0-9]{16}", advantage_job_id):
        raise ValueError("Expected server-generated v10.8.47 job ID.")
    bars, manifest, directory = read_verified_control_snapshot(
        source_job_id, snapshot_root=snapshot_root,
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
        or baseline44.get("source_validation_job_id") != validation_job_id
        or baseline44.get("source_execution_job_id") != execution_job_id
        or baseline44.get("control_reference_parity") != "verified"
        or baseline45.get("source_liquidity_job_id") != liquidity_job_id
        or baseline46.get("source_liquidity_job_id") != liquidity_job_id
        or baseline46.get("source_tcn_job_id") != tcn_job_id
        or baseline46.get("research_kind") != "control_deep_pairwise_rank_hybrid"
    ):
        raise ValueError("v10.8.47 requires exact SHA-matched v10.8.41/42/44/45/46 chain.")

    scientific_cap = _number((baseline41.get("oos") or {}).get("strategy_ending_capital"))
    executable_cap = _number((baseline42.get("feasible_oos") or {}).get("strategy_ending_capital"))
    liquidity_cap = _number((baseline44.get("liquidity_aware") or {}).get("strategy_ending_capital"))
    tcn_cap = _number((baseline45.get("tcn_portfolio") or {}).get("strategy_ending_capital"))
    rank_cap = _number((baseline46.get("deep_rank_portfolio") or {}).get("strategy_ending_capital"))
    if None in (scientific_cap, executable_cap, liquidity_cap, tcn_cap, rank_cap):
        raise ValueError("Missing benchmark hierarchy capital for v10.8.47.")

    cutoff = str(manifest["completed_session"])
    inclusive = (
        pd.Timestamp(cutoff, tz="UTC") + pd.Timedelta(days=1)
        - pd.Timedelta(seconds=1)
    ).isoformat()
    config = build_control_config(CONFIG).model_copy(update={
        "analysis_end_date": inclusive, "end_date": cutoff,
    })
    _, _, calendar_audit = prepare_operational_control_panel(
        bars, completed_session=cutoff, config=config,
    )
    if int(baseline41.get("calendar_sessions", -1)) != calendar_audit.calendar_sessions:
        raise ValueError("v10.8.47 canonical calendar differs from Control.")

    def on_progress(pct: float, stage: str, completed: int) -> None:
        if progress:
            progress(str(stage)[:150], max(0, min(100, int(pct))), 100)

    paths, training_folds, decisions = run_counterfactual_meta_veto_pair(
        bars, config, calculate_reference_fees, apply_slippage,
        progress_callback=on_progress,
    )
    baseline = paths["liquidity_baseline"]
    meta = paths["meta_veto"]
    if len(baseline.predictions) != 1554 or len(meta.predictions) != 1554:
        raise ValueError("v10.8.47 paired replay must contain exactly 1,554 OOS sessions.")

    reproduced = float(baseline.metrics["strategy_ending_capital"])
    if not math.isclose(reproduced, float(liquidity_cap), rel_tol=0, abs_tol=1e-6):
        raise ValueError(
            f"v10.8.44 parity failed before meta-veto: expected {liquidity_cap:.12f}, got {reproduced:.12f}."
        )
    if (baseline.predictions["cash"] < -1e-8).any() or (meta.predictions["cash"] < -1e-8).any():
        raise ValueError("v10.8.47 generated negative CASH.")

    original_folds = (baseline41.get("oos") or {}).get("walk_forward_folds") or []
    baseline_folds = _validate_folds(
        baseline.predictions, original_folds, float(config.initial_capital),
    )
    meta_folds = _validate_folds(
        meta.predictions, original_folds, float(config.initial_capital),
    )

    decision_frame = pd.DataFrame(decisions)
    veto_count = int(decision_frame["veto_applied"].sum()) if not decision_frame.empty else 0
    enabled_folds = int(sum(bool(x["model_enabled"]) for x in training_folds))
    output = directory / "validation" / "v10.8.47" / advantage_job_id
    if output.exists():
        raise FileExistsError("Never overwrite v10.8.47 research artifacts.")
    output.mkdir(parents=True)

    for name, run in (("liquidity_baseline", baseline), ("meta_veto", meta)):
        curve = run.predictions.copy()
        curve["drawdown"] = (
            curve["strategy_equity"]
            / curve["strategy_equity"].cummax().clip(lower=1e-12) - 1.0
        )
        curve.to_csv(output / f"{name}_capital_curve.csv", float_format="%.17g")
        run.trades.to_csv(
            output / f"{name}_fills.csv", index=False, float_format="%.17g",
        )
    pd.DataFrame(training_folds).to_csv(
        output / "meta_veto_training_folds.csv", index=False, float_format="%.17g",
    )
    decision_frame.to_csv(
        output / "meta_veto_decisions.csv", index=False, float_format="%.17g",
    )
    pd.DataFrame(baseline_folds).assign(path="liquidity_baseline").to_csv(
        output / "liquidity_baseline_folds.csv", index=False, float_format="%.17g",
    )
    pd.DataFrame(meta_folds).assign(path="meta_veto").to_csv(
        output / "meta_veto_capital_folds.csv", index=False, float_format="%.17g",
    )
    aligned = pd.concat([
        baseline.predictions["strategy_equity"].rename("liquidity_baseline"),
        meta.predictions["strategy_equity"].rename("meta_veto"),
    ], axis=1)
    aligned.to_csv(output / "aligned_capital_curves.csv", float_format="%.17g")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(aligned.index, aligned["liquidity_baseline"], label="v10.8.44 Liquidity-Aware")
    ax.plot(aligned.index, aligned["meta_veto"], label="v10.8.47 Meta-Veto")
    ax.set_yscale("log")
    ax.set_title("Control Counterfactual Advantage Meta-Veto · research only")
    ax.set_ylabel("USD, log scale")
    ax.grid(alpha=.2)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output / "paired_capital.png", dpi=140)
    plt.close(fig)

    def metrics(run):
        result = {key: _number(run.metrics.get(key)) for key in PORTFOLIO_KEYS}
        result["terminal_holdings_asset"] = str(run.metrics.get("terminal_holdings_asset"))
        return result

    report = {
        "schema_version": 1,
        "research_kind": "control_counterfactual_advantage_meta_veto",
        "strategy_mode": MODE,
        "source_job_id": source_job_id,
        "source_validation_job_id": validation_job_id,
        "source_execution_job_id": execution_job_id,
        "source_liquidity_job_id": liquidity_job_id,
        "source_tcn_job_id": tcn_job_id,
        "source_ranking_job_id": ranking_job_id,
        "advantage_job_id": advantage_job_id,
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
        },
        "v1044_parity": {
            "status": "verified",
            "expected_ending_capital": liquidity_cap,
            "reproduced_ending_capital": reproduced,
            "absolute_difference": abs(reproduced - float(liquidity_cap)),
        },
        "training_folds": training_folds,
        "enabled_fold_count": enabled_folds,
        "veto_count": veto_count,
        "liquidity_baseline": metrics(baseline),
        "meta_veto_portfolio": metrics(meta),
        "method": {
            "role": "veto_only",
            "control_default_action": True,
            "target": "weighted_forward_return(candidate)-weighted_forward_return(incumbent)",
            "target_is_exact_delta_capital": False,
            "cash_actions_overridable": False,
            "oos_tuning": False,
        },
        "report_directory": str(output),
        "artifacts": [
            "summary.json",
            "liquidity_baseline_capital_curve.csv",
            "liquidity_baseline_fills.csv",
            "meta_veto_capital_curve.csv",
            "meta_veto_fills.csv",
            "meta_veto_training_folds.csv",
            "meta_veto_decisions.csv",
            "liquidity_baseline_folds.csv",
            "meta_veto_capital_folds.csv",
            "aligned_capital_curves.csv",
            "paired_capital.png",
        ],
        "order_eligible": False,
        "order_submission": "never",
        "source_download": "never",
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    if progress:
        progress("completed", 1, 1)
    return report
