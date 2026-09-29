"""Offline, SHA-pinned TCN research job; no trading or source modification.

Reports predictive diagnostics separately from portfolio performance. The
LightGBM and liquidity-aware references are PREVIOUSLY VERIFIED jobs, not
newly re-executed in this experiment. This historical OOS is exploratory.
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
from ..tcc_v106_reference.execution import calculate_reference_fees, apply_slippage
from ..tcc_v106_reference import research_challengers as scientific
from .control_snapshot_validation import read_verified_control_snapshot
from .control_deep_learning_tcn import (
    MODE, FEATURES, TARGET, WINDOW, TRAIN_STRIDE, HIDDEN, DILATIONS, EPOCHS,
    PATIENCE, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, FIXED_SWITCH_MARGIN,
    run_tcn_challenger,
)
from .operational_control_contract import prepare_operational_control_panel
from .control_execution_sensitivity import _validate_folds

Progress = Callable[[str, int, int], None]
KEYS = (
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


def _float(x: Any) -> float | None:
    if x is None:
        return None
    value = float(x)
    return value if math.isfinite(value) else None


def _predictive_metrics(
    signals: pd.DataFrame, group: str,
) -> dict[str, Any]:
    valid = signals.loc[signals["realized_utility_if_mature"].notna()]
    if valid.empty:
        raise ValueError("No mature OOS targets for a diagnostic comparison.")
    a = valid["predicted_utility"].to_numpy(dtype=float)
    b = valid["realized_utility_if_mature"].to_numpy(dtype=float)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Non-finite mature OOS predictive result.")
    error = a-b
    correlation = pd.Series(a).corr(pd.Series(b), method="spearman")
    return {
        "group": group,
        "mature_target_rows": len(valid),
        "unmatured_target_rows": len(signals)-len(valid),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "sign_accuracy": float(np.mean(np.sign(a)==np.sign(b))),
        "zero_prediction_mae": float(np.mean(np.abs(b))),
        "spearman": float(correlation) if pd.notna(correlation) else None,
    }


def run_control_tcn_research(
    *,
    source_job_id: str, validation_job_id: str, execution_job_id: str,
    liquidity_job_id: str, research_job_id: str,
    expected_sha256: str,
    baseline41: dict[str, Any], baseline42: dict[str, Any],
    baseline44: dict[str, Any],
    progress: Progress | None = None,
    snapshot_root: Path | None = None,
) -> dict[str, Any]:
    if not re.fullmatch(r"control-tcn-[a-f0-9]{16}", research_job_id):
        raise ValueError("Expected exact server-generated TCN research job ID.")
    if progress:
        progress("verify_immutable_snapshot", 0, 1)
    bars, manifest, directory = read_verified_control_snapshot(
        source_job_id, snapshot_root=snapshot_root, expected_sha256=expected_sha256,
    )
    if (
        len(bars)!=55
        or any(
            source.get("source_snapshot_sha256") != expected_sha256
            or source.get("source_job_id") != source_job_id
            or source.get("source_unchanged") is not True
            or source.get("order_submission") != "never"
            for source in (baseline41, baseline42, baseline44)
        )
        or not (baseline41.get("original_shadow") or {}).get("reproduced")
        or any((source.get("numeric_input_integrity") or {}).get("status")!="verified"
               for source in (baseline41, baseline42, baseline44))
        or int((baseline41.get("numeric_input_integrity") or {}).get("checked_assets",-1))!=55
        or int((baseline42.get("numeric_input_integrity") or {}).get("assets",-1))!=55
        or int((baseline44.get("numeric_input_integrity") or {}).get("assets",-1))!=55
        or baseline42.get("source_validation_job_id") != validation_job_id
        or baseline44.get("source_validation_job_id") != validation_job_id
        or baseline44.get("source_execution_job_id") != execution_job_id
        or baseline44.get("control_reference_parity") != "verified"
        or baseline44.get("research_kind") != "control_liquidity_aware_fixed_hypothesis"
        or len(baseline44.get("fold_comparison") or []) != 6
        or int((baseline41.get("oos") or {}).get("walk_forward_fold_count",-1))!=3
    ):
        raise ValueError("Missing exact SHA-matched v10.8.41/42/44 Control research contract.")
    control_cap = _float((baseline42.get("feasible_oos") or {}).get("strategy_ending_capital"))
    prior_control = _float((baseline44.get("control_reference") or {}).get("strategy_ending_capital"))
    liquidity_cap = _float((baseline44.get("liquidity_aware") or {}).get("strategy_ending_capital"))
    if (
        control_cap is None or prior_control is None or liquidity_cap is None
        or not math.isclose(control_cap,prior_control,rel_tol=0,abs_tol=1e-6)
    ):
        raise ValueError("Historical Control capital is not reproducibly pinned.")
    cutoff = str(manifest["completed_session"])
    inclusive = (pd.Timestamp(cutoff, tz="UTC") + pd.Timedelta(days=1)
                 - pd.Timedelta(seconds=1)).isoformat()
    config = build_control_config(CONFIG).model_copy(update={
        "analysis_end_date": inclusive, "end_date": cutoff,
    })
    _, _, audit = prepare_operational_control_panel(
        bars, completed_session=cutoff, config=config,
    )
    if int(baseline41.get("calendar_sessions",-1)) != audit.calendar_sessions:
        raise ValueError("Frozen original calendar differs from model context.")
    (frames, common_dates, symbols, folds, decision_dates,
     decision_to_fold, decision_metadata) = scientific._build_execution_context(
        bars, config,
    )
    if len(frames)!=55 or len(symbols)!=55 or len(folds)!=3 or len(decision_dates)!=1555:
        raise ValueError("Expected 55 assets and exactly 1554 OOS sessions.")
    original_folds = (baseline41.get("oos") or {}).get("walk_forward_folds") or []
    if len(original_folds)!=3:
        raise ValueError("Missing original fold calendar.")
    for prior, fold in zip(original_folds,folds):
        if (
            int(prior["fold_id"]) != int(fold["fold_id"])
            or pd.Timestamp(prior["test_start"]).date()!=pd.Timestamp(fold["test_start"]).date()
            or pd.Timestamp(prior["test_end"]).date()!=pd.Timestamp(fold["test_end"]).date()
        ):
            raise ValueError("TCN chronological fold does not match source Control fold.")
    if progress:
        progress("training_only_tcn", 0, 1)

    def stage(frac: float, label: str):
        if progress:
            progress(label[:140],int(frac*100),100)
    run, signals, training, account_audit = run_tcn_challenger(
        frames, common_dates, symbols, folds, decision_dates, decision_to_fold,
        decision_metadata, config, calculate_reference_fees, apply_slippage,
        progress=stage,
    )
    if not run.predictions.index.equals(pd.DatetimeIndex(decision_dates[1:])):
        raise ValueError("TCN OOS calendar differs from source Control.")
    if len(account_audit)!=1554 or len(signals)!=1554*55:
        raise ValueError("Incomplete TCN prior-close or per-asset OOS audit.")
    if (signals.groupby("decision_date")["asset"].nunique()!=55).any():
        raise ValueError("Incomplete candidate coverage in temporal TCN inference.")
    folds_report = _validate_folds(
        run.predictions, original_folds, float(config.initial_capital),
    )
    if not math.isclose(
        folds_report[-1]["strategy_ending_capital"],
        float(run.metrics["strategy_ending_capital"]),
        rel_tol=0, abs_tol=1e-6,
    ):
        raise ValueError("TCN ending capital/fold accounting mismatch.")
    for row in run.trades.to_dict("records"):
        if (
            int(row["quantity"]) > int(row["maximum_fill_quantity"])
            or float(row["realized_volume_participation"]) > .10+1e-10
        ):
            raise ValueError("TCN broke original execution capacity constraint.")
    diagnostics = [_predictive_metrics(signals,"all_mature_oos")]
    for fid in (1,2,3):
        diagnostics.append(_predictive_metrics(
            signals.loc[signals["fold_id"]==fid],f"fold_{fid}",
        ))
    output = directory / "validation" / "v10.8.45" / research_job_id
    if output.exists():
        raise FileExistsError("Never overwrite a completed deep-learning research result.")
    output.mkdir(parents=True)
    equity = run.predictions.copy()
    equity["drawdown"] = equity["strategy_equity"]/equity["strategy_equity"].cummax()-1
    equity.to_csv(output/"tcn_capital_curve.csv",float_format="%.17g")
    run.trades.to_csv(output/"tcn_fills.csv",index=False,float_format="%.17g")
    signals.to_csv(output/"tcn_oos_scores.csv",index=False,float_format="%.17g")
    pd.DataFrame(training).to_csv(output/"tcn_training_folds.csv",index=False)
    pd.DataFrame(folds_report).to_csv(output/"tcn_capital_folds.csv",index=False)
    pd.DataFrame(diagnostics).to_csv(output/"tcn_predictive_metrics.csv",index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(equity.index,equity["strategy_equity"],label="TinyTCN liquidity-aware")
    ax.set_title("Experimental TCN OOS · 10% constrained execution (no orders)")
    ax.set_yscale("log")
    ax.set_ylabel("USD, log scale, marked-to-market")
    ax.grid(alpha=.2)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output/"tcn_capital.png",dpi=140)
    plt.close(fig)
    report = {
        "schema_version":1,
        "research_kind":"control_tcn_fixed_temporal_baseline",
        "strategy_mode":MODE,
        "source_job_id":source_job_id,
        "source_validation_job_id":validation_job_id,
        "source_execution_job_id":execution_job_id,
        "source_liquidity_job_id":liquidity_job_id,
        "research_job_id":research_job_id,
        "source_snapshot_sha256":expected_sha256,
        "source_unchanged":True,
        "numeric_input_integrity":{"status":"verified","assets":55},
        "completed_session":cutoff,
        "source_baselines_previously_verified_not_retrained_in_this_job":{
            "control_cap10_cost_usd":control_cap,
            "liquidity_aware_v1044_usd":liquidity_cap,
        },
        "architecture":{
            "model":"pooled_cpu_temporal_convolutional_network",
            "window":WINDOW,"train_stride":TRAIN_STRIDE,"features":list(FEATURES),
            "target":TARGET,"hidden_channels":HIDDEN,
            "causal_dilations":list(DILATIONS),
            "epochs_max":EPOCHS,"patience":PATIENCE,
            "batch_size":BATCH_SIZE,"learning_rate":LEARNING_RATE,
            "weight_decay":WEIGHT_DECAY,
            "seed":42,"fixed_switch_margin":FIXED_SWITCH_MARGIN,
            "feature_scaler":"fit only on train segment for each fold",
            "calibration":"train-only chronological validation, purge at least target horizon",
            "final_training":"all labels must mature strictly before OOS first session",
            "execution_scenario":"identical to v10.8.42: cap10 plus hypothetical costs",
        },
        "training_fold_reports":training,
        "predictive_metrics":diagnostics,
        "tcn_portfolio":{
            **{key:_float(run.metrics.get(key)) for key in KEYS},
            "terminal_holdings_asset":str(run.metrics["terminal_holdings_asset"]),
            "walk_forward_fold_count":3,
            "walk_forward_folds":folds_report,
        },
        "capital_delta_vs_previous_liquidity_usd":(
            float(run.metrics["strategy_ending_capital"])-liquidity_cap
        ),
        "caveats":[
            "EXPLORATORY: this same historical OOS informed method design; no untouched independent confirmation and no automatic model selection or Winner promotion.",
            "The TCN input uses only current/past causal features. Future 60-session utility is used solely as a label with strict maturity in training, or retrospective diagnostic on OOS.",
            "The frozen LightGBM source has an inherited next-session availability filter; it is not invoked to generate TCN scores. Original baselines are historical reports, not retrained here.",
            "Calibration chooses epochs only from pre-OOS validation with purge. Final training and standardization only use past data and mature labels.",
            "The liquidity overlay is unchanged from v10.8.44, fixed 10% prior-known capacity, with 10% execution-day volume as ex-post cap only.",
            "Original full spread/impact and fee assumptions are hypothetical; daily OHLCV does not prove broker or auction fills; ending holdings are marked and not liquidated.",
            "One pooled network per fold with shared weights over 55 assets; not 55 independent deep nets. Fixed margin 0.0025, not OOS optimized.",
            "No source overwrite, Alpaca data download, live order, TCC change, operational strategy_mode, or Winner promotion.",
        ],
        "report_directory":str(output),
        "artifacts":[
            "summary.json","tcn_capital_curve.csv","tcn_fills.csv",
            "tcn_oos_scores.csv","tcn_training_folds.csv",
            "tcn_capital_folds.csv","tcn_predictive_metrics.csv",
            "tcn_capital.png",
        ],
        "source_download":"never","order_submission":"never","order_eligible":False,
    }
    (output/"summary.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False,default=str)+"\n",
        encoding="utf-8",
    )
    if progress: progress("completed",1,1)
    return report
