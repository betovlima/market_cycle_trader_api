"""SHA-pinned pairwise Deep Ranking experiment for MCT Control research."""
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
from .control_deep_ranking import (
    MODE, TARGET, HORIZON, TRAIN_DATE_STRIDE, MIN_ASSETS_PER_DATE,
    MAX_EPOCHS, PATIENCE, LEARNING_RATE, WEIGHT_DECAY,
    run_deep_rank_hybrid,
)
from .control_snapshot_validation import read_verified_control_snapshot
from .control_execution_sensitivity import _validate_folds
from .operational_control_contract import prepare_operational_control_panel

Progress = Callable[[str, int, int], None]
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


def _ranking_metrics(frame: pd.DataFrame, label: str) -> dict[str, Any]:
    mature = frame.loc[frame["realized_utility_if_mature"].notna()].copy()
    if mature.empty:
        raise ValueError("No mature Deep Ranking OOS labels.")
    daily_spearman = []
    pair_correct = 0
    pair_total = 0
    top_percentiles = []
    top_realized = []
    for _, group in mature.groupby("decision_date", sort=True):
        if len(group) < 2:
            continue
        scores = group["rank_score"].to_numpy(dtype=float)
        targets = group["realized_utility_if_mature"].to_numpy(dtype=float)
        corr = pd.Series(scores).corr(pd.Series(targets), method="spearman")
        if pd.notna(corr):
            daily_spearman.append(float(corr))
        dy = targets[:, None] - targets[None, :]
        ds = scores[:, None] - scores[None, :]
        upper = np.triu(np.ones_like(dy, dtype=bool), 1)
        valid = upper & (dy != 0)
        pair_total += int(valid.sum())
        pair_correct += int(
            (np.sign(dy[valid]) == np.sign(ds[valid])).sum()
        )
        winner = int(np.argmax(scores))
        realized_rank = (
            pd.Series(targets).rank(method="average", pct=True).iloc[winner]
        )
        top_percentiles.append(float(realized_rank))
        top_realized.append(float(targets[winner]))
    return {
        "group": label,
        "mature_rows": len(mature),
        "mature_dates": int(mature["decision_date"].nunique()),
        "pairwise_accuracy": pair_correct / max(pair_total, 1),
        "pair_count": pair_total,
        "mean_daily_spearman": (
            float(np.mean(daily_spearman)) if daily_spearman else None
        ),
        "median_daily_spearman": (
            float(np.median(daily_spearman)) if daily_spearman else None
        ),
        "mean_predicted_top_realized_percentile": (
            float(np.mean(top_percentiles)) if top_percentiles else None
        ),
        "mean_predicted_top_realized_utility": (
            float(np.mean(top_realized)) if top_realized else None
        ),
    }


def run_deep_ranking_research(
    *,
    source_job_id: str,
    validation_job_id: str,
    execution_job_id: str,
    liquidity_job_id: str,
    tcn_job_id: str,
    ranking_job_id: str,
    expected_sha256: str,
    baseline41: dict[str, Any],
    baseline42: dict[str, Any],
    baseline44: dict[str, Any],
    baseline45: dict[str, Any],
    progress: Progress | None = None,
    snapshot_root: Path | None = None,
) -> dict[str, Any]:
    if not re.fullmatch(r"control-rank-[a-f0-9]{16}", ranking_job_id):
        raise ValueError("Expected server-generated Deep Ranking job ID.")
    if progress:
        progress("verify_immutable_snapshot", 0, 1)
    bars, manifest, directory = read_verified_control_snapshot(
        source_job_id, snapshot_root=snapshot_root, expected_sha256=expected_sha256,
    )
    sources = (baseline41, baseline42, baseline44, baseline45)
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
        or baseline45.get("source_validation_job_id") != validation_job_id
        or baseline45.get("source_execution_job_id") != execution_job_id
        or baseline45.get("source_liquidity_job_id") != liquidity_job_id
        or baseline45.get("research_kind") != "control_tcn_fixed_temporal_baseline"
    ):
        raise ValueError("Deep Ranking requires exact v10.8.41/42/44/45 source chain.")
    scientific_cap = _number(
        (baseline41.get("oos") or {}).get("strategy_ending_capital")
    )
    control_exec = _number(
        (baseline42.get("feasible_oos") or {}).get("strategy_ending_capital")
    )
    liquidity_cap = _number(
        (baseline44.get("liquidity_aware") or {}).get("strategy_ending_capital")
    )
    tcn_cap = _number(
        (baseline45.get("tcn_portfolio") or {}).get("strategy_ending_capital")
    )
    if None in (scientific_cap, control_exec, liquidity_cap, tcn_cap):
        raise ValueError("Missing benchmark hierarchy capital.")
    if not math.isclose(
        control_exec,
        float((baseline44.get("control_reference") or {}).get(
            "strategy_ending_capital", float("nan")
        )),
        rel_tol=0, abs_tol=1e-6,
    ):
        raise ValueError("v10.8.44 reference no longer matches constrained Control.")
    cutoff = str(manifest["completed_session"])
    inclusive = (
        pd.Timestamp(cutoff, tz="UTC") + pd.Timedelta(days=1)
        - pd.Timedelta(seconds=1)
    ).isoformat()
    config = build_control_config(CONFIG).model_copy(update={
        "analysis_end_date": inclusive, "end_date": cutoff,
    })
    _, _, audit = prepare_operational_control_panel(
        bars, completed_session=cutoff, config=config,
    )
    if int(baseline41.get("calendar_sessions", -1)) != audit.calendar_sessions:
        raise ValueError("Deep Ranking canonical calendar differs from Control.")
    (
        _, common_dates, symbols, folds, all_decision_dates,
        _, _,
    ) = scientific._build_execution_context(bars, config)
    if len(symbols) != 55 or len(folds) != 3 or len(all_decision_dates) != 1555:
        raise ValueError("Expected original 55 assets / 3 folds / 1554 OOS sessions.")
    original_folds = (baseline41.get("oos") or {}).get("walk_forward_folds") or []
    for old, new in zip(original_folds, folds):
        if (
            int(old["fold_id"]) != int(new["fold_id"])
            or pd.Timestamp(old["test_start"]).date()
            != pd.Timestamp(new["test_start"]).date()
            or pd.Timestamp(old["test_end"]).date()
            != pd.Timestamp(new["test_end"]).date()
        ):
            raise ValueError("Deep Ranking fold calendar differs from frozen Control.")

    def on_progress(pct: float, stage: str, completed: int) -> None:
        if progress:
            progress(str(stage)[:150], max(0, min(100, int(pct))), 100)

    run, scores, training_reports, account_audit = run_deep_rank_hybrid(
        bars, config, calculate_reference_fees, apply_slippage,
        progress_callback=on_progress,
    )
    if len(run.predictions) != 1554 or len(account_audit) != 1554:
        raise ValueError("Deep Ranking portfolio/account audit is incomplete.")
    if len(scores) != 1554 * 55:
        raise ValueError("Deep Ranking must export 55 candidate scores per OOS date.")
    if (scores.groupby("decision_date")["asset"].nunique() != 55).any():
        raise ValueError("Deep Ranking candidate coverage incomplete.")
    expected_index = pd.DatetimeIndex(all_decision_dates[1:])
    if not run.predictions.index.equals(expected_index):
        raise ValueError("Deep Ranking OOS dates differ from Control.")
    fold_report = _validate_folds(
        run.predictions, original_folds, float(config.initial_capital),
    )
    if not math.isclose(
        float(run.metrics["strategy_ending_capital"]),
        float(fold_report[-1]["strategy_ending_capital"]),
        rel_tol=0, abs_tol=1e-6,
    ):
        raise ValueError("Deep Ranking capital/fold accounting mismatch.")
    if (run.predictions["cash"] < -1e-8).any():
        raise ValueError("Deep Ranking generated negative CASH.")
    if not run.trades.empty:
        quantity = pd.to_numeric(run.trades["quantity"], errors="coerce")
        maximum = pd.to_numeric(
            run.trades["maximum_fill_quantity"], errors="coerce",
        )
        participation = pd.to_numeric(
            run.trades["realized_volume_participation"], errors="coerce",
        )
        if (quantity > maximum).any() or (participation > .10 + 1e-10).any():
            raise ValueError("Deep Ranking violated fixed execution capacity.")

    predictive = [_ranking_metrics(scores, "all_mature_oos")]
    for fold_id in (1, 2, 3):
        predictive.append(_ranking_metrics(
            scores.loc[scores["fold_id"] == fold_id],
            f"fold_{fold_id}",
        ))

    output = directory / "validation" / "v10.8.46" / ranking_job_id
    if output.exists():
        raise FileExistsError("Never overwrite Deep Ranking research artifacts.")
    output.mkdir(parents=True)
    curve = run.predictions.copy()
    curve["drawdown"] = (
        curve["strategy_equity"]
        / curve["strategy_equity"].cummax().clip(lower=1e-12) - 1
    )
    curve.to_csv(output / "deep_rank_capital_curve.csv", float_format="%.17g")
    run.trades.to_csv(
        output / "deep_rank_fills.csv", index=False, float_format="%.17g",
    )
    scores.to_csv(
        output / "deep_rank_oos_scores.csv", index=False, float_format="%.17g",
    )
    pd.DataFrame(training_reports).to_csv(
        output / "deep_rank_training_folds.csv", index=False,
    )
    pd.DataFrame(fold_report).to_csv(
        output / "deep_rank_capital_folds.csv", index=False,
        float_format="%.17g",
    )
    pd.DataFrame(predictive).to_csv(
        output / "deep_rank_predictive_metrics.csv", index=False,
        float_format="%.17g",
    )
    curve[[
        "decision_date", "signal_asset", "previous_asset", "selected_asset",
        "cash", "shares", "cash_weight", "requested_quantity",
        "executed_quantity", "unfilled_quantity", "trade_action",
        "trade_reason", "walk_forward_fold",
    ]].to_csv(
        output / "deep_rank_decisions.csv", float_format="%.17g",
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(curve.index, curve["strategy_equity"], label="Deep Pairwise Rank + Control utility")
    ax.set_yscale("log")
    ax.set_title("Deep Ranking · constrained OOS marked equity (research only)")
    ax.set_ylabel("USD, log scale")
    ax.grid(alpha=.2)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output / "deep_rank_capital.png", dpi=140)
    plt.close(fig)

    metrics = {
        key: _number(run.metrics.get(key)) for key in PORTFOLIO_KEYS
    }
    metrics["terminal_holdings_asset"] = str(
        run.metrics["terminal_holdings_asset"]
    )
    report = {
        "schema_version": 1,
        "research_kind": "control_deep_pairwise_rank_hybrid",
        "strategy_mode": MODE,
        "source_job_id": source_job_id,
        "source_validation_job_id": validation_job_id,
        "source_execution_job_id": execution_job_id,
        "source_liquidity_job_id": liquidity_job_id,
        "source_tcn_job_id": tcn_job_id,
        "ranking_job_id": ranking_job_id,
        "source_snapshot_sha256": expected_sha256,
        "source_unchanged": True,
        "numeric_input_integrity": {"status": "verified", "assets": 55},
        "completed_session": cutoff,
        "benchmark_hierarchy": {
            "scientific_control_usd": scientific_cap,
            "execution_constrained_control_usd": control_exec,
            "liquidity_aware_v1044_usd": liquidity_cap,
            "failed_tiny_tcn_v1045_usd": tcn_cap,
        },
        "architecture": {
            "temporal_encoder": "same small causal TinyTCN encoder family",
            "learning_objective": "all-pairs within-date logistic ranking",
            "absolute_utility_source": "original fold-specific LightGBM Control",
            "candidate_order_source": "Deep Pairwise Ranker",
            "liquidity_overlay": "unchanged v10.8.44 prior-close capacity",
            "target_for_ordering": TARGET,
            "target_horizon_max_sessions": HORIZON,
            "train_date_stride": TRAIN_DATE_STRIDE,
            "minimum_assets_per_training_date": MIN_ASSETS_PER_DATE,
            "epochs_max": MAX_EPOCHS,
            "patience": PATIENCE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "feature_scaler": "training dates only per fold",
            "cash_threshold_and_switch_margin": "original LightGBM Control scale",
        },
        "training_fold_reports": training_reports,
        "predictive_metrics": predictive,
        "deep_rank_portfolio": {
            **metrics,
            "walk_forward_fold_count": 3,
            "walk_forward_folds": fold_report,
        },
        "capital_delta_vs_scientific_control_usd": (
            metrics["strategy_ending_capital"] - scientific_cap
        ),
        "capital_delta_vs_execution_control_usd": (
            metrics["strategy_ending_capital"] - control_exec
        ),
        "capital_delta_vs_liquidity_v1044_usd": (
            metrics["strategy_ending_capital"] - liquidity_cap
        ),
        "caveats": [
            "Historical OOS is exploratory because v10.8.45 failure motivated this ranking objective; it is not untouched confirmation.",
            "The scientific Control USD 5.89M remains the primary model benchmark; USD 528.7k is its execution-constrained scenario, not a replacement benchmark.",
            "Deep Learning only orders candidates. Absolute utility, CASH threshold and switch margin remain anchored to original LightGBM Control after the fixed liquidity overlay.",
            "Pairwise training uses only labels whose complete 60-session horizon matures before calibration/test boundaries; no OOS capital tunes architecture.",
            "Same 10% execution scenario, assumed spread/impact and whole-share accounting as v10.8.42/v10.8.44; daily OHLCV does not prove real broker fills.",
            "No Alpaca download, source overwrite, operational strategy registration, TCC edit, Winner promotion or real order.",
        ],
        "report_directory": str(output),
        "artifacts": [
            "summary.json", "deep_rank_capital_curve.csv",
            "deep_rank_fills.csv", "deep_rank_oos_scores.csv",
            "deep_rank_training_folds.csv", "deep_rank_capital_folds.csv",
            "deep_rank_predictive_metrics.csv", "deep_rank_decisions.csv",
            "deep_rank_capital.png",
        ],
        "source_download": "never",
        "order_submission": "never",
        "order_eligible": False,
    }
    (output / "summary.json").write_text(
        json.dumps(
            report, ensure_ascii=False, allow_nan=False, indent=2, default=str,
        ) + "\n",
        encoding="utf-8",
    )
    if progress:
        progress("completed", 1, 1)
    return report
