from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from ..core.environment import load_project_environment

# The backtest engine runs in a separate Python process. Load the API .env in
# this process as well instead of relying only on environment inheritance from
# Uvicorn. Real system/Railway variables keep priority because override=False.
load_project_environment()

from ..infrastructure.persistence.mongo_repository import (
    JOBS_COLLECTION,
    bson_value,
    create_client,
    ensure_database,
    get_database,
    replace_comparison,
    replace_run_result,
)
from ..schemas.requests import BacktestExecutionRequest, BacktestRequest
from ..services.reproducibility import build_reproducibility_manifest
from .capital_rotation import run_rotation_models
from .market_data import load_market_bars, validate_and_clean_bars


def configure_console_utf8() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def round_fee_to_cent(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        return 0.0
    return math.ceil((value - 1e-12) * 100.0) / 100.0


def calculate_reference_fees(
    side: str,
    quantity: float,
    price: float,
    config: BacktestRequest,
) -> dict[str, float]:
    if quantity <= 0 or price <= 0:
        return {
            "commission_fee": 0.0,
            "sec_fee": 0.0,
            "taf_fee": 0.0,
            "cat_fee": 0.0,
            "total_fee": 0.0,
        }
    normalized_side = side.upper()
    trade_value = quantity * price
    commission = round_fee_to_cent(trade_value * config.commission_rate)
    cat = round_fee_to_cent(quantity * config.cat_fee_per_share)
    sec = 0.0
    taf = 0.0
    if normalized_side == "SELL":
        sec = round_fee_to_cent(trade_value * config.sec_fee_rate)
        taf = round_fee_to_cent(min(quantity * config.taf_fee_per_share, config.taf_fee_cap))
    elif normalized_side != "BUY":
        raise ValueError(f"Unsupported side: {side}")
    return {
        "commission_fee": commission,
        "sec_fee": sec,
        "taf_fee": taf,
        "cat_fee": cat,
        "total_fee": commission + sec + taf + cat,
    }


def apply_slippage(price: float, side: str, config: BacktestRequest) -> float:
    adjustment = config.slippage_bps / 10_000
    return price * (1 + adjustment if side == "BUY" else 1 - adjustment)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    return parser.parse_args()


def load_config(db: Any, job_id: str) -> BacktestExecutionRequest:
    job = db[JOBS_COLLECTION].find_one({"id": job_id}, {"_id": 0, "request": 1})
    if job is None:
        raise ValueError(f"Backtest job not found: {job_id}")
    return BacktestExecutionRequest.model_validate(job.get("request") or {})


def emit_progress(percent: float, stage: str, completed_runs: int = 0) -> None:
    safe_stage = str(stage).replace("|", "/").strip()
    print(f"JOB_PROGRESS|{float(percent):.1f}|{int(completed_runs)}|{safe_stage}", flush=True)


def emit_trade(trade: dict[str, Any]) -> None:
    normalized = bson_value(trade)
    for key in ("timestamp", "entry_timestamp"):
        value = normalized.get(key)
        if isinstance(value, (datetime, pd.Timestamp)):
            normalized[key] = pd.Timestamp(value).isoformat()
    print(
        "JOB_TRADE|"
        + json.dumps(normalized, ensure_ascii=False, separators=(",", ":"), default=str),
        flush=True,
    )


def flatten_rotation_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "backend",
        "model_family",
        "random_seed",
        "repetition_index",
        "repetition_count",
        "strategy_mode",
        "strategy_label",
        "assets",
        "decision_horizon_days",
        "decision_horizon_label",
        "benchmark_name",
        "strategy_ending_capital",
        "strategy_return",
        "buy_hold_ending_capital",
        "buy_hold_return",
        "excess_return",
        "strategy_maximum_drawdown",
        "buy_hold_maximum_drawdown",
        "strategy_sharpe",
        "buy_hold_sharpe",
        "strategy_cagr",
        "buy_hold_cagr",
        "compound_log_growth",
        "risk_adjusted_compound_score",
        "walk_forward_fold_count",
        "walk_forward_purge_days",
        "downside_penalty",
        "drawdown_penalty",
        "effective_switch_margin",
        "market_exposure",
        "cash_days",
        "simulated_buys",
        "simulated_sells",
        "capital_rotations",
        "cycles_per_year",
        "average_holding_days",
        "average_holding_bars",
        "average_holding_minutes",
        "overnight_positions_allowed",
        "intraday_rotations_allowed",
        "maximum_entries_per_session",
        "maximum_exits_per_session",
        "invested_sessions",
        "winning_sessions",
        "session_win_rate",
        "reference_benchmark_name",
        "reference_buy_hold_ending_capital",
        "reference_buy_hold_return",
        "reference_buy_hold_maximum_drawdown",
        "reference_buy_hold_sharpe",
        "reference_buy_hold_cagr",
        "geometric_trade_return",
        "total_transaction_fees",
        "turnover_ratio",
        "effective_compute_device",
        "gpu_name",
        "deterministic_execution",
        "numeric_thread_limit",
        "xgb_n_jobs",
        "strategy_configuration_sha256",
        "market_data_signature_sha256",
        "python_version",
        "xgboost_version",
        "scikit_learn_version",
        "numpy_version",
        "pandas_version",
        "scipy_version",
        "threadpoolctl_version",
    )
    row = {key: metrics.get(key) for key in keys}
    row.update({"symbol": "PORTFOLIO", "portfolio_rotation": True})
    return row


def run_job(job_id: str, config: BacktestExecutionRequest, db: Any) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    bars_by_symbol: dict[str, pd.DataFrame] = {}
    failures: list[dict[str, str]] = []
    emit_progress(2.0, "Preparing shared-capital rotation")
    total_assets = max(1, len(config.assets))
    for asset_position, symbol in enumerate(config.assets, start=1):
        emit_progress(
            3.0 + 12.0 * ((asset_position - 1) / total_assets),
            f"Loading market data {asset_position}/{total_assets} — {symbol}",
        )
        try:
            raw = load_market_bars(symbol, config)
            bars_by_symbol[symbol] = validate_and_clean_bars(raw, config)
            emit_progress(
                3.0 + 12.0 * (asset_position / total_assets),
                f"Loaded market data {asset_position}/{total_assets} — {symbol}",
            )
        except Exception as exc:
            failures.append({"symbol": symbol, "backend": "data_load", "error": str(exc)})
            print(f"ERROR loading {symbol}: {exc}", file=sys.stderr, flush=True)
    if len(bars_by_symbol) < 2:
        raise ValueError("Compound rotation needs at least two successfully loaded assets.")
    reproducibility = build_reproducibility_manifest(config, bars_by_symbol)
    emit_progress(17.0, "Building aligned daily panel and walk-forward folds")
    results = run_rotation_models(
        bars_by_symbol,
        config,
        calculate_reference_fees,
        apply_slippage,
        progress_callback=emit_progress,
        trade_callback=emit_trade,
    )
    comparisons: list[dict[str, Any]] = []
    total_results = max(1, len(results))
    for result_position, result in enumerate(results, start=1):
        result.metrics.update(reproducibility)
        result.summary += "\n\nREPRODUCIBILITY\n"
        result.summary += (
            f"Configuration SHA-256: {reproducibility['strategy_configuration_sha256']}\n"
        )
        result.summary += (
            f"Market data SHA-256: {reproducibility['market_data_signature_sha256']}\n"
        )
        result.summary += f"Python: {reproducibility.get('python_version')}\n"
        result.summary += f"XGBoost: {reproducibility.get('xgboost_version')}\n"
        emit_progress(
            92.0 + 6.0 * ((result_position - 1) / total_results),
            f"Saving {result.metrics.get('strategy_label', result.backend)} results to MongoDB",
            len(results),
        )
        replace_run_result(
            db,
            job_id=job_id,
            symbol="PORTFOLIO",
            backend=result.backend,
            metrics=result.metrics,
            summary=result.summary,
            predictions=result.predictions,
            trades=result.trades,
            batch_size=config.mongo_write_batch_size,
        )
        comparisons.append(bson_value(flatten_rotation_metrics(result.metrics)))
        print(
            f"PORTFOLIO/{result.backend}: Strategy={result.metrics['strategy_return']:.2%} | "
            f"Benchmark={result.metrics['buy_hold_return']:.2%}",
            flush=True,
        )
    emit_progress(99.0, "Finalizing comparison and reports", len(results))
    return comparisons, failures


def main() -> None:
    configure_console_utf8()
    args = parse_args()
    client = create_client()
    try:
        db = get_database(client)
        ensure_database(db)
        config = load_config(db, args.job_id)
        print("Local simulation only. No broker order will be created.", flush=True)
        print(f"Assets: {', '.join(config.assets)}", flush=True)
        print(f"Strategy: {config.strategy_mode}", flush=True)
        print(f"Models: {', '.join(config.rotation_models)}", flush=True)
        print(
            f"Training history: {config.start_date} → "
            f"{config.effective_analysis_end_date or 'latest available session'}",
            flush=True,
        )
        print(
            f"Requested analysis window: {config.analysis_start_date} → "
            f"{config.analysis_end_date or 'latest available session'}",
            flush=True,
        )
        print(
            "Champion walk-forward schedule locked; the public date range "
            "changes only the simulated account window.",
            flush=True,
        )
        comparisons, failures = run_job(args.job_id, config, db)
        comparisons.sort(key=lambda item: str(item.get("backend", "")))
        replace_comparison(
            db,
            job_id=args.job_id,
            comparison=comparisons,
            failures=failures,
            effective_config=bson_value(config.model_dump(mode="python")),
        )
        if not comparisons:
            raise SystemExit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()
