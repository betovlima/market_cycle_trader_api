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



def emit_progress_detail(detail: dict[str, Any]) -> None:
    safe = {
        key: value
        for key, value in detail.items()
        if key in {
            "run_index",
            "run_count",
            "fold_index",
            "fold_count",
            "phase",
            "trained_models",
            "total_models",
            "device",
        }
    }
    print(
        "JOB_DETAIL|"
        + json.dumps(safe, ensure_ascii=False, separators=(",", ":"), default=str),
        flush=True,
    )


def emit_xgboost_technical(message: str) -> None:
    safe_message = str(message).replace("\n", " ").replace("\r", " ").strip()
    if safe_message:
        print(f"XGB_TECH|{safe_message}", flush=True)


def emit_research_technical(message: str) -> None:
    safe_message = str(message).replace("\n", " ").replace("\r", " ").strip()
    if safe_message:
        print(f"RESEARCH_TECH|{safe_message}", flush=True)


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
        "lightgbm_version",
        "lightgbm_settings_revision",
        "lightgbm_profile_id",
        "torch_version",
        "iqn_settings_revision",
        "iqn_profile_id",
        "strategy_configuration_sha256",
        "market_data_signature_sha256",
        "market_data_history_complete",
        "market_data_incomplete_assets",
        "market_data_backfilled_assets",
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
    calendar_anchor_assets = set(config.calendar_anchor_assets)
    for asset_position, symbol in enumerate(config.assets, start=1):
        emit_progress(
            3.0 + 12.0 * ((asset_position - 1) / total_assets),
            f"Loading market data {asset_position}/{total_assets} — {symbol}",
        )
        try:
            asset_config = (
                config
                if symbol in calendar_anchor_assets
                else config.model_copy(update={"market_data_require_complete_history": False})
            )
            raw = load_market_bars(symbol, asset_config)
            cleaned = validate_and_clean_bars(raw, asset_config)
            bars_by_symbol[symbol] = cleaned
            provenance = dict(cleaned.attrs.get("market_data_provenance", {}))
            first_session = pd.Timestamp(cleaned.index.min()).date().isoformat()
            last_session = pd.Timestamp(cleaned.index.max()).date().isoformat()
            backfill_rows = int(provenance.get("history_backfill_rows") or 0)
            source_label = str(
                provenance.get("effective_provider")
                or provenance.get("provider")
                or config.market_data_provider
            )
            access_path = str(provenance.get("research_access_path") or "mongodb_only")
            print(
                "MARKET_DATA|"
                f"{symbol}|rows={len(cleaned)}|start={first_session}|end={last_session}|"
                f"source={source_label}|access={access_path}|backfill_rows={backfill_rows}|"
                f"complete={bool(provenance.get('history_complete', True))}",
                flush=True,
            )
            emit_progress(
                3.0 + 12.0 * (asset_position / total_assets),
                (
                    f"Loaded market data {asset_position}/{total_assets} — {symbol} "
                    f"({first_session} → {last_session}, {source_label})"
                ),
            )
        except Exception as exc:
            failures.append({"symbol": symbol, "backend": "data_load", "error": str(exc)})
            print(f"ERROR loading {symbol}: {exc}", file=sys.stderr, flush=True)
    missing_assets = [symbol for symbol in config.assets if symbol not in bars_by_symbol]
    if missing_assets:
        raise ValueError(
            "Research market data failed to load for the complete configured universe: "
            + ", ".join(missing_assets)
        )
    if len(bars_by_symbol) < 2:
        raise ValueError("Compound rotation needs at least two successfully loaded assets.")

    
    
    
    reproducibility = build_reproducibility_manifest(config, bars_by_symbol)
    expected_signature = str(
        getattr(config, "expected_market_data_signature_sha256", None) or ""
    ).strip().lower()
    actual_signature = str(reproducibility.get("market_data_signature_sha256") or "").strip().lower()
    if expected_signature and actual_signature != expected_signature:
        raise RuntimeError(
            f"MarketDataSignatureMismatch: expected {expected_signature}, got {actual_signature or 'missing'}"
        )

    emit_progress(17.0, "Building aligned daily panel and walk-forward folds")
    results = run_rotation_models(
        bars_by_symbol,
        config,
        calculate_reference_fees,
        apply_slippage,
        progress_callback=emit_progress,
        trade_callback=emit_trade,
        progress_detail_callback=emit_progress_detail,
        technical_log_callback=(
            emit_xgboost_technical
            if str(getattr(config, "research_model_family", "xgboost_utility")) == "xgboost_utility"
            else emit_research_technical
        ),
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
        result.summary += (
            f"Complete requested history: {reproducibility.get('market_data_history_complete')}\n"
        )
        result.summary += (
            "Backfilled assets: "
            f"{', '.join(reproducibility.get('market_data_backfilled_assets') or []) or 'none'}\n"
        )
        result.summary += f"Python: {reproducibility.get('python_version')}\n"
        model_family = str(getattr(config, "research_model_family", "xgboost_utility"))
        if model_family == "lightgbm_utility":
            result.summary += f"LightGBM: {reproducibility.get('lightgbm_version')}\n"
        elif model_family == "iqn":
            result.summary += f"PyTorch: {reproducibility.get('torch_version')}\n"
        else:
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
        model_family = str(getattr(config, "research_model_family", "xgboost_utility"))
        model_label = {
            "xgboost_utility": "XGBoost Utility",
            "lightgbm_utility": "LightGBM Utility",
            "iqn": "IQN",
        }.get(model_family, model_family)
        print(f"Research model: {model_label}", flush=True)
        print(
            f"Research execution: seed={config.random_state}, "
            f"repetitions={config.rotation_xgb_repetitions}, "
            f"deterministic={config.deterministic_execution}",
            flush=True,
        )
        print(
            f"Research market data: mode={getattr(config, 'research_market_data_mode', 'database_only')}; "
            f"cutoff={config.analysis_end_date or config.end_date or 'unresolved'}",
            flush=True,
        )
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
            "Champion walk-forward schedule and execution period are locked in MongoDB.",
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
