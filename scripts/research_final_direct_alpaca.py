from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from market_cycle_trader_api.core.environment import load_project_environment

load_project_environment()

from market_cycle_trader_api.engine.capital_rotation import (
    _build_walk_forward_folds,
    _fold_performance,
    prepare_rotation_panel,
    run_rotation_models,
)
from market_cycle_trader_api.engine.compound_risk_overlay import (
    allocation_execution_enabled,
)
from market_cycle_trader_api.infrastructure.market_data.alpaca import (
    download_stock_bars,
)
from market_cycle_trader_api.infrastructure.market_data.alpaca_corporate_actions import (
    corporate_actions_for_symbol,
    corporate_actions_sha256,
    download_corporate_actions,
)
from market_cycle_trader_api.schemas.requests import (
    BacktestExecutionRequest,
    BacktestRequest,
)


DEFAULT_REQUEST = (
    ROOT / "config" / "final_research_alpaca_direct.json"
)
DEFAULT_OUTPUT = "output/final_research_alpaca_direct"
OHLCV = ("open", "high", "low", "close", "volume")


def _checkpoint(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )


def _utc(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return (
        stamp.tz_localize("UTC")
        if stamp.tzinfo is None
        else stamp.tz_convert("UTC")
    )


def _credentials_from_environment() -> dict[str, str]:
    load_project_environment()
    api_key_id = str(
        os.getenv("ALPACA_API_KEY_ID")
        or os.getenv("APCA_API_KEY_ID")
        or ""
    ).strip()
    secret_key = str(
        os.getenv("ALPACA_SECRET_KEY")
        or os.getenv("ALPACA_API_SECRET_KEY")
        or os.getenv("APCA_API_SECRET_KEY")
        or ""
    ).strip()
    if not api_key_id or not secret_key:
        raise RuntimeError(
            "Alpaca credentials must come from the environment. "
            "Set ALPACA_API_KEY_ID and ALPACA_SECRET_KEY in .env. "
            "This final research script does not read credentials from MongoDB."
        )
    return {
        "api_key_id": api_key_id,
        "secret_key": secret_key,
    }


def _round_fee_to_cent(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        return 0.0
    return math.ceil(
        (float(value) - 1e-12) * 100.0
    ) / 100.0


def _calculate_reference_fees(
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

    trade_value = quantity * price
    commission = _round_fee_to_cent(
        trade_value * config.commission_rate
    )
    cat = _round_fee_to_cent(
        quantity * config.cat_fee_per_share
    )
    sec = 0.0
    taf = 0.0
    if side.upper() == "SELL":
        sec = _round_fee_to_cent(
            trade_value * config.sec_fee_rate
        )
        taf = _round_fee_to_cent(
            min(
                quantity * config.taf_fee_per_share,
                config.taf_fee_cap,
            )
        )

    return {
        "commission_fee": commission,
        "sec_fee": sec,
        "taf_fee": taf,
        "cat_fee": cat,
        "total_fee": commission + sec + taf + cat,
    }


def _apply_slippage(
    price: float,
    side: str,
    config: BacktestRequest,
) -> float:
    adjustment = config.slippage_bps / 10_000.0
    return price * (
        1 + adjustment
        if side.upper() == "BUY"
        else 1 - adjustment
    )


def _split_normalize(
    raw: pd.DataFrame,
    actions: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    result = raw.copy()
    session_dates = (
        pd.DatetimeIndex(result.index)
        .tz_convert("UTC")
        .normalize()
    )
    applied: list[dict[str, Any]] = []

    split_actions = [
        action
        for action in actions
        if action.get("action_type")
        in {"forward_split", "reverse_split"}
        and action.get("ex_date")
        and action.get("old_rate") is not None
        and action.get("new_rate") is not None
    ]
    split_actions.sort(
        key=lambda item: str(item.get("ex_date"))
    )

    for action in split_actions:
        ex_date = _utc(
            action["ex_date"]
        ).normalize()
        old_rate = float(action["old_rate"])
        new_rate = float(action["new_rate"])
        if not (
            old_rate > 0
            and new_rate > 0
        ):
            continue

        price_factor = old_rate / new_rate
        volume_factor = new_rate / old_rate
        mask = session_dates < ex_date
        if not mask.any():
            continue

        for column in (
            "open",
            "high",
            "low",
            "close",
        ):
            result.loc[mask, column] = (
                result.loc[mask, column].astype(float)
                * price_factor
            )
        result.loc[mask, "volume"] = (
            result.loc[mask, "volume"].astype(float)
            * volume_factor
        )
        applied.append(
            {
                "action_type": action.get(
                    "action_type"
                ),
                "ex_date": str(
                    action.get("ex_date")
                ),
                "process_date": str(
                    action.get("process_date")
                ),
                "old_rate": old_rate,
                "new_rate": new_rate,
                "price_factor": price_factor,
                "volume_factor": volume_factor,
            }
        )

    return result, applied


def _structural_identity_issue(
    symbol: str,
    actions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    normalized = str(
        symbol
    ).strip().upper()

    for action in actions:
        action_type = str(
            action.get("action_type") or ""
        )
        if action_type not in {
            "stock_merger",
            "stock_and_cash_merger",
            "cash_merger",
        }:
            continue

        acquiree = str(
            action.get("acquiree_symbol") or ""
        ).strip().upper()
        acquirer = str(
            action.get("acquirer_symbol") or ""
        ).strip().upper()

        if (
            acquiree == normalized
            and acquirer
            and acquirer != normalized
        ):
            return {
                "reason": "structural_identity_change",
                "action_type": action_type,
                "process_date": action.get(
                    "process_date"
                ),
                "effective_date": action.get(
                    "effective_date"
                ),
                "acquiree_symbol": acquiree,
                "acquirer_symbol": acquirer,
            }

    return None


def _metrics(
    result: Any,
    folds: list[dict[str, Any]],
    initial_capital: float,
) -> dict[str, Any]:
    fold_rows = _fold_performance(
        result.predictions,
        folds,
        initial_capital,
    )
    worst_fold_return = (
        min(
            float(row["strategy_return"])
            for row in fold_rows
        )
        if fold_rows
        else None
    )
    predictive = deepcopy(
        result.metrics.get(
            "lightgbm_predictive_diagnostics"
        )
        or {}
    )
    simulation_profile = deepcopy(
        result.metrics.get(
            "simulation_profile"
        )
        or {}
    )

    strategy_ending = float(
        result.metrics.get(
            "strategy_ending_capital"
        )
        or 0.0
    )
    buy_hold_ending = float(
        result.metrics.get(
            "buy_hold_ending_capital"
        )
        or 0.0
    )
    strategy_return = float(
        result.metrics.get(
            "strategy_return"
        )
        or 0.0
    )
    buy_hold_return = float(
        result.metrics.get(
            "buy_hold_return"
        )
        or 0.0
    )
    strategy_cagr = float(
        result.metrics.get(
            "strategy_cagr"
        )
        or 0.0
    )
    buy_hold_cagr = float(
        result.metrics.get(
            "buy_hold_cagr"
        )
        or 0.0
    )
    strategy_sharpe = float(
        result.metrics.get(
            "strategy_sharpe"
        )
        or 0.0
    )
    buy_hold_sharpe = float(
        result.metrics.get(
            "buy_hold_sharpe"
        )
        or 0.0
    )
    strategy_maxdd = float(
        result.metrics.get(
            "strategy_maximum_drawdown"
        )
        or 0.0
    )
    buy_hold_maxdd = float(
        result.metrics.get(
            "buy_hold_maximum_drawdown"
        )
        or 0.0
    )

    metrics = {
        "ending_capital": strategy_ending,
        "sharpe": strategy_sharpe,
        "maximum_drawdown": strategy_maxdd,
        "cagr": strategy_cagr,
        "strategy_return": strategy_return,
        "buy_hold_ending_capital": buy_hold_ending,
        "buy_hold_return": buy_hold_return,
        "buy_hold_cagr": buy_hold_cagr,
        "buy_hold_sharpe": buy_hold_sharpe,
        "buy_hold_maximum_drawdown": buy_hold_maxdd,
        "benchmark_name": result.metrics.get(
            "benchmark_name"
        ),
        "strategy_vs_buy_hold_capital_ratio": (
            strategy_ending / buy_hold_ending
            if buy_hold_ending > 0
            else None
        ),
        "strategy_vs_buy_hold_excess_capital": (
            strategy_ending
            - buy_hold_ending
        ),
        "strategy_vs_buy_hold_excess_return": (
            strategy_return
            - buy_hold_return
        ),
        "strategy_vs_buy_hold_cagr_spread": (
            strategy_cagr
            - buy_hold_cagr
        ),
        "strategy_vs_buy_hold_sharpe_spread": (
            strategy_sharpe
            - buy_hold_sharpe
        ),
        "strategy_vs_buy_hold_drawdown_spread": (
            strategy_maxdd
            - buy_hold_maxdd
        ),
        "worst_fold_return": worst_fold_return,
        "folds": fold_rows,
        "requested_compute_device": result.metrics.get(
            "requested_compute_device"
        ),
        "effective_compute_device": result.metrics.get(
            "effective_compute_device"
        ),
        "compute_device_probe_errors": result.metrics.get(
            "compute_device_probe_errors"
        ),
        "predictive_diagnostics": predictive,
        "validation_mae": predictive.get(
            "validation_mae_mean"
        ),
        "validation_rmse": predictive.get(
            "validation_rmse_mean"
        ),
        "train_mae": predictive.get(
            "train_mae_mean"
        ),
        "train_rmse": predictive.get(
            "train_rmse_mean"
        ),
        "generalization_gap_rmse": predictive.get(
            "generalization_gap_rmse_mean"
        ),
        "best_iteration_mean": predictive.get(
            "best_iteration_mean"
        ),
        "early_stopping_model_fraction": predictive.get(
            "early_stopping_model_fraction_mean"
        ),
        "simulation_profile": simulation_profile,
        "simulation_total_seconds": (
            simulation_profile.get("total_seconds")
        ),
        "simulation_benchmark_seconds": (
            simulation_profile.get(
                "benchmark_seconds"
            )
        ),
        "simulation_market_regime_seconds": (
            simulation_profile.get(
                "market_regime_seconds"
            )
        ),
        "simulation_policy_seconds": (
            simulation_profile.get(
                "policy_seconds"
            )
        ),
        "simulation_accounting_seconds": (
            simulation_profile.get(
                "accounting_seconds"
            )
        ),
        "simulation_session_count": (
            simulation_profile.get(
                "session_count"
            )
        ),
        "eligible": True,
    }

    metrics.update(
        {
            key: value
            for key, value in result.metrics.items()
            if (
                str(key).startswith(
                    "soft_horizon_consensus_"
                )
                or str(key).startswith(
                    "oos_inference_cache_"
                )
            )
        }
    )
    return metrics


def _run(
    *,
    label: str,
    frames: dict[str, pd.DataFrame],
    config: BacktestExecutionRequest,
    folds: list[dict[str, Any]],
) -> tuple[Any, dict[str, Any]]:
    print(
        f"[final-research] starting {label}",
        flush=True,
    )
    results = run_rotation_models(
        frames,
        config,
        _calculate_reference_fees,
        _apply_slippage,
        progress_callback=lambda p, stage, completed: print(
            f"[final-research] {label} "
            f"progress={p:.1f}% "
            f"completed={completed} "
            f"stage={stage}",
            flush=True,
        ),
        technical_log_callback=lambda message: print(
            f"[technical] {label} {message}",
            flush=True,
        ),
    )
    if not results:
        raise RuntimeError(
            f"{label} returned no result."
        )

    result = results[0]
    metrics = _metrics(
        result,
        folds,
        float(config.initial_capital),
    )
    print(
        f"[final-research] completed {label} "
        f"capital={metrics['ending_capital']:,.2f} "
        f"buy_hold="
        f"{float(metrics.get('buy_hold_ending_capital') or 0.0):,.2f} "
        f"sharpe={metrics['sharpe']:.4f} "
        f"maxdd={metrics['maximum_drawdown']:.4%} "
        f"worst_fold="
        f"{float(metrics['worst_fold_return']):.4%}",
        flush=True,
    )
    return result, metrics


def _write_frame(
    frame: pd.DataFrame,
    path: Path,
) -> str:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    export = frame.reset_index()
    timestamp_column = str(
        export.columns[0]
    )
    if timestamp_column != "timestamp":
        export = export.rename(
            columns={
                timestamp_column: "timestamp"
            }
        )
    export["timestamp"] = (
        pd.to_datetime(
            export["timestamp"],
            utc=True,
        )
        .map(
            lambda value: (
                pd.Timestamp(value).isoformat()
            )
        )
    )
    export.to_csv(
        path,
        index=False,
        lineterminator="\n",
        float_format="%.15g",
    )
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def _request_hash(
    request_payload: dict[str, Any],
) -> str:
    canonical = json.dumps(
        request_payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(
        canonical
    ).hexdigest()


def _load_request(
    path: Path,
    end_date_override: str | None,
) -> tuple[
    BacktestExecutionRequest,
    dict[str, Any],
]:
    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )
    if end_date_override:
        normalized_end = date.fromisoformat(
            str(end_date_override)
        ).isoformat()
        payload["end_date"] = normalized_end
        payload["analysis_end_date"] = normalized_end

    payload["mongo_cache_enabled"] = False
    payload["market_data_provider"] = "alpaca"
    payload["alpaca_adjustment"] = "raw"
    payload[
        "market_data_history_backfill_enabled"
    ] = False
    payload[
        "market_data_history_backfill_provider"
    ] = "alpaca"

    request = (
        BacktestExecutionRequest.model_validate(
            payload
        )
    )
    if request.mongo_cache_enabled:
        raise RuntimeError(
            "Final research must run with "
            "mongo_cache_enabled=false."
        )
    return request, payload


def _prepare_output(
    output_dir: Path,
    *,
    replace_output: bool,
) -> None:
    if output_dir.exists():
        if not replace_output:
            raise RuntimeError(
                f"Output directory already exists: "
                f"{output_dir}. "
                "Use --replace-output to perform a new "
                "fresh Alpaca download."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


def _comparison_row(
    label: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "variant": label,
        "ending_capital": metrics.get(
            "ending_capital"
        ),
        "strategy_return": metrics.get(
            "strategy_return"
        ),
        "cagr": metrics.get("cagr"),
        "sharpe": metrics.get("sharpe"),
        "maximum_drawdown": metrics.get(
            "maximum_drawdown"
        ),
        "worst_fold_return": metrics.get(
            "worst_fold_return"
        ),
        "buy_hold_ending_capital": metrics.get(
            "buy_hold_ending_capital"
        ),
        "buy_hold_return": metrics.get(
            "buy_hold_return"
        ),
        "buy_hold_cagr": metrics.get(
            "buy_hold_cagr"
        ),
        "buy_hold_sharpe": metrics.get(
            "buy_hold_sharpe"
        ),
        "buy_hold_maximum_drawdown": metrics.get(
            "buy_hold_maximum_drawdown"
        ),
        "strategy_vs_buy_hold_capital_ratio": (
            metrics.get(
                "strategy_vs_buy_hold_capital_ratio"
            )
        ),
        "simulation_total_seconds": metrics.get(
            "simulation_total_seconds"
        ),
        "simulation_policy_seconds": metrics.get(
            "simulation_policy_seconds"
        ),
        "oos_inference_cache_build_seconds": (
            metrics.get(
                "oos_inference_cache_build_seconds"
            )
        ),
        "soft_horizon_consensus_changed_base_actions": (
            metrics.get(
                "soft_horizon_consensus_changed_base_actions"
            )
        ),
        "soft_horizon_consensus_change_rate": (
            metrics.get(
                "soft_horizon_consensus_change_rate"
            )
        ),
        "soft_horizon_consensus_average_support": (
            metrics.get(
                "soft_horizon_consensus_average_support"
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Final reproducibility run: download fresh RAW "
            "bars and corporate actions directly from Alpaca, "
            "use no MongoDB market/config data, reconstruct "
            "splits locally, then run Control vs Soft Horizon "
            "Consensus."
        )
    )
    parser.add_argument(
        "--request-json",
        default=str(DEFAULT_REQUEST),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help=(
            "Optional explicit closed research end date. "
            "Omit it to use the frozen date in the JSON config."
        ),
    )
    parser.add_argument(
        "--penalty-strength",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--corporate-actions-chunk-size",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--replace-output",
        action="store_true",
        help=(
            "Delete an existing output directory and perform "
            "a completely fresh Alpaca download."
        ),
    )
    args = parser.parse_args()

    if float(args.penalty_strength) < 0:
        raise ValueError(
            "--penalty-strength cannot be negative."
        )

    request_path = Path(
        args.request_json
    ).expanduser().resolve()
    if not request_path.is_file():
        raise FileNotFoundError(
            f"Request JSON not found: {request_path}"
        )

    request, request_payload = _load_request(
        request_path,
        args.end_date,
    )
    source_request_sha = _request_hash(
        request_payload
    )
    if allocation_execution_enabled(request):
        raise ValueError(
            "Final soft-consensus research is limited to "
            "the canonical single-position rotation policy."
        )

    output_dir = (
        ROOT / args.output_dir
    ).resolve()
    _prepare_output(
        output_dir,
        replace_output=bool(
            args.replace_output
        ),
    )

    snapshot_dir = output_dir / "snapshot"
    raw_dir = snapshot_dir / "raw"
    normalized_dir = (
        snapshot_dir / "split_normalized"
    )
    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    normalized_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    credentials = (
        _credentials_from_environment()
    )
    research_start = str(
        request.start_date
    )
    research_end = str(
        request.analysis_end_date
        or request.end_date
    )
    if not research_end:
        raise RuntimeError(
            "Final research requires a closed end date."
        )

    end_exclusive = (
        _utc(research_end).normalize()
        + pd.Timedelta(days=1)
    )
    corporate_actions_start = (
        date.fromisoformat(
            research_start
        )
        - timedelta(days=366)
    ).isoformat()

    print(
        "[final-research] database_access=NONE",
        flush=True,
    )
    print(
        "[final-research] "
        "market_data_source=ALPACA_DIRECT_RAW",
        flush=True,
    )
    print(
        f"[final-research] period="
        f"{research_start}..{research_end}",
        flush=True,
    )
    print(
        f"[final-research] feed="
        f"{request.alpaca_historical_feed}",
        flush=True,
    )

    corporate_actions = (
        download_corporate_actions(
            api_key_id=credentials[
                "api_key_id"
            ],
            secret_key=credentials[
                "secret_key"
            ],
            symbols=list(request.assets),
            start=corporate_actions_start,
            end=research_end,
            chunk_size=int(
                args.corporate_actions_chunk_size
            ),
            progress_callback=(
                lambda completed, total, chunk: print(
                    "[alpaca-ca] "
                    f"progress={completed}/{total} "
                    f"symbols={chunk[0]}..{chunk[-1]}",
                    flush=True,
                )
            ),
        )
    )
    corporate_actions_path = (
        snapshot_dir
        / "corporate_actions.json"
    )
    _checkpoint(
        corporate_actions_path,
        {
            "source": (
                "alpaca_corporate_actions_api"
            ),
            "query_start": (
                corporate_actions_start
            ),
            "query_end": research_end,
            "actions": corporate_actions,
        },
    )
    ca_sha = corporate_actions_sha256(
        corporate_actions
    )

    frames: dict[str, pd.DataFrame] = {}
    exclusions: list[dict[str, Any]] = []
    data_diagnostics: list[
        dict[str, Any]
    ] = []
    snapshot_entries: list[
        dict[str, Any]
    ] = []

    for position, symbol in enumerate(
        request.assets,
        start=1,
    ):
        print(
            f"[alpaca-bars] "
            f"{position}/{len(request.assets)} "
            f"{symbol} downloading RAW",
            flush=True,
        )
        raw = download_stock_bars(
            api_key_id=credentials[
                "api_key_id"
            ],
            secret_key=credentials[
                "secret_key"
            ],
            symbol=symbol,
            timeframe=request.timeframe,
            start=research_start,
            end=end_exclusive,
            feed=request.alpaca_historical_feed,
            adjustment="raw",
        )
        if raw.empty:
            raise RuntimeError(
                f"Fresh Alpaca download returned no "
                f"RAW bars for {symbol}."
            )

        raw = raw[
            [
                column
                for column in OHLCV
                if column in raw.columns
            ]
        ].copy()
        raw_path = (
            raw_dir / f"{symbol}.csv"
        )
        raw_sha = _write_frame(
            raw,
            raw_path,
        )

        actions = (
            corporate_actions_for_symbol(
                corporate_actions,
                symbol,
            )
        )
        issue = _structural_identity_issue(
            symbol,
            actions,
        )
        if issue is not None:
            exclusions.append(
                {
                    "symbol": symbol,
                    **issue,
                }
            )
            snapshot_entries.append(
                {
                    "symbol": symbol,
                    "raw_rows": int(len(raw)),
                    "raw_first_timestamp": (
                        pd.Timestamp(
                            raw.index.min()
                        ).isoformat()
                    ),
                    "raw_last_timestamp": (
                        pd.Timestamp(
                            raw.index.max()
                        ).isoformat()
                    ),
                    "raw_sha256": raw_sha,
                    "corporate_action_count": (
                        len(actions)
                    ),
                    "excluded": True,
                    "exclusion_reason": (
                        issue["reason"]
                    ),
                }
            )
            print(
                f"[alpaca-bars] {symbol} excluded "
                f"reason={issue['reason']}",
                flush=True,
            )
            continue

        normalized, applied = (
            _split_normalize(
                raw,
                actions,
            )
        )
        normalized_path = (
            normalized_dir
            / f"{symbol}.csv"
        )
        normalized_sha = _write_frame(
            normalized,
            normalized_path,
        )
        frames[symbol] = normalized

        row = {
            "symbol": symbol,
            "raw_rows": int(len(raw)),
            "raw_first_timestamp": (
                pd.Timestamp(
                    raw.index.min()
                ).isoformat()
            ),
            "raw_last_timestamp": (
                pd.Timestamp(
                    raw.index.max()
                ).isoformat()
            ),
            "raw_sha256": raw_sha,
            "split_normalized_sha256": (
                normalized_sha
            ),
            "corporate_action_count": (
                len(actions)
            ),
            "splits_applied": int(
                len(applied)
            ),
            "excluded": False,
        }
        snapshot_entries.append(row)
        data_diagnostics.append(
            {
                **row,
                "applied_splits": applied,
            }
        )
        print(
            f"[alpaca-bars] {symbol} "
            f"rows={len(raw)} "
            f"splits={len(applied)}",
            flush=True,
        )

    eligible = list(frames)
    if len(eligible) < 2:
        raise RuntimeError(
            "Fewer than two eligible assets remain "
            "after fresh Alpaca reconstruction."
        )

    anchors = [
        symbol
        for symbol
        in request.calendar_anchor_assets
        if symbol in frames
    ]
    if len(anchors) < 2:
        raise RuntimeError(
            "Fresh download left fewer than two "
            "calendar anchors."
        )

    references = [
        symbol
        for symbol
        in request.research_reference_assets
        if symbol in frames
    ]
    if len(references) < 2:
        references = list(anchors)

    reference_set = set(references)
    candidates = [
        symbol
        for symbol
        in request.research_candidate_assets
        if (
            symbol in frames
            and symbol not in reference_set
        )
    ]

    base_settings = deepcopy(
        request.research_model_settings
    )
    base_lightgbm = deepcopy(
        base_settings.get("lightgbm") or {}
    )
    base_lightgbm[
        "early_stopping_enabled"
    ] = False
    base_settings["lightgbm"] = (
        base_lightgbm
    )
    base_settings["horizon_voting"] = {
        "enabled": False,
    }
    base_settings[
        "soft_horizon_consensus"
    ] = {
        "enabled": False,
    }

    base_config = request.model_copy(
        update={
            "research_model_settings": (
                base_settings
            ),
            "assets": eligible,
            "calendar_anchor_assets": anchors,
            "research_reference_assets": (
                references
            ),
            "research_candidate_assets": (
                candidates
            ),
            "market_data_provider": "alpaca",
            "alpaca_adjustment": "raw",
            "market_data_history_backfill_enabled": False,
            "mongo_cache_enabled": False,
            "expected_market_data_signature_sha256": None,
            "research_market_data_snapshot_id": None,
        }
    )

    frozen_request_payload = (
        base_config.model_dump(
            mode="json"
        )
    )
    frozen_request_path = (
        output_dir
        / "frozen_request.json"
    )
    _checkpoint(
        frozen_request_path,
        frozen_request_payload,
    )
    effective_request_sha = _request_hash(
        frozen_request_payload
    )

    snapshot_signature_payload = {
        "provider": "alpaca",
        "feed": (
            request.alpaca_historical_feed
        ),
        "bars_adjustment": "raw",
        "research_start": research_start,
        "research_end": research_end,
        "corporate_actions_query_start": (
            corporate_actions_start
        ),
        "corporate_actions_sha256": (
            ca_sha
        ),
        "source_request_sha256": source_request_sha,
        "effective_request_sha256": effective_request_sha,
        "assets": snapshot_entries,
    }
    snapshot_id = _request_hash(
        snapshot_signature_payload
    )
    snapshot_manifest = {
        "schema_version": 1,
        "api_version": "10.8.75",
        "snapshot_id": snapshot_id,
        "downloaded_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "database_access": "none",
        "source": (
            "fresh_alpaca_direct_api"
        ),
        "provider": "alpaca",
        "feed": (
            request.alpaca_historical_feed
        ),
        "bars_adjustment": "raw",
        "split_normalization": (
            "local_point_in_time_corporate_actions"
        ),
        "research_start": research_start,
        "research_end": research_end,
        "corporate_actions_query_start": (
            corporate_actions_start
        ),
        "corporate_actions_query_end": (
            research_end
        ),
        "corporate_actions_count": (
            len(corporate_actions)
        ),
        "corporate_actions_sha256": (
            ca_sha
        ),
        "source_request_path": str(request_path),
        "source_request_sha256": source_request_sha,
        "effective_request_sha256": effective_request_sha,
        "configured_asset_count": (
            len(request.assets)
        ),
        "eligible_asset_count": (
            len(eligible)
        ),
        "eligible_assets": eligible,
        "excluded_assets": exclusions,
        "assets": snapshot_entries,
    }
    _checkpoint(
        snapshot_dir
        / "manifest.json",
        snapshot_manifest,
    )

    _, common_dates = (
        prepare_rotation_panel(
            frames,
            base_config,
        )
    )
    folds = _build_walk_forward_folds(
        common_dates,
        base_config,
    )

    control_result, control_metrics = _run(
        label="CONTROL",
        frames=frames,
        config=base_config,
        folds=folds,
    )

    challenger_settings = deepcopy(
        base_settings
    )
    challenger_settings[
        "soft_horizon_consensus"
    ] = {
        "enabled": True,
        "penalty_strength": float(
            args.penalty_strength
        ),
    }
    challenger_config = (
        base_config.model_copy(
            update={
                "research_model_settings": (
                    challenger_settings
                ),
            }
        )
    )
    challenger_result, challenger_metrics = (
        _run(
            label="SOFT_HORIZON_CONSENSUS",
            frames=frames,
            config=challenger_config,
            folds=folds,
        )
    )

    control_result.predictions.to_csv(
        output_dir
        / "control_predictions.csv",
        index=True,
    )
    control_result.trades.to_csv(
        output_dir
        / "control_trades.csv",
        index=False,
    )
    challenger_result.predictions.to_csv(
        output_dir
        / "soft_horizon_consensus_predictions.csv",
        index=True,
    )
    challenger_result.trades.to_csv(
        output_dir
        / "soft_horizon_consensus_trades.csv",
        index=False,
    )

    consensus_columns = [
        column
        for column
        in challenger_result.predictions.columns
        if str(column).startswith(
            "soft_horizon_consensus_"
        )
    ]
    if consensus_columns:
        challenger_result.predictions[
            consensus_columns
        ].to_csv(
            output_dir
            / "soft_horizon_consensus_decisions.csv",
            index=True,
        )

    pd.DataFrame(
        [
            _comparison_row(
                "CONTROL",
                control_metrics,
            ),
            _comparison_row(
                "SOFT_HORIZON_CONSENSUS",
                challenger_metrics,
            ),
        ]
    ).to_csv(
        output_dir
        / "strategy_comparison.csv",
        index=False,
    )

    diagnostics_export: list[
        dict[str, Any]
    ] = []
    for item in data_diagnostics:
        copied = dict(item)
        copied["applied_splits"] = json.dumps(
            copied.get(
                "applied_splits"
            )
            or [],
            sort_keys=True,
            default=str,
        )
        diagnostics_export.append(copied)

    pd.DataFrame(
        diagnostics_export
    ).to_csv(
        output_dir
        / "data_diagnostics.csv",
        index=False,
    )
    pd.DataFrame(
        exclusions
    ).to_csv(
        output_dir
        / "excluded_assets.csv",
        index=False,
    )

    control_capital = float(
        control_metrics["ending_capital"]
    )
    challenger_capital = float(
        challenger_metrics[
            "ending_capital"
        ]
    )
    summary = {
        "schema_version": 1,
        "api_version": "10.8.75",
        "experiment": (
            "final-direct-alpaca-no-db-v1"
        ),
        "snapshot_id": snapshot_id,
        "database_access": "none",
        "market_data": {
            "source": (
                "fresh_alpaca_direct_api"
            ),
            "bars_adjustment": "raw",
            "feed": (
                request.alpaca_historical_feed
            ),
            "local_split_normalization": True,
            "snapshot_manifest": (
                "snapshot/manifest.json"
            ),
            "raw_snapshot_directory": (
                "snapshot/raw"
            ),
            "normalized_snapshot_directory": (
                "snapshot/split_normalized"
            ),
            "corporate_actions_file": (
                "snapshot/corporate_actions.json"
            ),
        },
        "methodology": {
            "request_source": (
                "local frozen JSON file"
            ),
            "mongo_market_data_used": False,
            "mongo_configuration_used": False,
            "mongo_results_written": False,
            "credentials_source": (
                "environment_only"
            ),
            "control": (
                "canonical weighted multi-horizon "
                "LightGBM utility policy"
            ),
            "challenger": (
                "Control plus soft multi-horizon "
                "rank consensus"
            ),
            "penalty_strength": float(
                args.penalty_strength
            ),
            "same_frozen_download_for_both_variants": True,
            "buy_hold_benchmark_enabled": True,
            "batched_oos_inference": True,
        },
        "configured_assets": list(
            request.assets
        ),
        "eligible_assets": eligible,
        "excluded_assets": exclusions,
        "control": control_metrics,
        "soft_horizon_consensus": (
            challenger_metrics
        ),
        "comparison": {
            "capital_difference": (
                challenger_capital
                - control_capital
            ),
            "capital_ratio": (
                challenger_capital
                / control_capital
                if control_capital > 0
                else None
            ),
            "cagr_difference": (
                float(
                    challenger_metrics[
                        "cagr"
                    ]
                )
                - float(
                    control_metrics["cagr"]
                )
            ),
            "sharpe_difference": (
                float(
                    challenger_metrics[
                        "sharpe"
                    ]
                )
                - float(
                    control_metrics[
                        "sharpe"
                    ]
                )
            ),
            "maximum_drawdown_difference": (
                float(
                    challenger_metrics[
                        "maximum_drawdown"
                    ]
                )
                - float(
                    control_metrics[
                        "maximum_drawdown"
                    ]
                )
            ),
            "worst_fold_difference": (
                float(
                    challenger_metrics[
                        "worst_fold_return"
                    ]
                )
                - float(
                    control_metrics[
                        "worst_fold_return"
                    ]
                )
            ),
        },
    }
    _checkpoint(
        output_dir / "summary.json",
        summary,
    )

    print("")
    print(
        "[done] FINAL DIRECT ALPACA "
        "NO-DB RESEARCH COMPLETED",
        flush=True,
    )
    print(
        f"[done] snapshot_id={snapshot_id}",
        flush=True,
    )
    print(
        f"[done] CONTROL="
        f"{control_capital:,.2f}",
        flush=True,
    )
    print(
        f"[done] SOFT_HORIZON_CONSENSUS="
        f"{challenger_capital:,.2f}",
        flush=True,
    )
    print(
        f"[done] delta="
        f"{challenger_capital - control_capital:,.2f}",
        flush=True,
    )
    print(
        f"[done] output={output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
