from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import date, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import requests


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
from market_cycle_trader_api.schemas.requests import (
    BacktestExecutionRequest,
    BacktestRequest,
)


DEFAULT_CONFIG = ROOT / "research" / "final_research_v10_8_75.json"
DEFAULT_OUTPUT = ROOT / "output" / "final_fresh_alpaca_standalone"
CORPORATE_ACTIONS_ENDPOINT = "https://data.alpaca.markets/v1/corporate-actions"
OHLCV = ("open", "high", "low", "close", "volume")
REQUEST_TYPES = (
    "forward_split",
    "reverse_split",
    "unit_split",
    "cash_dividend",
    "stock_dividend",
    "spin_off",
    "cash_merger",
    "stock_merger",
    "stock_and_cash_merger",
    "redemption",
    "name_change",
    "worthless_removal",
    "rights_distribution",
)
ARRAY_TO_TYPE = {
    "forward_splits": "forward_split",
    "reverse_splits": "reverse_split",
    "unit_splits": "unit_split",
    "cash_dividends": "cash_dividend",
    "stock_dividends": "stock_dividend",
    "spin_offs": "spin_off",
    "cash_mergers": "cash_merger",
    "stock_mergers": "stock_merger",
    "stock_and_cash_mergers": "stock_and_cash_merger",
    "redemptions": "redemption",
    "name_changes": "name_change",
    "worthless_removals": "worthless_removal",
    "rights_distributions": "rights_distribution",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return (
        stamp.tz_localize("UTC")
        if stamp.tzinfo is None
        else stamp.tz_convert("UTC")
    )


def _environment_credentials() -> dict[str, str]:
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
            "Alpaca credentials are required only from the environment. "
            "Set ALPACA_API_KEY_ID and ALPACA_SECRET_KEY."
        )
    return {
        "api_key_id": api_key_id,
        "secret_key": secret_key,
    }


def _round_fee_to_cent(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        return 0.0
    return math.ceil((value - 1e-12) * 100.0) / 100.0


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


def _request_corporate_actions_page(
    *,
    headers: dict[str, str],
    symbols: list[str],
    start: str,
    end: str,
    page_token: str | None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "symbols": ",".join(symbols),
        "types": ",".join(REQUEST_TYPES),
        "start": start,
        "end": end,
        "region": "us",
        "data_quality": "complete",
        "limit": 1000,
        "sort": "asc",
    }
    if page_token:
        params["page_token"] = page_token

    last_error: Exception | None = None
    for attempt in range(4):
        try:
            response = requests.get(
                CORPORATE_ACTIONS_ENDPOINT,
                headers=headers,
                params=params,
                timeout=30,
            )
            if response.status_code == 429 and attempt < 3:
                time.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError(
                    "Unexpected Alpaca corporate-actions response."
                )
            return payload
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(2 ** attempt)
    raise RuntimeError(
        "Alpaca corporate-actions request failed: "
        f"{last_error}"
    ) from last_error


def _flatten_corporate_actions(
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    groups = payload.get("corporate_actions") or {}
    if not isinstance(groups, dict):
        return result
    for array_name, values in groups.items():
        if not isinstance(values, list):
            continue
        action_type = ARRAY_TO_TYPE.get(
            str(array_name),
            str(array_name),
        )
        for value in values:
            if not isinstance(value, dict):
                continue
            document = dict(value)
            document["action_type"] = action_type
            document["source_array"] = str(array_name)
            result.append(document)
    return result


def _download_corporate_actions(
    *,
    credentials: dict[str, str],
    symbols: list[str],
    research_start: str,
    research_end: str,
    lookback_days: int,
    chunk_size: int = 40,
) -> list[dict[str, Any]]:
    start = (
        date.fromisoformat(research_start)
        - timedelta(days=int(lookback_days))
    ).isoformat()
    headers = {
        "APCA-API-KEY-ID": credentials["api_key_id"],
        "APCA-API-SECRET-KEY": credentials["secret_key"],
    }
    all_actions: list[dict[str, Any]] = []
    size = max(1, min(100, int(chunk_size)))

    for offset in range(0, len(symbols), size):
        chunk = symbols[offset : offset + size]
        page_token: str | None = None
        while True:
            payload = _request_corporate_actions_page(
                headers=headers,
                symbols=chunk,
                start=start,
                end=research_end,
                page_token=page_token,
            )
            all_actions.extend(
                _flatten_corporate_actions(payload)
            )
            page_token = payload.get("next_page_token")
            if not page_token:
                break
        print(
            "[fresh-ca] "
            f"{min(offset + len(chunk), len(symbols))}/"
            f"{len(symbols)} symbols",
            flush=True,
        )

    unique: dict[str, dict[str, Any]] = {}
    for index, action in enumerate(all_actions):
        identifier = str(action.get("id") or "").strip()
        if not identifier:
            identifier = _sha256_bytes(
                json.dumps(
                    action,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            )
        unique[identifier] = action

    return sorted(
        unique.values(),
        key=lambda item: (
            str(item.get("process_date") or ""),
            str(item.get("ex_date") or ""),
            str(item.get("effective_date") or ""),
            str(item.get("action_type") or ""),
            str(item.get("id") or ""),
        ),
    )


def _actions_for_symbol(
    actions: list[dict[str, Any]],
    symbol: str,
) -> list[dict[str, Any]]:
    normalized = str(symbol).strip().upper()
    symbol_fields = (
        "symbol",
        "source_symbol",
        "old_symbol",
        "new_symbol",
        "acquirer_symbol",
        "acquiree_symbol",
    )
    selected = [
        action
        for action in actions
        if any(
            str(action.get(field) or "").strip().upper()
            == normalized
            for field in symbol_fields
        )
    ]
    return sorted(
        selected,
        key=lambda item: (
            str(item.get("process_date") or ""),
            str(item.get("ex_date") or ""),
            str(item.get("effective_date") or ""),
        ),
    )


def _structural_identity_issue(
    symbol: str,
    actions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    normalized = str(symbol).strip().upper()
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
                "process_date": action.get("process_date"),
                "effective_date": action.get(
                    "effective_date"
                ),
                "acquiree_symbol": acquiree,
                "acquirer_symbol": acquirer,
            }
    return None


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
        if not (old_rate > 0 and new_rate > 0):
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


def _frame_data_sha256(
    frame: pd.DataFrame,
) -> str:
    canonical = frame.to_csv(
        index=True,
        float_format="%.17g",
        lineterminator="\n",
    ).encode("utf-8")
    return _sha256_bytes(canonical)


def _write_frame_csv(
    frame: pd.DataFrame,
    path: Path,
) -> dict[str, str]:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    frame.to_csv(
        path,
        index=True,
        compression="gzip",
        float_format="%.17g",
        lineterminator="\n",
    )
    return {
        "file_sha256": _sha256_file(path),
        "data_sha256": _frame_data_sha256(frame),
    }


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
        result.metrics.get("simulation_profile")
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
        result.metrics.get("strategy_return")
        or 0.0
    )
    buy_hold_return = float(
        result.metrics.get("buy_hold_return")
        or 0.0
    )
    strategy_cagr = float(
        result.metrics.get("strategy_cagr")
        or 0.0
    )
    buy_hold_cagr = float(
        result.metrics.get("buy_hold_cagr")
        or 0.0
    )
    strategy_sharpe = float(
        result.metrics.get("strategy_sharpe")
        or 0.0
    )
    buy_hold_sharpe = float(
        result.metrics.get("buy_hold_sharpe")
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
            strategy_ending - buy_hold_ending
        ),
        "strategy_vs_buy_hold_excess_return": (
            strategy_return - buy_hold_return
        ),
        "strategy_vs_buy_hold_cagr_spread": (
            strategy_cagr - buy_hold_cagr
        ),
        "strategy_vs_buy_hold_sharpe_spread": (
            strategy_sharpe - buy_hold_sharpe
        ),
        "strategy_vs_buy_hold_drawdown_spread": (
            strategy_maxdd - buy_hold_maxdd
        ),
        "worst_fold_return": worst_fold_return,
        "folds": fold_rows,
        "requested_compute_device": (
            result.metrics.get(
                "requested_compute_device"
            )
        ),
        "effective_compute_device": (
            result.metrics.get(
                "effective_compute_device"
            )
        ),
        "compute_device_probe_errors": (
            result.metrics.get(
                "compute_device_probe_errors"
            )
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
        "early_stopping_model_fraction": (
            predictive.get(
                "early_stopping_model_fraction_mean"
            )
        ),
        "simulation_profile": simulation_profile,
        "simulation_total_seconds": (
            simulation_profile.get(
                "total_seconds"
            )
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


def _run_variant(
    *,
    label: str,
    frames: dict[str, pd.DataFrame],
    config: BacktestExecutionRequest,
    folds: list[dict[str, Any]],
) -> tuple[Any, dict[str, Any]]:
    print(
        f"[final-study] starting {label}",
        flush=True,
    )
    results = run_rotation_models(
        frames,
        config,
        _calculate_reference_fees,
        _apply_slippage,
        progress_callback=lambda p, stage, completed: print(
            f"[final-study] {label} "
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
        f"[final-study] completed {label} "
        f"capital={metrics['ending_capital']:,.2f} "
        f"buy_hold="
        f"{float(metrics['buy_hold_ending_capital']):,.2f} "
        f"sharpe={metrics['sharpe']:.4f} "
        f"maxdd={metrics['maximum_drawdown']:.4%} "
        f"worst_fold={metrics['worst_fold_return']:.4%}",
        flush=True,
    )
    return result, metrics


def _comparison_row(
    label: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "variant": label,
        "ending_capital": metrics.get(
            "ending_capital"
        ),
        "return": metrics.get(
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
        "strategy_vs_buy_hold_excess_capital": (
            metrics.get(
                "strategy_vs_buy_hold_excess_capital"
            )
        ),
        "simulation_total_seconds": metrics.get(
            "simulation_total_seconds"
        ),
        "oos_inference_cache_build_seconds": (
            metrics.get(
                "oos_inference_cache_build_seconds"
            )
        ),
        "soft_horizon_consensus_change_rate": (
            metrics.get(
                "soft_horizon_consensus_change_rate"
            )
        ),
        "soft_horizon_consensus_changed_base_actions": (
            metrics.get(
                "soft_horizon_consensus_changed_base_actions"
            )
        ),
    }


def _checkpoint(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Final standalone MCT research run: download fresh RAW bars and "
            "corporate actions directly from Alpaca, persist a local immutable "
            "snapshot, reconstruct splits locally, and run Control versus "
            "Soft Horizon Consensus without MongoDB."
        )
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT),
    )
    parser.add_argument(
        "--reuse-local-snapshot",
        action="store_true",
        help=(
            "Reuse the already downloaded local snapshot. "
            "Default behavior always downloads fresh data from Alpaca."
        ),
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    input_dir = output_dir / "input"
    raw_dir = input_dir / "raw"
    split_dir = input_dir / "split"

    frozen = json.loads(
        config_path.read_text(encoding="utf-8")
    )
    request_payload = deepcopy(
        frozen["request"]
    )
    request = BacktestExecutionRequest.model_validate(
        request_payload
    )
    if allocation_execution_enabled(request):
        raise ValueError(
            "Final v10.8.75 study is limited to the canonical "
            "single-position rotation policy."
        )
    if bool(request.mongo_cache_enabled):
        raise ValueError(
            "Final standalone study requires mongo_cache_enabled=false."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    split_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config_sha256 = _sha256_file(config_path)
    snapshot_manifest_path = (
        input_dir / "snapshot_manifest.json"
    )
    corporate_actions_path = (
        input_dir / "corporate_actions.json"
    )

    credentials = _environment_credentials()
    start = _utc(
        request.start_date
    ).normalize()
    end = _utc(
        request.analysis_end_date
        or request.end_date
    ).normalize()
    api_end = end + pd.Timedelta(days=1)

    fresh_options = (
        frozen.get("fresh_alpaca") or {}
    )
    feed = str(
        fresh_options.get("feed")
        or request.alpaca_historical_feed
    ).lower()
    adjustment = str(
        fresh_options.get("bars_adjustment")
        or "raw"
    ).lower()
    if adjustment != "raw":
        raise ValueError(
            "Final standalone study must download Alpaca adjustment=raw."
        )
    lookback_days = int(
        fresh_options.get(
            "corporate_action_lookback_days",
            366,
        )
    )

    bars_by_symbol: dict[
        str,
        pd.DataFrame,
    ] = {}
    corporate_actions: list[
        dict[str, Any]
    ] = []
    snapshot_files: list[
        dict[str, Any]
    ] = []

    if args.reuse_local_snapshot:
        if not snapshot_manifest_path.exists():
            raise RuntimeError(
                "--reuse-local-snapshot was requested but "
                "snapshot_manifest.json does not exist."
            )
        manifest = json.loads(
            snapshot_manifest_path.read_text(
                encoding="utf-8"
            )
        )
        if (
            manifest.get("config_sha256")
            != config_sha256
        ):
            raise RuntimeError(
                "Local snapshot config SHA-256 does not match "
                "the current final-study configuration."
            )
        corporate_actions = json.loads(
            corporate_actions_path.read_text(
                encoding="utf-8"
            )
        )
        for symbol in request.assets:
            raw_path = (
                raw_dir / f"{symbol}.csv.gz"
            )
            if not raw_path.exists():
                raise RuntimeError(
                    "Missing local RAW snapshot file: "
                    f"{raw_path}"
                )
            frame = pd.read_csv(
                raw_path,
                index_col=0,
                parse_dates=True,
                compression="gzip",
            )
            frame.index = pd.to_datetime(
                frame.index,
                utc=True,
            )
            bars_by_symbol[symbol] = frame
        print(
            "[final-study] reused local immutable snapshot",
            flush=True,
        )
    else:
        corporate_actions = _download_corporate_actions(
            credentials=credentials,
            symbols=list(request.assets),
            research_start=request.start_date,
            research_end=(
                request.analysis_end_date
                or request.end_date
            ),
            lookback_days=lookback_days,
        )
        corporate_actions_path.write_text(
            json.dumps(
                corporate_actions,
                indent=2,
                sort_keys=True,
                default=str,
            ),
            encoding="utf-8",
        )
        snapshot_files.append(
            {
                "path": str(
                    corporate_actions_path.relative_to(
                        output_dir
                    )
                ),
                "sha256": _sha256_file(
                    corporate_actions_path
                ),
                "kind": "corporate_actions",
                "rows": len(corporate_actions),
            }
        )

        print(
            "[final-study] downloading fresh RAW Alpaca bars "
            f"assets={len(request.assets)} "
            f"feed={feed} "
            f"start={start.date()} "
            f"end={end.date()}",
            flush=True,
        )
        for position, symbol in enumerate(
            request.assets,
            start=1,
        ):
            print(
                "[fresh-bars] "
                f"{position}/{len(request.assets)} "
                f"{symbol}",
                flush=True,
            )
            frame = download_stock_bars(
                api_key_id=credentials[
                    "api_key_id"
                ],
                secret_key=credentials[
                    "secret_key"
                ],
                symbol=symbol,
                timeframe=request.timeframe,
                start=start,
                end=api_end,
                feed=feed,
                adjustment="raw",
            )
            if frame.empty:
                raise RuntimeError(
                    "Fresh Alpaca returned no RAW bars "
                    f"for {symbol}."
                )
            raw_path = (
                raw_dir / f"{symbol}.csv.gz"
            )
            hashes = _write_frame_csv(
                frame,
                raw_path,
            )
            snapshot_files.append(
                {
                    "path": str(
                        raw_path.relative_to(
                            output_dir
                        )
                    ),
                    "file_sha256": hashes["file_sha256"],
                    "data_sha256": hashes["data_sha256"],
                    "kind": "raw_bars",
                    "symbol": symbol,
                    "rows": int(len(frame)),
                    "first": pd.Timestamp(
                        frame.index.min()
                    ).isoformat(),
                    "last": pd.Timestamp(
                        frame.index.max()
                    ).isoformat(),
                }
            )
            bars_by_symbol[symbol] = frame

        combined_signature_payload = json.dumps(
            sorted(
                (
                    item["path"],
                    item.get("data_sha256") or item.get("sha256"),
                )
                for item in snapshot_files
            ),
            separators=(",", ":"),
        ).encode("utf-8")
        snapshot_manifest = {
            "schema_version": 1,
            "api_version": "10.8.75",
            "experiment": (
                "final-fresh-alpaca-standalone-v1"
            ),
            "mongo_used": False,
            "data_source": (
                "Alpaca APIs downloaded at execution time"
            ),
            "bars_adjustment": "raw",
            "feed": feed,
            "start": request.start_date,
            "end": (
                request.analysis_end_date
                or request.end_date
            ),
            "asset_count": len(
                request.assets
            ),
            "assets": list(
                request.assets
            ),
            "config_path": str(
                config_path.relative_to(ROOT)
            ),
            "config_sha256": config_sha256,
            "files": snapshot_files,
            "snapshot_sha256": _sha256_bytes(
                combined_signature_payload
            ),
            "corporate_action_count": len(
                corporate_actions
            ),
        }
        _checkpoint(
            snapshot_manifest_path,
            snapshot_manifest,
        )
        print(
            "[final-study] local snapshot "
            f"sha256={snapshot_manifest['snapshot_sha256']}",
            flush=True,
        )

    frames: dict[
        str,
        pd.DataFrame,
    ] = {}
    exclusions: list[
        dict[str, Any]
    ] = []
    data_diagnostics: list[
        dict[str, Any]
    ] = []

    for position, symbol in enumerate(
        request.assets,
        start=1,
    ):
        raw = bars_by_symbol.get(symbol)
        if raw is None or raw.empty:
            exclusions.append(
                {
                    "symbol": symbol,
                    "reason": "missing_raw_history",
                }
            )
            continue
        actions = _actions_for_symbol(
            corporate_actions,
            symbol,
        )
        identity_issue = _structural_identity_issue(
            symbol,
            actions,
        )
        if identity_issue is not None:
            exclusions.append(
                {
                    "symbol": symbol,
                    **identity_issue,
                }
            )
            print(
                "[normalize] "
                f"{symbol} excluded "
                f"reason={identity_issue['reason']}",
                flush=True,
            )
            continue

        reconstructed, applied = (
            _split_normalize(
                raw,
                actions,
            )
        )
        split_path = (
            split_dir / f"{symbol}.csv.gz"
        )
        split_hashes = _write_frame_csv(
            reconstructed,
            split_path,
        )
        frames[symbol] = reconstructed
        data_diagnostics.append(
            {
                "symbol": symbol,
                "raw_rows": int(len(raw)),
                "first": pd.Timestamp(
                    raw.index.min()
                ).isoformat(),
                "last": pd.Timestamp(
                    raw.index.max()
                ).isoformat(),
                "corporate_actions": len(actions),
                "splits_applied": len(applied),
                "split_file_sha256": split_hashes["file_sha256"],
                "split_data_sha256": split_hashes["data_sha256"],
            }
        )
        print(
            "[normalize] "
            f"{position}/{len(request.assets)} "
            f"{symbol} "
            f"rows={len(raw)} "
            f"actions={len(actions)} "
            f"splits={len(applied)}",
            flush=True,
        )

    eligible = list(frames)
    anchors = [
        item
        for item
        in request.calendar_anchor_assets
        if item in frames
    ]
    references = [
        item
        for item
        in request.research_reference_assets
        if item in frames
    ]
    if len(anchors) < 2:
        raise RuntimeError(
            "Fewer than two calendar anchors remain "
            "after standalone structural checks."
        )
    if len(references) < 2:
        references = list(anchors)
    reference_set = set(references)
    candidates = [
        item
        for item
        in request.research_candidate_assets
        if (
            item in frames
            and item not in reference_set
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
    base_settings["lightgbm"] = base_lightgbm
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
            "research_model_settings": base_settings,
            "assets": eligible,
            "calendar_anchor_assets": anchors,
            "research_reference_assets": references,
            "research_candidate_assets": candidates,
            "market_data_provider": "alpaca",
            "alpaca_adjustment": "split",
            "mongo_cache_enabled": False,
            "expected_market_data_signature_sha256": None,
            "research_market_data_snapshot_id": None,
        }
    )

    _, common_dates = prepare_rotation_panel(
        frames,
        base_config,
    )
    folds = _build_walk_forward_folds(
        common_dates,
        base_config,
    )

    control_result, control_metrics = (
        _run_variant(
            label="CONTROL",
            frames=frames,
            config=base_config,
            folds=folds,
        )
    )

    challenger_settings = deepcopy(
        base_settings
    )
    challenger_settings[
        "soft_horizon_consensus"
    ] = deepcopy(
        frozen.get("challenger", {}).get(
            "soft_horizon_consensus"
        )
        or {
            "enabled": True,
            "penalty_strength": 1.0,
        }
    )
    challenger_config = (
        base_config.model_copy(
            update={
                "research_model_settings": (
                    challenger_settings
                )
            }
        )
    )
    challenger_result, challenger_metrics = (
        _run_variant(
            label="SOFT_HORIZON_CONSENSUS",
            frames=frames,
            config=challenger_config,
            folds=folds,
        )
    )

    control_result.predictions.to_csv(
        output_dir / "control_predictions.csv",
        index=True,
    )
    control_result.trades.to_csv(
        output_dir / "control_trades.csv",
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

    comparison = pd.DataFrame(
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
    )
    comparison.to_csv(
        output_dir / "strategy_comparison.csv",
        index=False,
    )
    pd.DataFrame(
        data_diagnostics
    ).to_csv(
        output_dir / "data_diagnostics.csv",
        index=False,
    )
    pd.DataFrame(
        exclusions
    ).to_csv(
        output_dir / "excluded_assets.csv",
        index=False,
    )

    manifest = json.loads(
        snapshot_manifest_path.read_text(
            encoding="utf-8"
        )
    )
    control_capital = float(
        control_metrics["ending_capital"]
    )
    challenger_capital = float(
        challenger_metrics["ending_capital"]
    )
    summary = {
        "schema_version": 1,
        "api_version": "10.8.75",
        "experiment": (
            "final-fresh-alpaca-standalone-v1"
        ),
        "mongo_used": False,
        "configuration": {
            "path": str(
                config_path.relative_to(ROOT)
            ),
            "sha256": config_sha256,
        },
        "snapshot": {
            "sha256": manifest.get(
                "snapshot_sha256"
            ),
            "bars_adjustment_downloaded": (
                "raw"
            ),
            "split_reconstruction": (
                "local point-in-time corporate actions"
            ),
            "feed": feed,
            "start": request.start_date,
            "end": (
                request.analysis_end_date
                or request.end_date
            ),
        },
        "configured_assets": len(
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
                    challenger_metrics["cagr"]
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
                    control_metrics["sharpe"]
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
        "methodology": {
            "database_market_data_used": False,
            "database_configuration_used": False,
            "alpaca_downloaded_fresh": (
                not bool(
                    args.reuse_local_snapshot
                )
            ),
            "raw_bars_persisted_locally": True,
            "corporate_actions_persisted_locally": True,
            "local_snapshot_sha256": (
                manifest.get("snapshot_sha256")
            ),
            "soft_consensus_penalty_strength": (
                challenger_settings[
                    "soft_horizon_consensus"
                ].get("penalty_strength")
            ),
            "buy_hold_included": True,
            "batched_oos_inference": True,
        },
    }
    _checkpoint(
        output_dir / "summary.json",
        summary,
    )

    print("")
    print(
        "[done] FINAL FRESH ALPACA STANDALONE completed",
        flush=True,
    )
    print(
        "[done] mongo_used=false",
        flush=True,
    )
    print(
        "[done] snapshot_sha256="
        f"{manifest.get('snapshot_sha256')}",
        flush=True,
    )
    print(
        "[done] CONTROL="
        f"{control_capital:,.2f}",
        flush=True,
    )
    print(
        "[done] SOFT_HORIZON_CONSENSUS="
        f"{challenger_capital:,.2f}",
        flush=True,
    )
    print(
        "[done] output="
        f"{output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
