from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
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
from market_cycle_trader_api.schemas.requests import (
    BacktestExecutionRequest,
    BacktestRequest,
)


API_VERSION = "10.8.75"
EXPERIMENT = "final-alpaca-direct-soft-horizon-validation-v1"
BARS_ENDPOINT = "https://data.alpaca.markets/v2/stocks/{symbol}/bars"
CORPORATE_ACTIONS_ENDPOINT = "https://data.alpaca.markets/v1/corporate-actions"
DEFAULT_OUTPUT = "output/final_alpaca_direct_validation"
OHLCV = ("open", "high", "low", "close", "volume")
CA_ARRAY_TO_TYPE = {
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
    "reorganizations": "reorganization",
    "partial_calls": "partial_call",
    "capital_gains_distributions": "capital_gains_distribution",
}


def _utc(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return (
        stamp.tz_localize("UTC")
        if stamp.tzinfo is None
        else stamp.tz_convert("UTC")
    )


def _environment_value(*names: str) -> str:
    for name in names:
        value = str(os.getenv(name) or "").strip()
        if value:
            return value
    return ""


def _alpaca_headers() -> dict[str, str]:
    key = _environment_value(
        "ALPACA_API_KEY_ID",
        "APCA_API_KEY_ID",
    )
    secret = _environment_value(
        "ALPACA_SECRET_KEY",
        "ALPACA_API_SECRET_KEY",
        "APCA_API_SECRET_KEY",
    )
    if not key or not secret:
        raise RuntimeError(
            "Alpaca credentials are not configured. "
            "Set ALPACA_API_KEY_ID and ALPACA_SECRET_KEY."
        )
    return {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }


def _request_json(
    url: str,
    *,
    headers: dict[str, str],
    params: dict[str, Any],
    attempts: int = 6,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(max(1, attempts)):
        try:
            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=60,
            )
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                sleep_seconds = (
                    float(retry_after)
                    if retry_after
                    else min(30.0, 2.0 ** attempt)
                )
                if attempt + 1 < attempts:
                    time.sleep(sleep_seconds)
                    continue
            if 500 <= response.status_code < 600 and attempt + 1 < attempts:
                time.sleep(min(30.0, 2.0 ** attempt))
                continue
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError(
                    f"Unexpected JSON payload from {url}."
                )
            return payload
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(min(30.0, 2.0 ** attempt))
                continue
    raise RuntimeError(
        f"Alpaca request failed after {attempts} attempts: "
        f"url={url} error={last_error}"
    ) from last_error


def _bar_frame_from_payload(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=list(OHLCV))
    frame = pd.DataFrame(rows)
    rename = {
        "t": "timestamp",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
        "n": "trade_count",
        "vw": "vwap",
    }
    frame = frame.rename(
        columns={
            key: value
            for key, value in rename.items()
            if key in frame.columns
        }
    )
    required = ["timestamp", *OHLCV]
    missing = [
        column
        for column in required
        if column not in frame.columns
    ]
    if missing:
        raise RuntimeError(
            f"Alpaca bars response is missing columns: {missing}"
        )
    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"],
        utc=True,
        errors="coerce",
    )
    frame = frame.dropna(subset=["timestamp"])
    frame = frame.set_index("timestamp").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    numeric = [
        column
        for column in (
            "open",
            "high",
            "low",
            "close",
            "volume",
            "trade_count",
            "vwap",
        )
        if column in frame.columns
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )
    frame = frame.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna(subset=list(OHLCV))
    ordered = [
        column
        for column in (
            "open",
            "high",
            "low",
            "close",
            "volume",
            "trade_count",
            "vwap",
        )
        if column in frame.columns
    ]
    return frame[ordered]


def _download_raw_bars(
    *,
    symbol: str,
    timeframe: str,
    feed: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    headers: dict[str, str],
) -> pd.DataFrame:
    url = BARS_ENDPOINT.format(symbol=symbol)
    rows: list[dict[str, Any]] = []
    page_token: str | None = None
    while True:
        params: dict[str, Any] = {
            "timeframe": timeframe,
            "start": start.isoformat(),
            "end": (
                end
                + pd.Timedelta(hours=23, minutes=59, seconds=59)
            ).isoformat(),
            "limit": 10000,
            "adjustment": "raw",
            "feed": feed,
            "sort": "asc",
        }
        if page_token:
            params["page_token"] = page_token
        payload = _request_json(
            url,
            headers=headers,
            params=params,
        )
        page_rows = payload.get("bars") or []
        if not isinstance(page_rows, list):
            raise RuntimeError(
                f"Unexpected bars payload for {symbol}."
            )
        rows.extend(
            item
            for item in page_rows
            if isinstance(item, dict)
        )
        page_token = payload.get("next_page_token")
        if not page_token:
            break
    return _bar_frame_from_payload(rows)


def _flatten_corporate_actions(
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    groups = payload.get("corporate_actions") or {}
    if not isinstance(groups, dict):
        return []
    output: list[dict[str, Any]] = []
    for array_name, values in groups.items():
        if not isinstance(values, list):
            continue
        action_type = CA_ARRAY_TO_TYPE.get(
            str(array_name),
            str(array_name).removesuffix("s"),
        )
        for value in values:
            if not isinstance(value, dict):
                continue
            document = dict(value)
            document["action_type"] = action_type
            document["source_array"] = str(array_name)
            output.append(document)
    return output


def _download_corporate_actions(
    *,
    symbols: list[str],
    research_start: date,
    research_end: date,
    headers: dict[str, str],
    chunk_size: int = 40,
) -> list[dict[str, Any]]:
    query_start = research_start - timedelta(days=366)
    documents: list[dict[str, Any]] = []
    size = max(1, min(100, int(chunk_size)))
    for offset in range(0, len(symbols), size):
        chunk = symbols[offset:offset + size]
        page_token: str | None = None
        while True:
            params: dict[str, Any] = {
                "symbols": ",".join(chunk),
                "start": query_start.isoformat(),
                "end": research_end.isoformat(),
                "data_quality": "complete",
                "limit": 1000,
                "sort": "asc",
            }
            if page_token:
                params["page_token"] = page_token
            payload = _request_json(
                CORPORATE_ACTIONS_ENDPOINT,
                headers=headers,
                params=params,
            )
            documents.extend(
                _flatten_corporate_actions(payload)
            )
            page_token = payload.get("next_page_token")
            if not page_token:
                break
        print(
            "[download] corporate-actions "
            f"{min(offset + len(chunk), len(symbols))}/{len(symbols)}",
            flush=True,
        )
    documents.sort(
        key=lambda item: (
            str(item.get("action_type") or ""),
            str(
                item.get("symbol")
                or item.get("source_symbol")
                or item.get("old_symbol")
                or ""
            ),
            str(item.get("process_date") or ""),
            str(item.get("ex_date") or ""),
            str(item.get("id") or ""),
        )
    )
    return documents


def _actions_for_symbol(
    documents: list[dict[str, Any]],
    symbol: str,
) -> list[dict[str, Any]]:
    target = str(symbol).strip().upper()
    fields = (
        "symbol",
        "source_symbol",
        "old_symbol",
        "new_symbol",
        "acquirer_symbol",
        "acquiree_symbol",
    )
    output = [
        item
        for item in documents
        if any(
            str(item.get(field) or "").strip().upper()
            == target
            for field in fields
        )
    ]
    output.sort(
        key=lambda item: (
            str(item.get("process_date") or ""),
            str(item.get("ex_date") or ""),
            str(item.get("effective_date") or ""),
        )
    )
    return output


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

    splits = [
        action
        for action in actions
        if action.get("action_type")
        in {"forward_split", "reverse_split"}
        and action.get("ex_date")
        and action.get("old_rate") is not None
        and action.get("new_rate") is not None
    ]
    splits.sort(
        key=lambda item: str(item.get("ex_date"))
    )

    for action in splits:
        ex_date = _utc(
            action["ex_date"]
        ).normalize()
        old_rate = float(action["old_rate"])
        new_rate = float(action["new_rate"])
        if not (
            np.isfinite(old_rate)
            and np.isfinite(new_rate)
            and old_rate > 0
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
                "action_type": action.get("action_type"),
                "id": action.get("id"),
                "ex_date": str(action.get("ex_date")),
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


def _round_fee_to_cent(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        return 0.0
    return (
        math.ceil((value - 1e-12) * 100.0)
        / 100.0
    )


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
        "total_fee": (
            commission + sec + taf + cat
        ),
    }


def _apply_slippage(
    price: float,
    side: str,
    config: BacktestRequest,
) -> float:
    adjustment = (
        config.slippage_bps / 10_000.0
    )
    return price * (
        1 + adjustment
        if side.upper() == "BUY"
        else 1 - adjustment
    )


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
    simulation = deepcopy(
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
    output = {
        "ending_capital": strategy_ending,
        "strategy_return": strategy_return,
        "cagr": strategy_cagr,
        "sharpe": strategy_sharpe,
        "maximum_drawdown": strategy_maxdd,
        "worst_fold_return": worst_fold_return,
        "folds": fold_rows,
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
        "simulation_profile": simulation,
        "simulation_total_seconds": simulation.get(
            "total_seconds"
        ),
        "simulation_policy_seconds": simulation.get(
            "policy_seconds"
        ),
        "simulation_accounting_seconds": simulation.get(
            "accounting_seconds"
        ),
        "oos_inference_cache_build_seconds": result.metrics.get(
            "oos_inference_cache_build_seconds"
        ),
        "oos_inference_cache_predict_calls": result.metrics.get(
            "oos_inference_cache_predict_calls"
        ),
    }
    output.update(
        {
            key: value
            for key, value in result.metrics.items()
            if str(key).startswith(
                "soft_horizon_consensus_"
            )
        }
    )
    return output


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_canonical_frame(
    path: Path,
    frame: pd.DataFrame,
) -> str:
    output = frame.reset_index().copy()
    first = output.columns[0]
    if first != "timestamp":
        output = output.rename(
            columns={first: "timestamp"}
        )
    output["timestamp"] = pd.to_datetime(
        output["timestamp"],
        utc=True,
    ).map(
        lambda value: value.isoformat()
    )
    csv_text = output.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.12g",
    )
    path.write_text(
        csv_text,
        encoding="utf-8",
        newline="",
    )
    return _sha256_bytes(
        csv_text.encode("utf-8")
    )


def _canonical_json_bytes(
    value: Any,
) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _run_variant(
    *,
    label: str,
    frames: dict[str, pd.DataFrame],
    config: BacktestExecutionRequest,
    folds: list[dict[str, Any]],
) -> tuple[Any, dict[str, Any]]:
    print(
        f"[final] starting {label}",
        flush=True,
    )
    results = run_rotation_models(
        frames,
        config,
        _calculate_reference_fees,
        _apply_slippage,
        progress_callback=lambda p, stage, completed: print(
            f"[final] {label} "
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
        f"[final] completed {label} "
        f"capital={metrics['ending_capital']:,.2f} "
        f"buy_hold={metrics['buy_hold_ending_capital']:,.2f} "
        f"cagr={metrics['cagr']:.4%} "
        f"sharpe={metrics['sharpe']:.4f} "
        f"maxdd={metrics['maximum_drawdown']:.4%} "
        f"worst_fold={metrics['worst_fold_return']:.4%}",
        flush=True,
    )
    return result, metrics


def _prepare_output(
    path: Path,
    *,
    replace: bool,
) -> None:
    if path.exists():
        if not replace:
            raise FileExistsError(
                f"{path} already exists. Final validation never "
                "mixes two downloads. Use a new --output-dir or "
                "--replace-output intentionally."
            )
        shutil.rmtree(path)
    path.mkdir(
        parents=True,
        exist_ok=False,
    )


def _load_frozen_request(
    path: Path,
) -> tuple[BacktestExecutionRequest, dict[str, Any]]:
    payload = json.loads(
        path.read_text(encoding="utf-8")
    )
    request_payload = (
        payload.get("request")
        if isinstance(payload, dict)
        and isinstance(payload.get("request"), dict)
        else payload
    )
    if not isinstance(request_payload, dict):
        raise ValueError(
            "Request JSON must contain a BacktestExecutionRequest "
            "or {'request': {...}}."
        )
    request = BacktestExecutionRequest.model_validate(
        request_payload
    )
    return request, payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Final DB-free market-data validation: download fresh RAW SIP bars "
            "and corporate actions directly from Alpaca, freeze them locally, "
            "locally normalize splits, then run Control and the already-frozen "
            "Soft Horizon Consensus challenger."
        )
    )
    parser.add_argument(
        "--request-json",
        required=True,
        help=(
            "Frozen BacktestExecutionRequest exported before final validation. "
            "The final run itself never reads market data or configuration from MongoDB."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--replace-output",
        action="store_true",
    )
    parser.add_argument(
        "--penalty-strength",
        type=float,
        default=1.0,
    )
    args = parser.parse_args()

    if float(args.penalty_strength) != 1.0:
        raise ValueError(
            "Final validation freezes penalty_strength=1.0. "
            "Do not tune on the final dataset."
        )

    request_path = (
        ROOT / args.request_json
    ).resolve()
    output_dir = (
        ROOT / args.output_dir
    ).resolve()
    _prepare_output(
        output_dir,
        replace=bool(args.replace_output),
    )
    snapshot_dir = output_dir / "snapshot"
    raw_dir = snapshot_dir / "raw_bars"
    results_dir = output_dir / "results"
    raw_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)

    request, request_document = _load_frozen_request(
        request_path
    )
    if allocation_execution_enabled(request):
        raise ValueError(
            "Final soft-consensus validation is defined for the "
            "canonical single-position rotation policy."
        )

    start = _utc(
        request.start_date
    ).normalize()
    end = _utc(
        request.analysis_end_date
        or request.end_date
    ).normalize()
    if end < start:
        raise ValueError(
            "Research end precedes research start."
        )

    symbols = [
        str(item).strip().upper()
        for item in request.assets
        if str(item).strip()
    ]
    if not symbols:
        raise ValueError(
            "Frozen request contains no assets."
        )
    feed = str(
        request.alpaca_historical_feed
        or "sip"
    ).strip().lower()
    timeframe = str(request.timeframe)

    headers = _alpaca_headers()
    downloaded_at = datetime.now(
        timezone.utc
    ).isoformat()

    print(
        "[final] FINAL DATASET IS FROZEN FOR VALIDATION",
        flush=True,
    )
    print(
        "[final] source=Alpaca direct HTTP API",
        flush=True,
    )
    print(
        "[final] database_market_data_used=false",
        flush=True,
    )
    print(
        f"[final] period={start.date()}..{end.date()} "
        f"feed={feed} adjustment=raw assets={len(symbols)}",
        flush=True,
    )

    raw_frames: dict[str, pd.DataFrame] = {}
    file_hashes: dict[str, str] = {}
    data_diagnostics: list[dict[str, Any]] = []

    for position, symbol in enumerate(
        symbols,
        start=1,
    ):
        frame = _download_raw_bars(
            symbol=symbol,
            timeframe=timeframe,
            feed=feed,
            start=start,
            end=end,
            headers=headers,
        )
        raw_frames[symbol] = frame
        file_name = f"{symbol}.csv"
        file_hashes[
            f"raw_bars/{file_name}"
        ] = _write_canonical_frame(
            raw_dir / file_name,
            frame,
        )
        data_diagnostics.append(
            {
                "symbol": symbol,
                "raw_rows": int(len(frame)),
                "first_bar": (
                    str(frame.index.min())
                    if not frame.empty
                    else None
                ),
                "last_bar": (
                    str(frame.index.max())
                    if not frame.empty
                    else None
                ),
            }
        )
        print(
            f"[download] bars "
            f"{position}/{len(symbols)} "
            f"{symbol} rows={len(frame)}",
            flush=True,
        )

    corporate_actions = _download_corporate_actions(
        symbols=symbols,
        research_start=start.date(),
        research_end=end.date(),
        headers=headers,
    )
    ca_bytes = _canonical_json_bytes(
        corporate_actions
    )
    ca_path = snapshot_dir / "corporate_actions.json"
    ca_path.write_bytes(ca_bytes)
    file_hashes[
        "corporate_actions.json"
    ] = _sha256_bytes(ca_bytes)

    frozen_request = request.model_dump(
        mode="json"
    )
    request_bytes = _canonical_json_bytes(
        frozen_request
    )
    request_copy = snapshot_dir / "request.json"
    request_copy.write_bytes(request_bytes)
    file_hashes[
        "request.json"
    ] = _sha256_bytes(request_bytes)

    frames: dict[str, pd.DataFrame] = {}
    exclusions: list[dict[str, Any]] = []
    split_diagnostics: list[dict[str, Any]] = []

    for symbol in symbols:
        raw = raw_frames[symbol]
        actions = _actions_for_symbol(
            corporate_actions,
            symbol,
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
            print(
                f"[final] excluded {symbol}: "
                f"{issue['reason']}",
                flush=True,
            )
            continue
        if raw.empty:
            exclusions.append(
                {
                    "symbol": symbol,
                    "reason": "missing_raw_history",
                }
            )
            continue

        normalized, applied = _split_normalize(
            raw,
            actions,
        )
        frames[symbol] = normalized
        split_diagnostics.append(
            {
                "symbol": symbol,
                "raw_rows": int(len(raw)),
                "corporate_actions": int(
                    len(actions)
                ),
                "splits_applied": int(
                    len(applied)
                ),
                "split_details": applied,
            }
        )

    eligible = list(frames)
    anchors = [
        symbol
        for symbol
        in request.calendar_anchor_assets
        if symbol in frames
    ]
    if len(anchors) < 2:
        raise RuntimeError(
            "Fresh Alpaca dataset left fewer than two valid "
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

    research_settings = deepcopy(
        request.research_model_settings
    )
    lightgbm = deepcopy(
        research_settings.get("lightgbm") or {}
    )
    lightgbm[
        "early_stopping_enabled"
    ] = False
    research_settings["lightgbm"] = lightgbm
    research_settings["horizon_voting"] = {
        "enabled": False,
    }
    research_settings[
        "soft_horizon_consensus"
    ] = {
        "enabled": False,
    }

    base_config = request.model_copy(
        update={
            "research_model_settings": (
                research_settings
            ),
            "assets": eligible,
            "calendar_anchor_assets": anchors,
            "research_reference_assets": references,
            "research_candidate_assets": candidates,
            "market_data_provider": "alpaca",
            "alpaca_historical_feed": feed,
            "alpaca_adjustment": "split",
            "expected_market_data_signature_sha256": None,
            "research_market_data_snapshot_id": None,
            "research_market_data_mode": "database_only",
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

    manifest_base = {
        "schema_version": 1,
        "api_version": API_VERSION,
        "experiment": EXPERIMENT,
        "downloaded_at_utc": downloaded_at,
        "source": "alpaca_direct_api",
        "database_market_data_used": False,
        "database_corporate_actions_used": False,
        "database_result_persistence_used": False,
        "bars_endpoint": (
            "https://data.alpaca.markets/v2/stocks/{symbol}/bars"
        ),
        "corporate_actions_endpoint": (
            CORPORATE_ACTIONS_ENDPOINT
        ),
        "bars_adjustment_downloaded": "raw",
        "local_price_normalization": (
            "forward_split_and_reverse_split_only"
        ),
        "feed": feed,
        "timeframe": timeframe,
        "research_start": str(start.date()),
        "research_end": str(end.date()),
        "requested_assets": symbols,
        "eligible_assets": eligible,
        "excluded_assets": exclusions,
        "file_sha256": dict(
            sorted(file_hashes.items())
        ),
    }
    snapshot_hash = _sha256_bytes(
        _canonical_json_bytes(
            {
                "request": frozen_request,
                "files": dict(
                    sorted(file_hashes.items())
                ),
                "eligible_assets": eligible,
                "excluded_assets": exclusions,
            }
        )
    )
    manifest_base[
        "snapshot_sha256"
    ] = snapshot_hash
    (
        snapshot_dir / "manifest.json"
    ).write_text(
        json.dumps(
            manifest_base,
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        data_diagnostics
    ).to_csv(
        results_dir
        / "download_diagnostics.csv",
        index=False,
    )
    pd.DataFrame(
        split_diagnostics
    ).to_csv(
        results_dir
        / "split_diagnostics.csv",
        index=False,
    )
    pd.DataFrame(
        exclusions
    ).to_csv(
        results_dir
        / "excluded_assets.csv",
        index=False,
    )

    control_result, control_metrics = (
        _run_variant(
            label="FINAL_CONTROL",
            frames=frames,
            config=base_config,
            folds=folds,
        )
    )

    challenger_settings = deepcopy(
        research_settings
    )
    challenger_settings[
        "soft_horizon_consensus"
    ] = {
        "enabled": True,
        "penalty_strength": 1.0,
    }
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
            label="FINAL_SOFT_HORIZON_CONSENSUS",
            frames=frames,
            config=challenger_config,
            folds=folds,
        )
    )

    control_result.predictions.to_csv(
        results_dir
        / "control_predictions.csv",
        index=True,
    )
    control_result.trades.to_csv(
        results_dir
        / "control_trades.csv",
        index=False,
    )
    challenger_result.predictions.to_csv(
        results_dir
        / "soft_horizon_consensus_predictions.csv",
        index=True,
    )
    challenger_result.trades.to_csv(
        results_dir
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
            results_dir
            / "soft_horizon_consensus_decisions.csv",
            index=True,
        )

    control_capital = float(
        control_metrics["ending_capital"]
    )
    challenger_capital = float(
        challenger_metrics["ending_capital"]
    )

    comparison = pd.DataFrame(
        [
            {
                "variant": "CONTROL",
                **{
                    key: control_metrics.get(key)
                    for key in (
                        "ending_capital",
                        "strategy_return",
                        "cagr",
                        "sharpe",
                        "maximum_drawdown",
                        "worst_fold_return",
                        "buy_hold_ending_capital",
                        "buy_hold_return",
                        "buy_hold_cagr",
                        "buy_hold_sharpe",
                        "buy_hold_maximum_drawdown",
                        "strategy_vs_buy_hold_capital_ratio",
                        "simulation_total_seconds",
                        "oos_inference_cache_build_seconds",
                    )
                },
            },
            {
                "variant": "SOFT_HORIZON_CONSENSUS",
                **{
                    key: challenger_metrics.get(key)
                    for key in (
                        "ending_capital",
                        "strategy_return",
                        "cagr",
                        "sharpe",
                        "maximum_drawdown",
                        "worst_fold_return",
                        "buy_hold_ending_capital",
                        "buy_hold_return",
                        "buy_hold_cagr",
                        "buy_hold_sharpe",
                        "buy_hold_maximum_drawdown",
                        "strategy_vs_buy_hold_capital_ratio",
                        "simulation_total_seconds",
                        "oos_inference_cache_build_seconds",
                    )
                },
            },
        ]
    )
    comparison.to_csv(
        results_dir
        / "strategy_comparison.csv",
        index=False,
    )

    final_summary = {
        "schema_version": 1,
        "api_version": API_VERSION,
        "experiment": EXPERIMENT,
        "final_validation": True,
        "dataset_role": (
            "frozen_final_validation_dataset"
        ),
        "parameter_tuning_on_this_dataset": False,
        "optuna_used": False,
        "caro_used": False,
        "penalty_strength": 1.0,
        "snapshot_sha256": snapshot_hash,
        "market_data_source": (
            "fresh_alpaca_direct_api"
        ),
        "database_market_data_used": False,
        "period": {
            "start": str(start.date()),
            "end": str(end.date()),
        },
        "universe": {
            "requested": symbols,
            "eligible": eligible,
            "excluded": exclusions,
        },
        "methodology": {
            "bars": "RAW SIP from Alpaca",
            "corporate_actions": (
                "fresh direct Alpaca corporate-actions API"
            ),
            "split_handling": (
                "local forward/reverse split normalization"
            ),
            "dividend_price_back_adjustment": False,
            "control": (
                "frozen canonical weighted multi-horizon LightGBM policy"
            ),
            "challenger": (
                "frozen soft multi-horizon rank consensus, penalty_strength=1.0"
            ),
            "buy_and_hold": (
                "equal-weight initial purchase, no periodic rebalance, final liquidation"
            ),
            "hyperparameter_optimization": (
                "none on final validation data"
            ),
        },
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
                - float(control_metrics["cagr"])
            ),
            "sharpe_difference": (
                float(
                    challenger_metrics["sharpe"]
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
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "requests": requests.__version__,
        },
        "request_source_document": {
            "path": str(request_path),
            "source_job_id": (
                request_document.get(
                    "source_job_id"
                )
                if isinstance(
                    request_document,
                    dict,
                )
                else None
            ),
        },
    }
    (
        results_dir / "summary.json"
    ).write_text(
        json.dumps(
            final_summary,
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )

    archive_base = (
        output_dir.parent
        / output_dir.name
    )
    shutil.make_archive(
        str(archive_base),
        "zip",
        root_dir=output_dir,
    )

    print("")
    print(
        "[done] FINAL ALPACA DIRECT VALIDATION COMPLETED",
        flush=True,
    )
    print(
        f"[done] snapshot_sha256={snapshot_hash}",
        flush=True,
    )
    print(
        f"[done] CONTROL={control_capital:,.2f}",
        flush=True,
    )
    print(
        "[done] SOFT_HORIZON_CONSENSUS="
        f"{challenger_capital:,.2f}",
        flush=True,
    )
    print(
        "[done] database_market_data_used=false",
        flush=True,
    )
    print(
        f"[done] output={output_dir}",
        flush=True,
    )
    print(
        f"[done] archive={archive_base}.zip",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
