from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import date, timedelta
import hashlib
import importlib.metadata
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
from market_cycle_trader_api.infrastructure.market_data.alpaca import (
    download_stock_bars,
)
from market_cycle_trader_api.schemas.requests import (
    BacktestExecutionRequest,
)


BARS_ENDPOINT = "https://data.alpaca.markets/v2/stocks/bars"
CORPORATE_ACTIONS_ENDPOINT = (
    "https://data.alpaca.markets/v1/corporate-actions"
)
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

DEFAULT_CONFIG = "research/soft_horizon_7m_direct_alpaca_v10_8_81.json"
DEFAULT_OUTPUT = "output/soft_horizon_7m_direct_alpaca_v10881"
SCRIPT_VERSION = "soft-horizon-7m-direct-alpaca-v1.1.0"


def _utc(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return (
        stamp.tz_localize("UTC")
        if stamp.tzinfo is None
        else stamp.tz_convert("UTC")
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )


def _first_env(*names: str) -> tuple[str | None, str | None]:
    for name in names:
        value = str(os.getenv(name) or "").strip()
        if value:
            return name, value
    return None, None


def _alpaca_headers() -> tuple[dict[str, str], dict[str, str]]:
    key_name, key = _first_env(
        "ALPACA_API_KEY_ID",
        "APCA_API_KEY_ID",
    )
    secret_name, secret = _first_env(
        "ALPACA_SECRET_KEY",
        "ALPACA_API_SECRET_KEY",
        "APCA_API_SECRET_KEY",
    )
    if not key or not secret:
        raise RuntimeError(
            "Standalone final research requires Alpaca credentials only "
            "from environment variables. Set ALPACA_API_KEY_ID and "
            "ALPACA_SECRET_KEY (APCA_* aliases are also accepted)."
        )
    return (
        {
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": secret,
        },
        {
            "api_key_environment": str(key_name),
            "secret_key_environment": str(secret_name),
        },
    )


def _request_json(
    url: str,
    *,
    headers: dict[str, str],
    params: dict[str, Any],
    attempts: int = 5,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=60,
            )
            if response.status_code == 429:
                raise RuntimeError(
                    "Alpaca rate limit (HTTP 429)."
                )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError(
                    "Unexpected non-object Alpaca response."
                )
            return payload
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                delay = min(30, 2 ** attempt)
                print(
                    f"[alpaca] retry={attempt + 1}/{attempts - 1} "
                    f"delay={delay}s error={type(exc).__name__}: {exc}",
                    flush=True,
                )
                time.sleep(delay)
    raise RuntimeError(
        f"Alpaca request failed after {attempts} attempts: {last_error}"
    ) from last_error


def _bar_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp": _utc(row["t"]).isoformat(),
        "open": float(row["o"]),
        "high": float(row["h"]),
        "low": float(row["l"]),
        "close": float(row["c"]),
        "volume": float(row["v"]),
        "trade_count": (
            int(row["n"])
            if row.get("n") is not None
            else None
        ),
        "vwap": (
            float(row["vw"])
            if row.get("vw") is not None
            else None
        ),
    }


def _download_bars(
    *,
    headers: dict[str, str],
    request: BacktestExecutionRequest,
    snapshot_dir: Path,
    chunk_size: int,
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Download bars with the exact Alpaca semantics used by the 10.8.74 snapshot.

    The original alpaca_market_bars_raw_20260919 collection was built by
    download_fresh_alpaca_snapshot.py, which called download_stock_bars()
    once per symbol with SIP, 1Day, adjustment=raw, and end + 1 calendar day.
    This standalone runner deliberately reuses that same downloader while
    writing the returned frames to immutable local CSV files instead of MongoDB.
    """
    del chunk_size  # Kept in the CLI for compatibility; original loader was per-symbol.

    bars_dir = snapshot_dir / "raw_bars"
    bars_dir.mkdir(parents=True, exist_ok=True)

    symbols = list(request.assets)
    timeframe = str(request.timeframe)
    feed = str(request.alpaca_historical_feed or "sip").strip().lower()
    adjustment = "raw"
    start = pd.Timestamp(request.start_date)
    end_text = request.analysis_end_date or request.end_date
    end = pd.Timestamp(end_text)
    api_end = end + pd.Timedelta(days=1)

    api_key_id = str(headers.get("APCA-API-KEY-ID") or "").strip()
    secret_key = str(headers.get("APCA-API-SECRET-KEY") or "").strip()
    if not api_key_id or not secret_key:
        raise RuntimeError("Alpaca API credentials are not configured.")

    files: dict[str, Path] = {}
    row_counts: dict[str, int] = {}
    symbol_summaries: list[dict[str, Any]] = []

    print(
        f"[alpaca-bars] loader=10.8.74-download_stock_bars "
        f"assets={len(symbols)} feed={feed} adjustment={adjustment} "
        f"timeframe={timeframe} start={start.date()} end={end.date()}",
        flush=True,
    )

    for position, symbol in enumerate(symbols, start=1):
        print(
            f"[alpaca-bars] {position}/{len(symbols)} downloading {symbol}...",
            flush=True,
        )
        frame = download_stock_bars(
            api_key_id=api_key_id,
            secret_key=secret_key,
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=api_end,
            feed=feed,
            adjustment=adjustment,
        )
        if frame is None or frame.empty:
            raise RuntimeError(
                f"Fresh Alpaca RAW snapshot returned no bars for {symbol}."
            )

        frame = frame.copy().sort_index()
        frame = frame[~frame.index.duplicated(keep="last")]
        for column in ("open", "high", "low", "close", "volume"):
            frame[column] = pd.to_numeric(
                frame[column],
                errors="raise",
            ).astype(float)

        target = bars_dir / f"{symbol}.csv"
        frame.reset_index().to_csv(
            target,
            index=False,
            float_format="%.17g",
        )
        files[symbol] = target
        row_counts[symbol] = int(len(frame))

        first = pd.Timestamp(frame.index.min()).date().isoformat()
        last = pd.Timestamp(frame.index.max()).date().isoformat()
        symbol_summaries.append(
            {
                "symbol": symbol,
                "rows": int(len(frame)),
                "first": first,
                "last": last,
            }
        )
        print(
            f"[alpaca-bars] {symbol} rows={len(frame)} "
            f"first={first} last={last}",
            flush=True,
        )

    return files, {
        "loader": "market_cycle_trader_api.infrastructure.market_data.alpaca.download_stock_bars",
        "loader_semantics": "matches download_fresh_alpaca_snapshot.py used for raw_20260919",
        "feed": feed,
        "adjustment": adjustment,
        "timeframe": timeframe,
        "requested_start": start.date().isoformat(),
        "requested_end": end.date().isoformat(),
        "api_end_exclusive_upper_bound": api_end.date().isoformat(),
        "asset_count": len(symbols),
        "row_counts": row_counts,
        "total_rows": int(sum(row_counts.values())),
        "symbols": symbol_summaries,
    }


def _flatten_corporate_actions(
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    groups = payload.get("corporate_actions") or {}
    if not isinstance(groups, dict):
        return documents
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
            documents.append(document)
    return documents


def _download_corporate_actions(
    *,
    headers: dict[str, str],
    request: BacktestExecutionRequest,
    snapshot_dir: Path,
    chunk_size: int,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    research_start = date.fromisoformat(
        request.start_date
    )
    query_start = (
        research_start - timedelta(days=366)
    ).isoformat()
    query_end = str(
        request.analysis_end_date
        or request.end_date
    )
    symbols = sorted(
        {
            str(symbol).strip().upper()
            for symbol in request.assets
            if str(symbol).strip()
        }
    )
    resolved_chunk_size = max(1, min(100, int(chunk_size)))
    documents: list[dict[str, Any]] = []

    for offset in range(0, len(symbols), resolved_chunk_size):
        chunk = symbols[offset : offset + resolved_chunk_size]
        page_token: str | None = None
        page_count = 0
        while True:
            params: dict[str, Any] = {
                "symbols": ",".join(chunk),
                "types": ",".join(REQUEST_TYPES),
                "start": query_start,
                "end": query_end,
                "region": "us",
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
            page_count += 1
            documents.extend(
                _flatten_corporate_actions(payload)
            )
            page_token = payload.get("next_page_token")
            if not page_token:
                break

        print(
            f"[corporate-actions] "
            f"progress={min(offset + len(chunk), len(symbols))}/{len(symbols)} "
            f"chunk={chunk[0]}..{chunk[-1]} pages={page_count}",
            flush=True,
        )

    def sort_key(item: dict[str, Any]) -> tuple[str, ...]:
        return (
            str(item.get("action_type") or ""),
            str(
                item.get("symbol")
                or item.get("source_symbol")
                or ""
            ),
            str(item.get("process_date") or ""),
            str(item.get("ex_date") or ""),
            str(item.get("effective_date") or ""),
            str(item.get("id") or ""),
        )

    documents = sorted(
        documents,
        key=sort_key,
    )
    target = snapshot_dir / "corporate_actions.jsonl"
    with target.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        for document in documents:
            handle.write(
                json.dumps(
                    document,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                )
                + "\n"
            )

    counts: dict[str, int] = {}
    for item in documents:
        action_type = str(
            item.get("action_type") or "unknown"
        )
        counts[action_type] = (
            counts.get(action_type, 0) + 1
        )

    return target, documents, {
        "endpoint": CORPORATE_ACTIONS_ENDPOINT,
        "query_start": query_start,
        "query_end": query_end,
        "types": list(REQUEST_TYPES),
        "document_count": len(documents),
        "counts_by_type": counts,
    }


def _actions_for_symbol(
    documents: list[dict[str, Any]],
    symbol: str,
) -> list[dict[str, Any]]:
    normalized = str(symbol).strip().upper()
    fields = (
        "symbol",
        "source_symbol",
        "old_symbol",
        "new_symbol",
        "acquirer_symbol",
        "acquiree_symbol",
    )
    result = []
    for item in documents:
        if any(
            str(item.get(field) or "").strip().upper()
            == normalized
            for field in fields
        ):
            result.append(item)
    return result


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
            old_rate > 0.0
            and new_rate > 0.0
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


def _load_raw_bar_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"],
        utc=True,
    )
    frame = frame.set_index("timestamp").sort_index()
    for column in (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trade_count",
        "vwap",
    ):
        if column in frame.columns:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )
    # The 10.8.74 Mongo snapshot returned split-adjustable OHLCV as
    # floating-point values. Preserve that dtype when the same RAW data
    # is transported through local CSV files so split normalization is
    # numerically equivalent and pandas never rejects fractional values.
    for column in ("open", "high", "low", "close", "volume"):
        if column in frame.columns:
            frame[column] = frame[column].astype(float)
    return frame


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
    config: BacktestExecutionRequest,
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
    config: BacktestExecutionRequest,
) -> float:
    adjustment = config.slippage_bps / 10_000.0
    if side.upper() == "BUY":
        return price * (1.0 + adjustment)
    return price * (1.0 - adjustment)


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
        "strategy_return": strategy_return,
        "cagr": strategy_cagr,
        "sharpe": strategy_sharpe,
        "maximum_drawdown": strategy_maxdd,
        "worst_fold_return": worst_fold_return,
        "folds": fold_rows,
        "buy_hold_ending_capital": (
            buy_hold_ending
        ),
        "buy_hold_return": buy_hold_return,
        "buy_hold_cagr": buy_hold_cagr,
        "buy_hold_sharpe": buy_hold_sharpe,
        "buy_hold_maximum_drawdown": (
            buy_hold_maxdd
        ),
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
        "predictive_diagnostics": predictive,
        "simulation_profile": simulation,
        "simulation_total_seconds": (
            simulation.get("total_seconds")
        ),
        "simulation_policy_seconds": (
            simulation.get("policy_seconds")
        ),
        "simulation_accounting_seconds": (
            simulation.get("accounting_seconds")
        ),
        "simulation_session_count": (
            simulation.get("session_count")
        ),
    }

    for key, value in result.metrics.items():
        if (
            str(key).startswith(
                "soft_horizon_consensus_"
            )
            or str(key).startswith(
                "oos_inference_cache_"
            )
        ):
            metrics[str(key)] = value
    return metrics


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
            f"[final] {label} progress={p:.1f}% "
            f"completed={completed} stage={stage}",
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
        f"sharpe={metrics['sharpe']:.4f} "
        f"maxdd={metrics['maximum_drawdown']:.4%} "
        f"worst_fold={metrics['worst_fold_return']:.4%}",
        flush=True,
    )
    return result, metrics


def _runtime_versions() -> dict[str, str | None]:
    packages = (
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "lightgbm",
        "requests",
    )
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = (
                importlib.metadata.version(package)
            )
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    versions["python"] = sys.version.split()[0]
    versions["platform"] = platform.platform()
    return versions


def _load_config(
    path: Path,
) -> tuple[dict[str, Any], BacktestExecutionRequest]:
    document = json.loads(
        path.read_text(encoding="utf-8")
    )
    if not isinstance(document, dict):
        raise ValueError(
            "Final research config must be a JSON object."
        )
    request_payload = document.get("request")
    if not isinstance(request_payload, dict):
        raise ValueError(
            "Final research config must contain request."
        )
    request = BacktestExecutionRequest.model_validate(
        request_payload
    )
    if request.mongo_cache_enabled:
        raise ValueError(
            "Final standalone research requires "
            "mongo_cache_enabled=false."
        )
    if (
        request.research_market_data_mode
        != "database_only"
    ):
        raise ValueError(
            "10.8.74 reproduction requires "
            "research_market_data_mode=database_only."
        )
    if request.alpaca_adjustment != "raw":
        raise ValueError(
            "Final standalone research must download "
            "Alpaca adjustment=raw."
        )
    if not request.end_date:
        raise ValueError(
            "Final standalone research requires a "
            "closed end_date."
        )

    # Reproduction contract: do not allow this runner to drift away from
    # the 10.8.74 experiment that produced ~US$7.38M.  Only the physical
    # data transport is allowed to change.
    if request.start_date != "2016-01-01" or request.end_date != "2026-09-18":
        raise ValueError(
            "10.8.74 reproduction requires the frozen "
            "2016-01-01 -> 2026-09-18 research window."
        )
    if len(request.assets) != 56 or "DOC" not in request.assets or "CLMT" not in request.assets:
        raise ValueError(
            "10.8.74 reproduction requires the original 56-asset request "
            "including DOC and CLMT before the original structural guard."
        )
    if request.rotation_accelerator != "cuda":
        raise ValueError(
            "This direct Alpaca replay requires rotation_accelerator=cuda."
        )
    if request.rotation_allow_cpu_fallback:
        raise ValueError(
            "This direct Alpaca replay requires "
            "rotation_allow_cpu_fallback=false."
        )
    if request.deterministic_execution:
        raise ValueError(
            "GPU execution requires deterministic_execution=false."
        )
    consensus = (
        request.research_model_settings.get("soft_horizon_consensus")
        or {}
    )
    if float(consensus.get("penalty_strength", 0.0)) != 1.0:
        raise ValueError(
            "10.8.74 reproduction requires soft horizon "
            "penalty_strength=1.0."
        )
    return document, request


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Final standalone MCT research: fresh Alpaca RAW bars and "
            "corporate actions -> immutable local snapshot -> local split "
            "normalization -> Control + Soft Horizon Consensus. "
            "No MongoDB data is read or written."
        )
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--replace-snapshot",
        action="store_true",
        help=(
            "Delete an existing local snapshot and download "
            "fresh data again from Alpaca."
        ),
    )
    parser.add_argument(
        "--reuse-snapshot",
        action="store_true",
        help=(
            "Reuse the existing local snapshot instead of "
            "calling Alpaca. Mutually exclusive with --replace-snapshot."
        ),
    )
    parser.add_argument(
        "--bars-chunk-size",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--corporate-actions-chunk-size",
        type=int,
        default=40,
    )
    args = parser.parse_args()

    if (
        args.replace_snapshot
        and args.reuse_snapshot
    ):
        raise ValueError(
            "--replace-snapshot and --reuse-snapshot "
            "are mutually exclusive."
        )

    config_path = (
        ROOT / args.config
    ).resolve()
    output_dir = (
        ROOT / args.output_dir
    ).resolve()
    snapshot_dir = output_dir / "snapshot"
    results_dir = output_dir / "results"

    config_document, request = _load_config(
        config_path
    )
    config_sha256 = _canonical_sha256(
        config_document
    )

    if snapshot_dir.exists():
        if args.replace_snapshot:
            shutil.rmtree(snapshot_dir)
        elif not args.reuse_snapshot:
            raise RuntimeError(
                f"Snapshot already exists: {snapshot_dir}. "
                "Use --replace-snapshot for a fresh Alpaca download "
                "or --reuse-snapshot to reproduce an existing snapshot."
            )

    snapshot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    headers, credential_metadata = (
        _alpaca_headers()
    )

    if args.reuse_snapshot:
        manifest_path = (
            snapshot_dir / "manifest.json"
        )
        if not manifest_path.exists():
            raise RuntimeError(
                "Cannot reuse snapshot without manifest.json."
            )
        manifest = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )
        if (
            manifest.get("config_sha256")
            != config_sha256
        ):
            raise RuntimeError(
                "Existing snapshot was produced from a "
                "different final research configuration."
            )
        bars_files = {
            symbol: (
                snapshot_dir
                / "raw_bars"
                / f"{symbol}.csv"
            )
            for symbol in request.assets
        }
        missing = [
            symbol
            for symbol, path in bars_files.items()
            if not path.exists()
        ]
        if missing:
            raise RuntimeError(
                "Existing snapshot is incomplete: "
                + ", ".join(missing)
            )
        ca_path = (
            snapshot_dir
            / "corporate_actions.jsonl"
        )
        expected_hashes = (
            manifest.get("file_hashes") or {}
        )
        for relative_path, expected_sha in expected_hashes.items():
            local_path = snapshot_dir / str(relative_path)
            if not local_path.exists():
                raise RuntimeError(
                    f"Snapshot manifest file is missing: {relative_path}"
                )
            actual_sha = _sha256_file(local_path)
            if actual_sha != str(expected_sha):
                raise RuntimeError(
                    f"Snapshot integrity mismatch for {relative_path}: "
                    f"expected={expected_sha}, actual={actual_sha}"
                )

        corporate_actions = [
            json.loads(line)
            for line in ca_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        print(
            f"[snapshot] reusing id="
            f"{manifest.get('snapshot_sha256')}",
            flush=True,
        )
    else:
        downloaded_at = pd.Timestamp.now(
            tz="UTC"
        ).isoformat()
        bars_files, bars_metadata = (
            _download_bars(
                headers=headers,
                request=request,
                snapshot_dir=snapshot_dir,
                chunk_size=int(
                    args.bars_chunk_size
                ),
            )
        )
        (
            ca_path,
            corporate_actions,
            ca_metadata,
        ) = _download_corporate_actions(
            headers=headers,
            request=request,
            snapshot_dir=snapshot_dir,
            chunk_size=int(
                args.corporate_actions_chunk_size
            ),
        )

        file_hashes = {
            f"raw_bars/{symbol}.csv": (
                _sha256_file(path)
            )
            for symbol, path
            in sorted(bars_files.items())
        }
        file_hashes[
            "corporate_actions.jsonl"
        ] = _sha256_file(ca_path)

        snapshot_identity = {
            "script_version": SCRIPT_VERSION,
            "config_sha256": config_sha256,
            "bars": bars_metadata,
            "corporate_actions": ca_metadata,
            "file_hashes": file_hashes,
        }
        snapshot_sha256 = (
            _canonical_sha256(
                snapshot_identity
            )
        )
        manifest = {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "downloaded_at": downloaded_at,
            "database_access": False,
            "source": "alpaca_direct_http",
            "credential_source": credential_metadata,
            "config_path": str(
                config_path.relative_to(ROOT)
            ),
            "config_sha256": config_sha256,
            "snapshot_sha256": snapshot_sha256,
            "runtime_versions": _runtime_versions(),
            **snapshot_identity,
        }
        _checkpoint(
            snapshot_dir / "manifest.json",
            manifest,
        )
        print(
            f"[snapshot] fresh Alpaca snapshot complete "
            f"id={snapshot_sha256}",
            flush=True,
        )

    frames: dict[str, pd.DataFrame] = {}
    exclusions: list[dict[str, Any]] = []
    data_diagnostics: list[dict[str, Any]] = []

    for position, symbol in enumerate(
        request.assets,
        start=1,
    ):
        raw = _load_raw_bar_file(
            bars_files[symbol]
        )
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
                f"[data] {position}/{len(request.assets)} "
                f"{symbol} excluded "
                f"reason={issue['reason']}",
                flush=True,
            )
            continue

        reconstructed, applied = (
            _split_normalize(
                raw,
                actions,
            )
        )
        frames[symbol] = reconstructed
        data_diagnostics.append(
            {
                "symbol": symbol,
                "raw_rows": int(len(raw)),
                "first_raw_session": (
                    raw.index.min().isoformat()
                    if len(raw)
                    else None
                ),
                "last_raw_session": (
                    raw.index.max().isoformat()
                    if len(raw)
                    else None
                ),
                "corporate_actions": int(
                    len(actions)
                ),
                "splits_applied": int(
                    len(applied)
                ),
            }
        )

    if len(frames) < 2:
        raise RuntimeError(
            "Fewer than two structurally valid assets remain."
        )

    eligible = list(frames)
    anchors = [
        symbol
        for symbol in request.calendar_anchor_assets
        if symbol in frames
    ]
    if len(anchors) < 2:
        raise RuntimeError(
            "Too few calendar anchors remain after "
            "structural-identity filtering."
        )
    references = [
        symbol
        for symbol in request.research_reference_assets
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
    base_settings[
        "horizon_voting"
    ] = {"enabled": False}
    base_settings[
        "soft_horizon_consensus"
    ] = {"enabled": False}

    base_config = request.model_copy(
        update={
            "assets": eligible,
            "calendar_anchor_assets": anchors,
            "research_reference_assets": references,
            "research_candidate_assets": candidates,
            "research_model_settings": base_settings,
            # Preserve the exact model-facing configuration used by the
            # reproduced 10.8.74 experiment.  The transport changed to a
            # local immutable Alpaca snapshot, but run_rotation_models still
            # receives the same RAW+split semantic configuration as 10.8.74.
            "market_data_provider": "alpaca",
            "alpaca_adjustment": "split",
            "research_market_data_mode": "database_only",
            "expected_market_data_signature_sha256": None,
            "research_market_data_snapshot_id": None,
        }
    )

    if allocation_execution_enabled(
        base_config
    ):
        raise ValueError(
            "Final standalone study is locked to the "
            "canonical single-position rotation policy."
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
    ] = {
        "enabled": True,
        "penalty_strength": float(
            (
                request.research_model_settings.get(
                    "soft_horizon_consensus"
                )
                or {}
            ).get(
                "penalty_strength",
                1.0,
            )
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
    (
        challenger_result,
        challenger_metrics,
    ) = _run_variant(
        label="SOFT_HORIZON_CONSENSUS",
        frames=frames,
        config=challenger_config,
        folds=folds,
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
                        "simulation_policy_seconds",
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
                        "simulation_policy_seconds",
                    )
                },
                "soft_horizon_consensus_changed_base_actions": (
                    challenger_metrics.get(
                        "soft_horizon_consensus_changed_base_actions"
                    )
                ),
                "soft_horizon_consensus_change_rate": (
                    challenger_metrics.get(
                        "soft_horizon_consensus_change_rate"
                    )
                ),
            },
        ]
    )
    comparison.to_csv(
        results_dir
        / "strategy_comparison.csv",
        index=False,
    )
    pd.DataFrame(
        data_diagnostics
    ).to_csv(
        results_dir
        / "data_diagnostics.csv",
        index=False,
    )
    pd.DataFrame(
        exclusions
    ).to_csv(
        results_dir
        / "excluded_assets.csv",
        index=False,
    )

    resolved_request = (
        challenger_config.model_dump(
            mode="json"
        )
    )
    _checkpoint(
        results_dir
        / "resolved_final_request.json",
        resolved_request,
    )
    shutil.copyfile(
        config_path,
        results_dir / "frozen_research_config.json",
    )
    shutil.copyfile(
        snapshot_dir / "manifest.json",
        results_dir
        / "snapshot_manifest.json",
    )

    control_capital = float(
        control_metrics["ending_capital"]
    )
    challenger_capital = float(
        challenger_metrics["ending_capital"]
    )
    summary = {
        "schema_version": 1,
        "api_version": "10.8.81",
        "experiment": (
            "raw-split-soft-horizon-consensus-direct-alpaca-v1"
        ),
        "database_access": False,
        "tuning_enabled": False,
        "optimizer": None,
        "source": "alpaca_direct_http",
        "lineage": {
            "base_api_version": "10.8.74",
            "base_branch": "research/api-v10.8.74-soft-horizon-consensus-batched-inference",
            "base_commit": "05b765df0496c905a4195948027a9b3b7adf2bce",
            "base_experiment": "raw-split-soft-horizon-consensus-v2",
            "base_source_job_id": "20260918T234903-52bd06f3",
            "change_scope": "market_data_transport_only",
        },
        "snapshot_sha256": (
            manifest["snapshot_sha256"]
        ),
        "config_sha256": config_sha256,
        "research_window": {
            "start": request.start_date,
            "end": request.end_date,
            "analysis_start": (
                request.analysis_start_date
            ),
            "analysis_end": (
                request.analysis_end_date
            ),
        },
        "data_methodology": {
            "bars": "fresh Alpaca RAW SIP daily bars",
            "corporate_actions": (
                "fresh Alpaca corporate-actions API snapshot"
            ),
            "split_normalization": (
                "local pre-ex-date price/volume unit normalization"
            ),
            "dividend_back_adjustment": False,
            "structural_identity_guard": True,
            "mongo_read": False,
            "mongo_write": False,
        },
        "model_methodology": {
            "lightgbm_parameters_frozen": True,
            "soft_horizon_consensus_parameters_frozen": True,
            "optuna_used": False,
            "caro_used": False,
            "deterministic_execution": (
                request.deterministic_execution
            ),
            "random_state": request.random_state,
        },
        "asset_count_requested": len(
            request.assets
        ),
        "asset_count_eligible": len(
            eligible
        ),
        "eligible_assets": eligible,
        "excluded_assets": exclusions,
        "runtime_versions": _runtime_versions(),
        "control": control_metrics,
        "soft_horizon_consensus": (
            challenger_metrics
        ),
        "reference_10_8_74": {
            "control_ending_capital": 5551143.963971565,
            "soft_horizon_consensus_ending_capital": 7376955.5577371465,
            "soft_horizon_consensus_changed_base_actions": 11,
        },
        "reproduction_audit": {
            "control_capital_delta_vs_10_8_74": (
                control_capital - 5551143.963971565
            ),
            "soft_capital_delta_vs_10_8_74": (
                challenger_capital - 7376955.5577371465
            ),
            "control_capital_ratio_vs_10_8_74": (
                control_capital / 5551143.963971565
            ),
            "soft_capital_ratio_vs_10_8_74": (
                challenger_capital / 7376955.5577371465
            ),
            "changed_base_actions_delta_vs_10_8_74": (
                int(
                    challenger_metrics.get(
                        "soft_horizon_consensus_changed_base_actions"
                    )
                    or 0
                )
                - 11
            ),
        },
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
                    control_metrics[
                        "cagr"
                    ]
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
        results_dir / "summary.json",
        summary,
    )

    print("")
    print(
        "[done] FINAL standalone Alpaca research completed",
        flush=True,
    )
    print(
        f"[done] snapshot_sha256="
        f"{manifest['snapshot_sha256']}",
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
        f"[done] reference_10.8.74 CONTROL=5,551,143.96 "
        f"delta={control_capital - 5551143.963971565:,.2f}",
        flush=True,
    )
    print(
        f"[done] reference_10.8.74 SOFT=7,376,955.56 "
        f"delta={challenger_capital - 7376955.5577371465:,.2f}",
        flush=True,
    )
    print(
        f"[done] output={output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
