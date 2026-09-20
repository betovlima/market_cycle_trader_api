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
import time
from typing import Any
from urllib.parse import quote

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
BARS_ENDPOINT = "https://data.alpaca.markets/v2/stocks/{symbol}/bars"
CORPORATE_ACTIONS_ENDPOINT = "https://data.alpaca.markets/v1/corporate-actions"
DEFAULT_CONFIG = ROOT / "config" / "final_research_alpaca_direct_v10875.json"
DEFAULT_OUTPUT = ROOT / "output" / "final_research_alpaca_direct"
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
    "reorganization",
    "partial_call",
    "capital_gains_distribution",
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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return _sha256_bytes(payload)


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


def _credentials() -> tuple[str, str]:
    key = str(
        os.getenv("ALPACA_API_KEY_ID") or ""
    ).strip()
    secret = str(
        os.getenv("ALPACA_SECRET_KEY") or ""
    ).strip()
    if not key or not secret:
        raise RuntimeError(
            "ALPACA_API_KEY_ID and ALPACA_SECRET_KEY must be "
            "present in the environment or project .env. "
            "MongoDB credentials are not used by this script."
        )
    return key, secret


def _headers() -> dict[str, str]:
    key, secret = _credentials()
    return {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }


def _request_json(
    url: str,
    *,
    headers: dict[str, str],
    params: dict[str, Any],
    retries: int = 6,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=60,
            )
            if response.status_code == 429:
                retry_after = response.headers.get(
                    "Retry-After"
                )
                wait = (
                    float(retry_after)
                    if retry_after
                    else min(60.0, 2.0 ** attempt)
                )
                print(
                    f"[alpaca] rate limit; sleeping {wait:.1f}s",
                    flush=True,
                )
                time.sleep(wait)
                continue
            if response.status_code >= 500:
                raise RuntimeError(
                    f"Alpaca server error {response.status_code}: "
                    f"{response.text[:500]}"
                )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError(
                    "Alpaca returned a non-object JSON response."
                )
            return payload
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(
                    min(30.0, 2.0 ** attempt)
                )
    raise RuntimeError(
        f"Alpaca request failed after {retries} attempts: "
        f"{last_error}"
    ) from last_error


def _bars_to_frame(
    bars: list[dict[str, Any]],
) -> pd.DataFrame:
    if not bars:
        return pd.DataFrame(
            columns=list(OHLCV)
        )

    rows = []
    for bar in bars:
        rows.append(
            {
                "timestamp": bar.get("t"),
                "open": bar.get("o"),
                "high": bar.get("h"),
                "low": bar.get("l"),
                "close": bar.get("c"),
                "volume": bar.get("v"),
            }
        )

    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"],
        utc=True,
        errors="coerce",
    )
    frame = (
        frame.dropna(subset=["timestamp"])
        .set_index("timestamp")
        .sort_index()
    )
    frame = frame[
        ~frame.index.duplicated(keep="last")
    ]
    for column in OHLCV:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )
    frame = frame.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna(
        subset=list(OHLCV)
    )
    return frame


def _download_raw_bars(
    *,
    symbol: str,
    start: str,
    end: str,
    timeframe: str,
    feed: str,
    headers: dict[str, str],
) -> pd.DataFrame:
    page_token: str | None = None
    bars: list[dict[str, Any]] = []

    while True:
        params: dict[str, Any] = {
            "timeframe": timeframe,
            "start": start,
            "end": end,
            "limit": 10_000,
            "adjustment": "raw",
            "feed": feed,
            "sort": "asc",
        }
        if page_token:
            params["page_token"] = page_token

        payload = _request_json(
            BARS_ENDPOINT.format(
                symbol=quote(symbol, safe="")
            ),
            headers=headers,
            params=params,
        )
        page_bars = payload.get("bars") or []
        if not isinstance(page_bars, list):
            raise RuntimeError(
                f"Unexpected bars response for {symbol}."
            )
        bars.extend(
            item
            for item in page_bars
            if isinstance(item, dict)
        )
        page_token = payload.get(
            "next_page_token"
        )
        if not page_token:
            break

    return _bars_to_frame(bars)


def _frame_csv_bytes(
    frame: pd.DataFrame,
) -> bytes:
    rendered = frame.reset_index().to_csv(
        index=False,
        float_format="%.12g",
        date_format="%Y-%m-%dT%H:%M:%S%z",
        lineterminator="\n",
    )
    return rendered.encode("utf-8")


def _flatten_corporate_actions(
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    groups = payload.get(
        "corporate_actions"
    ) or {}
    if not isinstance(groups, dict):
        return []

    result: list[dict[str, Any]] = []
    for array_name, values in groups.items():
        if not isinstance(values, list):
            continue
        action_type = ARRAY_TO_TYPE.get(
            str(array_name),
            str(array_name),
        )
        for item in values:
            if not isinstance(item, dict):
                continue
            document = dict(item)
            document["action_type"] = action_type
            document["source_array"] = str(
                array_name
            )
            result.append(document)
    return result


def _download_corporate_actions(
    *,
    symbols: list[str],
    research_start: str,
    end: str,
    headers: dict[str, str],
    chunk_size: int = 40,
) -> list[dict[str, Any]]:
    start = (
        date.fromisoformat(research_start)
        - timedelta(days=366)
    ).isoformat()

    documents: list[dict[str, Any]] = []
    chunk_size = max(
        1,
        min(100, int(chunk_size)),
    )
    for offset in range(
        0,
        len(symbols),
        chunk_size,
    ):
        chunk = symbols[
            offset: offset + chunk_size
        ]
        page_token: str | None = None
        while True:
            params: dict[str, Any] = {
                "symbols": ",".join(chunk),
                "types": ",".join(
                    REQUEST_TYPES
                ),
                "start": start,
                "end": end,
                "data_quality": "complete",
                "limit": 1000,
                "sort": "asc",
            }
            if page_token:
                params[
                    "page_token"
                ] = page_token

            payload = _request_json(
                CORPORATE_ACTIONS_ENDPOINT,
                headers=headers,
                params=params,
            )
            documents.extend(
                _flatten_corporate_actions(
                    payload
                )
            )
            page_token = payload.get(
                "next_page_token"
            )
            if not page_token:
                break

        print(
            "[alpaca] corporate actions "
            f"{min(offset + len(chunk), len(symbols))}"
            f"/{len(symbols)}",
            flush=True,
        )

    deduped: dict[str, dict[str, Any]] = {}
    for index, document in enumerate(
        documents
    ):
        key = str(
            document.get("id")
            or _canonical_json_sha256(
                document
            )
            or index
        )
        deduped[key] = document

    return sorted(
        deduped.values(),
        key=lambda item: (
            str(
                item.get(
                    "action_type"
                )
                or ""
            ),
            str(
                item.get("symbol")
                or item.get(
                    "source_symbol"
                )
                or ""
            ),
            str(
                item.get(
                    "process_date"
                )
                or ""
            ),
            str(
                item.get("id")
                or ""
            ),
        ),
    )


def _canonical_actions_bytes(
    documents: list[dict[str, Any]],
) -> bytes:
    lines = [
        json.dumps(
            item,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        for item in documents
    ]
    return (
        "\n".join(lines) + "\n"
    ).encode("utf-8")


def _actions_for_symbol(
    documents: list[dict[str, Any]],
    symbol: str,
) -> list[dict[str, Any]]:
    normalized = str(
        symbol
    ).strip().upper()
    keys = (
        "symbol",
        "source_symbol",
        "old_symbol",
        "new_symbol",
        "acquirer_symbol",
        "acquiree_symbol",
    )
    values = []
    for document in documents:
        if any(
            str(
                document.get(key)
                or ""
            ).strip().upper()
            == normalized
            for key in keys
        ):
            values.append(
                document
            )
    return sorted(
        values,
        key=lambda item: (
            str(
                item.get(
                    "process_date"
                )
                or ""
            ),
            str(
                item.get("ex_date")
                or ""
            ),
            str(
                item.get(
                    "effective_date"
                )
                or ""
            ),
        ),
    )


def _split_normalize(
    raw: pd.DataFrame,
    actions: list[dict[str, Any]],
) -> tuple[
    pd.DataFrame,
    list[dict[str, Any]],
]:
    result = raw.copy()
    session_dates = (
        pd.DatetimeIndex(
            result.index
        )
        .tz_convert("UTC")
        .normalize()
    )
    applied: list[dict[str, Any]] = []

    split_actions = [
        action
        for action in actions
        if (
            action.get("action_type")
            in {
                "forward_split",
                "reverse_split",
            }
            and action.get("ex_date")
            and action.get(
                "old_rate"
            )
            is not None
            and action.get(
                "new_rate"
            )
            is not None
        )
    ]
    split_actions.sort(
        key=lambda item: str(
            item.get("ex_date")
        )
    )

    for action in split_actions:
        ex_date = _utc(
            action["ex_date"]
        ).normalize()
        old_rate = float(
            action["old_rate"]
        )
        new_rate = float(
            action["new_rate"]
        )
        if not (
            old_rate > 0
            and new_rate > 0
        ):
            continue

        price_factor = (
            old_rate / new_rate
        )
        volume_factor = (
            new_rate / old_rate
        )
        mask = session_dates < ex_date
        if not mask.any():
            continue

        for column in (
            "open",
            "high",
            "low",
            "close",
        ):
            result.loc[
                mask,
                column,
            ] = (
                result.loc[
                    mask,
                    column,
                ].astype(float)
                * price_factor
            )
        result.loc[
            mask,
            "volume",
        ] = (
            result.loc[
                mask,
                "volume",
            ].astype(float)
            * volume_factor
        )
        applied.append(
            {
                "action_type": action.get(
                    "action_type"
                ),
                "ex_date": str(
                    action.get(
                        "ex_date"
                    )
                ),
                "process_date": str(
                    action.get(
                        "process_date"
                    )
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
            action.get(
                "action_type"
            )
            or ""
        )
        if action_type not in {
            "stock_merger",
            "stock_and_cash_merger",
            "cash_merger",
        }:
            continue
        acquiree = str(
            action.get(
                "acquiree_symbol"
            )
            or ""
        ).strip().upper()
        acquirer = str(
            action.get(
                "acquirer_symbol"
            )
            or ""
        ).strip().upper()
        if (
            acquiree == normalized
            and acquirer
            and acquirer != normalized
        ):
            return {
                "reason": (
                    "structural_identity_change"
                ),
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


def _load_request(
    path: Path,
) -> tuple[
    BacktestExecutionRequest,
    dict[str, Any],
]:
    document = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )
    raw_request = (
        document.get("request")
        if isinstance(
            document,
            dict,
        )
        else None
    )
    if not isinstance(
        raw_request,
        dict,
    ):
        raise ValueError(
            "Config must contain a "
            "top-level 'request' object."
        )

    request = (
        BacktestExecutionRequest
        .model_validate(
            raw_request
        )
    )
    return request, document


def _download_snapshot(
    *,
    request: BacktestExecutionRequest,
    snapshot_dir: Path,
    replace: bool,
    resume: bool,
) -> dict[str, Any]:
    if snapshot_dir.exists():
        if replace:
            shutil.rmtree(snapshot_dir)
        elif not resume:
            raise RuntimeError(
                f"Snapshot directory already exists: {snapshot_dir}. "
                "Use --replace-snapshot for a new direct Alpaca download, "
                "--resume-download to continue an interrupted download, "
                "or --reuse-snapshot to run an existing completed snapshot."
            )

    bars_dir = snapshot_dir / "bars"
    bars_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    headers = _headers()
    downloaded_at = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )
    end = str(
        request.analysis_end_date
        or request.end_date
    )
    if not end:
        raise ValueError(
            "Final research requires a fixed closed analysis_end_date."
        )

    bar_manifest: dict[
        str,
        dict[str, Any],
    ] = {}
    for position, symbol in enumerate(
        request.assets,
        start=1,
    ):
        target = bars_dir / f"{symbol}.csv"
        if resume and target.is_file() and target.stat().st_size > 0:
            print(
                f"[alpaca] bars {position}/{len(request.assets)} "
                f"{symbol} resume=existing",
                flush=True,
            )
            payload = target.read_bytes()
            resumed = pd.read_csv(target)
            if "timestamp" in resumed.columns:
                resumed["timestamp"] = pd.to_datetime(
                    resumed["timestamp"],
                    utc=True,
                    errors="coerce",
                )
                resumed = resumed.dropna(subset=["timestamp"])
            bar_manifest[symbol] = {
                "rows": int(len(resumed)),
                "first_timestamp": (
                    str(resumed["timestamp"].min())
                    if len(resumed)
                    else None
                ),
                "last_timestamp": (
                    str(resumed["timestamp"].max())
                    if len(resumed)
                    else None
                ),
                "sha256": _sha256_bytes(payload),
                "file": str(target.relative_to(snapshot_dir)),
                "resumed": True,
            }
            continue

        print(
            f"[alpaca] bars "
            f"{position}/{len(request.assets)} "
            f"{symbol}",
            flush=True,
        )
        frame = _download_raw_bars(
            symbol=symbol,
            start=request.start_date,
            end=end,
            timeframe=request.timeframe,
            feed=request.alpaca_historical_feed,
            headers=headers,
        )
        payload = _frame_csv_bytes(
            frame
        )
        target.write_bytes(
            payload
        )
        bar_manifest[
            symbol
        ] = {
            "rows": int(
                len(frame)
            ),
            "first_timestamp": (
                str(frame.index.min())
                if not frame.empty
                else None
            ),
            "last_timestamp": (
                str(frame.index.max())
                if not frame.empty
                else None
            ),
            "sha256": (
                _sha256_bytes(
                    payload
                )
            ),
            "file": str(
                target.relative_to(
                    snapshot_dir
                )
            ),
            "resumed": False,
        }
        _checkpoint(
            snapshot_dir / "download_state.json",
            {
                "schema_version": 1,
                "api_version": API_VERSION,
                "completed_bars": list(bar_manifest),
                "completed_count": len(bar_manifest),
                "asset_count": len(request.assets),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    actions = (
        _download_corporate_actions(
            symbols=list(
                request.assets
            ),
            research_start=(
                request.start_date
            ),
            end=end,
            headers=headers,
        )
    )
    actions_payload = (
        _canonical_actions_bytes(
            actions
        )
    )
    actions_file = (
        snapshot_dir
        / "corporate_actions.jsonl"
    )
    actions_file.write_bytes(
        actions_payload
    )

    request_payload = (
        request.model_dump(
            mode="json"
        )
    )
    request_sha = (
        _canonical_json_sha256(
            request_payload
        )
    )
    actions_sha = (
        _sha256_bytes(
            actions_payload
        )
    )
    combined_sha = (
        _canonical_json_sha256(
            {
                "request_sha256": request_sha,
                "bars": {
                    symbol: item[
                        "sha256"
                    ]
                    for symbol, item
                    in sorted(
                        bar_manifest.items()
                    )
                },
                "corporate_actions_sha256": actions_sha,
            }
        )
    )

    manifest = {
        "schema_version": 1,
        "api_version": API_VERSION,
        "source": (
            "Alpaca Market Data API direct"
        ),
        "database_used": False,
        "downloaded_at": downloaded_at,
        "bars_endpoint": BARS_ENDPOINT,
        "corporate_actions_endpoint": (
            CORPORATE_ACTIONS_ENDPOINT
        ),
        "timeframe": request.timeframe,
        "feed": request.alpaca_historical_feed,
        "bars_adjustment": "raw",
        "start_date": request.start_date,
        "analysis_end_date": end,
        "asset_count": len(
            request.assets
        ),
        "request_sha256": request_sha,
        "bars": bar_manifest,
        "corporate_actions": {
            "document_count": len(
                actions
            ),
            "sha256": actions_sha,
            "file": (
                "corporate_actions.jsonl"
            ),
            "types": list(
                REQUEST_TYPES
            ),
            "data_quality": (
                "complete"
            ),
        },
        "snapshot_sha256": combined_sha,
    }
    _checkpoint(
        snapshot_dir
        / "manifest.json",
        manifest,
    )
    state_path = snapshot_dir / "download_state.json"
    if state_path.exists():
        state_path.unlink()
    return manifest


def _verify_snapshot(
    snapshot_dir: Path,
) -> dict[str, Any]:
    manifest_path = (
        snapshot_dir
        / "manifest.json"
    )
    if not manifest_path.is_file():
        raise RuntimeError(
            "Snapshot manifest is missing: "
            f"{manifest_path}"
        )
    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    for symbol, item in (
        manifest.get("bars")
        or {}
    ).items():
        path = (
            snapshot_dir
            / str(item["file"])
        )
        if not path.is_file():
            raise RuntimeError(
                f"Missing bar file for {symbol}: {path}"
            )
        actual = _sha256_bytes(
            path.read_bytes()
        )
        if actual != item.get(
            "sha256"
        ):
            raise RuntimeError(
                f"SHA-256 mismatch for {symbol}."
            )

    ca = (
        manifest.get(
            "corporate_actions"
        )
        or {}
    )
    ca_path = (
        snapshot_dir
        / str(
            ca.get(
                "file",
                "corporate_actions.jsonl",
            )
        )
    )
    if not ca_path.is_file():
        raise RuntimeError(
            "Corporate-actions file is missing."
        )
    if _sha256_bytes(
        ca_path.read_bytes()
    ) != ca.get("sha256"):
        raise RuntimeError(
            "Corporate-actions SHA-256 mismatch."
        )

    expected_snapshot = (
        _canonical_json_sha256(
            {
                "request_sha256": manifest[
                    "request_sha256"
                ],
                "bars": {
                    symbol: item[
                        "sha256"
                    ]
                    for symbol, item
                    in sorted(
                        (
                            manifest.get(
                                "bars"
                            )
                            or {}
                        ).items()
                    )
                },
                "corporate_actions_sha256": ca[
                    "sha256"
                ],
            }
        )
    )
    if expected_snapshot != manifest.get(
        "snapshot_sha256"
    ):
        raise RuntimeError(
            "Snapshot manifest SHA-256 is inconsistent."
        )
    return manifest


def _load_snapshot_frames(
    *,
    request: BacktestExecutionRequest,
    snapshot_dir: Path,
) -> tuple[
    dict[str, pd.DataFrame],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    actions_path = (
        snapshot_dir
        / "corporate_actions.jsonl"
    )
    actions = []
    for line in actions_path.read_text(
        encoding="utf-8"
    ).splitlines():
        if line.strip():
            actions.append(
                json.loads(line)
            )

    frames: dict[
        str,
        pd.DataFrame,
    ] = {}
    exclusions: list[
        dict[str, Any]
    ] = []
    diagnostics: list[
        dict[str, Any]
    ] = []

    for position, symbol in enumerate(
        request.assets,
        start=1,
    ):
        path = (
            snapshot_dir
            / "bars"
            / f"{symbol}.csv"
        )
        if not path.is_file():
            exclusions.append(
                {
                    "symbol": symbol,
                    "reason": (
                        "missing_local_raw_file"
                    ),
                }
            )
            continue

        frame = pd.read_csv(
            path
        )
        if frame.empty:
            exclusions.append(
                {
                    "symbol": symbol,
                    "reason": (
                        "missing_raw_history"
                    ),
                }
            )
            continue

        frame["timestamp"] = (
            pd.to_datetime(
                frame["timestamp"],
                utc=True,
                errors="coerce",
            )
        )
        frame = (
            frame.dropna(
                subset=["timestamp"]
            )
            .set_index("timestamp")
            .sort_index()
        )
        for column in OHLCV:
            frame[column] = (
                pd.to_numeric(
                    frame[column],
                    errors="coerce",
                )
            )
        frame = frame.replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna(
            subset=list(OHLCV)
        )

        symbol_actions = (
            _actions_for_symbol(
                actions,
                symbol,
            )
        )
        issue = (
            _structural_identity_issue(
                symbol,
                symbol_actions,
            )
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

        normalized, applied = (
            _split_normalize(
                frame,
                symbol_actions,
            )
        )
        frames[
            symbol
        ] = normalized
        diagnostics.append(
            {
                "symbol": symbol,
                "raw_rows": len(
                    frame
                ),
                "corporate_actions": (
                    len(
                        symbol_actions
                    )
                ),
                "splits_applied": len(
                    applied
                ),
                "first_timestamp": (
                    str(
                        frame.index.min()
                    )
                ),
                "last_timestamp": (
                    str(
                        frame.index.max()
                    )
                ),
            }
        )
        print(
            f"[data] {position}/{len(request.assets)} "
            f"{symbol} rows={len(frame)} "
            f"actions={len(symbol_actions)} "
            f"splits={len(applied)}",
            flush=True,
        )

    return (
        frames,
        exclusions,
        diagnostics,
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
            float(
                row[
                    "strategy_return"
                ]
            )
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
        "strategy_return": (
            strategy_return
        ),
        "cagr": strategy_cagr,
        "sharpe": strategy_sharpe,
        "maximum_drawdown": (
            strategy_maxdd
        ),
        "worst_fold_return": (
            worst_fold_return
        ),
        "folds": fold_rows,
        "buy_hold_ending_capital": (
            buy_hold_ending
        ),
        "buy_hold_return": (
            buy_hold_return
        ),
        "buy_hold_cagr": (
            buy_hold_cagr
        ),
        "buy_hold_sharpe": (
            buy_hold_sharpe
        ),
        "buy_hold_maximum_drawdown": (
            buy_hold_maxdd
        ),
        "benchmark_name": (
            result.metrics.get(
                "benchmark_name"
            )
        ),
        "strategy_vs_buy_hold_capital_ratio": (
            strategy_ending
            / buy_hold_ending
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
        "predictive_diagnostics": (
            predictive
        ),
        "simulation_profile": (
            simulation_profile
        ),
        "simulation_total_seconds": (
            simulation_profile.get(
                "total_seconds"
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
    }
    metrics.update(
        {
            key: value
            for key, value
            in result.metrics.items()
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
        float(
            config.initial_capital
        ),
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


def _comparison_row(
    label: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    fields = {
        "variant": label,
        "ending_capital": metrics.get(
            "ending_capital"
        ),
        "return": metrics.get(
            "strategy_return"
        ),
        "cagr": metrics.get("cagr"),
        "sharpe": metrics.get(
            "sharpe"
        ),
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
        "strategy_vs_buy_hold_capital_ratio": metrics.get(
            "strategy_vs_buy_hold_capital_ratio"
        ),
        "simulation_total_seconds": metrics.get(
            "simulation_total_seconds"
        ),
        "simulation_policy_seconds": metrics.get(
            "simulation_policy_seconds"
        ),
        "oos_inference_cache_build_seconds": metrics.get(
            "oos_inference_cache_build_seconds"
        ),
        "soft_horizon_consensus_changed_base_actions": metrics.get(
            "soft_horizon_consensus_changed_base_actions"
        ),
        "soft_horizon_consensus_change_rate": metrics.get(
            "soft_horizon_consensus_change_rate"
        ),
        "soft_horizon_consensus_average_support": metrics.get(
            "soft_horizon_consensus_average_support"
        ),
    }
    return fields


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Final standalone MCT research: download fresh RAW SIP bars "
            "and corporate actions directly from Alpaca, persist a local "
            "checksummed snapshot, then run Control vs Soft Horizon "
            "Consensus without reading market data or configuration from MongoDB."
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
        "--penalty-strength",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
    )
    parser.add_argument(
        "--reuse-snapshot",
        action="store_true",
    )
    parser.add_argument(
        "--resume-download",
        action="store_true",
    )
    parser.add_argument(
        "--replace-snapshot",
        action="store_true",
    )
    args = parser.parse_args()

    selected_snapshot_modes = sum(
        bool(value)
        for value in (
            args.reuse_snapshot,
            args.resume_download,
            args.replace_snapshot,
        )
    )
    if selected_snapshot_modes > 1:
        raise ValueError(
            "--reuse-snapshot, --resume-download and --replace-snapshot "
            "are mutually exclusive."
        )
    if float(
        args.penalty_strength
    ) < 0.0:
        raise ValueError(
            "--penalty-strength cannot be negative."
        )

    config_path = Path(
        args.config
    ).expanduser().resolve()
    output_dir = Path(
        args.output_dir
    ).expanduser().resolve()
    snapshot_dir = (
        output_dir
        / "snapshot"
    )
    results_dir = (
        output_dir
        / "results"
    )
    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    request, config_document = (
        _load_request(
            config_path
        )
    )
    if allocation_execution_enabled(
        request
    ):
        raise ValueError(
            "Final v10.8.75 research is limited to the canonical "
            "single-position rotation policy."
        )

    if args.reuse_snapshot:
        manifest = _verify_snapshot(
            snapshot_dir
        )
    else:
        manifest = _download_snapshot(
            request=request,
            snapshot_dir=snapshot_dir,
            replace=bool(args.replace_snapshot),
            resume=bool(args.resume_download),
        )
        manifest = _verify_snapshot(
            snapshot_dir
        )

    _checkpoint(
        results_dir
        / "request_used.json",
        {
            "schema_version": 1,
            "api_version": API_VERSION,
            "source_file": str(
                config_path
            ),
            "config_document": config_document,
            "validated_request": (
                request.model_dump(
                    mode="json"
                )
            ),
            "request_sha256": (
                manifest[
                    "request_sha256"
                ]
            ),
            "snapshot_sha256": (
                manifest[
                    "snapshot_sha256"
                ]
            ),
        },
    )

    if args.download_only:
        print("")
        print(
            "[final] download-only completed",
            flush=True,
        )
        print(
            f"[final] snapshot={snapshot_dir}",
            flush=True,
        )
        print(
            f"[final] snapshot_sha256="
            f"{manifest['snapshot_sha256']}",
            flush=True,
        )
        return 0

    frames, exclusions, diagnostics = (
        _load_snapshot_frames(
            request=request,
            snapshot_dir=snapshot_dir,
        )
    )
    if len(frames) < 2:
        raise RuntimeError(
            "Fewer than two eligible assets remain after "
            "fresh-data and structural-identity checks."
        )

    eligible = list(frames)
    anchors = [
        item
        for item
        in request.calendar_anchor_assets
        if item in frames
    ]
    if len(anchors) < 2:
        raise RuntimeError(
            "Fresh snapshot removed too many calendar anchors."
        )

    references = [
        item
        for item
        in request.research_reference_assets
        if item in frames
    ]
    if len(references) < 2:
        references = list(
            anchors
        )

    reference_set = set(
        references
    )
    candidates = [
        item
        for item
        in request.research_candidate_assets
        if (
            item in frames
            and item
            not in reference_set
        )
    ]

    base_settings = deepcopy(
        request.research_model_settings
    )
    lightgbm = deepcopy(
        base_settings.get(
            "lightgbm"
        )
        or {}
    )
    lightgbm[
        "early_stopping_enabled"
    ] = False
    base_settings[
        "lightgbm"
    ] = lightgbm
    base_settings[
        "horizon_voting"
    ] = {
        "enabled": False,
    }
    base_settings[
        "soft_horizon_consensus"
    ] = {
        "enabled": False,
    }

    base_config = (
        request.model_copy(
            update={
                "research_model_settings": (
                    base_settings
                ),
                "assets": eligible,
                "calendar_anchor_assets": anchors,
                "research_reference_assets": references,
                "research_candidate_assets": candidates,
                "market_data_provider": "alpaca",
                "alpaca_adjustment": "split",
                "market_data_history_backfill_enabled": False,
                "mongo_cache_enabled": False,
                "expected_market_data_signature_sha256": None,
                "research_market_data_snapshot_id": None,
            }
        )
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

    (
        challenger_result,
        challenger_metrics,
    ) = _run_variant(
        label=(
            "SOFT_HORIZON_CONSENSUS"
        ),
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

    pd.DataFrame(
        diagnostics
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
        results_dir
        / "strategy_comparison.csv",
        index=False,
    )

    control_capital = float(
        control_metrics[
            "ending_capital"
        ]
    )
    challenger_capital = float(
        challenger_metrics[
            "ending_capital"
        ]
    )
    summary = {
        "schema_version": 1,
        "api_version": API_VERSION,
        "experiment": (
            "final-alpaca-direct-soft-horizon-consensus-v1"
        ),
        "database_used": False,
        "data_source": (
            "Alpaca Market Data API direct"
        ),
        "snapshot": {
            "directory": str(
                snapshot_dir
            ),
            "snapshot_sha256": (
                manifest[
                    "snapshot_sha256"
                ]
            ),
            "downloaded_at": (
                manifest[
                    "downloaded_at"
                ]
            ),
            "feed": (
                manifest["feed"]
            ),
            "bars_adjustment": (
                manifest[
                    "bars_adjustment"
                ]
            ),
            "corporate_actions_data_quality": (
                manifest[
                    "corporate_actions"
                ][
                    "data_quality"
                ]
            ),
        },
        "eligible_assets": eligible,
        "excluded_assets": exclusions,
        "penalty_strength": float(
            args.penalty_strength
        ),
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
        "methodology": {
            "fresh_download": True,
            "mongo_market_data": False,
            "mongo_configuration": False,
            "raw_bars": True,
            "local_split_normalization": True,
            "structural_identity_guard": True,
            "soft_horizon_consensus": True,
            "batched_oos_inference": True,
            "same_cost_model": True,
            "buy_and_hold_reported": True,
            "fixed_analysis_end_date": (
                request.analysis_end_date
            ),
        },
    }
    _checkpoint(
        results_dir
        / "summary.json",
        summary,
    )

    print("")
    print(
        "[final] research completed",
        flush=True,
    )
    print(
        f"[final] database_used=false",
        flush=True,
    )
    print(
        f"[final] snapshot_sha256="
        f"{manifest['snapshot_sha256']}",
        flush=True,
    )
    print(
        f"[final] CONTROL="
        f"{control_capital:,.2f}",
        flush=True,
    )
    print(
        "[final] SOFT_HORIZON_CONSENSUS="
        f"{challenger_capital:,.2f}",
        flush=True,
    )
    print(
        f"[final] results={results_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
