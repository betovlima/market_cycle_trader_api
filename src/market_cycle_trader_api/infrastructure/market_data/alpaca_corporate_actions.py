from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Callable

import requests


ENDPOINT = "https://data.alpaca.markets/v1/corporate-actions"
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


def _headers(api_key_id: str, secret_key: str) -> dict[str, str]:
    key = str(api_key_id or "").strip()
    secret = str(secret_key or "").strip()
    if not key or not secret:
        raise RuntimeError("Alpaca API credentials are not configured.")
    return {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }


def _flatten_page(payload: dict[str, Any]) -> list[dict[str, Any]]:
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


def _request_page(
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
        "start": str(start),
        "end": str(end),
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
                ENDPOINT,
                headers=headers,
                params=params,
                timeout=30,
            )
            if response.status_code == 429 and attempt < 3:
                time.sleep(2**attempt)
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
                time.sleep(2**attempt)

    raise RuntimeError(
        "Alpaca corporate-actions request failed: "
        f"{last_error}"
    ) from last_error


def download_corporate_actions(
    *,
    api_key_id: str,
    secret_key: str,
    symbols: list[str],
    start: str,
    end: str,
    chunk_size: int = 40,
    progress_callback: Callable[[int, int, list[str]], None] | None = None,
) -> list[dict[str, Any]]:
    unique_symbols = sorted(
        {
            str(symbol).strip().upper()
            for symbol in symbols
            if str(symbol).strip()
        }
    )
    if not unique_symbols:
        return []

    headers = _headers(api_key_id, secret_key)
    resolved_chunk_size = max(
        1,
        min(100, int(chunk_size)),
    )
    result: list[dict[str, Any]] = []

    for offset in range(
        0,
        len(unique_symbols),
        resolved_chunk_size,
    ):
        chunk = unique_symbols[
            offset : offset + resolved_chunk_size
        ]
        page_token: str | None = None

        while True:
            payload = _request_page(
                headers=headers,
                symbols=chunk,
                start=start,
                end=end,
                page_token=page_token,
            )
            result.extend(_flatten_page(payload))
            page_token = payload.get("next_page_token")
            if not page_token:
                break

        if progress_callback is not None:
            progress_callback(
                min(
                    offset + len(chunk),
                    len(unique_symbols),
                ),
                len(unique_symbols),
                chunk,
            )

    return sorted(
        result,
        key=lambda item: (
            str(item.get("action_type") or ""),
            str(
                item.get("symbol")
                or item.get("source_symbol")
                or item.get("old_symbol")
                or item.get("new_symbol")
                or item.get("acquiree_symbol")
                or item.get("acquirer_symbol")
                or ""
            ),
            str(item.get("process_date") or ""),
            str(item.get("ex_date") or ""),
            str(item.get("effective_date") or ""),
            str(item.get("id") or ""),
        ),
    )


def corporate_actions_for_symbol(
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
    result = [
        action
        for action in actions
        if any(
            str(action.get(field) or "").strip().upper()
            == normalized
            for field in symbol_fields
        )
    ]
    return sorted(
        result,
        key=lambda item: (
            str(item.get("process_date") or ""),
            str(item.get("ex_date") or ""),
            str(item.get("effective_date") or ""),
            str(item.get("id") or ""),
        ),
    )


def corporate_actions_sha256(
    actions: list[dict[str, Any]],
) -> str:
    canonical = json.dumps(
        actions,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
