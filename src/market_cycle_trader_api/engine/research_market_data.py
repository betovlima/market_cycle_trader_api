"""Research-only RAW market-data protocol aligned with the standalone TCC.

Paper/live market-data behavior is intentionally left unchanged. Backtest
Simulation and frozen Model Tuning snapshots can opt into RAW Alpaca bars,
causal split normalization and structural identity exclusion.
"""

from __future__ import annotations

from datetime import date, timedelta
import time
from typing import Any

import pandas as pd
import requests

from ..infrastructure.persistence.mongo_repository import (
    ALPACA_CORPORATE_ACTIONS_COLLECTION,
    create_client,
    get_alpaca_credentials,
    get_database,
    utc_now,
)
from .market_data import (
    _attach_provenance,
    effective_execution_end_date,
    load_market_bars,
    load_mongo_market_bars,
)

RAW_TOTAL_CAUSAL_PROTOCOL = "raw_total_causal_v1"
LEGACY_ADJUSTED_PROTOCOL = "legacy_adjusted"

CORPORATE_ACTIONS_ENDPOINT = "https://data.alpaca.markets/v1/corporate-actions"
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


class StructuralResearchAssetExclusion(RuntimeError):
    """Raised when a ticker cannot represent one continuous research identity."""

    def __init__(self, details: dict[str, Any]):
        self.details = dict(details)
        symbol = str(details.get("symbol") or "UNKNOWN")
        action_type = str(details.get("action_type") or "unknown")
        acquirer = str(details.get("acquirer_symbol") or "unknown")
        super().__init__(
            f"{symbol}: structural identity change ({action_type}) "
            f"to {acquirer}; excluded from the research universe."
        )


def effective_research_config(config: Any) -> Any:
    """Return the homologated research config without mutating operational config."""
    protocol = str(
        getattr(config, "research_market_data_protocol", RAW_TOTAL_CAUSAL_PROTOCOL)
        or RAW_TOTAL_CAUSAL_PROTOCOL
    )
    if protocol != RAW_TOTAL_CAUSAL_PROTOCOL:
        return config

    settings = dict(getattr(config, "research_model_settings", {}) or {})
    lightgbm = dict(settings.get("lightgbm") or {})
    if lightgbm:
        lightgbm["n_jobs"] = 1
        lightgbm["early_stopping_enabled"] = False
        settings["lightgbm"] = lightgbm

    return config.model_copy(
        update={
            "alpaca_adjustment": "raw",
            "deterministic_execution": True,
            "numeric_thread_limit": 1,
            "research_model_settings": settings,
            "research_market_data_refresh_mode": "full",
        }
    )


def _request_json(
    *,
    headers: dict[str, str],
    params: dict[str, Any],
    max_attempts: int = 6,
) -> dict[str, Any]:
    delay = 2.0
    for attempt in range(1, max_attempts + 1):
        response = requests.get(
            CORPORATE_ACTIONS_ENDPOINT,
            headers=headers,
            params=params,
            timeout=60,
        )
        if response.status_code == 200:
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError(
                    "Alpaca Corporate Actions returned a non-object response."
                )
            return payload
        if response.status_code == 429 or 500 <= response.status_code < 600:
            if attempt == max_attempts:
                break
            retry_after = response.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else delay
            time.sleep(wait)
            delay = min(delay * 2.0, 30.0)
            continue
        raise RuntimeError(
            "Alpaca Corporate Actions HTTP "
            f"{response.status_code}: {response.text[:500]}"
        )
    raise RuntimeError(
        "Unable to retrieve Alpaca Corporate Actions after repeated attempts."
    )


def _flatten_corporate_actions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    groups = payload.get("corporate_actions") or {}
    if not isinstance(groups, dict):
        return documents
    for array_name, values in groups.items():
        if not isinstance(values, list):
            continue
        action_type = ARRAY_TO_TYPE.get(str(array_name), str(array_name))
        for value in values:
            if isinstance(value, dict):
                item = dict(value)
                item["action_type"] = action_type
                item["source_array"] = str(array_name)
                documents.append(item)
    sort_fields = (
        "action_type",
        "symbol",
        "source_symbol",
        "process_date",
        "ex_date",
        "effective_date",
        "id",
    )
    documents.sort(
        key=lambda item: tuple(
            str(item.get(field) or "") for field in sort_fields
        )
    )
    return documents


def _action_matches_symbol(item: dict[str, Any], symbol: str) -> bool:
    normalized = str(symbol).strip().upper()
    fields = (
        "symbol",
        "source_symbol",
        "old_symbol",
        "new_symbol",
        "acquirer_symbol",
        "acquiree_symbol",
    )
    return any(
        str(item.get(field) or "").strip().upper() == normalized
        for field in fields
    )


def _download_corporate_actions(
    symbol: str,
    config: Any,
) -> tuple[list[dict[str, Any]], str, str]:
    credentials = get_alpaca_credentials()
    query_start = (
        date.fromisoformat(str(config.start_date)) - timedelta(days=366)
    ).isoformat()
    query_end = str(effective_execution_end_date(config) or "")
    if not query_end:
        raise RuntimeError(
            "A locked research end date is required for Corporate Actions."
        )

    headers = {
        "APCA-API-KEY-ID": credentials["api_key_id"],
        "APCA-API-SECRET-KEY": credentials["secret_key"],
    }
    documents: list[dict[str, Any]] = []
    page_token: str | None = None
    while True:
        params: dict[str, Any] = {
            "symbols": str(symbol).strip().upper(),
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
        payload = _request_json(headers=headers, params=params)
        documents.extend(_flatten_corporate_actions(payload))
        page_token = payload.get("next_page_token")
        if not page_token:
            break

    filtered = [
        item
        for item in documents
        if _action_matches_symbol(item, symbol)
    ]
    return filtered, query_start, query_end


def _load_research_corporate_actions(
    symbol: str,
    config: Any,
) -> list[dict[str, Any]]:
    normalized = str(symbol).strip().upper()
    protocol = RAW_TOTAL_CAUSAL_PROTOCOL
    target_end = str(effective_execution_end_date(config) or "")
    if not target_end:
        raise RuntimeError(
            "A locked research end date is required for Corporate Actions."
        )

    client = create_client()
    try:
        collection = get_database(client)[ALPACA_CORPORATE_ACTIONS_COLLECTION]
        cached = collection.find_one(
            {"symbol": normalized, "protocol": protocol},
            {"_id": 0},
        )
        access_mode = str(
            getattr(config, "research_market_data_mode", "database_only")
        )
        refresh_mode = str(
            getattr(config, "research_market_data_refresh_mode", "reuse")
        )
        if cached is not None and (
            access_mode != "backtest_bootstrap_missing"
            or refresh_mode != "full"
        ):
            cached_end = str(cached.get("query_end") or "")
            if cached_end and cached_end >= target_end:
                return [
                    dict(item)
                    for item in list(cached.get("actions") or [])
                    if isinstance(item, dict)
                ]

        if access_mode != "backtest_bootstrap_missing":
            raise RuntimeError(
                "CorporateActionsMissingInMongoDB: RAW total-causal research "
                f"actions for {normalized} are not cached through {target_end}. "
                "Model tuning is database-only and never downloads market data."
            )

        actions, query_start, query_end = _download_corporate_actions(
            normalized,
            config,
        )
        collection.replace_one(
            {"symbol": normalized, "protocol": protocol},
            {
                "symbol": normalized,
                "protocol": protocol,
                "source": "alpaca",
                "query_start": query_start,
                "query_end": query_end,
                "actions": actions,
                "action_count": len(actions),
                "updated_at": utc_now(),
            },
            upsert=True,
        )
        return actions
    finally:
        client.close()


def structural_identity_issue(
    symbol: str,
    actions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    normalized = str(symbol).strip().upper()
    for action in actions:
        action_type = str(action.get("action_type") or "")
        if action_type not in {
            "stock_merger",
            "stock_and_cash_merger",
            "cash_merger",
        }:
            continue
        acquiree = (
            str(action.get("acquiree_symbol") or "").strip().upper()
        )
        acquirer = (
            str(action.get("acquirer_symbol") or "").strip().upper()
        )
        if acquiree == normalized and acquirer and acquirer != normalized:
            return {
                "symbol": normalized,
                "reason": "structural_identity_change",
                "action_type": action_type,
                "process_date": action.get("process_date"),
                "effective_date": action.get("effective_date"),
                "acquiree_symbol": acquiree,
                "acquirer_symbol": acquirer,
            }
    return None


def split_normalize(
    raw: pd.DataFrame,
    actions: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Replicate the split normalization used by the homologated TCC main."""
    result = raw.copy()
    source_attrs = dict(getattr(raw, "attrs", {}))
    session_dates = pd.DatetimeIndex(result.index).tz_convert("UTC").normalize()
    applied: list[dict[str, Any]] = []

    splits = [
        action
        for action in actions
        if action.get("action_type") in {"forward_split", "reverse_split"}
        and action.get("ex_date")
        and action.get("old_rate") is not None
        and action.get("new_rate") is not None
    ]
    splits.sort(key=lambda item: str(item.get("ex_date")))

    for action in splits:
        ex_date = pd.Timestamp(action["ex_date"])
        ex_date = (
            ex_date.tz_localize("UTC")
            if ex_date.tzinfo is None
            else ex_date.tz_convert("UTC")
        ).normalize()
        old_rate = float(action["old_rate"])
        new_rate = float(action["new_rate"])
        if old_rate <= 0.0 or new_rate <= 0.0:
            continue

        price_factor = old_rate / new_rate
        volume_factor = new_rate / old_rate
        mask = session_dates < ex_date
        if not mask.any():
            continue

        for column in ("open", "high", "low", "close"):
            result.loc[mask, column] = (
                pd.to_numeric(result.loc[mask, column], errors="coerce")
                * price_factor
            )
        result.loc[mask, "volume"] = (
            pd.to_numeric(result.loc[mask, "volume"], errors="coerce")
            * volume_factor
        )
        applied.append(
            {
                "action_type": action.get("action_type"),
                "ex_date": str(action.get("ex_date")),
                "process_date": str(action.get("process_date")),
                "old_rate": old_rate,
                "new_rate": new_rate,
                "price_factor": price_factor,
                "volume_factor": volume_factor,
                "normalization_direction": "pre_ex_date_history",
            }
        )

    result.attrs.update(source_attrs)
    return result, applied


def _dividend_events(
    actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        dict(action)
        for action in actions
        if str(action.get("action_type") or "")
        in {"cash_dividend", "stock_dividend"}
    ]


def load_research_market_bars(symbol: str, config: Any) -> pd.DataFrame:
    """Load research data without changing paper/live market-data semantics."""
    protocol = str(
        getattr(config, "research_market_data_protocol", RAW_TOTAL_CAUSAL_PROTOCOL)
        or RAW_TOTAL_CAUSAL_PROTOCOL
    )
    if protocol == LEGACY_ADJUSTED_PROTOCOL:
        return load_market_bars(symbol, config)
    if protocol != RAW_TOTAL_CAUSAL_PROTOCOL:
        raise ValueError(f"Unsupported research market-data protocol: {protocol}")

    raw_config = effective_research_config(config)
    frame = load_mongo_market_bars(symbol, raw_config)

    # Frozen tuning snapshots already contain the processed research frame.
    if str(
        getattr(raw_config, "research_market_data_snapshot_id", None) or ""
    ).strip():
        provenance = dict(
            frame.attrs.get("market_data_provenance", {})
        )
        provenance["research_market_data_protocol"] = protocol
        provenance["effective_adjustment"] = "raw"
        return _attach_provenance(frame, provenance)

    actions = _load_research_corporate_actions(symbol, raw_config)
    issue = structural_identity_issue(symbol, actions)
    if issue is not None:
        raise StructuralResearchAssetExclusion(issue)

    normalized, applied = split_normalize(frame, actions)
    dividends = _dividend_events(actions)
    provenance = dict(
        normalized.attrs.get("market_data_provenance", {})
    )
    provenance.update(
        {
            "research_market_data_protocol": protocol,
            "source_adjustment": "raw",
            "effective_adjustment": "raw_plus_causal_split_normalization",
            "corporate_action_source": "alpaca",
            "corporate_action_count": int(len(actions)),
            "splits_applied": int(len(applied)),
            "split_events": applied,
            "split_normalization_direction": "pre_ex_date_history",
            "split_normalization_uses_future_events": True,
            "dividend_event_count": int(len(dividends)),
            "dividend_adjustment_applied": False,
            "dividend_events_used_by_model": False,
            "structural_identity_verified": True,
        }
    )
    return _attach_provenance(normalized, provenance)
