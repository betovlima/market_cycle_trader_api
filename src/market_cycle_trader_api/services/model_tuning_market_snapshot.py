from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..engine.market_data import load_market_bars, validate_and_clean_bars
from ..engine.market_data_snapshot import (
    TUNING_MARKET_SNAPSHOT_SCHEMA_VERSION,
    encode_market_frame,
    json_safe_provenance,
)
from ..infrastructure.persistence.mongo_repository import (
    MODEL_TUNING_MARKET_SNAPSHOTS_COLLECTION,
    bson_value,
    utc_now,
)
from ..schemas.requests import BacktestExecutionRequest
from .reproducibility import market_data_manifest


class TuningMarketSnapshotMismatch(RuntimeError):
    pass


class TuningMarketSnapshotMissing(RuntimeError):
    pass


def market_snapshot_exists(db: Any, snapshot_id: str) -> bool:
    normalized = str(snapshot_id or "").strip().lower()
    if not normalized:
        return False
    return db[MODEL_TUNING_MARKET_SNAPSHOTS_COLLECTION].find_one(
        {
            "snapshot_id": normalized,
            "kind": "manifest",
            "ready": True,
            "schema_version": TUNING_MARKET_SNAPSHOT_SCHEMA_VERSION,
        },
        {"_id": 1},
    ) is not None


def freeze_tuning_market_snapshot(
    db: Any,
    request_payload: dict[str, Any],
    *,
    expected_signature: str | None = None,
) -> dict[str, Any]:
    






    payload = deepcopy(request_payload)
    payload["research_market_data_mode"] = "database_only"
    payload["research_market_data_snapshot_id"] = None
    payload["expected_market_data_signature_sha256"] = None
    config = BacktestExecutionRequest.model_validate(payload)

    frames: dict[str, Any] = {}
    for symbol in config.assets:
        raw = load_market_bars(symbol, config)
        frames[symbol] = validate_and_clean_bars(raw, config)

    signature, manifests = market_data_manifest(frames)
    signature = str(signature).strip().lower()
    expected = str(expected_signature or "").strip().lower()
    if expected and signature != expected:
        raise TuningMarketSnapshotMismatch(
            "SourceMarketDataSnapshotMismatch: the certified research reference "
            f"was evaluated on market-data signature {expected}, but the current MongoDB "
            f"research data resolves to {signature}. The tuning campaign cannot safely reuse "
            "the certified Control against a different market-data snapshot. Run a new certified "
            "Simulation Backtest before starting another tuning campaign."
        )

    collection = db[MODEL_TUNING_MARKET_SNAPSHOTS_COLLECTION]
    existing = collection.find_one(
        {
            "snapshot_id": signature,
            "kind": "manifest",
            "ready": True,
            "schema_version": TUNING_MARKET_SNAPSHOT_SCHEMA_VERSION,
        },
        {"_id": 0, "snapshot_id": 1, "asset_count": 1, "row_count": 1, "signature": 1},
    )
    if existing is not None:
        return dict(existing)

    
    collection.delete_many({"snapshot_id": signature})
    total_rows = 0
    now = utc_now()
    for symbol in config.assets:
        frame = frames[symbol]
        encoded, columns = encode_market_frame(frame)
        provenance = json_safe_provenance(dict(frame.attrs.get("market_data_provenance", {})))
        collection.insert_one(
            {
                "_id": f"{signature}:{symbol}",
                "snapshot_id": signature,
                "kind": "symbol",
                "schema_version": TUNING_MARKET_SNAPSHOT_SCHEMA_VERSION,
                "symbol": symbol,
                "interval": config.timeframe,
                "feed": config.alpaca_historical_feed,
                "adjustment": config.alpaca_adjustment,
                "columns": columns,
                "payload": encoded,
                "rows": int(len(frame)),
                "first_timestamp": manifests[symbol].get("first_timestamp"),
                "last_timestamp": manifests[symbol].get("last_timestamp"),
                "provenance": bson_value(provenance),
                "created_at": now,
            }
        )
        total_rows += int(len(frame))

    manifest = {
        "_id": f"{signature}:manifest",
        "snapshot_id": signature,
        "signature": signature,
        "kind": "manifest",
        "schema_version": TUNING_MARKET_SNAPSHOT_SCHEMA_VERSION,
        "ready": True,
        "assets": list(config.assets),
        "asset_count": len(config.assets),
        "row_count": total_rows,
        "start_date": config.start_date,
        "end_date": config.analysis_end_date or config.end_date,
        "research_snapshot_cutoff": config.analysis_end_date or config.end_date,
        "interval": config.timeframe,
        "feed": config.alpaca_historical_feed,
        "adjustment": config.alpaca_adjustment,
        "market_data_manifests": bson_value(manifests),
        "created_at": now,
    }
    collection.insert_one(manifest)
    return {key: value for key, value in manifest.items() if key != "_id"}


def require_tuning_market_snapshot(db: Any, snapshot_id: str) -> dict[str, Any]:
    normalized = str(snapshot_id or "").strip().lower()
    document = db[MODEL_TUNING_MARKET_SNAPSHOTS_COLLECTION].find_one(
        {
            "snapshot_id": normalized,
            "kind": "manifest",
            "ready": True,
            "schema_version": TUNING_MARKET_SNAPSHOT_SCHEMA_VERSION,
        },
        {"_id": 0},
    )
    if document is None:
        raise TuningMarketSnapshotMissing(
            f"Frozen tuning market-data snapshot {normalized or 'missing'} is not available."
        )
    return dict(document)
