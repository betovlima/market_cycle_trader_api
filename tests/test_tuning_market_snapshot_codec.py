from __future__ import annotations

import numpy as np
import pandas as pd

from market_cycle_trader_api.engine.market_data_snapshot import decode_market_frame, encode_market_frame
from market_cycle_trader_api.schemas.requests import BacktestExecutionRequest


def test_snapshot_codec_preserves_market_frame_bits() -> None:
    index = pd.date_range("2026-01-02", periods=4, freq="B", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [100.125, 101.25, 102.5, 103.75],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            "volume": [1000.0, 1100.0, 1200.0, 1300.0],
            "vwap": [100.4, 101.4, 102.4, 103.4],
            "trade_count": [10.0, 11.0, 12.0, 13.0],
        },
        index=index,
    )
    payload, columns = encode_market_frame(frame)
    restored = decode_market_frame(payload, columns)

    assert list(restored.columns) == list(frame.columns)
    assert np.array_equal(restored.index.asi8, frame.index.asi8)
    assert np.array_equal(restored.to_numpy(dtype=np.float64), frame.to_numpy(dtype=np.float64))


def test_snapshot_codec_normalizes_microsecond_datetime_index_to_nanoseconds() -> None:
    base = pd.date_range("2016-01-04", periods=4, freq="B", tz="UTC")
    index = base.as_unit("us")
    assert str(index.dtype) == "datetime64[us, UTC]"

    frame = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            "volume": [1000.0, 1100.0, 1200.0, 1300.0],
        },
        index=index,
    )

    payload, columns = encode_market_frame(frame)
    restored = decode_market_frame(payload, columns)

    assert restored.index[0] == pd.Timestamp("2016-01-04", tz="UTC")
    assert restored.index[-1] == pd.Timestamp("2016-01-07", tz="UTC")
    assert restored.index.min().year == 2016
    assert np.array_equal(restored.to_numpy(dtype=np.float64), frame.to_numpy(dtype=np.float64))


def test_execution_request_accepts_sha256_snapshot_id() -> None:
    from pathlib import Path
    import json

    root = Path(__file__).resolve().parents[1]
    base = json.loads((root / "src" / "market_cycle_trader_api" / "parameterizations" / "winner-v1.13.2.json").read_text())
    digest = "a" * 64
    request = BacktestExecutionRequest.model_validate(
        {
            **base,
            "analysis_start_date": base["start_date"],
            "analysis_end_date": base["end_date"],
            "calendar_anchor_assets": base["assets"],
            "research_market_data_snapshot_id": digest,
        }
    )
    assert request.research_market_data_snapshot_id == digest


class _SnapshotCollection:
    def __init__(self) -> None:
        self.docs: list[dict] = []

    def find_one(self, query, projection=None):
        del projection
        for document in self.docs:
            if all(document.get(key) == value for key, value in query.items()):
                return dict(document)
        return None

    def delete_many(self, query):
        self.docs = [
            document
            for document in self.docs
            if not all(document.get(key) == value for key, value in query.items())
        ]

    def insert_one(self, document):
        self.docs.append(dict(document))


class _SnapshotDb:
    def __init__(self) -> None:
        self.collection = _SnapshotCollection()

    def __getitem__(self, name):
        from market_cycle_trader_api.infrastructure.persistence.mongo_repository import MODEL_TUNING_MARKET_SNAPSHOTS_COLLECTION

        assert name == MODEL_TUNING_MARKET_SNAPSHOTS_COLLECTION
        return self.collection


def test_freeze_snapshot_is_content_addressed_and_deduplicated(monkeypatch) -> None:
    from pathlib import Path
    import json

    from market_cycle_trader_api.services import model_tuning_market_snapshot as service

    root = Path(__file__).resolve().parents[1]
    base = json.loads((root / "src" / "market_cycle_trader_api" / "parameterizations" / "winner-v1.13.2.json").read_text())
    assets = base["assets"][:2]
    payload = {
        **base,
        "assets": assets,
        "analysis_start_date": base["start_date"],
        "analysis_end_date": base["end_date"],
        "calendar_anchor_assets": assets,
    }
    index = pd.date_range("2016-01-04", periods=1000, freq="B", tz="UTC")

    def fake_load(symbol, config):
        del config
        offset = 1.0 if symbol == assets[0] else 2.0
        frame = pd.DataFrame(
            {
                "open": np.arange(len(index), dtype=float) + 100.0 + offset,
                "high": np.arange(len(index), dtype=float) + 101.0 + offset,
                "low": np.arange(len(index), dtype=float) + 99.0 + offset,
                "close": np.arange(len(index), dtype=float) + 100.5 + offset,
                "volume": np.full(len(index), 1000.0 + offset),
            },
            index=index,
        )
        frame.attrs["market_data_provenance"] = {
            "history_complete": True,
            "historical_feed": base["alpaca_historical_feed"],
            "adjustment": base["alpaca_adjustment"],
            "provider": "alpaca",
            "effective_provider": "alpaca",
        }
        return frame

    monkeypatch.setattr(service, "load_market_bars", fake_load)
    monkeypatch.setattr(service, "validate_and_clean_bars", lambda frame, config: frame)
    db = _SnapshotDb()

    first = service.freeze_tuning_market_snapshot(db, payload)
    count_after_first = len(db.collection.docs)
    second = service.freeze_tuning_market_snapshot(db, payload)

    assert first["snapshot_id"] == first["signature"]
    assert second["snapshot_id"] == first["snapshot_id"]
    assert len(db.collection.docs) == count_after_first
    assert count_after_first == len(assets) + 1


def test_frozen_snapshot_round_trip_preserves_real_mongo_microsecond_dates(monkeypatch) -> None:
    from pathlib import Path
    import json

    from market_cycle_trader_api.engine import market_data as market_data_engine
    from market_cycle_trader_api.services import model_tuning_market_snapshot as service

    root = Path(__file__).resolve().parents[1]
    base = json.loads((root / "src" / "market_cycle_trader_api" / "parameterizations" / "winner-v1.13.2.json").read_text())
    assets = base["assets"][:2]
    request_payload = {
        **base,
        "assets": assets,
        "analysis_start_date": "2016-01-01",
        "analysis_end_date": "2026-08-11",
        "calendar_anchor_assets": assets,
    }
    index = pd.date_range("2016-01-04", periods=1000, freq="B", tz="UTC").as_unit("us")

    def fake_load(symbol, config):
        del config
        offset = 1.0 if symbol == assets[0] else 2.0
        frame = pd.DataFrame(
            {
                "open": np.arange(len(index), dtype=float) + 100.0 + offset,
                "high": np.arange(len(index), dtype=float) + 101.0 + offset,
                "low": np.arange(len(index), dtype=float) + 99.0 + offset,
                "close": np.arange(len(index), dtype=float) + 100.5 + offset,
                "volume": np.full(len(index), 1000.0 + offset),
            },
            index=index,
        )
        frame.attrs["market_data_provenance"] = {"history_complete": True}
        return frame

    monkeypatch.setattr(service, "load_market_bars", fake_load)
    monkeypatch.setattr(service, "validate_and_clean_bars", lambda frame, config: frame)
    db = _SnapshotDb()
    frozen = service.freeze_tuning_market_snapshot(db, request_payload)

    class Client:
        def close(self):
            return None

    monkeypatch.setattr(market_data_engine, "create_client", lambda: Client())
    monkeypatch.setattr(market_data_engine, "get_database", lambda client: db)

    config = BacktestExecutionRequest.model_validate(
        {
            **request_payload,
            "research_market_data_mode": "database_only",
            "research_market_data_snapshot_id": frozen["snapshot_id"],
        }
    )
    restored = market_data_engine._load_frozen_tuning_snapshot_bars(assets[0], config)

    assert len(restored) == len(index)
    assert restored.index.min() == pd.Timestamp("2016-01-04", tz="UTC")
    assert restored.index.max().year >= 2019


def test_schema_v1_snapshot_is_not_reused(monkeypatch) -> None:
    from pathlib import Path
    import json

    from market_cycle_trader_api.engine.market_data_snapshot import TUNING_MARKET_SNAPSHOT_SCHEMA_VERSION
    from market_cycle_trader_api.services import model_tuning_market_snapshot as service

    root = Path(__file__).resolve().parents[1]
    base = json.loads((root / "src" / "market_cycle_trader_api" / "parameterizations" / "winner-v1.13.2.json").read_text())
    assets = base["assets"][:2]
    payload = {
        **base,
        "assets": assets,
        "analysis_start_date": base["start_date"],
        "analysis_end_date": base["end_date"],
        "calendar_anchor_assets": assets,
    }
    index = pd.date_range("2016-01-04", periods=1000, freq="B", tz="UTC").as_unit("us")

    def fake_load(symbol, config):
        del config
        offset = float(assets.index(symbol) + 1)
        frame = pd.DataFrame(
            {
                "open": np.arange(len(index), dtype=float) + 100.0 + offset,
                "high": np.arange(len(index), dtype=float) + 101.0 + offset,
                "low": np.arange(len(index), dtype=float) + 99.0 + offset,
                "close": np.arange(len(index), dtype=float) + 100.5 + offset,
                "volume": np.full(len(index), 1000.0 + offset),
            },
            index=index,
        )
        frame.attrs["market_data_provenance"] = {"history_complete": True}
        return frame

    monkeypatch.setattr(service, "load_market_bars", fake_load)
    monkeypatch.setattr(service, "validate_and_clean_bars", lambda frame, config: frame)
    db = _SnapshotDb()
    first = service.freeze_tuning_market_snapshot(db, payload)
    snapshot_id = first["snapshot_id"]

    for document in db.collection.docs:
        if document.get("snapshot_id") == snapshot_id:
            document["schema_version"] = 1

    rebuilt = service.freeze_tuning_market_snapshot(db, payload)
    matching = [doc for doc in db.collection.docs if doc.get("snapshot_id") == snapshot_id]

    assert rebuilt["snapshot_id"] == snapshot_id
    assert matching
    assert all(doc.get("schema_version") == TUNING_MARKET_SNAPSHOT_SCHEMA_VERSION for doc in matching)


def test_snapshot_round_trip_preserves_research_signature_across_mongo_dtypes() -> None:
    from market_cycle_trader_api.services.reproducibility import market_data_manifest

    index = pd.date_range("2016-01-04", periods=4, freq="B", tz="UTC").as_unit("us")
    frame = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            "volume": [1000, 1100, 1200, 1300],
        },
        index=index,
    )
    frame.attrs["market_data_provenance"] = {
        "history_complete": True,
        "historical_feed": "sip",
        "adjustment": "all",
    }

    before, _ = market_data_manifest({"AAPL": frame})
    payload, columns = encode_market_frame(frame)
    restored = decode_market_frame(payload, columns)
    restored.attrs["market_data_provenance"] = dict(frame.attrs["market_data_provenance"])
    after, _ = market_data_manifest({"AAPL": restored})

    assert before == after
