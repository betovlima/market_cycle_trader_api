from __future__ import annotations

import threading
from copy import deepcopy
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any

from market_cycle_trader_api.auth.config import get_auth_settings


class InMemoryReviewerAccessStore:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        self.lock = threading.RLock()

    def ensure_indexes(self) -> None:
        return None

    def create(self, document: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self.records[document["_id"]] = deepcopy(document)
            return deepcopy(document)

    def get(self, access_id: str) -> dict[str, Any] | None:
        with self.lock:
            item = self.records.get(access_id)
            return deepcopy(item) if item else None

    def list(self) -> list[dict[str, Any]]:
        with self.lock:
            return sorted(
                (deepcopy(item) for item in self.records.values()),
                key=lambda item: item.get("created_at") or datetime.min.replace(tzinfo=UTC),
                reverse=True,
            )

    def update(self, access_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        with self.lock:
            item = self.records.get(access_id)
            if item is None:
                return None
            item.update(deepcopy(updates))
            return deepcopy(item)

    def delete(self, access_id: str) -> bool:
        with self.lock:
            return self.records.pop(access_id, None) is not None


class MongoReviewerAccessStore:
    def __init__(self) -> None:
        from market_cycle_trader_api.core.runtime import get_database

        self.collection = get_database()["reviewer_guest_access"]

    def ensure_indexes(self) -> None:
        self.collection.create_index("code_hash", unique=True)
        self.collection.create_index([("status", 1), ("expires_at", 1)])
        self.collection.create_index("created_at")

    def create(self, document: dict[str, Any]) -> dict[str, Any]:
        self.collection.insert_one(document)
        return document

    def get(self, access_id: str) -> dict[str, Any] | None:
        return self.collection.find_one({"_id": access_id})

    def list(self) -> list[dict[str, Any]]:
        return list(self.collection.find({}).sort("created_at", -1))

    def update(self, access_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        from pymongo import ReturnDocument

        return self.collection.find_one_and_update(
            {"_id": access_id},
            {"$set": updates},
            return_document=ReturnDocument.AFTER,
        )

    def delete(self, access_id: str) -> bool:
        return bool(self.collection.delete_one({"_id": access_id}).deleted_count)


@lru_cache(maxsize=1)
def get_reviewer_access_store() -> InMemoryReviewerAccessStore | MongoReviewerAccessStore:
    settings = get_auth_settings()
    if settings.auth_storage == "memory":
        return InMemoryReviewerAccessStore()
    return MongoReviewerAccessStore()
