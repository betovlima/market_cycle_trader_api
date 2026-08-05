from __future__ import annotations

import threading
import uuid
from copy import deepcopy
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any

from market_cycle_trader_api.auth.config import get_auth_settings as get_settings


def _aware_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _session_is_active(document: dict[str, Any], now: datetime) -> bool:
    expires_at = _aware_utc(document.get("expires_at"))
    return not document.get("revoked") and expires_at is not None and expires_at > now


class InMemoryAccessStore:
    def __init__(self) -> None:
        self.invitations: dict[str, dict[str, Any]] = {}
        self.sessions: dict[str, dict[str, Any]] = {}
        self.logs: list[dict[str, Any]] = []
        self.lock = threading.RLock()

    def ensure_indexes(self) -> None:
        now = datetime.now(UTC)
        with self.lock:
            legacy_ids: set[str] = set()
            for invitation_id, item in self.invitations.items():
                if not item.get("authorized_email"):
                    item["status"] = "legacy_unverified"
                    item.setdefault("legacy_marked_at", now)
                    item["token_hash"] = f"legacy-disabled:{uuid.uuid4()}"
                    item.setdefault("max_active_sessions", 1 if item.get("role") in {"trader", "admin"} else 2)
                    legacy_ids.add(invitation_id)
            for session in self.sessions.values():
                if session.get("invitation_id") in legacy_ids:
                    session["revoked"] = True

    def create_invitation(self, document: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self.invitations[document["_id"]] = deepcopy(document)
            return deepcopy(document)

    def get_invitation(self, invitation_id: str) -> dict[str, Any] | None:
        with self.lock:
            item = self.invitations.get(invitation_id)
            return deepcopy(item) if item else None

    def get_invitation_by_token_hash(self, token_hash: str) -> dict[str, Any] | None:
        with self.lock:
            for item in self.invitations.values():
                if item.get("token_hash") == token_hash:
                    return deepcopy(item)
        return None

    def find_claimed_invitations(
        self,
        identity_subject: str,
        identity_email: str,
        now: datetime,
    ) -> list[dict[str, Any]]:
        normalized_email = str(identity_email or "").strip().casefold()
        with self.lock:
            matches = []
            for item in self.invitations.values():
                expires_at = _aware_utc(item.get("expires_at"))
                if (
                    item.get("status") == "claimed"
                    and str(item.get("claimed_subject") or "") == identity_subject
                    and str(item.get("claimed_email") or "").strip().casefold() == normalized_email
                    and expires_at is not None
                    and expires_at > now
                ):
                    matches.append(deepcopy(item))
            return sorted(
                matches,
                key=lambda item: item.get("created_at") or now,
                reverse=True,
            )

    def upsert_primary_administrator(
        self,
        document: dict[str, Any],
    ) -> dict[str, Any]:
        with self.lock:
            existing = self.invitations.get(document["_id"])
            if existing is None:
                self.invitations[document["_id"]] = deepcopy(document)
                existing = self.invitations[document["_id"]]
            return deepcopy(existing)

    def list_invitations(self) -> list[dict[str, Any]]:
        with self.lock:
            return sorted(
                (deepcopy(item) for item in self.invitations.values()),
                key=lambda item: item["created_at"],
                reverse=True,
            )

    def update_invitation(self, invitation_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        with self.lock:
            if invitation_id not in self.invitations:
                return None
            self.invitations[invitation_id].update(deepcopy(updates))
            return deepcopy(self.invitations[invitation_id])

    def claim_invitation(
        self,
        invitation_id: str,
        expected_token_hash: str,
        now: datetime,
        updates: dict[str, Any],
    ) -> dict[str, Any] | None:
        with self.lock:
            item = self.invitations.get(invitation_id)
            expires_at = _aware_utc(item.get("expires_at")) if item else None
            if (
                not item
                or item.get("status") != "pending_verification"
                or item.get("token_hash") != expected_token_hash
                or expires_at is None
                or expires_at <= now
            ):
                return None
            item.update(deepcopy(updates))
            return deepcopy(item)

    def delete_invitation(self, invitation_id: str) -> bool:
        with self.lock:
            return self.invitations.pop(invitation_id, None) is not None

    def create_limited_session(
        self,
        document: dict[str, Any],
        max_active_sessions: int,
        now: datetime,
    ) -> int:
        with self.lock:
            active = sorted(
                (
                    item
                    for item in self.sessions.values()
                    if item.get("invitation_id") == document["invitation_id"]
                    and _session_is_active(item, now)
                ),
                key=lambda item: item.get("created_at") or now,
            )
            remove_count = max(0, len(active) - max_active_sessions + 1)
            for item in active[:remove_count]:
                item["revoked"] = True
            self.sessions[document["_id"]] = deepcopy(document)
            return remove_count

    def count_active_sessions(self, invitation_id: str, now: datetime) -> int:
        with self.lock:
            return sum(
                1
                for item in self.sessions.values()
                if item.get("invitation_id") == invitation_id and _session_is_active(item, now)
            )

    def trim_active_sessions(self, invitation_id: str, max_active_sessions: int, now: datetime) -> int:
        with self.lock:
            active = sorted(
                (
                    item
                    for item in self.sessions.values()
                    if item.get("invitation_id") == invitation_id and _session_is_active(item, now)
                ),
                key=lambda item: item.get("created_at") or now,
                reverse=True,
            )
            removed = 0
            for item in active[max_active_sessions:]:
                item["revoked"] = True
                removed += 1
            return removed

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self.lock:
            item = self.sessions.get(session_id)
            return deepcopy(item) if item else None

    def touch_session(self, session_id: str, last_activity_at: datetime, idle_expires_at: datetime) -> dict[str, Any] | None:
        with self.lock:
            item = self.sessions.get(session_id)
            if not item or item.get("revoked"):
                return None
            item["last_activity_at"] = last_activity_at
            item["idle_expires_at"] = idle_expires_at
            return deepcopy(item)

    def terminate_sessions(self, invitation_id: str) -> int:
        count = 0
        with self.lock:
            for item in self.sessions.values():
                if item.get("invitation_id") == invitation_id and not item.get("revoked"):
                    item["revoked"] = True
                    count += 1
        return count

    def revoke_session(self, session_id: str) -> bool:
        with self.lock:
            item = self.sessions.get(session_id)
            if not item:
                return False
            item["revoked"] = True
            return True

    def record_log(self, document: dict[str, Any]) -> None:
        with self.lock:
            self.logs.append(deepcopy(document))

    def list_logs(self, limit: int) -> list[dict[str, Any]]:
        with self.lock:
            return sorted(
                (deepcopy(item) for item in self.logs),
                key=lambda item: item["created_at"],
                reverse=True,
            )[:limit]


class MongoAccessStore:
    def __init__(self) -> None:
        from market_cycle_trader_api.core.runtime import get_database

        self.database = get_database()
        self.invitations = self.database["trader_invitations"]
        self.sessions = self.database["trader_sessions"]
        self.logs = self.database["trader_access_logs"]

    def ensure_indexes(self) -> None:
        now = datetime.now(UTC)
        legacy = list(
            self.invitations.find(
                {
                    "status": {"$ne": "legacy_unverified"},
                    "$or": [
                        {"authorized_email": {"$exists": False}},
                        {"authorized_email": None},
                        {"authorized_email": ""},
                    ],
                },
                {"_id": 1, "role": 1},
            )
        )
        if legacy:
            legacy_ids = [item["_id"] for item in legacy]
            for item in legacy:
                self.invitations.update_one(
                    {"_id": item["_id"]},
                    {
                        "$set": {
                            "status": "legacy_unverified",
                            "legacy_marked_at": now,
                            "updated_at": now,
                            "token_hash": f"legacy-disabled:{uuid.uuid4()}",
                            "max_active_sessions": 1 if item.get("role") in {"trader", "admin"} else 2,
                        }
                    },
                )
            self.sessions.update_many(
                {"invitation_id": {"$in": legacy_ids}, "revoked": False},
                {"$set": {"revoked": True, "revoked_at": now}},
            )

        self.invitations.create_index("token_hash", unique=True)
        self.invitations.create_index([("authorized_email", 1), ("created_at", -1)])
        self.invitations.create_index([("claimed_subject", 1), ("expires_at", 1)])
        self.invitations.create_index([("status", 1), ("expires_at", 1)])
        self.sessions.create_index("expires_at", expireAfterSeconds=0)
        self.sessions.create_index([("invitation_id", 1), ("revoked", 1), ("created_at", -1)])
        self.sessions.create_index([("identity_subject", 1), ("revoked", 1)])
        self.logs.create_index("created_at")
        self.logs.create_index([("invitation_id", 1), ("created_at", -1)])

    def create_invitation(self, document):
        self.invitations.insert_one(document)
        return document

    def get_invitation(self, invitation_id):
        return self.invitations.find_one({"_id": invitation_id})

    def get_invitation_by_token_hash(self, token_hash):
        return self.invitations.find_one({"token_hash": token_hash})

    def find_claimed_invitations(self, identity_subject, identity_email, now):
        return list(
            self.invitations.find(
                {
                    "status": "claimed",
                    "claimed_subject": identity_subject,
                    "claimed_email": str(identity_email or "").strip().casefold(),
                    "expires_at": {"$gt": now},
                }
            ).sort("created_at", -1)
        )

    def upsert_primary_administrator(self, document):
        from pymongo import ReturnDocument

        return self.invitations.find_one_and_update(
            {"_id": document["_id"]},
            {"$setOnInsert": document},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )

    def list_invitations(self):
        return list(self.invitations.find({}).sort("created_at", -1))

    def update_invitation(self, invitation_id, updates):
        from pymongo import ReturnDocument

        return self.invitations.find_one_and_update(
            {"_id": invitation_id},
            {"$set": updates},
            return_document=ReturnDocument.AFTER,
        )

    def claim_invitation(self, invitation_id, expected_token_hash, now, updates):
        from pymongo import ReturnDocument

        return self.invitations.find_one_and_update(
            {
                "_id": invitation_id,
                "status": "pending_verification",
                "token_hash": expected_token_hash,
                "expires_at": {"$gt": now},
            },
            {"$set": updates},
            return_document=ReturnDocument.AFTER,
        )

    def delete_invitation(self, invitation_id):
        return bool(self.invitations.delete_one({"_id": invitation_id}).deleted_count)

    def create_limited_session(self, document, max_active_sessions, now):
        active = list(
            self.sessions.find(
                {
                    "invitation_id": document["invitation_id"],
                    "revoked": False,
                    "expires_at": {"$gt": now},
                },
                {"_id": 1, "created_at": 1},
            ).sort("created_at", 1)
        )
        remove_count = max(0, len(active) - max_active_sessions + 1)
        if remove_count:
            ids = [item["_id"] for item in active[:remove_count]]
            self.sessions.update_many(
                {"_id": {"$in": ids}},
                {"$set": {"revoked": True, "revoked_at": now}},
            )
        self.sessions.insert_one(document)
        return remove_count

    def count_active_sessions(self, invitation_id, now):
        return int(
            self.sessions.count_documents(
                {
                    "invitation_id": invitation_id,
                    "revoked": False,
                    "expires_at": {"$gt": now},
                }
            )
        )

    def trim_active_sessions(self, invitation_id, max_active_sessions, now):
        active = list(
            self.sessions.find(
                {
                    "invitation_id": invitation_id,
                    "revoked": False,
                    "expires_at": {"$gt": now},
                },
                {"_id": 1, "created_at": 1},
            ).sort("created_at", -1)
        )
        ids = [item["_id"] for item in active[max_active_sessions:]]
        if not ids:
            return 0
        return int(
            self.sessions.update_many(
                {"_id": {"$in": ids}},
                {"$set": {"revoked": True, "revoked_at": now}},
            ).modified_count
        )

    def get_session(self, session_id):
        return self.sessions.find_one({"_id": session_id})

    def touch_session(self, session_id, last_activity_at, idle_expires_at):
        from pymongo import ReturnDocument

        return self.sessions.find_one_and_update(
            {"_id": session_id, "revoked": False},
            {"$set": {"last_activity_at": last_activity_at, "idle_expires_at": idle_expires_at}},
            return_document=ReturnDocument.AFTER,
        )

    def terminate_sessions(self, invitation_id):
        now = datetime.now(UTC)
        return int(
            self.sessions.update_many(
                {"invitation_id": invitation_id, "revoked": False},
                {"$set": {"revoked": True, "revoked_at": now}},
            ).modified_count
        )

    def revoke_session(self, session_id):
        return bool(
            self.sessions.update_one(
                {"_id": session_id},
                {"$set": {"revoked": True, "revoked_at": datetime.now(UTC)}},
            ).modified_count
        )

    def record_log(self, document):
        self.logs.insert_one(document)

    def list_logs(self, limit):
        return list(self.logs.find({}).sort("created_at", -1).limit(limit))


@lru_cache(maxsize=1)
def get_access_store() -> InMemoryAccessStore | MongoAccessStore:
    settings = get_settings()
    if settings.auth_storage == "memory":
        return InMemoryAccessStore()
    return MongoAccessStore()
