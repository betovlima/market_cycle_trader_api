from __future__ import annotations

import threading
from copy import deepcopy
from functools import lru_cache
from typing import Any

from market_cycle_trader_api.auth.config import get_auth_settings as get_settings


class InMemoryAccessStore:
    def __init__(self) -> None:
        self.invitations: dict[str, dict[str, Any]] = {}
        self.sessions: dict[str, dict[str, Any]] = {}
        self.logs: list[dict[str, Any]] = []
        self.lock = threading.RLock()

    def ensure_indexes(self) -> None:
        return None

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

    def delete_invitation(self, invitation_id: str) -> bool:
        with self.lock:
            return self.invitations.pop(invitation_id, None) is not None

    def create_session(self, document: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self.sessions[document["_id"]] = deepcopy(document)
            return deepcopy(document)

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self.lock:
            item = self.sessions.get(session_id)
            return deepcopy(item) if item else None

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
        self.invitations.create_index("token_hash", unique=True)
        self.invitations.create_index([("guest_name", 1), ("created_at", -1)])
        self.invitations.create_index([("status", 1), ("expires_at", 1)])
        self.sessions.create_index("expires_at", expireAfterSeconds=0)
        self.sessions.create_index([("invitation_id", 1), ("revoked", 1)])
        self.logs.create_index("created_at")
        self.logs.create_index([("invitation_id", 1), ("created_at", -1)])

    def create_invitation(self, document):
        self.invitations.insert_one(document); return document
    def get_invitation(self, invitation_id):
        return self.invitations.find_one({"_id": invitation_id})
    def get_invitation_by_token_hash(self, token_hash):
        return self.invitations.find_one({"token_hash": token_hash})
    def list_invitations(self):
        return list(self.invitations.find({}).sort("created_at", -1))
    def update_invitation(self, invitation_id, updates):
        from pymongo import ReturnDocument
        return self.invitations.find_one_and_update({"_id": invitation_id}, {"$set": updates}, return_document=ReturnDocument.AFTER)
    def delete_invitation(self, invitation_id):
        return bool(self.invitations.delete_one({"_id": invitation_id}).deleted_count)
    def create_session(self, document):
        self.sessions.insert_one(document); return document
    def get_session(self, session_id):
        return self.sessions.find_one({"_id": session_id})
    def terminate_sessions(self, invitation_id):
        return int(self.sessions.update_many({"invitation_id": invitation_id, "revoked": False}, {"$set": {"revoked": True}}).modified_count)
    def revoke_session(self, session_id):
        return bool(self.sessions.update_one({"_id": session_id}, {"$set": {"revoked": True}}).modified_count)
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
