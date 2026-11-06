"""Live Winner operational controls. Start/Pause/Exit only/Stop govern Alpaca Live."""
from __future__ import annotations

from typing import Any

from pymongo import ReturnDocument

from ..infrastructure.persistence.mongo_repository import utc_now
from ..infrastructure.trading.alpaca_live import live_credentials
from ..infrastructure.trading.canonical_live_oauth import (
    canonical_live_auth_mode, canonical_oauth_orders_enabled,
    create_canonical_live_oauth_client,
)

LIVE_CONTROL_COLLECTION = "live_trader_control"
LIVE_CONTROL_HISTORY_COLLECTION = "live_trader_control_history"
VALID_MODES = frozenset({"active", "paused", "exit_only", "stopped"})
CONTROL_ID = "default"


class LiveControlConflict(RuntimeError):
    pass


def _current(db: Any) -> dict[str, Any]:
    # $setOnInsert ensures never writing into Paper collections.
    db[LIVE_CONTROL_COLLECTION].update_one(
        {"_id": CONTROL_ID},
        {"$setOnInsert": {
            "control_mode": "stopped",
            "revision": 0,
            "updated_at": utc_now(),
            "updated_by": "system",
            "reason": "Live initial state is stopped; order execution disabled.",
        }},
        upsert=True,
    )
    return db[LIVE_CONTROL_COLLECTION].find_one({"_id": CONTROL_ID})


def live_control_status(db: Any) -> dict[str, Any]:
    state = _current(db)
    mode = str(state.get("control_mode") or "stopped")
    auth_mode = canonical_live_auth_mode(db)
    orders_armed = canonical_oauth_orders_enabled(db)
    execution_allowed = (
        mode in {"active", "exit_only"}
        and (
            auth_mode != "oauth"
            or orders_armed
        )
    )
    return {
        "control_mode": mode,
        "revision": int(state.get("revision") or 0),
        "status": mode,
        "phase": "live_operational" if mode in {"active", "exit_only"} else "live_inactive",
        "enabled": mode in {"active", "exit_only"},
        "monitoring_enabled": mode in {"active", "exit_only"},
        "execution_enabled": execution_allowed,
        "scheduler_alive": False,
        "canonical_broker_authentication": auth_mode,
        "oauth_orders_allowed": orders_armed,
        "oauth_account_suffix": str(state.get("oauth_account_id") or "")[-6:] or None,
        "order_submission_enabled": execution_allowed,
        "last_control_update_at": state.get("updated_at"),
        "last_control_updated_by": state.get("updated_by"),
        "last_control_reason": state.get("reason"),
    }


def set_live_control_mode(
    db: Any, *, mode: str, expected_revision: int,
    actor_email: str, reason: str,
) -> dict[str, Any]:
    if mode not in VALID_MODES:
        raise ValueError("Invalid Live mode.")
    if not actor_email:
        raise ValueError("An authenticated administrator is required.")
    if mode in {"active", "exit_only"}:
        # The operator must have a valid connection for the selected transport.
        # In OAuth mode, never authorize a Live transition from API keys.
        if canonical_live_auth_mode(db) == "oauth":
            create_canonical_live_oauth_client(db)
        else:
            live_credentials()
    previous = _current(db)
    if int(previous.get("revision", 0)) != int(expected_revision):
        raise LiveControlConflict("Live mode revision changed. Refresh and retry.")
    if previous.get("control_mode") == mode:
        return live_control_status(db)
    now = utc_now()
    latest = db[LIVE_CONTROL_COLLECTION].find_one_and_update(
        {
            "_id": CONTROL_ID,
            "revision": expected_revision,
            "control_mode": previous["control_mode"],
        },
        {"$set": {
            "control_mode": mode,
            "updated_at": now,
            "updated_by": actor_email,
            "reason": reason[:500],
        }, "$inc": {"revision": 1}},
        return_document=ReturnDocument.AFTER,
    )
    if latest is None:
        raise LiveControlConflict("Live mode changed concurrently. Refresh and retry.")
    db[LIVE_CONTROL_HISTORY_COLLECTION].insert_one({
        "created_at": now, "updated_by": actor_email,
        "old_mode": previous["control_mode"],
        "new_mode": mode,
        "revision": int(latest["revision"]),
        "reason": reason[:500],
        "execution_enabled": mode in {"active", "exit_only"},
        "market": "live",
    })
    return live_control_status(db)



def canonical_live_broker_settings(db: Any) -> dict[str, Any]:
    """Safe administrator view of persisted broker transport and real-order gate."""
    state = _current(db)
    status = live_control_status(db)
    return {
        "mode": status["canonical_broker_authentication"],
        "oauth_orders_allowed": status["oauth_orders_allowed"],
        "oauth_account_suffix": status["oauth_account_suffix"],
        "control_mode": status["control_mode"],
        "order_submission_enabled": status["order_submission_enabled"],
        "revision": int(state.get("revision") or 0),
        "can_bind": status["control_mode"] == "stopped",
        "can_enable_orders": status["control_mode"] == "stopped",
        "updated_at": state.get("broker_updated_at"),
        "updated_by": state.get("broker_updated_by"),
    }


def update_canonical_live_broker(
    db: Any, *, identity: Any, expected_revision: int,
    enable_orders: bool, confirmation: str,
) -> dict[str, Any]:
    """Bind the current administrator's verified OAuth account, or change its gate.

    Requires optimistic locking and the Live robot stopped before binding or
    enabling order permission. Disabling an armed gate is also allowed while
    active, so the administrator can stop subsequent new submissions.
    """
    if not bool(getattr(identity, "is_admin", False)):
        raise LiveControlConflict("Administrator access is required.")
    state = _current(db)
    revision = int(state.get("revision") or 0)
    if expected_revision != revision:
        raise LiveControlConflict("Live configuration changed. Refresh and retry.")
    mode = str(state.get("control_mode") or "stopped")
    currently_oauth = state.get("broker_auth_mode") == "oauth"
    binding_change = not currently_oauth
    if (binding_change or enable_orders) and mode != "stopped":
        raise LiveControlConflict("Stop the Live robot before binding OAuth or enabling orders.")
    if enable_orders:
        if not currently_oauth or state.get("oauth_orders_allowed") is not False:
            raise LiveControlConflict("First bind the verified OAuth account with orders disabled.")
        if confirmation != "ENABLE_LIVE_REAL_ORDERS":
            raise LiveControlConflict("Explicit confirmation of real orders is required.")
    elif confirmation != "BIND_VERIFIED_OAUTH_NO_ORDERS" and confirmation != "DISABLE_LIVE_REAL_ORDERS":
        raise LiveControlConflict("Explicit confirmation of the broker operation is required.")
    if not enable_orders and confirmation == "DISABLE_LIVE_REAL_ORDERS" and not currently_oauth:
        raise LiveControlConflict("No OAuth broker is configured.")

    account_id = str(state.get("oauth_account_id") or "")
    subject = str(state.get("oauth_subject") or "")
    if binding_change or enable_orders or confirmation == "BIND_VERIFIED_OAUTH_NO_ORDERS":
        from ..api.routers.alpaca_oauth import CONNECTIONS, _key, connected_trading_client
        subject = str(getattr(identity, "subject", "") or "")
        saved = db[CONNECTIONS].find_one({"_id": _key(subject)})
        if (not isinstance(saved, dict)
            or saved.get("subject") != subject
            or saved.get("role") != "admin"
            or saved.get("environment") != "live"
            or saved.get("trading_authorized") is not True
            or saved.get("oauth_scope") != "trading"
            or saved.get("read_only") is not False):
            raise LiveControlConflict("Verify and authorize an administrator OAuth trading account first.")
        broker = connected_trading_client(db, identity)
        verified_id = str(getattr(broker.get_account(), "id", "") or "").strip()
        if not verified_id or verified_id != str(saved.get("account_id") or ""):
            raise LiveControlConflict("The Alpaca OAuth account identity could not be verified.")
        if currently_oauth and (account_id != verified_id or
                                 subject != str(state.get("oauth_subject") or "")):
            raise LiveControlConflict("Bound Live OAuth account differs; stop and reconcile it first.")
        account_id = verified_id
    if not account_id or not subject:
        raise LiveControlConflict("An administrator OAuth account must be bound first.")

    now = utc_now()
    update = db[LIVE_CONTROL_COLLECTION].find_one_and_update(
        {"_id": CONTROL_ID, "revision": revision,
         "control_mode": state.get("control_mode", "stopped")},
        {"$set": {
            "broker_auth_mode": "oauth",
            "oauth_subject": subject,
            "oauth_account_id": account_id,
            "oauth_orders_allowed": bool(enable_orders),
            "broker_updated_at": now,
            "broker_updated_by": str(getattr(identity, "email", "") or subject),
        }, "$inc": {"revision": 1}},
        return_document=ReturnDocument.AFTER,
    )
    if not isinstance(update, dict):
        raise LiveControlConflict("Live configuration changed concurrently. Refresh and retry.")
    db[LIVE_CONTROL_HISTORY_COLLECTION].insert_one({
        "created_at": now, "updated_by": str(getattr(identity, "email", "") or subject),
        "old_mode": str(state.get("broker_auth_mode") or "api_keys"),
        "new_mode": "oauth",
        "oauth_orders_before": bool(state.get("oauth_orders_allowed")),
        "oauth_orders_after": bool(enable_orders),
        "account_suffix": account_id[-6:],
        "revision": revision + 1,
        "reason": "Administrator-controlled canonical OAuth broker authorization.",
        "market": "live",
    })
    return canonical_live_broker_settings(db)
