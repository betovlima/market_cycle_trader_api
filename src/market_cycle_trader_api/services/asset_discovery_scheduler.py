from __future__ import annotations

import threading

from ..core.runtime import database
from .asset_discovery import (
    AssetDiscoveryConflict,
    automatic_slot_due,
    mark_automatic_slot,
    start_asset_discovery,
)

_SCHEDULER_STOP = threading.Event()
_SCHEDULER_THREAD: threading.Thread | None = None


def _scheduler_loop() -> None:
    while not _SCHEDULER_STOP.wait(60.0):
        try:
            db = database()
            slot = automatic_slot_due(db)
            if not slot:
                continue
            try:
                start_asset_discovery(db, source="automatic")
            except AssetDiscoveryConflict:
                continue
            mark_automatic_slot(db, slot)
        except Exception:
            continue


def start_asset_discovery_scheduler() -> None:
    global _SCHEDULER_THREAD
    if _SCHEDULER_THREAD is not None and _SCHEDULER_THREAD.is_alive():
        return
    _SCHEDULER_STOP.clear()
    _SCHEDULER_THREAD = threading.Thread(
        target=_scheduler_loop,
        name="asset-discovery-scheduler",
        daemon=True,
    )
    _SCHEDULER_THREAD.start()


def stop_asset_discovery_scheduler() -> None:
    _SCHEDULER_STOP.set()
