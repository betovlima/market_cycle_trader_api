from __future__ import annotations

import hashlib
import re
from typing import Any


_INSTALLED = False
_PRIVATE_FAILURE_FIELD = "_technical_failure_diagnostics"
_PUBLIC_FAILURE_FIELD = "technical_failure_summary"
_MAX_PUBLIC_FAILURE_GROUPS = 12


def _sanitize_message(symbol: str, error: Exception) -> str:
    message = str(error or "").strip() or error.__class__.__name__
    if symbol:
        message = re.sub(re.escape(symbol), "{symbol}", message, flags=re.IGNORECASE)
    message = re.sub(r"\s+", " ", message).strip()
    message = re.sub(
        r"(?i)\b(api[_\s-]?key|secret|token)\b\s*[:=]\s*[^\s,;]+",
        lambda match: f"{match.group(1)}=<redacted>",
        message,
    )
    return message[:300]


def _failure_signature(stage: str, error_type: str, message: str) -> str:
    payload = f"{stage}|{error_type}|{message}".encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()[:16]


def _record_failure(service: Any, db: Any, symbol: str, stage: str, error: Exception) -> None:
    try:
        document = service._campaign(db) or {}
        run_id = str(document.get("run_id") or "").strip()
        if not run_id:
            return
        error_type = error.__class__.__name__
        message = _sanitize_message(symbol, error)
        signature = _failure_signature(stage, error_type, message)
        base = f"{_PRIVATE_FAILURE_FIELD}.{signature}"
        service_db = db[service.COLLECTION]
        service_db.update_one(
            {"_id": service.CURRENT_ID, "run_id": run_id},
            {
                "$inc": {f"{base}.count": 1},
                "$set": {
                    f"{base}.stage": stage,
                    f"{base}.error_type": error_type,
                    f"{base}.message": message,
                    "updated_at": service.utc_now(),
                },
            },
        )
    except Exception:
        return


def _build_public_summary(document: dict[str, Any]) -> list[dict[str, Any]]:
    raw = document.get(_PRIVATE_FAILURE_FIELD)
    if not isinstance(raw, dict):
        return []
    rows: list[dict[str, Any]] = []
    for value in raw.values():
        if not isinstance(value, dict):
            continue
        try:
            count = max(0, int(value.get("count") or 0))
        except (TypeError, ValueError):
            count = 0
        if count <= 0:
            continue
        rows.append(
            {
                "stage": str(value.get("stage") or "unknown"),
                "error_type": str(value.get("error_type") or "Exception"),
                "message": str(value.get("message") or "")[:300],
                "count": count,
            }
        )
    rows.sort(key=lambda item: (-int(item.get("count") or 0), str(item.get("stage") or ""), str(item.get("error_type") or "")))
    return rows[:_MAX_PUBLIC_FAILURE_GROUPS]


def _persist_public_summary(service: Any, db: Any, run_id: str) -> None:
    try:
        document = service._campaign(db) or {}
        if str(document.get("run_id") or "") != str(run_id or ""):
            return
        summary = _build_public_summary(document)
        update: dict[str, Any] = {
            "$set": {
                _PUBLIC_FAILURE_FIELD: service.bson_value(summary),
                "technical_failure_summary_count": int(sum(int(item.get("count") or 0) for item in summary)),
                "updated_at": service.utc_now(),
            },
            "$unset": {_PRIVATE_FAILURE_FIELD: ""},
        }
        db[service.COLLECTION].update_one(
            {"_id": service.CURRENT_ID, "run_id": run_id},
            update,
        )
    except Exception:
        return


def install_asset_discovery_failure_diagnostics() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from . import asset_discovery as service

    original_candidate_frame = service._candidate_frame
    original_score_candidate = service._score_candidate
    original_finish = service._finish
    original_export = service.export_asset_discovery
    active_db: list[Any | None] = [None]

    if getattr(original_candidate_frame, "_asset_discovery_failure_diagnostics", False):
        _INSTALLED = True
        return

    def diagnostic_candidate_frame(
        db: Any,
        symbol: str,
        config: Any,
        end_session: Any,
        *,
        credentials: dict[str, str] | None = None,
    ) -> Any:
        active_db[0] = db
        try:
            return original_candidate_frame(
                db,
                symbol,
                config,
                end_session,
                credentials=credentials,
            )
        except Exception as exc:
            _record_failure(service, db, str(symbol or ""), "candidate_frame", exc)
            raise

    def diagnostic_score_candidate(bundle: Any, symbol: str, frame: Any, baseline_returns: Any) -> dict[str, Any]:
        try:
            return original_score_candidate(bundle, symbol, frame, baseline_returns)
        except Exception as exc:
            db = active_db[0]
            if db is not None:
                _record_failure(service, db, str(symbol or ""), "score_candidate", exc)
            raise

    def diagnostic_finish(
        db: Any,
        run_id: str,
        status: str,
        message: str,
        *,
        results: list[dict[str, Any]] | None = None,
    ) -> None:
        _persist_public_summary(service, db, run_id)
        original_finish(db, run_id, status, message, results=results)

    def diagnostic_export(db: Any, *, front_version: str | None = None) -> dict[str, Any]:
        document = service._campaign(db) or {}
        run_id = str(document.get("run_id") or "").strip()
        if run_id and isinstance(document.get(_PRIVATE_FAILURE_FIELD), dict):
            _persist_public_summary(service, db, run_id)
        payload = dict(original_export(db, front_version=front_version))
        storage_policy = dict(payload.get("storage_policy") or {})
        storage_policy["technical_failure_symbols_persisted"] = False
        storage_policy["technical_failure_summary_persisted"] = True
        payload["storage_policy"] = storage_policy
        return payload

    setattr(diagnostic_candidate_frame, "_asset_discovery_failure_diagnostics", True)
    service._candidate_frame = diagnostic_candidate_frame
    service._score_candidate = diagnostic_score_candidate
    service._finish = diagnostic_finish
    service.export_asset_discovery = diagnostic_export
    _INSTALLED = True
