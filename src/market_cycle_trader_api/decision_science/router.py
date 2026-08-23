from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from ..auth.security import SessionIdentity, require_capability
from ..core.runtime import database
from .service import build_export, history, latest, run_analysis

router = APIRouter(prefix="/api/decision-science", tags=["decision-science"])
require_view = require_capability("temporal_intelligence.view")
require_run = require_capability("temporal_intelligence.start")
require_export = require_capability("temporal_intelligence.export")


@router.get("/history")
def decision_science_history(
    _identity: Annotated[SessionIdentity, Depends(require_view)],
    limit: int = Query(default=30, ge=1, le=100),
) -> dict[str, Any]:
    return history(database(), limit=limit)


@router.get("/latest")
def latest_decision_science(
    _identity: Annotated[SessionIdentity, Depends(require_view)],
    run_id: str | None = Query(default=None),
) -> dict[str, Any] | None:
    return latest(database(), run_id=run_id)


@router.post("/{run_id}/analyze")
def analyze_decision_science(
    run_id: str,
    _identity: Annotated[SessionIdentity, Depends(require_run)],
) -> dict[str, Any]:
    try:
        return run_analysis(database(), run_id)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc

@router.get("/{analysis_id}/export.zip")
def export_decision_science(
    analysis_id: str,
    _identity: Annotated[SessionIdentity, Depends(require_export)],
) -> Response:
    try:
        content = build_export(database(), analysis_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return Response(
        content=content,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="decision_science_{analysis_id}.zip"'},
    )

