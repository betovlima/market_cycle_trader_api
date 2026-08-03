from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Query

from ...core.runtime import database
from ...services.dashboard import dashboard_job_detail, dashboard_summary

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/summary")
def get_dashboard_summary(
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
) -> dict[str, Any]:
    """Return a read-only, strategy-neutral dashboard summary."""

    return dashboard_summary(database(), limit=limit)


@router.get("/jobs/{job_id}")
def get_dashboard_job(job_id: str) -> dict[str, Any]:
    """Return strategy-neutral metrics and the public equity series for one job."""

    return dashboard_job_detail(database(), job_id)
