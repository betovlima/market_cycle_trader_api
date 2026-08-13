from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from ...core.runtime import database
from ...services.admin_rotations import admin_job_rotations

router = APIRouter(prefix="/api/admin/jobs", tags=["administration"])


@router.get("/{job_id}/rotations")
def get_admin_job_rotations(job_id: str) -> dict[str, Any]:
    

    return admin_job_rotations(database(), job_id)
