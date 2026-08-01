from __future__ import annotations

import io
import json
import zipfile

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from ...services.jobs import require_job
from ...services.results import build_results, csv_bytes, csv_response, diagnostic_csv_rows

router = APIRouter(tags=["exports"])


@router.get("/api/jobs/{job_id}/comparison.csv")
def export_comparison(job_id: str) -> Response:
    require_job(job_id)
    results = build_results(job_id)
    return csv_response(results.get("comparison", []), f"comparison_{job_id}.csv")


@router.get("/api/jobs/{job_id}/export.zip")
def export_zip(job_id: str) -> Response:
    require_job(job_id)
    results = build_results(job_id)
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("comparison.csv", csv_bytes(results.get("comparison", [])))
        for index, run in enumerate(results.get("runs", []), start=1):
            folder = f"result_{index}"
            archive.writestr(
                f"{folder}/metrics.json",
                json.dumps(run.get("metrics", {}), indent=2, ensure_ascii=False),
            )
            archive.writestr(
                f"{folder}/equity.csv",
                csv_bytes(run.get("series", [])),
            )
            archive.writestr(
                f"{folder}/trades.csv",
                csv_bytes(run.get("trades", [])),
            )
            archive.writestr(
                f"{folder}/diagnostics.csv",
                csv_bytes(diagnostic_csv_rows(run.get("diagnostics", {}))),
            )
    payload = archive_buffer.getvalue()
    if not payload:
        raise HTTPException(status_code=404, detail="Export is not available.")
    return Response(
        content=payload,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="analysis_{job_id}.zip"',
            "Content-Length": str(len(payload)),
            "Cache-Control": "no-store",
        },
    )
