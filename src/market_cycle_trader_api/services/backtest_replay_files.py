"""Read-only exports of the exact persisted analytics for one completed backtest.

These files are generated from stored records, not original scientific artifacts
or a newly run simulation. No filesystem path supplied by the client is opened.
"""
from __future__ import annotations

import csv
from hashlib import sha256
from io import BytesIO, StringIO
import json
import re
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile


EQUITY_COLUMNS = ("timestamp", "simulation_equity", "reference_equity", "drawdown")
ROTATION_COLUMNS = (
    "executed_at", "from_asset", "to_asset", "realized_pnl",
    "transaction_fees", "buy_reason", "sell_reason",
)
SCORE_COLUMNS = (
    "execution_date", "decision_date", "current_asset", "current_score",
    "best_asset", "best_score", "finite_score_count", "decision_reason",
    "final_action_asset", "holding_days_at_decision", "effective_switch_margin",
    "best_vs_current_gap", "min_hold_guard_applied",
    "switch_margin_guard_applied", "cash_threshold_guard_applied",
    "minimum_expected_edge_guard_applied",
    *(item for rank in range(1, 6) for item in (f"top_{rank}_asset", f"top_{rank}_score")),
)
PUBLIC_FILES = (
    "equity_history.csv",
    "rotations.csv",
    "decision_scores.csv",
    "metrics.json",
    "analytics.json",
    "manifest.json",
    "package.zip",
)
_DESCRIPTIONS = {
    "equity_history.csv": "Historical capital and drawdown per recorded session",
    "rotations.csv": "Recorded transitions and transaction values",
    "decision_scores.csv": "Only utility scores actually saved at execution time",
    "metrics.json": "Stored metrics and summary diagnostics",
    "analytics.json": "Full read-only analytics payload used by the replay",
    "manifest.json": "File checksums and the origin of the exported data",
    "package.zip": "Available files and manifest together",
}


def _csv_cell(value: Any) -> Any:
    # Avoid spreadsheet formula execution in text cells; numeric cells stay numeric.
    if value is None:
        return ""
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r", "\n")):
        return "'" + value
    return value


def _csv_bytes(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> bytes:
    output = StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([_csv_cell(row.get(column)) for column in columns])
    return output.getvalue().encode("utf-8-sig")


def _score_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for row in rows:
        item = {key: row.get(key) for key in SCORE_COLUMNS if not key.startswith("top_")}
        for rank, entry in enumerate(row.get("top_scores") or [], start=1):
            if rank > 5:
                break
            item[f"top_{rank}_asset"] = entry.get("asset")
            item[f"top_{rank}_score"] = entry.get("score")
        flattened.append(item)
    return flattened


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n").encode("utf-8")


def _files_and_manifest(analytics: dict[str, Any]) -> tuple[dict[str, bytes], dict[str, Any]]:
    result: dict[str, bytes] = {}
    rows: dict[str, int] = {}
    equity = analytics.get("equity") or []
    rotations = analytics.get("rotations") or []
    scores = analytics.get("decision_scores") or []
    if equity:
        result["equity_history.csv"] = _csv_bytes(equity, EQUITY_COLUMNS)
        rows["equity_history.csv"] = len(equity)
    if rotations:
        result["rotations.csv"] = _csv_bytes(rotations, ROTATION_COLUMNS)
        rows["rotations.csv"] = len(rotations)
    if scores:
        result["decision_scores.csv"] = _csv_bytes(_score_rows(scores), SCORE_COLUMNS)
        rows["decision_scores.csv"] = len(scores)
    if analytics.get("metrics") or analytics.get("rotation_summary"):
        result["metrics.json"] = _json_bytes({
            key: analytics.get(key) for key in (
                "metrics", "rotation_summary", "monthly_returns", "consistency",
                "drawdown_episodes", "asset_attribution", "transition_matrix",
                "holding_buckets", "trade_dependency",
            )
        })
    result["analytics.json"] = _json_bytes(analytics)
    manifest = {
        "schema_version": 1,
        "job_id": analytics.get("job_id"),
        "created_at": analytics.get("created_at"),
        "finished_at": analytics.get("finished_at"),
        "source": "persisted_backtest_analytics",
        "scope": "One completed Backtest selected in Annual Replay; all years",
        "limitations": (
            "Exports are generated from the persisted analytics, not original "
            "research inputs, training datasets, model binaries or source code. "
            "They do not by themselves prove scientific reproducibility."
        ),
        "csv_note": "Untrusted text is escaped for spreadsheet formula safety; analytics.json retains stored public values.",
        "files": [
            {
                "name": name,
                "bytes": len(body),
                "sha256": sha256(body).hexdigest(),
                **({"rows": rows[name]} if name in rows else {}),
            }
            for name, body in result.items()
        ],
    }
    result["manifest.json"] = _json_bytes(manifest)
    return result, manifest


def available_backtest_files(analytics: dict[str, Any]) -> dict[str, Any]:
    files, manifest = _files_and_manifest(analytics)
    listing = [
        {"name": name, "bytes": len(files[name]), "description": _DESCRIPTIONS[name]}
        for name in PUBLIC_FILES if name in files
    ]
    listing.append({"name": "package.zip", "bytes": None, "description": _DESCRIPTIONS["package.zip"]})
    return {
        "job_id": manifest["job_id"],
        "source": manifest["source"],
        "files": listing,
    }


def backtest_file(analytics: dict[str, Any], filename: str) -> tuple[bytes, str, str]:
    """Returns bytes, content-type, and a safe Content-Disposition filename."""
    if filename not in PUBLIC_FILES:
        raise KeyError(filename)
    files, _ = _files_and_manifest(analytics)
    if filename == "package.zip":
        output = BytesIO()
        with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
            for name, body in files.items():
                archive.writestr(name, body)
        body = output.getvalue()
    else:
        if filename not in files:
            raise KeyError(filename)
        body = files[filename]
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", str(analytics.get("job_id") or ""))[:70] or "selected"
    content_type = (
        "application/zip" if filename.endswith(".zip")
        else "text/csv; charset=utf-8" if filename.endswith(".csv")
        else "application/json; charset=utf-8"
    )
    return body, content_type, f"backtest_{safe_id}_{filename}"
