from __future__ import annotations

import csv
import io
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from fastapi import HTTPException
from fastapi.responses import Response, StreamingResponse

from ..core.runtime import database
from ..infrastructure.persistence.mongo_repository import (
    COMPARISONS_COLLECTION,
    PREDICTIONS_COLLECTION,
    RUNS_COLLECTION,
    TRADES_COLLECTION,
)
from .diagnostics.performance import build_performance_diagnostics
from .serialization import clean_mongo_rows, downsample_documents, iso_value


def diagnostic_csv_rows(diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sections = {
        "underperformance_periods": "UNDERPERFORMANCE_PERIOD",
        "exit_diagnostics": "EXIT_COUNTERFACTUAL",
        "rotation_diagnostics": "ROTATION_COUNTERFACTUAL",
        "q_delta_buckets": "Q_DELTA_BUCKET",
        "holding_distribution": "HOLDING_DISTRIBUTION",
        "asset_performance": "ASSET_PERFORMANCE",
        "session_performance": "SESSION_PERFORMANCE",
    }
    for section, diagnostic_type in sections.items():
        for item in diagnostics.get(section, []):
            row = dict(item)
            row["diagnostic_type"] = diagnostic_type
            if section == "underperformance_periods":
                row["dominant_assets"] = ", ".join(
                    f"{entry.get('asset')}:{entry.get('days')}"
                    for entry in item.get("dominant_assets", [])
                )
            rows.append(row)
    return rows


def build_run_payload(run: dict[str, Any]) -> dict[str, Any]:
    db = database()
    job_id = str(run["job_id"])
    symbol = str(run["symbol"])
    backend = str(run["backend"])
    run_filter = {"job_id": job_id, "symbol": symbol, "backend": backend}
    raw_predictions = list(
        db[PREDICTIONS_COLLECTION].find(run_filter, {"_id": 0}).sort("timestamp", 1)
    )
    trades = list(
        db[TRADES_COLLECTION]
        .find(run_filter, {"_id": 0})
        .sort([("timestamp", 1), ("sequence", 1)])
    )
    diagnostics = build_performance_diagnostics(
        db,
        raw_predictions,
        trades,
        run.get("metrics", {}),
    )
    series = [
        {
            "timestamp": iso_value(row.get("timestamp")),
            "strategyEquity": iso_value(row.get("strategy_equity")),
            "buyHoldEquity": iso_value(row.get("buy_hold_equity")),
        }
        for row in downsample_documents(raw_predictions)
    ]
    return {
        "key": f"{symbol}_{backend}",
        "symbol": symbol,
        "backend": backend,
        "metrics": iso_value(run.get("metrics", {})),
        "summary": run.get("summary", ""),
        "series": series,
        "trades": clean_mongo_rows(trades),
        "diagnostics": iso_value(diagnostics),
        "downloads": {
            "predictions": f"/api/jobs/{job_id}/runs/{symbol}/{backend}/predictions.csv",
            "trades": f"/api/jobs/{job_id}/runs/{symbol}/{backend}/trades.csv",
            "diagnostics": f"/api/jobs/{job_id}/runs/{symbol}/{backend}/diagnostics.csv",
            "summary": f"/api/jobs/{job_id}/runs/{symbol}/{backend}/summary.txt",
            "chart": f"/api/jobs/{job_id}/runs/{symbol}/{backend}/chart.png",
        },
    }


def build_robustness_summary(comparison_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in comparison_rows:
        if row.get("portfolio_rotation"):
            family = str(row.get("model_family") or row.get("backend") or "")
            groups.setdefault(family, []).append(row)
    summaries: list[dict[str, Any]] = []
    for family, rows in groups.items():
        capitals = pd.Series([float(row["strategy_ending_capital"]) for row in rows], dtype=float)
        excess = pd.Series([float(row["excess_return"]) for row in rows], dtype=float)
        cagrs = pd.Series([float(row["strategy_cagr"]) for row in rows], dtype=float)
        drawdowns = pd.Series([float(row["strategy_maximum_drawdown"]) for row in rows], dtype=float)
        sharpe = pd.Series([float(row["strategy_sharpe"]) for row in rows], dtype=float)
        summaries.append(
            {
                "model_family": family,
                "model_label": "XGBoost Utility" if family == "xgboost_utility" else family,
                "runs": len(rows),
                "beat_buy_hold_runs": int((excess > 0).sum()),
                "beat_buy_hold_rate": float((excess > 0).mean()),
                "ending_capital_min": float(capitals.min()),
                "ending_capital_median": float(capitals.median()),
                "ending_capital_mean": float(capitals.mean()),
                "ending_capital_max": float(capitals.max()),
                "excess_return_median": float(excess.median()),
                "cagr_median": float(cagrs.median()),
                "drawdown_median": float(drawdowns.median()),
                "drawdown_worst": float(drawdowns.min()),
                "sharpe_median": float(sharpe.median()),
                "seeds": [row.get("random_seed") for row in rows],
            }
        )
    return summaries


def build_results(job_id: str) -> dict[str, Any]:
    db = database()
    comparison = db[COMPARISONS_COLLECTION].find_one({"job_id": job_id}, {"_id": 0})
    if comparison is None:
        raise HTTPException(status_code=404, detail="Backtest results are not available yet.")
    runs = list(
        db[RUNS_COLLECTION]
        .find({"job_id": job_id}, {"_id": 0})
        .sort([("symbol", 1), ("backend", 1)])
    )
    comparison_rows = comparison.get("results", [])
    run_payloads = [build_run_payload(run) for run in runs]
    first_metrics = run_payloads[0].get("metrics", {}) if run_payloads else {}
    reproducibility = {
        key: first_metrics.get(key)
        for key in (
            "strategy_configuration_sha256",
            "market_data_signature_sha256",
            "market_data_signatures",
            "runtime_versions",
            "deterministic_execution",
            "numeric_thread_limit",
            "xgb_n_jobs",
        )
        if key in first_metrics
    }
    return {
        "jobId": job_id,
        "comparison": iso_value(comparison_rows),
        "robustnessSummary": iso_value(build_robustness_summary(comparison_rows)),
        "runs": run_payloads,
        "failures": iso_value(comparison.get("failures", [])),
        "effectiveConfig": iso_value(comparison.get("effective_config", {})),
        "reproducibility": iso_value(reproducibility),
        "downloads": {
            "zip": f"/api/jobs/{job_id}/export.zip",
            "comparison": f"/api/jobs/{job_id}/comparison.csv",
            "comparisonChart": f"/api/jobs/{job_id}/comparison.png",
        },
    }


def csv_bytes(rows: list[dict[str, Any]], excluded_fields: set[str] | None = None) -> bytes:
    if not rows:
        return b""
    excluded = excluded_fields or {"_id", "job_id"}
    cleaned_rows = [
        {key: iso_value(value) for key, value in row.items() if key not in excluded}
        for row in rows
    ]
    fieldnames: list[str] = []
    for row in cleaned_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(cleaned_rows)
    return buffer.getvalue().encode("utf-8-sig")


def csv_response(
    rows: list[dict[str, Any]],
    filename: str,
    excluded_fields: set[str] | None = None,
) -> StreamingResponse:
    return StreamingResponse(
        io.BytesIO(csv_bytes(rows, excluded_fields=excluded_fields)),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def require_run(job_id: str, symbol: str, backend: str) -> dict[str, Any]:
    run = database()[RUNS_COLLECTION].find_one(
        {"job_id": job_id, "symbol": symbol.upper(), "backend": backend.lower()}
    )
    if run is None:
        raise HTTPException(status_code=404, detail="Strategy result not found.")
    return run


def figure_response(figure: Any, filename: str) -> Response:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    return Response(
        content=buffer.getvalue(),
        media_type="image/png",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )
