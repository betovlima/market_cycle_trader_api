from __future__ import annotations

import csv
import io
from typing import Any

import matplotlib.pyplot as plt
from fastapi import HTTPException
from fastapi.responses import Response

from ..core.runtime import database
from ..infrastructure.persistence.mongo_repository import (
    COMPARISONS_COLLECTION,
    PREDICTIONS_COLLECTION,
    RUNS_COLLECTION,
    TRADES_COLLECTION,
)
from .diagnostics.performance import build_performance_diagnostics
from .serialization import downsample_documents, iso_value


SENSITIVE_DIAGNOSTIC_TOKENS = (
    "reason",
    "cause",
    "score",
    "utility",
    "seed",
    "model",
    "backend",
    "configuration",
    "training",
    "calibration",
    "feed",
    "margin",
)


def _public_diagnostics(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _public_diagnostics(item)
            for key, item in value.items()
            if not any(token in str(key).lower() for token in SENSITIVE_DIAGNOSTIC_TOKENS)
        }
    if isinstance(value, list):
        return [_public_diagnostics(item) for item in value]
    return iso_value(value)

PUBLIC_METRIC_MAP = {
    "initial_capital": "initial_capital",
    "strategy_ending_capital": "ending_capital",
    "strategy_return": "return",
    "buy_hold_ending_capital": "benchmark_ending_capital",
    "buy_hold_return": "benchmark_return",
    "excess_return": "excess_return",
    "strategy_cagr": "cagr",
    "strategy_sharpe": "sharpe",
    "strategy_sortino": "sortino",
    "strategy_maximum_drawdown": "maximum_drawdown",
    "capital_rotations": "rotations",
    "rotations": "rotations",
    "average_holding_days": "average_holding_days",
    "market_exposure": "exposure",
    "exposure": "exposure",
    "walk_forward_fold_count": "fold_count",
    "fold_count": "fold_count",
    "positive_fold_rate": "positive_fold_rate",
    "beat_buy_hold_fold_rate": "beat_benchmark_fold_rate",
    "worst_fold_return": "worst_fold_return",
    "median_fold_return": "median_fold_return",
    "median_fold_excess_return": "median_fold_excess_return",
    "robust_score": "robust_score",
}



def _public_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for source, target in PUBLIC_METRIC_MAP.items():
        if source in metrics and target not in payload:
            payload[target] = iso_value(metrics.get(source))
    return payload


def _public_trade(document: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "timestamp",
        "date",
        "action",
        "asset",
        "execution_price",
        "price",
        "quantity",
        "total_fee",
        "realized_pnl",
        "position_return",
        "cash_after_trade",
        "capital_after_trade",
        "cash_after",
        "cash",
    )
    return {key: iso_value(document.get(key)) for key in allowed if document.get(key) is not None}


def diagnostic_csv_rows(diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sections = {
        "underperformance_periods": "UNDERPERFORMANCE_PERIOD",
        "exit_diagnostics": "EXIT_COUNTERFACTUAL",
        "rotation_diagnostics": "ROTATION_COUNTERFACTUAL",
        "holding_distribution": "HOLDING_DISTRIBUTION",
        "asset_performance": "ASSET_PERFORMANCE",
        "session_performance": "SESSION_PERFORMANCE",
    }
    for section, diagnostic_type in sections.items():
        for item in diagnostics.get(section, []):
            row = dict(item)
            row.pop("q_delta", None)
            row.pop("score", None)
            row.pop("predicted_utility", None)
            row["diagnostic_type"] = diagnostic_type
            rows.append(row)
    return rows


def build_run_payload(run: dict[str, Any], index: int) -> dict[str, Any]:
    db = database()
    job_id = str(run["job_id"])
    symbol = str(run["symbol"])
    backend = str(run["backend"])
    run_filter = {"job_id": job_id, "symbol": symbol, "backend": backend}
    raw_predictions = list(db[PREDICTIONS_COLLECTION].find(run_filter, {"_id": 0}).sort("timestamp", 1))
    trades = list(db[TRADES_COLLECTION].find(run_filter, {"_id": 0}).sort([("timestamp", 1), ("sequence", 1)]))
    diagnostics = build_performance_diagnostics(db, raw_predictions, trades, run.get("metrics", {}))
    series = [
        {
            "timestamp": iso_value(row.get("timestamp")),
            "portfolioEquity": iso_value(row.get("strategy_equity")),
            "benchmarkEquity": iso_value(row.get("buy_hold_equity")),
        }
        for row in downsample_documents(raw_predictions)
    ]
    return {
        "key": f"result_{index}",
        "label": "Portfolio result",
        "metrics": _public_metrics(run.get("metrics", {})),
        "series": series,
        "trades": [_public_trade(item) for item in trades],
        "diagnostics": _public_diagnostics(diagnostics),
    }



def _operational_result(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    aggregate = [
        row
        for row in rows
        if bool((row.get("metrics") or {}).get("seed_ensemble") or row.get("seed_ensemble"))
        or str(row.get("backend") or "").lower().endswith("ensemble")
    ]
    return [aggregate[-1] if aggregate else rows[0]]

def _public_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        public = {"key": f"result_{index}", "label": "Portfolio result"}
        public.update(_public_metrics(row))
        result.append(public)
    return result


def build_results(job_id: str) -> dict[str, Any]:
    db = database()
    comparison = db[COMPARISONS_COLLECTION].find_one({"job_id": job_id}, {"_id": 0})
    if comparison is None:
        raise HTTPException(status_code=404, detail="Results are not available yet.")
    runs = list(db[RUNS_COLLECTION].find({"job_id": job_id}, {"_id": 0}).sort([("symbol", 1), ("backend", 1)]))
    public_comparison = _operational_result(list(comparison.get("results", [])))
    public_runs = _operational_result(runs)
    return {
        "jobId": job_id,
        "comparison": _public_comparison(public_comparison),
        "runs": [build_run_payload(run, index) for index, run in enumerate(public_runs, start=1)],
        "failures": [
            {"message": "An analysis result could not be completed."}
            for _ in comparison.get("failures", [])
        ],
        "downloads": {
            "zip": f"/api/jobs/{job_id}/export.zip",
            "comparison": f"/api/jobs/{job_id}/comparison.csv",
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


def csv_response(rows: list[dict[str, Any]], filename: str, excluded_fields: set[str] | None = None) -> Response:
    payload = csv_bytes(rows, excluded_fields=excluded_fields)
    return Response(
        content=payload,
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(payload)),
            "Cache-Control": "no-store",
        },
    )


def require_run(job_id: str, symbol: str, backend: str) -> dict[str, Any]:
    run = database()[RUNS_COLLECTION].find_one({"job_id": job_id, "symbol": symbol.upper(), "backend": backend.lower()})
    if run is None:
        raise HTTPException(status_code=404, detail="Result not found.")
    return run


def figure_response(figure: Any, filename: str) -> Response:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    return Response(content=buffer.getvalue(), media_type="image/png", headers={"Content-Disposition": f'inline; filename="{filename}"'})
