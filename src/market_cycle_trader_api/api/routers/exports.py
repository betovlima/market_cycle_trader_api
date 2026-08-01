from __future__ import annotations

import io
import json
import zipfile
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse

from ...core.runtime import database
from ...infrastructure.persistence.mongo_repository import COMPARISONS_COLLECTION, PREDICTIONS_COLLECTION, RUNS_COLLECTION, TRADES_COLLECTION
from ...services.diagnostics.performance import build_performance_diagnostics
from ...services.jobs import require_job
from ...services.results import csv_bytes, csv_response, diagnostic_csv_rows, figure_response, require_run
from ...services.serialization import iso_value

router = APIRouter(tags=["exports"])

@router.get("/api/jobs/{job_id}/comparison.csv")
def export_comparison(job_id: str) -> StreamingResponse:
    require_job(job_id)
    document = database()[COMPARISONS_COLLECTION].find_one(
        {"job_id": job_id},
        {"_id": 0, "results": 1},
    )
    if document is None:
        raise HTTPException(status_code=404, detail="Comparison not found.")
    return csv_response(
        document.get("results", []),
        f"comparison_{job_id}.csv",
    )


@router.get("/api/jobs/{job_id}/runs/{symbol}/{backend}/predictions.csv")
def export_predictions(
    job_id: str,
    symbol: str,
    backend: str,
) -> StreamingResponse:
    require_run(job_id, symbol, backend)
    rows = list(
        database()[PREDICTIONS_COLLECTION]
        .find(
            {
                "job_id": job_id,
                "symbol": symbol.upper(),
                "backend": backend.lower(),
            },
            {"_id": 0},
        )
        .sort("timestamp", 1)
    )
    return csv_response(
        rows,
        f"{symbol.upper()}_{backend.lower()}_predictions.csv",
        excluded_fields={"_id", "job_id", "symbol", "backend"},
    )


@router.get("/api/jobs/{job_id}/runs/{symbol}/{backend}/trades.csv")
def export_trades(
    job_id: str,
    symbol: str,
    backend: str,
) -> StreamingResponse:
    require_run(job_id, symbol, backend)
    rows = list(
        database()[TRADES_COLLECTION]
        .find(
            {
                "job_id": job_id,
                "symbol": symbol.upper(),
                "backend": backend.lower(),
            },
            {"_id": 0},
        )
        .sort([("timestamp", 1), ("sequence", 1)])
    )
    return csv_response(
        rows,
        f"{symbol.upper()}_{backend.lower()}_trades.csv",
        excluded_fields={"_id", "job_id", "symbol", "backend"},
    )


@router.get("/api/jobs/{job_id}/runs/{symbol}/{backend}/diagnostics.csv")
def get_run_diagnostics_csv(
    job_id: str,
    symbol: str,
    backend: str,
) -> StreamingResponse:
    require_job(job_id)
    db = database()
    run_filter = {
        "job_id": job_id,
        "symbol": symbol,
        "backend": backend,
    }
    run = db[RUNS_COLLECTION].find_one(run_filter, {"_id": 0})
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found.")

    predictions = list(
        db[PREDICTIONS_COLLECTION]
        .find(run_filter, {"_id": 0})
        .sort("timestamp", 1)
    )
    trades = list(
        db[TRADES_COLLECTION]
        .find(run_filter, {"_id": 0})
        .sort([("timestamp", 1), ("sequence", 1)])
    )
    diagnostics = build_performance_diagnostics(
        db,
        predictions,
        trades,
        run.get("metrics", {}),
    )
    return csv_response(
        diagnostic_csv_rows(diagnostics),
        f"{symbol}_{backend}_diagnostics.csv",
    )


@router.get("/api/jobs/{job_id}/runs/{symbol}/{backend}/summary.txt")
def export_summary(job_id: str, symbol: str, backend: str) -> Response:
    run = require_run(job_id, symbol, backend)
    return Response(
        content=str(run.get("summary", "")),
        media_type="text/plain; charset=utf-8",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{symbol.upper()}_'
                f'{backend.lower()}_summary.txt"'
            )
        },
    )


@router.get("/api/jobs/{job_id}/runs/{symbol}/{backend}/chart.png")
def export_chart(job_id: str, symbol: str, backend: str) -> Response:
    require_run(job_id, symbol, backend)
    rows = list(
        database()[PREDICTIONS_COLLECTION]
        .find(
            {
                "job_id": job_id,
                "symbol": symbol.upper(),
                "backend": backend.lower(),
            },
            {
                "_id": 0,
                "timestamp": 1,
                "close": 1,
                "strategy_equity": 1,
                "buy_hold_equity": 1,
            },
        )
        .sort("timestamp", 1)
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Prediction series not found.")

    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)

    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(frame["timestamp"], frame["strategy_equity"], label="Strategy")
    axis.plot(frame["timestamp"], frame["buy_hold_equity"], label="Buy and hold")
    axis.set_title(f"{symbol.upper()} · {backend.lower()}")
    axis.set_ylabel("Equity")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    return figure_response(
        figure,
        f"{symbol.upper()}_{backend.lower()}_chart.png",
    )


@router.get("/api/jobs/{job_id}/comparison.png")
def export_comparison_chart(job_id: str) -> Response:
    document = database()[COMPARISONS_COLLECTION].find_one(
        {"job_id": job_id},
        {"_id": 0, "results": 1},
    )
    if document is None:
        raise HTTPException(status_code=404, detail="Comparison not found.")

    frame = pd.DataFrame(document.get("results", []))
    if frame.empty:
        raise HTTPException(status_code=404, detail="Comparison is empty.")

    labels = frame["symbol"].astype(str) + " · " + frame["backend"].astype(str)
    x = range(len(frame))

    figure, axis = plt.subplots(figsize=(12, 5))
    width = 0.38
    axis.bar(
        [item - width / 2 for item in x],
        frame["strategy_return"] * 100,
        width,
        label="Strategy",
    )
    axis.bar(
        [item + width / 2 for item in x],
        frame["buy_hold_return"] * 100,
        width,
        label="Buy and hold",
    )
    axis.set_xticks(list(x), labels, rotation=30, ha="right")
    axis.set_ylabel("Return (%)")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    return figure_response(figure, f"comparison_{job_id}.png")


@router.get("/api/jobs/{job_id}/export.zip")
def export_zip(job_id: str) -> StreamingResponse:
    require_job(job_id)
    db = database()

    comparison = db[COMPARISONS_COLLECTION].find_one(
        {"job_id": job_id},
        {"_id": 0},
    )
    if comparison is None:
        raise HTTPException(status_code=404, detail="Comparison not found.")

    runs = list(
        db[RUNS_COLLECTION]
        .find({"job_id": job_id}, {"_id": 0})
        .sort([("symbol", 1), ("backend", 1)])
    )

    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(
        archive_buffer,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr(
            "multi_asset_comparison.csv",
            csv_bytes(comparison.get("results", [])),
        )
        archive.writestr(
            "failures.csv",
            csv_bytes(comparison.get("failures", [])),
        )
        for run in runs:
            symbol = str(run["symbol"])
            backend = str(run["backend"])
            folder = f"{symbol}_{backend}"

            run_filter = {
                "job_id": job_id,
                "symbol": symbol,
                "backend": backend,
            }
            predictions = list(
                db[PREDICTIONS_COLLECTION]
                .find(run_filter, {"_id": 0})
                .sort("timestamp", 1)
            )
            trades = list(
                db[TRADES_COLLECTION]
                .find(run_filter, {"_id": 0})
                .sort([("timestamp", 1), ("sequence", 1)])
            )
            metrics = run.get("metrics", {})
            diagnostics = build_performance_diagnostics(
                db,
                predictions,
                trades,
                metrics,
            )

            archive.writestr(
                f"{folder}/{symbol}_{backend}_metrics.json",
                json.dumps(
                    iso_value(metrics),
                    indent=2,
                    ensure_ascii=False,
                ),
            )
            walk_forward_folds = metrics.get("walk_forward_folds", [])
            if walk_forward_folds:
                archive.writestr(
                    f"{folder}/{symbol}_{backend}_walk_forward_folds.csv",
                    csv_bytes(walk_forward_folds),
                )
            archive.writestr(
                f"{folder}/{symbol}_{backend}_summary.txt",
                str(run.get("summary", "")),
            )
            archive.writestr(
                f"{folder}/{symbol}_{backend}_predictions.csv",
                csv_bytes(
                    predictions,
                    excluded_fields={"_id", "job_id", "symbol", "backend"},
                ),
            )
            archive.writestr(
                f"{folder}/{symbol}_{backend}_trades.csv",
                csv_bytes(
                    trades,
                    excluded_fields={"_id", "job_id", "symbol", "backend"},
                ),
            )
            archive.writestr(
                f"{folder}/{symbol}_{backend}_diagnostics.csv",
                csv_bytes(diagnostic_csv_rows(diagnostics)),
            )
            archive.writestr(
                f"{folder}/{symbol}_{backend}_diagnostics.json",
                json.dumps(
                    iso_value(diagnostics),
                    indent=2,
                    ensure_ascii=False,
                ),
            )

            if str(metrics.get("strategy_mode", "")) == "COMPOUND_ROTATION_DAY_TRADE_OPEN_CLOSE":
                decision_rows: list[dict[str, Any]] = []
                value_rows: list[dict[str, Any]] = []
                for prediction in predictions:
                    decision_rows.append(
                        {
                            key: value
                            for key, value in prediction.items()
                            if key in {
                                "timestamp",
                                "decision_date",
                                "selected_asset",
                                "trade_action",
                                "trade_reason",
                                "session_return",
                                "decision_score",
                                "walk_forward_fold",
                                "fold_test_start",
                                "fold_test_end",
                                "strategy_equity",
                                "buy_hold_equity",
                                "reference_buy_hold_equity",
                            }
                        }
                    )
                    value_rows.append(
                        {
                            key: value
                            for key, value in prediction.items()
                            if key.startswith("decision_value_")
                            or key in {
                                "timestamp",
                                "decision_date",
                                "selected_asset",
                                "decision_score",
                                "q_gap_best_vs_second",
                                "walk_forward_fold",
                            }
                        }
                    )

                archive.writestr(
                    f"{folder}/{symbol}_{backend}_session_decisions.csv",
                    csv_bytes(decision_rows),
                )
                archive.writestr(
                    f"{folder}/{symbol}_{backend}_decision_values.csv",
                    csv_bytes(value_rows),
                )

    archive_buffer.seek(0)
    return StreamingResponse(
        archive_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": (
                f'attachment; filename="market_cycle_trader_{job_id}.zip"'
            )
        },
    )
