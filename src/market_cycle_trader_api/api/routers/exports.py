from __future__ import annotations

import io
import json
import zipfile
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from ...core.runtime import database
from ...infrastructure.persistence.mongo_repository import COMPARISONS_COLLECTION, PREDICTIONS_COLLECTION, RUNS_COLLECTION, TRADES_COLLECTION
from ...services.diagnostics.performance import build_performance_diagnostics
from ...services.experiment_manifest import build_experiment_manifest
from ...services.jobs import require_job
from ...services.results import csv_bytes, csv_response, diagnostic_csv_rows, figure_response, require_run
from ...services.serialization import iso_value

router = APIRouter(tags=["exports"])

XGBOOST_DECISION_DIAGNOSTIC_FIELDS = (
    "timestamp",
    "decision_date",
    "walk_forward_fold",
    "fold_test_start",
    "fold_test_end",
    "current_asset",
    "current_score",
    "strategy_selective_opportunity_enabled",
    "strategy_opportunity_cash_gate_enabled",
    "strategy_absolute_utility_cash_gate_enabled",
    "absolute_utility_best_score",
    "absolute_utility_entry_threshold",
    "absolute_utility_exit_threshold",
    "absolute_utility_active_threshold",
    "absolute_utility_accepted",
    "absolute_utility_hysteresis_market_hold",
    "absolute_utility_hysteresis_cash_block",
    "opportunity_probability",
    "opportunity_confidence",
    "opportunity_threshold",
    "opportunity_entry_threshold",
    "opportunity_exit_threshold",
    "opportunity_active_threshold",
    "opportunity_threshold_basis",
    "opportunity_target_basis",
    "opportunity_target_horizon_sessions",
    "opportunity_adaptive_refresh_count",
    "opportunity_regularized_to_base_policy",
    "opportunity_validation_alpha",
    "opportunity_validation_exposure_ratio",
    "opportunity_decision_value",
    "opportunity_accepted",
    "opportunity_hysteresis_market_hold",
    "opportunity_hysteresis_cash_block",
    "cash_gate_base_position_before",
    "cash_gate_base_holding_days_before",
    "cash_gate_base_action_asset",
    "cash_gate_base_action_score",
    "cash_gate_base_decision_reason",
    "cash_gate_changed_base_action",
    "cash_gate_counterfactual_asset",
    "cash_gate_counterfactual_open_to_close_return",
    "cash_gate_avoided_loss_return",
    "cash_gate_missed_gain_return",
    "holding_days_at_decision",
    "raw_best_asset",
    "raw_best_score",
    "best_asset",
    "best_score",
    "second_asset",
    "second_score",
    "best_vs_second_gap",
    "best_vs_current_gap",
    "best_vs_cash_gap",
    "cash_score",
    "current_asset_rank",
    "universe_score_mean",
    "universe_score_std",
    "current_score_zscore",
    "best_score_zscore",
    "best_vs_second_zscore",
    "positive_score_count",
    "finite_score_count",
    "rotation_cash_threshold",
    "rotation_min_expected_edge",
    "base_switch_margin",
    "calibrated_switch_margin",
    "effective_switch_margin",
    "final_action_asset",
    "final_action_score",
    "decision_reason",
    "decision_is_rotation",
    "decision_is_entry",
    "decision_is_exit_to_cash",
    "min_hold_guard_applied",
    "switch_margin_guard_applied",
    "cash_threshold_guard_applied",
    "minimum_expected_edge_guard_applied",
    "top_1_asset",
    "top_1_score",
    "top_2_asset",
    "top_2_score",
    "top_3_asset",
    "top_3_score",
    "position_risk_diagnostics_schema_version",
    "position_entry_timestamp",
    "position_entry_price",
    "position_entry_score",
    "position_return_since_entry",
    "position_peak_return",
    "position_drawdown_from_peak",
    "position_mfe_so_far",
    "position_mae_so_far",
    "score_change_from_entry",
    "days_current_not_top1",
    "consecutive_days_current_not_top1",
    "market_regime_diagnostics_schema_version",
    "spy_return_5",
    "spy_return_20",
    "spy_realized_volatility_20",
    "universe_breadth_5",
    "universe_breadth_20",
    "universe_breadth_5_valid_assets",
    "universe_breadth_20_valid_assets",
    "selected_asset",
    "previous_asset",
    "trade_action",
    "strategy_equity",
    "buy_hold_equity",
)


def _xgboost_decision_rows(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: row.get(key) for key in XGBOOST_DECISION_DIAGNOSTIC_FIELDS}
        for row in predictions
        if row.get("decision_diagnostics_schema_version") is not None
    ]

@router.get("/api/jobs/{job_id}/comparison.csv")
def export_comparison(job_id: str) -> Response:
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
) -> Response:
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
) -> Response:
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


@router.get("/api/jobs/{job_id}/runs/{symbol}/{backend}/decision-diagnostics.csv")
def export_decision_diagnostics(
    job_id: str,
    symbol: str,
    backend: str,
) -> Response:
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
        _xgboost_decision_rows(rows),
        f"{symbol.upper()}_{backend.lower()}_decision_diagnostics.csv",
    )


@router.get("/api/jobs/{job_id}/runs/{symbol}/{backend}/diagnostics.csv")
def get_run_diagnostics_csv(
    job_id: str,
    symbol: str,
    backend: str,
) -> Response:
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
def export_zip(job_id: str) -> Response:
    job = require_job(job_id)
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
        archive.writestr(
            "strategy_manifest.json",
            json.dumps(
                iso_value(
                    {
                        "job_id": job_id,
                        "status": job.get("status"),
                        "strategy_profile_id": job.get("strategy_profile_id"),
                        "strategy_profile_name": job.get("strategy_profile_name"),
                        "strategy_profile_revision": job.get("strategy_profile_revision"),
                        "strategy_configuration_hash": job.get("strategy_configuration_hash"),
                        "winner_engine_compatibility": job.get("winner_engine_compatibility"),
                        "numeric_thread_environment_applied": job.get("numeric_thread_environment_applied"),
                        "created_at": job.get("created_at"),
                        "started_at": job.get("started_at"),
                        "finished_at": job.get("finished_at"),
                        "configuration": job.get("request"),
                    }
                ),
                indent=2,
                ensure_ascii=False,
            ),
        )
        archive.writestr(
            "experiment_manifest.json",
            json.dumps(
                iso_value(build_experiment_manifest(job, runs)),
                indent=2,
                ensure_ascii=False,
            ),
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
            diagnostics_error: str | None = None
            try:
                diagnostics = build_performance_diagnostics(
                    db,
                    predictions,
                    trades,
                    metrics,
                )
            except Exception as exc:  
                diagnostics = {}
                diagnostics_error = f"{type(exc).__name__}: {exc}"

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
            decision_rows = _xgboost_decision_rows(predictions)
            if decision_rows:
                archive.writestr(
                    f"{folder}/{symbol}_{backend}_decision_diagnostics.csv",
                    csv_bytes(decision_rows),
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
            if diagnostics_error:
                archive.writestr(
                    f"{folder}/{symbol}_{backend}_diagnostics_error.txt",
                    diagnostics_error,
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

    payload = archive_buffer.getvalue()
    return Response(
        content=payload,
        media_type="application/zip",
        headers={
            "Content-Disposition": (
                f'attachment; filename="market_cycle_trader_{job_id}.zip"'
            ),
            "Content-Length": str(len(payload)),
            "Cache-Control": "no-store",
        },
    )
