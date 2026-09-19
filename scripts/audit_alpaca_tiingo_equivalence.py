from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from market_cycle_trader_api.core.environment import load_project_environment

load_project_environment()

from market_cycle_trader_api.engine.capital_rotation import (
    ROTATION_FEATURES,
    build_rotation_frame,
    run_rotation_models,
)
from market_cycle_trader_api.infrastructure.persistence.mongo_repository import (
    ALPACA_MARKET_BARS_COLLECTION,
    JOBS_COLLECTION,
    TIINGO_MARKET_BARS_COLLECTION,
    create_client,
    get_database,
)
from market_cycle_trader_api.schemas.requests import BacktestExecutionRequest, BacktestRequest


OHLCV = ("open", "high", "low", "close", "volume")
PRICE_COLUMNS = ("open", "high", "low", "close")


def _round_fee_to_cent(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        return 0.0
    return math.ceil((value - 1e-12) * 100.0) / 100.0


def _calculate_reference_fees(
    side: str,
    quantity: float,
    price: float,
    config: BacktestRequest,
) -> dict[str, float]:
    """Exact fee contract used by the normal compound-rotation backtest.

    Kept local to this standalone audit to avoid importing the backtest CLI
    module, which initializes service packages and creates a circular import
    through Asset Discovery.
    """
    if quantity <= 0 or price <= 0:
        return {
            "commission_fee": 0.0,
            "sec_fee": 0.0,
            "taf_fee": 0.0,
            "cat_fee": 0.0,
            "total_fee": 0.0,
        }

    normalized_side = side.upper()
    trade_value = quantity * price
    commission = _round_fee_to_cent(trade_value * config.commission_rate)
    cat = _round_fee_to_cent(quantity * config.cat_fee_per_share)
    sec = 0.0
    taf = 0.0

    if normalized_side == "SELL":
        sec = _round_fee_to_cent(trade_value * config.sec_fee_rate)
        taf = _round_fee_to_cent(
            min(quantity * config.taf_fee_per_share, config.taf_fee_cap)
        )
    elif normalized_side != "BUY":
        raise ValueError(f"Unsupported side: {side}")

    return {
        "commission_fee": commission,
        "sec_fee": sec,
        "taf_fee": taf,
        "cat_fee": cat,
        "total_fee": commission + sec + taf + cat,
    }


def _apply_slippage(
    price: float,
    side: str,
    config: BacktestRequest,
) -> float:
    adjustment = config.slippage_bps / 10_000
    return price * (1 + adjustment if side == "BUY" else 1 - adjustment)


def _utc(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError(f"Invalid timestamp: {value}")
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


def _latest_tiingo_job(db: Any, job_id: str | None) -> dict[str, Any]:
    if job_id:
        job = db[JOBS_COLLECTION].find_one({"id": str(job_id)})
    else:
        job = db[JOBS_COLLECTION].find_one(
            {
                "internal_job": {"$ne": True},
                "status": "completed",
                "request.market_data_provider": "tiingo",
            },
            sort=[("finished_at", -1), ("created_at", -1)],
        )
    if job is None:
        raise RuntimeError(
            "No completed Tiingo backtest job was found. "
            "Run the Tiingo backtest first or pass --job-id."
        )
    request = job.get("request")
    if not isinstance(request, dict):
        raise RuntimeError("Selected job has no immutable request snapshot.")
    return job


def _read_cache(
    collection: Any,
    *,
    symbol: str,
    interval: str,
    feed: str,
    adjustment: str,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp | None,
) -> pd.DataFrame:
    query: dict[str, Any] = {
        "symbol": str(symbol).strip().upper(),
        "interval": interval,
        "feed": feed,
        "adjustment": adjustment,
        "timestamp": {"$gte": start.to_pydatetime()},
    }
    if end_exclusive is not None:
        query["timestamp"]["$lt"] = end_exclusive.to_pydatetime()

    projection = {"_id": 0, "timestamp": 1, **{column: 1 for column in OHLCV}}
    rows = list(collection.find(query, projection).sort("timestamp", 1))
    if not rows:
        return pd.DataFrame(columns=list(OHLCV))

    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    for column in OHLCV:
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=list(OHLCV))
    frame = frame[(frame[list(PRICE_COLUMNS)] > 0).all(axis=1)]
    frame = frame[frame["volume"] >= 0]
    return frame[list(OHLCV)]


def _first_and_last(frame: pd.DataFrame) -> tuple[str | None, str | None]:
    if frame.empty:
        return None, None
    return (
        pd.Timestamp(frame.index.min()).date().isoformat(),
        pd.Timestamp(frame.index.max()).date().isoformat(),
    )


def _coverage_ready(
    frame: pd.DataFrame,
    *,
    start: pd.Timestamp,
    tolerance_days: int,
    minimum_rows: int,
) -> tuple[bool, str]:
    if frame.empty:
        return False, "missing"
    first = _utc(frame.index.min()).normalize()
    if first > start.normalize() + pd.Timedelta(days=tolerance_days):
        return False, "late_history_start"
    if len(frame) < minimum_rows:
        return False, "insufficient_rows"
    return True, "ready"


def _relative_difference(left: pd.Series, right: pd.Series) -> pd.Series:
    denominator = left.abs().replace(0.0, np.nan)
    return (right - left).abs() / denominator


def _normalize_session_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize daily-bar timestamps to the UTC calendar session date.

    Alpaca daily bars are stored at New York midnight converted to UTC
    (04:00/05:00 depending on daylight saving time), while Tiingo EOD bars are
    stored at 00:00 UTC. For 1Day equivalence we compare the trading session
    date, not the provider-specific timestamp representation.
    """
    result = frame.copy()
    result.index = pd.to_datetime(result.index, utc=True).normalize()
    result = result[~result.index.duplicated(keep="last")].sort_index()
    return result


def _compare_asset(
    symbol: str,
    alpaca: pd.DataFrame,
    tiingo: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    alpaca = _normalize_session_index(alpaca)
    tiingo = _normalize_session_index(tiingo)
    alpaca_dates = pd.DatetimeIndex(alpaca.index)
    tiingo_dates = pd.DatetimeIndex(tiingo.index)
    common_dates = alpaca_dates.intersection(tiingo_dates).sort_values()
    only_alpaca = alpaca_dates.difference(tiingo_dates).sort_values()
    only_tiingo = tiingo_dates.difference(alpaca_dates).sort_values()

    missing_rows: list[dict[str, Any]] = []
    for timestamp in only_alpaca:
        missing_rows.append(
            {
                "symbol": symbol,
                "timestamp": pd.Timestamp(timestamp).isoformat(),
                "present_in": "alpaca_only",
            }
        )
    for timestamp in only_tiingo:
        missing_rows.append(
            {
                "symbol": symbol,
                "timestamp": pd.Timestamp(timestamp).isoformat(),
                "present_in": "tiingo_only",
            }
        )

    if common_dates.empty:
        return (
            {
                "symbol": symbol,
                "common_rows": 0,
                "alpaca_only_rows": len(only_alpaca),
                "tiingo_only_rows": len(only_tiingo),
            },
            pd.DataFrame(),
            pd.DataFrame(missing_rows),
        )

    left = alpaca.loc[common_dates, list(OHLCV)].astype(float)
    right = tiingo.loc[common_dates, list(OHLCV)].astype(float)

    detail = pd.DataFrame(index=common_dates)
    detail.index.name = "timestamp"
    detail["symbol"] = symbol
    material = pd.Series(False, index=common_dates)

    summary: dict[str, Any] = {
        "symbol": symbol,
        "common_rows": int(len(common_dates)),
        "alpaca_only_rows": int(len(only_alpaca)),
        "tiingo_only_rows": int(len(only_tiingo)),
    }

    for column in OHLCV:
        detail[f"alpaca_{column}"] = left[column]
        detail[f"tiingo_{column}"] = right[column]
        absolute = (right[column] - left[column]).abs()
        relative = _relative_difference(left[column], right[column])
        detail[f"{column}_abs_diff"] = absolute
        detail[f"{column}_rel_diff"] = relative

        finite_relative = relative.replace([np.inf, -np.inf], np.nan).dropna()
        summary[f"{column}_mean_abs_diff"] = float(absolute.mean())
        summary[f"{column}_max_abs_diff"] = float(absolute.max())
        summary[f"{column}_mean_rel_diff"] = (
            float(finite_relative.mean()) if not finite_relative.empty else None
        )
        summary[f"{column}_p95_rel_diff"] = (
            float(finite_relative.quantile(0.95)) if not finite_relative.empty else None
        )
        summary[f"{column}_max_rel_diff"] = (
            float(finite_relative.max()) if not finite_relative.empty else None
        )

        threshold = 1e-4 if column in PRICE_COLUMNS else 0.01
        material = material | (relative.fillna(0.0) > threshold)

    detail["material_difference"] = material
    exact = np.ones(len(common_dates), dtype=bool)
    for column in OHLCV:
        exact &= np.equal(left[column].to_numpy(), right[column].to_numpy())
    detail["exact_ohlcv_match"] = exact

    summary["exact_match_rows"] = int(np.sum(exact))
    summary["exact_match_rate"] = float(np.mean(exact))
    summary["material_difference_rows"] = int(material.sum())
    summary["material_difference_rate"] = float(material.mean())

    close_ratio = (right["close"] / left["close"]).replace([np.inf, -np.inf], np.nan).dropna()
    summary["tiingo_to_alpaca_close_ratio_median"] = (
        float(close_ratio.median()) if not close_ratio.empty else None
    )
    summary["tiingo_to_alpaca_close_ratio_p05"] = (
        float(close_ratio.quantile(0.05)) if not close_ratio.empty else None
    )
    summary["tiingo_to_alpaca_close_ratio_p95"] = (
        float(close_ratio.quantile(0.95)) if not close_ratio.empty else None
    )

    detail = detail.reset_index()
    detail = detail[
        (~detail["exact_ohlcv_match"])
        | detail["material_difference"]
    ].copy()
    return summary, detail, pd.DataFrame(missing_rows)


TARGET_COLUMNS = (
    "forward_net_log_return",
    "forward_cash_edge",
    "forward_movement_capture",
    "forward_trend_persistence",
    "forward_risk_adjusted_utility",
)


def _compare_model_inputs(
    symbol: str,
    alpaca: pd.DataFrame,
    tiingo: pd.DataFrame,
    config: BacktestExecutionRequest,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    left = build_rotation_frame(_normalize_session_index(alpaca), config)
    right = build_rotation_frame(_normalize_session_index(tiingo), config)
    common = left.index.intersection(right.index).sort_values()
    columns = [
        column
        for column in [*ROTATION_FEATURES, *TARGET_COLUMNS]
        if column in left.columns and column in right.columns
    ]
    summaries: list[dict[str, Any]] = []
    if common.empty:
        return summaries, left, right

    for column in columns:
        a = pd.to_numeric(left.loc[common, column], errors="coerce")
        t = pd.to_numeric(right.loc[common, column], errors="coerce")
        valid = pd.DataFrame({"alpaca": a, "tiingo": t}).replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        if valid.empty:
            continue
        delta = valid["tiingo"] - valid["alpaca"]
        abs_delta = delta.abs()
        corr = (
            float(valid["alpaca"].corr(valid["tiingo"]))
            if len(valid) >= 2
            else None
        )
        summaries.append(
            {
                "symbol": symbol,
                "column": column,
                "kind": "target" if column in TARGET_COLUMNS else "feature",
                "common_rows": int(len(valid)),
                "mean_abs_diff": float(abs_delta.mean()),
                "p95_abs_diff": float(abs_delta.quantile(0.95)),
                "max_abs_diff": float(abs_delta.max()),
                "correlation": corr,
            }
        )

    return summaries, left, right


def _first_divergence_input_snapshot(
    first_divergence: dict[str, Any] | None,
    alpaca_inputs: dict[str, pd.DataFrame],
    tiingo_inputs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    if not first_divergence:
        return pd.DataFrame()
    timestamp = _utc(first_divergence["timestamp"]).normalize()
    symbols = list(
        dict.fromkeys(
            [
                str(first_divergence.get("alpaca_asset") or "").strip().upper(),
                str(first_divergence.get("tiingo_asset") or "").strip().upper(),
            ]
        )
    )
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        if not symbol:
            continue
        left = alpaca_inputs.get(symbol)
        right = tiingo_inputs.get(symbol)
        if left is None or right is None or timestamp not in left.index or timestamp not in right.index:
            continue
        for column in [*ROTATION_FEATURES, *TARGET_COLUMNS]:
            if column not in left.columns or column not in right.columns:
                continue
            a = left.at[timestamp, column]
            t = right.at[timestamp, column]
            if pd.isna(a) or pd.isna(t):
                continue
            rows.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "symbol": symbol,
                    "column": column,
                    "kind": "target" if column in TARGET_COLUMNS else "feature",
                    "alpaca_value": float(a),
                    "tiingo_value": float(t),
                    "abs_diff": float(abs(float(t) - float(a))),
                }
            )
    return pd.DataFrame(rows)


def _compare_pretest_inputs(
    alpaca_inputs: dict[str, pd.DataFrame],
    tiingo_inputs: dict[str, pd.DataFrame],
    first_test_session: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []
    cutoff = _utc(first_test_session).normalize()

    for symbol in sorted(set(alpaca_inputs).intersection(tiingo_inputs)):
        left = alpaca_inputs[symbol]
        right = tiingo_inputs[symbol]
        common = left.index.intersection(right.index)
        common = common[common < cutoff].sort_values()
        if common.empty:
            continue

        for column in [*ROTATION_FEATURES, *TARGET_COLUMNS]:
            if column not in left.columns or column not in right.columns:
                continue
            a = pd.to_numeric(left.loc[common, column], errors="coerce")
            t = pd.to_numeric(right.loc[common, column], errors="coerce")
            valid = pd.DataFrame({"alpaca": a, "tiingo": t}).replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            if valid.empty:
                continue

            delta = valid["tiingo"] - valid["alpaca"]
            abs_delta = delta.abs()
            corr = (
                float(valid["alpaca"].corr(valid["tiingo"]))
                if len(valid) >= 2
                else None
            )
            kind = "target" if column in TARGET_COLUMNS else "feature"
            summaries.append(
                {
                    "symbol": symbol,
                    "column": column,
                    "kind": kind,
                    "common_rows": int(len(valid)),
                    "first_session": pd.Timestamp(valid.index.min()).isoformat(),
                    "last_session": pd.Timestamp(valid.index.max()).isoformat(),
                    "mean_abs_diff": float(abs_delta.mean()),
                    "p95_abs_diff": float(abs_delta.quantile(0.95)),
                    "max_abs_diff": float(abs_delta.max()),
                    "correlation": corr,
                }
            )

            for timestamp in abs_delta.nlargest(min(5, len(abs_delta))).index:
                anomalies.append(
                    {
                        "symbol": symbol,
                        "timestamp": pd.Timestamp(timestamp).isoformat(),
                        "column": column,
                        "kind": kind,
                        "alpaca_value": float(valid.at[timestamp, "alpaca"]),
                        "tiingo_value": float(valid.at[timestamp, "tiingo"]),
                        "abs_diff": float(abs_delta.loc[timestamp]),
                    }
                )

    return pd.DataFrame(summaries), pd.DataFrame(anomalies)


def _decision_column(frame: pd.DataFrame) -> str | None:
    for column in (
        "final_action_asset",
        "selected_asset",
        "best_asset",
        "raw_best_asset",
        "target_asset",
    ):
        if column in frame.columns:
            return column
    return None


def _compare_predictions(
    alpaca: pd.DataFrame,
    tiingo: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if alpaca is None or tiingo is None or alpaca.empty or tiingo.empty:
        return {"available": False, "reason": "empty_predictions"}, pd.DataFrame()

    left = _normalize_session_index(alpaca)
    right = _normalize_session_index(tiingo)
    common = left.index.intersection(right.index).sort_values()
    if common.empty:
        return {"available": False, "reason": "no_common_prediction_dates"}, pd.DataFrame()

    left_col = _decision_column(left)
    right_col = _decision_column(right)
    if left_col is None or right_col is None:
        return {
            "available": False,
            "reason": "decision_asset_column_not_found",
            "alpaca_columns": list(left.columns),
            "tiingo_columns": list(right.columns),
        }, pd.DataFrame()

    rows: list[dict[str, Any]] = []
    first: dict[str, Any] | None = None
    divergence_count = 0

    score_columns = [
        column
        for column in (
            "best_score",
            "raw_best_score",
            "final_action_score",
            "current_score",
            "second_score",
            "best_vs_second_gap",
        )
        if column in left.columns and column in right.columns
    ]

    for timestamp in common:
        left_asset = str(left.at[timestamp, left_col])
        right_asset = str(right.at[timestamp, right_col])
        if left_asset == right_asset:
            continue
        divergence_count += 1
        row: dict[str, Any] = {
            "timestamp": pd.Timestamp(timestamp).isoformat(),
            "alpaca_asset": left_asset,
            "tiingo_asset": right_asset,
        }
        for column in score_columns:
            row[f"alpaca_{column}"] = left.at[timestamp, column]
            row[f"tiingo_{column}"] = right.at[timestamp, column]
        rows.append(row)
        if first is None:
            first = dict(row)

    return {
        "available": True,
        "decision_column_alpaca": left_col,
        "decision_column_tiingo": right_col,
        "common_prediction_dates": int(len(common)),
        "divergent_decision_dates": int(divergence_count),
        "divergent_decision_rate": float(divergence_count / len(common)),
        "first_divergence": first,
    }, pd.DataFrame(rows)


def _run_source(
    label: str,
    frames: dict[str, pd.DataFrame],
    config: BacktestExecutionRequest,
) -> Any:
    print(f"[model] Starting controlled {label} replay with {len(frames)} assets...", flush=True)

    def progress(percent: float, stage: str, completed: int) -> None:
        print(
            f"[model] {label} progress={percent:.1f}% completed={completed} stage={stage}",
            flush=True,
        )

    results = run_rotation_models(
        frames,
        config,
        _calculate_reference_fees,
        _apply_slippage,
        progress_callback=progress,
    )
    if not results:
        raise RuntimeError(f"{label} replay returned no results.")
    result = results[0]
    print(
        f"[model] {label} ending_capital="
        f"{float(result.metrics.get('strategy_ending_capital') or 0.0):,.2f}",
        flush=True,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare cached Alpaca and Tiingo OHLCV candle-by-candle and rerun "
            "the same MCT model on the largest common, fully modelable universe."
        )
    )
    parser.add_argument("--job-id", default=None, help="Completed Tiingo Backtest job id.")
    parser.add_argument(
        "--output-dir",
        default="output/alpaca_tiingo_equivalence_audit",
    )
    parser.add_argument(
        "--skip-model-replay",
        action="store_true",
        help="Run only candle/cache equivalence, without two LightGBM replays.",
    )
    args = parser.parse_args()

    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    client = create_client()
    try:
        db = get_database(client)
        job = _latest_tiingo_job(db, args.job_id)
        request = BacktestExecutionRequest.model_validate(job["request"])

        start = _utc(request.start_date).normalize()
        requested_end_text = request.analysis_end_date or request.end_date
        requested_end = _utc(requested_end_text).normalize() if requested_end_text else None
        end_exclusive = (
            requested_end + pd.Timedelta(days=1)
            if requested_end is not None
            else None
        )

        alpaca_collection = db[ALPACA_MARKET_BARS_COLLECTION]
        tiingo_collection = db[TIINGO_MARKET_BARS_COLLECTION]

        minimum_rows = (
            int(request.rotation_minimum_training_rows)
            + max(int(item) for item in request.rotation_target_horizons)
            + int(request.rotation_purge_days)
        )
        tolerance_days = int(request.market_data_history_start_tolerance_days)

        raw: dict[str, dict[str, pd.DataFrame]] = {}
        coverage_rows: list[dict[str, Any]] = []
        eligible_assets: list[str] = []

        print(
            f"[data] job={job.get('id')} assets={len(request.assets)} "
            f"start={request.start_date} requested_end={requested_end_text}",
            flush=True,
        )

        for position, symbol in enumerate(request.assets, start=1):
            alpaca = _read_cache(
                alpaca_collection,
                symbol=symbol,
                interval=request.timeframe,
                feed=request.alpaca_historical_feed,
                adjustment=request.alpaca_adjustment,
                start=start,
                end_exclusive=end_exclusive,
            )
            tiingo = _read_cache(
                tiingo_collection,
                symbol=symbol,
                interval=request.timeframe,
                feed="eod",
                adjustment=request.alpaca_adjustment,
                start=start,
                end_exclusive=end_exclusive,
            )
            raw[symbol] = {"alpaca": alpaca, "tiingo": tiingo}

            alpaca_ready, alpaca_reason = _coverage_ready(
                alpaca,
                start=start,
                tolerance_days=tolerance_days,
                minimum_rows=minimum_rows,
            )
            tiingo_ready, tiingo_reason = _coverage_ready(
                tiingo,
                start=start,
                tolerance_days=tolerance_days,
                minimum_rows=minimum_rows,
            )
            if alpaca_ready and tiingo_ready:
                eligible_assets.append(symbol)

            alpaca_first, alpaca_last = _first_and_last(alpaca)
            tiingo_first, tiingo_last = _first_and_last(tiingo)
            coverage_rows.append(
                {
                    "symbol": symbol,
                    "alpaca_rows": len(alpaca),
                    "tiingo_rows": len(tiingo),
                    "alpaca_first": alpaca_first,
                    "alpaca_last": alpaca_last,
                    "tiingo_first": tiingo_first,
                    "tiingo_last": tiingo_last,
                    "alpaca_ready": alpaca_ready,
                    "tiingo_ready": tiingo_ready,
                    "alpaca_reason": alpaca_reason,
                    "tiingo_reason": tiingo_reason,
                }
            )
            print(
                f"[data] {position}/{len(request.assets)} {symbol}: "
                f"alpaca={len(alpaca)} tiingo={len(tiingo)} "
                f"eligible={alpaca_ready and tiingo_ready}",
                flush=True,
            )

        if len(eligible_assets) < 2:
            raise RuntimeError(
                "Fewer than two assets have modelable history in both caches."
            )

        latest_candidates: list[pd.Timestamp] = []
        if requested_end is not None:
            latest_candidates.append(requested_end)
        for symbol in eligible_assets:
            latest_candidates.append(_utc(raw[symbol]["alpaca"].index.max()).normalize())
            latest_candidates.append(_utc(raw[symbol]["tiingo"].index.max()).normalize())
        common_end = min(latest_candidates)
        common_end_exclusive = common_end + pd.Timedelta(days=1)

        alpaca_frames: dict[str, pd.DataFrame] = {}
        tiingo_frames: dict[str, pd.DataFrame] = {}
        asset_summaries: list[dict[str, Any]] = []
        detail_frames: list[pd.DataFrame] = []
        missing_frames: list[pd.DataFrame] = []

        for symbol in eligible_assets:
            alpaca = raw[symbol]["alpaca"].loc[
                (raw[symbol]["alpaca"].index >= start)
                & (raw[symbol]["alpaca"].index < common_end_exclusive)
            ].copy()
            tiingo = raw[symbol]["tiingo"].loc[
                (raw[symbol]["tiingo"].index >= start)
                & (raw[symbol]["tiingo"].index < common_end_exclusive)
            ].copy()
            alpaca_frames[symbol] = alpaca
            tiingo_frames[symbol] = tiingo
            summary, detail, missing = _compare_asset(symbol, alpaca, tiingo)
            asset_summaries.append(summary)
            if not detail.empty:
                detail_frames.append(detail)
            if not missing.empty:
                missing_frames.append(missing)

        model_input_summaries: list[dict[str, Any]] = []
        alpaca_model_inputs: dict[str, pd.DataFrame] = {}
        tiingo_model_inputs: dict[str, pd.DataFrame] = {}
        for symbol in eligible_assets:
            input_summary, left_inputs, right_inputs = _compare_model_inputs(
                symbol,
                alpaca_frames[symbol],
                tiingo_frames[symbol],
                request,
            )
            model_input_summaries.extend(input_summary)
            alpaca_model_inputs[symbol] = left_inputs
            tiingo_model_inputs[symbol] = right_inputs

        coverage = pd.DataFrame(coverage_rows)
        coverage.to_csv(output_dir / "coverage.csv", index=False)

        per_asset = pd.DataFrame(asset_summaries)
        per_asset.to_csv(output_dir / "per_asset_equivalence.csv", index=False)

        differences = (
            pd.concat(detail_frames, ignore_index=True)
            if detail_frames
            else pd.DataFrame()
        )
        differences.to_csv(output_dir / "candle_differences.csv", index=False)

        missing_dates = (
            pd.concat(missing_frames, ignore_index=True)
            if missing_frames
            else pd.DataFrame()
        )
        missing_dates.to_csv(output_dir / "missing_dates.csv", index=False)

        model_input_equivalence = pd.DataFrame(model_input_summaries)
        model_input_equivalence.to_csv(
            output_dir / "model_input_equivalence.csv",
            index=False,
        )

        anchors = [symbol for symbol in request.calendar_anchor_assets if symbol in eligible_assets]
        if len(anchors) < 2:
            anchors = list(eligible_assets)
        references = [symbol for symbol in request.research_reference_assets if symbol in eligible_assets]
        if len(references) < 2:
            references = list(eligible_assets)
        reference_set = set(references)
        candidates = [
            symbol
            for symbol in request.research_candidate_assets
            if symbol in eligible_assets and symbol not in reference_set
        ]

        controlled_request = request.model_copy(
            update={
                "assets": list(eligible_assets),
                "analysis_end_date": common_end.date().isoformat(),
                "calendar_anchor_assets": anchors,
                "research_reference_assets": references,
                "research_candidate_assets": candidates,
                "expected_market_data_signature_sha256": None,
                "research_market_data_snapshot_id": None,
                "research_market_data_mode": "database_only",
            }
        )

        excluded = [
            row
            for row in coverage_rows
            if row["symbol"] not in set(eligible_assets)
        ]

        summary: dict[str, Any] = {
            "schema_version": 1,
            "source_job_id": job.get("id"),
            "source_job_api_version": job.get("api_version"),
            "configured_asset_count": len(request.assets),
            "common_modelable_asset_count": len(eligible_assets),
            "common_modelable_assets": eligible_assets,
            "excluded_assets": excluded,
            "minimum_rows_required": minimum_rows,
            "comparison_start": start.date().isoformat(),
            "comparison_end": common_end.date().isoformat(),
            "alpaca_feed": request.alpaca_historical_feed,
            "tiingo_feed": "eod",
            "adjustment": request.alpaca_adjustment,
            "candle_comparison": {
                "assets_with_any_non_exact_rows": int(
                    sum(
                        1
                        for row in asset_summaries
                        if float(row.get("exact_match_rate") or 0.0) < 1.0
                    )
                ),
                "total_common_rows": int(
                    sum(int(row.get("common_rows") or 0) for row in asset_summaries)
                ),
                "total_exact_match_rows": int(
                    sum(int(row.get("exact_match_rows") or 0) for row in asset_summaries)
                ),
                "total_material_difference_rows": int(
                    sum(
                        int(row.get("material_difference_rows") or 0)
                        for row in asset_summaries
                    )
                ),
                "alpaca_only_dates": int(
                    sum(int(row.get("alpaca_only_rows") or 0) for row in asset_summaries)
                ),
                "tiingo_only_dates": int(
                    sum(int(row.get("tiingo_only_rows") or 0) for row in asset_summaries)
                ),
            },
            "model_input_comparison": {
                "rows": int(len(model_input_summaries)),
                "feature_rows": int(
                    sum(1 for row in model_input_summaries if row.get("kind") == "feature")
                ),
                "target_rows": int(
                    sum(1 for row in model_input_summaries if row.get("kind") == "target")
                ),
            },
            "controlled_request": controlled_request.model_dump(mode="json"),
        }

        if not args.skip_model_replay:
            alpaca_result = _run_source("ALPACA", alpaca_frames, controlled_request)
            tiingo_result = _run_source("TIINGO", tiingo_frames, controlled_request)

            alpaca_result.predictions.to_csv(
                output_dir / "alpaca_predictions.csv",
                index=True,
            )
            tiingo_result.predictions.to_csv(
                output_dir / "tiingo_predictions.csv",
                index=True,
            )
            alpaca_result.trades.to_csv(output_dir / "alpaca_trades.csv", index=False)
            tiingo_result.trades.to_csv(output_dir / "tiingo_trades.csv", index=False)

            decision_summary, decision_rows = _compare_predictions(
                alpaca_result.predictions,
                tiingo_result.predictions,
            )
            decision_rows.to_csv(
                output_dir / "decision_divergences.csv",
                index=False,
            )

            first_input_snapshot = _first_divergence_input_snapshot(
                decision_summary.get("first_divergence"),
                alpaca_model_inputs,
                tiingo_model_inputs,
            )
            first_input_snapshot.to_csv(
                output_dir / "first_divergence_model_inputs.csv",
                index=False,
            )

            first_test_session = min(
                _normalize_session_index(alpaca_result.predictions).index.min(),
                _normalize_session_index(tiingo_result.predictions).index.min(),
            )
            pretest_summary, pretest_anomalies = _compare_pretest_inputs(
                alpaca_model_inputs,
                tiingo_model_inputs,
                first_test_session,
            )
            pretest_summary.to_csv(
                output_dir / "fold1_pretest_model_input_equivalence.csv",
                index=False,
            )
            pretest_anomalies.to_csv(
                output_dir / "fold1_pretest_model_input_anomalies.csv",
                index=False,
            )

            alpaca_capital = float(
                alpaca_result.metrics.get("strategy_ending_capital") or 0.0
            )
            tiingo_capital = float(
                tiingo_result.metrics.get("strategy_ending_capital") or 0.0
            )
            summary["controlled_replay"] = {
                "alpaca": {
                    "ending_capital": alpaca_capital,
                    "cagr": alpaca_result.metrics.get("strategy_cagr"),
                    "sharpe": alpaca_result.metrics.get("strategy_sharpe"),
                    "maximum_drawdown": alpaca_result.metrics.get(
                        "strategy_maximum_drawdown"
                    ),
                    "switches": alpaca_result.metrics.get("switches"),
                },
                "tiingo": {
                    "ending_capital": tiingo_capital,
                    "cagr": tiingo_result.metrics.get("strategy_cagr"),
                    "sharpe": tiingo_result.metrics.get("strategy_sharpe"),
                    "maximum_drawdown": tiingo_result.metrics.get(
                        "strategy_maximum_drawdown"
                    ),
                    "switches": tiingo_result.metrics.get("switches"),
                },
                "alpaca_minus_tiingo_ending_capital": alpaca_capital - tiingo_capital,
                "alpaca_to_tiingo_capital_ratio": (
                    alpaca_capital / tiingo_capital
                    if tiingo_capital > 0
                    else None
                ),
                "decision_divergence": decision_summary,
                "fold1_pretest_input_attribution": {
                    "first_test_session": pd.Timestamp(first_test_session).isoformat(),
                    "summary_rows": int(len(pretest_summary)),
                    "anomaly_rows": int(len(pretest_anomalies)),
                },
            }

        summary_path = output_dir / "summary.json"
        summary_path.write_text(
            json.dumps(_json_value(summary), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print("", flush=True)
        print("[done] Alpaca x Tiingo equivalence audit completed.", flush=True)
        print(f"[done] output={output_dir}", flush=True)
        print(f"[done] common_assets={len(eligible_assets)}", flush=True)
        print(f"[done] comparison_end={common_end.date().isoformat()}", flush=True)
        if excluded:
            print(
                "[done] excluded="
                + ", ".join(str(item["symbol"]) for item in excluded),
                flush=True,
            )
        print(f"[done] summary={summary_path}", flush=True)
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
