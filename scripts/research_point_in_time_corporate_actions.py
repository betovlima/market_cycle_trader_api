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

import market_cycle_trader_api.engine.capital_rotation as capital_rotation
import market_cycle_trader_api.engine.research_challengers as research_challengers
from market_cycle_trader_api.infrastructure.persistence.mongo_repository import (
    JOBS_COLLECTION,
    create_client,
    get_database,
)
from market_cycle_trader_api.schemas.requests import BacktestExecutionRequest, BacktestRequest


RAW_COLLECTION = "alpaca_market_bars_raw_20260919"
SPLIT_REFERENCE_COLLECTION = "alpaca_market_bars_split_20260919"
CA_COLLECTION = "alpaca_corporate_actions_20260919"
OHLCV = ("open", "high", "low", "close", "volume")
CA_FEATURES = [
    "ca_ex_dividend_yield_today",
    "ca_known_dividend_yield_next_5",
    "ca_known_dividend_yield_next_20",
    "ca_known_dividend_yield_next_60",
    "ca_dividend_yield_trailing_60",
]


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
    if quantity <= 0 or price <= 0:
        return {
            "commission_fee": 0.0,
            "sec_fee": 0.0,
            "taf_fee": 0.0,
            "cat_fee": 0.0,
            "total_fee": 0.0,
        }
    trade_value = quantity * price
    commission = _round_fee_to_cent(trade_value * config.commission_rate)
    cat = _round_fee_to_cent(quantity * config.cat_fee_per_share)
    sec = 0.0
    taf = 0.0
    if side.upper() == "SELL":
        sec = _round_fee_to_cent(trade_value * config.sec_fee_rate)
        taf = _round_fee_to_cent(
            min(quantity * config.taf_fee_per_share, config.taf_fee_cap)
        )
    return {
        "commission_fee": commission,
        "sec_fee": sec,
        "taf_fee": taf,
        "cat_fee": cat,
        "total_fee": commission + sec + taf + cat,
    }


def _apply_slippage(price: float, side: str, config: BacktestRequest) -> float:
    adjustment = config.slippage_bps / 10_000.0
    return price * (1 + adjustment if side.upper() == "BUY" else 1 - adjustment)


def _latest_job(db: Any, job_id: str | None) -> dict[str, Any]:
    if job_id:
        job = db[JOBS_COLLECTION].find_one({"id": str(job_id)})
    else:
        job = db[JOBS_COLLECTION].find_one(
            {"internal_job": {"$ne": True}, "status": "completed"},
            sort=[("finished_at", -1), ("created_at", -1)],
        )
    if job is None or not isinstance(job.get("request"), dict):
        raise RuntimeError("No completed Backtest job with immutable request snapshot was found.")
    return job


def _utc(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _read_bars(
    collection: Any,
    *,
    symbol: str,
    interval: str,
    feed: str,
    adjustment: str,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> pd.DataFrame:
    rows = list(
        collection.find(
            {
                "symbol": symbol,
                "interval": interval,
                "feed": feed,
                "adjustment": adjustment,
                "timestamp": {
                    "$gte": start.to_pydatetime(),
                    "$lt": end_exclusive.to_pydatetime(),
                },
            },
            {"_id": 0, "timestamp": 1, **{column: 1 for column in OHLCV}},
        ).sort("timestamp", 1)
    )
    if not rows:
        return pd.DataFrame(columns=list(OHLCV))
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    for column in OHLCV:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=list(OHLCV))
    return frame


def _corporate_actions(collection: Any, symbol: str) -> list[dict[str, Any]]:
    return list(
        collection.find(
            {
                "$or": [
                    {"symbol": symbol},
                    {"source_symbol": symbol},
                ]
            },
            {"_id": 0},
        ).sort([("process_date", 1), ("ex_date", 1)])
    )


def _split_normalize(raw: pd.DataFrame, actions: list[dict[str, Any]]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    result = raw.copy()
    session_dates = pd.DatetimeIndex(result.index).tz_convert("UTC").normalize()
    applied: list[dict[str, Any]] = []

    split_actions = [
        action
        for action in actions
        if action.get("action_type") in {"forward_split", "reverse_split"}
        and action.get("ex_date")
        and action.get("old_rate") is not None
        and action.get("new_rate") is not None
    ]
    split_actions.sort(key=lambda item: str(item.get("ex_date")))

    for action in split_actions:
        ex_date = _utc(action["ex_date"]).normalize()
        old_rate = float(action["old_rate"])
        new_rate = float(action["new_rate"])
        if not (old_rate > 0 and new_rate > 0):
            continue
        price_factor = old_rate / new_rate
        volume_factor = new_rate / old_rate
        mask = session_dates < ex_date
        if not mask.any():
            continue
        for column in ("open", "high", "low", "close"):
            result.loc[mask, column] = result.loc[mask, column].astype(float) * price_factor
        result.loc[mask, "volume"] = result.loc[mask, "volume"].astype(float) * volume_factor
        applied.append(
            {
                "action_type": action.get("action_type"),
                "ex_date": str(action.get("ex_date")),
                "process_date": str(action.get("process_date")),
                "old_rate": old_rate,
                "new_rate": new_rate,
                "price_factor": price_factor,
                "volume_factor": volume_factor,
            }
        )
    return result, applied


def _session_position(index: pd.DatetimeIndex, date_value: Any) -> int | None:
    if date_value is None:
        return None
    target = _utc(date_value).normalize()
    normalized = index.tz_convert("UTC").normalize()
    pos = int(normalized.searchsorted(target, side="left"))
    if pos >= len(normalized):
        return None
    return pos


def _attach_dividend_features(
    frame: pd.DataFrame,
    actions: list[dict[str, Any]],
) -> tuple[pd.DataFrame, int]:
    result = frame.copy()
    index = pd.DatetimeIndex(result.index)
    sessions = index.tz_convert("UTC").normalize()
    n = len(result)

    for feature in CA_FEATURES:
        result[feature] = 0.0

    dividends: list[dict[str, Any]] = []
    for action in actions:
        if action.get("action_type") != "cash_dividend":
            continue
        if action.get("rate") is None or not action.get("ex_date") or not action.get("process_date"):
            continue
        rate = float(action["rate"])
        if not np.isfinite(rate) or rate <= 0:
            continue
        ex_pos = _session_position(index, action["ex_date"])
        process_pos = _session_position(index, action["process_date"])
        if ex_pos is None or process_pos is None:
            continue
        dividends.append(
            {
                "rate": rate,
                "ex_pos": ex_pos,
                "process_pos": process_pos,
                "special": bool(action.get("special", False)),
            }
        )

    closes = pd.to_numeric(result["close"], errors="coerce").to_numpy(dtype=float)
    values = {feature: np.zeros(n, dtype=float) for feature in CA_FEATURES}

    for i in range(n):
        close = closes[i]
        if not np.isfinite(close) or close <= 0:
            continue

        today = 0.0
        trailing = 0.0
        next_5 = 0.0
        next_20 = 0.0
        next_60 = 0.0

        for dividend in dividends:
            # Point-in-time rule: an event is usable only from the session on
            # which Alpaca's process_date says it was known to this data source.
            if dividend["process_pos"] > i:
                continue
            ex_pos = int(dividend["ex_pos"])
            rate = float(dividend["rate"])

            if ex_pos == i:
                today += rate
            if i - 59 <= ex_pos <= i:
                trailing += rate
            if i < ex_pos <= i + 5:
                next_5 += rate
            if i < ex_pos <= i + 20:
                next_20 += rate
            if i < ex_pos <= i + 60:
                next_60 += rate

        values["ca_ex_dividend_yield_today"][i] = today / close
        values["ca_dividend_yield_trailing_60"][i] = trailing / close
        values["ca_known_dividend_yield_next_5"][i] = next_5 / close
        values["ca_known_dividend_yield_next_20"][i] = next_20 / close
        values["ca_known_dividend_yield_next_60"][i] = next_60 / close

    for feature, array in values.items():
        result[feature] = array

    return result, len(dividends)


def _split_reference_error(
    reconstructed: pd.DataFrame,
    reference: pd.DataFrame,
) -> dict[str, Any]:
    if reconstructed.empty or reference.empty:
        return {"available": False}
    left = reconstructed.copy()
    right = reference.copy()
    left.index = pd.DatetimeIndex(left.index).tz_convert("UTC").normalize()
    right.index = pd.DatetimeIndex(right.index).tz_convert("UTC").normalize()
    common = left.index.intersection(right.index)
    if common.empty:
        return {"available": False}
    a = pd.to_numeric(left.loc[common, "close"], errors="coerce")
    b = pd.to_numeric(right.loc[common, "close"], errors="coerce")
    valid = pd.DataFrame({"reconstructed": a, "reference": b}).dropna()
    rel = ((valid["reconstructed"] - valid["reference"]).abs() / valid["reference"].abs().replace(0, np.nan)).dropna()
    return {
        "available": True,
        "common_rows": int(len(valid)),
        "mean_close_relative_error": float(rel.mean()) if len(rel) else None,
        "p95_close_relative_error": float(rel.quantile(0.95)) if len(rel) else None,
        "max_close_relative_error": float(rel.max()) if len(rel) else None,
    }


def _metrics(result: Any) -> dict[str, Any]:
    return {
        "ending_capital": result.metrics.get("strategy_ending_capital"),
        "cagr": result.metrics.get("strategy_cagr"),
        "sharpe": result.metrics.get("strategy_sharpe"),
        "maximum_drawdown": result.metrics.get("strategy_maximum_drawdown"),
        "switches": result.metrics.get("switches"),
    }


def _run(label: str, frames: dict[str, pd.DataFrame], config: BacktestExecutionRequest) -> Any:
    print(f"[model] starting={label} assets={len(frames)}", flush=True)
    results = capital_rotation.run_rotation_models(
        frames,
        config,
        _calculate_reference_fees,
        _apply_slippage,
        progress_callback=lambda p, stage, completed: print(
            f"[model] {label} progress={p:.1f}% completed={completed} stage={stage}",
            flush=True,
        ),
    )
    if not results:
        raise RuntimeError(f"{label} returned no results.")
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
            "Controlled point-in-time corporate-action experiment: RAW Alpaca bars, "
            "locally reconstructed split continuity, and dividend features known by process_date."
        )
    )
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--raw-collection", default=RAW_COLLECTION)
    parser.add_argument("--split-reference-collection", default=SPLIT_REFERENCE_COLLECTION)
    parser.add_argument("--corporate-actions-collection", default=CA_COLLECTION)
    parser.add_argument(
        "--output-dir",
        default="output/point_in_time_corporate_actions",
    )
    args = parser.parse_args()

    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    client = create_client()
    original_features = list(capital_rotation.ROTATION_FEATURES)
    try:
        db = get_database(client)
        job = _latest_job(db, args.job_id)
        request = BacktestExecutionRequest.model_validate(job["request"])

        start = _utc(request.start_date).normalize()
        end = _utc(request.analysis_end_date or request.end_date).normalize()
        end_exclusive = end + pd.Timedelta(days=1)
        minimum_rows = (
            int(request.rotation_minimum_training_rows)
            + max(int(item) for item in request.rotation_target_horizons)
            + int(request.rotation_purge_days)
        )

        raw_collection = db[str(args.raw_collection)]
        split_reference_collection = db[str(args.split_reference_collection)]
        ca_collection = db[str(args.corporate_actions_collection)]

        base_frames: dict[str, pd.DataFrame] = {}
        feature_frames: dict[str, pd.DataFrame] = {}
        diagnostics: list[dict[str, Any]] = []

        for position, symbol in enumerate(request.assets, start=1):
            raw = _read_bars(
                raw_collection,
                symbol=symbol,
                interval=request.timeframe,
                feed=request.alpaca_historical_feed,
                adjustment="raw",
                start=start,
                end_exclusive=end_exclusive,
            )
            if raw.empty or len(raw) < minimum_rows:
                print(f"[data] {symbol} excluded raw_rows={len(raw)}", flush=True)
                continue

            actions = _corporate_actions(ca_collection, symbol)
            split_frame, applied_splits = _split_normalize(raw, actions)
            feature_frame, dividend_count = _attach_dividend_features(split_frame, actions)

            split_reference = _read_bars(
                split_reference_collection,
                symbol=symbol,
                interval=request.timeframe,
                feed=request.alpaca_historical_feed,
                adjustment="split",
                start=start,
                end_exclusive=end_exclusive,
            )
            split_error = _split_reference_error(split_frame, split_reference)

            base_frames[symbol] = split_frame
            feature_frames[symbol] = feature_frame
            diagnostics.append(
                {
                    "symbol": symbol,
                    "raw_rows": len(raw),
                    "corporate_actions": len(actions),
                    "cash_dividends": dividend_count,
                    "splits_applied": len(applied_splits),
                    **{f"split_validation_{key}": value for key, value in split_error.items()},
                }
            )
            print(
                f"[data] {position}/{len(request.assets)} {symbol} raw={len(raw)} "
                f"actions={len(actions)} dividends={dividend_count} splits={len(applied_splits)}",
                flush=True,
            )

        if len(base_frames) < 2:
            raise RuntimeError("Fewer than two assets are modelable in the RAW snapshot.")

        eligible = list(base_frames)
        anchors = [symbol for symbol in request.calendar_anchor_assets if symbol in base_frames]
        if len(anchors) < 2:
            anchors = eligible
        references = [symbol for symbol in request.research_reference_assets if symbol in base_frames]
        if len(references) < 2:
            references = eligible
        reference_set = set(references)
        candidates = [
            symbol
            for symbol in request.research_candidate_assets
            if symbol in base_frames and symbol not in reference_set
        ]

        controlled_request = request.model_copy(
            update={
                "assets": eligible,
                "calendar_anchor_assets": anchors,
                "research_reference_assets": references,
                "research_candidate_assets": candidates,
                "alpaca_adjustment": "split",
                "expected_market_data_signature_sha256": None,
                "research_market_data_snapshot_id": None,
                "research_market_data_mode": "database_only",
            }
        )

        # Control: exactly the existing feature family, but fed from our own
        # RAW -> split reconstruction.
        baseline = _run("RAW_SPLIT_PRICE_ONLY", base_frames, controlled_request)
        baseline.predictions.to_csv(output_dir / "baseline_predictions.csv", index=True)
        baseline.trades.to_csv(output_dir / "baseline_trades.csv", index=False)

        # Experiment: same prices, same target and same hyperparameters.
        # Only point-in-time dividend context is added to the LightGBM inputs.
        for feature in CA_FEATURES:
            if feature not in capital_rotation.ROTATION_FEATURES:
                capital_rotation.ROTATION_FEATURES.append(feature)
        research_challengers.ROTATION_FEATURES = capital_rotation.ROTATION_FEATURES

        experiment = _run(
            "RAW_SPLIT_PIT_DIVIDEND_FEATURES",
            feature_frames,
            controlled_request,
        )
        experiment.predictions.to_csv(output_dir / "experiment_predictions.csv", index=True)
        experiment.trades.to_csv(output_dir / "experiment_trades.csv", index=False)

        diagnostic_frame = pd.DataFrame(diagnostics)
        diagnostic_frame.to_csv(output_dir / "data_diagnostics.csv", index=False)

        baseline_capital = float(baseline.metrics.get("strategy_ending_capital") or 0.0)
        experiment_capital = float(experiment.metrics.get("strategy_ending_capital") or 0.0)

        summary = {
            "schema_version": 1,
            "experiment": "point-in-time-corporate-actions-v1",
            "source_job_id": job.get("id"),
            "raw_collection": str(args.raw_collection),
            "split_reference_collection": str(args.split_reference_collection),
            "corporate_actions_collection": str(args.corporate_actions_collection),
            "asset_count": len(eligible),
            "features_added": CA_FEATURES,
            "point_in_time_rule": "corporate action process_date <= decision session",
            "split_rule": "RAW prices normalized locally with forward/reverse split old_rate/new_rate",
            "target": "unchanged price-based forward risk-adjusted utility",
            "portfolio_dividend_cash_credit": False,
            "baseline": _metrics(baseline),
            "experiment_with_dividend_features": _metrics(experiment),
            "ending_capital_delta": experiment_capital - baseline_capital,
            "ending_capital_ratio": (
                experiment_capital / baseline_capital
                if baseline_capital > 0
                else None
            ),
            "limitations": [
                "This version tests dividend information as model context only.",
                "Cash dividends are not yet credited to portfolio cash.",
                "The training target remains price-based in this version.",
                "process_date is used as the best available point-in-time proxy from the REST snapshot.",
            ],
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str),
            encoding="utf-8",
        )

        print("")
        print("[done] point-in-time corporate-action experiment completed")
        print(f"[done] baseline_capital={baseline_capital:,.2f}")
        print(f"[done] experiment_capital={experiment_capital:,.2f}")
        print(f"[done] delta={experiment_capital - baseline_capital:,.2f}")
        print(f"[done] output={output_dir}")
        return 0
    finally:
        capital_rotation.ROTATION_FEATURES[:] = original_features
        research_challengers.ROTATION_FEATURES = capital_rotation.ROTATION_FEATURES
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
