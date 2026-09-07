from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from market_cycle_trader_api.engine.capital_rotation import (  # noqa: E402
    ROTATION_FEATURES,
    _annualized_sharpe,
    _build_walk_forward_folds,
    _cagr,
    _maximum_drawdown,
    build_rotation_frame,
)
from market_cycle_trader_api.engine.research_challengers import _lightgbm_fit_models  # noqa: E402
from market_cycle_trader_api.infrastructure.persistence.mongo_repository import MongoRepository  # noqa: E402
from market_cycle_trader_api.services.strategy_configuration import get_strategy_profile  # noqa: E402

SCRIPT_VERSION = "asset-timing-vs-buyhold-v1"
INITIAL_CAPITAL = 10_000.0


@dataclass(frozen=True)
class FoldMetrics:
    symbol: str
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    sessions: int
    timing_return: float
    buy_hold_return: float
    excess_return: float
    timing_ending_capital: float
    buy_hold_ending_capital: float
    timing_max_drawdown: float
    buy_hold_max_drawdown: float
    timing_sharpe: float | None
    buy_hold_sharpe: float | None
    market_exposure: float
    cash_days: int
    buy_count: int
    sell_count: int
    average_holding_days: float | None
    profitable_trade_rate: float | None
    average_trade_return: float | None
    median_trade_return: float | None
    return_while_in_market: float
    return_while_in_cash: float
    beat_buy_hold: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Revalidates every Strategy asset and every complete local-Mongo candidate by "
            "point-in-time LightGBM timing versus buy-and-hold. The full Strategy backtest is "
            "not used for qualification."
        )
    )
    parser.add_argument("--strategy-sequence", type=int, default=10)
    parser.add_argument("--snapshot-end", required=True)
    parser.add_argument("--mongo-uri", default=os.getenv("MONGO_URI") or "mongodb://localhost:27017")
    parser.add_argument("--database", default=os.getenv("MONGO_DB") or os.getenv("MONGODB_DB") or "market_cycle_trader")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--candidate-symbols", nargs="*", default=None)
    parser.add_argument("--output-root", default="research_output")
    parser.add_argument(
        "--qualification-rule",
        choices=["positive-median-and-majority", "all-folds-positive"],
        default="positive-median-and-majority",
    )
    return parser.parse_args()


def _is_local_mongo(uri: str) -> bool:
    raw = str(uri or "").lower()
    return any(token in raw for token in ("localhost", "127.0.0.1", "::1"))


def _sha256_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _strategy_config(profile: Any) -> Any:
    for name in ("config", "configuration", "strategy_config"):
        value = getattr(profile, name, None)
        if value is not None:
            return value
    if isinstance(profile, dict):
        for name in ("config", "configuration", "strategy_config"):
            if profile.get(name) is not None:
                return profile[name]
    raise RuntimeError("Could not resolve Strategy configuration snapshot.")


def _strategy_assets(profile: Any, config: Any) -> list[str]:
    candidates = []
    if isinstance(profile, dict):
        candidates.extend([profile.get("assets"), profile.get("selected_assets")])
    else:
        candidates.extend([getattr(profile, "assets", None), getattr(profile, "selected_assets", None)])
    candidates.extend([getattr(config, "assets", None)])
    if isinstance(config, dict):
        candidates.append(config.get("assets"))
    for value in candidates:
        if isinstance(value, (list, tuple)) and value:
            return sorted({str(item).strip().upper() for item in value if str(item).strip()})
    raise RuntimeError("Strategy has no selected assets.")


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _model_copy(config: Any, updates: dict[str, Any]) -> Any:
    if hasattr(config, "model_copy"):
        return config.model_copy(update=updates)
    if isinstance(config, dict):
        result = dict(config)
        result.update(updates)
        return result
    for key, value in updates.items():
        setattr(config, key, value)
    return config


def _market_collection(db: Any) -> Any:
    names = set(db.list_collection_names())
    preferred = [
        "alpaca_market_bars",
        "market_bars",
        "market_data",
        "bars",
    ]
    for name in preferred:
        if name in names:
            return db[name]
    raise RuntimeError("Could not locate local Mongo market-bar collection.")


def _distinct_symbols(collection: Any) -> list[str]:
    for field in ("symbol", "ticker", "asset"):
        try:
            values = collection.distinct(field)
        except Exception:
            continue
        symbols = sorted({str(value).strip().upper() for value in values if str(value).strip()})
        if symbols:
            return symbols
    return []


def _read_symbol_frame(collection: Any, symbol: str, start: str, end: str) -> pd.DataFrame:
    fields = {"_id": 0}
    query_variants = [
        {"symbol": symbol},
        {"ticker": symbol},
        {"asset": symbol},
    ]
    rows: list[dict[str, Any]] = []
    for query in query_variants:
        try:
            rows = list(collection.find(query, fields))
        except Exception:
            rows = []
        if rows:
            break
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    ts_col = next((name for name in ("timestamp", "time", "datetime", "date", "t") if name in frame.columns), None)
    if ts_col is None:
        return pd.DataFrame()
    frame[ts_col] = pd.to_datetime(frame[ts_col], utc=True, errors="coerce")
    frame = frame.dropna(subset=[ts_col]).set_index(ts_col).sort_index()
    rename = {}
    aliases = {
        "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume",
        "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume",
    }
    for source, target in aliases.items():
        if source in frame.columns and target not in frame.columns:
            rename[source] = target
    frame = frame.rename(columns=rename)
    required = ["open", "high", "low", "close", "volume"]
    if any(column not in frame.columns for column in required):
        return pd.DataFrame()
    frame = frame[required].apply(pd.to_numeric, errors="coerce").dropna()
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    frame = frame[(frame.index >= start_ts) & (frame.index <= end_ts)]
    return frame[~frame.index.duplicated(keep="last")]


def _required_sessions(strategy_frame: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(strategy_frame.index).normalize().unique().sort_values()


def _history_complete(frame: pd.DataFrame, required: pd.DatetimeIndex) -> tuple[bool, int]:
    if frame.empty:
        return False, len(required)
    actual = pd.DatetimeIndex(frame.index).normalize().unique()
    missing = required.difference(actual)
    return len(missing) == 0, len(missing)


def _fit_single_asset_model(frame: pd.DataFrame, train_dates: pd.DatetimeIndex, config: Any, seed: int) -> Any:
    rep_config = _model_copy(config, {"random_state": int(seed)})
    models = _lightgbm_fit_models(
        {"ASSET": frame},
        ["ASSET"],
        train_dates,
        rep_config,
        phase="asset_timing_research",
    )
    model = models.get("ASSET")
    if model is None:
        raise RuntimeError("LightGBM could not fit the asset timing model.")
    return model


def _decision_state(score: float, invested: bool, config: Any) -> bool:
    entry = float(_config_value(config, "opportunity_utility_entry_threshold", 0.0))
    exit_ = float(_config_value(config, "opportunity_utility_exit_threshold", entry))
    threshold = exit_ if invested else entry
    return bool(np.isfinite(score) and score >= threshold)


def _simulate_fold(
    raw_bars: pd.DataFrame,
    feature_frame: pd.DataFrame,
    model: Any,
    test_dates: pd.DatetimeIndex,
    config: Any,
    symbol: str,
    fold_number: int,
    train_dates: pd.DatetimeIndex,
) -> FoldMetrics:
    test_dates = pd.DatetimeIndex(test_dates).sort_values()
    if len(test_dates) < 2:
        raise RuntimeError(f"{symbol}: fold {fold_number} has insufficient test dates.")

    prices = raw_bars.reindex(test_dates)[["open", "close"]].dropna()
    test_dates = test_dates.intersection(prices.index)
    if len(test_dates) < 2:
        raise RuntimeError(f"{symbol}: fold {fold_number} has insufficient aligned prices.")

    fee = max(0.0, float(_config_value(config, "commission_rate", 0.0)))
    slip = max(0.0, float(_config_value(config, "slippage_bps", 0.0))) / 10000.0
    capital = INITIAL_CAPITAL
    shares = 0.0
    invested = False
    pending_target: bool | None = None
    equity_values: list[float] = []
    equity_dates: list[pd.Timestamp] = []
    buy_count = 0
    sell_count = 0
    cash_days = 0
    in_market_days = 0
    holding_days = 0
    holding_spans: list[int] = []
    trade_returns: list[float] = []
    entry_value: float | None = None
    asset_returns_in_market: list[float] = []
    asset_returns_in_cash: list[float] = []

    previous_close: float | None = None
    previous_invested = False
    for index, date in enumerate(test_dates):
        row = prices.loc[date]
        open_price = float(row["open"])
        close_price = float(row["close"])

        if pending_target is not None and pending_target != invested:
            if pending_target:
                execution = open_price * (1.0 + slip)
                capital_after_fee = capital * (1.0 - fee)
                shares = capital_after_fee / execution
                capital = 0.0
                invested = True
                buy_count += 1
                holding_days = 0
                entry_value = shares * execution
            else:
                execution = open_price * (1.0 - slip)
                gross = shares * execution
                capital = gross * (1.0 - fee)
                if entry_value and entry_value > 0:
                    trade_returns.append(capital / entry_value - 1.0)
                shares = 0.0
                invested = False
                sell_count += 1
                if holding_days > 0:
                    holding_spans.append(holding_days)
                holding_days = 0
                entry_value = None

        if invested:
            in_market_days += 1
            holding_days += 1
            equity = shares * close_price
        else:
            cash_days += 1
            equity = capital
        equity_values.append(float(equity))
        equity_dates.append(pd.Timestamp(date))

        if previous_close and previous_close > 0:
            daily_asset_return = close_price / previous_close - 1.0
            (asset_returns_in_market if previous_invested else asset_returns_in_cash).append(float(daily_asset_return))
        previous_close = close_price
        previous_invested = invested

        if index >= len(test_dates) - 1:
            continue
        if date not in feature_frame.index:
            pending_target = invested
            continue
        features = feature_frame.loc[[date], ROTATION_FEATURES]
        if features.isna().any(axis=None):
            pending_target = invested
            continue
        score = float(model.predict(features)[0])
        pending_target = _decision_state(score, invested, config)

    if invested:
        last_close = float(prices.loc[test_dates[-1], "close"])
        capital = shares * last_close * (1.0 - fee)
        if entry_value and entry_value > 0:
            trade_returns.append(capital / entry_value - 1.0)
        sell_count += 1
        if holding_days > 0:
            holding_spans.append(holding_days)
        equity_values[-1] = float(capital)

    equity_curve = pd.Series(equity_values, index=pd.DatetimeIndex(equity_dates), dtype=float)
    first_open = float(prices.loc[test_dates[0], "open"])
    last_close = float(prices.loc[test_dates[-1], "close"])
    bh_shares = INITIAL_CAPITAL * (1.0 - fee) / (first_open * (1.0 + slip))
    buy_hold_ending = bh_shares * last_close * (1.0 - slip) * (1.0 - fee)
    buy_hold_curve = INITIAL_CAPITAL * prices.loc[test_dates, "close"].astype(float) / float(prices.loc[test_dates[0], "close"])
    buy_hold_curve.iloc[-1] = buy_hold_ending

    timing_return = float(capital / INITIAL_CAPITAL - 1.0)
    buy_hold_return = float(buy_hold_ending / INITIAL_CAPITAL - 1.0)
    timing_sharpe = _safe_float(_annualized_sharpe(equity_curve))
    buy_hold_sharpe = _safe_float(_annualized_sharpe(buy_hold_curve))

    return FoldMetrics(
        symbol=symbol,
        fold=int(fold_number),
        train_start=pd.Timestamp(train_dates[0]).date().isoformat(),
        train_end=pd.Timestamp(train_dates[-1]).date().isoformat(),
        test_start=pd.Timestamp(test_dates[0]).date().isoformat(),
        test_end=pd.Timestamp(test_dates[-1]).date().isoformat(),
        sessions=int(len(test_dates)),
        timing_return=timing_return,
        buy_hold_return=buy_hold_return,
        excess_return=float(timing_return - buy_hold_return),
        timing_ending_capital=float(capital),
        buy_hold_ending_capital=float(buy_hold_ending),
        timing_max_drawdown=float(_maximum_drawdown(equity_curve)),
        buy_hold_max_drawdown=float(_maximum_drawdown(buy_hold_curve)),
        timing_sharpe=timing_sharpe,
        buy_hold_sharpe=buy_hold_sharpe,
        market_exposure=float(in_market_days / max(1, len(test_dates))),
        cash_days=int(cash_days),
        buy_count=int(buy_count),
        sell_count=int(sell_count),
        average_holding_days=float(np.mean(holding_spans)) if holding_spans else None,
        profitable_trade_rate=float(np.mean(np.asarray(trade_returns) > 0.0)) if trade_returns else None,
        average_trade_return=float(np.mean(trade_returns)) if trade_returns else None,
        median_trade_return=float(np.median(trade_returns)) if trade_returns else None,
        return_while_in_market=float(np.prod(1.0 + np.asarray(asset_returns_in_market)) - 1.0) if asset_returns_in_market else 0.0,
        return_while_in_cash=float(np.prod(1.0 + np.asarray(asset_returns_in_cash)) - 1.0) if asset_returns_in_cash else 0.0,
        beat_buy_hold=bool(timing_return > buy_hold_return),
    )


def _aggregate(symbol: str, source: str, folds: list[FoldMetrics]) -> dict[str, Any]:
    excess = np.asarray([row.excess_return for row in folds], dtype=float)
    timing_returns = np.asarray([row.timing_return for row in folds], dtype=float)
    bh_returns = np.asarray([row.buy_hold_return for row in folds], dtype=float)
    beat_rate = float(np.mean([row.beat_buy_hold for row in folds]))
    timing_dd = np.asarray([row.timing_max_drawdown for row in folds], dtype=float)
    bh_dd = np.asarray([row.buy_hold_max_drawdown for row in folds], dtype=float)
    return {
        "symbol": symbol,
        "source": source,
        "fold_count": int(len(folds)),
        "beat_buy_hold_fold_count": int(sum(row.beat_buy_hold for row in folds)),
        "beat_buy_hold_fold_rate": beat_rate,
        "median_fold_excess_return": float(np.median(excess)),
        "mean_fold_excess_return": float(np.mean(excess)),
        "worst_fold_excess_return": float(np.min(excess)),
        "best_fold_excess_return": float(np.max(excess)),
        "fold_excess_std": float(np.std(excess, ddof=0)),
        "median_timing_return": float(np.median(timing_returns)),
        "median_buy_hold_return": float(np.median(bh_returns)),
        "mean_drawdown_improvement": float(np.mean(timing_dd - bh_dd)),
        "mean_market_exposure": float(np.mean([row.market_exposure for row in folds])),
        "total_buy_count": int(sum(row.buy_count for row in folds)),
        "total_sell_count": int(sum(row.sell_count for row in folds)),
        "mean_return_while_in_market": float(np.mean([row.return_while_in_market for row in folds])),
        "mean_return_while_in_cash": float(np.mean([row.return_while_in_cash for row in folds])),
    }


def _qualifies(row: dict[str, Any], rule: str) -> bool:
    if rule == "all-folds-positive":
        return bool(row["beat_buy_hold_fold_rate"] == 1.0 and row["worst_fold_excess_return"] > 0.0)
    return bool(
        row["median_fold_excess_return"] > 0.0
        and row["beat_buy_hold_fold_rate"] >= 0.75
        and row["mean_return_while_in_market"] > row["mean_return_while_in_cash"]
    )


def main() -> int:
    args = parse_args()
    if not _is_local_mongo(args.mongo_uri):
        raise RuntimeError("This research script only accepts a local MongoDB connection.")

    repository = MongoRepository(args.mongo_uri, args.database)
    db = repository.db
    profile = get_strategy_profile(db, sequence=int(args.strategy_sequence))
    if profile is None:
        raise RuntimeError(f"Strategy #{args.strategy_sequence} was not found.")
    config = _strategy_config(profile)
    baseline_assets = _strategy_assets(profile, config)
    strategy_start = str(_config_value(config, "start_date", "2016-01-01"))[:10]
    snapshot_end = str(args.snapshot_end)[:10]

    collection = _market_collection(db)
    cached_symbols = _distinct_symbols(collection)
    if args.candidate_symbols:
        requested = sorted({str(item).strip().upper() for item in args.candidate_symbols if str(item).strip()})
        universe = sorted(set(baseline_assets) | set(requested))
    else:
        universe = sorted(set(baseline_assets) | set(cached_symbols))

    anchor_symbol = next((symbol for symbol in baseline_assets if symbol in cached_symbols), None)
    if anchor_symbol is None:
        raise RuntimeError("Could not locate any Strategy asset in local Mongo market cache.")
    anchor = _read_symbol_frame(collection, anchor_symbol, strategy_start, snapshot_end)
    if anchor.empty:
        raise RuntimeError(f"Could not load anchor history for {anchor_symbol}.")
    required = _required_sessions(anchor)

    output = Path(args.output_root) / f"asset_timing_strategy_{args.strategy_sequence}_{snapshot_end}"
    output.mkdir(parents=True, exist_ok=True)

    histories: dict[str, pd.DataFrame] = {}
    integrity_rows: list[dict[str, Any]] = []
    for position, symbol in enumerate(universe, start=1):
        frame = _read_symbol_frame(collection, symbol, strategy_start, snapshot_end)
        complete, missing = _history_complete(frame, required)
        source = "strategy_existing" if symbol in baseline_assets else "candidate"
        integrity_rows.append({
            "symbol": symbol,
            "source": source,
            "rows": int(len(frame)),
            "history_complete": bool(complete),
            "missing_sessions": int(missing),
            "actual_start": frame.index.min().date().isoformat() if not frame.empty else None,
            "actual_end": frame.index.max().date().isoformat() if not frame.empty else None,
        })
        if complete:
            histories[symbol] = frame
        print(f"History {position}/{len(universe)} {symbol}: {'OK' if complete else 'INCOMPLETE'}")

    pd.DataFrame(integrity_rows).to_csv(output / "asset_history_integrity.csv", index=False)
    required_existing = [symbol for symbol in baseline_assets if symbol not in histories]
    if required_existing:
        raise RuntimeError("Existing Strategy assets failed full-history validation: " + ", ".join(required_existing))

    fold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    research_config = _model_copy(
        config,
        {
            "assets": baseline_assets,
            "calendar_anchor_assets": baseline_assets[:2],
            "research_market_data_mode": "database_only",
        },
    )

    for position, symbol in enumerate(sorted(histories), start=1):
        raw = histories[symbol]
        rotation = build_rotation_frame(raw, research_config)
        common_dates = pd.DatetimeIndex(rotation.index).sort_values()
        folds = _build_walk_forward_folds(common_dates, research_config)
        if not folds:
            print(f"Timing {position}/{len(histories)} {symbol}: no folds")
            continue
        symbol_folds: list[FoldMetrics] = []
        for fold_position, fold in enumerate(folds, start=1):
            train_dates = common_dates[: int(fold["train_end_index"])]
            test_dates = pd.DatetimeIndex(fold["decision_dates"])[:-1]
            model = _fit_single_asset_model(rotation, train_dates, research_config, seed=int(_config_value(config, "random_state", 42)) + fold_position)
            metrics = _simulate_fold(raw, rotation, model, test_dates, research_config, symbol, fold_position, train_dates)
            symbol_folds.append(metrics)
            fold_rows.append(asdict(metrics))
        source = "strategy_existing" if symbol in baseline_assets else "candidate"
        aggregate = _aggregate(symbol, source, symbol_folds)
        aggregate["qualified"] = _qualifies(aggregate, args.qualification_rule)
        summary_rows.append(aggregate)
        pd.DataFrame(fold_rows).to_csv(output / "asset_timing_folds.csv", index=False)
        pd.DataFrame(summary_rows).to_csv(output / "asset_timing_summary.csv", index=False)
        print(
            f"Timing {position}/{len(histories)} {symbol}: "
            f"beat_rate={aggregate['beat_buy_hold_fold_rate']:.2%} "
            f"median_excess={aggregate['median_fold_excess_return']:.2%} "
            f"qualified={aggregate['qualified']}"
        )

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        raise RuntimeError("No asset timing results were produced.")
    summary = summary.sort_values(
        ["qualified", "median_fold_excess_return", "beat_buy_hold_fold_rate", "mean_drawdown_improvement", "symbol"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    summary["timing_rank"] = np.arange(1, len(summary) + 1)
    summary.to_csv(output / "asset_timing_summary.csv", index=False)

    qualified = summary.loc[summary["qualified"], "symbol"].astype(str).tolist()
    retained_existing = [symbol for symbol in baseline_assets if symbol in qualified]
    removed_existing = [symbol for symbol in baseline_assets if symbol not in qualified]
    added_candidates = [symbol for symbol in qualified if symbol not in baseline_assets]

    frozen = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "strategy_sequence": int(args.strategy_sequence),
        "strategy_start": strategy_start,
        "snapshot_end": snapshot_end,
        "qualification_rule": args.qualification_rule,
        "full_strategy_backtest_used_for_selection": False,
        "existing_strategy_assets_revalidated": True,
        "original_assets": baseline_assets,
        "universe_evaluated": summary["symbol"].astype(str).tolist(),
        "qualified_assets": qualified,
        "retained_existing_assets": retained_existing,
        "removed_existing_assets": removed_existing,
        "added_candidate_assets": added_candidates,
    }
    frozen["qualified_assets_sha256"] = _sha256_json(qualified)
    frozen["decision_snapshot_sha256"] = _sha256_json(frozen)
    (output / "timing_validation_snapshot_frozen.json").write_text(json.dumps(frozen, indent=2, default=str), encoding="utf-8")

    manifest = {
        "script_version": SCRIPT_VERSION,
        "mongo_local_only": True,
        "alpaca_network_used": False,
        "mongo_writes": False,
        "full_strategy_backtest_run": False,
        "selection_basis": "point-in-time LightGBM timing versus same-asset buy-and-hold",
        "same_model_family_as_strategy": str(_config_value(config, "research_model_family", "")),
        "strategy_mode": str(_config_value(config, "strategy_mode", "")),
        "rotation_features": list(ROTATION_FEATURES),
        "qualification_rule": args.qualification_rule,
        "result_counts": {
            "evaluated": int(len(summary)),
            "qualified": int(len(qualified)),
            "retained_existing": int(len(retained_existing)),
            "removed_existing": int(len(removed_existing)),
            "added_candidates": int(len(added_candidates)),
        },
        "decision_snapshot_sha256": frozen["decision_snapshot_sha256"],
    }
    (output / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print("\nFrozen qualified universe")
    print(f"Original Strategy assets : {len(baseline_assets)}")
    print(f"Retained existing assets : {len(retained_existing)}")
    print(f"Removed existing assets  : {len(removed_existing)}")
    print(f"Added candidates         : {len(added_candidates)}")
    print(f"Final qualified universe : {len(qualified)}")
    print(f"Snapshot: {output / 'timing_validation_snapshot_frozen.json'}")
    print("\nNo full Strategy backtest has been executed. Run the companion final-backtest script only after reviewing this frozen snapshot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
