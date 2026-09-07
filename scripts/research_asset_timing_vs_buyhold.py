from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from pymongo import MongoClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import research_asset_signature_leave_one_out as common  # noqa: E402

from market_cycle_trader_api.engine.absolute_utility_cash_gate import (  # noqa: E402
    absolute_utility_cash_gate_enabled,
)
from market_cycle_trader_api.engine.capital_rotation import (  # noqa: E402
    ROTATION_FEATURES,
    _annualized_sharpe,
    _build_walk_forward_folds,
    _maximum_drawdown,
    _simple_policy_growth,
    _utility_policy,
    build_rotation_frame,
)
from market_cycle_trader_api.engine.research_challengers import _lightgbm_fit_models  # noqa: E402
from market_cycle_trader_api.schemas.requests import BacktestRequest  # noqa: E402

SCRIPT_VERSION = "asset-timing-vs-buyhold-v1.1"
EXPERIMENT_NAME = "point_in_time_asset_timing_vs_same_asset_buy_hold"
INITIAL_CAPITAL = 10_000.0
DEFAULT_WORKERS = 4
SUPPORTED_INDIVIDUAL_TIMING_MODES = {
    "COMPOUND_ROTATION_SWING_XGBOOST",
    "COMPOUND_ROTATION_SWING_ABSOLUTE_UTILITY_CASH_GATE",
}


@dataclass(frozen=True)
class FoldMetrics:
    symbol: str
    fold: int
    train_start: str
    train_end: str
    calibration_start: str
    calibration_end: str
    test_start: str
    test_end: str
    sessions: int
    calibrated_switch_margin: float
    effective_switch_margin: float
    calibration_growth_score: float
    timing_return: float
    buy_hold_return: float
    excess_return: float
    timing_ending_capital: float
    buy_hold_ending_capital: float
    timing_max_drawdown: float
    buy_hold_max_drawdown: float
    drawdown_improvement: float
    timing_sharpe: float | None
    buy_hold_sharpe: float | None
    sharpe_improvement: float | None
    market_exposure: float
    cash_days: int
    market_days: int
    buy_count: int
    sell_count: int
    average_holding_days: float | None
    profitable_trade_rate: float | None
    average_trade_return: float | None
    median_trade_return: float | None
    return_while_in_market: float
    return_while_in_cash: float
    upside_capture: float | None
    downside_avoidance: float | None
    beat_buy_hold: bool


def _log(message: str) -> None:
    stamp = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{stamp}] {message}", flush=True)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_write_text(path, json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _load_frames_allow_incomplete(
    collection: Any,
    assets: list[str],
    identity: dict[str, str],
    start_date: pd.Timestamp,
    snapshot_end: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    start = start_date.tz_localize("UTC").to_pydatetime()
    end = (snapshot_end + pd.Timedelta(days=1)).tz_localize("UTC").to_pydatetime()
    query = {
        "symbol": {"$in": assets},
        **identity,
        "timestamp": {"$gte": start, "$lt": end},
    }
    projection = {
        "_id": 0,
        "symbol": 1,
        "timestamp": 1,
        "open": 1,
        "high": 1,
        "low": 1,
        "close": 1,
        "volume": 1,
        "vwap": 1,
        "trade_count": 1,
    }
    rows = list(collection.find(query, projection).sort([("symbol", 1), ("timestamp", 1)]))
    if not rows:
        raise RuntimeError("Local MongoDB returned no market bars for the research universe.")
    raw = pd.DataFrame(rows)
    raw["symbol"] = raw["symbol"].astype(str).str.upper()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    frames: dict[str, pd.DataFrame] = {}
    for symbol, group in raw.groupby("symbol", sort=False):
        frame = group.drop(columns=["symbol"]).set_index("timestamp").sort_index()
        frames[str(symbol)] = frame[~frame.index.duplicated(keep="last")]
    return frames


def _history_diagnostics(
    frames: dict[str, pd.DataFrame],
    assets: list[str],
    baseline_assets: set[str],
    expected: pd.DatetimeIndex,
) -> tuple[list[dict[str, Any]], list[str]]:
    diagnostics: list[dict[str, Any]] = []
    complete: list[str] = []
    for symbol in assets:
        frame = frames.get(symbol, pd.DataFrame())
        observed = (
            pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True).normalize().tz_localize(None)).unique()
            if not frame.empty
            else pd.DatetimeIndex([])
        )
        missing = expected.difference(observed)
        is_complete = bool(len(observed) and len(missing) == 0)
        diagnostics.append(
            {
                "symbol": symbol,
                "source": "strategy_existing" if symbol in baseline_assets else "candidate",
                "observed_rows": int(len(frame)),
                "expected_sessions": int(len(expected)),
                "missing_sessions": int(len(missing)),
                "actual_start": observed.min().date().isoformat() if len(observed) else None,
                "actual_end": observed.max().date().isoformat() if len(observed) else None,
                "history_complete": is_complete,
            }
        )
        if is_complete:
            complete.append(symbol)
    return diagnostics, complete


def _fit_models(
    symbol: str,
    frame: pd.DataFrame,
    train_dates: pd.DatetimeIndex,
    config: BacktestRequest,
    *,
    phase: str,
) -> dict[str, Any]:
    return _lightgbm_fit_models(
        {symbol: frame},
        [symbol],
        train_dates,
        config,
        phase=phase,
    )


def _calibrate_policy(
    symbol: str,
    frame: pd.DataFrame,
    common_dates: pd.DatetimeIndex,
    fold: dict[str, Any],
    config: BacktestRequest,
    seed: int,
) -> tuple[Callable[[pd.Timestamp, int, int], tuple[int, float]], float, float, float]:
    rep_config = config.model_copy(update={"random_state": int(seed)})
    train_dates = common_dates[: int(fold["train_end_index"])]
    calibration_dates = common_dates[
        int(fold["calibration_start_index"]): int(fold["calibration_end_index"])
    ]
    final_fit_dates = common_dates[: int(fold["final_fit_end_index"])]

    calibration_models = _fit_models(
        symbol,
        frame,
        train_dates,
        rep_config,
        phase=f"asset_timing_{symbol}_fold_{fold['fold_id']}_calibration",
    )
    candidate_margins = tuple(float(value) for value in rep_config.rotation_switch_margin_candidates)
    if not candidate_margins:
        raise RuntimeError(f"{symbol}: rotation_switch_margin_candidates is empty.")
    best_candidate = candidate_margins[0]
    best_score = float("-inf")
    margin_config = (
        rep_config.model_copy(update={"strategy_mode": "COMPOUND_ROTATION_SWING_XGBOOST"})
        if absolute_utility_cash_gate_enabled(rep_config)
        else rep_config
    )
    for candidate in candidate_margins:
        calibration_policy = _utility_policy(
            calibration_models,
            {symbol: frame},
            [symbol],
            margin_config,
            candidate,
        )
        score = _simple_policy_growth(
            calibration_policy,
            {symbol: frame},
            [symbol],
            calibration_dates,
            rep_config,
        )
        if float(score) > best_score:
            best_score = float(score)
            best_candidate = float(candidate)

    final_models = _fit_models(
        symbol,
        frame,
        final_fit_dates,
        rep_config,
        phase=f"asset_timing_{symbol}_fold_{fold['fold_id']}_final",
    )
    effective_margin = max(float(rep_config.rotation_switch_margin), float(best_candidate))
    policy = _utility_policy(
        final_models,
        {symbol: frame},
        [symbol],
        rep_config,
        effective_margin,
        cash_gate_base_state=(
            {"position": 0, "holding_days": 0, "pending_sample": None}
            if absolute_utility_cash_gate_enabled(rep_config)
            else None
        ),
        fold_id=int(fold["fold_id"]),
        calibrated_switch_margin=float(best_candidate),
    )
    return policy, float(best_candidate), float(effective_margin), float(best_score)


def _simulate_fold(
    symbol: str,
    raw_bars: pd.DataFrame,
    frame: pd.DataFrame,
    fold: dict[str, Any],
    policy: Callable[[pd.Timestamp, int, int], tuple[int, float]],
    config: BacktestRequest,
    calibrated_margin: float,
    effective_margin: float,
    calibration_score: float,
) -> FoldMetrics:
    decision_dates = pd.DatetimeIndex(fold["decision_dates"]).sort_values()
    if len(decision_dates) < 2:
        raise RuntimeError(f"{symbol}: fold {fold['fold_id']} has insufficient decision dates.")
    test_dates = decision_dates[1:]
    raw = raw_bars.reindex(test_dates)[["open", "close"]].dropna()
    test_dates = test_dates.intersection(raw.index)
    if len(test_dates) < 2:
        raise RuntimeError(f"{symbol}: fold {fold['fold_id']} has insufficient aligned test prices.")

    commission = max(0.0, float(config.commission_rate))
    slippage = max(0.0, float(config.slippage_bps)) / 10_000.0
    capital = INITIAL_CAPITAL
    shares = 0.0
    current_position = 0
    holding_days = 0
    entry_cost_basis: float | None = None
    holding_spans: list[int] = []
    trade_returns: list[float] = []
    buys = 0
    sells = 0
    cash_days = 0
    market_days = 0
    equity_values: list[float] = []
    equity_dates: list[pd.Timestamp] = []
    session_market_returns: list[float] = []
    session_cash_returns: list[float] = []
    all_positive_session_returns: list[float] = []
    captured_positive_returns: list[float] = []
    all_negative_session_returns: list[float] = []
    avoided_negative_returns: list[float] = []

    for index, decision_date in enumerate(decision_dates[:-1]):
        execution_date = pd.Timestamp(decision_dates[index + 1])
        if execution_date not in raw.index:
            continue
        target_position, _ = policy(pd.Timestamp(decision_date), int(current_position), int(holding_days))
        row = raw.loc[execution_date]
        open_price = float(row["open"])
        close_price = float(row["close"])

        if int(target_position) != int(current_position):
            if current_position > 0:
                execution_price = open_price * (1.0 - slippage)
                proceeds = shares * execution_price * (1.0 - commission)
                if entry_cost_basis and entry_cost_basis > 0:
                    trade_returns.append(float(proceeds / entry_cost_basis - 1.0))
                capital = float(proceeds)
                shares = 0.0
                sells += 1
                if holding_days > 0:
                    holding_spans.append(int(holding_days))
                holding_days = 0
                entry_cost_basis = None
            if int(target_position) > 0:
                execution_price = open_price * (1.0 + slippage)
                spendable = capital * (1.0 - commission)
                shares = float(spendable / execution_price)
                entry_cost_basis = float(capital)
                capital = 0.0
                buys += 1
                current_position = 1
                holding_days = 0
            else:
                current_position = 0
        else:
            current_position = int(target_position)

        session_return = float(close_price / open_price - 1.0)
        if session_return > 0:
            all_positive_session_returns.append(session_return)
        elif session_return < 0:
            all_negative_session_returns.append(session_return)

        if current_position > 0:
            market_days += 1
            holding_days += 1
            equity = float(shares * close_price)
            session_market_returns.append(session_return)
            if session_return > 0:
                captured_positive_returns.append(session_return)
        else:
            cash_days += 1
            equity = float(capital)
            session_cash_returns.append(session_return)
            if session_return < 0:
                avoided_negative_returns.append(session_return)

        equity_dates.append(execution_date)
        equity_values.append(equity)

    if current_position > 0 and len(test_dates):
        final_close = float(raw.loc[test_dates[-1], "close"])
        final_proceeds = shares * final_close * (1.0 - slippage) * (1.0 - commission)
        if entry_cost_basis and entry_cost_basis > 0:
            trade_returns.append(float(final_proceeds / entry_cost_basis - 1.0))
        capital = float(final_proceeds)
        shares = 0.0
        sells += 1
        if holding_days > 0:
            holding_spans.append(int(holding_days))
        if equity_values:
            equity_values[-1] = capital

    if not equity_values:
        raise RuntimeError(f"{symbol}: fold {fold['fold_id']} produced no OOS equity points.")
    equity_curve = pd.Series(equity_values, index=pd.DatetimeIndex(equity_dates), dtype=float)

    first_open = float(raw.loc[test_dates[0], "open"])
    last_close = float(raw.loc[test_dates[-1], "close"])
    buy_hold_shares = INITIAL_CAPITAL * (1.0 - commission) / (first_open * (1.0 + slippage))
    buy_hold_ending = buy_hold_shares * last_close * (1.0 - slippage) * (1.0 - commission)
    buy_hold_curve = INITIAL_CAPITAL * raw.loc[test_dates, "close"].astype(float) / float(raw.loc[test_dates[0], "close"])
    buy_hold_curve.iloc[-1] = float(buy_hold_ending)

    timing_return = float(capital / INITIAL_CAPITAL - 1.0)
    buy_hold_return = float(buy_hold_ending / INITIAL_CAPITAL - 1.0)
    timing_dd = float(_maximum_drawdown(equity_curve))
    buy_hold_dd = float(_maximum_drawdown(buy_hold_curve))
    timing_sharpe = _safe_float(_annualized_sharpe(equity_curve))
    buy_hold_sharpe = _safe_float(_annualized_sharpe(buy_hold_curve))

    positive_total = float(sum(all_positive_session_returns))
    negative_total = float(sum(abs(value) for value in all_negative_session_returns))
    upside_capture = (
        float(sum(captured_positive_returns) / positive_total)
        if positive_total > 1e-12
        else None
    )
    downside_avoidance = (
        float(sum(abs(value) for value in avoided_negative_returns) / negative_total)
        if negative_total > 1e-12
        else None
    )

    def compound(values: list[float]) -> float:
        return float(np.prod(1.0 + np.asarray(values, dtype=float)) - 1.0) if values else 0.0

    return FoldMetrics(
        symbol=symbol,
        fold=int(fold["fold_id"]),
        train_start=pd.Timestamp(fold["train_start"]).date().isoformat(),
        train_end=pd.Timestamp(fold["train_end"]).date().isoformat(),
        calibration_start=pd.Timestamp(fold["calibration_start"]).date().isoformat(),
        calibration_end=pd.Timestamp(fold["calibration_end"]).date().isoformat(),
        test_start=pd.Timestamp(fold["test_start"]).date().isoformat(),
        test_end=pd.Timestamp(fold["test_end"]).date().isoformat(),
        sessions=int(len(equity_curve)),
        calibrated_switch_margin=float(calibrated_margin),
        effective_switch_margin=float(effective_margin),
        calibration_growth_score=float(calibration_score),
        timing_return=timing_return,
        buy_hold_return=buy_hold_return,
        excess_return=float(timing_return - buy_hold_return),
        timing_ending_capital=float(capital),
        buy_hold_ending_capital=float(buy_hold_ending),
        timing_max_drawdown=timing_dd,
        buy_hold_max_drawdown=buy_hold_dd,
        drawdown_improvement=float(timing_dd - buy_hold_dd),
        timing_sharpe=timing_sharpe,
        buy_hold_sharpe=buy_hold_sharpe,
        sharpe_improvement=(
            float(timing_sharpe - buy_hold_sharpe)
            if timing_sharpe is not None and buy_hold_sharpe is not None
            else None
        ),
        market_exposure=float(market_days / max(1, market_days + cash_days)),
        cash_days=int(cash_days),
        market_days=int(market_days),
        buy_count=int(buys),
        sell_count=int(sells),
        average_holding_days=float(np.mean(holding_spans)) if holding_spans else None,
        profitable_trade_rate=float(np.mean(np.asarray(trade_returns) > 0.0)) if trade_returns else None,
        average_trade_return=float(np.mean(trade_returns)) if trade_returns else None,
        median_trade_return=float(np.median(trade_returns)) if trade_returns else None,
        return_while_in_market=compound(session_market_returns),
        return_while_in_cash=compound(session_cash_returns),
        upside_capture=upside_capture,
        downside_avoidance=downside_avoidance,
        beat_buy_hold=bool(timing_return > buy_hold_return),
    )


def _aggregate(symbol: str, source: str, folds: list[FoldMetrics]) -> dict[str, Any]:
    excess = np.asarray([row.excess_return for row in folds], dtype=float)
    timing = np.asarray([row.timing_return for row in folds], dtype=float)
    buy_hold = np.asarray([row.buy_hold_return for row in folds], dtype=float)
    beat_rate = float(np.mean([row.beat_buy_hold for row in folds]))
    compound_timing = float(np.prod(1.0 + timing) - 1.0)
    compound_buy_hold = float(np.prod(1.0 + buy_hold) - 1.0)
    timing_sharpe = [row.timing_sharpe for row in folds if row.timing_sharpe is not None]
    bh_sharpe = [row.buy_hold_sharpe for row in folds if row.buy_hold_sharpe is not None]
    upside = [row.upside_capture for row in folds if row.upside_capture is not None]
    downside = [row.downside_avoidance for row in folds if row.downside_avoidance is not None]
    trade_returns = [row.average_trade_return for row in folds if row.average_trade_return is not None]
    profitable = [row.profitable_trade_rate for row in folds if row.profitable_trade_rate is not None]
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
        "compound_oos_timing_return": compound_timing,
        "compound_oos_buy_hold_return": compound_buy_hold,
        "compound_oos_excess_return": float(compound_timing - compound_buy_hold),
        "median_timing_return": float(np.median(timing)),
        "median_buy_hold_return": float(np.median(buy_hold)),
        "mean_drawdown_improvement": float(np.mean([row.drawdown_improvement for row in folds])),
        "median_timing_sharpe": float(np.median(timing_sharpe)) if timing_sharpe else None,
        "median_buy_hold_sharpe": float(np.median(bh_sharpe)) if bh_sharpe else None,
        "mean_market_exposure": float(np.mean([row.market_exposure for row in folds])),
        "mean_upside_capture": float(np.mean(upside)) if upside else None,
        "mean_downside_avoidance": float(np.mean(downside)) if downside else None,
        "mean_return_while_in_market": float(np.mean([row.return_while_in_market for row in folds])),
        "mean_return_while_in_cash": float(np.mean([row.return_while_in_cash for row in folds])),
        "total_buy_count": int(sum(row.buy_count for row in folds)),
        "total_sell_count": int(sum(row.sell_count for row in folds)),
        "mean_trade_return": float(np.mean(trade_returns)) if trade_returns else None,
        "mean_profitable_trade_rate": float(np.mean(profitable)) if profitable else None,
        "calibrated_switch_margin_mean": float(np.mean([row.calibrated_switch_margin for row in folds])),
        "effective_switch_margin_mean": float(np.mean([row.effective_switch_margin for row in folds])),
    }


def _qualifies(row: dict[str, Any], rule: str) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if rule == "all-folds-positive":
        checks = {
            "all_folds_beat_buy_hold": float(row["beat_buy_hold_fold_rate"]) == 1.0,
            "worst_fold_excess_positive": float(row["worst_fold_excess_return"]) > 0.0,
            "compound_oos_excess_positive": float(row["compound_oos_excess_return"]) > 0.0,
        }
    else:
        checks = {
            "majority_of_folds_beat_buy_hold": float(row["beat_buy_hold_fold_rate"]) > 0.5,
            "median_fold_excess_positive": float(row["median_fold_excess_return"]) > 0.0,
            "compound_oos_excess_positive": float(row["compound_oos_excess_return"]) > 0.0,
            "market_periods_better_than_cash_periods": (
                float(row["mean_return_while_in_market"]) > float(row["mean_return_while_in_cash"])
            ),
        }
    for name, accepted in checks.items():
        if not accepted:
            reasons.append(name)
    return bool(all(checks.values())), reasons


def _analyse_asset(
    symbol: str,
    source: str,
    raw: pd.DataFrame,
    config: BacktestRequest,
    random_state: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.perf_counter()
    frame = build_rotation_frame(raw, config)
    common_dates = pd.DatetimeIndex(frame.index).sort_values()
    folds = _build_walk_forward_folds(common_dates, config)
    fold_rows: list[FoldMetrics] = []
    for fold in folds:
        fold_id = int(fold["fold_id"])
        seed = int(random_state) + fold_id
        policy, calibrated, effective, calibration_score = _calibrate_policy(
            symbol,
            frame,
            common_dates,
            fold,
            config,
            seed,
        )
        fold_rows.append(
            _simulate_fold(
                symbol,
                raw,
                frame,
                fold,
                policy,
                config,
                calibrated,
                effective,
                calibration_score,
            )
        )
    aggregate = _aggregate(symbol, source, fold_rows)
    aggregate["elapsed_seconds"] = float(time.perf_counter() - started)
    return aggregate, [asdict(row) for row in fold_rows]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Revalidate all current Strategy assets and complete local-Mongo candidates by "
            "point-in-time Strategy LightGBM timing versus same-asset buy-and-hold. "
            "No full Strategy backtest participates in qualification."
        )
    )
    parser.add_argument("--strategy-sequence", type=int, default=10)
    parser.add_argument("--strategy-id", default=None)
    parser.add_argument("--snapshot-end", default=None)
    parser.add_argument("--mongo-uri", default=None)
    parser.add_argument("--database", default=None)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--candidate-symbols", nargs="*", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--allow-remote-mongo", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--qualification-rule",
        choices=["positive-median-and-majority", "all-folds-positive"],
        default="positive-median-and-majority",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    common.load_project_environment(args.env_file)

    mongo_uri = str(
        args.mongo_uri
        or os.getenv("MONGO_URL")
        or os.getenv("MONGO_URI")
        or "mongodb://localhost:27017"
    ).strip()
    database_name = str(args.database or os.getenv("MONGO_DATABASE") or "").strip()
    if not database_name:
        raise RuntimeError("MONGO_DATABASE is required in .env or via --database.")
    common._assert_local_mongo(mongo_uri, bool(args.allow_remote_mongo))

    workers = max(
        1,
        int(
            args.workers
            if args.workers is not None
            else (os.getenv("ASSET_DISCOVERY_REPLAY_WORKERS") or DEFAULT_WORKERS)
        ),
    )
    os.environ.setdefault("MCT_MODEL_THREADS_OVERRIDE", "1")

    client = MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=3_000,
        connectTimeoutMS=3_000,
        maxPoolSize=max(8, workers + 2),
        retryWrites=False,
    )
    client.admin.command("ping")
    db = client[database_name]

    strategy = common._strategy_document(db, args.strategy_sequence, args.strategy_id)
    configuration = common._configuration(strategy)
    config = BacktestRequest.model_validate(configuration)
    strategy_mode = str(config.strategy_mode)
    if strategy_mode not in SUPPORTED_INDIVIDUAL_TIMING_MODES:
        raise RuntimeError(
            "This v1 individual-timing experiment supports the current LightGBM base/absolute-utility "
            f"rotation policy only. Strategy mode received: {strategy_mode}."
        )
    if str(config.research_model_family) != "lightgbm_utility":
        raise RuntimeError("This experiment requires the Strategy LightGBM Utility model snapshot.")

    baseline_assets = [str(symbol).strip().upper() for symbol in config.assets if str(symbol).strip()]
    baseline_assets = list(dict.fromkeys(baseline_assets))
    baseline_set = set(baseline_assets)
    if len(baseline_assets) < 2:
        raise RuntimeError("The selected Strategy must contain at least two assets.")

    start_date = common._normalize_date(config.start_date)
    identity = common._market_identity(configuration)
    collection = db[common.ALPACA_MARKET_BARS_COLLECTION]
    snapshot_end = (
        common._normalize_date(args.snapshot_end)
        if args.snapshot_end
        else common._latest_common_session(collection, baseline_assets, identity)
    )

    if args.candidate_symbols:
        external = sorted({str(item).strip().upper() for item in args.candidate_symbols if str(item).strip()} - baseline_set)
        candidate_source = "explicit_candidate_symbols"
    else:
        cached = {
            str(item).strip().upper()
            for item in collection.distinct("symbol", identity)
            if str(item).strip()
        }
        external = sorted(cached - baseline_set)
        candidate_source = "all_local_mongo_cached_symbols"
    universe = [*baseline_assets, *external]

    output_dir = Path(
        args.output_dir
        or PROJECT_ROOT
        / "research_output"
        / f"asset_timing_strategy_{int(strategy.get('strategy_sequence') or args.strategy_sequence)}_{snapshot_end.date().isoformat()}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _log(
        f"Start: Strategy #{int(strategy.get('strategy_sequence') or args.strategy_sequence)}, "
        f"existing={len(baseline_assets)}, external_cached={len(external)}, workers={workers}."
    )
    _log("Step 1/6 - Loading the complete research universe from local MongoDB in one query...")
    load_started = time.perf_counter()
    frames = _load_frames_allow_incomplete(collection, universe, identity, start_date, snapshot_end)
    expected = common._expected_sessions(start_date, snapshot_end)
    diagnostics, complete_assets = _history_diagnostics(frames, universe, baseline_set, expected)
    _write_csv(output_dir / "asset_history_integrity.csv", pd.DataFrame(diagnostics))
    missing_existing = [symbol for symbol in baseline_assets if symbol not in set(complete_assets)]
    if missing_existing:
        raise RuntimeError(
            "Existing Strategy assets failed Full Strategy History revalidation: " + ", ".join(missing_existing)
        )
    valid_candidates = [symbol for symbol in external if symbol in set(complete_assets)]
    research_assets = [*baseline_assets, *valid_candidates]
    research_frames = {symbol: frames[symbol] for symbol in research_assets}
    _log(
        f"Loaded {sum(len(frame) for frame in research_frames.values()):,} OHLCV rows in "
        f"{time.perf_counter() - load_started:.1f}s. Complete universe={len(research_assets)}."
    )

    client.close()
    _log("MongoDB connection closed. All timing validation below runs from RAM/CPU only.")

    random_state = int(args.random_state if args.random_state is not None else config.random_state)
    result_path = output_dir / "asset_timing_summary.csv"
    fold_path = output_dir / "asset_timing_folds.csv"
    results: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    completed: set[str] = set()
    if not args.no_resume and result_path.exists():
        existing = pd.read_csv(result_path)
        if not existing.empty and "symbol" in existing.columns:
            results = existing.to_dict(orient="records")
            completed = set(existing["symbol"].dropna().astype(str).str.upper())
        if fold_path.exists():
            fold_rows = pd.read_csv(fold_path).to_dict(orient="records")
        if completed:
            _log(f"Resume: {len(completed)} assets already validated.")

    _log("Step 2/6 - Point-in-time LightGBM timing vs same-asset Buy & Hold...")
    pending = [symbol for symbol in research_assets if symbol not in completed]
    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(pending)))) as executor:
        futures = {
            executor.submit(
                _analyse_asset,
                symbol,
                "strategy_existing" if symbol in baseline_set else "candidate",
                research_frames[symbol],
                config,
                random_state,
            ): symbol
            for symbol in pending
        }
        done_count = len(completed)
        for future in as_completed(futures):
            symbol = futures[future]
            aggregate, asset_folds = future.result()
            qualified, reasons = _qualifies(aggregate, args.qualification_rule)
            aggregate["qualified"] = qualified
            aggregate["qualification_fail_reasons"] = "|".join(reasons)
            results = [row for row in results if str(row.get("symbol") or "").upper() != symbol]
            fold_rows = [row for row in fold_rows if str(row.get("symbol") or "").upper() != symbol]
            results.append(aggregate)
            fold_rows.extend(asset_folds)
            done_count += 1
            result_frame = pd.DataFrame(results).sort_values("symbol").reset_index(drop=True)
            fold_frame = pd.DataFrame(fold_rows).sort_values(["symbol", "fold"]).reset_index(drop=True)
            _write_csv(result_path, result_frame)
            _write_csv(fold_path, fold_frame)
            _log(
                f"Timing {done_count}/{len(research_assets)} - {symbol}: "
                f"beat_BH={float(aggregate['beat_buy_hold_fold_rate']):.1%}, "
                f"median_excess={float(aggregate['median_fold_excess_return']):.2%}, "
                f"compound_excess={float(aggregate['compound_oos_excess_return']):.2%}, "
                f"qualified={qualified}."
            )

    _log("Step 3/6 - Ranking by intrinsic timing quality, not by similarity to the original 56 assets...")
    summary = pd.DataFrame(results)
    if summary.empty:
        raise RuntimeError("No asset timing results were produced.")
    summary = summary.sort_values(
        [
            "qualified",
            "compound_oos_excess_return",
            "median_fold_excess_return",
            "beat_buy_hold_fold_rate",
            "mean_drawdown_improvement",
            "symbol",
        ],
        ascending=[False, False, False, False, False, True],
    ).reset_index(drop=True)
    summary["timing_rank"] = np.arange(1, len(summary) + 1)
    _write_csv(result_path, summary)

    _log("Step 4/6 - Freezing all qualification evidence before any full Strategy Backtest...")
    qualified_assets = summary.loc[summary["qualified"] == True, "symbol"].astype(str).tolist()  # noqa: E712
    retained_existing = [symbol for symbol in baseline_assets if symbol in set(qualified_assets)]
    removed_existing = [symbol for symbol in baseline_assets if symbol not in set(qualified_assets)]
    added_candidates = [symbol for symbol in qualified_assets if symbol not in baseline_set]
    rejected_candidates = [symbol for symbol in valid_candidates if symbol not in set(qualified_assets)]

    ranking_records = summary.to_dict(orient="records")
    ranking_sha256 = _sha256_json(ranking_records)
    qualified_assets_sha256 = _sha256_json(qualified_assets)
    frozen = {
        "schema_version": 2,
        "experiment": EXPERIMENT_NAME,
        "script_version": SCRIPT_VERSION,
        "strategy_id": str(strategy.get("_id") or ""),
        "strategy_sequence": int(strategy.get("strategy_sequence") or args.strategy_sequence),
        "strategy_revision": int(strategy.get("revision") or 0),
        "strategy_configuration_hash": strategy.get("configuration_hash"),
        "strategy_model_family": str(config.research_model_family),
        "strategy_mode": str(config.strategy_mode),
        "strategy_start": start_date.date().isoformat(),
        "snapshot_end": snapshot_end.date().isoformat(),
        "qualification_rule": args.qualification_rule,
        "candidate_source": candidate_source,
        "full_strategy_backtest_used_for_selection": False,
        "same_asset_buy_hold_used_as_benchmark": True,
        "existing_strategy_assets_revalidated": True,
        "original_assets": baseline_assets,
        "complete_external_candidates": valid_candidates,
        "qualified_assets": qualified_assets,
        "retained_existing_assets": retained_existing,
        "removed_existing_assets": removed_existing,
        "added_candidate_assets": added_candidates,
        "rejected_candidate_assets": rejected_candidates,
        "ranking_sha256": ranking_sha256,
        "qualified_assets_sha256": qualified_assets_sha256,
    }
    frozen["decision_snapshot_sha256"] = _sha256_json(frozen)
    _write_json(output_dir / "timing_validation_snapshot_frozen.json", frozen)

    _log("Step 5/6 - Writing reproducibility manifest...")
    manifest = {
        "schema_version": 2,
        "experiment": EXPERIMENT_NAME,
        "script_version": SCRIPT_VERSION,
        "data_source": "local_mongodb_only",
        "mongo_writes": False,
        "alpaca_network_used": False,
        "full_strategy_backtest_run": False,
        "full_strategy_backtest_used_for_selection": False,
        "selection_basis": "point-in-time Strategy LightGBM timing versus same-asset buy-and-hold",
        "full_strategy_history_required": True,
        "expected_xnys_sessions": int(len(expected)),
        "rotation_features": list(ROTATION_FEATURES),
        "walk_forward_protocol_source": "Strategy BacktestRequest + engine._build_walk_forward_folds",
        "switch_margin_calibration": "same Strategy calibration procedure, inside each fold using only prior calibration data",
        "execution_timing": "decision at close, position change at next session open",
        "qualification_rule": args.qualification_rule,
        "workers": workers,
        "model_threads_per_worker": int(os.getenv("MCT_MODEL_THREADS_OVERRIDE") or 1),
        "random_state": random_state,
        "result_counts": {
            "original_strategy_assets": len(baseline_assets),
            "complete_external_candidates": len(valid_candidates),
            "evaluated": len(summary),
            "qualified": len(qualified_assets),
            "retained_existing": len(retained_existing),
            "removed_existing": len(removed_existing),
            "added_candidates": len(added_candidates),
            "rejected_candidates": len(rejected_candidates),
        },
        "ranking_sha256": ranking_sha256,
        "qualified_assets_sha256": qualified_assets_sha256,
        "decision_snapshot_sha256": frozen["decision_snapshot_sha256"],
    }
    _write_json(output_dir / "experiment_manifest.json", manifest)

    _log("Step 6/6 - Qualification complete; full Strategy Backtest intentionally not started.")
    _log(
        f"Final frozen universe: {len(qualified_assets)} assets = "
        f"{len(retained_existing)} retained existing + {len(added_candidates)} new candidates."
    )
    if removed_existing:
        _log("Existing assets removed by revalidation: " + ", ".join(removed_existing))
    if added_candidates:
        _log("New qualified assets: " + ", ".join(added_candidates))
    _log(f"Frozen snapshot: {output_dir / 'timing_validation_snapshot_frozen.json'}")
    _log("Review the frozen files, then run research_asset_timing_final_backtest.py exactly once.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
