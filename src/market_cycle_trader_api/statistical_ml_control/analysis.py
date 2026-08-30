from __future__ import annotations

from collections import Counter
import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..classification_evaluation import roc_curve_payload
from .config import ACTIONS, ANALYSIS_VERSION, SCHEMA_VERSION


def _finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _stamp(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _day_key(value: Any) -> str | None:
    stamp = _stamp(value)
    return stamp.date().isoformat() if stamp is not None else None


def _robust_scale(values: list[float]) -> tuple[float, float]:
    clean = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=float)
    if len(clean) == 0:
        return 0.0, 1.0
    median = float(np.median(clean))
    mad = float(np.median(np.abs(clean - median)))
    scale = 1.4826 * mad
    if not math.isfinite(scale) or scale <= 1e-9:
        std = float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0
        scale = std if math.isfinite(std) and std > 1e-9 else 1.0
    return median, scale


def _robust_z(value: float | None, history: list[float]) -> float | None:
    if value is None or len(history) < 5:
        return None
    center, scale = _robust_scale(history)
    return float((value - center) / scale)


def _tail_percentile(value: float | None, history: list[float]) -> float | None:
    if value is None or len(history) < 5:
        return None
    absolute = abs(float(value))
    reference = [abs(float(item)) for item in history if math.isfinite(float(item))]
    if not reference:
        return None
    return float(sum(item <= absolute for item in reference) / len(reference))


def _cross_section_z(values: dict[str, float | None]) -> dict[str, float | None]:
    clean = [float(value) for value in values.values() if value is not None and math.isfinite(float(value))]
    if len(clean) < 3:
        return {symbol: None for symbol in values}
    center, scale = _robust_scale(clean)
    return {
        symbol: None if value is None else float((float(value) - center) / scale)
        for symbol, value in values.items()
    }


def _future_return(series: pd.Series | None, execution_at: pd.Timestamp, horizon: int) -> float | None:
    if series is None or execution_at not in series.index:
        return None
    loc = series.index.get_loc(execution_at)
    if not isinstance(loc, (int, np.integer)) or int(loc) + int(horizon) >= len(series):
        return None
    start = _finite(series.iloc[int(loc)])
    end = _finite(series.iloc[int(loc) + int(horizon)])
    if start in {None, 0.0} or end is None:
        return None
    return float(end / start - 1.0)


def _open_series(observation_rows: list[dict[str, Any]]) -> dict[str, pd.Series]:
    frame = pd.DataFrame(observation_rows)
    if frame.empty:
        return {}
    frame["execution_date"] = pd.to_datetime(frame.get("execution_date"), utc=True, errors="coerce")
    frame["execution_open"] = pd.to_numeric(frame.get("execution_open"), errors="coerce")
    frame["symbol"] = frame.get("symbol").astype(str).str.upper()
    frame = frame.dropna(subset=["execution_date", "execution_open", "symbol"])
    result: dict[str, pd.Series] = {}
    for symbol, group in frame.groupby("symbol"):
        series = (
            group.drop_duplicates("execution_date")
            .sort_values("execution_date")
            .set_index("execution_date")["execution_open"]
        )
        result[str(symbol).upper()] = series
    return result


def _path_stats(values: list[dict[str, Any]]) -> dict[str, Any]:
    if not values:
        return {}
    frame = pd.DataFrame(values).copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "value"]).sort_values("timestamp")
    if frame.empty:
        return {}
    series = frame["value"].to_numpy(dtype=float)
    initial = float(frame.iloc[0].get("starting_value") or series[0])
    previous = np.concatenate(([initial], series[:-1]))
    daily_returns = series / previous - 1.0
    std = float(np.std(daily_returns, ddof=1)) if len(daily_returns) > 1 else 0.0
    sharpe = float(np.mean(daily_returns) / std * math.sqrt(252.0)) if std > 1e-12 else None
    peaks = np.maximum.accumulate(series)
    drawdowns = series / peaks - 1.0
    elapsed_years = max((frame.iloc[-1]["timestamp"] - frame.iloc[0]["timestamp"]).days / 365.25, 1.0 / 365.25)
    cagr = float((series[-1] / initial) ** (1.0 / elapsed_years) - 1.0) if initial > 0 and series[-1] > 0 else None
    work = frame[["timestamp", "value"]].copy()
    work["month"] = work["timestamp"].dt.strftime("%Y-%m")
    monthly: list[dict[str, Any]] = []
    prior = initial
    for month, group in work.groupby("month", sort=True):
        end_value = float(group.iloc[-1]["value"])
        monthly.append({"month": str(month), "return": float(end_value / prior - 1.0) if prior else None})
        prior = end_value
    worst = min((row for row in monthly if row["return"] is not None), key=lambda row: float(row["return"]), default=None)
    return {
        "initial_capital": initial,
        "ending_capital": float(series[-1]),
        "total_return": float(series[-1] / initial - 1.0) if initial else None,
        "cagr": cagr,
        "sharpe": sharpe,
        "maximum_drawdown": float(np.min(drawdowns)) if len(drawdowns) else None,
        "worst_month": worst,
        "monthly_returns": monthly,
    }


def _equity_path(reference_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous_value: float | None = None
    for item in reference_rows:
        timestamp = _stamp(item.get("timestamp"))
        value = _finite(item.get("strategy_equity"))
        if timestamp is None or value is None:
            continue
        starting = previous_value if previous_value is not None else _finite(item.get("initial_capital")) or value
        rows.append({"timestamp": timestamp.isoformat(), "stamp": timestamp, "value": value, "starting_value": starting})
        previous_value = value
    rows.sort(key=lambda row: row["stamp"])
    return rows


def _model(settings: dict[str, Any], seed_offset: int) -> Pipeline:
    seed = int(settings.get("random_state") or 42) + int(seed_offset)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("model", RandomForestClassifier(
            n_estimators=int(settings.get("n_estimators") or 320),
            max_depth=int(settings.get("max_depth") or 5),
            min_samples_leaf=int(settings.get("min_samples_leaf") or 8),
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        )),
    ])


def _positive_probability(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    classes = list(model.named_steps["model"].classes_)
    if 1 not in classes:
        return np.zeros(len(frame), dtype=float)
    return model.predict_proba(frame)[:, classes.index(1)]


def _safe_auc(y_true: pd.Series | np.ndarray, probabilities: np.ndarray) -> float | None:
    truth = np.asarray(y_true, dtype=int)
    if len(truth) == 0 or len(np.unique(truth)) < 2:
        return None
    return float(roc_auc_score(truth, probabilities))


def _threshold_candidates(settings: dict[str, Any]) -> list[float]:
    raw = settings.get("probability_threshold_candidates") or [0.55, 0.60, 0.65, 0.70, 0.75]
    values = sorted({float(value) for value in raw if 0.0 < float(value) < 1.0})
    return values or [0.65]


def _select_threshold(y_true: pd.Series, probabilities: np.ndarray, settings: dict[str, Any]) -> tuple[float, float | None]:
    truth = np.asarray(y_true, dtype=int)
    if len(truth) == 0 or len(np.unique(truth)) < 2:
        default = float(settings.get("default_probability_threshold") or 0.65)
        return default, None
    ranked: list[tuple[float, float, float]] = []
    for threshold in _threshold_candidates(settings):
        prediction = probabilities >= threshold
        score = float(balanced_accuracy_score(truth, prediction))
        ranked.append((score, -abs(threshold - 0.65), threshold))
    score, _, threshold = max(ranked)
    return float(threshold), float(score)


def _binary_metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float, *, origin: str) -> dict[str, Any]:
    truth = np.asarray(y_true, dtype=int)
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    prediction = clipped >= float(threshold)
    return {
        "rows": int(len(truth)),
        "positive_rate": float(np.mean(truth)) if len(truth) else None,
        "auc": _safe_auc(truth, clipped),
        "brier": float(brier_score_loss(truth, clipped)) if len(truth) else None,
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)) if len(truth) else None,
        "threshold": float(threshold),
        "roc": roc_curve_payload(truth, clipped, operating_threshold=float(threshold), threshold_origin=origin),
    }


def _weighted_asset_utility(
    open_series: pd.Series | None,
    execution_at: pd.Timestamp,
    settings: dict[str, Any],
) -> tuple[float | None, dict[int, float | None]]:
    horizons = [int(value) for value in (settings.get("horizons_sessions") or [1, 3, 5])]
    weights_raw = settings.get("horizon_weights") or {"1": 0.45, "3": 0.35, "5": 0.20}
    weights = {int(key): float(value) for key, value in weights_raw.items()}
    returns = {horizon: _future_return(open_series, execution_at, horizon) for horizon in horizons}
    if any(returns[horizon] is None for horizon in horizons):
        return None, returns
    total_weight = sum(max(0.0, weights.get(horizon, 0.0)) for horizon in horizons)
    if total_weight <= 0:
        return None, returns
    weighted = sum(weights.get(horizon, 0.0) * float(returns[horizon]) for horizon in horizons) / total_weight
    downside = max(0.0, -min(float(returns[horizon]) for horizon in horizons))
    penalty = float(settings.get("downside_penalty") or 0.50)
    return float(weighted - penalty * downside), returns


def _alternative_score(row: dict[str, Any]) -> float | None:
    return _finite(row.get("risk_adjusted_entry_score")) or _finite(row.get("entry_rank_score"))


def _best_alternative(
    candidates: list[dict[str, Any]],
    *,
    current_symbol: str,
) -> dict[str, Any] | None:
    eligible = []
    for row in candidates:
        symbol = str(row.get("symbol") or "").strip().upper()
        score = _alternative_score(row)
        if not symbol or symbol == current_symbol or symbol == "CASH" or score is None:
            continue
        eligible.append((float(score), str(symbol), row))
    if not eligible:
        return None
    eligible.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return dict(eligible[0][2])


def _prepare_rows(
    reference_rows: list[dict[str, Any]],
    observation_rows: list[dict[str, Any]],
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    lookback = max(20, int(settings.get("lookback_sessions") or 60))
    minimum_history = max(10, int(settings.get("minimum_history_sessions") or 30))

    observations = [dict(row) for row in observation_rows if isinstance(row, dict)]
    by_decision_symbol: dict[tuple[str, str], dict[str, Any]] = {}
    symbol_rows: dict[str, list[dict[str, Any]]] = {}
    for row in observations:
        symbol = str(row.get("symbol") or "").strip().upper()
        decision_day = _day_key(row.get("timestamp"))
        if not symbol or not decision_day:
            continue
        row["symbol"] = symbol
        row["decision_day"] = decision_day
        by_decision_symbol[(decision_day, symbol)] = row
        symbol_rows.setdefault(symbol, []).append(row)

    for rows in symbol_rows.values():
        rows.sort(key=lambda row: _stamp(row.get("timestamp")) or pd.Timestamp.min.tz_localize("UTC"))

    close_returns_by_day: dict[str, dict[str, float | None]] = {}
    open_gaps_by_exec_day: dict[str, dict[str, float | None]] = {}
    histories: dict[str, dict[str, list[float]]] = {}
    enriched: dict[tuple[str, str], dict[str, Any]] = {}
    for symbol, rows in symbol_rows.items():
        close_history: list[float] = []
        gap_history: list[float] = []
        previous_close: float | None = None
        for row in rows:
            decision_day = row["decision_day"]
            execution_day = _day_key(row.get("execution_date"))
            close = _finite(row.get("decision_close"))
            open_price = _finite(row.get("execution_open"))
            close_return = None if previous_close in {None, 0.0} or close is None else float(close / previous_close - 1.0)
            open_gap = None if close in {None, 0.0} or open_price is None else float(open_price / close - 1.0)
            close_z = _robust_z(close_return, close_history[-lookback:]) if len(close_history) >= minimum_history else None
            gap_z = _robust_z(open_gap, gap_history[-lookback:]) if len(gap_history) >= minimum_history else None
            close_tail = _tail_percentile(close_return, close_history[-lookback:]) if len(close_history) >= minimum_history else None
            gap_tail = _tail_percentile(open_gap, gap_history[-lookback:]) if len(gap_history) >= minimum_history else None
            copy = dict(row)
            copy.update({
                "close_return_1d": close_return,
                "opening_gap": open_gap,
                "close_return_robust_z": close_z,
                "opening_gap_robust_z": gap_z,
                "close_abs_tail_percentile": close_tail,
                "opening_gap_abs_tail_percentile": gap_tail,
            })
            enriched[(decision_day, symbol)] = copy
            close_returns_by_day.setdefault(decision_day, {})[symbol] = close_return
            if execution_day:
                open_gaps_by_exec_day.setdefault(execution_day, {})[symbol] = open_gap
            if close_return is not None:
                close_history.append(close_return)
            if open_gap is not None:
                gap_history.append(open_gap)
            if close is not None:
                previous_close = close
        histories[symbol] = {"close": close_history, "gap": gap_history}

    close_cross = {day: _cross_section_z(values) for day, values in close_returns_by_day.items()}
    gap_cross = {day: _cross_section_z(values) for day, values in open_gaps_by_exec_day.items()}
    open_series = _open_series(observations)
    observations_by_day: dict[str, list[dict[str, Any]]] = {}
    for (decision_day, _symbol), row in enriched.items():
        observations_by_day.setdefault(decision_day, []).append(row)

    result: list[dict[str, Any]] = []
    for reference in reference_rows:
        if not isinstance(reference, dict):
            continue
        execution_at = _stamp(reference.get("timestamp"))
        decision_day = _day_key(reference.get("decision_date"))
        execution_day = _day_key(reference.get("timestamp"))
        symbol = str(reference.get("selected_asset") or "").strip().upper()
        if execution_at is None or not decision_day or not execution_day or not symbol or symbol == "CASH":
            continue
        observation = enriched.get((decision_day, symbol))
        if not observation:
            continue
        utility, horizon_returns = _weighted_asset_utility(open_series.get(symbol), execution_at, settings)
        if utility is None:
            continue
        alternative = _best_alternative(observations_by_day.get(decision_day, []), current_symbol=symbol)
        alternative_symbol = str((alternative or {}).get("symbol") or "").upper() or None
        alternative_utility = None
        alternative_horizon_returns: dict[int, float | None] = {}
        if alternative_symbol:
            alternative_utility, alternative_horizon_returns = _weighted_asset_utility(
                open_series.get(alternative_symbol), execution_at, settings
            )
        minimum_cash_edge = float(settings.get("minimum_cash_edge") or 0.005)
        minimum_rotation_edge = float(settings.get("minimum_rotation_edge") or 0.005)
        target_rotate = int(bool(
            alternative_symbol
            and alternative_utility is not None
            and float(alternative_utility) >= float(utility) + minimum_rotation_edge
            and float(alternative_utility) >= 0.0 + minimum_rotation_edge
        ))
        target_cash = int(bool(
            0.0 >= float(utility) + minimum_cash_edge
            and (alternative_utility is None or 0.0 >= float(alternative_utility) + minimum_cash_edge)
        ))
        target_action = "ROTATE" if target_rotate else "CASH" if target_cash else "FOLLOW_BASE"
        finite_count = _finite(reference.get("finite_score_count"))
        positive_count = _finite(reference.get("positive_score_count"))
        positive_share = None if finite_count in {None, 0.0} or positive_count is None else float(positive_count / finite_count)
        rank_percentile = _finite(observation.get("entry_rank_percentile"))
        risk_safety = _finite(observation.get("all_horizon_risk_safety"))
        risk_health = _finite(observation.get("incumbent_risk_health"))
        opportunity_risk_divergence = None
        if rank_percentile is not None and risk_safety is not None:
            opportunity_risk_divergence = float(rank_percentile * max(0.0, 1.0 - risk_safety))
        close_z = _finite(observation.get("close_return_robust_z"))
        gap_z = _finite(observation.get("opening_gap_robust_z"))
        close_tail = _finite(observation.get("close_abs_tail_percentile"))
        gap_tail = _finite(observation.get("opening_gap_abs_tail_percentile"))
        shock_tail_score = max([value for value in (close_tail, gap_tail) if value is not None], default=None)
        extreme_tail = float(settings.get("extreme_tail_percentile") or 0.98)
        extreme_z = float(settings.get("extreme_robust_z") or 3.0)
        statistical_close_shock = int(bool(
            (close_tail is not None and close_tail >= extreme_tail)
            or (close_z is not None and abs(close_z) >= extreme_z)
        ))
        statistical_open_shock = int(bool(
            (gap_tail is not None and gap_tail >= extreme_tail)
            or (gap_z is not None and abs(gap_z) >= extreme_z)
        ))
        opportunity_floor = float(settings.get("opportunity_conflict_min_percentile") or 0.75)
        risk_ceiling = float(settings.get("risk_conflict_max_safety") or 0.25)
        opportunity_risk_conflict = int(bool(
            rank_percentile is not None and rank_percentile >= opportunity_floor
            and ((risk_safety is not None and risk_safety <= risk_ceiling) or (risk_health is not None and risk_health <= risk_ceiling))
        ))
        row = {
            "execution_at": execution_at.isoformat(),
            "decision_at": _stamp(reference.get("decision_date")).isoformat() if _stamp(reference.get("decision_date")) is not None else reference.get("decision_date"),
            "year": int(execution_at.year),
            "symbol": symbol,
            "previous_asset": str(reference.get("previous_asset") or "CASH").upper(),
            "base_action": reference.get("trade_action") or ("ROTATE" if bool(reference.get("decision_is_rotation")) else "HOLD"),
            "decision_reason": reference.get("decision_reason"),
            "strategy_equity": _finite(reference.get("strategy_equity")),
            "target_cash": target_cash,
            "target_rotate": target_rotate,
            "target_action": target_action,
            "asset_utility": utility,
            "alternative_symbol": alternative_symbol,
            "alternative_utility": alternative_utility,
            "alternative_entry_rank_score": _finite((alternative or {}).get("entry_rank_score")),
            "alternative_entry_rank_percentile": _finite((alternative or {}).get("entry_rank_percentile")),
            "alternative_risk_adjusted_entry_score": _finite((alternative or {}).get("risk_adjusted_entry_score")),
            "alternative_incumbent_risk_health": _finite((alternative or {}).get("incumbent_risk_health")),
            "alternative_all_horizon_risk_safety": _finite((alternative or {}).get("all_horizon_risk_safety")),
            "alternative_predicted_drawdown": _finite((alternative or {}).get("predicted_drawdown")),
            "alternative_short_profit_consensus": _finite((alternative or {}).get("short_profit_consensus")),
            "alternative_long_profit_confirmation": _finite((alternative or {}).get("long_profit_confirmation")),
            "alternative_horizon_agreement": _finite((alternative or {}).get("horizon_agreement")),
            "alternative_opening_gap": _finite((alternative or {}).get("opening_gap")),
            "alternative_opening_gap_robust_z": _finite((alternative or {}).get("opening_gap_robust_z")),
            "alternative_opening_gap_cross_section_robust_z": (gap_cross.get(execution_day) or {}).get(alternative_symbol) if alternative_symbol else None,
            "alternative_opening_gap_abs_tail_percentile": _finite((alternative or {}).get("opening_gap_abs_tail_percentile")),
            "alternative_vs_base_risk_adjusted_gap": (
                None if _finite((alternative or {}).get("risk_adjusted_entry_score")) is None or _finite(observation.get("risk_adjusted_entry_score")) is None
                else float(_finite((alternative or {}).get("risk_adjusted_entry_score")) - _finite(observation.get("risk_adjusted_entry_score")))
            ),
            "close_return_1d": _finite(observation.get("close_return_1d")),
            "close_return_robust_z": close_z,
            "close_cross_section_robust_z": (close_cross.get(decision_day) or {}).get(symbol),
            "close_abs_tail_percentile": close_tail,
            "opening_gap": _finite(observation.get("opening_gap")),
            "opening_gap_robust_z": gap_z,
            "opening_gap_cross_section_robust_z": (gap_cross.get(execution_day) or {}).get(symbol),
            "opening_gap_abs_tail_percentile": gap_tail,
            "shock_tail_score": shock_tail_score,
            "statistical_close_shock": statistical_close_shock,
            "statistical_open_shock": statistical_open_shock,
            "opportunity_risk_conflict": opportunity_risk_conflict,
            "entry_rank_score": _finite(observation.get("entry_rank_score")),
            "entry_rank_percentile": rank_percentile,
            "risk_adjusted_entry_score": _finite(observation.get("risk_adjusted_entry_score")),
            "incumbent_risk_health": risk_health,
            "all_horizon_risk_safety": risk_safety,
            "predicted_drawdown": _finite(observation.get("predicted_drawdown")),
            "short_profit_consensus": _finite(observation.get("short_profit_consensus")),
            "long_profit_confirmation": _finite(observation.get("long_profit_confirmation")),
            "horizon_agreement": _finite(observation.get("horizon_agreement")),
            "opportunity_risk_divergence": opportunity_risk_divergence,
            "raw_best_score": _finite(reference.get("raw_best_score")),
            "best_score_zscore": _finite(reference.get("best_score_zscore")),
            "current_score_zscore": _finite(reference.get("current_score_zscore")),
            "positive_score_share": positive_share,
            "universe_breadth_20": _finite(reference.get("universe_breadth_20")),
            "spy_return_5": _finite(reference.get("spy_return_5")),
            "spy_return_20": _finite(reference.get("spy_return_20")),
            "spy_realized_volatility_20": _finite(reference.get("spy_realized_volatility_20")),
            "position_return_since_entry": _finite(reference.get("position_return_since_entry")),
            "position_drawdown_from_peak": _finite(reference.get("position_drawdown_from_peak")),
            "decision_is_rotation": int(bool(reference.get("decision_is_rotation") or reference.get("trade_action") == "ROTATE")),
            "min_hold_guard_applied": int(bool(reference.get("min_hold_guard_applied"))),
        }
        for horizon, value in horizon_returns.items():
            row[f"asset_return_{int(horizon)}d"] = value
        for horizon, value in alternative_horizon_returns.items():
            row[f"alternative_return_{int(horizon)}d"] = value
        result.append(row)
    result.sort(key=lambda row: _stamp(row.get("execution_at")) or pd.Timestamp.min.tz_localize("UTC"))
    return result


CLOSE_FEATURES = (
    "close_return_1d",
    "close_return_robust_z",
    "close_cross_section_robust_z",
    "close_abs_tail_percentile",
    "statistical_close_shock",
    "opportunity_risk_conflict",
    "entry_rank_score",
    "entry_rank_percentile",
    "risk_adjusted_entry_score",
    "incumbent_risk_health",
    "all_horizon_risk_safety",
    "predicted_drawdown",
    "short_profit_consensus",
    "long_profit_confirmation",
    "horizon_agreement",
    "opportunity_risk_divergence",
    "raw_best_score",
    "best_score_zscore",
    "current_score_zscore",
    "positive_score_share",
    "universe_breadth_20",
    "spy_return_5",
    "spy_return_20",
    "spy_realized_volatility_20",
    "position_return_since_entry",
    "position_drawdown_from_peak",
    "decision_is_rotation",
    "min_hold_guard_applied",
    "alternative_entry_rank_score",
    "alternative_entry_rank_percentile",
    "alternative_risk_adjusted_entry_score",
    "alternative_incumbent_risk_health",
    "alternative_all_horizon_risk_safety",
    "alternative_predicted_drawdown",
    "alternative_short_profit_consensus",
    "alternative_long_profit_confirmation",
    "alternative_horizon_agreement",
    "alternative_vs_base_risk_adjusted_gap",
)

OPEN_FEATURES = CLOSE_FEATURES + (
    "opening_gap",
    "opening_gap_robust_z",
    "opening_gap_cross_section_robust_z",
    "opening_gap_abs_tail_percentile",
    "shock_tail_score",
    "statistical_open_shock",
    "alternative_opening_gap",
    "alternative_opening_gap_robust_z",
    "alternative_opening_gap_cross_section_robust_z",
    "alternative_opening_gap_abs_tail_percentile",
)


REGIME_SOURCE_FEATURES = (
    "positive_score_share",
    "universe_breadth_20",
    "spy_return_5",
    "spy_return_20",
    "spy_realized_volatility_20",
    "incumbent_risk_health",
    "all_horizon_risk_safety",
    "predicted_drawdown",
    "short_profit_consensus",
    "long_profit_confirmation",
    "horizon_agreement",
    "position_return_since_entry",
    "position_drawdown_from_peak",
    "best_score_zscore",
    "opportunity_risk_divergence",
)
REGIME_MAX_FEATURE_CLUSTERS = 6
REGIME_CONTEXT_FEATURES = (
    "regime_nearest_distance",
    "regime_second_distance",
    "regime_distance_margin",
    "regime_pca_x",
    "regime_pca_y",
    "regime_q1",
    "regime_q2",
    "regime_q3",
    "regime_q4",
) + tuple(f"regime_similarity_{idx}" for idx in range(REGIME_MAX_FEATURE_CLUSTERS))

REGIME_TRAJECTORY_COMPONENTS = (
    "regime_danger_similarity",
    "regime_danger_balance",
    "regime_danger_approach_3d",
    "regime_danger_approach_window",
    "regime_environment_deterioration_window",
    "regime_q4_persistence",
    "regime_defensive_persistence",
)


def _rolling_regime_source(frame: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    result = frame.copy().sort_values("execution_at").reset_index(drop=True)
    window = max(5, int(settings.get("regime_window_sessions") or 20))
    min_periods = max(3, min(window, window // 3))
    for feature in REGIME_SOURCE_FEATURES:
        values = pd.to_numeric(result.get(feature), errors="coerce")
        result[f"regime_source_{feature}"] = values.rolling(window=window, min_periods=min_periods).mean()
    return result


def _regime_matrix_columns() -> list[str]:
    return [f"regime_source_{feature}" for feature in REGIME_SOURCE_FEATURES]


def _select_regime_cluster_count(matrix: np.ndarray, settings: dict[str, Any]) -> tuple[int, float | None]:
    minimum = max(2, int(settings.get("regime_min_clusters") or 2))
    maximum = min(REGIME_MAX_FEATURE_CLUSTERS, int(settings.get("regime_max_clusters") or 6), max(2, len(matrix) - 1))
    candidates: list[tuple[float, int]] = []
    for count in range(minimum, maximum + 1):
        if len(matrix) <= count:
            continue
        labels = AgglomerativeClustering(n_clusters=count, linkage="ward").fit_predict(matrix)
        if len(set(labels)) < 2:
            continue
        try:
            score = float(silhouette_score(matrix, labels))
        except Exception:
            continue
        candidates.append((score, count))
    if not candidates:
        return minimum, None
    score, count = max(candidates, key=lambda item: (item[0], -item[1]))
    return int(count), float(score)


def _canonical_cluster_order(centroids: np.ndarray, columns: list[str]) -> list[int]:
    index = {name: position for position, name in enumerate(columns)}
    def value(row: np.ndarray, feature: str) -> float:
        pos = index.get(f"regime_source_{feature}")
        return 0.0 if pos is None else float(row[pos])
    scored = []
    for original, row in enumerate(centroids):
        defensive = (
            -value(row, "universe_breadth_20")
            -value(row, "spy_return_5")
            -value(row, "spy_return_20")
            -value(row, "incumbent_risk_health")
            -value(row, "all_horizon_risk_safety")
            +value(row, "spy_realized_volatility_20")
            +value(row, "predicted_drawdown")
            +value(row, "opportunity_risk_divergence")
        )
        scored.append((float(defensive), int(original)))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [original for _score, original in scored]


def _attach_regime_trajectory_raw(
    source: pd.DataFrame,
    *,
    settings: dict[str, Any],
) -> pd.DataFrame:
    if source.empty:
        return source.copy()
    result = source.copy().sort_values("execution_at").reset_index(drop=True)
    window = max(3, int(settings.get("regime_trajectory_window_sessions") or 5))
    danger_distance = pd.to_numeric(result.get("regime_danger_distance"), errors="coerce")
    danger_balance = pd.to_numeric(result.get("regime_danger_balance"), errors="coerce")
    pca_x = pd.to_numeric(result.get("regime_pca_x"), errors="coerce")
    pca_y = pd.to_numeric(result.get("regime_pca_y"), errors="coerce")
    q4 = pd.to_numeric(result.get("regime_q4"), errors="coerce").fillna(0.0)
    defensive = pd.to_numeric(result.get("regime_is_defensive_cluster"), errors="coerce").fillna(0.0)

    result["regime_danger_approach_1d"] = danger_distance.shift(1) - danger_distance
    result["regime_danger_approach_3d"] = danger_distance.shift(3) - danger_distance
    result["regime_danger_approach_window"] = danger_distance.shift(window) - danger_distance
    result["regime_danger_balance_delta_1d"] = danger_balance - danger_balance.shift(1)
    result["regime_danger_balance_delta_3d"] = danger_balance - danger_balance.shift(3)
    result["regime_environment_deterioration_window"] = -(pca_y - pca_y.shift(window)) / float(window)
    result["regime_q4_persistence"] = q4.rolling(window=window, min_periods=1).mean()
    result["regime_defensive_persistence"] = defensive.rolling(window=window, min_periods=1).mean()
    step = np.sqrt((pca_x - pca_x.shift(1)) ** 2 + (pca_y - pca_y.shift(1)) ** 2)
    result["regime_path_speed"] = step.rolling(window=window, min_periods=1).mean()
    return result


def _trajectory_score_contract(train: pd.DataFrame, settings: dict[str, Any]) -> dict[str, Any]:
    centers: dict[str, float] = {}
    scales: dict[str, float] = {}
    for feature in REGIME_TRAJECTORY_COMPONENTS:
        values = pd.to_numeric(train.get(feature), errors="coerce").dropna().tolist()
        center, scale = _robust_scale(values)
        centers[feature] = float(center)
        scales[feature] = float(scale)
    scored = _apply_trajectory_score(train, centers=centers, scales=scales)
    valid = pd.to_numeric(scored.get("regime_trajectory_score"), errors="coerce").dropna()
    quantile = min(max(float(settings.get("regime_trajectory_warning_quantile") or 0.90), 0.51), 0.99)
    threshold = float(valid.quantile(quantile)) if not valid.empty else 0.0
    return {"centers": centers, "scales": scales, "warning_quantile": quantile, "warning_threshold": threshold}


def _apply_trajectory_score(
    source: pd.DataFrame,
    *,
    centers: dict[str, float],
    scales: dict[str, float],
) -> pd.DataFrame:
    result = source.copy()
    components: list[np.ndarray] = []
    for feature in REGIME_TRAJECTORY_COMPONENTS:
        values = pd.to_numeric(result.get(feature), errors="coerce").to_numpy(dtype=float)
        center = float(centers.get(feature, 0.0))
        scale = max(float(scales.get(feature, 1.0)), 1e-9)
        z = np.clip((values - center) / scale, -5.0, 5.0)
        components.append(z)
    matrix = np.vstack(components).T if components else np.zeros((len(result), 0), dtype=float)
    with np.errstate(invalid="ignore"):
        score = np.nanmean(matrix, axis=1) if matrix.size else np.zeros(len(result), dtype=float)
    result["regime_trajectory_score"] = score
    return result


def _trajectory_with_history(
    history: pd.DataFrame,
    source: pd.DataFrame,
    *,
    settings: dict[str, Any],
) -> pd.DataFrame:
    if source.empty:
        return source.copy()
    window = max(3, int(settings.get("regime_trajectory_window_sessions") or 5))
    history_tail = history.tail(max(window, 5)).copy() if not history.empty else history.copy()
    history_tail["__trajectory_keep"] = 0
    current = source.copy()
    current["__trajectory_keep"] = 1
    combined = pd.concat([history_tail, current], ignore_index=True)
    combined = _attach_regime_trajectory_raw(combined, settings=settings)
    result = combined[combined["__trajectory_keep"] == 1].drop(columns=["__trajectory_keep"]).reset_index(drop=True)
    return result


def _regime_context_frames(
    train: pd.DataFrame,
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not bool(settings.get("regime_context_enabled", True)):
        return train, fit, validation, test, {"enabled": False}
    columns = _regime_matrix_columns()
    minimum_train = max(30, int(settings.get("regime_min_train_rows") or 120))
    if len(train) < minimum_train:
        return train, fit, validation, test, {"enabled": False, "reason": "insufficient_train_rows"}
    train_matrix_frame = train.reindex(columns=columns).apply(pd.to_numeric, errors="coerce")
    medians = train_matrix_frame.median(numeric_only=True).fillna(0.0)
    train_matrix_frame = train_matrix_frame.fillna(medians).fillna(0.0)
    scaler = StandardScaler()
    train_matrix = scaler.fit_transform(train_matrix_frame.to_numpy(dtype=float))
    cluster_count, silhouette = _select_regime_cluster_count(train_matrix, settings)
    labels = AgglomerativeClustering(n_clusters=cluster_count, linkage="ward").fit_predict(train_matrix)
    raw_centroids = np.vstack([train_matrix[labels == cluster].mean(axis=0) for cluster in range(cluster_count)])
    order = _canonical_cluster_order(raw_centroids, columns)
    centroids = raw_centroids[order]
    original_to_canonical = {original: canonical for canonical, original in enumerate(order)}

    pca = PCA(n_components=2, random_state=int(settings.get("random_state") or 42))
    pca.fit(train_matrix)
    opportunity_anchor = np.zeros(len(columns), dtype=float)
    environment_anchor = np.zeros(len(columns), dtype=float)
    for feature in ("short_profit_consensus", "long_profit_confirmation", "best_score_zscore", "positive_score_share"):
        if f"regime_source_{feature}" in columns:
            opportunity_anchor[columns.index(f"regime_source_{feature}")] = 1.0
    for feature in ("universe_breadth_20", "spy_return_5", "spy_return_20", "incumbent_risk_health", "all_horizon_risk_safety"):
        if f"regime_source_{feature}" in columns:
            environment_anchor[columns.index(f"regime_source_{feature}")] = 1.0
    for feature in ("spy_realized_volatility_20", "predicted_drawdown", "opportunity_risk_divergence"):
        if f"regime_source_{feature}" in columns:
            environment_anchor[columns.index(f"regime_source_{feature}")] = -1.0
    component_scores = [abs(float(np.dot(component, opportunity_anchor))) for component in pca.components_]
    x_component = int(np.argmax(component_scores))
    y_component = 1 - x_component
    x_sign = 1.0 if float(np.dot(pca.components_[x_component], opportunity_anchor)) >= 0.0 else -1.0
    y_sign = 1.0 if float(np.dot(pca.components_[y_component], environment_anchor)) >= 0.0 else -1.0
    temperature = float(settings.get("regime_distance_temperature") or 1.0)

    def transform(source: pd.DataFrame) -> pd.DataFrame:
        result = source.copy()
        matrix_frame = result.reindex(columns=columns).apply(pd.to_numeric, errors="coerce").fillna(medians).fillna(0.0)
        matrix = scaler.transform(matrix_frame.to_numpy(dtype=float))
        distances = np.linalg.norm(matrix[:, None, :] - centroids[None, :, :], axis=2)
        nearest = np.argsort(distances, axis=1)
        pca_raw = pca.transform(matrix)
        pca_x = pca_raw[:, x_component] * x_sign
        pca_y = pca_raw[:, y_component] * y_sign
        for idx in range(REGIME_MAX_FEATURE_CLUSTERS):
            if idx < cluster_count:
                result[f"regime_similarity_{idx}"] = np.exp(-distances[:, idx] / max(temperature, 1e-9))
            else:
                result[f"regime_similarity_{idx}"] = 0.0
        result["regime_cluster_id"] = nearest[:, 0].astype(int)
        result["regime_is_defensive_cluster"] = (nearest[:, 0] == (cluster_count - 1)).astype(int)
        result["regime_healthy_distance"] = distances[:, 0]
        result["regime_danger_distance"] = distances[:, cluster_count - 1]
        result["regime_healthy_similarity"] = np.exp(-distances[:, 0] / max(temperature, 1e-9))
        result["regime_danger_similarity"] = np.exp(-distances[:, cluster_count - 1] / max(temperature, 1e-9))
        result["regime_danger_balance"] = result["regime_danger_similarity"] - result["regime_healthy_similarity"]
        result["regime_nearest_distance"] = distances[np.arange(len(result)), nearest[:, 0]]
        if cluster_count > 1:
            result["regime_second_distance"] = distances[np.arange(len(result)), nearest[:, 1]]
            result["regime_distance_margin"] = result["regime_second_distance"] - result["regime_nearest_distance"]
        else:
            result["regime_second_distance"] = result["regime_nearest_distance"]
            result["regime_distance_margin"] = 0.0
        result["regime_pca_x"] = pca_x
        result["regime_pca_y"] = pca_y
        result["regime_q1"] = ((pca_x >= 0.0) & (pca_y >= 0.0)).astype(int)
        result["regime_q2"] = ((pca_x < 0.0) & (pca_y >= 0.0)).astype(int)
        result["regime_q3"] = ((pca_x < 0.0) & (pca_y < 0.0)).astype(int)
        result["regime_q4"] = ((pca_x >= 0.0) & (pca_y < 0.0)).astype(int)
        result["regime_quadrant"] = np.select(
            [result["regime_q1"] == 1, result["regime_q2"] == 1, result["regime_q3"] == 1, result["regime_q4"] == 1],
            ["Q1", "Q2", "Q3", "Q4"],
            default="Q0",
        )
        return result

    train_base = transform(train)
    fit_base = transform(fit)
    validation_base = transform(validation) if not validation.empty else validation.copy()
    test_base = transform(test)

    trajectory_enabled = bool(settings.get("regime_trajectory_enabled", True))
    trajectory_contract: dict[str, Any] = {"enabled": False}
    if trajectory_enabled:
        train_out = _attach_regime_trajectory_raw(train_base, settings=settings)
        fit_out = _trajectory_with_history(train_base.iloc[0:0], fit_base, settings=settings)
        validation_out = _trajectory_with_history(fit_base, validation_base, settings=settings) if not validation_base.empty else validation_base.copy()
        test_out = _trajectory_with_history(train_base, test_base, settings=settings)
        trajectory_contract = _trajectory_score_contract(train_out, settings)
        train_out = _apply_trajectory_score(train_out, centers=trajectory_contract["centers"], scales=trajectory_contract["scales"])
        fit_out = _apply_trajectory_score(fit_out, centers=trajectory_contract["centers"], scales=trajectory_contract["scales"])
        if not validation_out.empty:
            validation_out = _apply_trajectory_score(validation_out, centers=trajectory_contract["centers"], scales=trajectory_contract["scales"])
        test_out = _apply_trajectory_score(test_out, centers=trajectory_contract["centers"], scales=trajectory_contract["scales"])
        threshold = float(trajectory_contract["warning_threshold"])
        for target_frame in (train_out, fit_out, validation_out, test_out):
            if not target_frame.empty:
                target_frame["regime_trajectory_warning"] = (pd.to_numeric(target_frame["regime_trajectory_score"], errors="coerce") >= threshold).astype(int)
        trajectory_contract = {
            "enabled": True,
            "window_sessions": int(settings.get("regime_trajectory_window_sessions") or 5),
            "warning_quantile": float(trajectory_contract["warning_quantile"]),
            "warning_threshold": threshold,
            "components": list(REGIME_TRAJECTORY_COMPONENTS),
            "fit_scope": "outer-fold training years only",
            "decision_feature": False,
        }
    else:
        train_out, fit_out, validation_out, test_out = train_base, fit_base, validation_base, test_base

    cluster_counts = Counter(int(original_to_canonical.get(int(label), int(label))) for label in labels)
    return train_out, fit_out, validation_out, test_out, {
        "enabled": True,
        "cluster_count": int(cluster_count),
        "silhouette_score": silhouette,
        "train_rows": int(len(train)),
        "window_sessions": int(settings.get("regime_window_sessions") or 20),
        "cluster_counts": {str(key): int(value) for key, value in sorted(cluster_counts.items())},
        "pca_explained_variance_ratio": [float(value) for value in pca.explained_variance_ratio_.tolist()],
        "quadrant_definition": {"x": "opportunity-oriented PCA axis", "y": "market/risk-health-oriented PCA axis"},
        "fit_scope": "outer-fold training years only",
        "trajectory": trajectory_contract,
    }


CLOSE_MODEL_FEATURES = CLOSE_FEATURES + REGIME_CONTEXT_FEATURES
OPEN_MODEL_FEATURES = OPEN_FEATURES + REGIME_CONTEXT_FEATURES


def _mean_available(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(sum(clean) / len(clean)) if clean else None


def _constant_binary_metrics(y_true: pd.Series, probability: float, threshold: float, *, origin: str) -> dict[str, Any]:
    probabilities = np.full(len(y_true), float(probability), dtype=float)
    return _binary_metrics(y_true, probabilities, threshold, origin=origin)


def _binary_fold_model(
    *,
    train: pd.DataFrame,
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    features: tuple[str, ...],
    target: str,
    settings: dict[str, Any],
    seed_offset: int,
    default_threshold: float,
    fit_on_full_train: bool,
) -> tuple[np.ndarray, float, float | None, dict[str, Any]]:
    source_fit = train if fit_on_full_train else fit
    validation_source = validation
    if source_fit[target].nunique() < 2:
        probability = float(source_fit[target].mean()) if len(source_fit) else 0.0
        threshold = float(default_threshold)
        return (
            np.full(len(test), probability, dtype=float),
            threshold,
            None,
            _constant_binary_metrics(test[target], probability, threshold, origin="constant_training_class"),
        )
    model = _model(settings, seed_offset)
    model.fit(source_fit[list(features)], source_fit[target].astype(int))
    if not validation_source.empty and validation_source[target].nunique() >= 2:
        validation_prob = _positive_probability(model, validation_source[list(features)])
        threshold, validation_balanced = _select_threshold(validation_source[target], validation_prob, settings)
    else:
        source_prob = _positive_probability(model, source_fit[list(features)])
        threshold, validation_balanced = _select_threshold(source_fit[target], source_prob, settings)
    test_probability = _positive_probability(model, test[list(features)])
    metrics = _binary_metrics(test[target], test_probability, threshold, origin="inner_validation")
    return test_probability, float(threshold), validation_balanced, metrics


def _walk_forward(rows: list[dict[str, Any]], settings: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return [], []
    frame = _rolling_regime_source(frame, settings)
    first_test_year = int(settings.get("first_test_year") or max(int(frame["year"].min()) + 2, 2022))
    last_test_year = int(frame["year"].max())
    minimum_train = max(30, int(settings.get("min_train_rows") or 250))
    validation_share = min(max(float(settings.get("inner_validation_share") or 0.20), 0.10), 0.40)
    default_cash_threshold = float(settings.get("default_probability_threshold") or 0.65)
    default_rotation_threshold = float(settings.get("default_rotation_probability_threshold") or default_cash_threshold)
    arbitration_margin = float(settings.get("action_probability_margin") or 0.05)
    predictions: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []

    for test_year in range(first_test_year, last_test_year + 1):
        train = frame[frame["year"] < test_year].copy()
        test = frame[frame["year"] == test_year].copy()
        if len(train) < minimum_train or test.empty:
            continue
        validation_rows = max(20, int(round(len(train) * validation_share)))
        validation_rows = min(validation_rows, max(1, len(train) // 3))
        fit = train.iloc[:-validation_rows].copy() if validation_rows < len(train) else train.copy()
        validation = train.iloc[-validation_rows:].copy() if validation_rows < len(train) else train.iloc[0:0].copy()
        if len(fit) < max(20, minimum_train // 2):
            fit = train.copy()
            validation = train.iloc[0:0].copy()

        train, fit, validation, test, regime_context = _regime_context_frames(
            train, fit, validation, test, settings
        )
        close_features = CLOSE_MODEL_FEATURES if regime_context.get("enabled") else CLOSE_FEATURES
        open_features = OPEN_MODEL_FEATURES if regime_context.get("enabled") else OPEN_FEATURES

        close_cash_prob, close_cash_threshold, close_cash_validation, close_cash_metrics = _binary_fold_model(
            train=train, fit=fit, validation=validation, test=test,
            features=close_features, target="target_cash", settings=settings,
            seed_offset=11 + test_year, default_threshold=default_cash_threshold, fit_on_full_train=False,
        )
        close_rotate_prob, close_rotate_threshold, close_rotate_validation, close_rotate_metrics = _binary_fold_model(
            train=train, fit=fit, validation=validation, test=test,
            features=close_features, target="target_rotate", settings=settings,
            seed_offset=17 + test_year, default_threshold=default_rotation_threshold, fit_on_full_train=False,
        )
        open_cash_prob, open_cash_threshold, open_cash_validation, open_cash_metrics = _binary_fold_model(
            train=train, fit=fit, validation=validation, test=test,
            features=open_features, target="target_cash", settings=settings,
            seed_offset=29 + test_year, default_threshold=default_cash_threshold, fit_on_full_train=True,
        )
        open_rotate_prob, open_rotate_threshold, open_rotate_validation, open_rotate_metrics = _binary_fold_model(
            train=train, fit=fit, validation=validation, test=test,
            features=open_features, target="target_rotate", settings=settings,
            seed_offset=37 + test_year, default_threshold=default_rotation_threshold, fit_on_full_train=True,
        )
        close_auc = _mean_available([_finite(close_cash_metrics.get("auc")), _finite(close_rotate_metrics.get("auc"))])
        open_auc = _mean_available([_finite(open_cash_metrics.get("auc")), _finite(open_rotate_metrics.get("auc"))])
        folds.append({
            "test_year": int(test_year),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "close_threshold": close_cash_threshold,
            "open_threshold": open_cash_threshold,
            "close_cash_threshold": close_cash_threshold,
            "close_rotate_threshold": close_rotate_threshold,
            "open_cash_threshold": open_cash_threshold,
            "open_rotate_threshold": open_rotate_threshold,
            "close_validation_balanced_accuracy": close_cash_validation,
            "open_validation_balanced_accuracy": open_cash_validation,
            "close_rotate_validation_balanced_accuracy": close_rotate_validation,
            "open_rotate_validation_balanced_accuracy": open_rotate_validation,
            "close_metrics": {"auc": close_auc, "cash": close_cash_metrics, "rotate": close_rotate_metrics},
            "open_metrics": {"auc": open_auc, "cash": open_cash_metrics, "rotate": open_rotate_metrics},
            "regime_context": regime_context,
        })
        for position, ((_, source), close_cash, close_rotate, open_cash, open_rotate) in enumerate(zip(
            test.iterrows(), close_cash_prob, close_rotate_prob, open_cash_prob, open_rotate_prob
        )):
            close_cash_signal = float(close_cash) >= float(close_cash_threshold)
            close_rotate_signal = float(close_rotate) >= float(close_rotate_threshold)
            if close_cash_signal and float(close_cash) >= float(close_rotate) + arbitration_margin:
                close_action = "CASH"
            elif close_rotate_signal and source.get("alternative_symbol") and float(close_rotate) >= float(close_cash) + arbitration_margin:
                close_action = "ROTATE"
            else:
                close_action = "FOLLOW_BASE"
            open_cash_signal = float(open_cash) >= float(open_cash_threshold)
            open_rotate_signal = float(open_rotate) >= float(open_rotate_threshold)
            if open_cash_signal and float(open_cash) >= float(open_rotate) + arbitration_margin:
                action = "CASH"
            elif open_rotate_signal and source.get("alternative_symbol") and float(open_rotate) >= float(open_cash) + arbitration_margin:
                action = "ROTATE"
            else:
                action = "FOLLOW_BASE"
            predictions.append({
                "execution_at": source["execution_at"],
                "decision_at": source["decision_at"],
                "test_year": int(test_year),
                "symbol": source["symbol"],
                "previous_asset": source.get("previous_asset"),
                "base_action": source.get("base_action"),
                "target_cash": int(source["target_cash"]),
                "target_rotate": int(source["target_rotate"]),
                "target_action": source.get("target_action"),
                "policy_action": action,
                "close_policy_action": close_action,
                "alternative_symbol": source.get("alternative_symbol"),
                "asset_utility": _finite(source.get("asset_utility")),
                "alternative_utility": _finite(source.get("alternative_utility")),
                "close_cash_probability": float(close_cash),
                "close_rotate_probability": float(close_rotate),
                "open_cash_probability": float(open_cash),
                "open_rotate_probability": float(open_rotate),
                "close_threshold": float(close_cash_threshold),
                "open_threshold": float(open_cash_threshold),
                "close_cash_threshold": float(close_cash_threshold),
                "close_rotate_threshold": float(close_rotate_threshold),
                "open_cash_threshold": float(open_cash_threshold),
                "open_rotate_threshold": float(open_rotate_threshold),
                "close_signal": bool(close_cash_signal or close_rotate_signal),
                "open_signal": bool(open_cash_signal or open_rotate_signal),
                "opening_changed_decision": bool(close_action != action),
                "close_return_1d": _finite(source.get("close_return_1d")),
                "close_return_robust_z": _finite(source.get("close_return_robust_z")),
                "opening_gap": _finite(source.get("opening_gap")),
                "opening_gap_robust_z": _finite(source.get("opening_gap_robust_z")),
                "alternative_opening_gap": _finite(source.get("alternative_opening_gap")),
                "alternative_opening_gap_robust_z": _finite(source.get("alternative_opening_gap_robust_z")),
                "shock_tail_score": _finite(source.get("shock_tail_score")),
                "statistical_close_shock": int(source.get("statistical_close_shock") or 0),
                "statistical_open_shock": int(source.get("statistical_open_shock") or 0),
                "opportunity_risk_conflict": int(source.get("opportunity_risk_conflict") or 0),
                "entry_rank_percentile": _finite(source.get("entry_rank_percentile")),
                "risk_adjusted_entry_score": _finite(source.get("risk_adjusted_entry_score")),
                "alternative_risk_adjusted_entry_score": _finite(source.get("alternative_risk_adjusted_entry_score")),
                "alternative_vs_base_risk_adjusted_gap": _finite(source.get("alternative_vs_base_risk_adjusted_gap")),
                "incumbent_risk_health": _finite(source.get("incumbent_risk_health")),
                "all_horizon_risk_safety": _finite(source.get("all_horizon_risk_safety")),
                "predicted_drawdown": _finite(source.get("predicted_drawdown")),
                "opportunity_risk_divergence": _finite(source.get("opportunity_risk_divergence")),
                "universe_breadth_20": _finite(source.get("universe_breadth_20")),
                "regime_cluster_id": None if source.get("regime_cluster_id") is None else int(source.get("regime_cluster_id")),
                "regime_quadrant": source.get("regime_quadrant"),
                "regime_pca_x": _finite(source.get("regime_pca_x")),
                "regime_pca_y": _finite(source.get("regime_pca_y")),
                "regime_nearest_distance": _finite(source.get("regime_nearest_distance")),
                "regime_second_distance": _finite(source.get("regime_second_distance")),
                "regime_distance_margin": _finite(source.get("regime_distance_margin")),
                "regime_healthy_distance": _finite(source.get("regime_healthy_distance")),
                "regime_danger_distance": _finite(source.get("regime_danger_distance")),
                "regime_healthy_similarity": _finite(source.get("regime_healthy_similarity")),
                "regime_danger_similarity": _finite(source.get("regime_danger_similarity")),
                "regime_danger_balance": _finite(source.get("regime_danger_balance")),
                "regime_is_defensive_cluster": int(source.get("regime_is_defensive_cluster") or 0),
                "regime_danger_approach_1d": _finite(source.get("regime_danger_approach_1d")),
                "regime_danger_approach_3d": _finite(source.get("regime_danger_approach_3d")),
                "regime_danger_approach_window": _finite(source.get("regime_danger_approach_window")),
                "regime_danger_balance_delta_1d": _finite(source.get("regime_danger_balance_delta_1d")),
                "regime_danger_balance_delta_3d": _finite(source.get("regime_danger_balance_delta_3d")),
                "regime_environment_deterioration_window": _finite(source.get("regime_environment_deterioration_window")),
                "regime_q4_persistence": _finite(source.get("regime_q4_persistence")),
                "regime_defensive_persistence": _finite(source.get("regime_defensive_persistence")),
                "regime_path_speed": _finite(source.get("regime_path_speed")),
                "regime_trajectory_score": _finite(source.get("regime_trajectory_score")),
                "regime_trajectory_warning": int(source.get("regime_trajectory_warning") or 0),
                **{f"regime_similarity_{idx}": _finite(source.get(f"regime_similarity_{idx}")) for idx in range(REGIME_MAX_FEATURE_CLUSTERS)},
                **{
                    f"asset_return_{int(horizon)}d": _finite(source.get(f"asset_return_{int(horizon)}d"))
                    for horizon in [int(value) for value in (settings.get("horizons_sessions") or [1, 3, 5])]
                },
                **{
                    f"alternative_return_{int(horizon)}d": _finite(source.get(f"alternative_return_{int(horizon)}d"))
                    for horizon in [int(value) for value in (settings.get("horizons_sessions") or [1, 3, 5])]
                },
            })
    return predictions, folds

def _attach_forward_strategy_risk_labels(
    reference_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    if not bool(settings.get("regime_trajectory_enabled", True)):
        return [dict(item) for item in predictions]
    horizon = max(1, int(settings.get("regime_trajectory_target_horizon_sessions") or 5))
    severe_threshold = float(settings.get("regime_trajectory_severe_loss_threshold") or -0.05)
    equity = _equity_path(reference_rows)
    by_day = {_day_key(row.get("stamp")): index for index, row in enumerate(equity)}
    output: list[dict[str, Any]] = []
    for item in predictions:
        row = dict(item)
        index = by_day.get(_day_key(row.get("execution_at")))
        if index is None or index + 1 >= len(equity):
            output.append(row)
            continue
        current = float(equity[index]["value"])
        future = equity[index + 1:min(len(equity), index + horizon + 1)]
        if current <= 0.0 or not future:
            output.append(row)
            continue
        future_returns = [float(point["value"]) / current - 1.0 for point in future]
        trough_offset = int(np.argmin(np.asarray(future_returns, dtype=float))) + 1
        minimum_return = float(min(future_returns))
        row["trajectory_forward_min_return"] = minimum_return
        row["trajectory_forward_return"] = float(future_returns[-1])
        row["trajectory_severe_event"] = int(minimum_return <= severe_threshold)
        row["trajectory_trough_lead_sessions"] = trough_offset
        output.append(row)
    return output


def _trajectory_binary_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [
        item for item in rows
        if _finite(item.get("regime_trajectory_score")) is not None and item.get("trajectory_severe_event") in {0, 1}
    ]
    if not usable:
        return {"rows": 0, "auc": None, "warnings": 0, "severe_windows": 0, "true_warnings": 0, "precision": None, "recall": None, "balanced_accuracy": None, "median_trough_lead_sessions": None, "mean_trough_lead_sessions": None}
    y_true = np.asarray([int(item["trajectory_severe_event"]) for item in usable], dtype=int)
    scores = np.asarray([float(item["regime_trajectory_score"]) for item in usable], dtype=float)
    warnings = np.asarray([int(item.get("regime_trajectory_warning") or 0) for item in usable], dtype=int)
    auc = _safe_auc(y_true, scores)
    warning_count = int(warnings.sum())
    severe_count = int(y_true.sum())
    true_warning_mask = (warnings == 1) & (y_true == 1)
    true_warnings = int(true_warning_mask.sum())
    precision = float(true_warnings / warning_count) if warning_count else None
    recall = float(true_warnings / severe_count) if severe_count else None
    balanced = float(balanced_accuracy_score(y_true, warnings)) if len(set(y_true.tolist())) > 1 else None
    leads = [
        int(usable[index].get("trajectory_trough_lead_sessions") or 0)
        for index, matched in enumerate(true_warning_mask.tolist())
        if matched and int(usable[index].get("trajectory_trough_lead_sessions") or 0) > 0
    ]
    return {
        "rows": int(len(usable)),
        "auc": auc,
        "warnings": warning_count,
        "severe_windows": severe_count,
        "true_warnings": true_warnings,
        "precision": precision,
        "recall": recall,
        "balanced_accuracy": balanced,
        "median_trough_lead_sessions": float(np.median(leads)) if leads else None,
        "mean_trough_lead_sessions": float(np.mean(leads)) if leads else None,
    }


def _daily_regime_trajectory_diagnostic(
    predictions: list[dict[str, Any]],
    folds: list[dict[str, Any]],
    settings: dict[str, Any],
) -> dict[str, Any]:
    if not bool(settings.get("regime_trajectory_enabled", True)):
        return {"enabled": False, "decision_feature": False}
    overall = _trajectory_binary_metrics(predictions)
    fold_rows: list[dict[str, Any]] = []
    for fold in folds:
        year = int(fold.get("test_year") or 0)
        year_rows = [item for item in predictions if int(item.get("test_year") or 0) == year]
        metrics = _trajectory_binary_metrics(year_rows)
        trajectory_contract = ((fold.get("regime_context") or {}).get("trajectory") or {})
        fold_rows.append({
            "test_year": year,
            **metrics,
            "warning_threshold": _finite(trajectory_contract.get("warning_threshold")),
            "warning_quantile": _finite(trajectory_contract.get("warning_quantile")),
        })

    quadrant_rows: list[dict[str, Any]] = []
    for quadrant in ("Q1", "Q2", "Q3", "Q4"):
        subset = [item for item in predictions if str(item.get("regime_quadrant") or "Q0") == quadrant]
        metrics = _trajectory_binary_metrics(subset)
        quadrant_rows.append({"quadrant": quadrant, **metrics})

    top_warnings = sorted(
        [item for item in predictions if int(item.get("regime_trajectory_warning") or 0) == 1],
        key=lambda item: float(item.get("regime_trajectory_score") or 0.0),
        reverse=True,
    )[:40]
    warning_rows = [{
        "execution_at": item.get("execution_at"),
        "test_year": item.get("test_year"),
        "symbol": item.get("symbol"),
        "regime_cluster_id": item.get("regime_cluster_id"),
        "regime_quadrant": item.get("regime_quadrant"),
        "trajectory_score": item.get("regime_trajectory_score"),
        "danger_similarity": item.get("regime_danger_similarity"),
        "danger_approach_3d": item.get("regime_danger_approach_3d"),
        "danger_approach_window": item.get("regime_danger_approach_window"),
        "q4_persistence": item.get("regime_q4_persistence"),
        "defensive_persistence": item.get("regime_defensive_persistence"),
        "forward_min_return": item.get("trajectory_forward_min_return"),
        "severe_event": item.get("trajectory_severe_event"),
        "trough_lead_sessions": item.get("trajectory_trough_lead_sessions"),
    } for item in top_warnings]
    return {
        "enabled": True,
        "shadow_only": True,
        "decision_feature": False,
        "decision_policy_changed": False,
        "method": "daily causal regime trajectory from training-only fold centroids and rolling session state",
        "trajectory_window_sessions": int(settings.get("regime_trajectory_window_sessions") or 5),
        "target_horizon_sessions": int(settings.get("regime_trajectory_target_horizon_sessions") or 5),
        "severe_loss_threshold": float(settings.get("regime_trajectory_severe_loss_threshold") or -0.05),
        "warning_quantile": float(settings.get("regime_trajectory_warning_quantile") or 0.90),
        "overall": overall,
        "folds": fold_rows,
        "quadrants": quadrant_rows,
        "top_warnings": warning_rows,
    }


def _replay(reference_rows: list[dict[str, Any]], predictions: list[dict[str, Any]], settings: dict[str, Any]) -> dict[str, Any]:
    equity = _equity_path(reference_rows)
    if not equity:
        return {}
    predictions_by_day = {
        _day_key(row.get("execution_at")): row
        for row in predictions
        if str(row.get("policy_action") or "FOLLOW_BASE") in {"CASH", "ROTATE"}
    }
    factors_by_day: dict[str, float] = {}
    interventions: list[dict[str, Any]] = []
    for day, prediction in predictions_by_day.items():
        base_return = _finite(prediction.get("asset_return_1d"))
        if not day or base_return is None or 1.0 + base_return <= 1e-9:
            continue
        action = str(prediction.get("policy_action") or "FOLLOW_BASE")
        if action == "CASH":
            candidate_return = 0.0
        elif action == "ROTATE":
            candidate_return = _finite(prediction.get("alternative_return_1d"))
            if candidate_return is None or 1.0 + candidate_return <= 1e-9:
                continue
        else:
            continue
        factor = float((1.0 + float(candidate_return)) / (1.0 + float(base_return)))
        factors_by_day[day] = factor
        interventions.append({
            **prediction,
            "base_session_return": float(base_return),
            "candidate_session_return": float(candidate_return),
            "capital_factor_vs_base": factor,
        })

    initial = float(equity[0]["starting_value"])
    base_path: list[dict[str, Any]] = []
    candidate_path: list[dict[str, Any]] = []
    cumulative = 1.0
    for index, row in enumerate(equity):
        day = _day_key(row["stamp"])
        base_path.append({"timestamp": row["timestamp"], "value": float(row["value"]), "starting_value": initial})
        candidate_path.append({"timestamp": row["timestamp"], "value": float(row["value"] * cumulative), "starting_value": initial})
        if day in factors_by_day and index + 1 < len(equity):
            cumulative *= factors_by_day[day]
    base = _path_stats(base_path)
    candidate = _path_stats(candidate_path)
    base_end = _finite(base.get("ending_capital"))
    candidate_end = _finite(candidate.get("ending_capital"))
    monthly_map_base = {row["month"]: row["return"] for row in (base.get("monthly_returns") or [])}
    monthly_map_candidate = {row["month"]: row["return"] for row in (candidate.get("monthly_returns") or [])}
    monthly: list[dict[str, Any]] = []
    action_counts_by_month: dict[str, Counter] = {}
    for item in interventions:
        month = (_day_key(item.get("execution_at")) or "")[:7]
        action_counts_by_month.setdefault(month, Counter())[str(item.get("policy_action") or "FOLLOW_BASE")] += 1
    for month in sorted(set(monthly_map_base) | set(monthly_map_candidate)):
        base_return = monthly_map_base.get(month)
        candidate_return = monthly_map_candidate.get(month)
        counts = action_counts_by_month.get(month, Counter())
        monthly.append({
            "month": month,
            "control_return": base_return,
            "candidate_return": candidate_return,
            "delta_return": None if base_return is None or candidate_return is None else float(candidate_return - base_return),
            "cash_interventions": int(counts.get("CASH", 0)),
            "rotation_interventions": int(counts.get("ROTATE", 0)),
        })
    return {
        "method": "daily_market_open_hold_rotate_cash_shadow_arbitration",
        "interventions": len(interventions),
        "intervention_rows": interventions,
        "control": base,
        "candidate": {
            **candidate,
            "ending_capital_delta": None if base_end is None or candidate_end is None else float(candidate_end - base_end),
            "ending_capital_delta_rate": None if base_end in {None, 0.0} or candidate_end is None else float(candidate_end / base_end - 1.0),
        },
        "monthly": monthly,
    }

def _gate(name: str, passed: bool, observed: Any, requirement: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "observed": observed, "requirement": requirement}


def build_analysis(
    *,
    reference_rows: list[dict[str, Any]],
    observation_rows: list[dict[str, Any]],
    settings: dict[str, Any],
    run_id: str,
    processing_id: str,
    period_start: str,
    period_end: str,
) -> dict[str, Any]:
    rows = _prepare_rows(reference_rows, observation_rows, settings)
    if not rows:
        raise ValueError("Statistical & Predictive Controls require Winner reference rows and Temporal daily asset observations with open prices.")
    predictions, folds = _walk_forward(rows, settings)
    if not predictions:
        raise ValueError("Statistical & Predictive Controls could not produce chronological out-of-sample predictions.")
    predictions = _attach_forward_strategy_risk_labels(reference_rows, predictions, settings)
    daily_regime_trajectory = _daily_regime_trajectory_diagnostic(predictions, folds, settings)
    replay = _replay(reference_rows, predictions, settings)
    control = replay.get("control") if isinstance(replay.get("control"), dict) else {}
    candidate = replay.get("candidate") if isinstance(replay.get("candidate"), dict) else {}
    capital_lift = _finite(candidate.get("ending_capital_delta_rate")) or 0.0
    interventions = int(replay.get("interventions") or 0)
    open_aucs = [_finite(((fold.get("open_metrics") or {}).get("auc"))) for fold in folds]
    open_aucs = [value for value in open_aucs if value is not None]
    close_aucs = [_finite(((fold.get("close_metrics") or {}).get("auc"))) for fold in folds]
    close_aucs = [value for value in close_aucs if value is not None]
    open_cash_aucs = [_finite((((fold.get("open_metrics") or {}).get("cash") or {}).get("auc"))) for fold in folds]
    open_cash_aucs = [value for value in open_cash_aucs if value is not None]
    open_rotate_aucs = [_finite((((fold.get("open_metrics") or {}).get("rotate") or {}).get("auc"))) for fold in folds]
    open_rotate_aucs = [value for value in open_rotate_aucs if value is not None]
    close_cash_aucs = [_finite((((fold.get("close_metrics") or {}).get("cash") or {}).get("auc"))) for fold in folds]
    close_cash_aucs = [value for value in close_cash_aucs if value is not None]
    close_rotate_aucs = [_finite((((fold.get("close_metrics") or {}).get("rotate") or {}).get("auc"))) for fold in folds]
    close_rotate_aucs = [value for value in close_rotate_aucs if value is not None]
    mean_open_auc = _mean_available(open_aucs)
    mean_close_auc = _mean_available(close_aucs)
    mean_open_cash_auc = _mean_available(open_cash_aucs)
    mean_open_rotate_auc = _mean_available(open_rotate_aucs)
    mean_close_cash_auc = _mean_available(close_cash_aucs)
    mean_close_rotate_auc = _mean_available(close_rotate_aucs)
    control_dd = _finite(control.get("maximum_drawdown"))
    candidate_dd = _finite(candidate.get("maximum_drawdown"))
    control_worst = _finite(((control.get("worst_month") or {}).get("return")))
    candidate_worst = _finite(((candidate.get("worst_month") or {}).get("return")))
    min_interventions = int(settings.get("min_interventions") or 5)
    min_capital_lift = float(settings.get("min_capital_lift") or 0.02)
    min_mean_auc = float(settings.get("min_mean_open_auc") or 0.55)
    max_dd_degradation = float(settings.get("max_drawdown_degradation") or 0.01)
    max_worst_month_degradation = float(settings.get("max_worst_month_degradation") or 0.01)
    yearly_deltas: dict[int, float] = Counter()
    for item in replay.get("monthly") or []:
        if not isinstance(item, dict):
            continue
        month = str(item.get("month") or "")
        delta = _finite(item.get("delta_return"))
        if month and delta is not None:
            yearly_deltas[int(month[:4])] += delta
    positive_years = sum(1 for value in yearly_deltas.values() if value > 0)
    min_positive_years = int(settings.get("min_positive_oos_years") or 2)
    gates = [
        _gate("capital_lift", capital_lift >= min_capital_lift, capital_lift, f">= {min_capital_lift:.2%}"),
        _gate("minimum_interventions", interventions >= min_interventions, interventions, f">= {min_interventions}"),
        _gate("open_cash_auc", mean_open_cash_auc is not None and mean_open_cash_auc >= min_mean_auc, mean_open_cash_auc, f">= {min_mean_auc:.3f}"),
        _gate("open_rotate_auc", mean_open_rotate_auc is not None and mean_open_rotate_auc >= min_mean_auc, mean_open_rotate_auc, f">= {min_mean_auc:.3f}"),
        _gate("positive_oos_years", positive_years >= min_positive_years, positive_years, f">= {min_positive_years}"),
        _gate(
            "drawdown_safety",
            candidate_dd is not None and control_dd is not None and candidate_dd >= control_dd - max_dd_degradation,
            None if candidate_dd is None or control_dd is None else float(candidate_dd - control_dd),
            f">= -{max_dd_degradation:.2%}",
        ),
        _gate(
            "worst_month_safety",
            candidate_worst is not None and control_worst is not None and candidate_worst >= control_worst - max_worst_month_degradation,
            None if candidate_worst is None or control_worst is None else float(candidate_worst - control_worst),
            f">= -{max_worst_month_degradation:.2%}",
        ),
    ]
    approved = all(bool(item["passed"]) for item in gates)
    action_counts = Counter(str(item.get("policy_action") or "FOLLOW_BASE") for item in predictions)
    cash_interventions = int(action_counts.get("CASH", 0))
    rotation_interventions = int(action_counts.get("ROTATE", 0))
    opening_changes = sum(bool(item.get("opening_changed_decision")) for item in predictions)
    statistical_close_shocks = sum(bool(item.get("statistical_close_shock")) for item in predictions)
    statistical_open_shocks = sum(bool(item.get("statistical_open_shock")) for item in predictions)
    opportunity_risk_conflicts = sum(bool(item.get("opportunity_risk_conflict")) for item in predictions)
    regime_silhouettes = [
        _finite(((fold.get("regime_context") or {}).get("silhouette_score"))) for fold in folds
        if (fold.get("regime_context") or {}).get("enabled")
    ]
    regime_silhouettes = [value for value in regime_silhouettes if value is not None]
    mean_regime_silhouette = _mean_available(regime_silhouettes)
    regime_quadrant_counts = Counter(str(item.get("regime_quadrant") or "Q0") for item in predictions)
    regime_action_counts: dict[str, Counter] = {}
    for item in predictions:
        quadrant = str(item.get("regime_quadrant") or "Q0")
        regime_action_counts.setdefault(quadrant, Counter())[str(item.get("policy_action") or "FOLLOW_BASE")] += 1
    extreme_rows = sorted(
        [row for row in predictions if _finite(row.get("shock_tail_score")) is not None],
        key=lambda row: float(row.get("shock_tail_score") or 0.0),
        reverse=True,
    )[:30]
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "status": "completed",
        "run_id": str(run_id),
        "processing_id": str(processing_id),
        "period_start": str(period_start),
        "period_end": str(period_end),
        "protocol": {
            "purpose": "causal close/open HOLD-ROTATE-CASH arbitration before operational Strategy activation",
            "close_checkpoint": "uses information available at the completed decision close only",
            "open_checkpoint": "adds only the next regular-session opening price and gap before execution",
            "features_include": ["robust time-series shocks", "cross-sectional shocks", "opportunity-risk divergence", "existing Temporal risk/quality signals", "best risk-adjusted alternative versus base asset", "causal rolling regime distances and PCA quadrant context"],
            "regime_context": "rolling state; clustering/scaler/PCA fitted only on years before each outer test year",
            "daily_regime_trajectory": "shadow diagnostic of regime direction, speed and persistence; not used by HOLD/ROTATE/CASH models in this release",
            "daily_regime_trajectory_used_as_decision_feature": False,
            "monthly_diagnostic_cluster_used_as_feature": False,
            "candidate_actions": ["FOLLOW_BASE", "ROTATE", "CASH"],
            "rotation_candidate_selection": "best causal risk-adjusted alternative at the completed close",
            "chronological_validation": "expanding walk-forward by test year",
            "threshold_selection": "inner chronological validation only",
            "future_information_in_features": False,
            "future_information_in_labels": True,
            "strategy_decisions_changed": False,
            "shadow_only": True,
            "cash_return_assumption": 0.0,
            "settings": settings,
        },
        "decision": {
            "status": "approved" if approved else "rejected",
            "operationalize_next_release": bool(approved),
            "fallback_if_rejected": "preserve_original_strategy",
            "reason": "All statistical and predictive qualification gates passed." if approved else "At least one statistical or predictive gate failed; preserve the original Strategy.",
        },
        "summary": {
            "research_rows": len(rows),
            "oos_predictions": len(predictions),
            "actions": {action: int(action_counts.get(action, 0)) for action in ACTIONS},
            "cash_interventions": cash_interventions,
            "rotation_interventions": rotation_interventions,
            "total_interventions": interventions,
            "opening_changed_decision_count": int(opening_changes),
            "statistical_close_shock_count": int(statistical_close_shocks),
            "statistical_open_shock_count": int(statistical_open_shocks),
            "opportunity_risk_conflict_count": int(opportunity_risk_conflicts),
            "regime_context_enabled": bool(settings.get("regime_context_enabled", True)),
            "mean_regime_silhouette": mean_regime_silhouette,
            "regime_quadrant_counts": {key: int(value) for key, value in sorted(regime_quadrant_counts.items())},
            "actions_by_regime_quadrant": {
                quadrant: {action: int(counts.get(action, 0)) for action in ACTIONS}
                for quadrant, counts in sorted(regime_action_counts.items())
            },
            "daily_regime_trajectory_auc": ((daily_regime_trajectory.get("overall") or {}).get("auc")),
            "daily_regime_warning_count": int(((daily_regime_trajectory.get("overall") or {}).get("warnings") or 0)),
            "daily_regime_severe_window_count": int(((daily_regime_trajectory.get("overall") or {}).get("severe_windows") or 0)),
            "daily_regime_true_warning_count": int(((daily_regime_trajectory.get("overall") or {}).get("true_warnings") or 0)),
            "daily_regime_warning_precision": ((daily_regime_trajectory.get("overall") or {}).get("precision")),
            "daily_regime_warning_recall": ((daily_regime_trajectory.get("overall") or {}).get("recall")),
            "daily_regime_median_lead_sessions": ((daily_regime_trajectory.get("overall") or {}).get("median_trough_lead_sessions")),
            "mean_close_auc": mean_close_auc,
            "mean_open_auc": mean_open_auc,
            "mean_close_cash_auc": mean_close_cash_auc,
            "mean_open_cash_auc": mean_open_cash_auc,
            "mean_close_rotate_auc": mean_close_rotate_auc,
            "mean_open_rotate_auc": mean_open_rotate_auc,
            "control_ending_capital": control.get("ending_capital"),
            "candidate_ending_capital": candidate.get("ending_capital"),
            "ending_capital_delta": candidate.get("ending_capital_delta"),
            "ending_capital_delta_rate": candidate.get("ending_capital_delta_rate"),
            "control_maximum_drawdown": control.get("maximum_drawdown"),
            "candidate_maximum_drawdown": candidate.get("maximum_drawdown"),
            "control_worst_month": control.get("worst_month"),
            "candidate_worst_month": candidate.get("worst_month"),
        },
        "gates": gates,
        "folds": folds,
        "monthly": replay.get("monthly") or [],
        "extreme_sessions": extreme_rows,
        "daily_regime_trajectory": daily_regime_trajectory,
        "predictions": predictions,
        "replay": {
            "method": replay.get("method"),
            "interventions": replay.get("interventions"),
            "control": replay.get("control"),
            "candidate": replay.get("candidate"),
        },
    }
