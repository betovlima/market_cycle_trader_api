from __future__ import annotations

from collections import Counter, defaultdict, deque
import math
from typing import Any, Iterable

from .config import (
    FALLBACK_THRESHOLDS,
    MIN_HISTORY,
    QUALITY_PENALTIES,
    RECENT_ROTATION_WINDOW,
    SCHEMA_VERSION,
    STATE_HEALTHY,
    STATE_NO_OPPORTUNITY,
    STATES,
    STATE_WEAK,
    STATE_WHIPSAW,
    TRAILING_WINDOW,
)


def _number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _date_key(value: Any) -> str:
    text = str(value or "")
    return text[:10] if len(text) >= 10 else text


def _month_key(value: Any) -> str:
    key = _date_key(value)
    return key[:7] if len(key) >= 7 else ""


def _quantile(values: Iterable[float], q: float) -> float | None:
    ordered = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(q)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _threshold(history: deque[float], feature: str, q: float) -> float:
    suffix = f"q{int(round(q * 100))}"
    fallback = float(FALLBACK_THRESHOLDS[f"{feature}:{suffix}"])
    if len(history) < MIN_HISTORY:
        return fallback
    current = _quantile(history, q)
    return fallback if current is None else float(current)


def _forward_return(equity: list[float | None], index: int, horizon: int) -> float | None:
    if index < 0 or index + horizon >= len(equity):
        return None
    start = equity[index]
    end = equity[index + horizon]
    if start is None or end is None or start <= 0:
        return None
    return float(end / start - 1.0)


def _mean(values: Iterable[Any]) -> float | None:
    parsed = [_number(value) for value in values]
    clean = [value for value in parsed if value is not None]
    return float(sum(clean) / len(clean)) if clean else None


def _state_summary(rows: list[dict[str, Any]], state: str) -> dict[str, Any]:
    subset = [row for row in rows if row.get("state") == state]
    return {
        "state": state,
        "sessions": len(subset),
        "share": (len(subset) / len(rows)) if rows else 0.0,
        "average_quality_score": _mean(row.get("quality_score") for row in subset),
        "average_forward_return_1": _mean(row.get("realized_forward_return_1") for row in subset),
        "average_forward_return_5": _mean(row.get("realized_forward_return_5") for row in subset),
        "average_forward_return_10": _mean(row.get("realized_forward_return_10") for row in subset),
        "positive_forward_5_rate": (
            sum(1 for row in subset if (_number(row.get("realized_forward_return_5")) or 0.0) > 0.0)
            / max(1, sum(1 for row in subset if _number(row.get("realized_forward_return_5")) is not None))
        ),
        "severe_forward_5_rate": (
            sum(1 for row in subset if (_number(row.get("realized_forward_return_5")) or 0.0) <= -0.05)
            / max(1, sum(1 for row in subset if _number(row.get("realized_forward_return_5")) is not None))
        ),
    }


def build_analysis(
    winner_rows: list[dict[str, Any]],
    selected_asset_features: dict[str, dict[str, Any]],
    *,
    run_id: str,
    processing_id: str,
    period_start: str,
    period_end: str,
) -> dict[str, Any]:
    period_rows = [
        dict(row)
        for row in winner_rows
        if period_start <= _month_key(row.get("timestamp")) <= period_end
    ]
    period_rows.sort(key=lambda row: str(row.get("timestamp") or ""))
    if not period_rows:
        raise ValueError("Leadership Regime Analysis requires Strategy reference daily rows in the selected period.")

    histories: dict[str, deque[float]] = {
        feature: deque(maxlen=TRAILING_WINDOW)
        for feature in (
            "universe_breadth_20", "breadth_impulse", "spy_realized_volatility_20", "best_vs_second_gap",
            "position_drawdown_from_peak", "score_change_from_entry", "incumbent_risk_health",
            "all_horizon_risk_safety", "positive_score_share", "best_score_zscore",
        )
    }
    recent_rotations: deque[int] = deque(maxlen=RECENT_ROTATION_WINDOW)
    session_rows: list[dict[str, Any]] = []
    equity = [_number(row.get("strategy_equity")) for row in period_rows]

    for index, source in enumerate(period_rows):
        timestamp = source.get("timestamp")
        day_key = _date_key(timestamp)
        selected_asset = str(source.get("selected_asset") or source.get("strategy_research_control_asset") or "CASH")
        asset_features = selected_asset_features.get(f"{day_key}|{selected_asset}", {})

        breadth_5 = _number(source.get("universe_breadth_5"))
        breadth_20 = _number(source.get("universe_breadth_20"))
        breadth_impulse = (breadth_5 - breadth_20) if breadth_5 is not None and breadth_20 is not None else None
        positive_count = _number(source.get("positive_score_count"))
        finite_count = _number(source.get("finite_score_count"))
        positive_share = (positive_count / finite_count) if positive_count is not None and finite_count not in {None, 0.0} else None
        features = {
            "universe_breadth_20": breadth_20,
            "breadth_impulse": breadth_impulse,
            "spy_realized_volatility_20": _number(source.get("spy_realized_volatility_20")),
            "best_vs_second_gap": _number(source.get("best_vs_second_gap")),
            "position_drawdown_from_peak": _number(source.get("position_drawdown_from_peak")),
            "score_change_from_entry": _number(source.get("score_change_from_entry")),
            "incumbent_risk_health": _number(asset_features.get("incumbent_risk_health")),
            "all_horizon_risk_safety": _number(asset_features.get("all_horizon_risk_safety")),
            "positive_score_share": positive_share,
            "best_score_zscore": _number(source.get("best_score_zscore")),
        }
        thresholds = {
            "breadth_20_low": _threshold(histories["universe_breadth_20"], "universe_breadth_20", 0.35),
            "breadth_impulse_low": _threshold(histories["breadth_impulse"], "breadth_impulse", 0.25),
            "volatility_high": _threshold(histories["spy_realized_volatility_20"], "spy_realized_volatility_20", 0.75),
            "leader_gap_low": _threshold(histories["best_vs_second_gap"], "best_vs_second_gap", 0.35),
            "position_drawdown_low": _threshold(histories["position_drawdown_from_peak"], "position_drawdown_from_peak", 0.25),
            "score_change_low": _threshold(histories["score_change_from_entry"], "score_change_from_entry", 0.35),
            "risk_health_low": _threshold(histories["incumbent_risk_health"], "incumbent_risk_health", 0.35),
            "risk_safety_low": _threshold(histories["all_horizon_risk_safety"], "all_horizon_risk_safety", 0.35),
            "positive_share_low": _threshold(histories["positive_score_share"], "positive_score_share", 0.35),
            "best_score_weak": _threshold(histories["best_score_zscore"], "best_score_zscore", 0.35),
        }

        def low(feature: str, threshold_key: str) -> bool:
            value = features.get(feature)
            return value is not None and value <= thresholds[threshold_key]

        flags = {
            "breadth_low": low("universe_breadth_20", "breadth_20_low"),
            "breadth_impulse_low": low("breadth_impulse", "breadth_impulse_low"),
            "volatility_high": (
                features["spy_realized_volatility_20"] is not None
                and features["spy_realized_volatility_20"] >= thresholds["volatility_high"]
            ),
            "leader_gap_low": low("best_vs_second_gap", "leader_gap_low"),
            "position_drawdown_low": low("position_drawdown_from_peak", "position_drawdown_low"),
            "score_change_low": low("score_change_from_entry", "score_change_low"),
            "risk_health_low": low("incumbent_risk_health", "risk_health_low"),
            "risk_safety_low": low("all_horizon_risk_safety", "risk_safety_low"),
            "positive_share_low": low("positive_score_share", "positive_share_low"),
            "best_score_weak": low("best_score_zscore", "best_score_weak"),
            "rotation_pressure": sum(recent_rotations) >= 2,
        }
        current_rank = _number(source.get("current_asset_rank"))
        best_asset = str(source.get("best_asset") or source.get("top_1_asset") or "")
        is_top1 = selected_asset == best_asset or (current_rank is not None and current_rank <= 1.0)
        no_opportunity_evidence = sum(int(flags[key]) for key in ("breadth_low", "positive_share_low", "risk_safety_low", "best_score_weak"))
        whipsaw_evidence = sum(int(flags[key]) for key in ("leader_gap_low", "breadth_impulse_low", "position_drawdown_low", "risk_health_low"))
        weak_evidence = sum(int(flags[key]) for key in ("breadth_impulse_low", "leader_gap_low", "position_drawdown_low", "score_change_low", "risk_health_low", "volatility_high"))

        if selected_asset.upper() == "CASH" or no_opportunity_evidence >= 3:
            state = STATE_NO_OPPORTUNITY
        elif flags["rotation_pressure"] and whipsaw_evidence >= 2:
            state = STATE_WHIPSAW
        elif is_top1 and weak_evidence >= 3:
            state = STATE_WEAK
        else:
            state = STATE_HEALTHY

        penalty = sum(QUALITY_PENALTIES[key] for key, active in flags.items() if active)
        quality_score = max(0.0, min(100.0, 100.0 - float(penalty)))
        history_samples = min((len(history) for history in histories.values()), default=0)
        session_rows.append({
            "timestamp": timestamp,
            "month": _month_key(timestamp),
            "fold_id": source.get("walk_forward_fold") or source.get("decision_fold_id"),
            "selected_asset": selected_asset,
            "best_asset": best_asset or None,
            "current_asset_rank": current_rank,
            "state": state,
            "quality_score": quality_score,
            "classification_history_samples": history_samples,
            "classification_confidence": min(1.0, history_samples / max(1, MIN_HISTORY)),
            "recent_rotations_10": sum(recent_rotations),
            "signals": flags,
            "features": {
                **features,
                "universe_breadth_5": breadth_5,
                "spy_return_5": _number(source.get("spy_return_5")),
                "spy_return_20": _number(source.get("spy_return_20")),
                "position_return_since_entry": _number(source.get("position_return_since_entry")),
                "short_profit_consensus": _number(asset_features.get("short_profit_consensus")),
                "long_profit_confirmation": _number(asset_features.get("long_profit_confirmation")),
                "horizon_agreement": _number(asset_features.get("horizon_agreement")),
            },
            "thresholds": thresholds,
            "realized_forward_return_1": _forward_return(equity, index, 1),
            "realized_forward_return_5": _forward_return(equity, index, 5),
            "realized_forward_return_10": _forward_return(equity, index, 10),
        })

        for feature, value in features.items():
            if value is not None:
                histories[feature].append(value)
        recent_rotations.append(1 if _bool(source.get("decision_is_rotation")) else 0)

    monthly: list[dict[str, Any]] = []
    by_month: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_by_month: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row, source in zip(session_rows, period_rows):
        by_month[row["month"]].append(row)
        source_by_month[row["month"]].append(source)
    for month in sorted(by_month):
        rows = by_month[month]
        source_rows = source_by_month[month]
        counts = Counter(row["state"] for row in rows)
        dominant_state, dominant_count = counts.most_common(1)[0]
        first_equity = _number(source_rows[0].get("strategy_equity"))
        last_equity = _number(source_rows[-1].get("strategy_equity"))
        monthly_return = (last_equity / first_equity - 1.0) if first_equity not in {None, 0.0} and last_equity is not None else None
        monthly.append({
            "month": month,
            "sessions": len(rows),
            "monthly_return": monthly_return,
            "dominant_state": dominant_state,
            "dominant_share": dominant_count / len(rows),
            "state_counts": {state: int(counts.get(state, 0)) for state in STATES},
            "state_shares": {state: counts.get(state, 0) / len(rows) for state in STATES},
            "average_quality_score": _mean(row.get("quality_score") for row in rows),
            "average_breadth_5": _mean((row.get("features") or {}).get("universe_breadth_5") for row in rows),
            "average_breadth_20": _mean((row.get("features") or {}).get("universe_breadth_20") for row in rows),
            "average_breadth_impulse": _mean((row.get("features") or {}).get("breadth_impulse") for row in rows),
            "average_volatility_20": _mean((row.get("features") or {}).get("spy_realized_volatility_20") for row in rows),
            "average_leader_gap": _mean((row.get("features") or {}).get("best_vs_second_gap") for row in rows),
            "average_position_drawdown": _mean((row.get("features") or {}).get("position_drawdown_from_peak") for row in rows),
            "average_risk_health": _mean((row.get("features") or {}).get("incumbent_risk_health") for row in rows),
            "rotation_pressure_sessions": sum(1 for row in rows if (row.get("signals") or {}).get("rotation_pressure")),
        })

    strong_months = [row for row in monthly if (_number(row.get("monthly_return")) or 0.0) >= 0.10]
    severe_loss_months = [row for row in monthly if (_number(row.get("monthly_return")) or 0.0) <= -0.10]

    def cohort_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
        session_count = sum(int(item.get("sessions") or 0) for item in items)
        state_counts = Counter()
        for item in items:
            state_counts.update(item.get("state_counts") or {})
        return {
            "months": len(items),
            "average_monthly_return": _mean(item.get("monthly_return") for item in items),
            "sessions": session_count,
            "state_shares": {
                state: (state_counts.get(state, 0) / session_count if session_count else 0.0)
                for state in STATES
            },
            "average_breadth_impulse": _mean(item.get("average_breadth_impulse") for item in items),
            "average_volatility_20": _mean(item.get("average_volatility_20") for item in items),
            "average_leader_gap": _mean(item.get("average_leader_gap") for item in items),
            "average_position_drawdown": _mean(item.get("average_position_drawdown") for item in items),
            "average_risk_health": _mean(item.get("average_risk_health") for item in items),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "run_id": str(run_id),
        "processing_id": str(processing_id),
        "period_start": str(period_start),
        "period_end": str(period_end),
        "method": {
            "name": "causal_adaptive_leadership_regime_diagnostics",
            "trailing_window_sessions": TRAILING_WINDOW,
            "minimum_history_sessions": MIN_HISTORY,
            "recent_rotation_window_sessions": RECENT_ROTATION_WINDOW,
            "uses_realized_returns_for_classification": False,
            "realized_returns_role": "post_hoc_validation_only",
            "states": list(STATES),
        },
        "summary": {
            "sessions": len(session_rows),
            "states": [_state_summary(session_rows, state) for state in STATES],
            "strong_months": cohort_summary(strong_months),
            "severe_loss_months": cohort_summary(severe_loss_months),
        },
        "monthly": monthly,
        "sessions": session_rows,
    }
