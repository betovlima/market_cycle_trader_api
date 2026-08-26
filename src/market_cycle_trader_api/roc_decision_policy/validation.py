from __future__ import annotations

from typing import Any

import pandas as pd

from .metrics import equity_metrics, finite, metric_delta


def _stamp(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def control_metrics(run: dict[str, Any], *, start_month: str, end_month: str) -> dict[str, Any]:
    result = run.get("result") if isinstance(run.get("result"), dict) else {}
    multi = result.get("multi_horizon_metrics") if isinstance(result.get("multi_horizon_metrics"), dict) else {}
    capital = multi.get("shadow_capital") if isinstance(multi.get("shadow_capital"), dict) else {}
    economic_curve = [dict(row) for row in (capital.get("economic_curve") or []) if isinstance(row, dict)]
    economic_curve.sort(key=lambda row: _stamp(row.get("decision_timestamp")) or pd.Timestamp.max.tz_localize("UTC"))

    if economic_curve:
        selected: list[tuple[int, dict[str, Any]]] = []
        for index, row in enumerate(economic_curve):
            stamp = _stamp(row.get("decision_timestamp"))
            if stamp is None:
                continue
            month = stamp.strftime("%Y-%m")
            if start_month <= month <= end_month and finite(row.get("strategy_equity")) is not None:
                selected.append((index, row))
        if selected:
            first_index = selected[0][0]
            values = [float(finite(row.get("strategy_equity"))) for _, row in selected if finite(row.get("strategy_equity")) is not None]
            if first_index > 0:
                initial = finite(economic_curve[first_index - 1].get("strategy_equity"))
            else:
                initial = finite(capital.get("initial_capital"))
            if values and initial not in {None, 0.0}:
                scoped = equity_metrics(values, float(initial))
                scoped.update({
                    "switch_count": sum(
                        1
                        for _, row in selected
                        if str(row.get("action") or "").upper() in {"BUY", "SELL", "ROTATE"}
                    ),
                    "timing_override_count": sum(
                        1
                        for _, row in selected
                        if bool(row.get("temporal_timing_override"))
                    ),
                    "metric_scope": "selected_period_economic_curve",
                })
                return scoped

    return {
        "initial_capital": finite(capital.get("initial_capital")),
        "ending_capital": finite(capital.get("ending_capital")),
        "total_return": finite(capital.get("total_return")),
        "cagr": finite(capital.get("cagr")),
        "sharpe": finite(capital.get("sharpe")),
        "max_drawdown": finite(capital.get("max_drawdown")),
        "switch_count": capital.get("switch_count"),
        "timing_override_count": capital.get("timing_override_count"),
        "metric_scope": "full_temporal_result",
    }


def threshold_stability(calibrations: list[dict[str, Any]], entry_horizons: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon in entry_horizons:
        values = [float(item["threshold"]) for item in calibrations if item.get("eligible") and int(item.get("horizon") or 0) == int(horizon) and finite(item.get("threshold")) is not None]
        if not values:
            continue
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        rows.append({
            "horizon": int(horizon),
            "fold_count": int(len(values)),
            "threshold_mean": mean,
            "threshold_min": min(values),
            "threshold_max": max(values),
            "threshold_std": variance ** 0.5,
            "threshold_range": max(values) - min(values),
        })
    return rows


def build_comparison(challenger: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    delta = metric_delta(challenger, control)
    return {
        "control": control,
        "challenger": challenger,
        "delta": delta,
        "capital_improved": bool((delta.get("ending_capital") or 0.0) > 0.0),
        "sharpe_improved": bool((delta.get("sharpe") or 0.0) > 0.0),
        "drawdown_improved": bool((delta.get("max_drawdown") or 0.0) > 0.0),
    }
