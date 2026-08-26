from __future__ import annotations

import math
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


def _asset(value: Any) -> str:
    text = str(value or "CASH").strip().upper()
    return text or "CASH"


def temporal_capital(run: dict[str, Any]) -> dict[str, Any]:
    result = run.get("result") if isinstance(run.get("result"), dict) else {}
    multi = result.get("multi_horizon_metrics") if isinstance(result.get("multi_horizon_metrics"), dict) else {}
    return multi.get("shadow_capital") if isinstance(multi.get("shadow_capital"), dict) else {}


def scoped_temporal_rows(
    temporal_curve: list[dict[str, Any]],
    *,
    start_month: str,
    end_month: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    all_rows = [dict(row) for row in temporal_curve if isinstance(row, dict)]
    all_rows.sort(key=lambda row: _stamp(row.get("decision_timestamp")) or pd.Timestamp.max.tz_localize("UTC"))
    selected: list[dict[str, Any]] = []
    first_index = -1
    for index, row in enumerate(all_rows):
        stamp = _stamp(row.get("decision_timestamp"))
        if stamp is None:
            continue
        month = stamp.strftime("%Y-%m")
        if start_month <= month <= end_month:
            if first_index < 0:
                first_index = index
            selected.append(row)
    return all_rows, selected, first_index


def infer_cash_rotation_policy(run: dict[str, Any], temporal_curve: list[dict[str, Any]]) -> bool:
    capital = temporal_capital(run)
    reported = int(capital.get("switch_count") or 0)
    all_transitions = 0
    asset_transitions = 0
    for row in temporal_curve:
        previous = _asset(row.get("current_symbol"))
        target = _asset(row.get("target_symbol"))
        if previous == target:
            continue
        all_transitions += 1
        if previous != "CASH" and target != "CASH":
            asset_transitions += 1
    if reported == all_transitions:
        return True
    if reported == asset_transitions:
        return False
    return False


def control_metrics(
    run: dict[str, Any],
    temporal_curve: list[dict[str, Any]],
    *,
    start_month: str,
    end_month: str,
) -> dict[str, Any]:
    capital = temporal_capital(run)
    all_rows, selected, first_index = scoped_temporal_rows(
        temporal_curve,
        start_month=start_month,
        end_month=end_month,
    )
    values = [float(value) for row in selected if (value := finite(row.get("strategy_equity"))) is not None]
    if not values or first_index < 0:
        return {}
    if first_index == 0:
        initial = finite(capital.get("initial_capital"))
    else:
        initial = finite(all_rows[first_index - 1].get("strategy_equity"))
    if initial in {None, 0.0}:
        return {}

    count_cash = infer_cash_rotation_policy(run, all_rows)
    switch_count = 0
    for row in selected:
        previous = _asset(row.get("current_symbol"))
        target = _asset(row.get("target_symbol"))
        if previous == target:
            continue
        if count_cash or (previous != "CASH" and target != "CASH"):
            switch_count += 1

    metrics = equity_metrics(values, float(initial))
    metrics.update({
        "switch_count": int(switch_count),
        "timing_override_count": int(sum(1 for row in selected if bool(row.get("temporal_timing_override")))),
        "exposure": sum(1 for row in selected if _asset(row.get("target_symbol")) != "CASH") / max(1, len(selected)),
        "cash_days": int(sum(1 for row in selected if _asset(row.get("target_symbol")) == "CASH")),
        "metric_scope": "frozen_temporal_economic_curve",
        "count_cash_transitions_as_rotations": bool(count_cash),
    })
    return metrics


def _close(left: Any, right: Any) -> bool:
    a = finite(left)
    b = finite(right)
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-8)


def validate_control_parity(
    run: dict[str, Any],
    temporal_curve: list[dict[str, Any]],
    parity_replay: dict[str, Any],
    *,
    start_month: str,
    end_month: str,
) -> dict[str, Any]:
    control = control_metrics(run, temporal_curve, start_month=start_month, end_month=end_month)
    replay_metrics = parity_replay.get("metrics") if isinstance(parity_replay.get("metrics"), dict) else {}
    _, selected, _ = scoped_temporal_rows(temporal_curve, start_month=start_month, end_month=end_month)
    replay_equity = parity_replay.get("equity") if isinstance(parity_replay.get("equity"), list) else []

    decision_path = len(selected) == len(replay_equity) and all(
        _asset(source.get("target_symbol")) == _asset(replayed.get("target_symbol"))
        for source, replayed in zip(selected, replay_equity)
    )
    equity_curve = len(selected) == len(replay_equity) and all(
        _close(source.get("strategy_equity"), replayed.get("equity"))
        for source, replayed in zip(selected, replay_equity)
    )
    checks = {
        "decision_path": bool(decision_path),
        "equity_curve": bool(equity_curve),
        "initial_capital": _close(control.get("initial_capital"), replay_metrics.get("initial_capital")),
        "ending_capital": _close(control.get("ending_capital"), replay_metrics.get("ending_capital")),
        "max_drawdown": _close(control.get("max_drawdown"), replay_metrics.get("max_drawdown")),
        "switch_count": int(control.get("switch_count") or 0) == int(replay_metrics.get("switch_count") or 0),
        "timing_override_count": int(control.get("timing_override_count") or 0) == int(replay_metrics.get("temporal_timing_override_count") or 0),
        "equity_sessions": int(control.get("equity_sessions") or 0) == int(replay_metrics.get("equity_sessions") or 0),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "control": control,
        "replay": replay_metrics,
    }


def threshold_stability(calibrations: list[dict[str, Any]], entry_horizons: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon in entry_horizons:
        values = [
            float(item["threshold"])
            for item in calibrations
            if item.get("eligible")
            and int(item.get("horizon") or 0) == int(horizon)
            and finite(item.get("threshold")) is not None
        ]
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
