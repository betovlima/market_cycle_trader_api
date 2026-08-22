from __future__ import annotations

from typing import Any

from .utils import as_float


def reference_analytics(db: Any, processing_id: str) -> dict[str, Any]:
    from ..services.analytics import processing_analytics

    payload = processing_analytics(db, str(processing_id))
    if not isinstance(payload, dict):
        raise ValueError("Selected Strategy Research reference analytics are unavailable for MILP parity.")
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else None
    equity = payload.get("equity") if isinstance(payload.get("equity"), list) else None
    if metrics is None or not equity:
        raise ValueError("Selected Strategy Research reference analytics are incomplete for MILP parity.")
    return payload


def compare(reference: dict[str, Any], replay_metrics: dict[str, Any]) -> dict[str, Any]:
    metrics = reference.get("metrics") if isinstance(reference.get("metrics"), dict) else {}
    reference_equity = [row for row in (reference.get("equity") or []) if isinstance(row, dict)]
    replay_equity = [row for row in (replay_metrics.get("equity") or []) if isinstance(row, dict)]

    reference_capital = as_float(metrics.get("ending_capital"))
    replay_capital = as_float(replay_metrics.get("ending_capital"))
    reference_exposure = as_float(metrics.get("market_exposure"))
    replay_exposure = as_float(replay_metrics.get("market_exposure"))
    reference_cash = int(metrics.get("cash_days") or 0)
    replay_cash = int(replay_metrics.get("cash_days") or 0)
    reference_switches = int(metrics.get("capital_rotations") or metrics.get("position_changes") or 0)
    replay_switches = int(replay_metrics.get("capital_rotations") or 0)
    reference_sessions = len(reference_equity)
    replay_sessions = len(replay_equity)

    capital_delta = (
        replay_capital / reference_capital - 1.0
        if replay_capital is not None and reference_capital not in {None, 0.0}
        else None
    )
    exposure_delta = (
        replay_exposure - reference_exposure
        if replay_exposure is not None and reference_exposure is not None
        else None
    )

    decision_path_match = reference_sessions == replay_sessions
    equity_curve_match = decision_path_match
    max_equity_delta_rate = 0.0
    if decision_path_match:
        for source, replay in zip(reference_equity, replay_equity):
            source_asset = str(source.get("selected_asset") or "CASH").upper()
            replay_asset = str(replay.get("selected_asset") or "CASH").upper()
            if source_asset != replay_asset:
                decision_path_match = False
            source_value = as_float(source.get("simulation_equity"))
            replay_value = as_float(replay.get("simulation_equity"))
            if source_value in {None, 0.0} or replay_value is None:
                equity_curve_match = False
                continue
            delta = abs(replay_value / source_value - 1.0)
            max_equity_delta_rate = max(max_equity_delta_rate, delta)
            if delta > 1e-10:
                equity_curve_match = False

    checks = {
        "ending_capital": capital_delta is not None and abs(capital_delta) <= 1e-10,
        "cash_days": replay_cash == reference_cash,
        "market_exposure": exposure_delta is not None and abs(exposure_delta) <= 1e-12,
        "switches": replay_switches == reference_switches,
        "equity_sessions": replay_sessions == reference_sessions,
        "decision_path": decision_path_match,
        "equity_curve": equity_curve_match,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "ending_capital_delta_rate": capital_delta,
        "market_exposure_delta": exposure_delta,
        "maximum_equity_delta_rate": max_equity_delta_rate,
        "reference": {
            "ending_capital": reference_capital,
            "cash_days": reference_cash,
            "market_exposure": reference_exposure,
            "switches": reference_switches,
            "equity_sessions": reference_sessions,
        },
        "replay": {
            "ending_capital": replay_capital,
            "cash_days": replay_cash,
            "market_exposure": replay_exposure,
            "switches": replay_switches,
            "equity_sessions": replay_sessions,
        },
    }
