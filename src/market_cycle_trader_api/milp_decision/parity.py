from __future__ import annotations

from typing import Any

from .utils import as_datetime, as_float


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


def counts_cash_transitions_as_rotations(reference: dict[str, Any]) -> bool:
    protocol = reference.get("protocol") if isinstance(reference.get("protocol"), dict) else {}
    semantics = str(protocol.get("rotation_semantics") or "").strip().lower()
    if semantics == "all_position_changes":
        return True
    if semantics == "invested_asset_to_invested_asset":
        return False
    processing_kind = str(reference.get("processing_kind") or "").strip().lower()
    return processing_kind in {
        "strategy_research_temporal",
        "strategy_research_stateful",
        "strategy_research_decision_optimization",
    }


def reference_path(reference: dict[str, Any]) -> dict[str, Any]:
    equity = [dict(row) for row in (reference.get("equity") or []) if isinstance(row, dict)]
    equity = [row for row in equity if as_datetime(row.get("timestamp")) is not None and as_float(row.get("simulation_equity")) is not None]
    equity.sort(key=lambda row: as_datetime(row.get("timestamp")))
    if not equity:
        raise ValueError("Selected Strategy Research reference equity is unavailable for MILP parity.")

    rotations = [dict(row) for row in (reference.get("rotations") or []) if isinstance(row, dict)]
    rotations = [row for row in rotations if as_datetime(row.get("executed_at")) is not None]
    rotations.sort(key=lambda row: as_datetime(row.get("executed_at")))

    first_timestamp = as_datetime(equity[0].get("timestamp"))
    first_rotation = next(
        (row for row in rotations if as_datetime(row.get("executed_at")) == first_timestamp),
        None,
    )
    current = str((first_rotation or {}).get("from_asset") or equity[0].get("selected_asset") or "CASH").upper() or "CASH"
    initial_previous = current
    rotation_index = 0
    assets: list[str] = []
    values: list[float] = []
    timestamps: list[Any] = []

    for row in equity:
        timestamp = as_datetime(row.get("timestamp"))
        if timestamp is None:
            continue
        while rotation_index < len(rotations):
            rotation_timestamp = as_datetime(rotations[rotation_index].get("executed_at"))
            if rotation_timestamp is None or rotation_timestamp > timestamp:
                break
            current = str(rotations[rotation_index].get("to_asset") or "CASH").upper() or "CASH"
            rotation_index += 1
        explicit = str(row.get("selected_asset") or "").strip().upper()
        asset = explicit or current or "CASH"
        current = asset
        assets.append(asset)
        values.append(float(as_float(row.get("simulation_equity"))))
        timestamps.append(timestamp)

    if not values or len(values) != len(assets):
        raise ValueError("Selected Strategy Research reference path is incomplete for MILP parity.")
    return {
        "timestamps": timestamps,
        "equity": values,
        "assets": assets,
        "initial_previous_asset": initial_previous,
        "count_cash_transitions_as_rotations": counts_cash_transitions_as_rotations(reference),
    }


def compare(reference: dict[str, Any], replay_metrics: dict[str, Any]) -> dict[str, Any]:
    metrics = reference.get("metrics") if isinstance(reference.get("metrics"), dict) else {}
    normalized = reference_path(reference)
    reference_equity = normalized["equity"]
    reference_assets = normalized["assets"]
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
        for index, replay in enumerate(replay_equity):
            source_asset = str(reference_assets[index] or "CASH").upper()
            replay_asset = str(replay.get("selected_asset") or "CASH").upper()
            if source_asset != replay_asset:
                decision_path_match = False
            source_value = reference_equity[index]
            replay_value = as_float(replay.get("simulation_equity"))
            if source_value in {None, 0.0} or replay_value is None:
                equity_curve_match = False
                continue
            delta = abs(float(replay_value) / float(source_value) - 1.0)
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
