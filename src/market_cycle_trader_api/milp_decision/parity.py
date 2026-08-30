from __future__ import annotations

from typing import Any

from .utils import as_datetime, as_float


_ROTATION_ALL_POSITION_CHANGES = "all_position_changes"
_ROTATION_INVESTED_ASSET_CHANGES = "invested_asset_to_invested_asset"


def _asset(value: Any) -> str:
    return str(value or "CASH").strip().upper() or "CASH"


def _rotation_increment(previous: Any, target: Any, *, count_cash_transitions: bool) -> int:
    previous_asset = _asset(previous)
    target_asset = _asset(target)
    if previous_asset == target_asset:
        return 0
    if not count_cash_transitions and "CASH" in {previous_asset, target_asset}:
        return 0
    return 1


def _reported_switches(reference: dict[str, Any]) -> int | None:
    metrics = reference.get("metrics") if isinstance(reference.get("metrics"), dict) else {}
    value = metrics.get("capital_rotations")
    if value is None:
        value = metrics.get("position_changes")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _rotation_semantics(reference: dict[str, Any]) -> str:
    protocol = reference.get("protocol") if isinstance(reference.get("protocol"), dict) else {}
    explicit = str(protocol.get("rotation_semantics") or reference.get("rotation_semantics") or "").strip().lower()
    if explicit in {_ROTATION_ALL_POSITION_CHANGES, _ROTATION_INVESTED_ASSET_CHANGES}:
        return explicit

    expected_switches = _reported_switches(reference)
    rotations = [row for row in (reference.get("rotations") or []) if isinstance(row, dict)]
    if expected_switches is not None and rotations:
        all_position_changes = sum(
            _rotation_increment(row.get("from_asset"), row.get("to_asset"), count_cash_transitions=True)
            for row in rotations
        )
        invested_asset_changes = sum(
            _rotation_increment(row.get("from_asset"), row.get("to_asset"), count_cash_transitions=False)
            for row in rotations
        )
        all_matches = all_position_changes == expected_switches
        invested_matches = invested_asset_changes == expected_switches
        if all_matches != invested_matches:
            return _ROTATION_ALL_POSITION_CHANGES if all_matches else _ROTATION_INVESTED_ASSET_CHANGES

    processing_kind = str(reference.get("processing_kind") or "").strip().lower()
    if processing_kind in {
        "strategy_research_temporal",
        "strategy_research_stateful",
        "strategy_research_decision_optimization",
    }:
        return _ROTATION_ALL_POSITION_CHANGES
    return _ROTATION_INVESTED_ASSET_CHANGES


def _path_switches(assets: list[str], initial_previous_asset: str, *, count_cash_transitions: bool) -> int:
    previous = _asset(initial_previous_asset)
    switches = 0
    for target in assets:
        target_asset = _asset(target)
        switches += _rotation_increment(previous, target_asset, count_cash_transitions=count_cash_transitions)
        previous = target_asset
    return switches


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
    return _rotation_semantics(reference) == _ROTATION_ALL_POSITION_CHANGES


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
    current = _asset((first_rotation or {}).get("from_asset") or equity[0].get("selected_asset") or "CASH")
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
            current = _asset(rotations[rotation_index].get("to_asset"))
            rotation_index += 1
        explicit = str(row.get("selected_asset") or "").strip().upper()
        asset = explicit or current or "CASH"
        current = asset
        assets.append(asset)
        values.append(float(as_float(row.get("simulation_equity"))))
        timestamps.append(timestamp)

    if not values or len(values) != len(assets):
        raise ValueError("Selected Strategy Research reference path is incomplete for MILP parity.")
    semantics = _rotation_semantics(reference)
    return {
        "timestamps": timestamps,
        "equity": values,
        "assets": assets,
        "initial_previous_asset": initial_previous,
        "rotation_semantics": semantics,
        "count_cash_transitions_as_rotations": semantics == _ROTATION_ALL_POSITION_CHANGES,
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
    reference_switches_reported = _reported_switches(reference)
    try:
        replay_switches_reported = int(replay_metrics.get("capital_rotations")) if replay_metrics.get("capital_rotations") is not None else None
    except (TypeError, ValueError):
        replay_switches_reported = None
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
    replay_assets: list[str] = []
    if decision_path_match:
        for index, replay in enumerate(replay_equity):
            source_asset = _asset(reference_assets[index])
            replay_asset = _asset(replay.get("selected_asset"))
            replay_assets.append(replay_asset)
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
    else:
        replay_assets = [_asset(row.get("selected_asset")) for row in replay_equity]

    count_cash_transitions = bool(normalized["count_cash_transitions_as_rotations"])
    initial_previous = _asset(normalized["initial_previous_asset"])
    reference_switches = _path_switches(
        reference_assets,
        initial_previous,
        count_cash_transitions=count_cash_transitions,
    )
    replay_switches = _path_switches(
        replay_assets,
        initial_previous,
        count_cash_transitions=count_cash_transitions,
    )

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
        "rotation_semantics": normalized["rotation_semantics"],
        "reference": {
            "ending_capital": reference_capital,
            "cash_days": reference_cash,
            "market_exposure": reference_exposure,
            "switches": reference_switches,
            "reported_switches": reference_switches_reported,
            "equity_sessions": reference_sessions,
        },
        "replay": {
            "ending_capital": replay_capital,
            "cash_days": replay_cash,
            "market_exposure": replay_exposure,
            "switches": replay_switches,
            "reported_switches": replay_switches_reported,
            "equity_sessions": replay_sessions,
        },
    }
