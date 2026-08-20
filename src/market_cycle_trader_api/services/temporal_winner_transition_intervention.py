from __future__ import annotations

import math
import uuid
from typing import Any

import pandas as pd

from ..infrastructure.persistence.mongo_repository import (
    TEMPORAL_WINNER_TRANSITION_INTERVENTION_RESEARCH_COLLECTION,
    TEMPORAL_WINNER_TRANSITION_CONFIDENCE_RESEARCH_COLLECTION,
    TEMPORAL_WINNER_TRANSITION_RISK_RESEARCH_COLLECTION,
    bson_value,
    utc_now,
)
from .analytics import processing_analytics
from .temporal_winner_transition_attribution import get_winner_transition_attribution
from .temporal_winner_transition_risk import run_transition_risk_search_from_payloads
from .temporal_research_settings import temporal_research_settings_snapshot


class WinnerTransitionInterventionError(RuntimeError):
    pass


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _timestamp(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp


def _max_drawdown(values: list[float]) -> float | None:
    if not values:
        return None
    peak = values[0]
    maximum = 0.0
    for value in values:
        peak = max(peak, value)
        drawdown = value / peak - 1.0 if peak else 0.0
        maximum = min(maximum, drawdown)
    return float(maximum)


def _monthly_returns(equity: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not equity:
        return []
    frame = pd.DataFrame(equity)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp")
    frame["month"] = frame["timestamp"].dt.strftime("%Y-%m")
    rows: list[dict[str, Any]] = []
    previous = None
    for month, group in frame.groupby("month", sort=True):
        end_value = float(group.iloc[-1]["value"])
        start_value = float(group.iloc[0]["starting_value"]) if previous is None else float(previous)
        rows.append({
            "month": str(month),
            "start_value": start_value,
            "end_value": end_value,
            "return": float(end_value / start_value - 1.0) if start_value else None,
        })
        previous = end_value
    return rows


def _worst_month(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    values = [row for row in rows if _finite(row.get("return")) is not None]
    return min(values, key=lambda row: float(row["return"]), default=None)


def _equity_rows(analytics: dict[str, Any], years: set[int] | None = None) -> list[dict[str, Any]]:
    rows = []
    for row in analytics.get("equity") or []:
        if not isinstance(row, dict):
            continue
        stamp = _timestamp(row.get("timestamp"))
        value = _finite(row.get("simulation_equity"))
        if stamp is None or value is None:
            continue
        if years is not None and int(stamp.year) not in years:
            continue
        rows.append({"timestamp": row.get("timestamp"), "stamp": stamp, "value": value})
    rows.sort(key=lambda row: row["stamp"])
    return rows


def _starting_value(analytics: dict[str, Any], first_stamp: pd.Timestamp, fallback: float) -> float:
    previous: tuple[pd.Timestamp, float] | None = None
    for row in analytics.get("equity") or []:
        if not isinstance(row, dict):
            continue
        stamp = _timestamp(row.get("timestamp"))
        value = _finite(row.get("simulation_equity"))
        if stamp is None or value is None or stamp >= first_stamp:
            continue
        if previous is None or stamp > previous[0]:
            previous = (stamp, value)
    return float(previous[1]) if previous is not None else float(fallback)


def _path_stats(path: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(row["value"]) for row in path]
    monthly = _monthly_returns(path)
    return {
        "ending_capital": values[-1] if values else None,
        "maximum_drawdown": _max_drawdown(values),
        "worst_month": _worst_month(monthly),
        "monthly_returns": monthly,
    }




def _transition_key_from_rotation(row: dict[str, Any]) -> str:
    stamp = _timestamp(row.get("executed_at"))
    if stamp is None:
        return ""
    return "|".join((stamp.isoformat(), str(row.get("from_asset") or "CASH"), str(row.get("to_asset") or "CASH")))


def _equity_value_at_or_before(path: list[dict[str, Any]], stamp: pd.Timestamp) -> float | None:
    value = None
    for row in path:
        row_stamp = _timestamp(row.get("timestamp"))
        if row_stamp is None or row_stamp > stamp:
            break
        candidate = _finite(row.get("value"))
        if candidate is not None:
            value = candidate
    return value


def _equity_session_index(path: list[dict[str, Any]], stamp: pd.Timestamp) -> int | None:
    index = None
    for position, row in enumerate(path):
        row_stamp = _timestamp(row.get("timestamp"))
        if row_stamp is None:
            continue
        if row_stamp <= stamp:
            index = position
        else:
            break
    return index


def _shadow_rotation_sequence(
    analytics: dict[str, Any],
    replay: dict[str, Any],
    *,
    long_defer: bool = False,
) -> list[dict[str, Any]]:
    path = (((replay.get("equity") or {}).get("shadow")) or [])
    if not path:
        return []
    first_stamp = _timestamp(path[0].get("timestamp"))
    last_stamp = _timestamp(path[-1].get("timestamp"))
    if first_stamp is None or last_stamp is None:
        return []
    base = [
        dict(row) for row in analytics.get("rotations") or []
        if isinstance(row, dict)
        and (stamp := _timestamp(row.get("executed_at"))) is not None
        and first_stamp <= stamp <= last_stamp
    ]
    base.sort(key=lambda row: _timestamp(row.get("executed_at")) or pd.Timestamp.min.tz_localize("UTC"))
    effects = [dict(row) for row in replay.get("effects") or [] if isinstance(row, dict)]
    effects_by_execution = {
        _timestamp(effect.get("execution_at")): effect
        for effect in effects
        if _timestamp(effect.get("execution_at")) is not None
    }
    effects_by_rejoin = {
        _timestamp(effect.get("rejoin_at")): effect
        for effect in effects
        if _timestamp(effect.get("rejoin_at")) is not None
    }

    result: list[dict[str, Any]] = []
    pending_long: dict[str, Any] | None = None
    for rotation in base:
        stamp = _timestamp(rotation.get("executed_at"))
        if stamp is None:
            continue
        effect = effects_by_execution.get(stamp)
        if effect is not None:
            if long_defer:
                pending_long = effect
                continue
            shifted = dict(rotation)
            shifted["executed_at"] = effect.get("rejoin_at")
            shifted["shadow_intervention"] = "defer_one_session_then_rejoin"
            shifted["shadow_original_executed_at"] = rotation.get("executed_at")
            shifted["shadow_risk_score"] = effect.get("risk_score")
            shifted["shadow_risk_threshold"] = effect.get("risk_threshold")
            result.append(shifted)
            continue

        if long_defer and pending_long is not None and stamp == _timestamp(pending_long.get("rejoin_at")):
            direct = dict(rotation)
            direct["from_asset"] = pending_long.get("from_asset")
            direct["shadow_intervention"] = "keep_incumbent_until_next_control_transition"
            direct["shadow_original_executed_at"] = pending_long.get("execution_at")
            direct["shadow_risk_score"] = pending_long.get("risk_score")
            direct["shadow_risk_threshold"] = pending_long.get("risk_threshold")
            if str(direct.get("from_asset") or "CASH").upper() != str(direct.get("to_asset") or "CASH").upper():
                result.append(direct)
            pending_long = None
            continue

        if long_defer and stamp in effects_by_rejoin:
            continue
        result.append(dict(rotation))

    result.sort(key=lambda row: _timestamp(row.get("executed_at")) or pd.Timestamp.min.tz_localize("UTC"))

    previous_stamp = first_stamp
    previous_value = _finite(path[0].get("starting_value")) or _finite(path[0].get("value"))
    previous_index = _equity_session_index(path, previous_stamp)
    for rotation in result:
        stamp = _timestamp(rotation.get("executed_at"))
        if stamp is None:
            continue
        current_value = _equity_value_at_or_before(path, stamp)
        current_index = _equity_session_index(path, stamp)
        from_asset = str(rotation.get("from_asset") or "CASH").upper()
        if from_asset == "CASH":
            rotation["realized_pnl"] = None
            rotation["position_return"] = None
            rotation["holding_days"] = None
        elif current_value is not None and previous_value not in {None, 0.0}:
            rotation["realized_pnl"] = float(current_value - float(previous_value))
            rotation["position_return"] = float(current_value / float(previous_value) - 1.0)
            if current_index is not None and previous_index is not None:
                rotation["holding_days"] = max(0, int(current_index - previous_index))
        previous_stamp = stamp
        previous_value = current_value if current_value is not None else previous_value
        previous_index = current_index if current_index is not None else previous_index
    return result


def _attach_movement_heatmap(
    analytics: dict[str, Any],
    replay: dict[str, Any],
    *,
    long_defer: bool = False,
) -> dict[str, Any]:
    if not isinstance(replay, dict) or not (((replay.get("equity") or {}).get("shadow")) or []):
        return replay
    baseline_path = (((replay.get("equity") or {}).get("baseline")) or [])
    first_stamp = _timestamp(baseline_path[0].get("timestamp")) if baseline_path else None
    last_stamp = _timestamp(baseline_path[-1].get("timestamp")) if baseline_path else None
    baseline_rotations = []
    if first_stamp is not None and last_stamp is not None:
        baseline_rotations = [
            dict(row) for row in analytics.get("rotations") or []
            if isinstance(row, dict)
            and (stamp := _timestamp(row.get("executed_at"))) is not None
            and first_stamp <= stamp <= last_stamp
        ]
    return {
        **replay,
        "movement_heatmap": {
            "baseline_rotations": baseline_rotations,
            "shadow_rotations": _shadow_rotation_sequence(analytics, replay, long_defer=long_defer),
        },
    }


def _one_session_effects(
    analytics: dict[str, Any],
    scored_rows: list[dict[str, Any]],
    years: set[int] | None = None,
) -> list[dict[str, Any]]:
    equity = _equity_rows(analytics, years)
    if not equity:
        return []
    effects: list[dict[str, Any]] = []
    for row in scored_rows:
        if not bool(row.get("high_risk")):
            continue
        execution_at = _timestamp(row.get("execution_at"))
        if execution_at is None:
            continue
        if years is not None and int(execution_at.year) not in years:
            continue
        target_return = _finite(row.get("one_interval_target_return"))
        incumbent_return = _finite(row.get("one_interval_incumbent_return"))
        if target_return is None or incumbent_return is None or 1.0 + target_return <= 1e-9:
            continue
        rejoin = next((item for item in equity if item["stamp"] > execution_at), None)
        if rejoin is None:
            continue
        if years is not None and int(rejoin["stamp"].year) not in years:
            continue
        factor = float((1.0 + incumbent_return) / (1.0 + target_return))
        effects.append({
            "transition_key": row.get("transition_key"),
            "execution_at": row.get("execution_at"),
            "rejoin_at": rejoin.get("timestamp"),
            "from_asset": row.get("from_asset"),
            "to_asset": row.get("to_asset"),
            "risk_score": row.get("risk_score"),
            "risk_threshold": row.get("risk_threshold"),
            "rotation_value_added": row.get("rotation_value_added"),
            "one_interval_target_return": target_return,
            "one_interval_incumbent_return": incumbent_return,
            "one_interval_value_added": _finite(row.get("one_interval_value_added")),
            "capital_factor": factor,
        })
    effects.sort(key=lambda row: _timestamp(row.get("rejoin_at")) or pd.Timestamp.min.tz_localize("UTC"))
    return effects


def _replay_one_session(
    analytics: dict[str, Any],
    scored_rows: list[dict[str, Any]],
    *,
    years: set[int] | None = None,
) -> dict[str, Any]:
    equity = _equity_rows(analytics, years)
    if not equity:
        return {}
    effects = _one_session_effects(analytics, scored_rows, years)
    effects_by_time: dict[pd.Timestamp, list[dict[str, Any]]] = {}
    for effect in effects:
        stamp = _timestamp(effect.get("rejoin_at"))
        if stamp is not None:
            effects_by_time.setdefault(stamp, []).append(effect)

    starting_value = _starting_value(analytics, equity[0]["stamp"], float(equity[0]["value"]))
    baseline_equity: list[dict[str, Any]] = []
    shadow_equity: list[dict[str, Any]] = []
    cumulative_factor = 1.0
    for row in equity:
        for effect in effects_by_time.get(row["stamp"], []):
            cumulative_factor *= float(effect["capital_factor"])
        baseline_equity.append({
            "timestamp": row["timestamp"],
            "value": float(row["value"]),
            "starting_value": starting_value,
        })
        shadow_equity.append({
            "timestamp": row["timestamp"],
            "value": float(row["value"] * cumulative_factor),
            "starting_value": starting_value,
        })

    baseline = _path_stats(baseline_equity)
    shadow = _path_stats(shadow_equity)
    base_end = _finite(baseline.get("ending_capital"))
    shadow_end = _finite(shadow.get("ending_capital"))
    return {
        "method": "one_session_defer_then_rejoin_shadow",
        "interventions": len(effects),
        "effects": effects,
        "baseline": {key: value for key, value in baseline.items() if key != "monthly_returns"},
        "shadow": {
            **{key: value for key, value in shadow.items() if key != "monthly_returns"},
            "ending_capital_delta": float(shadow_end - base_end) if shadow_end is not None and base_end is not None else None,
            "ending_capital_delta_rate": float(shadow_end / base_end - 1.0) if shadow_end is not None and base_end not in {None, 0.0} else None,
        },
        "monthly_returns": {
            "baseline": baseline.get("monthly_returns") or [],
            "shadow": shadow.get("monthly_returns") or [],
        },
        "equity": {
            "baseline": baseline_equity,
            "shadow": shadow_equity,
        },
    }


def _control_replay(analytics: dict[str, Any], years: set[int] | None = None) -> dict[str, Any]:
    equity = _equity_rows(analytics, years)
    if not equity:
        return {}
    starting_value = _starting_value(analytics, equity[0]["stamp"], float(equity[0]["value"]))
    path = [{"timestamp": row["timestamp"], "value": float(row["value"]), "starting_value": starting_value} for row in equity]
    stats = _path_stats(path)
    return {
        "method": "control",
        "interventions": 0,
        "baseline": {key: value for key, value in stats.items() if key != "monthly_returns"},
        "shadow": {
            **{key: value for key, value in stats.items() if key != "monthly_returns"},
            "ending_capital_delta": 0.0,
            "ending_capital_delta_rate": 0.0,
        },
        "monthly_returns": {"baseline": stats.get("monthly_returns") or [], "shadow": stats.get("monthly_returns") or []},
        "equity": {"baseline": path, "shadow": path},
    }


def _tail_safe(replay: dict[str, Any]) -> bool:
    baseline = replay.get("baseline") if isinstance(replay.get("baseline"), dict) else {}
    shadow = replay.get("shadow") if isinstance(replay.get("shadow"), dict) else {}
    base_dd = _finite(baseline.get("maximum_drawdown"))
    shadow_dd = _finite(shadow.get("maximum_drawdown"))
    base_worst = baseline.get("worst_month") if isinstance(baseline.get("worst_month"), dict) else {}
    shadow_worst = shadow.get("worst_month") if isinstance(shadow.get("worst_month"), dict) else {}
    base_month = _finite(base_worst.get("return"))
    shadow_month = _finite(shadow_worst.get("return"))
    dd_safe = base_dd is None or shadow_dd is None or shadow_dd >= base_dd - 1e-12
    month_safe = base_month is None or shadow_month is None or shadow_month >= base_month - 1e-12
    return bool(dd_safe and month_safe)


def _training_decision(analytics: dict[str, Any], scored_rows: list[dict[str, Any]], prior_years: list[int]) -> dict[str, Any]:
    if not prior_years:
        return {
            "selected_mode": "control",
            "reason": "warmup_no_prior_oos_year",
            "prior_oos_years": [],
            "candidate": None,
        }
    replay = _replay_one_session(analytics, scored_rows, years=set(prior_years))
    delta = _finite(((replay.get("shadow") or {}).get("ending_capital_delta_rate")))
    safe = _tail_safe(replay)
    selected = bool(delta is not None and delta > 0.0 and safe)
    return {
        "selected_mode": "one_session_recheck" if selected else "control",
        "reason": "positive_capital_and_tail_safe" if selected else "insufficient_prior_oos_evidence",
        "prior_oos_years": list(prior_years),
        "candidate": {
            "mode": "one_session_recheck",
            "ending_capital_delta_rate": delta,
            "tail_safe": safe,
            "interventions": replay.get("interventions"),
            "baseline_maximum_drawdown": ((replay.get("baseline") or {}).get("maximum_drawdown")),
            "shadow_maximum_drawdown": ((replay.get("shadow") or {}).get("maximum_drawdown")),
            "baseline_worst_month": ((replay.get("baseline") or {}).get("worst_month")),
            "shadow_worst_month": ((replay.get("shadow") or {}).get("worst_month")),
        },
    }


def _selected_walk_forward_shadow(
    analytics: dict[str, Any],
    scored_rows: list[dict[str, Any]],
    selections: list[dict[str, Any]],
) -> dict[str, Any]:
    selected_years = {
        int(row["test_year"])
        for row in selections
        if str(row.get("selected_mode")) == "one_session_recheck"
    }
    selected_rows = [row for row in scored_rows if int(row.get("year") or 0) in selected_years]
    replay = _replay_one_session(analytics, selected_rows)
    replay["selected_years"] = sorted(selected_years)
    return replay


def _month_return(replay: dict[str, Any], month: str, side: str) -> float | None:
    rows = ((replay.get("monthly_returns") or {}).get(side) or [])
    row = next((item for item in rows if str(item.get("month")) == month), None)
    return _finite(row.get("return")) if isinstance(row, dict) else None


def run_transition_intervention_search_from_payloads(
    *,
    run_id: str,
    processing_id: str,
    start_month: str,
    end_month: str,
    transition_attribution: dict[str, Any],
    analytics: dict[str, Any],
    seed: int = 42,
    risk_search: dict[str, Any] | None = None,
    research_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    risk = risk_search if isinstance(risk_search, dict) else None
    reusable = bool(
        risk
        and ((risk.get("oos") or {}).get("scored_transitions"))
        and (((risk.get("shadow_replay") or {}).get("equity") or {}).get("shadow"))
        and isinstance(risk.get("research_settings"), dict)
    )
    frozen_research_settings = (risk or {}).get("research_settings") if isinstance((risk or {}).get("research_settings"), dict) else research_settings
    if not reusable:
        if not isinstance(frozen_research_settings, dict):
            raise WinnerTransitionInterventionError("Temporal research settings snapshot is required.")
        risk = run_transition_risk_search_from_payloads(
            run_id=run_id,
            processing_id=processing_id,
            start_month=start_month,
            end_month=end_month,
            transition_attribution=transition_attribution,
            analytics=analytics,
            research_settings=frozen_research_settings,
            seed=seed,
        )
    frozen_research_settings = risk.get("research_settings") if isinstance(risk.get("research_settings"), dict) else frozen_research_settings
    oos = risk.get("oos") if isinstance(risk.get("oos"), dict) else {}
    scored_rows = [dict(row) for row in oos.get("scored_transitions") or [] if isinstance(row, dict)]
    if not scored_rows:
        high_risk = [dict(row) for row in oos.get("high_risk_transitions") or [] if isinstance(row, dict)]
        if high_risk:
            scored_rows = high_risk
    if not scored_rows:
        raise WinnerTransitionInterventionError("No OOS Winner transition risk scores are available for intervention research.")

    oos_years = sorted({int(row.get("year")) for row in scored_rows if row.get("year") is not None})
    selections: list[dict[str, Any]] = []
    for test_year in oos_years:
        prior_years = [year for year in oos_years if year < test_year]
        decision = _training_decision(analytics, scored_rows, prior_years)
        test_rows = [row for row in scored_rows if int(row.get("year") or 0) == test_year]
        test_replay = (
            _replay_one_session(analytics, test_rows, years={test_year})
            if decision["selected_mode"] == "one_session_recheck"
            else _control_replay(analytics, years={test_year})
        )
        selections.append({
            "test_year": test_year,
            **decision,
            "test_result": {
                "interventions": test_replay.get("interventions"),
                "ending_capital_delta_rate": ((test_replay.get("shadow") or {}).get("ending_capital_delta_rate")),
                "baseline_maximum_drawdown": ((test_replay.get("baseline") or {}).get("maximum_drawdown")),
                "shadow_maximum_drawdown": ((test_replay.get("shadow") or {}).get("maximum_drawdown")),
                "baseline_worst_month": ((test_replay.get("baseline") or {}).get("worst_month")),
                "shadow_worst_month": ((test_replay.get("shadow") or {}).get("worst_month")),
            },
        })

    diagnostic_one_session = _attach_movement_heatmap(analytics, _replay_one_session(analytics, scored_rows))
    selected_shadow = _attach_movement_heatmap(analytics, _selected_walk_forward_shadow(analytics, scored_rows, selections))
    legacy_shadow = risk.get("shadow_replay") if isinstance(risk.get("shadow_replay"), dict) else {}
    if not ((legacy_shadow.get("equity") or {}).get("shadow")):
        legacy_shadow = {**legacy_shadow, "equity": {"baseline": [], "shadow": []}}
    legacy_shadow = _attach_movement_heatmap(analytics, legacy_shadow, long_defer=True)

    june = {
        "month": "2026-06",
        "baseline_return": _month_return(selected_shadow, "2026-06", "baseline"),
        "walk_forward_intervention_return": _month_return(selected_shadow, "2026-06", "shadow"),
        "one_session_all_oos_return": _month_return(diagnostic_one_session, "2026-06", "shadow"),
        "legacy_long_shadow_return": _month_return(legacy_shadow, "2026-06", "shadow"),
    }

    return bson_value({
        "schema_version": 1,
        "id": f"winner-transition-intervention-{utc_now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}",
        "run_id": str(run_id),
        "processing_id": str(processing_id),
        "period_start": start_month,
        "period_end": end_month,
        "created_at": utc_now(),
        "status": "completed",
        "research_settings": frozen_research_settings,
        "protocol": {
            "source_detector_family": "temporal_rejection",
            "source_detector": "winner_anchor_transition_risk_search_oos",
            "validation": "expanding_prior_oos_year_activation",
            "candidate_intervention": "defer_one_session_then_rejoin",
            "activation_rule": "positive_prior_oos_capital_and_no_worse_maxdd_or_worst_month",
            "unbounded_keep_until_next_transition": "reference_only_excluded_from_selection",
            "future_information_in_selection": False,
            "strategy_decisions_changed": False,
            "research_only": True,
        },
        "source_risk_search": {
            "id": risk.get("id"),
            "reused_persisted": reusable,
            "oos_metrics": oos.get("metrics"),
            "oos_years": oos_years,
        },
        "outer_results": selections,
        "walk_forward_selected_shadow": selected_shadow,
        "one_session_all_oos_shadow": diagnostic_one_session,
        "legacy_long_shadow_reference": legacy_shadow,
        "june_2026": june,
    })


def run_winner_transition_intervention_search(
    db: Any,
    run_id: str,
    *,
    processing_id: str,
    start_month: str,
    end_month: str,
    seed: int = 42,
) -> dict[str, Any]:
    attribution = get_winner_transition_attribution(db, run_id, start_month=start_month, end_month=end_month)
    analytics = processing_analytics(db, processing_id)
    stored_risk = db[TEMPORAL_WINNER_TRANSITION_RISK_RESEARCH_COLLECTION].find_one(
        {
            "run_id": str(run_id),
            "processing_id": str(processing_id),
            "period_start": str(start_month),
            "period_end": str(end_month),
            "status": "completed",
        },
        {"_id": 0},
        sort=[("created_at", -1)],
    )
    research_settings = temporal_research_settings_snapshot(db)
    result = run_transition_intervention_search_from_payloads(
        run_id=run_id,
        processing_id=processing_id,
        start_month=start_month,
        end_month=end_month,
        transition_attribution=attribution,
        analytics=analytics,
        seed=seed,
        risk_search=bson_value(stored_risk) if stored_risk is not None else None,
        research_settings=research_settings,
    )
    db[TEMPORAL_WINNER_TRANSITION_INTERVENTION_RESEARCH_COLLECTION].insert_one(dict(result))
    return result


def get_latest_winner_transition_intervention_search(
    db: Any,
    run_id: str,
    *,
    processing_id: str,
    start_month: str,
    end_month: str,
) -> dict[str, Any] | None:
    row = db[TEMPORAL_WINNER_TRANSITION_INTERVENTION_RESEARCH_COLLECTION].find_one(
        {
            "run_id": str(run_id),
            "processing_id": str(processing_id),
            "period_start": str(start_month),
            "period_end": str(end_month),
            "status": "completed",
        },
        {"_id": 0},
        sort=[("created_at", -1)],
    )
    if row is None:
        return None
    document = bson_value(row)
    analytics = processing_analytics(db, processing_id)
    document["walk_forward_selected_shadow"] = _attach_movement_heatmap(
        analytics, dict(document.get("walk_forward_selected_shadow") or {})
    )
    document["one_session_all_oos_shadow"] = _attach_movement_heatmap(
        analytics, dict(document.get("one_session_all_oos_shadow") or {})
    )
    document["legacy_long_shadow_reference"] = _attach_movement_heatmap(
        analytics, dict(document.get("legacy_long_shadow_reference") or {}), long_defer=True
    )
    return bson_value(document)



def _risk_margin(row: dict[str, Any]) -> float | None:
    score = _finite(row.get("risk_score"))
    threshold = _finite(row.get("risk_threshold"))
    if score is None or threshold is None:
        return None
    return float(score - threshold)


def _confidence_gated_rows(
    scored_rows: list[dict[str, Any]],
    *,
    years: set[int] | None = None,
    margin_threshold: float | None = None,
    active: bool = True,
) -> list[dict[str, Any]]:
    gated: list[dict[str, Any]] = []
    for source in scored_rows:
        year = int(source.get("year") or 0)
        if years is not None and year not in years:
            continue
        margin = _risk_margin(source)
        enabled = bool(
            active
            and source.get("high_risk")
            and margin is not None
            and margin_threshold is not None
            and margin >= margin_threshold - 1e-12
        )
        gated.append({**source, "risk_margin": margin, "high_risk": enabled})
    return gated


def _confidence_candidate(
    analytics: dict[str, Any],
    scored_rows: list[dict[str, Any]],
    prior_years: list[int],
    *,
    margin_quantiles: list[float],
    min_alerts: int,
) -> dict[str, Any]:
    if not prior_years:
        return {
            "selected_mode": "control",
            "reason": "warmup_no_prior_oos_year",
            "prior_oos_years": [],
            "selected_margin_quantile": None,
            "selected_margin_threshold": None,
            "candidates": [],
        }

    prior_set = set(prior_years)
    alerts = [
        row for row in scored_rows
        if int(row.get("year") or 0) in prior_set
        and bool(row.get("high_risk"))
        and _risk_margin(row) is not None
    ]
    if len(alerts) < min_alerts:
        return {
            "selected_mode": "control",
            "reason": "insufficient_prior_oos_alerts",
            "prior_oos_years": list(prior_years),
            "selected_margin_quantile": None,
            "selected_margin_threshold": None,
            "candidates": [],
        }

    margins = pd.Series([float(_risk_margin(row)) for row in alerts], dtype="float64")
    candidates: list[dict[str, Any]] = []
    for quantile in margin_quantiles:
        threshold = float(margins.quantile(quantile))
        gated = _confidence_gated_rows(
            scored_rows,
            years=prior_set,
            margin_threshold=threshold,
        )
        replay = _replay_one_session(analytics, gated, years=prior_set)
        delta = _finite(((replay.get("shadow") or {}).get("ending_capital_delta_rate")))
        safe = _tail_safe(replay)
        prior_year_results: list[dict[str, Any]] = []
        for prior_year in prior_years:
            year_rows = _confidence_gated_rows(
                scored_rows,
                years={prior_year},
                margin_threshold=threshold,
            )
            year_replay = _replay_one_session(analytics, year_rows, years={prior_year})
            prior_year_results.append({
                "year": int(prior_year),
                "ending_capital_delta_rate": ((year_replay.get("shadow") or {}).get("ending_capital_delta_rate")),
                "tail_safe": _tail_safe(year_replay),
                "interventions": int(year_replay.get("interventions") or 0),
            })
        year_consistent = all(
            (_finite(row.get("ending_capital_delta_rate")) or 0.0) >= -1e-12
            and bool(row.get("tail_safe"))
            for row in prior_year_results
        )
        candidates.append({
            "margin_quantile": float(quantile),
            "margin_threshold": threshold,
            "ending_capital_delta_rate": delta,
            "tail_safe": safe,
            "year_consistent": year_consistent,
            "prior_year_results": prior_year_results,
            "interventions": int(replay.get("interventions") or 0),
            "baseline_maximum_drawdown": ((replay.get("baseline") or {}).get("maximum_drawdown")),
            "shadow_maximum_drawdown": ((replay.get("shadow") or {}).get("maximum_drawdown")),
            "baseline_worst_month": ((replay.get("baseline") or {}).get("worst_month")),
            "shadow_worst_month": ((replay.get("shadow") or {}).get("worst_month")),
        })

    eligible = [
        row for row in candidates
        if (_finite(row.get("ending_capital_delta_rate")) or 0.0) > 0.0
        and bool(row.get("tail_safe"))
        and bool(row.get("year_consistent"))
        and int(row.get("interventions") or 0) > 0
    ]
    if not eligible:
        return {
            "selected_mode": "control",
            "reason": "no_positive_tail_safe_confidence_gate",
            "prior_oos_years": list(prior_years),
            "selected_margin_quantile": None,
            "selected_margin_threshold": None,
            "candidates": candidates,
        }

    selected = max(
        eligible,
        key=lambda row: (
            _finite(row.get("ending_capital_delta_rate")) or -1.0,
            float(row.get("margin_quantile") or 0.0),
            -int(row.get("interventions") or 0),
        ),
    )
    return {
        "selected_mode": "confidence_calibrated_one_session",
        "reason": "best_positive_tail_safe_prior_oos_confidence_gate",
        "prior_oos_years": list(prior_years),
        "selected_margin_quantile": selected.get("margin_quantile"),
        "selected_margin_threshold": selected.get("margin_threshold"),
        "selected_candidate": selected,
        "candidates": candidates,
    }


def _selected_confidence_shadow(
    analytics: dict[str, Any],
    scored_rows: list[dict[str, Any]],
    selections: list[dict[str, Any]],
) -> dict[str, Any]:
    selected_rows: list[dict[str, Any]] = []
    for selection in selections:
        year = int(selection.get("test_year") or 0)
        active = str(selection.get("selected_mode") or "") == "confidence_calibrated_one_session"
        threshold = _finite(selection.get("selected_margin_threshold"))
        selected_rows.extend(_confidence_gated_rows(
            scored_rows,
            years={year},
            margin_threshold=threshold,
            active=active,
        ))
    replay = _replay_one_session(analytics, selected_rows)
    replay["selected_years"] = [
        int(row.get("test_year"))
        for row in selections
        if str(row.get("selected_mode") or "") == "confidence_calibrated_one_session"
    ]
    return replay


def run_transition_confidence_calibration_from_payloads(
    *,
    run_id: str,
    processing_id: str,
    start_month: str,
    end_month: str,
    analytics: dict[str, Any],
    risk_search: dict[str, Any],
    intervention_search: dict[str, Any],
) -> dict[str, Any]:
    research_settings = risk_search.get("research_settings") if isinstance(risk_search.get("research_settings"), dict) else {}
    settings_payload = research_settings.get("settings") if isinstance(research_settings.get("settings"), dict) else research_settings
    confidence_settings = settings_payload.get("confidence") if isinstance(settings_payload.get("confidence"), dict) else {}
    if not confidence_settings:
        raise WinnerTransitionInterventionError("The source risk search does not contain a frozen temporal research settings snapshot. Run the risk search again.")
    margin_quantiles = [float(value) for value in confidence_settings["margin_quantiles"]]
    min_alerts = int(confidence_settings["min_alerts"])
    oos = risk_search.get("oos") if isinstance(risk_search.get("oos"), dict) else {}
    scored_rows = [dict(row) for row in oos.get("scored_transitions") or [] if isinstance(row, dict)]
    if not scored_rows:
        raise WinnerTransitionInterventionError("No OOS Winner transition scores are available for confidence calibration.")

    oos_years = sorted({int(row.get("year")) for row in scored_rows if row.get("year") is not None})
    selections: list[dict[str, Any]] = []
    for test_year in oos_years:
        prior_years = [year for year in oos_years if year < test_year]
        decision = _confidence_candidate(
            analytics,
            scored_rows,
            prior_years,
            margin_quantiles=margin_quantiles,
            min_alerts=min_alerts,
        )
        if decision["selected_mode"] == "confidence_calibrated_one_session":
            threshold = _finite(decision.get("selected_margin_threshold"))
            test_rows = _confidence_gated_rows(
                scored_rows,
                years={test_year},
                margin_threshold=threshold,
            )
            test_replay = _replay_one_session(analytics, test_rows, years={test_year})
        else:
            test_replay = _control_replay(analytics, years={test_year})
        selections.append({
            "test_year": test_year,
            **decision,
            "test_result": {
                "interventions": int(test_replay.get("interventions") or 0),
                "ending_capital_delta_rate": ((test_replay.get("shadow") or {}).get("ending_capital_delta_rate")),
                "tail_safe": _tail_safe(test_replay),
                "baseline_maximum_drawdown": ((test_replay.get("baseline") or {}).get("maximum_drawdown")),
                "shadow_maximum_drawdown": ((test_replay.get("shadow") or {}).get("maximum_drawdown")),
                "baseline_worst_month": ((test_replay.get("baseline") or {}).get("worst_month")),
                "shadow_worst_month": ((test_replay.get("shadow") or {}).get("worst_month")),
            },
        })

    selected_shadow = _attach_movement_heatmap(analytics, _selected_confidence_shadow(analytics, scored_rows, selections))
    june = {
        "month": "2026-06",
        "baseline_return": _month_return(selected_shadow, "2026-06", "baseline"),
        "confidence_calibrated_return": _month_return(selected_shadow, "2026-06", "shadow"),
    }
    return bson_value({
        "schema_version": 1,
        "id": f"winner-transition-confidence-{utc_now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}",
        "run_id": str(run_id),
        "processing_id": str(processing_id),
        "period_start": start_month,
        "period_end": end_month,
        "created_at": utc_now(),
        "status": "completed",
        "research_settings": research_settings,
        "protocol": {
            "source_detector_family": "temporal_rejection",
            "source_intervention": "defer_one_session_then_rejoin",
            "confidence_measure": "risk_score_minus_fold_risk_threshold",
            "margin_quantiles": margin_quantiles,
            "minimum_prior_alerts": min_alerts,
            "validation": "expanding_prior_oos_year_confidence_calibration",
            "selection_rule": "max_prior_oos_capital_subject_to_tail_safety_and_no_negative_prior_oos_year",
            "future_information_in_selection": False,
            "strategy_decisions_changed": False,
            "research_only": True,
        },
        "source_risk_search": {
            "id": risk_search.get("id"),
            "oos_metrics": oos.get("metrics"),
        },
        "source_intervention_search": {
            "id": intervention_search.get("id"),
            "walk_forward_capital": (((intervention_search.get("walk_forward_selected_shadow") or {}).get("shadow") or {}).get("ending_capital")),
        },
        "outer_results": selections,
        "walk_forward_calibrated_shadow": selected_shadow,
        "june_2026": june,
    })


def run_winner_transition_confidence_calibration(
    db: Any,
    run_id: str,
    *,
    processing_id: str,
    start_month: str,
    end_month: str,
) -> dict[str, Any]:
    risk = db[TEMPORAL_WINNER_TRANSITION_RISK_RESEARCH_COLLECTION].find_one(
        {
            "run_id": str(run_id),
            "processing_id": str(processing_id),
            "period_start": str(start_month),
            "period_end": str(end_month),
            "status": "completed",
        },
        {"_id": 0},
        sort=[("created_at", -1)],
    )
    intervention = db[TEMPORAL_WINNER_TRANSITION_INTERVENTION_RESEARCH_COLLECTION].find_one(
        {
            "run_id": str(run_id),
            "processing_id": str(processing_id),
            "period_start": str(start_month),
            "period_end": str(end_month),
            "status": "completed",
        },
        {"_id": 0},
        sort=[("created_at", -1)],
    )
    if risk is None:
        raise WinnerTransitionInterventionError("Run Winner transition risk search before confidence calibration.")
    if intervention is None:
        raise WinnerTransitionInterventionError("Run Winner transition intervention search before confidence calibration.")
    analytics = processing_analytics(db, processing_id)
    result = run_transition_confidence_calibration_from_payloads(
        run_id=run_id,
        processing_id=processing_id,
        start_month=start_month,
        end_month=end_month,
        analytics=analytics,
        risk_search=bson_value(risk),
        intervention_search=bson_value(intervention),
    )
    db[TEMPORAL_WINNER_TRANSITION_CONFIDENCE_RESEARCH_COLLECTION].insert_one(dict(result))
    return result


def get_latest_winner_transition_confidence_calibration(
    db: Any,
    run_id: str,
    *,
    processing_id: str,
    start_month: str,
    end_month: str,
) -> dict[str, Any] | None:
    row = db[TEMPORAL_WINNER_TRANSITION_CONFIDENCE_RESEARCH_COLLECTION].find_one(
        {
            "run_id": str(run_id),
            "processing_id": str(processing_id),
            "period_start": str(start_month),
            "period_end": str(end_month),
            "status": "completed",
        },
        {"_id": 0},
        sort=[("created_at", -1)],
    )
    if row is None:
        return None
    document = bson_value(row)
    analytics = processing_analytics(db, processing_id)
    document["walk_forward_calibrated_shadow"] = _attach_movement_heatmap(
        analytics, dict(document.get("walk_forward_calibrated_shadow") or {})
    )
    return bson_value(document)
