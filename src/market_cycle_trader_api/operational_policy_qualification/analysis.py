from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from ..alternative_action.analysis import _horizon_return, _series_by_symbol
from .config import (
    ACTION_PROBABILITY_THRESHOLD,
    ANALYSIS_VERSION,
    DOWNSIDE_PENALTY,
    FEATURES,
    HORIZONS,
    MAX_DRAWDOWN_DEGRADATION,
    MAX_SHARPE_DEGRADATION,
    MAX_WORST_MONTH_DEGRADATION,
    MIN_CAPITAL_LIFT,
    MIN_INTERVENTIONS,
    MIN_POSITIVE_OOS_YEARS,
    MIN_UTILITY_EDGE,
    ONE_SIDE_COST_BPS,
    RANDOM_STATE,
    SCHEMA_VERSION,
    UTILITY_WEIGHTS,
)

ACTIONS = ("ROTATE", "HOLD", "CASH")


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




def _max_drawdown(values: list[float]) -> float | None:
    if not values:
        return None
    peak = values[0]
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        if peak:
            worst = min(worst, value / peak - 1.0)
    return float(worst)


def _equity_rows(analytics: dict[str, Any], years: set[int] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in analytics.get("equity") or []:
        if not isinstance(row, dict):
            continue
        stamp = _stamp(row.get("timestamp"))
        value = _finite(row.get("simulation_equity"))
        if stamp is None or value is None:
            continue
        if years is not None and int(stamp.year) not in years:
            continue
        rows.append({"timestamp": row.get("timestamp"), "stamp": stamp, "value": value})
    rows.sort(key=lambda item: item["stamp"])
    return rows


def _starting_value(analytics: dict[str, Any], first_stamp: pd.Timestamp, fallback: float) -> float:
    metrics = analytics.get("metrics") if isinstance(analytics.get("metrics"), dict) else {}
    if first_stamp == _stamp((analytics.get("equity") or [{}])[0].get("timestamp")):
        initial = _finite(metrics.get("initial_capital"))
        if initial is not None:
            return initial
    previous: tuple[pd.Timestamp, float] | None = None
    for row in analytics.get("equity") or []:
        if not isinstance(row, dict):
            continue
        stamp = _stamp(row.get("timestamp"))
        value = _finite(row.get("simulation_equity"))
        if stamp is None or value is None or stamp >= first_stamp:
            continue
        if previous is None or stamp > previous[0]:
            previous = (stamp, value)
    return float(previous[1]) if previous is not None else float(fallback)


def _monthly_returns(path: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not path:
        return []
    frame = pd.DataFrame(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp")
    frame["month"] = frame["timestamp"].dt.strftime("%Y-%m")
    previous = None
    rows: list[dict[str, Any]] = []
    for month, group in frame.groupby("month", sort=True):
        end_value = float(group.iloc[-1]["value"])
        start_value = float(group.iloc[0]["starting_value"]) if previous is None else float(previous)
        rows.append({"month": str(month), "start_value": start_value, "end_value": end_value, "return": float(end_value / start_value - 1.0) if start_value else None})
        previous = end_value
    return rows


def _path_stats(path: list[dict[str, Any]]) -> dict[str, Any]:
    if not path:
        return {}
    values = np.asarray([float(row["value"]) for row in path], dtype=float)
    starting_value = float(path[0].get("starting_value") or values[0])
    stamps = pd.to_datetime([row["timestamp"] for row in path], utc=True)
    previous = np.concatenate(([starting_value], values[:-1]))
    daily_returns = values / previous - 1.0
    daily_std = float(np.std(daily_returns, ddof=1)) if len(daily_returns) > 1 else 0.0
    sharpe = float(np.mean(daily_returns) / daily_std * math.sqrt(252.0)) if daily_std > 1e-12 else None
    elapsed_years = max((stamps[-1] - stamps[0]).days / 365.25, 1.0 / 365.25)
    cagr = float((values[-1] / starting_value) ** (1.0 / elapsed_years) - 1.0) if starting_value > 0 and values[-1] > 0 else None
    monthly = _monthly_returns(path)
    worst_month = min((row for row in monthly if _finite(row.get("return")) is not None), key=lambda row: float(row["return"]), default=None)
    return {
        "initial_capital": starting_value,
        "ending_capital": float(values[-1]),
        "total_return": float(values[-1] / starting_value - 1.0) if starting_value else None,
        "cagr": cagr,
        "sharpe": sharpe,
        "maximum_drawdown": _max_drawdown(values.tolist()),
        "worst_month": worst_month,
        "monthly_returns": monthly,
    }

def _exact_transition_session(transition: dict[str, Any]) -> dict[str, Any]:
    decision = _stamp(transition.get("decision_at"))
    sessions = (((transition.get("trajectory") or {}).get("sessions")) or [])
    if decision is not None:
        for row in sessions:
            if isinstance(row, dict) and _stamp(row.get("decision_at")) == decision:
                return row
    return next((row for row in reversed(sessions) if isinstance(row, dict)), {})


def _feature_row(transition: dict[str, Any]) -> dict[str, Any]:
    session = _exact_transition_session(transition)
    temporal = session.get("temporal") if isinstance(session.get("temporal"), dict) else {}
    target = temporal.get("target") if isinstance(temporal.get("target"), dict) else {}
    incumbent = temporal.get("incumbent") if isinstance(temporal.get("incumbent"), dict) else {}
    delta = temporal.get("target_minus_incumbent") if isinstance(temporal.get("target_minus_incumbent"), dict) else {}
    finite_count = _finite(session.get("finite_score_count"))
    positive_count = _finite(session.get("positive_score_count"))
    positive_share = None if finite_count in {None, 0.0} or positive_count is None else float(positive_count / finite_count)
    row = {
        "top1_top2_gap": transition.get("winner_top1_top2_score_gap") if transition.get("winner_top1_top2_score_gap") is not None else session.get("top1_top2_gap"),
        "target_rank": session.get("target_rank"),
        "incumbent_rank": session.get("incumbent_rank"),
        "target_score": session.get("target_score"),
        "incumbent_score": session.get("incumbent_score"),
        "target_minus_incumbent_score": session.get("target_minus_incumbent_score"),
        "universe_score_mean": session.get("universe_score_mean"),
        "universe_score_std": session.get("universe_score_std"),
        "positive_score_share": positive_share,
    }
    fields = (
        "entry_rank_percentile", "opportunity_gate_score", "risk_adjusted_entry_score", "hold_score",
        "incumbent_persistence_score", "incumbent_risk_health", "short_profit_consensus",
        "long_profit_confirmation", "horizon_agreement", "all_horizon_risk_safety", "predicted_drawdown",
    )
    for field in fields:
        row[f"target_{field}"] = target.get(field)
        row[f"incumbent_{field}"] = incumbent.get(field)
        row[f"delta_{field}"] = delta.get(field)
    return {key: _finite(row.get(key)) for key in FEATURES}


def _action_cost(action: str) -> float:
    one_side = ONE_SIDE_COST_BPS / 10_000.0
    return 0.0 if action == "HOLD" else float(2.0 * one_side)


def _utility(action: str, returns: dict[int, float | None]) -> float | None:
    if any(returns.get(horizon) is None for horizon in HORIZONS):
        return None
    values = [float(returns[horizon]) for horizon in HORIZONS]
    weighted = sum(float(UTILITY_WEIGHTS[horizon]) * float(returns[horizon]) for horizon in HORIZONS)
    downside = max(0.0, -min(values))
    return float(weighted - DOWNSIDE_PENALTY * downside - _action_cost(action))


def _oracle_label(utilities: dict[str, float]) -> str:
    rotate = utilities["ROTATE"]
    hold = utilities["HOLD"]
    cash = utilities["CASH"]
    if cash >= max(rotate, hold) + MIN_UTILITY_EDGE:
        return "CASH"
    if hold >= max(rotate, cash) + MIN_UTILITY_EDGE:
        return "HOLD"
    return "ROTATE"


def _model(seed: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("model", RandomForestClassifier(
            n_estimators=240,
            max_depth=4,
            min_samples_leaf=5,
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


def _choose_action(p_cash: float, p_hold: float) -> str:
    if p_cash >= ACTION_PROBABILITY_THRESHOLD:
        return "CASH"
    if p_hold >= ACTION_PROBABILITY_THRESHOLD:
        return "HOLD"
    return "ROTATE"


def _transition_rows(
    transition_attribution: dict[str, Any],
    risk: dict[str, Any],
    market_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    series = _series_by_symbol(market_rows)
    risk_rows = (((risk.get("oos") or {}).get("scored_transitions")) or [])
    risk_map = {
        (_stamp(item.get("execution_at")), str(item.get("from_asset") or "").upper(), str(item.get("to_asset") or "").upper()): item
        for item in risk_rows if isinstance(item, dict)
    }
    result: list[dict[str, Any]] = []
    for transition in transition_attribution.get("items") or []:
        if not isinstance(transition, dict):
            continue
        execution = _stamp(transition.get("execution_at"))
        if execution is None:
            continue
        incumbent = str(transition.get("from_asset") or "CASH").upper()
        challenger = str(transition.get("to_asset") or "CASH").upper()
        action_returns: dict[str, dict[int, float | None]] = {action: {} for action in ACTIONS}
        for horizon in HORIZONS:
            action_returns["ROTATE"][horizon] = 0.0 if challenger == "CASH" else _horizon_return(series.get(challenger), execution, horizon)
            action_returns["HOLD"][horizon] = 0.0 if incumbent == "CASH" else _horizon_return(series.get(incumbent), execution, horizon)
            action_returns["CASH"][horizon] = 0.0
        utilities = {action: _utility(action, action_returns[action]) for action in ACTIONS}
        if any(value is None for value in utilities.values()):
            continue
        risk_row = risk_map.get((execution, incumbent, challenger)) or {}
        one_interval = transition.get("one_interval_outcome") if isinstance(transition.get("one_interval_outcome"), dict) else {}
        row = {
            "transition_key": f"{execution.isoformat()}|{incumbent}|{challenger}",
            "decision_at": transition.get("decision_at"),
            "execution_at": transition.get("execution_at"),
            "year": int(execution.year),
            "from_asset": incumbent,
            "to_asset": challenger,
            "fold_id": transition.get("fold_id"),
            "risk_score": risk_row.get("risk_score"),
            "risk_threshold": risk_row.get("risk_threshold"),
            "risk_high": bool(risk_row.get("high_risk")),
            "severe": bool(risk_row.get("severe")),
            "one_interval_target_return": one_interval.get("target_return"),
            "one_interval_incumbent_return": one_interval.get("incumbent_return"),
            "oracle_action": _oracle_label({action: float(utilities[action]) for action in ACTIONS}),
            **_feature_row(transition),
        }
        for action in ACTIONS:
            row[f"utility_{action.lower()}"] = float(utilities[action])
            for horizon in HORIZONS:
                row[f"{action.lower()}_return_{horizon}d"] = action_returns[action][horizon]
        result.append(row)
    result.sort(key=lambda row: _stamp(row.get("execution_at")) or pd.Timestamp.min.tz_localize("UTC"))
    return result


def _walk_forward_predictions(rows: list[dict[str, Any]], first_test_year: int, last_test_year: int) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    output: list[dict[str, Any]] = []
    for test_year in range(int(first_test_year), int(last_test_year) + 1):
        train = frame[frame["year"] < test_year].copy()
        test = frame[frame["year"] == test_year].copy()
        if len(train) < 30 or test.empty:
            continue
        cash_model = _model(RANDOM_STATE + 1)
        hold_model = _model(RANDOM_STATE + 2)
        cash_target = (train["oracle_action"] == "CASH").astype(int)
        hold_target = (train["oracle_action"] == "HOLD").astype(int)
        cash_model.fit(train[list(FEATURES)], cash_target)
        hold_model.fit(train[list(FEATURES)], hold_target)
        p_cash = _positive_probability(cash_model, test[list(FEATURES)])
        p_hold = _positive_probability(hold_model, test[list(FEATURES)])
        for (_, source), cash_probability, hold_probability in zip(test.iterrows(), p_cash, p_hold):
            action = _choose_action(float(cash_probability), float(hold_probability))
            rotate_utility = float(source["utility_rotate"])
            selected_utility = float(source[f"utility_{action.lower()}"])
            output.append({
                "transition_key": source["transition_key"],
                "decision_at": source["decision_at"],
                "execution_at": source["execution_at"],
                "test_year": int(test_year),
                "from_asset": source["from_asset"],
                "to_asset": source["to_asset"],
                "risk_high": bool(source.get("risk_high")),
                "risk_score": _finite(source.get("risk_score")),
                "oracle_action": source["oracle_action"],
                "policy_action": action,
                "p_cash": float(cash_probability),
                "p_hold": float(hold_probability),
                "threshold": ACTION_PROBABILITY_THRESHOLD,
                "selected_utility": selected_utility,
                "rotate_utility": rotate_utility,
                "utility_edge_vs_rotate": float(selected_utility - rotate_utility),
                "one_interval_target_return": _finite(source.get("one_interval_target_return")),
                "one_interval_incumbent_return": _finite(source.get("one_interval_incumbent_return")),
            })
    return output


def _replay(
    analytics: dict[str, Any],
    predictions: list[dict[str, Any]],
    years: set[int] | None = None,
) -> dict[str, Any]:
    equity = _equity_rows(analytics, years)
    if not equity:
        return {}
    by_rejoin: dict[pd.Timestamp, list[dict[str, Any]]] = {}
    effects: list[dict[str, Any]] = []
    for prediction in predictions:
        action = str(prediction.get("policy_action") or "ROTATE")
        if action == "ROTATE":
            continue
        execution = _stamp(prediction.get("execution_at"))
        if execution is None or (years is not None and int(execution.year) not in years):
            continue
        target = _finite(prediction.get("one_interval_target_return"))
        incumbent = _finite(prediction.get("one_interval_incumbent_return"))
        if target is None or 1.0 + target <= 1e-9:
            continue
        if action == "HOLD":
            if incumbent is None:
                continue
            factor = float((1.0 + incumbent) / (1.0 + target))
        else:
            factor = float(1.0 / (1.0 + target))
        rejoin = next((item for item in equity if item["stamp"] > execution), None)
        if rejoin is None:
            continue
        if years is not None and int(rejoin["stamp"].year) not in years:
            continue
        effect = {
            **prediction,
            "rejoin_at": rejoin["timestamp"],
            "capital_factor": factor,
        }
        effects.append(effect)
        by_rejoin.setdefault(rejoin["stamp"], []).append(effect)
    starting_value = _starting_value(analytics, equity[0]["stamp"], float(equity[0]["value"]))
    baseline_path: list[dict[str, Any]] = []
    candidate_path: list[dict[str, Any]] = []
    cumulative = 1.0
    for row in equity:
        for effect in by_rejoin.get(row["stamp"], []):
            cumulative *= float(effect["capital_factor"])
        baseline_path.append({"timestamp": row["timestamp"], "value": float(row["value"]), "starting_value": starting_value})
        candidate_path.append({"timestamp": row["timestamp"], "value": float(row["value"] * cumulative), "starting_value": starting_value})
    baseline = _path_stats(baseline_path)
    candidate = _path_stats(candidate_path)
    base_end = _finite(baseline.get("ending_capital"))
    candidate_end = _finite(candidate.get("ending_capital"))
    return {
        "method": "one_session_action_override_then_rejoin",
        "interventions": len(effects),
        "effects": effects,
        "baseline": baseline,
        "candidate": {
            **candidate,
            "ending_capital_delta": None if base_end is None or candidate_end is None else float(candidate_end - base_end),
            "ending_capital_delta_rate": None if base_end in {None, 0.0} or candidate_end is None else float(candidate_end / base_end - 1.0),
        },
        "equity": {"baseline": baseline_path, "candidate": candidate_path},
    }


def _gate(name: str, passed: bool, observed: Any, requirement: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "observed": observed, "requirement": requirement}


def build_analysis(
    *,
    transition_attribution: dict[str, Any],
    risk: dict[str, Any],
    market_rows: list[dict[str, Any]],
    analytics: dict[str, Any],
    run_id: str,
    processing_id: str,
    period_start: str,
    period_end: str,
) -> dict[str, Any]:
    rows = _transition_rows(transition_attribution, risk, market_rows)
    oos = risk.get("oos") if isinstance(risk.get("oos"), dict) else {}
    first_test_year = int(oos.get("first_test_year") or 2022)
    last_test_year = int(oos.get("last_test_year") or max((row["year"] for row in rows), default=first_test_year))
    predictions = _walk_forward_predictions(rows, first_test_year, last_test_year)
    replay = _replay(analytics, predictions)
    yearly: list[dict[str, Any]] = []
    positive_years = 0
    for year in range(first_test_year, last_test_year + 1):
        year_predictions = [row for row in predictions if int(row.get("test_year") or 0) == year]
        year_replay = _replay(analytics, year_predictions, years={year})
        delta = _finite(((year_replay.get("candidate") or {}).get("ending_capital_delta_rate")))
        if delta is not None and delta > 0:
            positive_years += 1
        counts = Counter(str(row.get("policy_action") or "ROTATE") for row in year_predictions)
        yearly.append({
            "test_year": year,
            "transitions": len(year_predictions),
            "interventions": int(sum(counts[action] for action in ("HOLD", "CASH"))),
            "actions": {action: int(counts.get(action, 0)) for action in ACTIONS},
            "ending_capital_delta_rate": delta,
            "candidate_ending_capital": ((year_replay.get("candidate") or {}).get("ending_capital")),
            "control_ending_capital": ((year_replay.get("baseline") or {}).get("ending_capital")),
        })
    candidate = replay.get("candidate") if isinstance(replay.get("candidate"), dict) else {}
    baseline = replay.get("baseline") if isinstance(replay.get("baseline"), dict) else {}
    capital_lift = _finite(candidate.get("ending_capital_delta_rate")) or 0.0
    candidate_sharpe = _finite(candidate.get("sharpe"))
    baseline_sharpe = _finite(baseline.get("sharpe"))
    candidate_dd = _finite(candidate.get("maximum_drawdown"))
    baseline_dd = _finite(baseline.get("maximum_drawdown"))
    candidate_worst = candidate.get("worst_month") if isinstance(candidate.get("worst_month"), dict) else {}
    baseline_worst = baseline.get("worst_month") if isinstance(baseline.get("worst_month"), dict) else {}
    candidate_worst_return = _finite(candidate_worst.get("return"))
    baseline_worst_return = _finite(baseline_worst.get("return"))
    intervention_count = int(replay.get("interventions") or 0)
    utility_edges = [_finite(row.get("utility_edge_vs_rotate")) for row in predictions if str(row.get("policy_action")) != "ROTATE"]
    utility_edges = [value for value in utility_edges if value is not None]
    mean_utility_edge = float(sum(utility_edges) / len(utility_edges)) if utility_edges else 0.0
    gates = [
        _gate("capital_lift", capital_lift >= MIN_CAPITAL_LIFT, capital_lift, f">= {MIN_CAPITAL_LIFT:.2%}"),
        _gate("minimum_interventions", intervention_count >= MIN_INTERVENTIONS, intervention_count, f">= {MIN_INTERVENTIONS}"),
        _gate("positive_oos_years", positive_years >= MIN_POSITIVE_OOS_YEARS, positive_years, f">= {MIN_POSITIVE_OOS_YEARS}"),
        _gate(
            "sharpe_safety",
            candidate_sharpe is not None and baseline_sharpe is not None and candidate_sharpe >= baseline_sharpe - MAX_SHARPE_DEGRADATION,
            None if candidate_sharpe is None or baseline_sharpe is None else float(candidate_sharpe - baseline_sharpe),
            f">= -{MAX_SHARPE_DEGRADATION:.3f}",
        ),
        _gate(
            "drawdown_safety",
            candidate_dd is not None and baseline_dd is not None and candidate_dd >= baseline_dd - MAX_DRAWDOWN_DEGRADATION,
            None if candidate_dd is None or baseline_dd is None else float(candidate_dd - baseline_dd),
            f">= -{MAX_DRAWDOWN_DEGRADATION:.2%}",
        ),
        _gate(
            "worst_month_safety",
            candidate_worst_return is not None and baseline_worst_return is not None and candidate_worst_return >= baseline_worst_return - MAX_WORST_MONTH_DEGRADATION,
            None if candidate_worst_return is None or baseline_worst_return is None else float(candidate_worst_return - baseline_worst_return),
            f">= -{MAX_WORST_MONTH_DEGRADATION:.2%}",
        ),
        _gate("positive_intervention_utility", intervention_count > 0 and mean_utility_edge > 0.0, mean_utility_edge, "> 0"),
    ]
    approved = all(bool(gate["passed"]) for gate in gates)
    counts = Counter(str(row.get("policy_action") or "ROTATE") for row in predictions)
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "status": "completed",
        "run_id": str(run_id),
        "processing_id": str(processing_id),
        "period_start": str(period_start),
        "period_end": str(period_end),
        "protocol": {
            "purpose": "final shadow qualification before operational policy activation",
            "training_scope": "all prior chronological transitions",
            "test_years": [first_test_year, last_test_year],
            "model": "hierarchical_random_forest_pairwise",
            "default_action": "ROTATE",
            "action_probability_threshold": ACTION_PROBABILITY_THRESHOLD,
            "candidate_actions": list(ACTIONS),
            "horizons_sessions": list(HORIZONS),
            "utility_weights": {str(key): value for key, value in UTILITY_WEIGHTS.items()},
            "downside_penalty": DOWNSIDE_PENALTY,
            "one_side_cost_bps": ONE_SIDE_COST_BPS,
            "minimum_utility_edge": MIN_UTILITY_EDGE,
            "replay_semantics": "one_session_override_then_rejoin_and_recheck",
            "future_information_in_features": False,
            "future_information_in_training_labels": True,
            "strategy_decisions_changed": False,
            "shadow_only": True,
            "threshold_tuned_on_test_data": False,
        },
        "decision": {
            "status": "approved" if approved else "rejected",
            "operationalize_next_release": bool(approved),
            "fallback_if_rejected": "preserve_original_strategy",
            "reason": "All frozen qualification gates passed." if approved else "At least one frozen qualification gate failed; preserve the original Strategy and end this selector research line.",
        },
        "summary": {
            "transition_rows": len(rows),
            "oos_predictions": len(predictions),
            "actions": {action: int(counts.get(action, 0)) for action in ACTIONS},
            "interventions": intervention_count,
            "positive_oos_years": positive_years,
            "mean_intervention_utility_edge_vs_rotate": mean_utility_edge,
            "control_ending_capital": baseline.get("ending_capital"),
            "candidate_ending_capital": candidate.get("ending_capital"),
            "ending_capital_delta": candidate.get("ending_capital_delta"),
            "ending_capital_delta_rate": candidate.get("ending_capital_delta_rate"),
            "control_sharpe": baseline.get("sharpe"),
            "candidate_sharpe": candidate.get("sharpe"),
            "control_maximum_drawdown": baseline.get("maximum_drawdown"),
            "candidate_maximum_drawdown": candidate.get("maximum_drawdown"),
            "control_worst_month": baseline.get("worst_month"),
            "candidate_worst_month": candidate.get("worst_month"),
        },
        "gates": gates,
        "yearly_oos": yearly,
        "predictions": predictions,
        "replay": replay,
    }
