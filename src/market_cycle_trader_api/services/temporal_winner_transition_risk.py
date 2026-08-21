from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..infrastructure.persistence.mongo_repository import (
    TEMPORAL_WINNER_TRANSITION_RISK_RESEARCH_COLLECTION,
    bson_value,
    utc_now,
)
from .analytics import processing_analytics
from .temporal_winner_transition_attribution import get_winner_transition_attribution
from .temporal_research_settings import temporal_research_settings_snapshot

FAMILY_FEATURES = {
    "temporal_rejection": (
        "temporal_3d_opportunity_gate_mean",
        "temporal_3d_entry_rank_mean",
        "temporal_3d_short_profit_mean",
    ),
    "fragile_leader": (
        "winner_5d_target_top1_rate",
        "winner_5d_incumbent_top1_rate",
        "winner_5d_target_top1_consecutive",
        "winner_5d_leader_change_count",
        "winner_5d_top1_top2_gap_latest",
        "winner_5d_target_minus_incumbent_score_latest",
        "winner_5d_target_minus_incumbent_score_delta",
        "temporal_3d_long_trend_mean",
        "temporal_3d_horizon_agreement_mean",
    ),
}
FAMILY_FEATURES["combined"] = tuple(dict.fromkeys(FAMILY_FEATURES["temporal_rejection"] + FAMILY_FEATURES["fragile_leader"]))


class WinnerTransitionRiskError(RuntimeError):
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


def _transition_key(executed_at: Any, from_asset: Any, to_asset: Any) -> tuple[str, str, str] | None:
    stamp = _timestamp(executed_at)
    if stamp is None:
        return None
    source = str(from_asset or "").strip().upper()
    target = str(to_asset or "").strip().upper()
    if not source or not target:
        return None
    return (stamp.isoformat(), source, target)


def _mean(values: list[Any]) -> float | None:
    clean = [_finite(value) for value in values]
    finite = [value for value in clean if value is not None]
    return float(sum(finite) / len(finite)) if finite else None


def _temporal_mean(sessions: list[dict[str, Any]], field: str, window: int = 3) -> float | None:
    values = []
    for session in sessions[-window:]:
        temporal = session.get("temporal") if isinstance(session.get("temporal"), dict) else {}
        delta = temporal.get("target_minus_incumbent") if isinstance(temporal.get("target_minus_incumbent"), dict) else {}
        values.append(delta.get(field))
    return _mean(values)


def _features(transition: dict[str, Any]) -> dict[str, float | None]:
    trajectory = transition.get("trajectory") if isinstance(transition.get("trajectory"), dict) else {}
    sessions = [row for row in trajectory.get("sessions") or [] if isinstance(row, dict)]
    windows = trajectory.get("windows") if isinstance(trajectory.get("windows"), dict) else {}
    winner_5d = windows.get("5") if isinstance(windows.get("5"), dict) else {}
    return {
        "temporal_3d_opportunity_gate_mean": _temporal_mean(sessions, "opportunity_gate_score"),
        "temporal_3d_entry_rank_mean": _temporal_mean(sessions, "entry_rank_score"),
        "temporal_3d_short_profit_mean": _temporal_mean(sessions, "short_profit_consensus"),
        "temporal_3d_long_trend_mean": _temporal_mean(sessions, "long_trend_support"),
        "temporal_3d_horizon_agreement_mean": _temporal_mean(sessions, "horizon_agreement"),
        "winner_5d_target_top1_rate": _finite(winner_5d.get("target_top1_rate")),
        "winner_5d_incumbent_top1_rate": _finite(winner_5d.get("incumbent_top1_rate")),
        "winner_5d_target_top1_consecutive": _finite(winner_5d.get("target_top1_consecutive")),
        "winner_5d_leader_change_count": _finite(winner_5d.get("leader_change_count")),
        "winner_5d_top1_top2_gap_latest": _finite(winner_5d.get("top1_top2_gap_latest")),
        "winner_5d_target_minus_incumbent_score_latest": _finite(winner_5d.get("target_minus_incumbent_score_latest")),
        "winner_5d_target_minus_incumbent_score_delta": _finite(winner_5d.get("target_minus_incumbent_score_delta")),
    }


def build_transition_risk_dataset(
    transition_attribution: dict[str, Any],
    rotations: list[dict[str, Any]],
    *,
    severe_threshold: float,
) -> list[dict[str, Any]]:
    rotation_map: dict[tuple[str, str, str], dict[str, Any]] = {}
    for rotation in rotations:
        key = _transition_key(rotation.get("executed_at"), rotation.get("from_asset"), rotation.get("to_asset"))
        if key is not None:
            rotation_map[key] = rotation

    rows: list[dict[str, Any]] = []
    for transition in transition_attribution.get("items") or []:
        if not isinstance(transition, dict):
            continue
        key = _transition_key(transition.get("execution_at"), transition.get("from_asset"), transition.get("to_asset"))
        if key is None:
            continue
        rotation = rotation_map.get(key)
        holding_interval = transition.get("holding_interval_outcome") if isinstance(transition.get("holding_interval_outcome"), dict) else {}
        rotation_value_added = _finite(rotation.get("rotation_value_added")) if rotation else None
        attributed_value_added = _finite(holding_interval.get("value_added"))
        value_added = rotation_value_added if rotation_value_added is not None else attributed_value_added
        if value_added is None:
            continue
        stamp = _timestamp(transition.get("execution_at"))
        if stamp is None:
            continue
        interval = transition.get("one_interval_outcome") if isinstance(transition.get("one_interval_outcome"), dict) else {}
        row = {
            "transition_key": "|".join(key),
            "decision_at": transition.get("decision_at"),
            "execution_at": transition.get("execution_at"),
            "year": int(stamp.year),
            "month": int(stamp.month),
            "from_asset": str(transition.get("from_asset") or "").upper(),
            "to_asset": str(transition.get("to_asset") or "").upper(),
            "rotation_value_added": value_added,
            "rotation_value_added_source": "analytics_rotation" if rotation_value_added is not None else "temporal_holding_interval",
            "severe": int(value_added <= float(severe_threshold)),
            "one_interval_value_added": _finite(interval.get("value_added")),
            "one_interval_target_return": _finite(interval.get("target_return")),
            "one_interval_incumbent_return": _finite(interval.get("incumbent_return")),
            "winner_top1_top2_score_gap": _finite(transition.get("winner_top1_top2_score_gap")),
        }
        row.update(_features(transition))
        rows.append(row)
    rows.sort(key=lambda row: _timestamp(row.get("execution_at")) or pd.Timestamp.min.tz_localize("UTC"))
    return rows


def _pipeline(seed: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=int(seed))),
    ])


def _can_fit(frame: pd.DataFrame, *, minimum_rows: int, min_train_severe: int) -> bool:
    if len(frame) < minimum_rows:
        return False
    counts = frame["severe"].value_counts()
    return int(counts.get(1, 0)) >= min_train_severe and int(counts.get(0, 0)) >= min_train_severe


def _auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def _classification_metrics(frame: pd.DataFrame, scores: np.ndarray, flags: np.ndarray) -> dict[str, Any]:
    labels = frame["severe"].to_numpy(dtype=int)
    severe_count = int(labels.sum())
    flagged_count = int(flags.sum())
    captured = int(((labels == 1) & flags).sum())
    precision = float(captured / flagged_count) if flagged_count else None
    recall = float(captured / severe_count) if severe_count else None
    f2 = None
    if precision is not None and recall is not None and 4.0 * precision + recall > 0:
        f2 = float(5.0 * precision * recall / (4.0 * precision + recall))
    value_added = frame["rotation_value_added"].to_numpy(dtype=float)
    flagged_values = value_added[flags]
    unflagged_values = value_added[~flags]
    return {
        "count": int(len(frame)),
        "severe_count": severe_count,
        "flagged_count": flagged_count,
        "captured_severe_count": captured,
        "alert_rate": float(flagged_count / len(frame)) if len(frame) else None,
        "precision": precision,
        "recall": recall,
        "f2": f2,
        "auc": _auc(labels, scores),
        "brier": float(brier_score_loss(labels, scores)) if len(np.unique(labels)) >= 2 else None,
        "mean_rotation_value_added_flagged": float(flagged_values.mean()) if len(flagged_values) else None,
        "mean_rotation_value_added_unflagged": float(unflagged_values.mean()) if len(unflagged_values) else None,
        "rotation_value_added_separation": (
            float(unflagged_values.mean() - flagged_values.mean())
            if len(flagged_values) and len(unflagged_values)
            else None
        ),
    }


def _fit_score(
    train: pd.DataFrame,
    test: pd.DataFrame,
    family: str,
    quantile: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    features = list(FAMILY_FEATURES[family])
    model = _pipeline(seed)
    model.fit(train[features], train["severe"].to_numpy(dtype=int))
    train_scores = model.predict_proba(train[features])[:, 1]
    test_scores = model.predict_proba(test[features])[:, 1]
    threshold = float(np.quantile(train_scores, quantile))
    flags = test_scores >= threshold
    return test_scores, flags, threshold


def _inner_predictions(
    train: pd.DataFrame,
    family: str,
    quantile: float,
    seed: int,
    *,
    min_inner_train_rows: int,
    min_train_severe: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray] | None:
    years = sorted(int(value) for value in train["year"].unique())
    pieces: list[pd.DataFrame] = []
    scores: list[np.ndarray] = []
    flags: list[np.ndarray] = []
    for year in years:
        inner_train = train[train["year"] < year]
        inner_test = train[train["year"] == year]
        if not len(inner_test) or not _can_fit(
            inner_train, minimum_rows=min_inner_train_rows, min_train_severe=min_train_severe
        ):
            continue
        fold_scores, fold_flags, _ = _fit_score(inner_train, inner_test, family, quantile, seed + year)
        pieces.append(inner_test)
        scores.append(fold_scores)
        flags.append(fold_flags)
    if not pieces:
        return None
    return pd.concat(pieces, axis=0), np.concatenate(scores), np.concatenate(flags)


def _select_candidate(
    train: pd.DataFrame,
    seed: int,
    *,
    risk_quantiles: list[float],
    default_risk_quantile: float,
    min_inner_train_rows: int,
    min_train_severe: int,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for family in FAMILY_FEATURES:
        for quantile in risk_quantiles:
            predicted = _inner_predictions(
                train,
                family,
                quantile,
                seed,
                min_inner_train_rows=min_inner_train_rows,
                min_train_severe=min_train_severe,
            )
            if predicted is None:
                continue
            frame, scores, flags = predicted
            metrics = _classification_metrics(frame, scores, flags)
            candidates.append({"family": family, "risk_quantile": quantile, "metrics": metrics})
    if not candidates:
        return {"family": "temporal_rejection", "risk_quantile": float(default_risk_quantile), "metrics": {}}
    return max(
        candidates,
        key=lambda item: (
            _finite(item["metrics"].get("f2")) or -1.0,
            _finite(item["metrics"].get("recall")) or -1.0,
            _finite(item["metrics"].get("auc")) or -1.0,
            -(_finite(item["metrics"].get("alert_rate")) or 1.0),
        ),
    )


def _family_oos(
    frame: pd.DataFrame,
    family: str,
    seed: int,
    *,
    quantile: float,
    min_outer_train_rows: int,
    min_train_severe: int,
) -> dict[str, Any]:
    pieces: list[pd.DataFrame] = []
    scores: list[np.ndarray] = []
    flags: list[np.ndarray] = []
    for year in sorted(int(value) for value in frame["year"].unique()):
        train = frame[frame["year"] < year]
        test = frame[frame["year"] == year]
        if not len(test) or not _can_fit(
            train, minimum_rows=min_outer_train_rows, min_train_severe=min_train_severe
        ):
            continue
        fold_scores, fold_flags, _ = _fit_score(train, test, family, quantile, seed + year)
        pieces.append(test)
        scores.append(fold_scores)
        flags.append(fold_flags)
    if not pieces:
        return {"family": family, "risk_quantile": quantile, "metrics": {}}
    joined = pd.concat(pieces, axis=0)
    return {
        "family": family,
        "risk_quantile": quantile,
        "metrics": _classification_metrics(joined, np.concatenate(scores), np.concatenate(flags)),
    }


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
    rows = []
    previous = None
    for month, group in frame.groupby("month", sort=True):
        end_value = float(group.iloc[-1]["value"])
        if previous is None:
            start_value = float(group.iloc[0]["starting_value"])
        else:
            start_value = previous
        rows.append({
            "month": str(month),
            "start_value": start_value,
            "end_value": end_value,
            "return": float(end_value / start_value - 1.0) if start_value else None,
        })
        previous = end_value
    return rows


def _shadow_replay(analytics: dict[str, Any], scored_rows: list[dict[str, Any]]) -> dict[str, Any]:
    equity = [dict(row) for row in analytics.get("equity") or [] if isinstance(row, dict)]
    rotations = [dict(row) for row in analytics.get("rotations") or [] if isinstance(row, dict)]
    equity = [row for row in equity if _timestamp(row.get("timestamp")) is not None and _finite(row.get("simulation_equity")) is not None]
    equity.sort(key=lambda row: _timestamp(row.get("timestamp")) or pd.Timestamp.min.tz_localize("UTC"))
    rotations = [row for row in rotations if _timestamp(row.get("executed_at")) is not None]
    rotations.sort(key=lambda row: _timestamp(row.get("executed_at")) or pd.Timestamp.min.tz_localize("UTC"))
    if not equity:
        return {}

    scored_map = {str(row.get("transition_key") or ""): row for row in scored_rows if row.get("high_risk")}
    effects: list[dict[str, Any]] = []
    skip_index: int | None = None
    for index, rotation in enumerate(rotations):
        if skip_index is not None and index == skip_index:
            skip_index = None
            continue
        key = _transition_key(rotation.get("executed_at"), rotation.get("from_asset"), rotation.get("to_asset"))
        encoded = "|".join(key) if key is not None else ""
        scored = scored_map.get(encoded)
        if scored is None or index + 1 >= len(rotations):
            continue
        control_return = _finite(rotation.get("subsequent_position_return"))
        incumbent_return = _finite(rotation.get("counterfactual_previous_asset_return"))
        if control_return is None or incumbent_return is None or 1.0 + control_return <= 1e-9:
            continue
        next_rotation = rotations[index + 1]
        effect_at = _timestamp(next_rotation.get("executed_at"))
        if effect_at is None:
            continue
        factor = float((1.0 + incumbent_return) / (1.0 + control_return))
        effects.append({
            "transition_key": encoded,
            "execution_at": rotation.get("executed_at"),
            "rejoin_at": next_rotation.get("executed_at"),
            "from_asset": rotation.get("from_asset"),
            "to_asset": rotation.get("to_asset"),
            "control_holding_return": control_return,
            "incumbent_counterfactual_return": incumbent_return,
            "rotation_value_added": _finite(rotation.get("rotation_value_added")),
            "capital_factor": factor,
            "risk_score": scored.get("risk_score"),
        })
        skip_index = index + 1

    base_equity = []
    shadow_equity = []
    cumulative_factor = 1.0
    effects_by_time: dict[pd.Timestamp, list[dict[str, Any]]] = {}
    for effect in effects:
        stamp = _timestamp(effect.get("rejoin_at"))
        if stamp is not None:
            effects_by_time.setdefault(stamp, []).append(effect)
    for row in equity:
        stamp = _timestamp(row.get("timestamp"))
        if stamp is None:
            continue
        for effect in effects_by_time.get(stamp, []):
            cumulative_factor *= float(effect["capital_factor"])
        base_value = float(row["simulation_equity"])
        base_equity.append({"timestamp": row.get("timestamp"), "value": base_value, "starting_value": float(equity[0]["simulation_equity"])})
        shadow_equity.append({"timestamp": row.get("timestamp"), "value": float(base_value * cumulative_factor), "starting_value": float(equity[0]["simulation_equity"])})

    base_monthly = _monthly_returns(base_equity)
    shadow_monthly = _monthly_returns(shadow_equity)
    base_values = [row["value"] for row in base_equity]
    shadow_values = [row["value"] for row in shadow_equity]
    base_worst = min(base_monthly, key=lambda item: item.get("return") if item.get("return") is not None else math.inf, default=None)
    shadow_worst = min(shadow_monthly, key=lambda item: item.get("return") if item.get("return") is not None else math.inf, default=None)
    base_end = base_values[-1]
    shadow_end = shadow_values[-1]
    return {
        "method": "stitched_expanding_walk_forward_keep_incumbent_until_next_control_transition_shadow",
        "interventions": len(effects),
        "effects": effects,
        "baseline": {
            "ending_capital": base_end,
            "maximum_drawdown": _max_drawdown(base_values),
            "worst_month": base_worst,
        },
        "shadow": {
            "ending_capital": shadow_end,
            "ending_capital_delta": float(shadow_end - base_end),
            "ending_capital_delta_rate": float(shadow_end / base_end - 1.0) if base_end else None,
            "maximum_drawdown": _max_drawdown(shadow_values),
            "worst_month": shadow_worst,
        },
        "monthly_returns": {
            "baseline": base_monthly,
            "shadow": shadow_monthly,
        },
        "equity": {
            "baseline": base_equity,
            "shadow": shadow_equity,
        },
    }


def run_transition_risk_search_from_payloads(
    *,
    run_id: str,
    processing_id: str,
    start_month: str,
    end_month: str,
    transition_attribution: dict[str, Any],
    analytics: dict[str, Any],
    research_settings: dict[str, Any],
    seed: int = 42,
) -> dict[str, Any]:
    settings_payload = research_settings.get("settings") if isinstance(research_settings.get("settings"), dict) else research_settings
    risk_settings = settings_payload.get("risk") if isinstance(settings_payload.get("risk"), dict) else {}
    severe_threshold = float(risk_settings["severe_threshold"])
    risk_quantiles = [float(value) for value in risk_settings["risk_quantiles"]]
    default_risk_quantile = float(risk_settings["default_risk_quantile"])
    min_outer_train_rows = int(risk_settings["min_outer_train_rows"])
    min_inner_train_rows = int(risk_settings["min_inner_train_rows"])
    min_train_severe = int(risk_settings["min_train_severe"])
    dataset = build_transition_risk_dataset(
        transition_attribution,
        list(analytics.get("rotations") or []),
        severe_threshold=severe_threshold,
    )
    if len(dataset) < min_outer_train_rows:
        attribution_items = [item for item in transition_attribution.get("items") or [] if isinstance(item, dict)]
        holding_outcomes = sum(
            1
            for item in attribution_items
            if isinstance(item.get("holding_interval_outcome"), dict)
            and _finite((item.get("holding_interval_outcome") or {}).get("value_added")) is not None
        )
        raise WinnerTransitionRiskError(
            f"Not enough attributed research transitions are available for chronological risk research: "
            f"found {len(dataset)}, requires at least {min_outer_train_rows}. "
            f"Attribution produced {len(attribution_items)} rotations and {holding_outcomes} complete holding outcomes."
        )
    frame = pd.DataFrame(dataset)
    outer_results: list[dict[str, Any]] = []
    scored_rows: list[dict[str, Any]] = []
    for year in sorted(int(value) for value in frame["year"].unique()):
        train = frame[frame["year"] < year]
        test = frame[frame["year"] == year]
        if not len(test) or not _can_fit(
            train, minimum_rows=min_outer_train_rows, min_train_severe=min_train_severe
        ):
            continue
        candidate = _select_candidate(
            train,
            seed + year,
            risk_quantiles=risk_quantiles,
            default_risk_quantile=default_risk_quantile,
            min_inner_train_rows=min_inner_train_rows,
            min_train_severe=min_train_severe,
        )
        family = str(candidate["family"])
        quantile = float(candidate["risk_quantile"])
        scores, flags, threshold = _fit_score(train, test, family, quantile, seed + year)
        metrics = _classification_metrics(test, scores, flags)
        holding_gains = []
        for (_, row), score, flagged in zip(test.iterrows(), scores, flags):
            item = row.to_dict()
            interval_value_added = _finite(item.get("one_interval_value_added"))
            if bool(flagged):
                holding_gains.append(-float(item["rotation_value_added"]))
            scored_rows.append({
                "transition_key": item["transition_key"],
                "decision_at": item["decision_at"],
                "execution_at": item["execution_at"],
                "from_asset": item["from_asset"],
                "to_asset": item["to_asset"],
                "year": int(item["year"]),
                "rotation_value_added": float(item["rotation_value_added"]),
                "severe": bool(item["severe"]),
                "risk_score": float(score),
                "high_risk": bool(flagged),
                "selected_family": family,
                "risk_quantile": quantile,
                "risk_threshold": threshold,
                "one_interval_value_added": interval_value_added,
                "one_interval_target_return": _finite(item.get("one_interval_target_return")),
                "one_interval_incumbent_return": _finite(item.get("one_interval_incumbent_return")),
                "one_interval_shadow_gain": -interval_value_added if bool(flagged) and interval_value_added is not None else 0.0,
            })
        outer_results.append({
            "test_year": year,
            "train_start_year": int(train["year"].min()),
            "train_end_year": int(train["year"].max()),
            "train_count": int(len(train)),
            "train_severe_count": int(train["severe"].sum()),
            "selected_family": family,
            "risk_quantile": quantile,
            "risk_threshold": threshold,
            "inner_selection": candidate,
            "metrics": metrics,
            "holding_period_shadow_gain_mean": float(np.mean(holding_gains)) if holding_gains else None,
            "holding_period_shadow_gain_sum": float(np.sum(holding_gains)) if holding_gains else 0.0,
        })
    if not scored_rows:
        raise WinnerTransitionRiskError("Chronological outer folds could not be formed from the available transition history.")

    scored_frame = pd.DataFrame(scored_rows)
    oos_scores = scored_frame["risk_score"].to_numpy(dtype=float)
    oos_flags = scored_frame["high_risk"].to_numpy(dtype=bool)
    overall = _classification_metrics(scored_frame, oos_scores, oos_flags)
    shadow = _shadow_replay(analytics, scored_rows)
    family_comparison = [
        _family_oos(
            frame,
            family,
            seed,
            quantile=default_risk_quantile,
            min_outer_train_rows=min_outer_train_rows,
            min_train_severe=min_train_severe,
        )
        for family in FAMILY_FEATURES
    ]

    final_candidate = _select_candidate(
        frame,
        seed + 100000,
        risk_quantiles=risk_quantiles,
        default_risk_quantile=default_risk_quantile,
        min_inner_train_rows=min_inner_train_rows,
        min_train_severe=min_train_severe,
    )
    final_family = str(final_candidate["family"])
    final_quantile = float(final_candidate["risk_quantile"])
    final_model = _pipeline(seed + 100001)
    final_features = list(FAMILY_FEATURES[final_family])
    final_model.fit(frame[final_features], frame["severe"].to_numpy(dtype=int))
    full_scores = final_model.predict_proba(frame[final_features])[:, 1]
    final_threshold = float(np.quantile(full_scores, final_quantile))

    severe_rows = scored_frame[scored_frame["severe"]].sort_values("rotation_value_added")
    high_risk_rows = scored_frame[scored_frame["high_risk"]].sort_values("risk_score", ascending=False)
    june_2026 = scored_frame[(scored_frame["year"] == 2026) & (pd.to_datetime(scored_frame["execution_at"], utc=True).dt.month == 6)].sort_values("execution_at")

    return bson_value({
        "schema_version": 1,
        "id": f"winner-transition-risk-{utc_now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}",
        "run_id": str(run_id),
        "processing_id": str(processing_id),
        "period_start": start_month,
        "period_end": end_month,
        "created_at": utc_now(),
        "status": "completed",
        "research_settings": research_settings,
        "research_target": {
            "name": "severe_winner_anchor_transition",
            "rotation_value_added_lte": severe_threshold,
            "intervention": "keep_incumbent_until_next_control_transition_shadow",
        },
        "protocol": {
            "validation": "expanding_chronological_yearly_outer_folds",
            "outer_min_train_rows": min_outer_train_rows,
            "inner_min_train_rows": min_inner_train_rows,
            "min_train_severe": min_train_severe,
            "risk_quantiles": risk_quantiles,
            "default_risk_quantile": default_risk_quantile,
            "seed": int(seed),
            "future_information_in_features": False,
            "shadow_replay_research_only": True,
            "shadow_rejoin": "next_control_transition",
            "overlapping_alerts_suppressed": True,
        },
        "dataset": {
            "count": int(len(frame)),
            "severe_count": int(frame["severe"].sum()),
            "severe_rate": float(frame["severe"].mean()),
            "first_execution_at": frame.iloc[0]["execution_at"],
            "last_execution_at": frame.iloc[-1]["execution_at"],
        },
        "family_comparison": family_comparison,
        "outer_results": outer_results,
        "oos": {
            "metrics": overall,
            "scored_count": int(len(scored_frame)),
            "first_test_year": int(scored_frame["year"].min()),
            "last_test_year": int(scored_frame["year"].max()),
            "scored_transitions": scored_frame.sort_values("execution_at").to_dict(orient="records"),
            "high_risk_transitions": high_risk_rows.head(30).to_dict(orient="records"),
            "severe_transitions": severe_rows.head(30).to_dict(orient="records"),
            "june_2026": june_2026.to_dict(orient="records"),
        },
        "shadow_replay": shadow,
        "final_refit": {
            "validation_role": "research_only_not_oos_evidence",
            "selected_family": final_family,
            "features": final_features,
            "risk_quantile": final_quantile,
            "risk_threshold": final_threshold,
            "inner_selection": final_candidate,
        },
    })


def run_winner_transition_risk_search(
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
    research_settings = temporal_research_settings_snapshot(db)
    try:
        result = run_transition_risk_search_from_payloads(
            run_id=run_id,
            processing_id=processing_id,
            start_month=start_month,
            end_month=end_month,
            transition_attribution=attribution,
            analytics=analytics,
            research_settings=research_settings,
            seed=seed,
        )
    except WinnerTransitionRiskError as exc:
        failed = bson_value({
            "schema_version": 2,
            "id": f"winner-transition-risk-{utc_now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}",
            "run_id": str(run_id),
            "processing_id": str(processing_id),
            "period_start": str(start_month),
            "period_end": str(end_month),
            "created_at": utc_now(),
            "status": "failed",
            "failure_message": str(exc),
            "attribution_count": int(attribution.get("count") or 0),
            "research_settings": research_settings,
        })
        db[TEMPORAL_WINNER_TRANSITION_RISK_RESEARCH_COLLECTION].insert_one(dict(failed))
        raise
    db[TEMPORAL_WINNER_TRANSITION_RISK_RESEARCH_COLLECTION].insert_one(dict(result))
    return result


def get_latest_winner_transition_risk_search(
    db: Any,
    run_id: str,
    *,
    processing_id: str,
    start_month: str,
    end_month: str,
) -> dict[str, Any] | None:
    row = db[TEMPORAL_WINNER_TRANSITION_RISK_RESEARCH_COLLECTION].find_one(
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
    return bson_value(row) if row is not None else None
