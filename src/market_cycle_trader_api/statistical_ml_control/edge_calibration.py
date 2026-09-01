from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any

import numpy as np
import pandas as pd

from .analysis import (
    OPEN_FEATURES,
    OPEN_MODEL_FEATURES,
    _binary_metrics,
    _finite,
    _mean_available,
    _model,
    _positive_probability,
    _prepare_rows,
    _regime_context_frames,
    _replay,
    _rolling_regime_source,
    _select_threshold,
)


EDGE_CALIBRATION_VERSION = "1.0.0"
DEFAULT_EDGE_CANDIDATES = (0.0, 0.001, 0.0025, 0.005, 0.0075, 0.01)


def _float_setting(settings: dict[str, Any], name: str, default: float) -> float:
    value = settings.get(name)
    return float(default if value is None else value)


def _int_setting(settings: dict[str, Any], name: str, default: int) -> int:
    value = settings.get(name)
    return int(default if value is None else value)


def _candidate_edges(configured: float) -> list[float]:
    return sorted({float(value) for value in (*DEFAULT_EDGE_CANDIDATES, configured) if 0.0 <= float(value) <= 1.0})


def _target_series(frame: pd.DataFrame, target: str, edge: float) -> pd.Series:
    asset = pd.to_numeric(frame["asset_utility"], errors="coerce")
    alternative = pd.to_numeric(frame["alternative_utility"], errors="coerce")
    alternative_available = frame["alternative_symbol"].fillna("").astype(str).str.strip().ne("") & alternative.notna()
    if target == "target_cash":
        values = asset.le(-float(edge)) & (~alternative_available | alternative.le(-float(edge)))
    elif target == "target_rotate":
        values = alternative_available & alternative.ge(asset + float(edge)) & alternative.ge(float(edge))
    else:
        raise ValueError(f"Unsupported calibration target: {target}")
    return values.fillna(False).astype(int)


def _with_target(frame: pd.DataFrame, target: str, edge: float) -> pd.DataFrame:
    result = frame.copy()
    result[target] = _target_series(result, target, edge)
    return result


def _positive_rate(frame: pd.DataFrame, target: str) -> float | None:
    if frame.empty:
        return None
    return float(pd.to_numeric(frame[target], errors="coerce").fillna(0).astype(int).mean())


def _inner_candidate(
    *,
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    features: tuple[str, ...],
    target: str,
    edge: float,
    settings: dict[str, Any],
    seed_offset: int,
    default_threshold: float,
) -> dict[str, Any]:
    fit_edge = _with_target(fit, target, edge)
    validation_edge = _with_target(validation, target, edge)
    fit_classes = int(fit_edge[target].nunique()) if not fit_edge.empty else 0
    validation_classes = int(validation_edge[target].nunique()) if not validation_edge.empty else 0
    record: dict[str, Any] = {
        "edge": float(edge),
        "fit_rows": int(len(fit_edge)),
        "validation_rows": int(len(validation_edge)),
        "fit_positive_rate": _positive_rate(fit_edge, target),
        "validation_positive_rate": _positive_rate(validation_edge, target),
        "fit_classes": fit_classes,
        "validation_classes": validation_classes,
        "eligible": False,
        "threshold": float(default_threshold),
        "balanced_accuracy": None,
        "auc": None,
        "brier": None,
    }
    if fit_edge.empty or validation_edge.empty or fit_classes < 2 or validation_classes < 2:
        record["ineligible_reason"] = "both classes are required in fit and inner validation"
        return record

    model = _model(settings, seed_offset)
    model.fit(fit_edge[list(features)], fit_edge[target].astype(int))
    probabilities = _positive_probability(model, validation_edge[list(features)])
    threshold, validation_balanced = _select_threshold(validation_edge[target], probabilities, settings)
    metrics = _binary_metrics(validation_edge[target], probabilities, threshold, origin="edge_inner_validation")
    record.update({
        "eligible": metrics.get("auc") is not None,
        "threshold": float(threshold),
        "balanced_accuracy": validation_balanced,
        "auc": _finite(metrics.get("auc")),
        "brier": _finite(metrics.get("brier")),
    })
    return record


def _selection_key(record: dict[str, Any], configured_edge: float) -> tuple[float, float, float, float, float]:
    if not bool(record.get("eligible")):
        return (-math.inf, -math.inf, -math.inf, -math.inf, -math.inf)
    balanced = _finite(record.get("balanced_accuracy"))
    auc = _finite(record.get("auc"))
    brier = _finite(record.get("brier"))
    edge = float(record["edge"])
    return (
        -math.inf if balanced is None else balanced,
        -math.inf if auc is None else auc,
        -math.inf if brier is None else -brier,
        -abs(edge - float(configured_edge)),
        -edge,
    )


def _select_edge(records: list[dict[str, Any]], configured_edge: float) -> dict[str, Any]:
    eligible = [record for record in records if bool(record.get("eligible"))]
    if eligible:
        selected = max(eligible, key=lambda record: _selection_key(record, configured_edge))
        return {**selected, "selection_origin": "inner_validation"}
    fallback = next((record for record in records if abs(float(record["edge"]) - float(configured_edge)) <= 1e-12), None)
    if fallback is None:
        fallback = {"edge": float(configured_edge), "threshold": 0.65}
    return {**fallback, "selection_origin": "configured_fallback"}


def _fit_outer_model(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: tuple[str, ...],
    target: str,
    edge: float,
    threshold: float,
    settings: dict[str, Any],
    seed_offset: int,
) -> tuple[np.ndarray, dict[str, Any], pd.DataFrame]:
    train_edge = _with_target(train, target, edge)
    test_edge = _with_target(test, target, edge)
    if train_edge[target].nunique() < 2:
        probability = float(train_edge[target].mean()) if len(train_edge) else 0.0
        probabilities = np.full(len(test_edge), probability, dtype=float)
        metrics = _binary_metrics(test_edge[target], probabilities, threshold, origin="constant_outer_training_class")
        return probabilities, metrics, test_edge
    model = _model(settings, seed_offset)
    model.fit(train_edge[list(features)], train_edge[target].astype(int))
    probabilities = _positive_probability(model, test_edge[list(features)])
    metrics = _binary_metrics(test_edge[target], probabilities, threshold, origin="nested_edge_inner_validation")
    return probabilities, metrics, test_edge


def _aggregate_grid(folds: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    selected_counts: Counter = Counter()
    for fold in folds:
        selected = ((fold.get(key) or {}).get("selected") or {})
        if selected.get("edge") is not None:
            selected_counts[float(selected["edge"])] += 1
        for row in ((fold.get(key) or {}).get("candidates") or []):
            grouped[float(row["edge"])].append(row)
    output: list[dict[str, Any]] = []
    for edge, rows in sorted(grouped.items()):
        output.append({
            "edge": edge,
            "folds": int(len(rows)),
            "eligible_folds": int(sum(bool(row.get("eligible")) for row in rows)),
            "selected_folds": int(selected_counts.get(edge, 0)),
            "mean_fit_positive_rate": _mean_available([_finite(row.get("fit_positive_rate")) for row in rows]),
            "mean_validation_positive_rate": _mean_available([_finite(row.get("validation_positive_rate")) for row in rows]),
            "mean_validation_balanced_accuracy": _mean_available([_finite(row.get("balanced_accuracy")) for row in rows]),
            "mean_validation_auc": _mean_available([_finite(row.get("auc")) for row in rows]),
            "mean_validation_brier": _mean_available([_finite(row.get("brier")) for row in rows]),
        })
    return output


def build_edge_calibration(
    *,
    reference_rows: list[dict[str, Any]],
    observation_rows: list[dict[str, Any]],
    settings: dict[str, Any],
) -> dict[str, Any]:
    rows = _prepare_rows(reference_rows, observation_rows, settings)
    if not rows:
        raise ValueError("Edge calibration requires prepared Statistical ML rows.")
    frame = _rolling_regime_source(pd.DataFrame(rows), settings)
    if frame.empty:
        raise ValueError("Edge calibration has no chronological rows.")

    configured_cash_edge = _float_setting(settings, "minimum_cash_edge", 0.005)
    configured_rotation_edge = _float_setting(settings, "minimum_rotation_edge", 0.005)
    cash_candidates = _candidate_edges(configured_cash_edge)
    rotation_candidates = _candidate_edges(configured_rotation_edge)
    first_test_year = _int_setting(settings, "first_test_year", max(int(frame["year"].min()) + 2, 2022))
    last_test_year = int(frame["year"].max())
    minimum_train = max(30, _int_setting(settings, "min_train_rows", 250))
    validation_share = min(max(_float_setting(settings, "inner_validation_share", 0.20), 0.10), 0.40)
    default_cash_threshold = _float_setting(settings, "default_probability_threshold", 0.65)
    default_rotation_threshold = _float_setting(settings, "default_rotation_probability_threshold", default_cash_threshold)
    arbitration_margin = _float_setting(settings, "action_probability_margin", 0.05)

    folds: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
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

        train, fit, validation, test, regime_context = _regime_context_frames(train, fit, validation, test, settings)
        features = OPEN_MODEL_FEATURES if regime_context.get("enabled") else OPEN_FEATURES

        cash_grid = [
            _inner_candidate(
                fit=fit,
                validation=validation,
                features=features,
                target="target_cash",
                edge=edge,
                settings=settings,
                seed_offset=101_000 + test_year * 100 + index,
                default_threshold=default_cash_threshold,
            )
            for index, edge in enumerate(cash_candidates)
        ]
        rotation_grid = [
            _inner_candidate(
                fit=fit,
                validation=validation,
                features=features,
                target="target_rotate",
                edge=edge,
                settings=settings,
                seed_offset=201_000 + test_year * 100 + index,
                default_threshold=default_rotation_threshold,
            )
            for index, edge in enumerate(rotation_candidates)
        ]
        selected_cash = _select_edge(cash_grid, configured_cash_edge)
        selected_rotation = _select_edge(rotation_grid, configured_rotation_edge)
        cash_edge = float(selected_cash["edge"])
        rotation_edge = float(selected_rotation["edge"])
        cash_threshold = float(selected_cash.get("threshold") or default_cash_threshold)
        rotation_threshold = float(selected_rotation.get("threshold") or default_rotation_threshold)

        cash_probabilities, cash_metrics, cash_test = _fit_outer_model(
            train=train,
            test=test,
            features=features,
            target="target_cash",
            edge=cash_edge,
            threshold=cash_threshold,
            settings=settings,
            seed_offset=301_000 + test_year,
        )
        rotation_probabilities, rotation_metrics, rotation_test = _fit_outer_model(
            train=train,
            test=test,
            features=features,
            target="target_rotate",
            edge=rotation_edge,
            threshold=rotation_threshold,
            settings=settings,
            seed_offset=401_000 + test_year,
        )

        folds.append({
            "test_year": int(test_year),
            "train_rows": int(len(train)),
            "fit_rows": int(len(fit)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
            "regime_context_enabled": bool(regime_context.get("enabled")),
            "cash": {
                "configured_edge": configured_cash_edge,
                "candidates": cash_grid,
                "selected": selected_cash,
                "oos": cash_metrics,
            },
            "rotation": {
                "configured_edge": configured_rotation_edge,
                "candidates": rotation_grid,
                "selected": selected_rotation,
                "oos": rotation_metrics,
            },
        })

        for position, ((_, source), cash_probability, rotation_probability) in enumerate(
            zip(test.iterrows(), cash_probabilities, rotation_probabilities)
        ):
            cash_signal = float(cash_probability) >= cash_threshold
            rotation_signal = float(rotation_probability) >= rotation_threshold
            if cash_signal and float(cash_probability) >= float(rotation_probability) + arbitration_margin:
                action = "CASH"
            elif rotation_signal and source.get("alternative_symbol") and float(rotation_probability) >= float(cash_probability) + arbitration_margin:
                action = "ROTATE"
            else:
                action = "FOLLOW_BASE"
            predictions.append({
                "execution_at": source.get("execution_at"),
                "decision_at": source.get("decision_at"),
                "test_year": int(test_year),
                "symbol": source.get("symbol"),
                "alternative_symbol": source.get("alternative_symbol"),
                "policy_action": action,
                "cash_probability": float(cash_probability),
                "rotation_probability": float(rotation_probability),
                "cash_threshold": cash_threshold,
                "rotation_threshold": rotation_threshold,
                "selected_cash_edge": cash_edge,
                "selected_rotation_edge": rotation_edge,
                "target_cash": int(cash_test.iloc[position]["target_cash"]),
                "target_rotate": int(rotation_test.iloc[position]["target_rotate"]),
                "asset_utility": _finite(source.get("asset_utility")),
                "alternative_utility": _finite(source.get("alternative_utility")),
                "asset_return_1d": _finite(source.get("asset_return_1d")),
                "alternative_return_1d": _finite(source.get("alternative_return_1d")),
            })

    if not folds or not predictions:
        raise ValueError("Edge calibration could not produce nested chronological OOS predictions.")

    replay = _replay(reference_rows, predictions, settings)
    cash_oos = [((fold.get("cash") or {}).get("oos") or {}) for fold in folds]
    rotation_oos = [((fold.get("rotation") or {}).get("oos") or {}) for fold in folds]
    selected_cash_counts = Counter(float(((fold.get("cash") or {}).get("selected") or {}).get("edge")) for fold in folds)
    selected_rotation_counts = Counter(float(((fold.get("rotation") or {}).get("selected") or {}).get("edge")) for fold in folds)
    action_counts = Counter(str(row.get("policy_action") or "FOLLOW_BASE") for row in predictions)

    return {
        "analysis_version": EDGE_CALIBRATION_VERSION,
        "status": "completed",
        "shadow_only": True,
        "decision_effect": "none",
        "selection_uses_outer_oos": False,
        "protocol": {
            "method": "nested chronological walk-forward edge calibration",
            "selection_scope": "inner chronological validation only",
            "outer_scope": "calendar-year OOS evaluation only",
            "selection_metric_order": ["balanced_accuracy", "auc", "lower_brier", "configured_edge_proximity"],
            "cash_candidates": cash_candidates,
            "rotation_candidates": rotation_candidates,
            "regime_context_enabled": bool(settings.get("regime_context_enabled", True)),
            "action_probability_margin": arbitration_margin,
            "note": "CASH and ROTATE target definitions are calibrated independently inside each outer fold; outer OOS results never select an edge.",
        },
        "configured": {
            "minimum_cash_edge": configured_cash_edge,
            "minimum_rotation_edge": configured_rotation_edge,
        },
        "cash_grid_summary": _aggregate_grid(folds, "cash"),
        "rotation_grid_summary": _aggregate_grid(folds, "rotation"),
        "selected_cash_edge_distribution": {str(key): int(value) for key, value in sorted(selected_cash_counts.items())},
        "selected_rotation_edge_distribution": {str(key): int(value) for key, value in sorted(selected_rotation_counts.items())},
        "nested_oos": {
            "mean_cash_auc": _mean_available([_finite(item.get("auc")) for item in cash_oos]),
            "mean_cash_brier": _mean_available([_finite(item.get("brier")) for item in cash_oos]),
            "mean_cash_balanced_accuracy": _mean_available([_finite(item.get("balanced_accuracy")) for item in cash_oos]),
            "mean_rotation_auc": _mean_available([_finite(item.get("auc")) for item in rotation_oos]),
            "mean_rotation_brier": _mean_available([_finite(item.get("brier")) for item in rotation_oos]),
            "mean_rotation_balanced_accuracy": _mean_available([_finite(item.get("balanced_accuracy")) for item in rotation_oos]),
            "action_counts": {str(key): int(value) for key, value in sorted(action_counts.items())},
            "replay": replay,
        },
        "folds": folds,
    }
