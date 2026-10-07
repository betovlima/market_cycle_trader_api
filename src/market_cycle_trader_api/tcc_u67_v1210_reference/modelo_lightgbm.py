from __future__ import annotations

import math
import time
from typing import Any, Callable

import numpy as np
import pandas as pd

from .rotacao import (
    ROTATION_FEATURES,
    SUPPORTED_ROTATION_MODES,
    RotationRunResult,
    _datas_decisao_analise,
    _construir_folds_walk_forward,
    _desempenho_folds,
    _utilidades_modelo,
    _precalcular_utilidades_modelo,
    _politica_agendada,
    _crescimento_politica_simples,
    _simular_exato,
    _politica_utilidade,
    preparar_painel_rotacao,
)

def _configuracoes_pesquisa(config: Any) -> dict[str, Any]:
    raw = getattr(config, "research_model_settings", {}) or {}
    return dict(raw) if isinstance(raw, dict) else {}

def _configuracoes_lightgbm(config: Any) -> dict[str, Any]:
    settings = _configuracoes_pesquisa(config)
    lightgbm = settings.get("lightgbm")
    if not isinstance(lightgbm, dict):
        raise ValueError("The LightGBM execution snapshot is missing its protected model settings.")
    required = {
        "n_estimators", "learning_rate", "max_depth", "num_leaves",
        "min_child_samples", "min_child_weight", "subsample", "subsample_freq",
        "colsample_bytree", "reg_alpha", "reg_lambda", "max_bin", "n_jobs",
    }
    missing = sorted(required.difference(lightgbm))
    if missing:
        raise ValueError("LightGBM research settings are incomplete: " + ", ".join(missing))
    resolved = dict(lightgbm)
    # v10.8.68: RMSE-based early stopping is deliberately disabled for the
    # economic ranking model. Predictive MAE/RMSE remain diagnostics only.
    resolved["early_stopping_enabled"] = False
    return resolved


def _selecionar_switch_margin_fold(
    config: Any,
    fold_id: int,
    candidate_scores: list[tuple[float, float]],
) -> dict[str, Any]:
    """Seleciona a margem do fold e permite override contrafactual auditavel.

    O override existe apenas para experimentos fatoriais predeclarados. O valor
    forcado precisa pertencer ao conjunto de candidatos ja congelado, evitando
    introduzir tuning novo durante a decomposicao causal.
    """
    if not candidate_scores:
        raise ValueError("Switch-margin calibration produced no candidates.")

    auto_candidate = float(candidate_scores[0][0])
    auto_score = float(candidate_scores[0][1])
    for candidate, score in candidate_scores[1:]:
        candidate = float(candidate)
        score = float(score)
        if score > auto_score:
            auto_candidate = candidate
            auto_score = score

    settings = _configuracoes_pesquisa(config)
    raw_overrides = settings.get("counterfactual_switch_margin_by_fold") or {}
    if not isinstance(raw_overrides, dict):
        raise ValueError(
            "counterfactual_switch_margin_by_fold must be a mapping."
        )

    raw_override = raw_overrides.get(str(int(fold_id)))
    if raw_override is None:
        raw_override = raw_overrides.get(int(fold_id))

    selected_candidate = auto_candidate
    selected_score = auto_score
    selection_source = "calibration"

    if raw_override is not None:
        forced = float(raw_override)
        if not np.isfinite(forced) or forced < 0.0:
            raise ValueError(
                f"Fold {fold_id}: forced switch margin must be finite and >= 0."
            )

        matched = None
        for candidate, score in candidate_scores:
            if abs(float(candidate) - forced) <= 1e-12:
                matched = (float(candidate), float(score))
                break
        if matched is None:
            allowed = ", ".join(
                f"{float(candidate):.10g}"
                for candidate, _ in candidate_scores
            )
            raise ValueError(
                f"Fold {fold_id}: forced switch margin {forced:.10g} "
                f"is outside frozen candidates [{allowed}]."
            )
        selected_candidate, selected_score = matched
        selection_source = "counterfactual_override"

    return {
        "selected_candidate_margin": float(selected_candidate),
        "selected_calibration_score": float(selected_score),
        "auto_candidate_margin": float(auto_candidate),
        "auto_calibration_score": float(auto_score),
        "selection_source": selection_source,
    }


_LIGHTGBM_DEVICE_PROBE_CACHE: dict[tuple[str, str], tuple[bool, str | None]] = {}

def _construir_contexto_execucao(
    bars_by_symbol: dict[str, pd.DataFrame],
    config: Any,
    *,
    calendar_override: pd.DatetimeIndex | None = None,
    calendar_source_label: str | None = None,
) -> tuple[
    dict[str, pd.DataFrame],
    pd.DatetimeIndex,
    str,
    list[str],
    list[dict[str, Any]],
    pd.DatetimeIndex,
    dict[pd.Timestamp, int],
    dict[pd.Timestamp, dict[str, Any]],
]:
    if config.strategy_mode not in SUPPORTED_ROTATION_MODES:
        raise ValueError(f"Unsupported research strategy mode: {config.strategy_mode}.")
    frames, common_dates, calendar_source_asset = preparar_painel_rotacao(
        bars_by_symbol,
        config,
        calendar_override=calendar_override,
        calendar_source_label=calendar_source_label,
    )
    symbols = sorted(frames)
    folds = _construir_folds_walk_forward(common_dates, config)
    all_decision_dates = _datas_decisao_analise(common_dates, folds, config)
    decision_to_fold: dict[pd.Timestamp, int] = {}
    decision_metadata: dict[pd.Timestamp, dict[str, Any]] = {}
    for fold in folds:
        for timestamp in fold["decision_dates"][:-1]:
            key = pd.Timestamp(timestamp)
            decision_to_fold[key] = int(fold["fold_id"])
            decision_metadata[key] = {
                "fold_id": int(fold["fold_id"]),
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
            }
    return (
        frames,
        common_dates,
        calendar_source_asset,
        symbols,
        folds,
        all_decision_dates,
        decision_to_fold,
        decision_metadata,
    )

def _profundidade_arvore(node: dict[str, Any] | None) -> int:
    if not isinstance(node, dict):
        return 0
    if "leaf_index" in node:
        return 1
    return 1 + max(
        _profundidade_arvore(node.get("left_child")),
        _profundidade_arvore(node.get("right_child")),
    )

def _no_publico_arvore(node: dict[str, Any] | None, feature_names: list[str]) -> dict[str, Any] | None:
    if not isinstance(node, dict):
        return None
    if "leaf_index" in node:
        return {
            "kind": "leaf",
            "leaf_index": int(node.get("leaf_index", -1)),
            "value": float(node.get("leaf_value", 0.0)),
            "count": int(node.get("leaf_count", 0) or 0),
        }

    feature_index = int(node.get("split_feature", -1))
    feature_name = (
        feature_names[feature_index]
        if 0 <= feature_index < len(feature_names)
        else f"feature_{feature_index}"
    )
    return {
        "kind": "split",
        "split_index": int(node.get("split_index", -1)),
        "feature": str(feature_name),
        "threshold": node.get("threshold"),
        "decision_type": str(node.get("decision_type") or "<="),
        "default_left": bool(node.get("default_left", False)),
        "gain": float(node.get("split_gain", 0.0) or 0.0),
        "value": float(node.get("internal_value", 0.0) or 0.0),
        "count": int(node.get("internal_count", 0) or 0),
        "left": _no_publico_arvore(node.get("left_child"), feature_names),
        "right": _no_publico_arvore(node.get("right_child"), feature_names),
    }

def _registro_ultima_arvore_lightgbm(
    model: Any,
    *,
    symbol: str,
    fold_id: int,
    fold_position: int,
    train_end: pd.Timestamp | None,
) -> dict[str, Any] | None:
    booster = getattr(model, "booster_", None)
    if booster is None:
        return None
    try:
        dump = booster.dump_model()
    except Exception:
        return None
    trees = dump.get("tree_info") if isinstance(dump, dict) else None
    if not isinstance(trees, list) or not trees:
        return None
    tree = trees[-1] if isinstance(trees[-1], dict) else {}
    structure = tree.get("tree_structure") if isinstance(tree, dict) else None
    feature_names = [str(item) for item in (dump.get("feature_names") or [])]
    public_root = _no_publico_arvore(structure, feature_names)
    if public_root is None:
        return None
    return {
        "schema_version": 1,
        "source": "latest_final_fold_selected_asset",
        "asset": str(symbol).upper(),
        "fold_id": int(fold_id),
        "fold_position": int(fold_position),
        "training_end": (pd.Timestamp(train_end).isoformat() if train_end is not None else None),
        "tree_index": int(tree.get("tree_index", len(trees) - 1)),
        "tree_count": int(len(trees)),
        "num_leaves": int(tree.get("num_leaves", 0) or 0),
        "depth": int(_profundidade_arvore(structure)),
        "shrinkage": float(tree.get("shrinkage", 0.0) or 0.0),
        "feature_names": feature_names,
        "root": public_root,
    }

def _diagnosticos_erro_regressao(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float | int | None]:
    actual_values = np.asarray(actual, dtype=np.float64)
    predicted_values = np.asarray(predicted, dtype=np.float64)
    valid = np.isfinite(actual_values) & np.isfinite(predicted_values)
    if not bool(valid.any()):
        return {
            "rows": 0,
            "absolute_error_sum": 0.0,
            "squared_error_sum": 0.0,
            "mae": None,
            "rmse": None,
        }
    residual = predicted_values[valid] - actual_values[valid]
    abs_sum = float(np.abs(residual).sum())
    sq_sum = float(np.square(residual).sum())
    rows = int(valid.sum())
    return {
        "rows": rows,
        "absolute_error_sum": abs_sum,
        "squared_error_sum": sq_sum,
        "mae": abs_sum / rows,
        "rmse": math.sqrt(sq_sum / rows),
    }

def _diagnosticos_ajuste_modelo_lightgbm(
    model: Any,
    train_frame: pd.DataFrame,
    *,
    target_column: str,
    configured_estimators: int,
) -> dict[str, Any]:
    train_prediction = model.predict(train_frame[ROTATION_FEATURES])
    train_diag = _diagnosticos_erro_regressao(
        train_frame[target_column].to_numpy(dtype=np.float64),
        np.asarray(train_prediction, dtype=np.float64),
    )

    gain = np.asarray(
        model.booster_.feature_importance(importance_type="gain"),
        dtype=np.float64,
    )
    gain = np.where(np.isfinite(gain), gain, 0.0)
    gain_total = float(gain.sum())
    importance = {
        feature: (float(value / gain_total) if gain_total > 0 else 0.0)
        for feature, value in zip(ROTATION_FEATURES, gain)
    }

    return {
        "configured_estimators": int(configured_estimators),
        "best_iteration": int(configured_estimators),
        "early_stopping_used": False,
        "train": train_diag,
        "feature_importance_gain": importance,
    }

def _avaliar_modelos_lightgbm_nas_datas(
    models: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    dates: pd.DatetimeIndex,
    *,
    target_column: str = "forward_risk_adjusted_utility",
) -> dict[str, float | int | None]:
    actual_values: list[float] = []
    predicted_values: list[float] = []

    for symbol in symbols:
        model = models.get(symbol)
        frame = frames.get(symbol)
        if model is None or frame is None or frame.empty:
            continue
        available_dates = pd.DatetimeIndex(dates).intersection(frame.index)
        if len(available_dates) == 0:
            continue
        sample = frame.loc[available_dates].dropna(
            subset=[target_column, *ROTATION_FEATURES]
        )
        if sample.empty:
            continue
        predicted = model.predict(sample[ROTATION_FEATURES])
        actual_values.extend(
            sample[target_column].to_numpy(dtype=np.float64).tolist()
        )
        predicted_values.extend(
            np.asarray(predicted, dtype=np.float64).tolist()
        )

    return _diagnosticos_erro_regressao(
        np.asarray(actual_values, dtype=np.float64),
        np.asarray(predicted_values, dtype=np.float64),
    )

def _agregar_diagnosticos_modelos_lightgbm(
    models: dict[str, Any],
    *,
    fold_id: int,
    validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    train_rows = 0
    train_abs = 0.0
    train_sq = 0.0
    configured_estimators: list[int] = []
    feature_gain = {feature: 0.0 for feature in ROTATION_FEATURES}

    for model in models.values():
        diag = getattr(model, "_fit_diagnostics", None)
        if not isinstance(diag, dict):
            continue
        train = diag.get("train") or {}
        train_rows += int(train.get("rows") or 0)
        train_abs += float(train.get("absolute_error_sum") or 0.0)
        train_sq += float(train.get("squared_error_sum") or 0.0)
        configured_estimators.append(int(diag.get("configured_estimators") or 0))
        for feature, value in (diag.get("feature_importance_gain") or {}).items():
            if feature in feature_gain:
                feature_gain[feature] += float(value or 0.0)

    model_count = len(models)
    feature_total = float(sum(feature_gain.values()))
    normalized_gain = {
        feature: (value / feature_total if feature_total > 0 else 0.0)
        for feature, value in feature_gain.items()
    }
    top_features = [
        {"feature": feature, "importance_gain": float(value)}
        for feature, value in sorted(
            normalized_gain.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]

    train_rmse = math.sqrt(train_sq / train_rows) if train_rows else None
    validation = dict(validation or {})
    validation_rows = int(validation.get("rows") or 0)
    validation_mae = (
        float(validation.get("mae"))
        if validation.get("mae") is not None
        else None
    )
    validation_rmse = (
        float(validation.get("rmse"))
        if validation.get("rmse") is not None
        else None
    )

    effective_estimators = (
        float(np.mean(configured_estimators))
        if configured_estimators
        else None
    )
    return {
        "fold_id": int(fold_id),
        "model_count": int(model_count),
        "early_stopping_model_count": 0,
        "early_stopping_model_fraction": 0.0,
        "best_iteration_mean": effective_estimators,
        "best_iteration_median": (
            float(np.median(configured_estimators))
            if configured_estimators
            else None
        ),
        "configured_estimators_mean": effective_estimators,
        "train_rows": int(train_rows),
        "train_mae": (train_abs / train_rows if train_rows else None),
        "train_rmse": train_rmse,
        "validation_rows": validation_rows,
        "validation_mae": validation_mae,
        "validation_rmse": validation_rmse,
        "generalization_gap_rmse": (
            float(validation_rmse) - float(train_rmse)
            if train_rmse is not None and validation_rmse is not None
            else None
        ),
        "validation_source": "existing_chronological_calibration_window",
        "feature_importance_source": "calibration_models",
        "feature_importance_gain": normalized_gain,
        "top_features_by_gain": top_features,
    }

def _agregar_diagnosticos_folds_lightgbm(
    folds: list[dict[str, Any]],
) -> dict[str, Any]:
    if not folds:
        return {}
    numeric_fields = (
        "train_mae",
        "train_rmse",
        "validation_mae",
        "validation_rmse",
        "generalization_gap_rmse",
        "best_iteration_mean",
        "configured_estimators_mean",
        "early_stopping_model_fraction",
    )
    output: dict[str, Any] = {"fold_count": len(folds)}
    for field in numeric_fields:
        values = [
            float(item[field])
            for item in folds
            if item.get(field) is not None and np.isfinite(float(item[field]))
        ]
        output_key = (
            field
            if field in {"best_iteration_mean", "configured_estimators_mean"}
            else f"{field}_mean"
        )
        output[output_key] = float(np.mean(values)) if values else None

    feature_gain = {feature: 0.0 for feature in ROTATION_FEATURES}
    for fold in folds:
        for feature, value in (fold.get("feature_importance_gain") or {}).items():
            if feature in feature_gain:
                feature_gain[feature] += float(value or 0.0)
    total = float(sum(feature_gain.values()))
    normalized = {
        feature: (value / total if total > 0 else 0.0)
        for feature, value in feature_gain.items()
    }
    output["feature_importance_gain"] = normalized
    output["top_features_by_gain"] = [
        {"feature": feature, "importance_gain": float(value)}
        for feature, value in sorted(
            normalized.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]
    output["early_stopping_enabled"] = False
    output["validation_source"] = "existing_chronological_calibration_window"
    return output

def _ajustar_modelos_lightgbm(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    train_dates: pd.DatetimeIndex,
    config: Any,
    *,
    phase: str,
    progress_callback: Callable[[int, int, str], None] | None = None,
    technical_log_callback: Callable[[str], None] | None = None,
    target_column: str = "forward_risk_adjusted_utility",
) -> dict[str, Any]:
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise RuntimeError(
            "LightGBM research requires lightgbm. Install requirements.txt."
        ) from exc

    minimum_rows = int(config.rotation_minimum_training_rows)
    settings = _configuracoes_lightgbm(config)
    active_device = "cpu"
    fitted: dict[str, Any] = {}
    started = time.perf_counter()

    def technical(message: str) -> None:
        if technical_log_callback is not None:
            technical_log_callback(message)

    technical(
        f"model=lightgbm phase={phase} event=fit_start device=cpu "
        f"models={len(symbols)} train_sessions={len(train_dates)} "
        f"estimators={int(settings['n_estimators'])} "
        f"seed={int(config.random_state)} early_stopping=false"
    )

    for position, symbol in enumerate(symbols, start=1):
        frame = frames[symbol].loc[train_dates].dropna(
            subset=[target_column, *ROTATION_FEATURES]
        )
        if len(frame) < minimum_rows:
            if progress_callback is not None:
                progress_callback(position, len(symbols), active_device)
            continue

        model = LGBMRegressor(
            objective="regression",
            boosting_type="gbdt",
            n_estimators=int(settings["n_estimators"]),
            learning_rate=float(settings["learning_rate"]),
            max_depth=int(settings["max_depth"]),
            num_leaves=int(settings["num_leaves"]),
            min_child_samples=int(settings["min_child_samples"]),
            min_child_weight=float(settings["min_child_weight"]),
            subsample=float(settings["subsample"]),
            subsample_freq=int(settings["subsample_freq"]),
            colsample_bytree=float(settings["colsample_bytree"]),
            reg_alpha=float(settings["reg_alpha"]),
            reg_lambda=float(settings["reg_lambda"]),
            max_bin=int(settings["max_bin"]),
            random_state=int(config.random_state),
            n_jobs=int(settings["n_jobs"]),
            device_type="cpu",
            deterministic=bool(config.deterministic_execution),
            force_col_wise=bool(config.deterministic_execution),
            verbosity=-1,
        )
        model.fit(
            frame[ROTATION_FEATURES],
            frame[target_column],
        )
        model._fit_diagnostics = _diagnosticos_ajuste_modelo_lightgbm(
            model,
            frame,
            target_column=target_column,
            configured_estimators=int(settings["n_estimators"]),
        )
        fitted[symbol] = model
        if progress_callback is not None:
            progress_callback(position, len(symbols), active_device)

    technical(
        f"model=lightgbm phase={phase} event=fit_complete device=cpu "
        f"models={len(fitted)} "
        f"duration_seconds={time.perf_counter() - started:.3f}"
    )
    return fitted


def _configuracoes_horizontes(config: Any) -> dict[str, Any]:
    horizons = [int(item) for item in config.rotation_target_horizons]
    weights = np.asarray(
        config.rotation_target_horizon_weights,
        dtype=float,
    )
    if len(horizons) != len(weights):
        raise ValueError(
            "Soft horizon consensus requires one weight per target horizon."
        )
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
        raise ValueError(
            "Soft horizon consensus requires finite positive horizon weights."
        )
    weights = weights / float(weights.sum())
    return {
        "horizons": horizons,
        "weights": [float(item) for item in weights],
    }


def _ajustar_modelos_horizontes(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    train_dates: pd.DatetimeIndex,
    config: Any,
    *,
    phase: str,
    progress_callback: Callable[[int, int, str], None] | None = None,
    technical_log_callback: Callable[[str], None] | None = None,
) -> dict[int, dict[str, Any]]:
    settings = _configuracoes_horizontes(config)
    horizons = list(settings["horizons"])
    fitted: dict[int, dict[str, Any]] = {}
    total = max(1, len(horizons) * len(symbols))

    for horizon_index, horizon in enumerate(horizons):
        def horizon_progress(
            position: int,
            symbol_total: int,
            device: str,
            *,
            _index=horizon_index,
        ) -> None:
            if progress_callback is None:
                return
            completed = _index * len(symbols) + int(position)
            progress_callback(completed, total, device)

        target_column = f"forward_horizon_utility_{int(horizon)}"
        fitted[int(horizon)] = _ajustar_modelos_lightgbm(
            frames,
            symbols,
            train_dates,
            config,
            phase=f"{phase}_h{int(horizon)}",
            progress_callback=horizon_progress,
            technical_log_callback=technical_log_callback,
            target_column=target_column,
        )
    return fitted


def _configuracoes_consenso_horizontes_soft(config: Any) -> dict[str, Any]:
    raw = (_configuracoes_pesquisa(config).get("soft_horizon_consensus") or {})
    horizon = _configuracoes_horizontes(config)
    return {
        "enabled": bool(raw.get("enabled", False))
        if isinstance(raw, dict)
        else False,
        "penalty_strength": (
            float(raw.get("penalty_strength", 1.0))
            if isinstance(raw, dict)
            else 1.0
        ),
        "horizons": list(horizon["horizons"]),
        "weights": list(horizon["weights"]),
        "mode": "weighted_rank_margin_modifier",
    }

def _registro_consenso_ranking_horizontes(
    horizon_models: dict[int, dict[str, Any]],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    timestamp: pd.Timestamp,
    config: Any,
    *,
    base_target: int,
    current_position: int,
    horizon_utility_caches: dict[int, dict[pd.Timestamp, np.ndarray]] | None = None,
) -> dict[str, Any]:
    settings = _configuracoes_consenso_horizontes_soft(config)
    aggregate = np.zeros(len(symbols) + 1, dtype=np.float64)
    available_weight = 0.0
    horizon_details: list[dict[str, Any]] = []

    for horizon, weight in zip(
        settings["horizons"],
        settings["weights"],
        strict=True,
    ):
        cache = (
            (horizon_utility_caches or {}).get(int(horizon))
            if horizon_utility_caches is not None
            else None
        )
        utilities = _utilidades_modelo(
            horizon_models.get(int(horizon), {}),
            frames,
            symbols,
            timestamp,
            config,
            utility_cache=cache,
        )
        ranked = sorted(
            (
                position
                for position in range(1, len(utilities))
                if np.isfinite(utilities[position])
            ),
            key=lambda position: (
                -float(utilities[position]),
                symbols[position - 1],
            ),
        )
        if not ranked:
            horizon_details.append(
                {
                    "horizon": int(horizon),
                    "weight": float(weight),
                    "available_assets": 0,
                    "winner_asset": None,
                    "base_target_rank": None,
                    "base_target_rank_score": None,
                    "current_position_rank": None,
                    "current_position_rank_score": None,
                }
            )
            continue

        available_weight += float(weight)
        denominator = max(1, len(ranked) - 1)
        rank_scores: dict[int, float] = {}
        for rank_index, position in enumerate(ranked):
            rank_score = (
                1.0
                if len(ranked) == 1
                else 1.0 - float(rank_index) / float(denominator)
            )
            rank_scores[int(position)] = float(rank_score)
            aggregate[int(position)] += float(weight) * float(rank_score)

        base_rank = (
            ranked.index(int(base_target)) + 1
            if int(base_target) in ranked
            else None
        )
        current_rank = (
            ranked.index(int(current_position)) + 1
            if int(current_position) in ranked
            else None
        )
        horizon_details.append(
            {
                "horizon": int(horizon),
                "weight": float(weight),
                "available_assets": int(len(ranked)),
                "winner_asset": symbols[ranked[0] - 1],
                "winner_score": float(utilities[ranked[0]]),
                "base_target_rank": base_rank,
                "base_target_rank_score": (
                    rank_scores.get(int(base_target))
                    if int(base_target) > 0
                    else None
                ),
                "current_position_rank": current_rank,
                "current_position_rank_score": (
                    rank_scores.get(int(current_position))
                    if int(current_position) > 0
                    else None
                ),
            }
        )

    if available_weight > 0:
        aggregate[1:] = aggregate[1:] / float(available_weight)

    finite_positions = [
        position
        for position in range(1, len(aggregate))
        if np.isfinite(aggregate[position])
    ]
    consensus_winner = (
        max(
            finite_positions,
            key=lambda position: (
                float(aggregate[position]),
                -position,
            ),
        )
        if finite_positions
        else 0
    )
    base_score = (
        float(aggregate[int(base_target)])
        if int(base_target) > 0 and int(base_target) < len(aggregate)
        else None
    )
    current_score = (
        float(aggregate[int(current_position)])
        if int(current_position) > 0 and int(current_position) < len(aggregate)
        else None
    )
    if base_score is None:
        support = None
        relative_component = None
    elif current_score is None:
        relative_component = float(base_score)
        support = float(base_score)
    else:
        relative_component = float(
            np.clip(
                0.5 + 0.5 * (float(base_score) - float(current_score)),
                0.0,
                1.0,
            )
        )
        support = float(
            0.5 * float(base_score) + 0.5 * relative_component
        )

    return {
        "consensus_winner_position": int(consensus_winner),
        "consensus_winner_asset": (
            symbols[consensus_winner - 1]
            if consensus_winner > 0
            else None
        ),
        "consensus_winner_score": (
            float(aggregate[consensus_winner])
            if consensus_winner > 0
            else None
        ),
        "base_target_rank_score": base_score,
        "current_position_rank_score": current_score,
        "relative_rank_component": relative_component,
        "soft_support": support,
        "available_horizon_weight": float(available_weight),
        "aggregate_rank_scores": {
            symbols[position - 1]: float(aggregate[position])
            for position in range(1, len(aggregate))
        },
        "horizons": horizon_details,
    }

def _politica_consenso_horizontes_soft(
    base_policy: Callable[[pd.Timestamp, int, int], tuple[int, float]],
    base_models: dict[str, Any],
    horizon_models: dict[int, dict[str, Any]],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    config: Any,
    *,
    base_switch_margin: float,
    base_utility_cache: dict[pd.Timestamp, np.ndarray] | None = None,
    horizon_utility_caches: dict[int, dict[pd.Timestamp, np.ndarray]] | None = None,
    decision_diagnostics: dict[pd.Timestamp, dict[str, Any]] | None = None,
) -> Callable[[pd.Timestamp, int, int], tuple[int, float]]:
    settings = _configuracoes_consenso_horizontes_soft(config)
    penalty_strength = max(0.0, float(settings["penalty_strength"]))

    def policy(
        timestamp: pd.Timestamp,
        current_position: int,
        holding_days: int,
    ) -> tuple[int, float]:
        base_target, base_score = base_policy(
            timestamp,
            current_position,
            holding_days,
        )
        key = pd.Timestamp(timestamp)

        if int(base_target) == 0:
            if decision_diagnostics is not None:
                decision_diagnostics.setdefault(key, {}).update(
                    {
                        "soft_horizon_consensus_enabled": True,
                        "soft_horizon_consensus_mode": str(settings["mode"]),
                        "soft_horizon_consensus_reason": "BASE_CASH_PRESERVED",
                        "soft_horizon_consensus_changed_base_action": False,
                    }
                )
            return 0, float(base_score)

        snapshot = _registro_consenso_ranking_horizontes(
            horizon_models,
            frames,
            symbols,
            timestamp,
            config,
            base_target=int(base_target),
            current_position=int(current_position),
            horizon_utility_caches=horizon_utility_caches,
        )

        if int(base_target) == int(current_position):
            final_target = int(base_target)
            reason = "BASE_HOLD_PRESERVED"
            base_gap = 0.0
            margin_multiplier = 1.0
            dynamic_margin = float(base_switch_margin)
        else:
            base_utilities = _utilidades_modelo(
                base_models,
                frames,
                symbols,
                timestamp,
                config,
                utility_cache=base_utility_cache,
            )
            target_utility = (
                float(base_utilities[int(base_target)])
                if int(base_target) < len(base_utilities)
                else float("nan")
            )
            current_utility = (
                float(base_utilities[int(current_position)])
                if 0 <= int(current_position) < len(base_utilities)
                else 0.0
            )
            base_gap = target_utility - current_utility
            support = snapshot.get("soft_support")
            if support is None or not np.isfinite(float(support)):
                margin_multiplier = 1.0
            else:
                margin_multiplier = 1.0 + penalty_strength * (
                    1.0 - float(np.clip(float(support), 0.0, 1.0))
                )
            dynamic_margin = float(base_switch_margin) * float(
                margin_multiplier
            )

            if not np.isfinite(base_gap):
                final_target = int(base_target)
                reason = "SOFT_CONSENSUS_NO_GAP_PRESERVE"
            elif float(base_gap) + 1e-15 >= float(dynamic_margin):
                final_target = int(base_target)
                reason = "SOFT_CONSENSUS_ACCEPT"
            else:
                final_target = (
                    int(current_position)
                    if int(current_position) > 0
                    else 0
                )
                reason = "SOFT_CONSENSUS_BLOCK_MARGINAL_SWITCH"

        if decision_diagnostics is not None:
            diagnostic = decision_diagnostics.setdefault(key, {})
            diagnostic.update(
                {
                    "soft_horizon_consensus_enabled": True,
                    "soft_horizon_consensus_mode": str(settings["mode"]),
                    "soft_horizon_consensus_penalty_strength": float(
                        penalty_strength
                    ),
                    "soft_horizon_consensus_base_target_asset": (
                        symbols[int(base_target) - 1]
                        if int(base_target) > 0
                        else "CASH"
                    ),
                    "soft_horizon_consensus_current_asset": (
                        symbols[int(current_position) - 1]
                        if int(current_position) > 0
                        else "CASH"
                    ),
                    "soft_horizon_consensus_winner_asset": snapshot.get(
                        "consensus_winner_asset"
                    ),
                    "soft_horizon_consensus_winner_score": snapshot.get(
                        "consensus_winner_score"
                    ),
                    "soft_horizon_consensus_base_target_rank_score": snapshot.get(
                        "base_target_rank_score"
                    ),
                    "soft_horizon_consensus_current_rank_score": snapshot.get(
                        "current_position_rank_score"
                    ),
                    "soft_horizon_consensus_relative_rank_component": snapshot.get(
                        "relative_rank_component"
                    ),
                    "soft_horizon_consensus_support": snapshot.get(
                        "soft_support"
                    ),
                    "soft_horizon_consensus_base_gap": (
                        float(base_gap)
                        if np.isfinite(float(base_gap))
                        else None
                    ),
                    "soft_horizon_consensus_base_switch_margin": float(
                        base_switch_margin
                    ),
                    "soft_horizon_consensus_margin_multiplier": float(
                        margin_multiplier
                    ),
                    "soft_horizon_consensus_dynamic_margin": float(
                        dynamic_margin
                    ),
                    "soft_horizon_consensus_horizons": list(
                        snapshot["horizons"]
                    ),
                    "soft_horizon_consensus_final_action_asset": (
                        symbols[int(final_target) - 1]
                        if int(final_target) > 0
                        else "CASH"
                    ),
                    "soft_horizon_consensus_reason": reason,
                    "soft_horizon_consensus_changed_base_action": (
                        int(final_target) != int(base_target)
                    ),
                }
            )
        return int(final_target), float(base_score)

    return policy

def _metricas_resultado_consenso_horizontes_soft(
    result: RotationRunResult,
) -> dict[str, Any]:
    predictions = result.predictions
    if not isinstance(predictions, pd.DataFrame) or predictions.empty:
        return {}
    column = "soft_horizon_consensus_enabled"
    if column not in predictions.columns:
        return {}
    rows = predictions.loc[
        predictions[column].fillna(False).astype(bool)
    ]
    if rows.empty:
        return {}

    reasons = rows.get(
        "soft_horizon_consensus_reason",
        pd.Series(dtype=object),
    )
    changed = rows.get(
        "soft_horizon_consensus_changed_base_action",
        pd.Series(dtype=bool),
    ).fillna(False).astype(bool)
    support = pd.to_numeric(
        rows.get(
            "soft_horizon_consensus_support",
            pd.Series(dtype=float),
        ),
        errors="coerce",
    )
    multiplier = pd.to_numeric(
        rows.get(
            "soft_horizon_consensus_margin_multiplier",
            pd.Series(dtype=float),
        ),
        errors="coerce",
    )
    return {
        "soft_horizon_consensus_enabled": True,
        "soft_horizon_consensus_decisions": int(len(rows)),
        "soft_horizon_consensus_changed_base_actions": int(changed.sum()),
        "soft_horizon_consensus_change_rate": float(changed.mean()),
        "soft_horizon_consensus_blocked_marginal_switches": int(
            (
                reasons
                == "SOFT_CONSENSUS_BLOCK_MARGINAL_SWITCH"
            ).sum()
        ),
        "soft_horizon_consensus_accepts": int(
            (reasons == "SOFT_CONSENSUS_ACCEPT").sum()
        ),
        "soft_horizon_consensus_average_support": (
            float(support.dropna().mean())
            if support.notna().any()
            else None
        ),
        "soft_horizon_consensus_median_support": (
            float(support.dropna().median())
            if support.notna().any()
            else None
        ),
        "soft_horizon_consensus_average_margin_multiplier": (
            float(multiplier.dropna().mean())
            if multiplier.notna().any()
            else None
        ),
    }

def executar_lightgbm(
    bars_by_symbol: dict[str, pd.DataFrame],
    config: Any,
    fee_calculator: Callable,
    slippage: Callable,
    *,
    progress_callback: Callable[[float, str, int], None] | None = None,
    trade_callback: Callable[[dict[str, Any]], None] | None = None,
    progress_detail_callback: Callable[[dict[str, Any]], None] | None = None,
    technical_log_callback: Callable[[str], None] | None = None,
) -> list[RotationRunResult]:
    (
        frames,
        common_dates,
        calendar_source_asset,
        symbols,
        folds,
        all_decision_dates,
        decision_to_fold,
        decision_metadata,
    ) = _construir_contexto_execucao(bars_by_symbol, config)

    repetitions = int(config.rotation_model_repetitions)
    seed_step = int(config.rotation_seed_step)
    total_folds = len(folds)
    total_models = len(symbols)
    soft = _configuracoes_consenso_horizontes_soft(config)

    def report(fraction: float, stage: str, completed: int) -> None:
        if progress_callback is not None:
            progress_callback(
                20.0 + 72.0 * max(0.0, min(1.0, fraction)),
                stage,
                completed,
            )

    def detail(**values: Any) -> None:
        if progress_detail_callback is not None:
            progress_detail_callback(values)

    if progress_callback is not None:
        progress_callback(
            18.0,
            (
                f"Prepared {len(symbols)} assets and {len(folds)} folds "
                "— LightGBM=CPU"
            ),
            0,
        )

    results: list[RotationRunResult] = []

    for repetition in range(repetitions):
        run_index = repetition + 1
        seed = int(config.random_state) + repetition * seed_step
        rep_config = config.copiar_modelo(update={"random_state": seed})
        policies: dict[int, Callable] = {}
        diagnostics: dict[pd.Timestamp, dict[str, Any]] = {}
        margin_details: list[dict[str, Any]] = []
        model_fold_diagnostics: list[dict[str, Any]] = []
        inference_cache_profiles: list[dict[str, Any]] = []
        latest_final_models: dict[str, Any] = {}
        latest_final_fold_id: int | None = None
        latest_final_fold_position: int | None = None
        latest_final_train_end: pd.Timestamp | None = None

        run_base = repetition / repetitions
        run_span = 1.0 / repetitions
        fold_span = (run_span * 0.90) / max(1, total_folds)

        for fold_position, fold in enumerate(folds, start=1):
            fold_base = run_base + (fold_position - 1) * fold_span
            fold_id = int(fold["fold_id"])
            train_dates = common_dates[: int(fold["train_end_index"])]
            calibration_dates = common_dates[
                int(fold["calibration_start_index"]):
                int(fold["calibration_end_index"])
            ]
            final_fit_dates = common_dates[: int(fold["final_fit_end_index"])]

            def phase_progress(label: str, start: float, end: float):
                def callback(
                    position: int,
                    total: int,
                    device: str,
                ) -> None:
                    fraction = position / max(1, total)
                    report(
                        fold_base
                        + fold_span
                        * (start + (end - start) * fraction),
                        (
                            f"Run {run_index}/{repetitions} — "
                            f"fold {fold_position}/{total_folds} — "
                            f"{label} {position}/{total}"
                        ),
                        repetition,
                    )
                    detail(
                        run_index=run_index,
                        run_count=repetitions,
                        fold_index=fold_position,
                        fold_count=total_folds,
                        phase=label.title(),
                        trained_models=position,
                        total_models=total,
                        device=device.upper(),
                    )

                return callback

            detail(
                run_index=run_index,
                run_count=repetitions,
                fold_index=fold_position,
                fold_count=total_folds,
                phase="Calibration training",
                trained_models=0,
                total_models=total_models,
                device="CPU",
            )
            calibration_models = _ajustar_modelos_lightgbm(
                frames,
                symbols,
                train_dates,
                rep_config,
                phase=(
                    f"run_{run_index}_fold_"
                    f"{fold_position}_calibration"
                ),
                progress_callback=phase_progress(
                    "calibration training",
                    0.02,
                    0.38,
                ),
                technical_log_callback=technical_log_callback,
            )
            calibration_predictive_diagnostics = (
                _avaliar_modelos_lightgbm_nas_datas(
                    calibration_models,
                    frames,
                    symbols,
                    calibration_dates,
                    target_column="forward_risk_adjusted_utility",
                )
            )
            model_fold_diagnostics.append(
                _agregar_diagnosticos_modelos_lightgbm(
                    calibration_models,
                    fold_id=fold_id,
                    validation=calibration_predictive_diagnostics,
                )
            )

            candidate_margins = tuple(
                float(value)
                for value in rep_config.rotation_switch_margin_candidates
            )
            candidate_scores: list[tuple[float, float]] = []
            for candidate in candidate_margins:
                calibration_policy = _politica_utilidade(
                    calibration_models,
                    frames,
                    symbols,
                    rep_config,
                    candidate,
                )
                score = _crescimento_politica_simples(
                    calibration_policy,
                    frames,
                    symbols,
                    calibration_dates,
                    rep_config,
                )
                candidate_scores.append((float(candidate), float(score)))

            margin_selection = _selecionar_switch_margin_fold(
                rep_config,
                fold_id,
                candidate_scores,
            )
            best_candidate = float(
                margin_selection["selected_candidate_margin"]
            )
            best_score = float(
                margin_selection["selected_calibration_score"]
            )
            auto_best_candidate = float(
                margin_selection["auto_candidate_margin"]
            )
            auto_best_score = float(
                margin_selection["auto_calibration_score"]
            )

            detail(
                run_index=run_index,
                run_count=repetitions,
                fold_index=fold_position,
                fold_count=total_folds,
                phase="Final training",
                trained_models=0,
                total_models=total_models,
                device="CPU",
            )
            final_models = _ajustar_modelos_lightgbm(
                frames,
                symbols,
                final_fit_dates,
                rep_config,
                phase=(
                    f"run_{run_index}_fold_"
                    f"{fold_position}_final"
                ),
                progress_callback=phase_progress(
                    "final training",
                    0.50,
                    0.78 if bool(soft["enabled"]) else 0.90,
                ),
                technical_log_callback=technical_log_callback,
            )

            final_horizon_models: dict[int, dict[str, Any]] = {}
            if bool(soft["enabled"]):
                detail(
                    run_index=run_index,
                    run_count=repetitions,
                    fold_index=fold_position,
                    fold_count=total_folds,
                    phase="Soft horizon consensus training",
                    trained_models=0,
                    total_models=(
                        len(symbols)
                        * len(config.rotation_target_horizons)
                    ),
                    device="CPU",
                )
                final_horizon_models = _ajustar_modelos_horizontes(
                    frames,
                    symbols,
                    final_fit_dates,
                    rep_config,
                    phase=(
                        f"run_{run_index}_fold_"
                        f"{fold_position}_soft_horizon"
                    ),
                    progress_callback=phase_progress(
                        "soft horizon consensus training",
                        0.80,
                        0.98,
                    ),
                    technical_log_callback=technical_log_callback,
                )

            latest_final_models = final_models
            latest_final_fold_id = fold_id
            latest_final_fold_position = fold_position
            latest_final_train_end = (
                pd.Timestamp(final_fit_dates[-1])
                if len(final_fit_dates)
                else None
            )

            fold_decision_dates = pd.DatetimeIndex(fold["decision_dates"])
            base_utility_cache, base_cache_profile = (
                _precalcular_utilidades_modelo(
                    final_models,
                    frames,
                    symbols,
                    fold_decision_dates,
                    rep_config,
                )
            )
            base_cache_profile.update(
                {
                    "fold_id": int(fold_id),
                    "model_role": "weighted_utility",
                }
            )
            inference_cache_profiles.append(base_cache_profile)

            horizon_utility_caches: dict[
                int,
                dict[pd.Timestamp, np.ndarray],
            ] = {}
            for horizon, horizon_models in final_horizon_models.items():
                cache, cache_profile = _precalcular_utilidades_modelo(
                    horizon_models,
                    frames,
                    symbols,
                    fold_decision_dates,
                    rep_config,
                )
                horizon_utility_caches[int(horizon)] = cache
                cache_profile.update(
                    {
                        "fold_id": int(fold_id),
                        "model_role": f"horizon_{int(horizon)}",
                        "horizon": int(horizon),
                    }
                )
                inference_cache_profiles.append(cache_profile)

            if technical_log_callback is not None:
                fold_profiles = [
                    item
                    for item in inference_cache_profiles
                    if int(item.get("fold_id") or 0) == int(fold_id)
                ]
                technical_log_callback(
                    "model=lightgbm event=oos_inference_cache_ready "
                    f"fold={fold_id} roles={len(fold_profiles)} "
                    "seconds="
                    f"{sum(float(item.get('cache_build_seconds') or 0.0) for item in fold_profiles):.3f} "
                    "predict_calls="
                    f"{sum(int(item.get('cache_predict_calls') or 0) for item in fold_profiles)}"
                )

            effective_margin = max(
                float(rep_config.rotation_switch_margin),
                float(best_candidate),
            )
            base_policy = _politica_utilidade(
                final_models,
                frames,
                symbols,
                rep_config,
                effective_margin,
                decision_diagnostics=diagnostics,
                fold_id=fold_id,
                calibrated_switch_margin=float(best_candidate),
                utility_cache=base_utility_cache,
            )
            if bool(soft["enabled"]):
                policies[fold_id] = _politica_consenso_horizontes_soft(
                    base_policy,
                    final_models,
                    final_horizon_models,
                    frames,
                    symbols,
                    rep_config,
                    base_switch_margin=float(effective_margin),
                    base_utility_cache=base_utility_cache,
                    horizon_utility_caches=horizon_utility_caches,
                    decision_diagnostics=diagnostics,
                )
            else:
                policies[fold_id] = base_policy

            margin_details.append(
                {
                    "fold_id": fold_id,
                    "soft_horizon_consensus_enabled": bool(
                        soft["enabled"]
                    ),
                    "soft_horizon_consensus_mode": (
                        str(soft["mode"])
                        if bool(soft["enabled"])
                        else None
                    ),
                    "soft_horizon_consensus_penalty_strength": (
                        float(soft["penalty_strength"])
                        if bool(soft["enabled"])
                        else None
                    ),
                    "calibrated_candidate_margin": float(best_candidate),
                    "effective_switch_margin": float(effective_margin),
                    "calibration_risk_adjusted_score": float(best_score),
                    "auto_calibrated_candidate_margin": float(
                        auto_best_candidate
                    ),
                    "auto_calibration_risk_adjusted_score": float(
                        auto_best_score
                    ),
                    "margin_selection_source": str(
                        margin_selection["selection_source"]
                    ),
                }
            )
            report(
                fold_base + fold_span,
                (
                    f"Run {run_index}/{repetitions} — "
                    f"fold {fold_position}/{total_folds} completed"
                ),
                repetition,
            )

        report(
            run_base + run_span * 0.94,
            (
                f"Run {run_index}/{repetitions} — "
                "simulating out-of-sample portfolio"
            ),
            repetition,
        )
        scheduled = _politica_agendada(policies, decision_to_fold)

        wrapped_trade_callback = None
        if trade_callback is not None:
            def wrapped_trade_callback(
                trade: dict[str, Any],
                *,
                _seed=seed,
                _run=run_index,
            ) -> None:
                payload = dict(trade)
                payload.update(
                    {
                        "model_family": "lightgbm_utility",
                        "random_seed": _seed,
                        "repetition_index": _run,
                        "model": (
                            "LightGBM Utility"
                            + (
                                f" · seed {_seed}"
                                if repetitions > 1
                                else ""
                            )
                        ),
                    }
                )
                trade_callback(payload)

        def simulation_progress(
            local_fraction: float,
            stage: str,
        ) -> None:
            fraction = max(
                0.0,
                min(1.0, float(local_fraction)),
            )
            report(
                run_base
                + run_span
                * (0.94 + 0.06 * fraction),
                f"Run {run_index}/{repetitions} — {stage}",
                repetition,
            )
            detail(
                run_index=run_index,
                run_count=repetitions,
                fold_index=total_folds,
                fold_count=total_folds,
                phase="OOS simulation",
                trained_models=int(round(fraction * 1000)),
                total_models=1000,
                device="CPU",
            )

        result = _simular_exato(
            "lightgbm_utility",
            scheduled,
            frames,
            symbols,
            all_decision_dates,
            rep_config,
            fee_calculator,
            slippage,
            decision_metadata=decision_metadata,
            policy_decision_diagnostics=diagnostics,
            trade_callback=wrapped_trade_callback,
            model_label="LightGBM Utility",
            method_line=(
                "- LightGBM Utility predicts the same weighted "
                "multi-horizon risk-adjusted utility target across "
                f"{rep_config.rotation_target_horizons}."
            ),
            simulation_progress_callback=simulation_progress,
        )

        backend = (
            "lightgbm_utility"
            if repetitions <= 1
            else f"lightgbm_utility_seed_{seed}"
        )
        result.backend = backend
        simulation_profile = (
            result.metrics.get("simulation_profile") or {}
        )

        if bool(soft["enabled"]):
            result.metrics.update(
                _metricas_resultado_consenso_horizontes_soft(result)
            )

        result.metrics["oos_inference_cache_profiles"] = list(
            inference_cache_profiles
        )
        result.metrics["oos_inference_cache_build_seconds"] = float(
            sum(
                float(item.get("cache_build_seconds") or 0.0)
                for item in inference_cache_profiles
            )
        )
        result.metrics["oos_inference_cache_predict_calls"] = int(
            sum(
                int(item.get("cache_predict_calls") or 0)
                for item in inference_cache_profiles
            )
        )

        if technical_log_callback is not None:
            technical_log_callback(
                "model=lightgbm event=oos_simulation_complete "
                f"run={run_index}/{repetitions} "
                f"sessions={simulation_profile.get('session_count')} "
                "benchmark_seconds="
                f"{float(simulation_profile.get('benchmark_seconds') or 0.0):.3f} "
                "market_regime_seconds="
                f"{float(simulation_profile.get('market_regime_seconds') or 0.0):.3f} "
                "policy_seconds="
                f"{float(simulation_profile.get('policy_seconds') or 0.0):.3f} "
                "accounting_seconds="
                f"{float(simulation_profile.get('accounting_seconds') or 0.0):.3f} "
                "total_seconds="
                f"{float(simulation_profile.get('total_seconds') or 0.0):.3f}"
            )

        latest_asset = None
        if (
            isinstance(result.predictions, pd.DataFrame)
            and not result.predictions.empty
        ):
            last_prediction = result.predictions.iloc[-1]
            for key in (
                "selected_asset",
                "final_action_asset",
                "best_asset",
                "raw_best_asset",
            ):
                raw_candidate = last_prediction.get(key)
                if raw_candidate is None or pd.isna(raw_candidate):
                    continue
                candidate = str(raw_candidate).strip().upper()
                if candidate and candidate != "CASH":
                    latest_asset = candidate
                    break
        if latest_asset is None and latest_final_models:
            latest_asset = next(reversed(latest_final_models))

        latest_tree = None
        if (
            latest_asset
            and latest_asset in latest_final_models
            and latest_final_fold_id is not None
            and latest_final_fold_position is not None
        ):
            latest_tree = _registro_ultima_arvore_lightgbm(
                latest_final_models[latest_asset],
                symbol=latest_asset,
                fold_id=latest_final_fold_id,
                fold_position=latest_final_fold_position,
                train_end=latest_final_train_end,
            )

        predictive_diagnostics = _agregar_diagnosticos_folds_lightgbm(
            model_fold_diagnostics
        )
        result.metrics.update(
            {
                "backend": backend,
                "model_family": "lightgbm_utility",
                "strategy_label": (
                    "LightGBM Utility"
                    + (
                        f" · seed {seed}"
                        if repetitions > 1
                        else ""
                    )
                ),
                "random_seed": seed,
                "repetition_index": run_index,
                "repetition_count": repetitions,
                "walk_forward_fold_count": len(folds),
                "walk_forward_folds": _desempenho_folds(
                    result.predictions,
                    folds,
                    float(rep_config.initial_capital),
                ),
                "effective_switch_margin": float(
                    np.mean(
                        [
                            item["effective_switch_margin"]
                            for item in margin_details
                        ]
                    )
                ),
                "effective_switch_margin_mean": float(
                    np.mean(
                        [
                            item["effective_switch_margin"]
                            for item in margin_details
                        ]
                    )
                ),
                "calibrated_switch_margin": float(
                    np.mean(
                        [
                            item["calibrated_candidate_margin"]
                            for item in margin_details
                        ]
                    )
                ),
                "calendar_source_asset": calendar_source_asset,
                "requested_compute_device": "cpu",
                "effective_compute_device": "cpu",
                "deterministic_execution": bool(
                    rep_config.deterministic_execution
                ),
                "numeric_thread_limit": int(
                    rep_config.numeric_thread_limit
                ),
                "decision_diagnostics_schema_version": (
                    13 if bool(soft["enabled"]) else 2
                ),
                "decision_diagnostics_rows": len(diagnostics),
                "lightgbm_settings_revision": _configuracoes_pesquisa(
                    rep_config
                ).get("settings_revision"),
                "lightgbm_profile_id": _configuracoes_pesquisa(
                    rep_config
                ).get("profile_id"),
                "lightgbm_early_stopping_enabled": False,
                "lightgbm_early_stopping_rounds": None,
                "lightgbm_validation_role": (
                    "diagnostic_only_existing_calibration_window"
                ),
                "lightgbm_fold_diagnostics": model_fold_diagnostics,
                "lightgbm_predictive_diagnostics": predictive_diagnostics,
                "latest_research_tree": latest_tree,
            }
        )

        margin_by_fold = {
            item["fold_id"]: item
            for item in margin_details
        }
        for item in result.metrics["walk_forward_folds"]:
            item.update(
                margin_by_fold.get(item["fold_id"], {})
            )

        try:
            import lightgbm
            result.metrics["lightgbm_version"] = str(
                lightgbm.__version__
            )
        except Exception:
            result.metrics["lightgbm_version"] = None

        results.append(result)
        report(
            run_index / repetitions,
            (
                f"LightGBM Utility run "
                f"{run_index}/{repetitions} completed"
            ),
            run_index,
        )

    results.sort(
        key=lambda item: int(
            item.metrics.get("repetition_index", 1)
        )
    )
    return results
