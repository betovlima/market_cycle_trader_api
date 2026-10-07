"""Protected operational replay for the TCC main U67 v1.21.0 Control.

The scientific implementation is vendored from the exact TCC main commit
recorded in tcc_u67_v1210_reference.contract. This adapter only supplies
current MCT bars, the current analysis cutoff and MCT progress callbacks.

It deliberately does not make the strategy live-Trader eligible.
"""
from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..core.config import TCC_U67_CONTROL_OPERATIONAL_MODE
from ..tcc_u67_v1210_reference.configuracao import (
    CONFIG as TCC_CONFIG,
    construir_configuracao_controle,
)
from ..tcc_u67_v1210_reference.contract import (
    EXPECTED_ENDING_CAPITAL,
    REPRODUCTION_VERSION,
    SOURCE_COMMIT,
    SOURCE_REPOSITORY,
    U67_REQUESTED_ASSETS,
)
from ..tcc_u67_v1210_reference.execucao import (
    aplicar_deslizamento,
    calcular_taxas_referencia,
)
from ..tcc_u67_v1210_reference.modelo_lightgbm import (
    _ajustar_modelos_lightgbm,
    _construir_contexto_execucao,
    _selecionar_switch_margin_fold,
)
from .live_lightgbm_signal import LiveLightGBMDecision
from ..tcc_u67_v1210_reference.rotacao import (
    _benchmark_pesos_iguais,
    _crescimento_politica_simples,
    _politica_agendada,
    _politica_utilidade,
    _precalcular_utilidades_modelo,
    _simular_exato,
    preparar_painel_rotacao,
    ROTATION_FEATURES,
)


U67_OPERATIONAL_CONTRACT = "tcc_main_u67_v1.21.0_control"
U67_REFERENCE_CHECKPOINT_CAPITAL = EXPECTED_ENDING_CAPITAL
U56_COUNT = 56
U67_COUNT = 67


def _value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def tcc_u67_contract_issues(config: Any) -> list[str]:
    """Return deviations from the protected U67 operational identity."""
    issues: list[str] = []
    mode = str(_value(config, "strategy_mode") or "")
    if mode != TCC_U67_CONTROL_OPERATIONAL_MODE:
        issues.append(
            "strategy_mode: expected="
            f"{TCC_U67_CONTROL_OPERATIONAL_MODE!r}, actual={mode!r}"
        )

    assets = tuple(
        str(item).strip().upper()
        for item in (_value(config, "assets", []) or [])
    )
    if assets != tuple(U67_REQUESTED_ASSETS):
        issues.append(
            "assets: expected exact U67 requested universe in scientific order"
        )

    exact = {
        "start_date": "2016-01-01",
        "timeframe": "1Day",
        "market_data_provider": "alpaca",
        "alpaca_historical_feed": "sip",
        "alpaca_adjustment": "raw",
        "rotation_horizon_days": 40,
        "rotation_target_horizons": [5, 10, 20, 40, 60],
        "rotation_target_horizon_weights": [0.10, 0.15, 0.20, 0.30, 0.25],
        "rotation_movement_capture_weight": 0.35,
        "rotation_trend_persistence_weight": 0.20,
        "rotation_minimum_training_rows": 700,
        "rotation_walk_forward_calibration_days": 126,
        "rotation_walk_forward_test_days": 504,
        "rotation_walk_forward_min_test_days": 126,
        "rotation_purge_days": 60,
        "rotation_downside_penalty": 0.20,
        "rotation_drawdown_penalty": 0.35,
        "rotation_min_holding_days": 2,
        "rotation_min_expected_edge": 0.001,
        "rotation_cash_threshold": 0.0,
        "rotation_switch_margin": 0.0005,
        "rotation_switch_margin_candidates": [0.0, 0.0025, 0.005, 0.01],
        "initial_capital": 10_000.0,
        "whole_shares": False,
        "slippage_bps": 0.0,
        "commission_rate": 0.0,
        "sec_fee_rate": 0.0000206,
        "taf_fee_per_share": 0.000195,
        "taf_fee_cap": 9.79,
        "cat_fee_per_share": 0.000003,
        "random_state": 42,
    }
    for name, expected in exact.items():
        actual = _value(config, name)
        if isinstance(expected, list):
            if list(actual or []) != expected:
                issues.append(
                    f"{name}: expected={expected!r}, actual={actual!r}"
                )
        elif isinstance(expected, float):
            try:
                matches = abs(float(actual) - expected) <= 1e-12
            except (TypeError, ValueError):
                matches = False
            if not matches:
                issues.append(
                    f"{name}: expected={expected!r}, actual={actual!r}"
                )
        elif actual != expected:
            issues.append(
                f"{name}: expected={expected!r}, actual={actual!r}"
            )

    end_date = _value(config, "end_date")
    if end_date not in {None, ""}:
        issues.append("end_date: U67 operational mode must remain open-ended")
    return issues


def assert_tcc_u67_operational_contract(config: Any) -> None:
    issues = tcc_u67_contract_issues(config)
    if issues:
        raise ValueError(
            "TCC U67 v1.21.0 operational contract mismatch: "
            + "; ".join(issues)
        )


def tcc_u67_expected_lightgbm_settings() -> dict[str, Any]:
    config = construir_configuracao_controle(
        TCC_CONFIG,
        assets=U67_REQUESTED_ASSETS,
    )
    settings = dict(config.research_model_settings or {})
    return dict(settings.get("lightgbm") or {})


def tcc_u67_model_values() -> dict[str, Any]:
    values = tcc_u67_expected_lightgbm_settings()
    values.pop("early_stopping_enabled", None)
    return values


def tcc_u67_model_snapshot_issues(snapshot: Any) -> list[str]:
    raw = snapshot if isinstance(snapshot, dict) else {}
    issues: list[str] = []
    family = str(raw.get("family") or "")
    if family != "lightgbm_utility":
        return [
            f"family: expected='lightgbm_utility', actual={family!r}"
        ]
    settings_snapshot = (
        raw.get("settings_snapshot")
        if isinstance(raw.get("settings_snapshot"), dict)
        else {}
    )
    actual = (
        settings_snapshot.get("lightgbm")
        if isinstance(settings_snapshot.get("lightgbm"), dict)
        else {}
    )
    expected = tcc_u67_expected_lightgbm_settings()
    for key, expected_value in expected.items():
        actual_value = (
            actual.get(key, False)
            if key == "early_stopping_enabled"
            else actual.get(key)
        )
        if isinstance(expected_value, float):
            try:
                matches = math.isclose(
                    float(actual_value),
                    float(expected_value),
                    rel_tol=0,
                    abs_tol=1e-12,
                )
            except (TypeError, ValueError):
                matches = False
        else:
            matches = actual_value == expected_value
        if not matches:
            issues.append(
                f"lightgbm.{key}: expected={expected_value!r}, "
                f"actual={actual_value!r}"
            )
    return issues


def assert_tcc_u67_model_snapshot(snapshot: Any) -> None:
    issues = tcc_u67_model_snapshot_issues(snapshot)
    if issues:
        raise ValueError(
            "TCC U67 v1.21.0 model snapshot mismatch: "
            + "; ".join(issues)
        )


def tcc_u67_strategy_updates() -> dict[str, Any]:
    """MCT request fields that define the protected U67 profile."""
    return {
        "assets": list(U67_REQUESTED_ASSETS),
        "strategy_mode": TCC_U67_CONTROL_OPERATIONAL_MODE,
        "start_date": "2016-01-01",
        "end_date": None,
        "timeframe": "1Day",
        "market_data_provider": "alpaca",
        "alpaca_historical_feed": "sip",
        "alpaca_live_feed": "iex",
        "alpaca_adjustment": "raw",
        "market_data_history_backfill_enabled": False,
        "market_data_history_backfill_provider": "alpaca",
        "market_data_history_start_tolerance_days": 10,
        "market_data_require_complete_history": True,
        "rotation_horizon_days": 40,
        "rotation_target_horizons": [5, 10, 20, 40, 60],
        "rotation_target_horizon_weights": [0.10, 0.15, 0.20, 0.30, 0.25],
        "rotation_movement_capture_weight": 0.35,
        "rotation_trend_persistence_weight": 0.20,
        "rotation_minimum_training_rows": 700,
        "rotation_walk_forward_enabled": True,
        "rotation_walk_forward_calibration_days": 126,
        "rotation_walk_forward_test_days": 504,
        "rotation_walk_forward_min_test_days": 126,
        "rotation_purge_days": 60,
        "rotation_downside_penalty": 0.20,
        "rotation_drawdown_penalty": 0.35,
        "rotation_min_holding_days": 2,
        "rotation_min_expected_edge": 0.001,
        "rotation_cash_threshold": 0.0,
        "rotation_switch_margin": 0.0005,
        "rotation_switch_margin_candidates": [0.0, 0.0025, 0.005, 0.01],
        "rotation_accelerator": "cpu",
        "rotation_allow_cpu_fallback": False,
        "rotation_xgb_repetitions": 1,
        "rotation_seed_step": 1000,
        "initial_capital": 10_000.0,
        "whole_shares": False,
        "slippage_bps": 0.0,
        "commission_rate": 0.0,
        "sec_fee_rate": 0.0000206,
        "taf_fee_per_share": 0.000195,
        "taf_fee_cap": 9.79,
        "cat_fee_per_share": 0.000003,
        "deterministic_execution": False,
        "numeric_thread_limit": 1,
        "random_state": 42,
    }


def _scientific_config(
    base_config: Any,
    *,
    eligible_assets: tuple[str, ...],
) -> Any:
    config = construir_configuracao_controle(
        TCC_CONFIG,
        assets=eligible_assets,
    )
    updates: dict[str, Any] = {}
    analysis_end = _value(base_config, "analysis_end_date")
    if analysis_end:
        updates["analysis_end_date"] = str(analysis_end)
    analysis_start = _value(base_config, "analysis_start_date")
    if analysis_start:
        updates["analysis_start_date"] = str(analysis_start)
    return config.copiar_modelo(update=updates) if updates else config


def run_tcc_u67_operational_backtest(
    bars_by_symbol: dict[str, pd.DataFrame],
    base_config: Any,
    *,
    progress_callback: Callable[[float, str, int], None] | None = None,
    trade_callback: Callable[[dict[str, Any]], None] | None = None,
    progress_detail_callback: Callable[[dict[str, Any]], None] | None = None,
    technical_log_callback: Callable[[str], None] | None = None,
) -> list[Any]:
    """Replay the protected U67 policy on current, already-audited MCT bars."""
    eligible_assets = tuple(
        symbol
        for symbol in U67_REQUESTED_ASSETS
        if symbol in bars_by_symbol
    )
    if len(eligible_assets) < 2:
        raise ValueError("TCC U67 requires at least two eligible assets.")

    u56_eligible = tuple(
        symbol
        for symbol in tuple(TCC_CONFIG.assets)
        if symbol in bars_by_symbol
    )
    if len(u56_eligible) < 2:
        raise ValueError(
            "TCC U67 requires an eligible subset of the original U56 "
            "to establish the scientific reference calendar."
        )

    config_u56 = construir_configuracao_controle(
        TCC_CONFIG,
        assets=u56_eligible,
    )
    _, reference_calendar, reference_source = preparar_painel_rotacao(
        {
            symbol: bars_by_symbol[symbol]
            for symbol in u56_eligible
        },
        config_u56,
    )

    config_u67 = _scientific_config(
        base_config,
        eligible_assets=eligible_assets,
    )
    (
        frames,
        common_dates,
        _calendar_source,
        symbols,
        folds,
        all_decision_dates,
        decision_to_fold,
        decision_metadata,
    ) = _construir_contexto_execucao(
        {
            symbol: bars_by_symbol[symbol]
            for symbol in eligible_assets
        },
        config_u67,
        calendar_override=reference_calendar,
        calendar_source_label=f"U56_FIXED:{reference_source}",
    )

    if progress_callback is not None:
        progress_callback(
            18.0,
            (
                "Prepared protected TCC U67 Control "
                f"assets={len(symbols)} folds={len(folds)}"
            ),
            0,
        )

    benchmark = _benchmark_pesos_iguais(
        {
            symbol: frames[symbol]
            for symbol in u56_eligible
            if symbol in frames
        },
        [
            symbol
            for symbol in u56_eligible
            if symbol in frames
        ],
        all_decision_dates[1:],
        float(config_u67.initial_capital),
        config_u67,
        calcular_taxas_referencia,
        aplicar_deslizamento,
    )

    policies: dict[int, Callable] = {}
    fold_count = len(folds)
    candidate_margins = tuple(
        float(value)
        for value in config_u67.rotation_switch_margin_candidates
    )

    for fold_position, fold in enumerate(folds, start=1):
        fold_id = int(fold["fold_id"])
        train_dates = common_dates[: int(fold["train_end_index"])]
        calibration_dates = common_dates[
            int(fold["calibration_start_index"]):
            int(fold["calibration_end_index"])
        ]
        final_fit_dates = common_dates[: int(fold["final_fit_end_index"])]
        decision_dates = pd.DatetimeIndex(fold["decision_dates"])

        def report_models(
            phase: str,
            start: float,
            end: float,
        ) -> Callable[[int, int, str], None]:
            def callback(position: int, total: int, device: str) -> None:
                local = position / max(1, total)
                if progress_callback is not None:
                    progress_callback(
                        start + (end - start) * local,
                        (
                            f"TCC U67 fold {fold_position}/{fold_count} "
                            f"{phase} {position}/{total}"
                        ),
                        fold_position - 1,
                    )
                if progress_detail_callback is not None:
                    progress_detail_callback(
                        {
                            "fold_index": fold_position,
                            "fold_count": fold_count,
                            "phase": phase,
                            "trained_models": position,
                            "total_models": total,
                            "device": device.upper(),
                        }
                    )
            return callback

        calibration_models = _ajustar_modelos_lightgbm(
            frames,
            symbols,
            train_dates,
            config_u67,
            phase=f"u67_fold_{fold_id}_calibration",
            progress_callback=report_models(
                "calibration training",
                20.0 + 30.0 * ((fold_position - 1) / max(1, fold_count)),
                20.0 + 30.0 * (fold_position / max(1, fold_count)),
            ),
            technical_log_callback=technical_log_callback,
        )
        calibration_cache, _ = _precalcular_utilidades_modelo(
            calibration_models,
            frames,
            symbols,
            calibration_dates,
            config_u67,
        )
        candidate_scores: list[tuple[float, float]] = []
        for margin in candidate_margins:
            policy = _politica_utilidade(
                calibration_models,
                frames,
                symbols,
                config_u67,
                margin,
                utility_cache=calibration_cache,
            )
            score = _crescimento_politica_simples(
                policy,
                frames,
                symbols,
                calibration_dates,
                config_u67,
            )
            candidate_scores.append((margin, float(score)))

        selection = _selecionar_switch_margin_fold(
            config_u67,
            fold_id,
            candidate_scores,
        )
        selected_margin = float(
            selection["selected_candidate_margin"]
        )
        effective_margin = max(
            float(config_u67.rotation_switch_margin),
            selected_margin,
        )

        final_models = _ajustar_modelos_lightgbm(
            frames,
            symbols,
            final_fit_dates,
            config_u67,
            phase=f"u67_fold_{fold_id}_final",
            progress_callback=report_models(
                "final training",
                52.0 + 30.0 * ((fold_position - 1) / max(1, fold_count)),
                52.0 + 30.0 * (fold_position / max(1, fold_count)),
            ),
            technical_log_callback=technical_log_callback,
        )
        decision_cache, _ = _precalcular_utilidades_modelo(
            final_models,
            frames,
            symbols,
            decision_dates,
            config_u67,
        )
        policies[fold_id] = _politica_utilidade(
            final_models,
            frames,
            symbols,
            config_u67,
            effective_margin,
            fold_id=fold_id,
            calibrated_switch_margin=selected_margin,
            utility_cache=decision_cache,
        )

    scheduled = _politica_agendada(
        policies,
        decision_to_fold,
    )
    result = _simular_exato(
        "tcc_u67_v1210_control",
        scheduled,
        frames,
        symbols,
        all_decision_dates,
        config_u67,
        calcular_taxas_referencia,
        aplicar_deslizamento,
        decision_metadata=decision_metadata,
        trade_callback=trade_callback,
        model_label="TCC U67 v1.21.0 Control",
        method_line=(
            "- Protected U67 policy vendored from TCC main; "
            "current MCT audited market data."
        ),
        benchmark_override=benchmark,
        benchmark_override_name=(
            "Fixed eligible U56 equal-weight buy-and-hold"
        ),
    )
    result.metrics.update(
        {
            "operational_contract": U67_OPERATIONAL_CONTRACT,
            "tcc_source_repository": SOURCE_REPOSITORY,
            "tcc_source_commit": SOURCE_COMMIT,
            "tcc_reproduction_version": REPRODUCTION_VERSION,
            "tcc_reference_checkpoint_capital": (
                U67_REFERENCE_CHECKPOINT_CAPITAL
            ),
            "tcc_requested_asset_count": U67_COUNT,
            "tcc_eligible_asset_count": len(symbols),
            "tcc_structural_reductions": sorted(
                set(U67_REQUESTED_ASSETS).difference(symbols)
            ),
            "tcc_reference_calendar_source": str(reference_source),
            "tcc_checkpoint_is_optimization_target": False,
        }
    )
    if progress_callback is not None:
        progress_callback(
            92.0,
            "TCC U67 operational replay completed",
            1,
        )
    return [result]


def _prepare_live_u67_panel(
    bars_by_symbol: dict[str, pd.DataFrame],
    config: Any,
) -> tuple[
    dict[str, pd.DataFrame],
    list[str],
    pd.DatetimeIndex,
    Any,
]:
    eligible_assets = tuple(
        symbol
        for symbol in U67_REQUESTED_ASSETS
        if symbol in bars_by_symbol
    )
    if len(eligible_assets) < 2:
        raise ValueError("TCC U67 live runtime requires at least two eligible assets.")

    u56_eligible = tuple(
        symbol
        for symbol in tuple(TCC_CONFIG.assets)
        if symbol in bars_by_symbol
    )
    if len(u56_eligible) < 2:
        raise ValueError(
            "TCC U67 live runtime requires an eligible U56 subset "
            "for the fixed scientific calendar."
        )

    config_u56 = construir_configuracao_controle(
        TCC_CONFIG,
        assets=u56_eligible,
    )
    _, reference_calendar, reference_source = preparar_painel_rotacao(
        {symbol: bars_by_symbol[symbol] for symbol in u56_eligible},
        config_u56,
    )
    config_u67 = _scientific_config(
        config,
        eligible_assets=eligible_assets,
    )
    frames, calendar, _ = preparar_painel_rotacao(
        {symbol: bars_by_symbol[symbol] for symbol in eligible_assets},
        config_u67,
        calendar_override=reference_calendar,
        calendar_source_label=f"U56_FIXED:{reference_source}",
    )
    symbols = [
        symbol
        for symbol in U67_REQUESTED_ASSETS
        if symbol in frames
    ]
    if len(symbols) < 2:
        raise ValueError("TCC U67 live runtime has fewer than two modelable assets.")
    return frames, symbols, pd.DatetimeIndex(calendar), config_u67


def _live_u67_utilities(
    models: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    timestamp: pd.Timestamp,
) -> np.ndarray:
    values = [0.0]
    for symbol in symbols:
        model = models.get(symbol)
        frame = frames.get(symbol)
        if model is None or frame is None or timestamp not in frame.index:
            values.append(float("-inf"))
            continue
        row = frame.loc[[timestamp], ROTATION_FEATURES]
        if row.empty or row.isna().any(axis=None):
            values.append(float("-inf"))
            continue
        values.append(float(model.predict(row)[0]))
    return np.asarray(values, dtype=np.float64)


def _apply_live_u67_policy(
    utilities: np.ndarray,
    symbols: list[str],
    config: Any,
    *,
    current_asset: str | None,
    holding_sessions: int,
    calibrated_margin: float,
) -> tuple[str, str, float]:
    labels = ["CASH", *symbols]
    current_label = str(current_asset or "CASH").strip().upper()
    if current_label not in labels:
        raise ValueError(
            f"Current position {current_label!r} is outside the effective U67 universe."
        )
    if current_label == "CASH" and holding_sessions:
        raise ValueError("CASH state cannot have positive holding_sessions.")
    if type(holding_sessions) is not int or holding_sessions < 0:
        raise ValueError("holding_sessions must be a nonnegative integer.")
    if not np.isfinite(utilities[1:]).any():
        raise ValueError("No finite U67 utility is available for the completed session.")

    current_position = labels.index(current_label)
    best = int(np.nanargmax(utilities))
    best_value = float(utilities[best])
    current_value = float(utilities[current_position])
    minimum = float(config.rotation_cash_threshold)
    entry_threshold = minimum + float(config.rotation_min_expected_edge)
    required = max(
        float(config.rotation_switch_margin),
        float(calibrated_margin),
    )

    if (
        current_position > 0
        and np.isfinite(current_value)
        and holding_sessions < int(config.rotation_min_holding_days)
    ):
        target = current_position
        score = current_value
    elif best == 0 or best_value <= minimum:
        target = 0
        score = 0.0
    elif current_position == 0:
        if best_value >= entry_threshold:
            target = best
            score = best_value
        else:
            target = 0
            score = 0.0
    elif best == current_position:
        target = current_position
        score = current_value
    elif best_value >= current_value + required:
        target = best
        score = best_value
    else:
        target = current_position
        score = current_value

    return labels[target], labels[best], float(score)


def build_live_tcc_u67_decision(
    bars_by_symbol: dict[str, pd.DataFrame],
    config: Any,
    *,
    current_asset: str | None,
    holding_sessions: int,
) -> LiveLightGBMDecision:
    """Build one protected next-open U67 decision from completed daily bars only."""
    assert_tcc_u67_operational_contract(config)
    frames, symbols, common_dates, scientific_config = _prepare_live_u67_panel(
        bars_by_symbol,
        config,
    )
    if len(common_dates) < 2:
        raise ValueError("TCC U67 live runtime has insufficient aligned history.")

    decision_date = pd.Timestamp(common_dates[-1])
    purge = max(
        int(scientific_config.rotation_purge_days),
        max(int(item) for item in scientific_config.rotation_target_horizons),
    )
    calibration_days = int(
        scientific_config.rotation_walk_forward_calibration_days
    )
    minimum_training_rows = int(
        scientific_config.rotation_minimum_training_rows
    )
    prospective_execution_index = len(common_dates)
    calibration_end_index = prospective_execution_index - purge
    calibration_start_index = calibration_end_index - calibration_days
    train_end_index = calibration_start_index - purge
    final_fit_end_index = prospective_execution_index - purge

    if (
        train_end_index < minimum_training_rows
        or calibration_start_index < 0
        or calibration_end_index <= calibration_start_index
        or final_fit_end_index <= 0
    ):
        raise ValueError(
            "Insufficient completed history for U67 live training/calibration."
        )

    train_dates = common_dates[:train_end_index]
    calibration_dates = common_dates[
        calibration_start_index:calibration_end_index
    ]
    final_fit_dates = common_dates[:final_fit_end_index]

    calibration_models = _ajustar_modelos_lightgbm(
        frames,
        symbols,
        train_dates,
        scientific_config,
        phase="u67_live_calibration",
    )
    if not calibration_models:
        raise ValueError("No U67 calibration models are available.")

    candidate_scores: list[tuple[float, float]] = []
    for margin in tuple(
        float(value)
        for value in scientific_config.rotation_switch_margin_candidates
    ):
        policy = _politica_utilidade(
            calibration_models,
            frames,
            symbols,
            scientific_config,
            margin,
        )
        score = _crescimento_politica_simples(
            policy,
            frames,
            symbols,
            calibration_dates,
            scientific_config,
        )
        candidate_scores.append((margin, float(score)))

    selection = _selecionar_switch_margin_fold(
        scientific_config,
        0,
        candidate_scores,
    )
    calibrated_margin = float(
        selection["selected_candidate_margin"]
    )
    effective_margin = max(
        float(scientific_config.rotation_switch_margin),
        calibrated_margin,
    )

    final_models = _ajustar_modelos_lightgbm(
        frames,
        symbols,
        final_fit_dates,
        scientific_config,
        phase="u67_live_final",
    )
    if not final_models:
        raise ValueError("No U67 final models are available.")

    utilities = _live_u67_utilities(
        final_models,
        frames,
        symbols,
        decision_date,
    )
    target_asset, raw_best_asset, selected_utility = _apply_live_u67_policy(
        utilities,
        symbols,
        scientific_config,
        current_asset=current_asset,
        holding_sessions=holding_sessions,
        calibrated_margin=calibrated_margin,
    )
    labels = ["CASH", *symbols]
    utility_map = {
        label: float(utilities[index])
        for index, label in enumerate(labels)
        if np.isfinite(float(utilities[index]))
    }

    return LiveLightGBMDecision(
        decision_date=decision_date,
        current_asset=str(current_asset or "CASH").strip().upper(),
        target_asset=target_asset,
        raw_best_asset=raw_best_asset,
        selected_utility=float(selected_utility),
        utilities=utility_map,
        cash_edges={},
        opportunity_probability=None,
        opportunity_confidence=None,
        opportunity_threshold=None,
        opportunity_accepted=None,
        effective_switch_margin=float(effective_margin),
        calibrated_candidate_margin=float(calibrated_margin),
        calibration_score=float(selection["selected_calibration_score"]),
        training_end=pd.Timestamp(train_dates[-1]),
        calibration_start=pd.Timestamp(calibration_dates[0]),
        calibration_end=pd.Timestamp(calibration_dates[-1]),
        final_fit_end=pd.Timestamp(final_fit_dates[-1]),
        effective_compute_device="cpu",
        compute_fallback_reason=None,
        random_state=int(scientific_config.random_state),
    )
