"""Protected operational replay for the TCC main U67 v1.21.0 Control.

The scientific implementation is vendored from the exact TCC main commit
recorded in tcc_u67_v1210_reference.contract. This adapter only supplies
current MCT bars, the current analysis cutoff and MCT progress callbacks.

It deliberately does not make the strategy live-Trader eligible.
"""
from __future__ import annotations

from typing import Any, Callable

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
from ..tcc_u67_v1210_reference.rotacao import (
    _benchmark_pesos_iguais,
    _crescimento_politica_simples,
    _politica_agendada,
    _politica_utilidade,
    _precalcular_utilidades_modelo,
    _simular_exato,
    preparar_painel_rotacao,
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
