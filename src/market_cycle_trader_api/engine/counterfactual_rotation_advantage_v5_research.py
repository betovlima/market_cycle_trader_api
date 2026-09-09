from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from . import research_challengers as challenger
from .capital_rotation import (
    RotationRunResult,
    _fold_performance,
    _scheduled_policy,
    _simulate_exact,
)
from ..services.counterfactual_rotation_advantage_v5 import (
    CounterfactualRotationAdvantageV5Dataset,
    build_counterfactual_rotation_advantage_v5_dataset,
    build_counterfactual_rotation_advantage_v5_policy,
    fit_counterfactual_rotation_advantage_v5,
)


def run_counterfactual_rotation_advantage_v5(
    bars_by_symbol: dict[str, pd.DataFrame],
    config: Any,
    fee_calculator: Callable,
    slippage: Callable,
    progress_callback: Callable[[float, str, int], None] | None = None,
    trade_callback: Callable[[dict[str, Any]], None] | None = None,
    progress_detail_callback: Callable[[dict[str, Any]], None] | None = None,
    technical_log_callback: Callable[[str], None] | None = None,
) -> list[RotationRunResult]:
    (
        frames,
        common_dates,
        symbols,
        folds,
        all_decision_dates,
        decision_to_fold,
        decision_metadata,
    ) = challenger._build_execution_context(bars_by_symbol, config)

    repetitions = int(config.rotation_xgb_repetitions)
    seed_step = int(config.rotation_seed_step)
    total_folds = len(folds)

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
                f"Prepared {len(symbols)} assets and {len(folds)} folds — "
                "LightGBM opportunity model + Counterfactual Rotation Advantage v5"
            ),
            0,
        )

    results: list[RotationRunResult] = []
    for repetition in range(repetitions):
        run_index = repetition + 1
        seed = int(config.random_state) + repetition * seed_step
        rep_config = config.model_copy(update={"random_state": seed})
        model_settings = challenger._lightgbm_settings(rep_config)
        policies: dict[int, Callable] = {}
        diagnostics: dict[pd.Timestamp, dict[str, Any]] = {}
        fold_models: list[dict[str, Any]] = []
        calibration_memory: list[CounterfactualRotationAdvantageV5Dataset] = []

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
                def callback(position: int, total: int, device: str) -> None:
                    fraction = position / max(1, total)
                    report(
                        fold_base + fold_span * (
                            start + (end - start) * fraction
                        ),
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

            calibration_models = challenger._lightgbm_fit_models(
                frames,
                symbols,
                train_dates,
                rep_config,
                phase=(
                    f"run_{run_index}_fold_{fold_position}_"
                    "calibration"
                ),
                progress_callback=phase_progress(
                    "opportunity calibration training", 0.02, 0.30
                ),
                technical_log_callback=technical_log_callback,
            )

            report(
                fold_base + fold_span * 0.36,
                (
                    f"Run {run_index}/{repetitions} — "
                    f"fold {fold_position}/{total_folds} — "
                    "building OOS temporal-memory block"
                ),
                repetition,
            )
            current_dataset = build_counterfactual_rotation_advantage_v5_dataset(
                source_fold_id=fold_id,
                utility_models=calibration_models,
                frames=frames,
                symbols=symbols,
                calibration_dates=calibration_dates,
                config=rep_config,
                technical_log_callback=technical_log_callback,
            )
            calibration_memory.append(current_dataset)

            report(
                fold_base + fold_span * 0.44,
                (
                    f"Run {run_index}/{repetitions} — "
                    f"fold {fold_position}/{total_folds} — "
                    f"fitting expanding temporal rotation memory "
                    f"({len(calibration_memory)} block(s))"
                ),
                repetition,
            )
            fitted_model = fit_counterfactual_rotation_advantage_v5(
                datasets=list(calibration_memory),
                config=rep_config,
                model_settings=model_settings,
                technical_log_callback=technical_log_callback,
            )
            fold_models.append(
                {"fold_id": fold_id, **fitted_model.metadata()}
            )

            final_models = challenger._lightgbm_fit_models(
                frames,
                symbols,
                final_fit_dates,
                rep_config,
                phase=f"run_{run_index}_fold_{fold_position}_final",
                progress_callback=phase_progress(
                    "opportunity final training", 0.55, 0.90
                ),
                technical_log_callback=technical_log_callback,
            )
            policies[fold_id] = build_counterfactual_rotation_advantage_v5_policy(
                fitted_model=fitted_model,
                utility_models=final_models,
                frames=frames,
                symbols=symbols,
                config=rep_config,
                diagnostics=diagnostics,
                fold_id=fold_id,
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
                "simulating OOS expanding-memory rotation portfolio"
            ),
            repetition,
        )
        scheduled = _scheduled_policy(policies, decision_to_fold)

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
                        "model_family": "lightgbm_counterfactual_rotation_advantage_v5",
                        "random_seed": _seed,
                        "repetition_index": _run,
                        "model": "LightGBM + Counterfactual Rotation Advantage v5",
                    }
                )
                trade_callback(payload)

        result = _simulate_exact(
            "lightgbm_counterfactual_rotation_advantage_v5",
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
            model_label="LightGBM + Counterfactual Rotation Advantage v5",
            method_line=(
                "- LightGBM identifies the current Top-1 growth opportunity; "
                "the rotation model learns the same persistent ROTATE-versus-HOLD "
                "target as v4, but its training memory expands with every prior "
                "out-of-sample calibration block available before the test fold. "
                "CASH remains intentionally excluded."
            ),
        )
        result.backend = "lightgbm_counterfactual_rotation_advantage_v5"
        result.metrics.update(
            {
                "backend": result.backend,
                "model_family": "lightgbm_counterfactual_rotation_advantage_v5",
                "strategy_label": "LightGBM + Counterfactual Rotation Advantage v5",
                "random_seed": seed,
                "repetition_index": run_index,
                "repetition_count": repetitions,
                "walk_forward_fold_count": len(folds),
                "walk_forward_folds": _fold_performance(
                    result.predictions,
                    folds,
                    float(rep_config.initial_capital),
                ),
                "counterfactual_rotation_advantage_v5_enabled": True,
                "counterfactual_rotation_advantage_v5_folds": fold_models,
                "expanding_temporal_memory_enabled": True,
                "cash_action_enabled": False,
                "effective_compute_device": "cpu",
                "deterministic_execution": bool(
                    rep_config.deterministic_execution
                ),
                "numeric_thread_limit": int(
                    rep_config.numeric_thread_limit
                ),
            }
        )
        results.append(result)

        report(
            run_base + run_span,
            f"Run {run_index}/{repetitions} completed",
            run_index,
        )

    return results
