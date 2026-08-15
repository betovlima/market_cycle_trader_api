from __future__ import annotations

from copy import deepcopy
import math
import os
import time
from typing import Any, Callable

import numpy as np
import pandas as pd

from .capital_rotation import (
    ROTATION_FEATURES,
    SUPPORTED_ROTATION_MODES,
    RotationRunResult,
    _analysis_decision_dates,
    _optimized_policy,
    _compound_risk_overlay_policy,
    _scheduled_allocation_policy,
    _simulate_optimized_allocation,
    _build_walk_forward_folds,
    _cash_gate_action_log_return,
    _fold_performance,
    _risk_adjusted_reward,
    _risk_off_enabled,
    _scheduled_policy,
    _simple_policy_growth,
    _simulate_exact,
    _training_transition_log_return,
    _xgb_policy,
    _xgb_utilities,
    prepare_rotation_panel,
)
from .optimized_allocation import fit_expected_return_calibrator
from .absolute_utility_cash_gate import absolute_utility_cash_gate_enabled
from .concentrated_allocation import concentrated_allocation_enabled, portfolio_allocation_enabled
from .compound_risk_overlay import allocation_execution_enabled, compound_risk_overlay_enabled
from .selective_opportunity import (
    AdaptiveOpportunityCashGate,
    build_base_policy_opportunity_samples,
    fit_adaptive_opportunity_cash_gate,
    fit_selective_opportunity_gate,
    opportunity_cash_gate_enabled,
    selective_opportunity_enabled,
)


def _effective_n_jobs(configured: int) -> int:
    raw = str(os.getenv("MCT_MODEL_THREADS_OVERRIDE") or "").strip()
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return int(configured)


def _research_settings(config: Any) -> dict[str, Any]:
    raw = getattr(config, "research_model_settings", {}) or {}
    return dict(raw) if isinstance(raw, dict) else {}


def _lightgbm_settings(config: Any) -> dict[str, Any]:
    settings = _research_settings(config)
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
    return dict(lightgbm)


def _build_execution_context(
    bars_by_symbol: dict[str, pd.DataFrame],
    config: Any,
) -> tuple[
    dict[str, pd.DataFrame],
    pd.DatetimeIndex,
    list[str],
    list[dict[str, Any]],
    pd.DatetimeIndex,
    dict[pd.Timestamp, int],
    dict[pd.Timestamp, dict[str, Any]],
]:
    if config.strategy_mode not in SUPPORTED_ROTATION_MODES:
        raise ValueError(f"Unsupported research strategy mode: {config.strategy_mode}.")
    if list(config.rotation_models) != ["xgboost_utility"]:
        raise ValueError("Research challengers cannot mutate the locked strategy model contract.")

    frames, common_dates = prepare_rotation_panel(bars_by_symbol, config)
    symbols = sorted(frames)
    folds = _build_walk_forward_folds(common_dates, config)
    all_decision_dates = _analysis_decision_dates(common_dates, folds, config)
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
        symbols,
        folds,
        all_decision_dates,
        decision_to_fold,
        decision_metadata,
    )


def _lightgbm_fit_models(
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
        raise RuntimeError("LightGBM research requires lightgbm. Install requirements.txt.") from exc

    anchor_assets = set(getattr(config, "calendar_anchor_assets", []) or [])
    minimum_rows = int(config.rotation_minimum_training_rows)
    settings = _lightgbm_settings(config)
    fitted: dict[str, Any] = {}
    started = time.perf_counter()

    def technical(message: str) -> None:
        if technical_log_callback is not None:
            technical_log_callback(message)

    technical(
        f"model=lightgbm phase={phase} event=fit_start device=cpu "
        f"models={len(symbols)} train_sessions={len(train_dates)} "
        f"estimators={int(settings['n_estimators'])} seed={int(config.random_state)}"
    )
    for position, symbol in enumerate(symbols, start=1):
        frame = frames[symbol].loc[train_dates].dropna(
            subset=[target_column, *ROTATION_FEATURES]
        )
        if len(frame) < minimum_rows:
            if symbol in anchor_assets:
                raise ValueError(
                    f"{symbol}: only {len(frame)} utility rows are available; "
                    f"{minimum_rows} are required for an anchor asset."
                )
            if progress_callback is not None:
                progress_callback(position, len(symbols), "cpu")
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
            n_jobs=_effective_n_jobs(int(settings["n_jobs"])),
            deterministic=bool(config.deterministic_execution),
            force_col_wise=bool(config.deterministic_execution),
            verbosity=-1,
        )
        model.fit(frame[ROTATION_FEATURES], frame[target_column])
        fitted[symbol] = model
        if progress_callback is not None:
            progress_callback(position, len(symbols), "cpu")
    technical(
        f"model=lightgbm phase={phase} event=fit_complete device=cpu "
        f"models={len(fitted)} duration_seconds={time.perf_counter() - started:.3f}"
    )
    return fitted


def _run_lightgbm(
    bars_by_symbol: dict[str, pd.DataFrame],
    config: Any,
    fee_calculator: Callable,
    slippage: Callable,
    *,
    progress_callback: Callable[[float, str, int], None] | None,
    trade_callback: Callable[[dict[str, Any]], None] | None,
    progress_detail_callback: Callable[[dict[str, Any]], None] | None,
    technical_log_callback: Callable[[str], None] | None,
) -> list[RotationRunResult]:
    (
        frames,
        common_dates,
        symbols,
        folds,
        all_decision_dates,
        decision_to_fold,
        decision_metadata,
    ) = _build_execution_context(bars_by_symbol, config)

    repetitions = int(config.rotation_xgb_repetitions)
    seed_step = int(config.rotation_seed_step)
    total_folds = len(folds)
    total_models = len(symbols)

    def report(fraction: float, stage: str, completed: int) -> None:
        if progress_callback is not None:
            progress_callback(20.0 + 72.0 * max(0.0, min(1.0, fraction)), stage, completed)

    def detail(**values: Any) -> None:
        if progress_detail_callback is not None:
            progress_detail_callback(values)

    if progress_callback is not None:
        progress_callback(
            18.0,
            f"Prepared {len(symbols)} assets and {len(folds)} folds — LightGBM=CPU",
            0,
        )

    results: list[RotationRunResult] = []
    for repetition in range(repetitions):
        run_index = repetition + 1
        seed = int(config.random_state) + repetition * seed_step
        rep_config = config.model_copy(update={"random_state": seed})
        policies: dict[int, Callable] = {}
        risk_overlay_state = {"position": 0, "holding_days": 0}
        cash_gate_base_state: dict[str, Any] = {"position": 0, "holding_days": 0, "pending_sample": None}
        cash_gate_oos_history: list[dict[str, Any]] = []
        diagnostics: dict[pd.Timestamp, dict[str, Any]] = {}
        margin_details: list[dict[str, Any]] = []
        run_base = repetition / repetitions
        run_span = 1.0 / repetitions
        fold_span = (run_span * 0.90) / max(1, total_folds)

        for fold_position, fold in enumerate(folds, start=1):
            fold_base = run_base + (fold_position - 1) * fold_span
            fold_id = int(fold["fold_id"])
            train_dates = common_dates[: int(fold["train_end_index"])]
            calibration_dates = common_dates[
                int(fold["calibration_start_index"]): int(fold["calibration_end_index"])
            ]
            final_fit_dates = common_dates[: int(fold["final_fit_end_index"])]

            def phase_progress(label: str, start: float, end: float):
                def callback(position: int, total: int, device: str) -> None:
                    fraction = position / max(1, total)
                    report(
                        fold_base + fold_span * (start + (end - start) * fraction),
                        f"Run {run_index}/{repetitions} — fold {fold_position}/{total_folds} — {label} {position}/{total}",
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
            calibration_models = _lightgbm_fit_models(
                frames,
                symbols,
                train_dates,
                rep_config,
                phase=f"run_{run_index}_fold_{fold_position}_calibration",
                progress_callback=phase_progress("calibration training", 0.02, 0.38),
                technical_log_callback=technical_log_callback,
            )
            calibration_cash_edge_models = None
            if _risk_off_enabled(rep_config):
                calibration_cash_edge_models = _lightgbm_fit_models(
                    frames,
                    symbols,
                    train_dates,
                    rep_config,
                    phase=f"run_{run_index}_fold_{fold_position}_calibration_cash_edge",
                    technical_log_callback=technical_log_callback,
                    target_column="forward_cash_edge",
                )
            opportunity_gate = None
            expected_return_calibrator = None
            label_horizon = max(int(item) for item in rep_config.rotation_target_horizons)
            if selective_opportunity_enabled(rep_config) and not opportunity_cash_gate_enabled(rep_config):
                opportunity_gate = fit_selective_opportunity_gate(
                    calibration_models,
                    frames,
                    symbols,
                    calibration_dates,
                    lambda fitted, panel, labels, ts: _xgb_utilities(fitted, panel, labels, ts, rep_config),
                    random_state=int(rep_config.random_state),
                    label_horizon=label_horizon,
                    hysteresis=False,
                )
            if portfolio_allocation_enabled(rep_config):
                expected_return_calibrator = fit_expected_return_calibrator(
                    calibration_models,
                    frames,
                    symbols,
                    calibration_dates,
                    lambda fitted, panel, labels, ts: _xgb_utilities(fitted, panel, labels, ts, rep_config),
                    label_horizon=label_horizon,
                )
            candidate_margins = tuple(float(value) for value in rep_config.rotation_switch_margin_candidates)
            best_candidate = candidate_margins[0]
            best_score = float("-inf")
            margin_config = (
                rep_config.model_copy(update={"strategy_mode": "COMPOUND_ROTATION_SWING_XGBOOST"})
                if selective_opportunity_enabled(rep_config) or absolute_utility_cash_gate_enabled(rep_config)
                else rep_config
            )
            for candidate in candidate_margins:
                calibration_policy = _xgb_policy(
                    calibration_models,
                    frames,
                    symbols,
                    margin_config,
                    candidate,
                    cash_edge_models=calibration_cash_edge_models,
                )
                score = _simple_policy_growth(
                    calibration_policy,
                    frames,
                    symbols,
                    calibration_dates,
                    rep_config,
                )
                if score > best_score:
                    best_score = score
                    best_candidate = candidate

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
            final_models = _lightgbm_fit_models(
                frames,
                symbols,
                final_fit_dates,
                rep_config,
                phase=f"run_{run_index}_fold_{fold_position}_final",
                progress_callback=phase_progress("final training", 0.50, 0.90),
                technical_log_callback=technical_log_callback,
            )
            final_cash_edge_models = None
            if _risk_off_enabled(rep_config):
                final_cash_edge_models = _lightgbm_fit_models(
                    frames,
                    symbols,
                    final_fit_dates,
                    rep_config,
                    phase=f"run_{run_index}_fold_{fold_position}_final_cash_edge",
                    technical_log_callback=technical_log_callback,
                    target_column="forward_cash_edge",
                )
            effective_margin = max(float(rep_config.rotation_switch_margin), float(best_candidate))
            if opportunity_cash_gate_enabled(rep_config):
                gate_base_config = rep_config.model_copy(update={"strategy_mode": "COMPOUND_ROTATION_SWING_XGBOOST"})
                calibration_base_policy = _xgb_policy(
                    calibration_models,
                    frames,
                    symbols,
                    gate_base_config,
                    effective_margin,
                    calibrated_switch_margin=float(best_candidate),
                )
                initial_gate_samples = build_base_policy_opportunity_samples(
                    calibration_models,
                    frames,
                    symbols,
                    calibration_dates,
                    lambda fitted, panel, labels, ts: _xgb_utilities(fitted, panel, labels, ts, rep_config),
                    calibration_base_policy,
                    lambda now, nxt, from_pos, to_pos: _cash_gate_action_log_return(
                        frames, symbols, now, nxt, from_pos, to_pos, rep_config
                    ),
                )
                opportunity_gate = fit_adaptive_opportunity_cash_gate(
                    initial_gate_samples,
                    random_state=int(rep_config.random_state),
                    shared_history=cash_gate_oos_history,
                    fold_id=fold_id,
                )
            if compound_risk_overlay_enabled(rep_config):
                policies[fold_id] = _compound_risk_overlay_policy(
                    final_models,
                    frames,
                    symbols,
                    rep_config,
                    effective_margin,
                    decision_diagnostics=diagnostics,
                    fold_id=fold_id,
                    calibrated_switch_margin=float(best_candidate),
                    state=risk_overlay_state,
                )
            elif portfolio_allocation_enabled(rep_config):
                policies[fold_id] = _optimized_policy(
                    final_models,
                    frames,
                    symbols,
                    rep_config,
                    opportunity_gate=opportunity_gate,
                    expected_return_calibrator=expected_return_calibrator,
                    decision_diagnostics=diagnostics,
                    fold_id=fold_id,
                )
            else:
                policies[fold_id] = _xgb_policy(
                    final_models,
                    frames,
                    symbols,
                    rep_config,
                    effective_margin,
                    cash_edge_models=final_cash_edge_models,
                    opportunity_gate=opportunity_gate,
                    cash_gate_base_state=cash_gate_base_state if (opportunity_cash_gate_enabled(rep_config) or absolute_utility_cash_gate_enabled(rep_config)) else None,
                    decision_diagnostics=diagnostics,
                    fold_id=fold_id,
                    calibrated_switch_margin=float(best_candidate),
                )
            margin_detail = {
                "fold_id": fold_id,
                "calibrated_candidate_margin": float(best_candidate),
                "effective_switch_margin": float(effective_margin),
                "calibration_risk_adjusted_score": float(best_score),
            }
            if absolute_utility_cash_gate_enabled(rep_config):
                margin_detail.update({
                    "absolute_utility_entry_threshold": float(rep_config.opportunity_utility_entry_threshold),
                    "absolute_utility_exit_threshold": float(rep_config.opportunity_utility_exit_threshold),
                    "absolute_utility_threshold_basis": "champion_top1_absolute_utility",
                })
            if opportunity_gate is not None:
                margin_detail.update(
                    {
                        "opportunity_threshold": float(opportunity_gate.threshold),
                        "opportunity_entry_threshold": (float(opportunity_gate.entry_threshold) if opportunity_gate.entry_threshold is not None else None),
                        "opportunity_exit_threshold": (float(opportunity_gate.exit_threshold) if opportunity_gate.exit_threshold is not None else None),
                        "opportunity_training_rows": int(opportunity_gate.training_rows),
                        "opportunity_positive_rate": float(opportunity_gate.positive_rate),
                        "opportunity_threshold_validation_rows": int(opportunity_gate.threshold_validation_rows),
                        "opportunity_threshold_validation_score": float(opportunity_gate.threshold_validation_score),
                        "opportunity_threshold_validation_accepted": int(opportunity_gate.threshold_validation_accepted),
                        "opportunity_threshold_validation_transitions": int(opportunity_gate.threshold_validation_transitions),
                        "opportunity_calibration_method": str(opportunity_gate.calibration_method),
                        "opportunity_threshold_basis": str(opportunity_gate.threshold_basis),
                        "opportunity_target_basis": str(getattr(opportunity_gate, "target_basis", "weighted_forward_net_log_return")),
                        "opportunity_target_horizon_sessions": getattr(opportunity_gate, "target_horizon_sessions", None),
                        "opportunity_regularized_to_base_policy": bool(getattr(opportunity_gate, "regularized_to_base_policy", False)),
                        "opportunity_threshold_validation_alpha": getattr(opportunity_gate, "threshold_validation_alpha", None),
                        "opportunity_threshold_validation_exposure_ratio": getattr(opportunity_gate, "threshold_validation_exposure_ratio", None),
                        "opportunity_refresh_interval_sessions": (int(opportunity_gate.refresh_interval) if isinstance(opportunity_gate, AdaptiveOpportunityCashGate) else None),
                        "opportunity_rolling_sample_window": (int(opportunity_gate.rolling_window) if isinstance(opportunity_gate, AdaptiveOpportunityCashGate) else None),
                    }
                )
            if expected_return_calibrator is not None:
                margin_detail.update(
                    {
                        "allocation_relative_alpha_calibration_method": str(expected_return_calibrator.method),
                        "allocation_relative_alpha_calibration_rows": int(expected_return_calibrator.sample_count),
                        "allocation_relative_alpha_mean": float(expected_return_calibrator.realized_alpha_mean),
                        "allocation_relative_alpha_std": float(expected_return_calibrator.realized_alpha_std),
                        "allocation_expected_return_calibration_method": str(expected_return_calibrator.method),
                        "allocation_expected_return_calibration_rows": int(expected_return_calibrator.sample_count),
                        "allocation_expected_return_mean": float(expected_return_calibrator.realized_return_mean),
                        "allocation_expected_return_std": float(expected_return_calibrator.realized_return_std),
                    }
                )
            margin_details.append(margin_detail)
            report(
                fold_base + fold_span,
                f"Run {run_index}/{repetitions} — fold {fold_position}/{total_folds} completed",
                repetition,
            )

        report(
            run_base + run_span * 0.94,
            f"Run {run_index}/{repetitions} — simulating out-of-sample portfolio",
            repetition,
        )
        scheduled = (
            _scheduled_allocation_policy(policies, decision_to_fold)
            if allocation_execution_enabled(rep_config)
            else _scheduled_policy(policies, decision_to_fold)
        )

        wrapped_trade_callback = None
        if trade_callback is not None:
            def wrapped_trade_callback(trade: dict[str, Any], *, _seed=seed, _run=run_index) -> None:
                payload = dict(trade)
                payload.update(
                    {
                        "model_family": "lightgbm_utility",
                        "random_seed": _seed,
                        "repetition_index": _run,
                        "model": "LightGBM Utility" + (f" · seed {_seed}" if repetitions > 1 else ""),
                    }
                )
                trade_callback(payload)

        simulator = _simulate_optimized_allocation if allocation_execution_enabled(rep_config) else _simulate_exact
        result = simulator(
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
                f"- LightGBM Utility predicts the same weighted multi-horizon risk-adjusted utility "
                f"target across {rep_config.rotation_target_horizons}."
            ),
        )
        backend = "lightgbm_utility" if repetitions <= 1 else f"lightgbm_utility_seed_{seed}"
        result.backend = backend
        result.metrics.update(
            {
                "backend": backend,
                "model_family": "lightgbm_utility",
                "strategy_label": "LightGBM Utility" + (f" · seed {seed}" if repetitions > 1 else ""),
                "random_seed": seed,
                "repetition_index": run_index,
                "repetition_count": repetitions,
                "walk_forward_fold_count": len(folds),
                "walk_forward_folds": _fold_performance(result.predictions, folds, float(rep_config.initial_capital)),
                "effective_switch_margin": float(np.mean([item["effective_switch_margin"] for item in margin_details])),
                "effective_switch_margin_mean": float(np.mean([item["effective_switch_margin"] for item in margin_details])),
                "calibrated_switch_margin": float(np.mean([item["calibrated_candidate_margin"] for item in margin_details])),
                "effective_compute_device": "cpu",
                "deterministic_execution": bool(rep_config.deterministic_execution),
                "numeric_thread_limit": int(rep_config.numeric_thread_limit),
                "decision_diagnostics_schema_version": (
                    11 if compound_risk_overlay_enabled(rep_config)
                    else 10 if concentrated_allocation_enabled(rep_config)
                    else 9 if portfolio_allocation_enabled(rep_config)
                    else 8 if absolute_utility_cash_gate_enabled(rep_config)
                    else 7 if opportunity_cash_gate_enabled(rep_config)
                    else 5 if selective_opportunity_enabled(rep_config)
                    else 3 if _risk_off_enabled(rep_config)
                    else 2
                ),
                "decision_diagnostics_rows": len(diagnostics),
                "lightgbm_settings_revision": _research_settings(rep_config).get("settings_revision"),
                "lightgbm_profile_id": _research_settings(rep_config).get("profile_id"),
            }
        )
        margin_by_fold = {item["fold_id"]: item for item in margin_details}
        for item in result.metrics["walk_forward_folds"]:
            item.update(margin_by_fold.get(item["fold_id"], {}))
        try:
            import lightgbm
            result.metrics["lightgbm_version"] = str(lightgbm.__version__)
        except Exception:
            result.metrics["lightgbm_version"] = None
        results.append(result)
        report(run_index / repetitions, f"LightGBM Utility run {run_index}/{repetitions} completed", run_index)

    results.sort(key=lambda item: int(item.metrics.get("repetition_index", 1)))
    return results


class _ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int) -> None:
        self.capacity = int(capacity)
        self.states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.bootstrap_discounts = np.zeros(self.capacity, dtype=np.float32)
        self.size = 0
        self.position = 0

    def add(self, state, action, reward, next_state, done, bootstrap_discount) -> None:
        index = self.position
        self.states[index] = state
        self.actions[index] = int(action)
        self.rewards[index] = float(reward)
        self.next_states[index] = next_state
        self.dones[index] = float(done)
        self.bootstrap_discounts[index] = float(bootstrap_discount)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        indices = rng.integers(0, self.size, size=int(batch_size))
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            self.bootstrap_discounts[indices],
        )


class _NStepAccumulator:
    def __init__(self, n_step: int, gamma: float) -> None:
        self.n_step = max(1, int(n_step))
        self.gamma = float(gamma)
        self.pending: list[tuple[np.ndarray, int, float, np.ndarray, bool]] = []

    def _emit_one(self):
        count = min(self.n_step, len(self.pending))
        first_state, first_action, _, _, _ = self.pending[0]
        reward = 0.0
        final_next_state = self.pending[count - 1][3]
        final_done = self.pending[count - 1][4]
        for offset in range(count):
            reward += (self.gamma ** offset) * float(self.pending[offset][2])
            if self.pending[offset][4]:
                count = offset + 1
                final_next_state = self.pending[offset][3]
                final_done = True
                break
        self.pending.pop(0)
        return (
            first_state,
            first_action,
            float(reward),
            final_next_state,
            bool(final_done),
            float(self.gamma ** count),
        )

    def append(self, state, action, reward, next_state, done):
        self.pending.append((state, int(action), float(reward), next_state, bool(done)))
        emitted = []
        if len(self.pending) >= self.n_step:
            emitted.append(self._emit_one())
        if done:
            while self.pending:
                emitted.append(self._emit_one())
        return emitted


class _IQNNetwork:
    def __init__(
        self,
        input_dim: int,
        action_count: int,
        hidden_dim: int,
        cosine_embedding_dim: int,
        device: str,
        seed: int,
    ) -> None:
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise RuntimeError("IQN research requires PyTorch. Install requirements.txt.") from exc

        class IQNModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.state_encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                )
                self.tau_projection = nn.Linear(cosine_embedding_dim, hidden_dim)
                self.value_head = nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, action_count),
                )
                basis = torch.arange(1, cosine_embedding_dim + 1, dtype=torch.float32) * math.pi
                self.register_buffer("cosine_basis", basis)

            def forward(self, states, taus):
                state_features = self.state_encoder(states).unsqueeze(1)
                cosine = torch.cos(taus.unsqueeze(-1) * self.cosine_basis.view(1, 1, -1))
                tau_features = torch.relu(self.tau_projection(cosine))
                return self.value_head(state_features * tau_features)

        self.torch = torch
        self.action_count = int(action_count)
        self.device = torch.device(device)
        torch.manual_seed(int(seed))
        if device == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        self.model = IQNModule().to(self.device)

    def quantiles(self, states, taus):
        return self.model(states, taus)


def _iqn_settings(config: Any) -> dict[str, Any]:
    settings = _research_settings(config)
    iqn = settings.get("iqn")
    if not isinstance(iqn, dict):
        raise ValueError("The IQN execution snapshot is missing its protected model settings.")
    required = {
        "training_steps", "episode_days", "replay_size", "learning_starts",
        "batch_size", "learning_rate", "gamma", "n_step", "quantile_samples",
        "target_quantile_samples", "action_quantile_samples", "evaluation_quantiles",
        "hidden_dim", "cosine_embedding_dim", "target_update_steps", "eval_every_steps",
        "epsilon_start", "epsilon_end", "early_stopping_enabled",
        "early_stopping_patience", "minimum_training_steps", "gradient_clip_norm",
        "huber_kappa",
    }
    missing = sorted(required.difference(iqn))
    if missing:
        raise ValueError("IQN research settings are incomplete: " + ", ".join(missing))
    return dict(iqn)


def _iqn_compute_device(config: Any) -> tuple[str, str | None, str | None]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("IQN research requires PyTorch. Install requirements.txt.") from exc
    requested = str(config.rotation_accelerator).strip().lower()
    available = bool(torch.cuda.is_available())
    if requested == "cpu":
        return "cpu", None, str(torch.__version__)
    if available:
        return "cuda", str(torch.cuda.get_device_name(0)), str(torch.__version__)
    if requested == "cuda" and not bool(config.rotation_allow_cpu_fallback):
        raise RuntimeError("IQN was configured for CUDA but PyTorch cannot access CUDA.")
    return "cpu", None, str(torch.__version__)


def _normalization(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    train_dates: pd.DatetimeIndex,
) -> dict[str, tuple[pd.Series, pd.Series]]:
    output: dict[str, tuple[pd.Series, pd.Series]] = {}
    for symbol in symbols:
        sample = frames[symbol].loc[train_dates, ROTATION_FEATURES].replace([np.inf, -np.inf], np.nan)
        mean = sample.mean().fillna(0.0)
        std = sample.std().replace(0, 1.0).fillna(1.0)
        output[symbol] = (mean, std)
    return output


def _feature_cache(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    dates: pd.DatetimeIndex,
    normalization: dict[str, tuple[pd.Series, pd.Series]],
) -> dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]]:
    feature_matrices: list[np.ndarray] = []
    availability_matrices: list[np.ndarray] = []
    for symbol in symbols:
        frame = frames[symbol]
        mean, std = normalization[symbol]
        raw = frame.loc[dates, ROTATION_FEATURES].replace([np.inf, -np.inf], np.nan)
        normalized = ((raw - mean) / std).clip(-8, 8).fillna(0.0)
        feature_matrices.append(normalized.to_numpy(dtype=np.float32))

        feature_ready = raw.notna().all(axis=1).to_numpy(dtype=bool)
        open_values = frame.loc[dates, "open"].to_numpy(dtype=np.float64)
        close_values = frame.loc[dates, "close"].to_numpy(dtype=np.float64)
        tradable = feature_ready & np.isfinite(open_values) & (open_values > 0) & np.isfinite(close_values) & (close_values > 0)
        availability_matrices.append(tradable.astype(np.float32)[:, None])

    combined_features = np.concatenate(feature_matrices, axis=1)
    availability = np.concatenate(availability_matrices, axis=1)
    return {
        pd.Timestamp(timestamp): (combined_features[index], availability[index])
        for index, timestamp in enumerate(dates)
    }


def _state_from_cache(
    cached: tuple[np.ndarray, np.ndarray],
    asset_count: int,
    current_position: int,
    holding_days: int,
) -> np.ndarray:
    base_features, availability = cached
    position = np.zeros(asset_count + 1, dtype=np.float32)
    position[int(current_position)] = 1.0
    holding = np.asarray([min(float(holding_days), 60.0) / 60.0], dtype=np.float32)
    return np.concatenate([base_features, position, holding, availability]).astype(np.float32, copy=False)


def _state_availability(state: np.ndarray, asset_count: int) -> np.ndarray:
    return np.asarray(state[-asset_count:], dtype=np.float32)


def _iqn_action_snapshot(
    network: _IQNNetwork,
    state: np.ndarray,
    asset_count: int,
    evaluation_quantiles: int,
) -> tuple[int, float, np.ndarray]:
    torch = network.torch
    with torch.no_grad():
        states = torch.as_tensor(state[None, :], dtype=torch.float32, device=network.device)
        taus = (
            (torch.arange(evaluation_quantiles, device=network.device, dtype=torch.float32) + 0.5)
            / evaluation_quantiles
        ).view(1, -1)
        values = network.quantiles(states, taus)[0].mean(dim=0)
        available = _state_availability(state, asset_count)
        mask = torch.as_tensor(
            np.concatenate([np.ones(1, dtype=np.float32), available]) > 0.5,
            dtype=torch.bool,
            device=network.device,
        )
        values = values.masked_fill(~mask, float("-inf"))
        action = int(torch.argmax(values).item())
        q_values = values.detach().cpu().numpy().astype(float)
        return action, float(q_values[action]), q_values


def _iqn_validation_score(
    network: _IQNNetwork,
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    dates: pd.DatetimeIndex,
    feature_cache: dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]],
    config: Any,
    settings: dict[str, Any],
) -> float:
    if len(dates) < 2:
        return float("-inf")
    position = 0
    holding = 0
    wealth = 1.0
    peak = 1.0
    utility = 0.0
    asset_count = len(symbols)
    for index in range(len(dates) - 1):
        now = pd.Timestamp(dates[index])
        nxt = pd.Timestamp(dates[index + 1])
        state = _state_from_cache(feature_cache[now], asset_count, position, holding)
        action, _, _ = _iqn_action_snapshot(
            network,
            state,
            asset_count,
            int(settings["evaluation_quantiles"]),
        )
        if position > 0 and holding < int(config.rotation_min_holding_days) and action != position:
            action = position
        log_return = _training_transition_log_return(frames, symbols, now, nxt, position, action, config)
        reward, wealth, peak = _risk_adjusted_reward(log_return, wealth, peak, config)
        utility += reward
        if action == position:
            holding = holding + 1 if action > 0 else 0
        else:
            position = action
            holding = 1 if action > 0 else 0
    return float(utility)


def _train_iqn(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    train_dates: pd.DatetimeIndex,
    calibration_dates: pd.DatetimeIndex,
    normalization: dict[str, tuple[pd.Series, pd.Series]],
    config: Any,
    settings: dict[str, Any],
    device: str,
    seed: int,
    *,
    progress_callback: Callable[[float], None] | None = None,
) -> tuple[_IQNNetwork, dict[str, Any]]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("IQN research requires PyTorch. Install requirements.txt.") from exc

    rng = np.random.default_rng(int(seed))
    cache_dates = train_dates.union(calibration_dates)
    cache = _feature_cache(frames, symbols, cache_dates, normalization)
    asset_count = len(symbols)
    sample_state = _state_from_cache(cache[pd.Timestamp(train_dates[0])], asset_count, 0, 0)
    online = _IQNNetwork(
        len(sample_state),
        asset_count + 1,
        int(settings["hidden_dim"]),
        int(settings["cosine_embedding_dim"]),
        device,
        seed,
    )
    target = _IQNNetwork(
        len(sample_state),
        asset_count + 1,
        int(settings["hidden_dim"]),
        int(settings["cosine_embedding_dim"]),
        device,
        seed + 1,
    )
    target.model.load_state_dict(online.model.state_dict())
    optimizer = torch.optim.Adam(online.model.parameters(), lr=float(settings["learning_rate"]))
    replay = _ReplayBuffer(int(settings["replay_size"]), len(sample_state))
    accumulator = _NStepAccumulator(int(settings["n_step"]), float(settings["gamma"]))

    total_steps = int(settings["training_steps"])
    episode_length = min(int(settings["episode_days"]), max(20, len(train_dates) - 2))
    max_start = max(1, len(train_dates) - episode_length - 1)
    start_index = int(rng.integers(0, max_start))
    date_index = start_index
    episode_end = min(len(train_dates) - 1, start_index + episode_length)
    position = 0
    holding = 0
    wealth = 1.0
    peak = 1.0

    best_score = float("-inf")
    best_state = deepcopy(online.model.state_dict())
    best_step = 0
    no_improvement = 0
    steps_used = 0
    stopped_early = False
    eval_every = max(250, int(settings["eval_every_steps"]))
    minimum_steps = int(settings["minimum_training_steps"])
    patience = int(settings["early_stopping_patience"])
    progress_interval = max(1, total_steps // 20)

    if progress_callback is not None:
        progress_callback(0.0)

    for step in range(total_steps):
        steps_used = step + 1
        now = pd.Timestamp(train_dates[date_index])
        nxt = pd.Timestamp(train_dates[date_index + 1])
        state = _state_from_cache(cache[now], asset_count, position, holding)
        availability = _state_availability(state, asset_count)
        allowed = [0, *[idx + 1 for idx, flag in enumerate(availability) if flag > 0.5]]

        fraction = step / max(1, total_steps - 1)
        epsilon = float(settings["epsilon_start"]) + fraction * (
            float(settings["epsilon_end"]) - float(settings["epsilon_start"])
        )
        if rng.random() < epsilon:
            action = int(rng.choice(allowed))
        else:
            action, _, _ = _iqn_action_snapshot(
                online,
                state,
                asset_count,
                int(settings["evaluation_quantiles"]),
            )
        if position > 0 and holding < int(config.rotation_min_holding_days) and action != position:
            action = position

        log_return = _training_transition_log_return(frames, symbols, now, nxt, position, action, config)
        reward, wealth, peak = _risk_adjusted_reward(log_return, wealth, peak, config)
        next_holding = holding + 1 if action == position and action > 0 else (1 if action > 0 else 0)
        next_state = _state_from_cache(cache[nxt], asset_count, action, next_holding)
        done = date_index + 1 >= episode_end
        for transition in accumulator.append(state, action, reward, next_state, done):
            replay.add(*transition)
        position = action
        holding = next_holding

        if replay.size >= int(settings["learning_starts"]):
            states, actions, rewards, next_states, dones, discounts = replay.sample(int(settings["batch_size"]), rng)
            states_t = torch.as_tensor(states, dtype=torch.float32, device=online.device)
            actions_t = torch.as_tensor(actions, dtype=torch.long, device=online.device)
            rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=online.device)
            next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=online.device)
            dones_t = torch.as_tensor(dones, dtype=torch.float32, device=online.device)
            discounts_t = torch.as_tensor(discounts, dtype=torch.float32, device=online.device)
            batch = states_t.shape[0]

            taus = torch.rand(
                (batch, int(settings["quantile_samples"])),
                device=online.device,
            )
            current_all = online.quantiles(states_t, taus)
            batch_index = torch.arange(batch, device=online.device)
            current = current_all[batch_index[:, None], torch.arange(current_all.shape[1], device=online.device)[None, :], actions_t[:, None]]

            with torch.no_grad():
                action_taus = torch.rand(
                    (batch, int(settings["action_quantile_samples"])),
                    device=online.device,
                )
                next_online_values = online.quantiles(next_states_t, action_taus).mean(dim=1)
                next_availability = next_states_t[:, -asset_count:] > 0.5
                action_mask = torch.cat(
                    [torch.ones((batch, 1), dtype=torch.bool, device=online.device), next_availability],
                    dim=1,
                )
                next_online_values = next_online_values.masked_fill(~action_mask, float("-inf"))
                next_actions = torch.argmax(next_online_values, dim=1)

                target_taus = torch.rand(
                    (batch, int(settings["target_quantile_samples"])),
                    device=online.device,
                )
                next_target_all = target.quantiles(next_states_t, target_taus)
                next_target = next_target_all[
                    batch_index[:, None],
                    torch.arange(next_target_all.shape[1], device=online.device)[None, :],
                    next_actions[:, None],
                ]
                target_quantiles = rewards_t[:, None] + (
                    discounts_t[:, None] * (1.0 - dones_t[:, None]) * next_target
                )

            td = target_quantiles[:, None, :] - current[:, :, None]
            abs_td = td.abs()
            kappa = float(settings["huber_kappa"])
            huber = torch.where(
                abs_td <= kappa,
                0.5 * td.pow(2),
                kappa * (abs_td - 0.5 * kappa),
            )
            quantile_weight = torch.abs(taus[:, :, None] - (td.detach() < 0).float())
            loss = (quantile_weight * huber / max(kappa, 1e-12)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online.model.parameters(), float(settings["gradient_clip_norm"]))
            optimizer.step()

        if (step + 1) % int(settings["target_update_steps"]) == 0:
            target.model.load_state_dict(online.model.state_dict())

        if (step + 1) % eval_every == 0:
            score = _iqn_validation_score(
                online,
                frames,
                symbols,
                calibration_dates,
                cache,
                config,
                settings,
            )
            if step + 1 >= minimum_steps or not bool(settings["early_stopping_enabled"]):
                if score > best_score + 1e-12:
                    best_score = score
                    best_state = deepcopy(online.model.state_dict())
                    best_step = step + 1
                    no_improvement = 0
                else:
                    no_improvement += 1
                if bool(settings["early_stopping_enabled"]) and no_improvement >= patience:
                    stopped_early = True
                    break

        if progress_callback is not None and ((step + 1) % progress_interval == 0 or step + 1 == total_steps):
            progress_callback((step + 1) / max(1, total_steps))

        date_index += 1
        if done:
            position = 0
            holding = 0
            wealth = 1.0
            peak = 1.0
            accumulator.pending.clear()
            start_index = int(rng.integers(0, max_start))
            date_index = start_index
            episode_end = min(len(train_dates) - 1, start_index + episode_length)

    if best_step == 0:
        best_state = deepcopy(online.model.state_dict())
        best_step = steps_used
        best_score = _iqn_validation_score(
            online,
            frames,
            symbols,
            calibration_dates,
            cache,
            config,
            settings,
        )
    online.model.load_state_dict(best_state)
    online.model.eval()
    if progress_callback is not None:
        progress_callback(1.0)
    return online, {
        "seed": int(seed),
        "requested_steps": total_steps,
        "steps_used": steps_used,
        "best_step": best_step,
        "best_validation_score": float(best_score),
        "stopped_early": bool(stopped_early),
        "n_step": int(settings["n_step"]),
    }


def _iqn_policy(
    network: _IQNNetwork,
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    normalization: dict[str, tuple[pd.Series, pd.Series]],
    config: Any,
    settings: dict[str, Any],
    *,
    fold_id: int,
    decision_diagnostics: dict[pd.Timestamp, dict[str, Any]],
) -> Callable[[pd.Timestamp, int, int], tuple[int, float]]:
    cache: dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]] = {}
    asset_count = len(symbols)

    def cached_state(timestamp: pd.Timestamp, current_position: int, holding_days: int) -> np.ndarray:
        key = pd.Timestamp(timestamp)
        if key not in cache:
            cache.update(_feature_cache(frames, symbols, pd.DatetimeIndex([key]), normalization))
        return _state_from_cache(cache[key], asset_count, current_position, holding_days)

    def policy(timestamp: pd.Timestamp, current_position: int, holding_days: int) -> tuple[int, float]:
        state = cached_state(timestamp, current_position, holding_days)
        raw_action, raw_score, values = _iqn_action_snapshot(
            network,
            state,
            asset_count,
            int(settings["evaluation_quantiles"]),
        )
        action = raw_action
        min_hold_guard = False
        if current_position > 0 and holding_days < int(config.rotation_min_holding_days) and action != current_position:
            action = current_position
            min_hold_guard = True
        labels = ["CASH", *symbols]
        finite_values = [(labels[index], float(value)) for index, value in enumerate(values) if np.isfinite(value)]
        ranked = sorted(finite_values[1:], key=lambda item: (-item[1], item[0]))
        current_value = float(values[current_position]) if np.isfinite(values[current_position]) else None
        final_value = float(values[action]) if np.isfinite(values[action]) else raw_score
        second = ranked[1] if len(ranked) > 1 else (None, None)
        best = ranked[0] if ranked else (None, None)
        decision_diagnostics[pd.Timestamp(timestamp)] = {
            "decision_diagnostics_schema_version": 2,
            "decision_fold_id": fold_id,
            "current_asset": labels[current_position],
            "current_score": current_value,
            "holding_days_at_decision": int(holding_days),
            "raw_best_asset": labels[raw_action],
            "raw_best_score": float(raw_score),
            "best_asset": best[0],
            "best_score": best[1],
            "second_asset": second[0],
            "second_score": second[1],
            "best_vs_second_gap": (
                float(best[1] - second[1]) if best[1] is not None and second[1] is not None else None
            ),
            "cash_score": float(values[0]) if np.isfinite(values[0]) else None,
            "final_action_asset": labels[action],
            "final_action_score": final_value,
            "decision_reason": "MIN_HOLD_GUARD" if min_hold_guard else "IQN_POLICY",
            "decision_is_rotation": bool(current_position > 0 and action > 0 and action != current_position),
            "decision_is_entry": bool(current_position == 0 and action > 0),
            "decision_is_exit_to_cash": bool(current_position > 0 and action == 0),
            "min_hold_guard_applied": min_hold_guard,
            "switch_margin_guard_applied": False,
            "cash_threshold_guard_applied": False,
            "minimum_expected_edge_guard_applied": False,
            "day_trade_constraint_applied": False,
            "q_current_position": current_value,
            "q_raw_best": float(raw_score),
            "q_final_action": final_value,
            "q_delta_final_vs_current": (
                float(final_value - current_value) if current_value is not None else None
            ),
            "q_gap_best_vs_second": (
                float(best[1] - second[1]) if best[1] is not None and second[1] is not None else None
            ),
            "raw_action_asset": labels[raw_action],
        }
        for rank in range(3):
            asset, score = ranked[rank] if rank < len(ranked) else (None, None)
            decision_diagnostics[pd.Timestamp(timestamp)][f"top_{rank + 1}_asset"] = asset
            decision_diagnostics[pd.Timestamp(timestamp)][f"top_{rank + 1}_score"] = score
        return int(action), float(final_value)

    return policy


def _run_iqn(
    bars_by_symbol: dict[str, pd.DataFrame],
    config: Any,
    fee_calculator: Callable,
    slippage: Callable,
    *,
    progress_callback: Callable[[float, str, int], None] | None,
    trade_callback: Callable[[dict[str, Any]], None] | None,
    progress_detail_callback: Callable[[dict[str, Any]], None] | None,
    technical_log_callback: Callable[[str], None] | None,
) -> list[RotationRunResult]:
    (
        frames,
        common_dates,
        symbols,
        folds,
        all_decision_dates,
        decision_to_fold,
        decision_metadata,
    ) = _build_execution_context(bars_by_symbol, config)
    settings = _iqn_settings(config)
    device, gpu_name, torch_version = _iqn_compute_device(config)
    repetitions = int(config.rotation_xgb_repetitions)
    seed_step = int(config.rotation_seed_step)
    total_folds = len(folds)

    def technical(message: str) -> None:
        if technical_log_callback is not None:
            technical_log_callback(message)

    def report(fraction: float, stage: str, completed: int) -> None:
        if progress_callback is not None:
            progress_callback(20.0 + 72.0 * max(0.0, min(1.0, fraction)), stage, completed)

    def detail(**values: Any) -> None:
        if progress_detail_callback is not None:
            progress_detail_callback(values)

    if progress_callback is not None:
        label = f"CUDA — {gpu_name}" if device == "cuda" and gpu_name else device.upper()
        progress_callback(18.0, f"Prepared {len(symbols)} assets and {len(folds)} folds — IQN={label}", 0)

    results: list[RotationRunResult] = []
    for repetition in range(repetitions):
        run_index = repetition + 1
        seed = int(config.random_state) + repetition * seed_step
        rep_config = config.model_copy(update={"random_state": seed})
        policies: dict[int, Callable] = {}
        diagnostics: dict[pd.Timestamp, dict[str, Any]] = {}
        training_details: list[dict[str, Any]] = []
        run_base = repetition / repetitions
        run_span = 1.0 / repetitions
        fold_span = (run_span * 0.92) / max(1, total_folds)

        technical(
            f"model=iqn event=run_start run={run_index}/{repetitions} seed={seed} "
            f"device={device} steps={int(settings['training_steps'])}"
        )
        for fold_position, fold in enumerate(folds, start=1):
            fold_base = run_base + (fold_position - 1) * fold_span
            fold_id = int(fold["fold_id"])
            train_dates = common_dates[: int(fold["train_end_index"])]
            calibration_dates = common_dates[
                int(fold["calibration_start_index"]): int(fold["calibration_end_index"])
            ]
            normalization = _normalization(frames, symbols, train_dates)

            def training_progress(local_fraction: float) -> None:
                report(
                    fold_base + fold_span * local_fraction,
                    f"Run {run_index}/{repetitions} — fold {fold_position}/{total_folds} — IQN training {local_fraction * 100:.0f}%",
                    repetition,
                )
                detail(
                    run_index=run_index,
                    run_count=repetitions,
                    fold_index=fold_position,
                    fold_count=total_folds,
                    phase="IQN training",
                    trained_models=int(round(local_fraction * int(settings["training_steps"]))),
                    total_models=int(settings["training_steps"]),
                    device=device.upper(),
                )

            network, train_info = _train_iqn(
                frames,
                symbols,
                train_dates,
                calibration_dates,
                normalization,
                rep_config,
                settings,
                device,
                seed,
                progress_callback=training_progress,
            )
            train_info.update({"fold_id": fold_id, "fold_position": fold_position})
            training_details.append(train_info)
            policies[fold_id] = _iqn_policy(
                network,
                frames,
                symbols,
                normalization,
                rep_config,
                settings,
                fold_id=fold_id,
                decision_diagnostics=diagnostics,
            )
            report(
                fold_base + fold_span,
                f"Run {run_index}/{repetitions} — fold {fold_position}/{total_folds} completed",
                repetition,
            )

        report(
            run_base + run_span * 0.96,
            f"Run {run_index}/{repetitions} — simulating out-of-sample portfolio",
            repetition,
        )
        scheduled = _scheduled_policy(policies, decision_to_fold)
        wrapped_trade_callback = None
        if trade_callback is not None:
            def wrapped_trade_callback(trade: dict[str, Any], *, _seed=seed, _run=run_index) -> None:
                payload = dict(trade)
                payload.update(
                    {
                        "model_family": "iqn",
                        "random_seed": _seed,
                        "repetition_index": _run,
                        "model": "IQN" + (f" · seed {_seed}" if repetitions > 1 else ""),
                    }
                )
                trade_callback(payload)

        result = _simulate_exact(
            "iqn",
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
            model_label="IQN",
            method_line=(
                "- IQN learns an implicit distribution of risk-adjusted long-horizon returns "
                "for CASH and each asset action from the same walk-forward market states."
            ),
        )
        backend = "iqn" if repetitions <= 1 else f"iqn_seed_{seed}"
        result.backend = backend
        result.metrics.update(
            {
                "backend": backend,
                "model_family": "iqn",
                "strategy_label": "IQN" + (f" · seed {seed}" if repetitions > 1 else ""),
                "random_seed": seed,
                "repetition_index": run_index,
                "repetition_count": repetitions,
                "walk_forward_fold_count": len(folds),
                "walk_forward_folds": _fold_performance(result.predictions, folds, float(rep_config.initial_capital)),
                "effective_compute_device": device,
                "gpu_name": gpu_name,
                "framework_version": torch_version,
                "torch_version": torch_version,
                "iqn_training_details": training_details,
                "iqn_settings_revision": _research_settings(config).get("settings_revision"),
                "iqn_profile_id": _research_settings(config).get("profile_id"),
                "decision_diagnostics_schema_version": 8 if absolute_utility_cash_gate_enabled(rep_config) else (7 if opportunity_cash_gate_enabled(rep_config) else (5 if selective_opportunity_enabled(rep_config) else (3 if _risk_off_enabled(rep_config) else 2))),
                "decision_diagnostics_rows": len(diagnostics),
            }
        )
        results.append(result)
        report(run_index / repetitions, f"IQN run {run_index}/{repetitions} completed", run_index)
        technical(
            f"model=iqn event=run_complete run={run_index}/{repetitions} seed={seed} "
            f"ending_capital={result.metrics.get('strategy_ending_capital')}"
        )

    results.sort(key=lambda item: int(item.metrics.get("repetition_index", 1)))
    return results


def run_research_challenger(
    model_family: str,
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
    if model_family == "lightgbm_utility":
        return _run_lightgbm(
            bars_by_symbol,
            config,
            fee_calculator,
            slippage,
            progress_callback=progress_callback,
            trade_callback=trade_callback,
            progress_detail_callback=progress_detail_callback,
            technical_log_callback=technical_log_callback,
        )
    if model_family == "iqn":
        if allocation_execution_enabled(config):
            raise ValueError("Portfolio Allocation / Compound Risk Overlay v3.12.0 supports Ranking Utility models (XGBoost/LightGBM); IQN allocation is not enabled in this release.")
        return _run_iqn(
            bars_by_symbol,
            config,
            fee_calculator,
            slippage,
            progress_callback=progress_callback,
            trade_callback=trade_callback,
            progress_detail_callback=progress_detail_callback,
            technical_log_callback=technical_log_callback,
        )
    raise ValueError(f"Unsupported research challenger: {model_family}")
