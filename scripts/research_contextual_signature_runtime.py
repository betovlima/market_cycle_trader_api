from __future__ import annotations

from contextlib import contextmanager
import hashlib
import time
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd


class PairedReplayMemoryCache:
    """RAM cache for replay work that is invariant to portfolio state.

    Safe cache layers:
    - market-derived rotation frames: process scoped;
    - execution context, fitted models and raw utility predictions: candidate-pair scoped.

    Positions, holding days, trades, diagnostics and simulator outputs are never
    cached because they depend on the counterfactual intervention.
    """

    def __init__(self, research_challengers: Any, capital_rotation: Any | None = None) -> None:
        self.module = research_challengers
        self.capital_rotation = capital_rotation
        self.original_build = research_challengers._build_execution_context
        self.original_fit = research_challengers._lightgbm_fit_models
        self.original_research_utilities = getattr(research_challengers, "_model_utilities", None)
        self.original_build_rotation_frame = (
            getattr(capital_rotation, "build_rotation_frame", None) if capital_rotation is not None else None
        )
        self.original_capital_utilities = (
            getattr(capital_rotation, "_model_utilities", None) if capital_rotation is not None else None
        )

        self.installed = False
        self.suspended = False
        self.active_key: tuple[Any, ...] | None = None
        self.state_key: str | None = None
        self.execution_context: Any | None = None
        self.models: dict[tuple[Any, ...], dict[str, Any]] = {}
        self.fit_seconds: dict[tuple[Any, ...], float] = {}
        self.feature_frames: dict[tuple[Any, ...], pd.DataFrame] = {}
        self.utility_predictions: dict[tuple[Any, ...], float] = {}
        self.utility_seconds: dict[tuple[Any, ...], float] = {}
        self.pair_started: float | None = None
        self.pair_start: dict[str, float] = {}
        self.market_identity: str | None = None
        self.market_frames_mb = 0.0

        self.completed_pairs = 0
        self.context_hits = 0
        self.context_misses = 0
        self.fit_hits = 0
        self.fit_misses = 0
        self.feature_hits = 0
        self.feature_misses = 0
        self.utility_hits = 0
        self.utility_misses = 0
        self.avoided_seconds = 0.0
        self.actual_fit_seconds = 0.0
        self.context_build_seconds = 0.0
        self.feature_build_seconds = 0.0
        self.utility_compute_seconds = 0.0
        self.validation: dict[str, Any] | None = None

    def install(self) -> None:
        if self.installed:
            return
        self.module._build_execution_context = self.cached_build
        self.module._lightgbm_fit_models = self.cached_fit
        if self.capital_rotation is not None and self.original_build_rotation_frame is not None:
            self.capital_rotation.build_rotation_frame = self.cached_build_rotation_frame
        if self.capital_rotation is not None and self.original_capital_utilities is not None:
            self.capital_rotation._model_utilities = self.cached_model_utilities
        if self.original_research_utilities is not None:
            self.module._model_utilities = self.cached_model_utilities
        self.installed = True

    def uninstall(self) -> None:
        if not self.installed:
            return
        self.clear_pair()
        self.module._build_execution_context = self.original_build
        self.module._lightgbm_fit_models = self.original_fit
        if self.capital_rotation is not None and self.original_build_rotation_frame is not None:
            self.capital_rotation.build_rotation_frame = self.original_build_rotation_frame
        if self.capital_rotation is not None and self.original_capital_utilities is not None:
            self.capital_rotation._model_utilities = self.original_capital_utilities
        if self.original_research_utilities is not None:
            self.module._model_utilities = self.original_research_utilities
        self.installed = False

    @contextmanager
    def suspended_cache(self) -> Iterator[None]:
        previous = self.suspended
        self.suspended = True
        try:
            yield
        finally:
            self.suspended = previous

    def set_market_frames(self, frames: dict[str, pd.DataFrame]) -> None:
        self.market_frames_mb = frames_memory_mb(frames)
        parts: list[str] = []
        for symbol in sorted(frames):
            frame = frames[symbol]
            first = str(frame.index[0]) if len(frame) else ""
            last = str(frame.index[-1]) if len(frame) else ""
            parts.append(f"{symbol}:{id(frame)}:{len(frame)}:{first}:{last}")
        self.market_identity = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
        self.feature_frames.clear()
        self.utility_predictions.clear()
        self.utility_seconds.clear()

    def ensure_state(self, value: Any) -> None:
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is not None:
            stamp = stamp.tz_convert("UTC").tz_localize(None)
        self.state_key = stamp.normalize().date().isoformat()

    def begin_pair(self, key: tuple[Any, ...]) -> None:
        self.clear_pair()
        self.active_key = tuple(key)
        self.pair_started = time.monotonic()
        self.pair_start = self._snapshot()

    def active_for(self, key: tuple[Any, ...]) -> bool:
        return self.active_key == tuple(key)

    def clear_pair(self) -> None:
        self.active_key = None
        self.execution_context = None
        self.models.clear()
        self.fit_seconds.clear()
        self.utility_predictions.clear()
        self.utility_seconds.clear()
        self.pair_started = None
        self.pair_start = {}

    def abort_pair(self) -> None:
        self.clear_pair()

    def _snapshot(self) -> dict[str, float]:
        return {
            "context_hits": float(self.context_hits),
            "context_misses": float(self.context_misses),
            "fit_hits": float(self.fit_hits),
            "fit_misses": float(self.fit_misses),
            "feature_hits": float(self.feature_hits),
            "feature_misses": float(self.feature_misses),
            "utility_hits": float(self.utility_hits),
            "utility_misses": float(self.utility_misses),
            "avoided_seconds": float(self.avoided_seconds),
        }

    @staticmethod
    def _feature_config_key(config: Any) -> tuple[Any, ...]:
        return (
            tuple(int(x) for x in getattr(config, "rotation_target_horizons", []) or []),
            tuple(float(x) for x in getattr(config, "rotation_target_horizon_weights", []) or []),
            float(getattr(config, "slippage_bps", 0.0)),
            float(getattr(config, "commission_rate", 0.0)),
            float(getattr(config, "rotation_downside_penalty", 0.0)),
            float(getattr(config, "rotation_drawdown_penalty", 0.0)),
            float(getattr(config, "rotation_movement_capture_weight", 0.0)),
            float(getattr(config, "rotation_trend_persistence_weight", 0.0)),
        )

    def cached_build_rotation_frame(self, bars: pd.DataFrame, config: Any) -> pd.DataFrame:
        if self.suspended or self.original_build_rotation_frame is None or self.market_identity is None:
            return self.original_build_rotation_frame(bars, config)  # type: ignore[misc]
        key = (self.market_identity, id(bars), self._feature_config_key(config))
        cached = self.feature_frames.get(key)
        if cached is not None:
            self.feature_hits += 1
            return cached
        started = time.perf_counter()
        result = self.original_build_rotation_frame(bars, config)
        self.feature_misses += 1
        self.feature_build_seconds += time.perf_counter() - started
        self.feature_frames[key] = result
        return result

    @staticmethod
    def fit_key(symbols: list[str], train_dates: Any, config: Any, phase: str, target_column: str) -> tuple[Any, ...]:
        dates = pd.DatetimeIndex(train_dates)
        first = int(pd.Timestamp(dates[0]).value) if len(dates) else None
        last = int(pd.Timestamp(dates[-1]).value) if len(dates) else None
        return (
            str(phase), str(target_column), tuple(str(x) for x in symbols), int(len(dates)),
            first, last, int(getattr(config, "random_state", 0)),
        )

    def cached_build(self, bars_by_symbol: dict[str, pd.DataFrame], config: Any):
        if self.suspended or self.active_key is None:
            return self.original_build(bars_by_symbol, config)
        if self.execution_context is not None:
            self.context_hits += 1
            return self.execution_context
        started = time.perf_counter()
        result = self.original_build(bars_by_symbol, config)
        self.context_misses += 1
        self.context_build_seconds += time.perf_counter() - started
        self.execution_context = result
        return result

    def cached_fit(
        self,
        frames: dict[str, pd.DataFrame],
        symbols: list[str],
        train_dates: Any,
        config: Any,
        *,
        phase: str,
        progress_callback: Callable[[int, int, str], None] | None = None,
        technical_log_callback: Callable[[str], None] | None = None,
        target_column: str = "forward_risk_adjusted_utility",
    ) -> dict[str, Any]:
        if self.suspended or self.active_key is None:
            return self.original_fit(
                frames, symbols, train_dates, config, phase=phase,
                progress_callback=progress_callback,
                technical_log_callback=technical_log_callback,
                target_column=target_column,
            )
        key = self.fit_key(symbols, train_dates, config, phase, target_column)
        cached = self.models.get(key)
        if cached is not None:
            self.fit_hits += 1
            self.avoided_seconds += float(self.fit_seconds.get(key, 0.0))
            if progress_callback is not None:
                for position in range(1, len(symbols) + 1):
                    progress_callback(position, len(symbols), "cpu-cache")
            if technical_log_callback is not None:
                technical_log_callback(f"model=lightgbm phase={phase} event=fit_memory_cache_hit")
            return dict(cached)
        started = time.perf_counter()
        fitted = self.original_fit(
            frames, symbols, train_dates, config, phase=phase,
            progress_callback=progress_callback,
            technical_log_callback=technical_log_callback,
            target_column=target_column,
        )
        elapsed = time.perf_counter() - started
        self.fit_misses += 1
        self.actual_fit_seconds += elapsed
        self.models[key] = dict(fitted)
        self.fit_seconds[key] = float(elapsed)
        return fitted

    def _prediction_key(self, model: Any, symbol: str, timestamp: pd.Timestamp) -> tuple[Any, ...]:
        return (self.active_key, id(model), str(symbol), int(pd.Timestamp(timestamp).value))

    def cached_model_utilities(
        self,
        models: dict[str, Any],
        frames: dict[str, pd.DataFrame],
        symbols: list[str],
        timestamp: pd.Timestamp,
        config: Any,
    ) -> np.ndarray:
        original = self.original_capital_utilities or self.original_research_utilities
        if original is None:
            raise RuntimeError("RAM cache could not resolve the original model-utility function.")
        if self.suspended or self.active_key is None:
            return original(models, frames, symbols, timestamp, config)

        values = [0.0]
        for symbol in symbols:
            model = models.get(symbol)
            if model is None:
                values.append(float("-inf"))
                continue
            key = self._prediction_key(model, symbol, pd.Timestamp(timestamp))
            if key in self.utility_predictions:
                self.utility_hits += 1
                self.avoided_seconds += float(self.utility_seconds.get(key, 0.0))
                values.append(float(self.utility_predictions[key]))
                continue
            started = time.perf_counter()
            single = original({symbol: model}, {symbol: frames[symbol]}, [symbol], timestamp, config)
            elapsed = time.perf_counter() - started
            value = float(single[1]) if len(single) > 1 else float("-inf")
            self.utility_misses += 1
            self.utility_compute_seconds += elapsed
            self.utility_predictions[key] = value
            self.utility_seconds[key] = float(elapsed)
            values.append(value)
        return np.asarray(values, dtype=np.float64)

    def finish_pair(self) -> dict[str, Any]:
        start = dict(self.pair_start)
        stats = {
            "context_hits": int(self.context_hits - int(start.get("context_hits", 0))),
            "context_misses": int(self.context_misses - int(start.get("context_misses", 0))),
            "fit_hits": int(self.fit_hits - int(start.get("fit_hits", 0))),
            "fit_misses": int(self.fit_misses - int(start.get("fit_misses", 0))),
            "feature_hits": int(self.feature_hits - int(start.get("feature_hits", 0))),
            "feature_misses": int(self.feature_misses - int(start.get("feature_misses", 0))),
            "utility_hits": int(self.utility_hits - int(start.get("utility_hits", 0))),
            "utility_misses": int(self.utility_misses - int(start.get("utility_misses", 0))),
            "avoided_seconds": float(self.avoided_seconds - start.get("avoided_seconds", 0.0)),
            "pair_elapsed_seconds": float(time.monotonic() - self.pair_started) if self.pair_started is not None else None,
        }
        self.completed_pairs += 1
        self.clear_pair()
        return stats

    def record_validation(self, result: dict[str, Any]) -> None:
        self.validation = dict(result)

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "scope": "features_process_pair_context_models_predictions",
            "completed_pairs": int(self.completed_pairs),
            "execution_context_hits": int(self.context_hits),
            "execution_context_misses": int(self.context_misses),
            "model_fit_hits": int(self.fit_hits),
            "model_fit_misses": int(self.fit_misses),
            "feature_frame_hits": int(self.feature_hits),
            "feature_frame_misses": int(self.feature_misses),
            "cached_feature_frames": int(len(self.feature_frames)),
            "utility_prediction_hits": int(self.utility_hits),
            "utility_prediction_misses": int(self.utility_misses),
            "cached_pair_predictions": int(len(self.utility_predictions)),
            "estimated_compute_seconds_avoided": float(self.avoided_seconds),
            "actual_model_fit_seconds": float(self.actual_fit_seconds),
            "execution_context_build_seconds": float(self.context_build_seconds),
            "feature_build_seconds": float(self.feature_build_seconds),
            "utility_compute_seconds": float(self.utility_compute_seconds),
            "market_frames_estimated_mb": float(self.market_frames_mb),
            "process_rss_mb": process_rss_mb(),
            "equivalence_validation": dict(self.validation) if self.validation is not None else None,
        }


def compare_replay_outputs(
    reference: tuple[dict[str, Any], Any, list[Any]],
    accelerated: tuple[dict[str, Any], Any, list[Any]],
    *,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    reference_metrics, reference_sessions, reference_captured = reference
    accelerated_metrics, accelerated_sessions, accelerated_captured = accelerated

    ref_capital = float(reference_metrics.get("ending_capital"))
    acc_capital = float(accelerated_metrics.get("ending_capital"))
    capital_relative_error = abs(acc_capital - ref_capital) / max(abs(ref_capital), 1e-12)
    sessions_identical = pd.DatetimeIndex(reference_sessions).equals(pd.DatetimeIndex(accelerated_sessions))
    predictions_identical = len(reference_captured) == len(accelerated_captured)
    trades_identical = predictions_identical

    if predictions_identical:
        for ref_result, acc_result in zip(reference_captured, accelerated_captured):
            try:
                pd.testing.assert_frame_equal(
                    getattr(ref_result, "predictions"), getattr(acc_result, "predictions"),
                    check_dtype=False, check_exact=False, rtol=tolerance, atol=tolerance,
                )
            except AssertionError:
                predictions_identical = False
                break
    if trades_identical:
        for ref_result, acc_result in zip(reference_captured, accelerated_captured):
            try:
                pd.testing.assert_frame_equal(
                    getattr(ref_result, "trades"), getattr(acc_result, "trades"),
                    check_dtype=False, check_exact=False, rtol=tolerance, atol=tolerance,
                )
            except AssertionError:
                trades_identical = False
                break

    passed = bool(
        capital_relative_error <= tolerance
        and sessions_identical
        and predictions_identical
        and trades_identical
    )
    return {
        "passed": passed,
        "tolerance": float(tolerance),
        "capital_relative_error": float(capital_relative_error),
        "sessions_identical": bool(sessions_identical),
        "predictions_identical": bool(predictions_identical),
        "trades_identical": bool(trades_identical),
    }


def process_rss_mb() -> float | None:
    try:
        import psutil  # type: ignore
        return float(psutil.Process().memory_info().rss / (1024.0 * 1024.0))
    except Exception:
        return None


def frames_memory_mb(frames: dict[str, pd.DataFrame]) -> float:
    total = 0
    for frame in frames.values():
        try:
            total += int(frame.memory_usage(index=True, deep=True).sum())
        except Exception:
            pass
    return float(total / (1024.0 * 1024.0))
