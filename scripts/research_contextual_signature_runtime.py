from __future__ import annotations

import time
from typing import Any, Callable

import pandas as pd


class PairedReplayMemoryCache:
    """Pair-scoped RAM cache for invariant replay preparation and fitted models."""

    def __init__(self, research_challengers: Any) -> None:
        self.module = research_challengers
        self.original_build = research_challengers._build_execution_context
        self.original_fit = research_challengers._lightgbm_fit_models
        self.installed = False
        self.active_key: tuple[Any, ...] | None = None
        self.execution_context: Any | None = None
        self.models: dict[tuple[Any, ...], dict[str, Any]] = {}
        self.fit_seconds: dict[tuple[Any, ...], float] = {}
        self.pair_started: float | None = None
        self.pair_start: dict[str, float] = {}
        self.market_frames_mb = 0.0
        self.completed_pairs = 0
        self.context_hits = 0
        self.context_misses = 0
        self.fit_hits = 0
        self.fit_misses = 0
        self.avoided_seconds = 0.0
        self.actual_fit_seconds = 0.0
        self.context_build_seconds = 0.0

    def install(self) -> None:
        if self.installed:
            return
        self.module._build_execution_context = self.cached_build
        self.module._lightgbm_fit_models = self.cached_fit
        self.installed = True

    def uninstall(self) -> None:
        if not self.installed:
            return
        self.clear_pair()
        self.module._build_execution_context = self.original_build
        self.module._lightgbm_fit_models = self.original_fit
        self.installed = False

    def begin_pair(self, key: tuple[Any, ...]) -> None:
        self.clear_pair()
        self.active_key = tuple(key)
        self.pair_started = time.monotonic()
        self.pair_start = {
            "context_hits": float(self.context_hits),
            "context_misses": float(self.context_misses),
            "fit_hits": float(self.fit_hits),
            "fit_misses": float(self.fit_misses),
            "avoided_seconds": float(self.avoided_seconds),
            "actual_fit_seconds": float(self.actual_fit_seconds),
            "context_build_seconds": float(self.context_build_seconds),
        }

    def active_for(self, key: tuple[Any, ...]) -> bool:
        return self.active_key == tuple(key)

    def clear_pair(self) -> None:
        self.active_key = None
        self.execution_context = None
        self.models.clear()
        self.fit_seconds.clear()
        self.pair_started = None
        self.pair_start = {}

    def abort_pair(self) -> None:
        self.clear_pair()

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
        if self.active_key is None:
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
        if self.active_key is None:
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

    def finish_pair(self) -> dict[str, Any]:
        start = dict(self.pair_start)
        stats = {
            "context_hits": int(self.context_hits - int(start.get("context_hits", 0))),
            "context_misses": int(self.context_misses - int(start.get("context_misses", 0))),
            "fit_hits": int(self.fit_hits - int(start.get("fit_hits", 0))),
            "fit_misses": int(self.fit_misses - int(start.get("fit_misses", 0))),
            "avoided_seconds": float(self.avoided_seconds - start.get("avoided_seconds", 0.0)),
            "pair_elapsed_seconds": float(time.monotonic() - self.pair_started) if self.pair_started is not None else None,
        }
        self.completed_pairs += 1
        self.clear_pair()
        return stats

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "scope": "one_candidate_policy_forced_pair",
            "completed_pairs": int(self.completed_pairs),
            "execution_context_hits": int(self.context_hits),
            "execution_context_misses": int(self.context_misses),
            "model_fit_hits": int(self.fit_hits),
            "model_fit_misses": int(self.fit_misses),
            "estimated_compute_seconds_avoided": float(self.avoided_seconds),
            "actual_model_fit_seconds": float(self.actual_fit_seconds),
            "execution_context_build_seconds": float(self.context_build_seconds),
            "market_frames_estimated_mb": float(self.market_frames_mb),
            "process_rss_mb": process_rss_mb(),
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
