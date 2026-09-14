from __future__ import annotations

from typing import Any

import pandas as pd

import research_contextual_signature_campaign as _impl
from research_contextual_signature_runtime import (
    PairedReplayMemoryCache,
    compare_replay_outputs,
    process_rss_mb,
)

for _export_name in dir(_impl):
    if not _export_name.startswith("_"):
        globals().setdefault(_export_name, getattr(_impl, _export_name))

SCRIPT_VERSION = "contextual-marginal-signature-v1.0.17.5"
_impl.SCRIPT_VERSION = SCRIPT_VERSION
_impl.prev.SCRIPT_VERSION = SCRIPT_VERSION


def _pair_key(kwargs: dict[str, Any]) -> tuple[Any, ...]:
    decision = pd.Timestamp(kwargs["decision"])
    if decision.tzinfo is not None:
        decision = decision.tz_convert("UTC").tz_localize(None)
    return (
        decision.normalize().date().isoformat(),
        tuple(
            str(item).strip().upper()
            for item in kwargs.get("reference_assets") or []
        ),
        str(kwargs.get("candidate") or "").strip().upper(),
    )


def _render_candidate_ranking(rows: list[dict[str, Any]]) -> str:
    frame = pd.DataFrame(rows)
    if frame.empty or "action_advantage_log" not in frame.columns:
        return "n/a"

    ordered = frame.sort_values(
        "action_advantage_log",
        ascending=False,
    )
    top = [
        f"{row.candidate} "
        f"{float(row.action_advantage_log):+.4f}"
        for row in ordered.head(3).itertuples(index=False)
    ]
    worst = ordered.iloc[-1]
    spread = (
        float(ordered.iloc[0]["action_advantage_log"])
        - float(worst["action_advantage_log"])
    )
    return (
        f"top3={', '.join(top)} | "
        f"worst={worst['candidate']} "
        f"{float(worst['action_advantage_log']):+.4f} | "
        f"spread={spread:+.4f}"
    )


def _validation_text(result: dict[str, Any]) -> str:
    return (
        f"capital relative error={result['capital_relative_error']:.3e} | "
        f"sessions={'YES' if result['sessions_identical'] else 'NO'} | "
        f"predictions={'YES' if result['predictions_identical'] else 'NO'} | "
        f"trades={'YES' if result['trades_identical'] else 'NO'}"
    )


def main() -> int:
    from market_cycle_trader_api.engine import (
        capital_rotation,
        research_challengers,
    )

    protocol = _impl.prev
    live = _impl.live

    cache = PairedReplayMemoryCache(
        research_challengers,
        capital_rotation,
    )
    cache.install()
    cache_disabled_reason: str | None = None

    original_run_replay = _impl._run_replay
    original_load_market_frames = protocol._load_market_frames
    original_log_context_summary = protocol._log_context_summary
    original_log_partial = protocol._log_partial
    original_partial_analysis = live.partial_analysis

    def cached_run_replay(**kwargs: Any):
        nonlocal cache_disabled_reason

        cache.ensure_state(kwargs["decision"])
        candidate = str(
            kwargs.get("candidate") or ""
        ).strip().upper()
        forced = bool(kwargs.get("forced"))
        config = kwargs.get("config")
        mode = str(
            getattr(
                config,
                "strategy_mode",
                "",
            )
        )
        eligible = (
            bool(candidate)
            and mode == "COMPOUND_ROTATION_SWING_XGBOOST"
        )
        if not eligible:
            return original_run_replay(**kwargs)

        if cache_disabled_reason is not None:
            with cache.suspended_cache():
                return original_run_replay(**kwargs)

        key = _pair_key(kwargs)
        if not forced:
            if cache.validation is None:
                live.console_log(
                    "    [cache-validation] first eligible policy "
                    "replay | uncached reference"
                )
                with cache.suspended_cache():
                    reference = original_run_replay(**kwargs)

                cache.begin_pair(key)
                try:
                    accelerated = original_run_replay(**kwargs)
                except BaseException:
                    cache.abort_pair()
                    raise

                validation = compare_replay_outputs(
                    reference,
                    accelerated,
                    tolerance=1e-12,
                )
                cache.record_validation(validation)
                live.console_log(
                    "    [cache-validation] "
                    + _validation_text(validation)
                )
                if not bool(validation["passed"]):
                    cache.abort_pair()
                    live.console_log(
                        "    [cache-validation] accelerated replay diverged; "
                        "running one second uncached replay to separate a cache "
                        "defect from underlying replay nondeterminism"
                    )
                    with cache.suspended_cache():
                        uncached_repeat = original_run_replay(**kwargs)
                    uncached_validation = compare_replay_outputs(
                        reference,
                        uncached_repeat,
                        tolerance=1e-12,
                    )
                    live.console_log(
                        "    [cache-validation-uncached] "
                        + _validation_text(uncached_validation)
                    )

                    if bool(uncached_validation["passed"]):
                        cache_disabled_reason = (
                            "accelerated replay differed while two uncached "
                            "replays were equivalent"
                        )
                        cache.record_validation(
                            {
                                **validation,
                                "cache_disabled": True,
                                "cache_disabled_reason": cache_disabled_reason,
                                "uncached_repeat_validation": uncached_validation,
                            }
                        )
                        live.console_log(
                            "    [cache-validation] SAFE FALLBACK | RAM acceleration "
                            "disabled for this process; campaign continues with the "
                            "trusted uncached protocol"
                        )
                        return reference

                    deterministic = bool(
                        getattr(config, "deterministic_execution", False)
                    )
                    xgb_n_jobs = getattr(config, "xgb_n_jobs", None)
                    numeric_thread_limit = getattr(
                        config,
                        "numeric_thread_limit",
                        None,
                    )
                    cache.record_validation(
                        {
                            **validation,
                            "cache_disabled": True,
                            "underlying_replay_deterministic": False,
                            "uncached_repeat_validation": uncached_validation,
                            "deterministic_execution": deterministic,
                            "xgb_n_jobs": xgb_n_jobs,
                            "numeric_thread_limit": numeric_thread_limit,
                        }
                    )
                    raise RuntimeError(
                        "Underlying uncached replay is not exactly reproducible; "
                        "RAM cache is not the only source of divergence. "
                        f"uncached check: {_validation_text(uncached_validation)} | "
                        f"deterministic_execution={deterministic}, "
                        f"xgb_n_jobs={xgb_n_jobs}, "
                        f"numeric_thread_limit={numeric_thread_limit}."
                    )

                live.console_log(
                    "    [cache-validation] PASS | invariant RAM "
                    "cache is numerically equivalent to uncached replay"
                )
                return accelerated

            cache.begin_pair(key)
            try:
                return original_run_replay(**kwargs)
            except BaseException:
                cache.abort_pair()
                raise

        if not cache.active_for(key):
            cache.abort_pair()
            live.console_log(
                "    [cache] safety fallback: paired cache key "
                "unavailable; forced arm will run uncached"
            )
            with cache.suspended_cache():
                return original_run_replay(**kwargs)

        try:
            result = original_run_replay(**kwargs)
        except BaseException:
            cache.abort_pair()
            raise

        stats = cache.finish_pair()
        live.console_log(
            f"    [cache] pair reuse | "
            f"panel={stats['context_hits']} | "
            f"LightGBM fits={stats['fit_hits']} reused/"
            f"{stats['fit_misses']} built | "
            f"feature frames={stats['feature_hits']} hits/"
            f"{stats['feature_misses']} builds | "
            f"utility predictions={stats['utility_hits']} hits/"
            f"{stats['utility_misses']} builds | "
            f"estimated compute avoided≈"
            f"{stats['avoided_seconds']:.1f}s | "
            f"pair elapsed={stats['pair_elapsed_seconds']:.1f}s"
        )
        return result

    def load_market_frames_with_memory(
        config: Any,
        market_data: Any,
    ):
        frames, provenance = original_load_market_frames(
            config,
            market_data,
        )
        cache.set_market_frames(frames)
        rss = process_rss_mb()
        rss_text = (
            "n/a"
            if rss is None
            else f"{rss:.1f} MB"
        )
        live.console_log(
            f"[memory] market snapshot resident in RAM | "
            f"symbols={len(frames)} | "
            f"DataFrames≈{cache.market_frames_mb:.1f} MB | "
            f"process RSS={rss_text}"
        )
        return frames, provenance

    def log_context_summary_with_ranking(
        rows: list[dict[str, Any]],
        decision_text: str,
        universe_name: str,
    ) -> None:
        original_log_context_summary(
            rows,
            decision_text,
            universe_name,
        )
        already_selected = sum(
            bool(
                row.get(
                    "policy_first_selected_candidate"
                )
            )
            for row in rows
        )
        live.console_log(
            f"[ranking] {live.display_date(decision_text)} | "
            f"{universe_name} | "
            f"{_render_candidate_ranking(rows)} | "
            f"policy already chose candidate="
            f"{already_selected}/{len(rows)}"
        )

    def partial_analysis_with_runtime(
        frame: pd.DataFrame,
        universe_count: int,
    ) -> dict[str, Any]:
        result = dict(
            original_partial_analysis(
                frame,
                universe_count,
            )
        )
        runtime = cache.summary()
        runtime["cache_disabled_reason"] = cache_disabled_reason
        result["runtime_memory_cache"] = runtime
        return result

    def log_partial_with_runtime(
        rows: list[dict[str, Any]],
        state_index: int,
        elapsed: float,
    ) -> dict[str, Any]:
        result = original_log_partial(
            rows,
            state_index,
            elapsed,
        )
        stats = cache.summary()
        rss = stats.get("process_rss_mb")
        rss_text = (
            "n/a"
            if rss is None
            else f"{float(rss):.1f} MB"
        )
        validation = (
            stats.get("equivalence_validation") or {}
        )
        if cache_disabled_reason is not None:
            validation_text = "SAFE_FALLBACK_UNCACHED"
        else:
            validation_text = (
                "PASS"
                if validation.get("passed")
                else "pending"
            )
        live.console_log(
            f"[cache] cumulative | "
            f"pairs={stats['completed_pairs']} | "
            f"LightGBM fits reused="
            f"{stats['model_fit_hits']} | "
            f"panel reuse="
            f"{stats['execution_context_hits']} | "
            f"feature hits="
            f"{stats['feature_frame_hits']} | "
            f"utility hits="
            f"{stats['utility_prediction_hits']} | "
            f"estimated compute avoided≈"
            f"{stats['estimated_compute_seconds_avoided']:.0f}s | "
            f"validation={validation_text} | "
            f"RSS={rss_text}"
        )

        frame = pd.DataFrame(rows)
        if not frame.empty:
            means = (
                frame.groupby("candidate")[
                    "action_advantage_log"
                ]
                .mean()
                .sort_values(ascending=False)
            )
            leaders = ", ".join(
                f"{candidate} {float(value):+.4f}"
                for candidate, value
                in means.head(3).items()
            )
            live.console_log(
                "[candidate] cumulative mean Y leaders | "
                f"{leaders}"
            )
        return result

    _impl._run_replay = cached_run_replay
    protocol._load_market_frames = (
        load_market_frames_with_memory
    )
    protocol._log_context_summary = (
        log_context_summary_with_ranking
    )
    protocol._log_partial = log_partial_with_runtime
    live.partial_analysis = partial_analysis_with_runtime

    live.console_log(
        f"[runtime] {SCRIPT_VERSION} | "
        "guarded RAM acceleration with safe uncached fallback enabled"
    )
    live.console_log(
        "[memory] reusable=invariant market feature frames + "
        "pair execution context + fitted LightGBM models + "
        "state-level raw utility predictions | "
        "never cached=portfolio state/trades/positions/"
        "holding days"
    )
    live.console_log(
        "[memory] first eligible candidate validates uncached-vs-cached; "
        "on mismatch a second uncached replay distinguishes cache divergence "
        "from underlying nondeterminism"
    )

    try:
        return int(_impl.main())
    finally:
        _impl._run_replay = original_run_replay
        protocol._load_market_frames = (
            original_load_market_frames
        )
        protocol._log_context_summary = (
            original_log_context_summary
        )
        protocol._log_partial = original_log_partial
        live.partial_analysis = original_partial_analysis
        cache.uninstall()


if __name__ == "__main__":
    raise SystemExit(main())
