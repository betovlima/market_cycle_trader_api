from __future__ import annotations

# Stable user-facing entrypoint. Internal research implementations may evolve,
# but the command, output folder and ZIP names remain unchanged.
from typing import Any

import pandas as pd

import research_contextual_marginal_signature_v1172 as _impl
from research_contextual_marginal_signature_v1172 import *  # noqa: F401,F403
from research_contextual_signature_runtime import (
    PairedReplayMemoryCache,
    compare_replay_outputs,
    process_rss_mb,
)

SCRIPT_VERSION = "contextual-marginal-signature-v1.0.17.4"
_impl.SCRIPT_VERSION = SCRIPT_VERSION
_impl.prev.SCRIPT_VERSION = SCRIPT_VERSION


def _pair_key(kwargs: dict[str, Any]) -> tuple[Any, ...]:
    decision = pd.Timestamp(kwargs["decision"])
    if decision.tzinfo is not None:
        decision = decision.tz_convert("UTC").tz_localize(None)
    return (
        decision.normalize().date().isoformat(),
        tuple(str(item).strip().upper() for item in kwargs.get("reference_assets") or []),
        str(kwargs.get("candidate") or "").strip().upper(),
    )


def _render_candidate_ranking(rows: list[dict[str, Any]]) -> str:
    frame = pd.DataFrame(rows)
    if frame.empty or "action_advantage_log" not in frame.columns:
        return "n/a"
    ordered = frame.sort_values("action_advantage_log", ascending=False)
    top = [
        f"{row.candidate} {float(row.action_advantage_log):+.4f}"
        for row in ordered.head(3).itertuples(index=False)
    ]
    worst = ordered.iloc[-1]
    spread = float(ordered.iloc[0]["action_advantage_log"]) - float(worst["action_advantage_log"])
    return (
        f"top3={', '.join(top)} | worst={worst['candidate']} {float(worst['action_advantage_log']):+.4f} | "
        f"spread={spread:+.4f}"
    )


def main() -> int:
    from market_cycle_trader_api.engine import capital_rotation, research_challengers

    cache = PairedReplayMemoryCache(research_challengers, capital_rotation)
    cache.install()

    original_run_replay = _impl._run_replay
    original_load_market_frames = _impl.prev._load_market_frames
    original_log_context_summary = _impl.prev._log_context_summary
    original_log_partial = _impl.prev._log_partial
    original_partial_analysis = _impl.live.partial_analysis

    def cached_run_replay(**kwargs: Any):
        cache.ensure_state(kwargs["decision"])
        candidate = str(kwargs.get("candidate") or "").strip().upper()
        forced = bool(kwargs.get("forced"))
        mode = str(getattr(kwargs.get("config"), "strategy_mode", ""))
        eligible = bool(candidate) and mode == "COMPOUND_ROTATION_SWING_XGBOOST"
        if not eligible:
            return original_run_replay(**kwargs)

        key = _pair_key(kwargs)
        if not forced:
            # The first candidate policy arm is executed once without any cache
            # and once with RAM acceleration.  The long campaign only continues
            # if capital, sessions, predictions and trades are equivalent.
            if cache.validation is None:
                _impl.live.console_log(
                    "    [cache-validation] first eligible policy replay | uncached reference"
                )
                with cache.suspended_cache():
                    reference = original_run_replay(**kwargs)
                cache.begin_pair(key)
                try:
                    accelerated = original_run_replay(**kwargs)
                except BaseException:
                    cache.abort_pair()
                    raise
                validation = compare_replay_outputs(reference, accelerated, tolerance=1e-12)
                cache.record_validation(validation)
                _impl.live.console_log(
                    "    [cache-validation] "
                    f"capital relative error={validation['capital_relative_error']:.3e} | "
                    f"sessions={'YES' if validation['sessions_identical'] else 'NO'} | "
                    f"predictions={'YES' if validation['predictions_identical'] else 'NO'} | "
                    f"trades={'YES' if validation['trades_identical'] else 'NO'}"
                )
                if not bool(validation["passed"]):
                    cache.abort_pair()
                    raise RuntimeError(
                        "RAM acceleration equivalence validation failed; campaign stopped before trusting cached results."
                    )
                _impl.live.console_log(
                    "    [cache-validation] PASS | invariant RAM cache is numerically equivalent to uncached replay"
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
            _impl.live.console_log(
                "    [cache] safety fallback: paired cache key unavailable; forced arm will run uncached"
            )
            with cache.suspended_cache():
                return original_run_replay(**kwargs)

        try:
            result = original_run_replay(**kwargs)
        except BaseException:
            cache.abort_pair()
            raise
        stats = cache.finish_pair()
        _impl.live.console_log(
            f"    [cache] pair reuse | panel={stats['context_hits']} | "
            f"LightGBM fits={stats['fit_hits']} reused/{stats['fit_misses']} built | "
            f"feature frames={stats['feature_hits']} hits/{stats['feature_misses']} builds | "
            f"utility predictions={stats['utility_hits']} hits/{stats['utility_misses']} builds | "
            f"estimated compute avoided≈{stats['avoided_seconds']:.1f}s | "
            f"pair elapsed={stats['pair_elapsed_seconds']:.1f}s"
        )
        return result

    def load_market_frames_with_memory(config: Any, market_data: Any):
        frames, provenance = original_load_market_frames(config, market_data)
        cache.set_market_frames(frames)
        rss = process_rss_mb()
        rss_text = "n/a" if rss is None else f"{rss:.1f} MB"
        _impl.live.console_log(
            f"[memory] market snapshot resident in RAM | symbols={len(frames)} | "
            f"DataFrames≈{cache.market_frames_mb:.1f} MB | process RSS={rss_text}"
        )
        return frames, provenance

    def log_context_summary_with_ranking(rows: list[dict[str, Any]], decision_text: str, universe_name: str) -> None:
        original_log_context_summary(rows, decision_text, universe_name)
        already_selected = sum(bool(row.get("policy_first_selected_candidate")) for row in rows)
        _impl.live.console_log(
            f"[ranking] {_impl.live.display_date(decision_text)} | {universe_name} | "
            f"{_render_candidate_ranking(rows)} | policy already chose candidate={already_selected}/{len(rows)}"
        )

    def partial_analysis_with_runtime(frame: pd.DataFrame, universe_count: int) -> dict[str, Any]:
        result = dict(original_partial_analysis(frame, universe_count))
        result["runtime_memory_cache"] = cache.summary()
        return result

    def log_partial_with_runtime(rows: list[dict[str, Any]], state_index: int, elapsed: float) -> dict[str, Any]:
        result = original_log_partial(rows, state_index, elapsed)
        stats = cache.summary()
        rss = stats.get("process_rss_mb")
        rss_text = "n/a" if rss is None else f"{float(rss):.1f} MB"
        validation = stats.get("equivalence_validation") or {}
        validation_text = "PASS" if validation.get("passed") else "pending"
        _impl.live.console_log(
            f"[cache] cumulative | pairs={stats['completed_pairs']} | "
            f"LightGBM fits reused={stats['model_fit_hits']} | panel reuse={stats['execution_context_hits']} | "
            f"feature hits={stats['feature_frame_hits']} | utility hits={stats['utility_prediction_hits']} | "
            f"estimated compute avoided≈{stats['estimated_compute_seconds_avoided']:.0f}s | "
            f"validation={validation_text} | RSS={rss_text}"
        )
        frame = pd.DataFrame(rows)
        if not frame.empty:
            means = frame.groupby("candidate")["action_advantage_log"].mean().sort_values(ascending=False)
            leaders = ", ".join(
                f"{candidate} {float(value):+.4f}" for candidate, value in means.head(3).items()
            )
            _impl.live.console_log(f"[candidate] cumulative mean Y leaders | {leaders}")
        return result

    _impl._run_replay = cached_run_replay
    _impl.prev._load_market_frames = load_market_frames_with_memory
    _impl.prev._log_context_summary = log_context_summary_with_ranking
    _impl.prev._log_partial = log_partial_with_runtime
    _impl.live.partial_analysis = partial_analysis_with_runtime

    _impl.live.console_log(f"[runtime] {SCRIPT_VERSION} | guarded RAM acceleration enabled")
    _impl.live.console_log(
        "[memory] reusable=invariant market feature frames + pair execution context + fitted LightGBM models + "
        "state-level raw utility predictions | never cached=portfolio state/trades/positions/holding days"
    )
    _impl.live.console_log(
        "[memory] first eligible candidate performs an automatic uncached-vs-cached equivalence check before continuing"
    )

    try:
        return int(_impl.main())
    finally:
        _impl._run_replay = original_run_replay
        _impl.prev._load_market_frames = original_load_market_frames
        _impl.prev._log_context_summary = original_log_context_summary
        _impl.prev._log_partial = original_log_partial
        _impl.live.partial_analysis = original_partial_analysis
        cache.uninstall()


if __name__ == "__main__":
    raise SystemExit(main())
