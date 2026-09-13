from __future__ import annotations

# Stable user-facing entrypoint. Internal research implementations may evolve,
# but the command, output folder and ZIP names remain unchanged.
from typing import Any

import pandas as pd

import research_contextual_marginal_signature_v1172 as _impl
from research_contextual_marginal_signature_v1172 import *  # noqa: F401,F403
from research_contextual_signature_runtime import (
    PairedReplayMemoryCache,
    frames_memory_mb,
    process_rss_mb,
)

SCRIPT_VERSION = "contextual-marginal-signature-v1.0.17.3"
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
    from market_cycle_trader_api.engine import research_challengers

    cache = PairedReplayMemoryCache(research_challengers)
    cache.install()

    original_run_replay = _impl._run_replay
    original_load_market_frames = _impl.prev._load_market_frames
    original_log_context_summary = _impl.prev._log_context_summary
    original_log_partial = _impl.prev._log_partial
    original_partial_analysis = _impl.live.partial_analysis

    def cached_run_replay(**kwargs: Any):
        candidate = str(kwargs.get("candidate") or "").strip().upper()
        forced = bool(kwargs.get("forced"))
        mode = str(getattr(kwargs.get("config"), "strategy_mode", ""))
        eligible = bool(candidate) and mode == "COMPOUND_ROTATION_SWING_XGBOOST"
        if not eligible:
            return original_run_replay(**kwargs)

        key = _pair_key(kwargs)
        if not forced:
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
            return original_run_replay(**kwargs)

        try:
            result = original_run_replay(**kwargs)
        except BaseException:
            cache.abort_pair()
            raise
        stats = cache.finish_pair()
        _impl.live.console_log(
            f"    [cache] pair reuse | panel={stats['context_hits']} hit | "
            f"LightGBM fits={stats['fit_hits']} reused/{stats['fit_misses']} built | "
            f"estimated compute avoided≈{stats['avoided_seconds']:.1f}s | "
            f"pair elapsed={stats['pair_elapsed_seconds']:.1f}s"
        )
        return result

    def load_market_frames_with_memory(config: Any, market_data: Any):
        frames, provenance = original_load_market_frames(config, market_data)
        cache.market_frames_mb = frames_memory_mb(frames)
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
        _impl.live.console_log(
            f"[cache] cumulative | pairs={stats['completed_pairs']} | "
            f"LightGBM fits reused={stats['model_fit_hits']} | panel reuse={stats['execution_context_hits']} | "
            f"estimated compute avoided≈{stats['estimated_compute_seconds_avoided']:.0f}s | RSS={rss_text}"
        )
        frame = pd.DataFrame(rows)
        if not frame.empty:
            means = frame.groupby("candidate")["action_advantage_log"].mean().sort_values(ascending=False)
            leaders = ", ".join(f"{candidate} {float(value):+.4f}" for candidate, value in means.head(3).items())
            _impl.live.console_log(f"[candidate] cumulative mean Y leaders | {leaders}")
        return result

    _impl._run_replay = cached_run_replay
    _impl.prev._load_market_frames = load_market_frames_with_memory
    _impl.prev._log_context_summary = log_context_summary_with_ranking
    _impl.prev._log_partial = log_partial_with_runtime
    _impl.live.partial_analysis = partial_analysis_with_runtime

    _impl.live.console_log(f"[runtime] {SCRIPT_VERSION} | paired RAM acceleration enabled")
    _impl.live.console_log(
        "[memory] cache scope=one candidate pair | reusable=prepared panel + fitted LightGBM models | "
        "simulator state/trades/positions are never cached"
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
