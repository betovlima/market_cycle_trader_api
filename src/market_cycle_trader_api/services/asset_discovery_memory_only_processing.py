from __future__ import annotations

from typing import Any


_INSTALLED = False


def _prefetch_is_active_for_current_campaign(service: Any, db: Any) -> bool:
    campaign = service._campaign(db) or {}
    status = str(campaign.get("status") or "").strip().lower()
    prefetch = campaign.get("candidate_history_prefetch") if isinstance(campaign.get("candidate_history_prefetch"), dict) else {}
    return status in service.ACTIVE_STATUSES and str(prefetch.get("status") or "").strip().lower() == "completed"


def install_asset_discovery_memory_only_processing() -> None:
    """Require prefetched market frames during the active Discovery calculation phase.

    Once candidate histories have been prefetched for the current campaign, the
    automatic scoring/integrity pipeline must not silently reach Alpaca again.
    A cache miss is treated as a pipeline-integrity failure so that network I/O
    cannot be mixed back into candidate-by-candidate calculation.

    Manual actions performed after the campaign (for example an explicit final
    validation after a process restart) keep the existing fallback behavior.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    from . import asset_discovery as service
    from . import asset_discovery_predictive_history_integrity as history

    original_candidate_frame = service._candidate_frame
    original_candidate_history_coverage = service._candidate_history_coverage

    def memory_only_candidate_frame(db: Any, symbol: str, config: Any, end_session: Any, *args: Any, **kwargs: Any) -> Any:
        if _prefetch_is_active_for_current_campaign(service, db):
            key = history._cache_key(symbol, config, end_session)
            frame, _coverage, failure = history._cache_lookup(key)
            if failure:
                raise RuntimeError(failure)
            if frame is None:
                raise RuntimeError(
                    f"candidate_history_cache_miss:{str(symbol or '').strip().upper()}"
                )
            return frame
        return original_candidate_frame(db, symbol, config, end_session, *args, **kwargs)

    def memory_only_candidate_history_coverage(
        db: Any,
        symbol: str,
        config: Any,
        end_session: Any,
        required_sessions: Any,
    ) -> Any:
        if _prefetch_is_active_for_current_campaign(service, db):
            key = history._cache_key(symbol, config, end_session)
            frame, coverage, failure = history._cache_lookup(key)
            if failure:
                raise RuntimeError(failure)
            if frame is None or coverage is None:
                raise RuntimeError(
                    f"candidate_history_cache_miss:{str(symbol or '').strip().upper()}"
                )
            return frame, coverage
        return original_candidate_history_coverage(
            db,
            symbol,
            config,
            end_session,
            required_sessions,
        )

    setattr(memory_only_candidate_frame, "_asset_discovery_memory_only_processing", True)
    setattr(memory_only_candidate_history_coverage, "_asset_discovery_memory_only_processing", True)
    service._candidate_frame = memory_only_candidate_frame
    service._candidate_history_coverage = memory_only_candidate_history_coverage
    _INSTALLED = True
