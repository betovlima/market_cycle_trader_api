"""Safe, non-executable response for retired intraday reanalysis endpoint.

After-close Winner training with one retry is the single permitted source
of a next-session Live signal. Expired windows are never rearmed and cannot
create a fresh intraday signal, even for analysis-only requests.
"""
from __future__ import annotations
from typing import Any


def queue_expired_live_analysis(
    db: Any, *, execution_session: str, actor_email: str,
) -> dict[str, Any]:
    if not actor_email:
        raise RuntimeError("An authenticated operator is required.")
    raise RuntimeError(
        "LiveIntradayReanalysisDisabled: the expired session remains under "
        "review. Only the next completed regular session may produce a "
        "new post-close Winner signal; no intraday retraining or order retry."
    )
