from __future__ import annotations

from typing import Any, Callable

from .inputs import candidate_request, load_frozen_bars, source_run, winner_override
from .preprocessing import prepare_training_context


def prepare_campaign_context(
    db: Any,
    strategy: dict[str, Any],
    model_snapshot: dict[str, Any],
    *,
    fold_count: int | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    run = source_run(db, strategy)
    request, _ = candidate_request(run, model_snapshot, {}, fold_count=fold_count)
    bars_by_symbol = load_frozen_bars(
        request,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
    )
    winner = winner_override(db, run)
    training = prepare_training_context(
        bars_by_symbol,
        request,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
    )
    return {
        "source_run": run,
        "bars_by_symbol": bars_by_symbol,
        "winner_override": winner,
        "training": training,
    }
