from __future__ import annotations

import math
from typing import Any

import pandas as pd

import research_contextual_marginal_signature_v1114 as diagnostics

base = diagnostics.base
storage = diagnostics.storage
progress = diagnostics.progress
sound = diagnostics.sound

SCRIPT_VERSION = "contextual-marginal-signature-v1.0.12"
EXPERIMENT_NAME = "contextual_marginal_signature_frozen_switch_margin"
TOLERANCE = 1e-12

_ORIGINAL_WINDOW_REQUEST = base._window_request
_ORIGINAL_DIAGNOSTIC_RUN = diagnostics._run_with_diagnostics
_BASELINE_MARGIN_BY_KEY: dict[tuple[str, tuple[str, ...]], dict[str, float]] = {}
_CURRENT_CONTEXT: dict[str, Any] | None = None


def _context_key(decision_session: Any, reference_assets: list[str]) -> tuple[str, tuple[str, ...]]:
    decision = pd.Timestamp(decision_session)
    if decision.tzinfo is not None:
        decision = decision.tz_convert("UTC").tz_localize(None)
    return decision.normalize().date().isoformat(), tuple(str(item).strip().upper() for item in reference_assets)


def _extract_margin(results: list[Any]) -> tuple[float, float]:
    calibrated_values: list[float] = []
    effective_values: list[float] = []
    for result in results:
        predictions = getattr(result, "predictions", None)
        if not isinstance(predictions, pd.DataFrame) or predictions.empty:
            continue
        if "calibrated_switch_margin" not in predictions.columns or "effective_switch_margin" not in predictions.columns:
            continue
        calibrated = pd.to_numeric(predictions["calibrated_switch_margin"], errors="coerce").dropna()
        effective = pd.to_numeric(predictions["effective_switch_margin"], errors="coerce").dropna()
        calibrated_values.extend(float(value) for value in calibrated.unique())
        effective_values.extend(float(value) for value in effective.unique())

    calibrated_unique = sorted({round(value, 15) for value in calibrated_values if math.isfinite(value)})
    effective_unique = sorted({round(value, 15) for value in effective_values if math.isfinite(value)})
    if len(calibrated_unique) != 1 or len(effective_unique) != 1:
        raise RuntimeError(
            "Frozen-margin protocol requires exactly one calibrated/effective switch margin per replay; "
            f"calibrated={calibrated_unique}, effective={effective_unique}."
        )
    return float(calibrated_unique[0]), float(effective_unique[0])


def _window_request_frozen(
    *,
    db: Any,
    config: Any,
    strategy_id: str,
    assets: list[str],
    reference_assets: list[str],
    candidate_assets: list[str],
    decision_session: Any,
    horizon_end: Any,
):
    global _CURRENT_CONTEXT
    key = _context_key(decision_session, reference_assets)
    candidate = str(candidate_assets[0]).strip().upper() if candidate_assets else None

    request_config = config
    if candidate is not None:
        baseline = _BASELINE_MARGIN_BY_KEY.get(key)
        if baseline is None:
            raise RuntimeError(
                "Frozen-margin challenger requested before its baseline margin was captured: "
                f"decision={key[0]}, reference_assets={len(key[1])}, candidate={candidate}."
            )
        calibrated_margin = float(baseline["calibrated_switch_margin"])
        request_config = config.model_copy(
            update={"rotation_switch_margin_candidates": [calibrated_margin]}
        )
        diagnostics._ORIGINAL_LOG(
            f"    [freeze] candidate={candidate}; baseline_calibrated_switch_margin={calibrated_margin:.6f}; "
            f"baseline_effective_switch_margin={baseline['effective_switch_margin']:.6f}"
        )

    _CURRENT_CONTEXT = {
        "key": key,
        "candidate": candidate,
        "is_baseline": candidate is None,
    }
    return _ORIGINAL_WINDOW_REQUEST(
        db=db,
        config=request_config,
        strategy_id=strategy_id,
        assets=assets,
        reference_assets=reference_assets,
        candidate_assets=candidate_assets,
        decision_session=decision_session,
        horizon_end=horizon_end,
    )


def _run_with_frozen_margin(frames: dict[str, Any], request: Any):
    context = dict(_CURRENT_CONTEXT or {})
    metrics, sessions, captured = _ORIGINAL_DIAGNOSTIC_RUN(frames, request)
    calibrated, effective = _extract_margin(captured)

    key = context.get("key")
    if not isinstance(key, tuple):
        raise RuntimeError("Frozen-margin replay completed without a request context key.")

    if bool(context.get("is_baseline")):
        _BASELINE_MARGIN_BY_KEY[key] = {
            "calibrated_switch_margin": calibrated,
            "effective_switch_margin": effective,
        }
        diagnostics._ORIGINAL_LOG(
            f"    [freeze] baseline captured; decision={key[0]}; calibrated_switch_margin={calibrated:.6f}; "
            f"effective_switch_margin={effective:.6f}"
        )
    else:
        baseline = _BASELINE_MARGIN_BY_KEY[key]
        expected_calibrated = float(baseline["calibrated_switch_margin"])
        expected_effective = float(baseline["effective_switch_margin"])
        if abs(calibrated - expected_calibrated) > TOLERANCE or abs(effective - expected_effective) > TOLERANCE:
            raise RuntimeError(
                "Frozen switch-margin invariant failed for "
                f"decision={key[0]}, candidate={context.get('candidate')}: "
                f"expected calibrated/effective={expected_calibrated}/{expected_effective}, "
                f"observed={calibrated}/{effective}."
            )
        diagnostics._ORIGINAL_LOG(
            f"    [freeze-ok] candidate={context.get('candidate')}; calibrated_switch_margin={calibrated:.6f}; "
            f"effective_switch_margin={effective:.6f}"
        )

    return metrics, sessions, captured


def main() -> int:
    # One-factor mechanism test: preserve the complete v1.0.11.4 audit protocol,
    # but force each challenger to use the switch-margin selected by its own
    # baseline universe/date. This separates direct candidate/path contribution
    # from the candidate-induced switch-margin recalibration observed in v1.0.11.4.
    diagnostics.SCRIPT_VERSION = SCRIPT_VERSION
    diagnostics.EXPERIMENT_NAME = EXPERIMENT_NAME
    base.SCRIPT_VERSION = SCRIPT_VERSION
    base.EXPERIMENT_NAME = EXPERIMENT_NAME
    storage.SCRIPT_VERSION = SCRIPT_VERSION
    progress.SCRIPT_VERSION = SCRIPT_VERSION
    sound.SCRIPT_VERSION = SCRIPT_VERSION

    base._window_request = _window_request_frozen
    diagnostics._run_with_diagnostics = _run_with_frozen_margin
    return diagnostics.main()


if __name__ == "__main__":
    raise SystemExit(main())
