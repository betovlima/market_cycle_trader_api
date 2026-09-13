from __future__ import annotations

import math
from typing import Any, Iterable

import pandas as pd

import research_contextual_marginal_signature_v1114 as diagnostics

base = diagnostics.base
storage = diagnostics.storage
progress = diagnostics.progress
sound = diagnostics.sound

SCRIPT_VERSION = "contextual-marginal-signature-v1.0.14.4"
EXPERIMENT_NAME = "contextual_marginal_signature_direct_effect_prevalence"
TOLERANCE = 1e-12

_ORIGINAL_WINDOW_REQUEST = base._window_request
_ORIGINAL_DIAGNOSTIC_RUN = diagnostics._run_with_diagnostics
_BASELINE_SCHEDULE_BY_KEY: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
_CURRENT_CONTEXT: dict[str, Any] | None = None


def _context_key(decision_session: Any, reference_assets: list[str]) -> tuple[str, tuple[str, ...]]:
    decision = pd.Timestamp(decision_session)
    if decision.tzinfo is not None:
        decision = decision.tz_convert("UTC").tz_localize(None)
    return (
        decision.normalize().date().isoformat(),
        tuple(str(item).strip().upper() for item in reference_assets),
    )


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _extract_margin_schedule(results: list[Any]) -> dict[str, Any]:
    repetitions: list[dict[str, Any]] = []
    relevant: dict[tuple[int, int], dict[str, float]] = {}

    for result_index, result in enumerate(results, start=1):
        metrics = dict(getattr(result, "metrics", {}) or {})
        repetition_index = int(metrics.get("repetition_index") or result_index)
        fold_count = int(metrics.get("walk_forward_fold_count") or 0)
        if fold_count <= 0:
            raise RuntimeError(
                "Fold-aware frozen-margin protocol requires walk_forward_fold_count in replay metrics."
            )

        fold_rows = list(metrics.get("walk_forward_folds") or [])
        observed_by_fold: dict[int, dict[str, float]] = {}
        for row in fold_rows:
            if not isinstance(row, dict):
                continue
            fold_id_raw = row.get("fold_id")
            calibrated = _finite(row.get("calibrated_candidate_margin"))
            effective = _finite(row.get("effective_switch_margin"))
            if fold_id_raw is None or calibrated is None or effective is None:
                continue
            fold_id = int(fold_id_raw)
            if fold_id < 1 or fold_id > fold_count:
                raise RuntimeError(
                    f"Invalid fold id {fold_id} for walk_forward_fold_count={fold_count}."
                )
            previous = observed_by_fold.get(fold_id)
            current = {
                "calibrated_switch_margin": float(calibrated),
                "effective_switch_margin": float(effective),
            }
            if previous is not None and (
                abs(previous["calibrated_switch_margin"] - current["calibrated_switch_margin"]) > TOLERANCE
                or abs(previous["effective_switch_margin"] - current["effective_switch_margin"]) > TOLERANCE
            ):
                raise RuntimeError(
                    f"Conflicting baseline switch margins for repetition={repetition_index}, fold={fold_id}."
                )
            observed_by_fold[fold_id] = current
            relevant[(repetition_index, fold_id)] = current

        if not observed_by_fold:
            raise RuntimeError(
                "Fold-aware frozen-margin protocol found no fold-specific margin diagnostics."
            )

        repetitions.append(
            {
                "repetition_index": repetition_index,
                "fold_count": fold_count,
                "relevant_folds": observed_by_fold,
            }
        )

    repetitions.sort(key=lambda item: int(item["repetition_index"]))
    return {"repetitions": repetitions, "relevant": relevant}


class _FoldScheduledMarginCandidates(list[float]):
    """Present one forced candidate only on OOS folds used by the analysis window.

    research_challengers calibrates switch margin once per walk-forward fold by
    iterating rotation_switch_margin_candidates.  This research-only list keeps
    normal calibration for folds outside the analysis window and yields the
    baseline-selected singleton for every fold that contributes OOS decisions.
    """

    def __init__(self, original: Iterable[float], schedule: dict[str, Any]) -> None:
        original_values = [float(value) for value in original]
        if not original_values:
            raise RuntimeError("rotation_switch_margin_candidates cannot be empty.")
        super().__init__(original_values)
        self._original = tuple(original_values)
        self._steps: list[dict[str, Any]] = []
        self._cursor = 0

        for repetition in list(schedule.get("repetitions") or []):
            repetition_index = int(repetition["repetition_index"])
            fold_count = int(repetition["fold_count"])
            relevant_folds = dict(repetition.get("relevant_folds") or {})
            for fold_id in range(1, fold_count + 1):
                observed = relevant_folds.get(fold_id)
                self._steps.append(
                    {
                        "repetition_index": repetition_index,
                        "fold_id": fold_id,
                        "forced_margin": (
                            float(observed["calibrated_switch_margin"])
                            if isinstance(observed, dict)
                            else None
                        ),
                    }
                )

    def __iter__(self):
        if self._cursor >= len(self._steps):
            raise RuntimeError(
                "Fold-aware frozen-margin schedule was exhausted before replay calibration completed."
            )
        step = self._steps[self._cursor]
        self._cursor += 1
        forced = step.get("forced_margin")
        values = (float(forced),) if forced is not None else self._original
        return iter(values)

    @property
    def consumed_steps(self) -> int:
        return int(self._cursor)

    @property
    def expected_steps(self) -> int:
        return int(len(self._steps))

    @property
    def steps(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._steps]


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

    request = _ORIGINAL_WINDOW_REQUEST(
        db=db,
        config=config,
        strategy_id=strategy_id,
        assets=assets,
        reference_assets=reference_assets,
        candidate_assets=candidate_assets,
        decision_session=decision_session,
        horizon_end=horizon_end,
    )

    schedule = None
    if candidate is not None:
        schedule = _BASELINE_SCHEDULE_BY_KEY.get(key)
        if schedule is None:
            raise RuntimeError(
                "Fold-aware frozen-margin challenger requested before baseline schedule was captured: "
                f"decision={key[0]}, reference_assets={len(key[1])}, candidate={candidate}."
            )
        diagnostics._ORIGINAL_LOG(
            f"    [freeze-folds] candidate={candidate}; baseline_relevant_folds="
            f"{len(schedule.get('relevant') or {})}"
        )

    _CURRENT_CONTEXT = {
        "key": key,
        "candidate": candidate,
        "is_baseline": candidate is None,
        "schedule": schedule,
    }
    return request


def _compare_relevant_schedules(expected: dict[str, Any], observed: dict[str, Any], candidate: str) -> None:
    expected_map = dict(expected.get("relevant") or {})
    observed_map = dict(observed.get("relevant") or {})
    missing = sorted(set(expected_map).difference(observed_map))
    if missing:
        raise RuntimeError(
            f"Frozen-margin challenger {candidate} is missing baseline OOS folds: {missing}."
        )

    for key, baseline in expected_map.items():
        challenger = observed_map[key]
        if (
            abs(float(challenger["calibrated_switch_margin"]) - float(baseline["calibrated_switch_margin"])) > TOLERANCE
            or abs(float(challenger["effective_switch_margin"]) - float(baseline["effective_switch_margin"])) > TOLERANCE
        ):
            raise RuntimeError(
                "Fold-aware frozen switch-margin invariant failed for "
                f"candidate={candidate}, repetition={key[0]}, fold={key[1]}: "
                f"expected calibrated/effective="
                f"{baseline['calibrated_switch_margin']}/{baseline['effective_switch_margin']}, "
                f"observed={challenger['calibrated_switch_margin']}/{challenger['effective_switch_margin']}."
            )


def _run_with_frozen_margin(frames: dict[str, Any], request: Any):
    context = dict(_CURRENT_CONTEXT or {})
    scheduled_candidates: _FoldScheduledMarginCandidates | None = None

    if not bool(context.get("is_baseline")):
        schedule = context.get("schedule")
        if not isinstance(schedule, dict):
            raise RuntimeError("Fold-aware frozen-margin challenger has no baseline schedule.")
        scheduled_candidates = _FoldScheduledMarginCandidates(
            list(request.rotation_switch_margin_candidates),
            schedule,
        )
        request = request.model_copy(
            update={"rotation_switch_margin_candidates": scheduled_candidates}
        )
        forced_steps = [step for step in scheduled_candidates.steps if step.get("forced_margin") is not None]
        rendered = ", ".join(
            f"r{step['repetition_index']}/f{step['fold_id']}={float(step['forced_margin']):.6f}"
            for step in forced_steps
        )
        diagnostics._ORIGINAL_LOG(f"    [freeze-fold-plan] {rendered}")

    metrics, sessions, captured = _ORIGINAL_DIAGNOSTIC_RUN(frames, request)

    if scheduled_candidates is not None and (
        scheduled_candidates.consumed_steps != scheduled_candidates.expected_steps
    ):
        raise RuntimeError(
            "Fold-aware frozen-margin calibration count mismatch: "
            f"consumed={scheduled_candidates.consumed_steps}, "
            f"expected={scheduled_candidates.expected_steps}."
        )

    observed = _extract_margin_schedule(captured)
    key = context.get("key")
    if not isinstance(key, tuple):
        raise RuntimeError("Fold-aware frozen-margin replay completed without a request context key.")

    if bool(context.get("is_baseline")):
        _BASELINE_SCHEDULE_BY_KEY[key] = observed
        rendered = ", ".join(
            f"r{rep}/f{fold}={values['calibrated_switch_margin']:.6f}"
            for (rep, fold), values in sorted(dict(observed.get("relevant") or {}).items())
        )
        diagnostics._ORIGINAL_LOG(
            f"    [freeze-folds] baseline captured; decision={key[0]}; {rendered}"
        )
    else:
        candidate = str(context.get("candidate") or "UNKNOWN")
        expected = _BASELINE_SCHEDULE_BY_KEY[key]
        _compare_relevant_schedules(expected, observed, candidate)
        diagnostics._ORIGINAL_LOG(
            f"    [freeze-folds-ok] candidate={candidate}; "
            f"matched_relevant_folds={len(expected.get('relevant') or {})}"
        )

    return metrics, sessions, captured


def main() -> int:
    # v1.0.14.4 corrects the frozen-margin protocol for analysis windows that
    # cross walk-forward fold boundaries.  The scientific hypothesis is
    # unchanged: only the baseline-selected switch margin is frozen, now at the
    # fold granularity at which the production engine actually calibrates it.
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
