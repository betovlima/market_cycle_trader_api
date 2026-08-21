from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from ...auth.security import SessionIdentity, require_capability
from ...core.runtime import database
from ...services.temporal_decision_context import (
    TemporalDecisionContextError,
    TemporalDecisionContextNotFound,
    get_temporal_decision_context,
)
from ...services.temporal_winner_transition_attribution import get_winner_transition_attribution
from ...services.temporal_winner_transition_risk import (
    WinnerTransitionRiskError,
    get_latest_winner_transition_risk_search,
    run_winner_transition_risk_search,
)
from ...services.temporal_winner_transition_intervention import (
    WinnerTransitionInterventionError,
    get_latest_winner_transition_intervention_search,
    run_winner_transition_intervention_search,
    get_latest_winner_transition_confidence_calibration,
    run_winner_transition_confidence_calibration,
)
from ...schemas.requests import StrategyResearchPipelineControlRequest, TemporalPolicySearchRequest, WinnerTransitionConfidenceCalibrationRequest, WinnerTransitionInterventionSearchRequest, WinnerTransitionRiskSearchRequest, WinnerTransitionStatefulReplayRequest
from ...services.temporal_winner_transition_stateful import (
    WinnerTransitionStatefulReplayError,
    get_latest_winner_transition_stateful_replay,
    materialize_winner_transition_stateful_candidate_a_strategy,
    run_winner_transition_stateful_replay,
)
from ...services.temporal_policy_search import (
    TemporalPolicySearchError,
    create_temporal_policy_search,
    get_latest_temporal_policy_search,
    get_temporal_policy_search,
    run_temporal_policy_caro,
    run_temporal_policy_comparison,
    run_temporal_policy_sampling,
    run_temporal_policy_study,
    run_temporal_policy_validation,
)
from ...services.temporal_intelligence import (
    TemporalIntelligenceConflict,
    TemporalIntelligenceNotFound,
    build_temporal_intelligence_export,
    get_latest_temporal_intelligence_run,
    get_temporal_intelligence_run,
    list_temporal_intelligence_history,
    control_strategy_research_pipeline,
    get_strategy_research_pipeline_state,
    get_strategy_research_pipeline_snapshot,
    materialize_temporal_intelligence_strategy,
    request_strategy_research_pipeline_pause,
    request_strategy_research_pipeline_stop,
    reset_strategy_research_pipeline,
    start_temporal_intelligence,
    stop_temporal_intelligence,
    validate_temporal_research_processing,
)

router = APIRouter(prefix="/api/temporal-intelligence", tags=["temporal-intelligence"])
require_temporal_view = require_capability("temporal_intelligence.view")
require_temporal_start = require_capability("temporal_intelligence.start")
require_temporal_stop = require_capability("temporal_intelligence.stop")
require_temporal_export = require_capability("temporal_intelligence.export")
require_temporal_materialize_strategy = require_capability("temporal_intelligence.materialize_strategy")


def _translate_error(exc: Exception) -> HTTPException:
    if isinstance(exc, TemporalIntelligenceNotFound):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    if isinstance(exc, TemporalIntelligenceConflict):
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc))


@router.get("/latest")
def latest_temporal_intelligence(
    _identity: Annotated[SessionIdentity, Depends(require_temporal_view)],
) -> dict[str, Any] | None:
    return get_latest_temporal_intelligence_run(database())


@router.get("/history")
def temporal_intelligence_history(
    _identity: Annotated[SessionIdentity, Depends(require_temporal_view)],
    limit: int = Query(default=30, ge=1, le=100),
) -> dict[str, Any]:
    items = list_temporal_intelligence_history(database(), limit=limit)
    return {"items": items, "count": len(items)}


@router.get("/{run_id}")
def temporal_intelligence_run(
    run_id: str,
    _identity: Annotated[SessionIdentity, Depends(require_temporal_view)],
) -> dict[str, Any]:
    try:
        return get_temporal_intelligence_run(database(), run_id)
    except TemporalIntelligenceNotFound as exc:
        raise _translate_error(exc) from exc


@router.get("/{run_id}/decision-context")
def temporal_intelligence_decision_context(
    run_id: str,
    _identity: Annotated[SessionIdentity, Depends(require_temporal_view)],
    start_month: str = Query(..., pattern=r"^\d{4}-\d{2}$"),
    end_month: str = Query(..., pattern=r"^\d{4}-\d{2}$"),
) -> dict[str, Any]:
    try:
        return get_temporal_decision_context(
            database(),
            run_id,
            start_month=start_month,
            end_month=end_month,
        )
    except TemporalDecisionContextNotFound as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except TemporalDecisionContextError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc


@router.get("/{run_id}/winner-transition-attribution")
def temporal_intelligence_winner_transition_attribution(
    run_id: str,
    _identity: Annotated[SessionIdentity, Depends(require_temporal_view)],
    start_month: str = Query(..., pattern=r"^\d{4}-\d{2}$"),
    end_month: str = Query(..., pattern=r"^\d{4}-\d{2}$"),
) -> dict[str, Any]:
    try:
        return get_winner_transition_attribution(
            database(),
            run_id,
            start_month=start_month,
            end_month=end_month,
        )
    except TemporalDecisionContextNotFound as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except TemporalDecisionContextError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc


@router.get("/{run_id}/winner-transition-risk-search/latest")
def latest_winner_transition_risk_research(
    run_id: str,
    _identity: Annotated[SessionIdentity, Depends(require_temporal_view)],
    processing_id: str = Query(..., min_length=1),
    start_month: str = Query(..., pattern=r"^\d{4}-\d{2}$"),
    end_month: str = Query(..., pattern=r"^\d{4}-\d{2}$"),
) -> dict[str, Any] | None:
    db = database()
    try:
        validate_temporal_research_processing(db, run_id, processing_id)
    except (TemporalIntelligenceConflict, TemporalIntelligenceNotFound) as exc:
        raise _translate_error(exc) from exc
    return get_latest_winner_transition_risk_search(
        db,
        run_id,
        processing_id=processing_id,
        start_month=start_month,
        end_month=end_month,
    )


@router.post("/{run_id}/winner-transition-risk-search")
def winner_transition_risk_research(
    run_id: str,
    payload: WinnerTransitionRiskSearchRequest,
    _identity: Annotated[SessionIdentity, Depends(require_temporal_start)],
) -> dict[str, Any]:
    db = database()
    try:
        validate_temporal_research_processing(db, run_id, payload.processing_id)
        return run_winner_transition_risk_search(
            db,
            run_id,
            processing_id=payload.processing_id,
            start_month=payload.start_month,
            end_month=payload.end_month,
            seed=payload.seed,
        )
    except (TemporalIntelligenceConflict, TemporalIntelligenceNotFound) as exc:
        raise _translate_error(exc) from exc
    except (WinnerTransitionRiskError, TemporalDecisionContextError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc


@router.get("/{run_id}/winner-transition-intervention-search/latest")
def latest_winner_transition_intervention_research(
    run_id: str,
    _identity: Annotated[SessionIdentity, Depends(require_temporal_view)],
    processing_id: str = Query(..., min_length=1),
    start_month: str = Query(..., pattern=r"^\d{4}-\d{2}$"),
    end_month: str = Query(..., pattern=r"^\d{4}-\d{2}$"),
) -> dict[str, Any] | None:
    db = database()
    try:
        validate_temporal_research_processing(db, run_id, processing_id)
    except (TemporalIntelligenceConflict, TemporalIntelligenceNotFound) as exc:
        raise _translate_error(exc) from exc
    return get_latest_winner_transition_intervention_search(
        db,
        run_id,
        processing_id=processing_id,
        start_month=start_month,
        end_month=end_month,
    )


@router.post("/{run_id}/winner-transition-intervention-search")
def winner_transition_intervention_research(
    run_id: str,
    payload: WinnerTransitionInterventionSearchRequest,
    _identity: Annotated[SessionIdentity, Depends(require_temporal_start)],
) -> dict[str, Any]:
    db = database()
    try:
        validate_temporal_research_processing(db, run_id, payload.processing_id)
        return run_winner_transition_intervention_search(
            db,
            run_id,
            processing_id=payload.processing_id,
            start_month=payload.start_month,
            end_month=payload.end_month,
            seed=payload.seed,
        )
    except (TemporalIntelligenceConflict, TemporalIntelligenceNotFound) as exc:
        raise _translate_error(exc) from exc
    except (WinnerTransitionInterventionError, WinnerTransitionRiskError, TemporalDecisionContextError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc


@router.get("/{run_id}/winner-transition-confidence-calibration/latest")
def latest_winner_transition_confidence_research(
    run_id: str,
    _identity: Annotated[SessionIdentity, Depends(require_temporal_view)],
    processing_id: str = Query(..., min_length=1),
    start_month: str = Query(..., pattern=r"^\d{4}-\d{2}$"),
    end_month: str = Query(..., pattern=r"^\d{4}-\d{2}$"),
) -> dict[str, Any] | None:
    db = database()
    try:
        validate_temporal_research_processing(db, run_id, processing_id)
    except (TemporalIntelligenceConflict, TemporalIntelligenceNotFound) as exc:
        raise _translate_error(exc) from exc
    return get_latest_winner_transition_confidence_calibration(
        db,
        run_id,
        processing_id=processing_id,
        start_month=start_month,
        end_month=end_month,
    )


@router.post("/{run_id}/winner-transition-confidence-calibration")
def winner_transition_confidence_research(
    run_id: str,
    payload: WinnerTransitionConfidenceCalibrationRequest,
    _identity: Annotated[SessionIdentity, Depends(require_temporal_start)],
) -> dict[str, Any]:
    db = database()
    try:
        validate_temporal_research_processing(db, run_id, payload.processing_id)
        return run_winner_transition_confidence_calibration(
            db,
            run_id,
            processing_id=payload.processing_id,
            start_month=payload.start_month,
            end_month=payload.end_month,
        )
    except (TemporalIntelligenceConflict, TemporalIntelligenceNotFound) as exc:
        raise _translate_error(exc) from exc
    except (WinnerTransitionInterventionError, WinnerTransitionRiskError, TemporalDecisionContextError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc


@router.get("/{run_id}/winner-transition-stateful-replay/latest")
def latest_winner_transition_stateful_research(
    run_id: str,
    _identity: Annotated[SessionIdentity, Depends(require_temporal_view)],
    processing_id: str = Query(..., min_length=1),
    start_month: str = Query(..., pattern=r"^\d{4}-\d{2}$"),
    end_month: str = Query(..., pattern=r"^\d{4}-\d{2}$"),
) -> dict[str, Any] | None:
    db = database()
    try:
        validate_temporal_research_processing(db, run_id, processing_id)
    except (TemporalIntelligenceConflict, TemporalIntelligenceNotFound) as exc:
        raise _translate_error(exc) from exc
    return get_latest_winner_transition_stateful_replay(
        db,
        run_id,
        processing_id=processing_id,
        start_month=start_month,
        end_month=end_month,
    )


@router.post("/{run_id}/winner-transition-stateful-replay")
def winner_transition_stateful_research(
    run_id: str,
    payload: WinnerTransitionStatefulReplayRequest,
    _identity: Annotated[SessionIdentity, Depends(require_temporal_start)],
) -> dict[str, Any]:
    db = database()
    try:
        validate_temporal_research_processing(db, run_id, payload.processing_id)
        return run_winner_transition_stateful_replay(
            db,
            run_id,
            processing_id=payload.processing_id,
            start_month=payload.start_month,
            end_month=payload.end_month,
        )
    except (TemporalIntelligenceConflict, TemporalIntelligenceNotFound) as exc:
        raise _translate_error(exc) from exc
    except (WinnerTransitionStatefulReplayError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc


@router.post("/{run_id}/winner-transition-stateful-replay/{replay_id}/candidate-a/strategy", status_code=201)
def create_strategy_from_stateful_candidate_a(
    run_id: str,
    replay_id: str,
    identity: Annotated[SessionIdentity, Depends(require_temporal_materialize_strategy)],
) -> dict[str, Any]:
    try:
        return materialize_winner_transition_stateful_candidate_a_strategy(
            database(),
            run_id,
            replay_id,
            actor_email=identity.email,
        )
    except (TemporalIntelligenceConflict, TemporalIntelligenceNotFound) as exc:
        raise _translate_error(exc) from exc
    except (WinnerTransitionStatefulReplayError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc


@router.get("/{run_id}/policy-search/latest")
def latest_temporal_policy_search(
    run_id: str,
    _identity: Annotated[SessionIdentity, Depends(require_temporal_view)],
    start_month: str | None = Query(default=None, pattern=r"^\d{4}-\d{2}$"),
    end_month: str | None = Query(default=None, pattern=r"^\d{4}-\d{2}$"),
) -> dict[str, Any] | None:
    try:
        return get_latest_temporal_policy_search(
            database(),
            run_id,
            start_month=start_month,
            end_month=end_month,
        )
    except TemporalPolicySearchError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc


@router.get("/{run_id}/policy-search/{search_id}")
def temporal_policy_search(
    run_id: str,
    search_id: str,
    _identity: Annotated[SessionIdentity, Depends(require_temporal_view)],
) -> dict[str, Any]:
    try:
        return get_temporal_policy_search(database(), run_id, search_id)
    except TemporalPolicySearchError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/{run_id}/policy-search", status_code=201)
def prepare_temporal_policy_search(
    run_id: str,
    payload: TemporalPolicySearchRequest,
    _identity: Annotated[SessionIdentity, Depends(require_temporal_start)],
) -> dict[str, Any]:
    try:
        return create_temporal_policy_search(
            database(),
            run_id,
            start_month=payload.start_month,
            end_month=payload.end_month,
            processing_id=payload.processing_id,
            lhs_trials=payload.lhs_trials,
            caro_trials=payload.caro_trials,
            seed=payload.seed,
        )
    except TemporalPolicySearchError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc


@router.post("/{run_id}/policy-search/{search_id}/sampling")
def temporal_policy_search_sampling(
    run_id: str,
    search_id: str,
    _identity: Annotated[SessionIdentity, Depends(require_temporal_start)],
) -> dict[str, Any]:
    try:
        return run_temporal_policy_sampling(database(), run_id, search_id)
    except (TemporalPolicySearchError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc


@router.post("/{run_id}/policy-search/{search_id}/caro")
def temporal_policy_search_caro(
    run_id: str,
    search_id: str,
    _identity: Annotated[SessionIdentity, Depends(require_temporal_start)],
) -> dict[str, Any]:
    try:
        return run_temporal_policy_caro(database(), run_id, search_id)
    except (TemporalPolicySearchError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc


@router.post("/{run_id}/policy-search/{search_id}/validation")
def temporal_policy_search_validation(
    run_id: str,
    search_id: str,
    _identity: Annotated[SessionIdentity, Depends(require_temporal_start)],
) -> dict[str, Any]:
    try:
        return run_temporal_policy_validation(database(), run_id, search_id)
    except (TemporalPolicySearchError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc


@router.post("/{run_id}/policy-search/{search_id}/comparison")
def temporal_policy_search_comparison(
    run_id: str,
    search_id: str,
    _identity: Annotated[SessionIdentity, Depends(require_temporal_start)],
) -> dict[str, Any]:
    try:
        return run_temporal_policy_comparison(database(), run_id, search_id)
    except (TemporalPolicySearchError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc


@router.post("/{run_id}/policy-search/{search_id}/study")
def temporal_policy_search_study(
    run_id: str,
    search_id: str,
    _identity: Annotated[SessionIdentity, Depends(require_temporal_start)],
) -> dict[str, Any]:
    try:
        return run_temporal_policy_study(database(), run_id, search_id)
    except (TemporalPolicySearchError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc


@router.get("/{run_id}/export.zip")
def export_temporal_intelligence(
    run_id: str,
    _identity: Annotated[SessionIdentity, Depends(require_temporal_export)],
    start_month: str | None = Query(None, pattern=r"^\d{4}-\d{2}$"),
    end_month: str | None = Query(None, pattern=r"^\d{4}-\d{2}$"),
) -> Response:
    try:
        content = build_temporal_intelligence_export(
            database(),
            run_id,
            start_month=start_month,
            end_month=end_month,
        )
    except (TemporalIntelligenceNotFound, TemporalIntelligenceConflict) as exc:
        raise _translate_error(exc) from exc
    return Response(
        content=content,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="temporal_intelligence_{run_id}.zip"'},
    )


@router.post("/{run_id}/strategy", status_code=201)
def create_strategy_from_temporal_intelligence(
    run_id: str,
    identity: Annotated[SessionIdentity, Depends(require_temporal_materialize_strategy)],
) -> dict[str, Any]:
    try:
        return materialize_temporal_intelligence_strategy(database(), run_id, actor_email=identity.email)
    except (TemporalIntelligenceNotFound, TemporalIntelligenceConflict) as exc:
        raise _translate_error(exc) from exc


@router.post("", status_code=202)
def create_temporal_intelligence(
    identity: Annotated[SessionIdentity, Depends(require_temporal_start)],
) -> dict[str, Any]:
    try:
        return start_temporal_intelligence(database(), actor_email=identity.email)
    except (TemporalIntelligenceConflict, ValueError, RuntimeError) as exc:
        raise _translate_error(exc) from exc


@router.get("/{run_id}/strategy-research/pipeline/snapshot")
def strategy_research_pipeline_snapshot(
    run_id: str,
    _identity: Annotated[SessionIdentity, Depends(require_temporal_view)],
) -> dict[str, Any]:
    try:
        return get_strategy_research_pipeline_snapshot(database(), run_id)
    except (TemporalIntelligenceConflict, TemporalIntelligenceNotFound) as exc:
        raise _translate_error(exc) from exc


@router.get("/{run_id}/strategy-research/pipeline")
def strategy_research_pipeline_state(
    run_id: str,
    _identity: Annotated[SessionIdentity, Depends(require_temporal_view)],
) -> dict[str, Any]:
    try:
        return get_strategy_research_pipeline_state(database(), run_id)
    except TemporalIntelligenceNotFound as exc:
        raise _translate_error(exc) from exc


@router.post("/{run_id}/strategy-research/pipeline/control")
def control_strategy_research_run(
    run_id: str,
    payload: StrategyResearchPipelineControlRequest,
    _identity: Annotated[SessionIdentity, Depends(require_temporal_start)],
) -> dict[str, Any]:
    try:
        return control_strategy_research_pipeline(
            database(),
            run_id,
            action=payload.action,
            stage=payload.stage,
            start_month=payload.start_month,
            end_month=payload.end_month,
            message=payload.message,
        )
    except (TemporalIntelligenceNotFound, TemporalIntelligenceConflict) as exc:
        raise _translate_error(exc) from exc


@router.post("/{run_id}/strategy-research/pipeline/pause")
def pause_strategy_research_run(
    run_id: str,
    _identity: Annotated[SessionIdentity, Depends(require_temporal_stop)],
) -> dict[str, Any]:
    try:
        return request_strategy_research_pipeline_pause(database(), run_id)
    except (TemporalIntelligenceNotFound, TemporalIntelligenceConflict) as exc:
        raise _translate_error(exc) from exc


@router.post("/{run_id}/strategy-research/pipeline/stop")
def stop_strategy_research_run(
    run_id: str,
    _identity: Annotated[SessionIdentity, Depends(require_temporal_stop)],
) -> dict[str, Any]:
    try:
        return request_strategy_research_pipeline_stop(database(), run_id)
    except (TemporalIntelligenceNotFound, TemporalIntelligenceConflict) as exc:
        raise _translate_error(exc) from exc


@router.post("/{run_id}/strategy-research/reset")
def reset_strategy_research_run(
    run_id: str,
    _identity: Annotated[SessionIdentity, Depends(require_temporal_start)],
) -> dict[str, Any]:
    try:
        return reset_strategy_research_pipeline(database(), run_id)
    except (TemporalIntelligenceNotFound, TemporalIntelligenceConflict) as exc:
        raise _translate_error(exc) from exc


@router.post("/{run_id}/stop")
def stop_temporal_intelligence_run(
    run_id: str,
    _identity: Annotated[SessionIdentity, Depends(require_temporal_stop)],
) -> dict[str, Any]:
    try:
        return stop_temporal_intelligence(database(), run_id)
    except (TemporalIntelligenceNotFound, TemporalIntelligenceConflict) as exc:
        raise _translate_error(exc) from exc
