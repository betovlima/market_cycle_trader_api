from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from .core.environment import load_project_environment


load_project_environment()

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .api.routers import (
    access_admin,
    admin_rotations,
    analytics,
    asset_discovery,
    admin_setup,
    admin_trader,
    auth,
    dashboard,
    exports,
    health,
    jobs,
    live_market,
    live_execution,
    model_research,
    model_tuning,
    paper_market,
    public_paper_portfolio,
    parameter_bootstrap,
    reviewer_access,
    strategy_configuration,
    strategy_lab,
    system_settings,
    tcc_v106_reference,
    tcc_u67_operational,
    tcc_u77_static_challenger,
    temporal_intelligence,
    temporal_research_settings,
    temporal_rotation_quality,
)
from .core.config import API_VERSION, cors_origins
from .milp_decision.router import router as milp_decision_router
from .decision_science.router import router as decision_science_router
from .roc_decision_policy.router import router as roc_decision_policy_router
from .auth.config import get_auth_settings
from .auth.replay_guest import replay_guest_denied
from .auth.security import SESSION_COOKIE_NAME, get_session_manager, require_admin_session, require_backtest_access, require_capability, require_portfolio_session, require_trader_access, require_trader_session
from .auth.access_service import get_access_service
from .auth.reviewer_access_service import get_reviewer_access_service
from .core.runtime import MONGO_STATUS, close_mongo, database, initialize_mongo
from .services.paper_market_scheduler import (
    start_paper_market_scheduler,
    stop_paper_market_scheduler,
)
from .services.live_observation_scheduler import start_live_observation_scheduler, stop_live_observation_scheduler
from .services.model_tuning import recover_integrated_model_tuning_runs
from .services.temporal_intelligence import recover_temporal_intelligence_runs
from .services.temporal_research_settings import ensure_temporal_research_settings
from .services.asset_discovery import purge_legacy_non_persistent_asset_discovery_records
from .roc_decision_policy.config import ensure_settings as ensure_roc_decision_policy_settings


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    initialize_mongo()
    if bool(MONGO_STATUS.get("available")):
        ensure_temporal_research_settings(database())
        ensure_roc_decision_policy_settings(database())
        purge_legacy_non_persistent_asset_discovery_records(database())
        recover_integrated_model_tuning_runs(database())
        recover_temporal_intelligence_runs(database())
    get_auth_settings().validate_runtime()
    get_access_service().ensure_storage()
    get_reviewer_access_service().ensure_storage()
    start_paper_market_scheduler()
    start_live_observation_scheduler()
    try:
        yield
    finally:
        stop_live_observation_scheduler()
        stop_paper_market_scheduler()
        close_mongo()


def create_app() -> FastAPI:
    application = FastAPI(
        title="Market Cycle Trader API",
        version=API_VERSION,
        description=(
            "MongoDB-backed market-cycle research and backtest API. "
            "Trading engines are isolated from HTTP orchestration."
        ),
        lifespan=lifespan,
        swagger_ui_parameters={
            # Native Swagger UI filtering searches OpenAPI tag groups.
            "filter": True,
            # Large APIs are easier to navigate with only tags initially visible.
            "docExpansion": "none",
            "tagsSorter": "alpha",
            "operationsSorter": "alpha",
            "deepLinking": True,
        },
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition", "Content-Length"],
    )

    @application.middleware("http")
    async def professor_replay_boundary(request: Request, call_next):
        token = request.cookies.get(SESSION_COOKIE_NAME)
        if token:
            try:
                identity = get_session_manager().decode_session_token(token)
            except HTTPException:
                # Normal authentication dependencies handle expired/invalid tokens.
                identity = None
            if identity is not None:
                if replay_guest_denied(identity.role, request.method, request.url.path):
                    return JSONResponse(
                        status_code=403,
                        content={"detail": "Only Annual Replay is available for this account."},
                    )
        return await call_next(request)

    application.include_router(health.router)
    application.include_router(auth.router)
    application.include_router(reviewer_access.public_router)
    application.include_router(reviewer_access.admin_router)
    application.include_router(access_admin.router)
    viewer_required = [Depends(require_trader_session)]
    backtest_access = [Depends(require_backtest_access)]
    backtest_export_access = [Depends(require_capability("backtest.export"))]
    admin_required = [Depends(require_admin_session)]
    research_access = [Depends(require_trader_access)]
    portfolio_required = [Depends(require_portfolio_session)]
    application.include_router(dashboard.router, dependencies=viewer_required)
    application.include_router(jobs.router, dependencies=backtest_access)
    application.include_router(model_research.router, dependencies=research_access)
    application.include_router(model_tuning.router, dependencies=research_access)
    application.include_router(
        tcc_v106_reference.router,
        dependencies=research_access,
    )
    application.include_router(
        tcc_u67_operational.router,
        dependencies=research_access,
    )
    application.include_router(
        tcc_u77_static_challenger.router,
        dependencies=research_access,
    )
    application.include_router(temporal_intelligence.router)
    application.include_router(milp_decision_router)
    application.include_router(decision_science_router)
    application.include_router(roc_decision_policy_router)
    application.include_router(temporal_rotation_quality.router)
    application.include_router(exports.router, dependencies=backtest_export_access)
    application.include_router(analytics.router)
    application.include_router(asset_discovery.router)
    application.include_router(paper_market.router, dependencies=admin_required)
    application.include_router(live_market.router, dependencies=admin_required)
    application.include_router(live_execution.router, dependencies=admin_required)
    application.include_router(public_paper_portfolio.router, dependencies=portfolio_required)
    application.include_router(admin_rotations.router, dependencies=admin_required)
    application.include_router(admin_trader.router, dependencies=admin_required)
    application.include_router(parameter_bootstrap.router, dependencies=admin_required)
    application.include_router(strategy_configuration.router, dependencies=admin_required)
    application.include_router(strategy_lab.router, dependencies=research_access)
    application.include_router(system_settings.router, dependencies=admin_required)
    application.include_router(temporal_research_settings.router, dependencies=admin_required)
    application.include_router(admin_setup.router, dependencies=admin_required)
    return application


app = create_app()
