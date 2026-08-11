from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from .core.environment import load_project_environment

# Load market_cycle_trader_api/.env before importing modules that read
# environment variables at import time, such as the MongoDB repository.
load_project_environment()

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routers import (
    access_admin,
    asset_discovery,
    admin_rotations,
    analytics,
    admin_setup,
    admin_trader,
    auth,
    dashboard,
    exports,
    health,
    jobs,
    model_research,
    paper_market,
    public_paper_portfolio,
    parameter_bootstrap,
    strategy_configuration,
    strategy_lab,
    system_settings,
)
from .core.config import API_VERSION, cors_origins
from .auth.config import get_auth_settings
from .auth.security import require_admin_session, require_portfolio_session, require_trader_session
from .auth.access_service import get_access_service
from .core.runtime import close_mongo, initialize_mongo
from .services.paper_market_scheduler import (
    start_paper_market_scheduler,
    stop_paper_market_scheduler,
)
from .services.asset_discovery_scheduler import (
    start_asset_discovery_scheduler,
    stop_asset_discovery_scheduler,
)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    initialize_mongo()
    get_auth_settings().validate_runtime()
    get_access_service().ensure_storage()
    start_paper_market_scheduler()
    start_asset_discovery_scheduler()
    try:
        yield
    finally:
        stop_asset_discovery_scheduler()
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
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition", "Content-Length"],
    )

    application.include_router(health.router)
    application.include_router(auth.router)
    application.include_router(access_admin.router)
    viewer_required = [Depends(require_trader_session)]
    admin_required = [Depends(require_admin_session)]
    portfolio_required = [Depends(require_portfolio_session)]
    application.include_router(dashboard.router, dependencies=viewer_required)
    application.include_router(jobs.router, dependencies=viewer_required)
    application.include_router(model_research.router, dependencies=admin_required)
    application.include_router(exports.router, dependencies=admin_required)
    application.include_router(analytics.router)
    application.include_router(paper_market.router, dependencies=admin_required)
    application.include_router(public_paper_portfolio.router, dependencies=portfolio_required)
    application.include_router(admin_rotations.router, dependencies=admin_required)
    application.include_router(admin_trader.router, dependencies=admin_required)
    application.include_router(parameter_bootstrap.router, dependencies=admin_required)
    application.include_router(strategy_configuration.router, dependencies=admin_required)
    application.include_router(strategy_lab.router, dependencies=admin_required)
    application.include_router(asset_discovery.router, dependencies=admin_required)
    application.include_router(system_settings.router, dependencies=admin_required)
    application.include_router(admin_setup.router, dependencies=admin_required)
    return application


app = create_app()
