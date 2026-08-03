from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from .core.environment import load_project_environment

# Load market_cycle_trader_api/.env before importing modules that read
# environment variables at import time, such as the MongoDB repository.
load_project_environment()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routers import (
    admin_setup,
    dashboard,
    exports,
    health,
    jobs,
    paper_market,
    public_paper_portfolio,
    parameter_bootstrap,
    strategy_configuration,
)
from .core.config import API_VERSION, cors_origins
from .core.runtime import close_mongo, initialize_mongo
from .services.paper_market_scheduler import (
    start_paper_market_scheduler,
    stop_paper_market_scheduler,
)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    initialize_mongo()
    start_paper_market_scheduler()
    try:
        yield
    finally:
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
    application.include_router(dashboard.router)
    application.include_router(jobs.router)
    application.include_router(exports.router)
    application.include_router(paper_market.router)
    application.include_router(public_paper_portfolio.router)
    application.include_router(parameter_bootstrap.router)
    application.include_router(strategy_configuration.router)
    application.include_router(admin_setup.router)
    return application


app = create_app()
