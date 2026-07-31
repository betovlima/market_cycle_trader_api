from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routers import config, exports, health, integrations, jobs
from .core.config import API_VERSION, cors_origins
from .core.runtime import close_mongo, initialize_mongo


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Own infrastructure startup and shutdown in one explicit lifecycle."""
    initialize_mongo()
    try:
        yield
    finally:
        close_mongo()


def create_app() -> FastAPI:
    """Build the HTTP application and compose feature routers."""
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
    )

    application.include_router(health.router)
    application.include_router(integrations.router)
    application.include_router(config.router)
    application.include_router(jobs.router)
    application.include_router(exports.router)
    return application


app = create_app()
