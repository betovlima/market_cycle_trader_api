from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from .core.environment import load_project_environment



load_project_environment()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routers import health, paper_market
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
        description="Private market execution API.",
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
    application.include_router(paper_market.router)
    return application


app = create_app()
