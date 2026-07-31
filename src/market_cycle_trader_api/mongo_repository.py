"""Compatibility facade for the legacy standalone engine entry point.

New application code imports the persistence adapter from
``market_cycle_trader_api.infrastructure.persistence.mongo_repository``.
The quantitative engine is still executable as a standalone Python script and
historically imports ``mongo_repository`` from the package directory, so this
facade preserves that contract without duplicating implementation logic.
"""
try:  # Package import (FastAPI / tests)
    from .infrastructure.persistence.mongo_repository import *  # noqa: F401,F403
except ImportError:  # Standalone engine script import
    from infrastructure.persistence.mongo_repository import *  # type: ignore # noqa: F401,F403
