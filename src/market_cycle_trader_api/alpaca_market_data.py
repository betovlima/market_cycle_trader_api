"""Compatibility facade for the legacy standalone engine entry point."""
try:  # Package import (FastAPI / tests)
    from .infrastructure.market_data.alpaca import *  # noqa: F401,F403
except ImportError:  # Standalone engine script import
    from infrastructure.market_data.alpaca import *  # type: ignore # noqa: F401,F403
