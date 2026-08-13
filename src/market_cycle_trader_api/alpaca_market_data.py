
try:  
    from .infrastructure.market_data.alpaca import *  # noqa: F401,F403
except ImportError:  
    from infrastructure.market_data.alpaca import *  # type: ignore # noqa: F401,F403
