from __future__ import annotations

CAPABILITY_NAMES = (
    "dashboard.view",
    "dashboard.strategy_intelligence.view",
    "backtest.view",
    "backtest.start",
    "backtest.export",
    "tuning.view",
    "tuning.start",
    "tuning.stop",
    "tuning.export",
    "tuning.logs.view",
    "tuning.promote",
    "research_models.view",
    "research_models.manage",
    "analytics.view",
    "portfolio.view",
    "asset_discovery.view",
    "administration.view",
    "settings.view",
    "research.manage",
    "admin.manage",
)

_ROLE_CAPABILITIES = {
    "viewer": {
        "dashboard.view",
        "backtest.view",
        "tuning.view",
        "research_models.view",
        "analytics.view",
    },
    "trader": {
        "dashboard.view",
        "dashboard.strategy_intelligence.view",
        "backtest.view",
        "backtest.start",
        "tuning.view",
        "research_models.view",
        "analytics.view",
        "portfolio.view",
    },
    "admin": set(CAPABILITY_NAMES),
}


def capabilities_for_role(role: str) -> dict[str, bool]:
    enabled = _ROLE_CAPABILITIES.get(str(role or "").lower(), set())
    return {name: name in enabled for name in CAPABILITY_NAMES}
