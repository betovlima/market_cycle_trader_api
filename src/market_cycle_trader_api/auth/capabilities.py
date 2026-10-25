from __future__ import annotations

CAPABILITY_NAMES = (
    "annual_replay.view",
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
    "temporal_intelligence.view",
    "temporal_intelligence.start",
    "temporal_intelligence.stop",
    "temporal_intelligence.export",
    "temporal_intelligence.materialize_strategy",
    "decision_science.view",
    "decision_science.export",
    "analytics.view",
    "asset_discovery.view",
    "asset_discovery.start",
    "asset_discovery.stop",
    "asset_discovery.export",
    "asset_discovery.create_strategy",
    "portfolio.view",
    "administration.view",
    "settings.view",
    "research.manage",
    "admin.manage",
)

_ROLE_CAPABILITIES = {
    "replay_guest": {
        "annual_replay.view",
    },
    "viewer": {
        "dashboard.view",
        "backtest.view",
        "annual_replay.view",
        "tuning.view",
        "research_models.view",
        "temporal_intelligence.view",
        "decision_science.view",
        "analytics.view",
    },
    "reviewer": {
        "dashboard.view",
        "dashboard.strategy_intelligence.view",
        "backtest.view",
        "annual_replay.view",
        "backtest.export",
        "research_models.view",
        "temporal_intelligence.view",
        "temporal_intelligence.export",
        "analytics.view",
        "asset_discovery.view",
        "asset_discovery.export",
    },
    "trader": {
        "dashboard.view",
        "dashboard.strategy_intelligence.view",
        "backtest.view",
        "annual_replay.view",
        "backtest.start",
        "tuning.view",
        "research_models.view",
        "temporal_intelligence.view",
        "decision_science.view",
        "analytics.view",
        "portfolio.view",
    },
    "admin": set(CAPABILITY_NAMES),
}


def capabilities_for_role(role: str) -> dict[str, bool]:
    enabled = _ROLE_CAPABILITIES.get(str(role or "").lower(), set())
    return {name: name in enabled for name in CAPABILITY_NAMES}
