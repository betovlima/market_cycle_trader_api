from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "src" / "market_cycle_trader_api" / "main.py"
ANALYTICS = ROOT / "src" / "market_cycle_trader_api" / "api" / "routers" / "analytics.py"


def test_viewer_can_read_dashboard_jobs_and_backtest_analytics_but_not_start_or_export() -> None:
    source = MAIN.read_text(encoding="utf-8")
    analytics = ANALYTICS.read_text(encoding="utf-8")
    assert "application.include_router(dashboard.router, dependencies=viewer_required)" in source
    assert "application.include_router(jobs.router, dependencies=backtest_access)" in source
    assert "backtest_access = [Depends(require_backtest_access)]" in source
    assert "application.include_router(exports.router, dependencies=admin_required)" in source
    assert 'Depends(require_trader_session)' in analytics
    assert '@router.get("/backtests/{job_id}")' in analytics


def test_portfolio_is_trader_or_admin_and_administration_is_admin_only() -> None:
    source = MAIN.read_text(encoding="utf-8")
    analytics = ANALYTICS.read_text(encoding="utf-8")
    assert "application.include_router(public_paper_portfolio.router, dependencies=portfolio_required)" in source
    assert '@router.get("/portfolio")' in analytics
    assert 'Depends(require_portfolio_session)' in analytics
    assert "application.include_router(paper_market.router, dependencies=admin_required)" in source
    assert "application.include_router(parameter_bootstrap.router, dependencies=admin_required)" in source
    assert "application.include_router(strategy_configuration.router, dependencies=admin_required)" in source
    assert "application.include_router(admin_setup.router, dependencies=admin_required)" in source
    assert "application.include_router(admin_rotations.router, dependencies=admin_required)" in source


def test_viewer_can_read_tuning_and_strategy_research_but_mutations_remain_admin_only() -> None:
    source = MAIN.read_text(encoding="utf-8")
    tuning = (ROOT / "src" / "market_cycle_trader_api" / "api" / "routers" / "model_tuning.py").read_text(encoding="utf-8")
    strategies = (ROOT / "src" / "market_cycle_trader_api" / "api" / "routers" / "strategy_lab.py").read_text(encoding="utf-8")
    assert "research_access = [Depends(require_trader_access)]" in source
    assert "application.include_router(model_research.router, dependencies=research_access)" in source
    assert "application.include_router(model_tuning.router, dependencies=research_access)" in source
    assert "application.include_router(strategy_lab.router, dependencies=research_access)" in source
    assert "ViewerIdentity = Annotated[SessionIdentity, Depends(require_trader_session)]" in strategies
    assert "def read_strategy_control(_: ViewerIdentity)" in strategies
    assert "def read_strategy(strategy_id: str, _: ViewerIdentity)" in strategies
    assert "_identity: Annotated[SessionIdentity, Depends(require_admin_session)]" in tuning[tuning.index("def export_tuning"):tuning.index("def adopt_tuning_candidate")]
