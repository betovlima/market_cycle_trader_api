from pathlib import Path


def test_continuous_robot_routes_and_controller_are_present() -> None:
    root = Path(__file__).resolve().parents[1]
    scheduler = (root / "src/market_cycle_trader_api/services/paper_market_scheduler.py").read_text()
    router = (root / "src/market_cycle_trader_api/api/routers/paper_market.py").read_text()
    public_router = (root / "src/market_cycle_trader_api/api/routers/public_paper_portfolio.py").read_text()

    assert "continuous_regular_sessions" in scheduler
    assert "_ensure_continuous_run" in scheduler
    assert "adopted_existing_run" in scheduler
    assert '@router.get("/robot/status")' in router
    assert '@router.post("/robot/stop")' in router
    assert '@router.get("/public-robot-status")' in public_router
