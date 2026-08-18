from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "market_cycle_trader_api"
FRONT = ROOT.parent / "market_cycle_trader" / "src"


def test_market_cutoffs_are_structurally_separated() -> None:
    market = (SRC / "engine" / "market_data.py").read_text(encoding="utf-8")
    temporal = (SRC / "services" / "temporal_intelligence.py").read_text(encoding="utf-8")
    strategy = (SRC / "services" / "strategy_lab.py").read_text(encoding="utf-8")

    assert "latest_safe_completed_xnys_session" in market
    assert "refresh_market_data_to_live_cutoff" in market
    assert '"certified_backtest_cutoff"' in strategy
    assert '"live_market_cutoff"' in strategy
    assert '"research_snapshot_cutoff"' in temporal
    assert "temporal_research_boundary_refresh" in temporal


def test_trader_scheduler_keeps_live_cutoff_current() -> None:
    paper = (SRC / "services" / "paper_trading.py").read_text(encoding="utf-8")
    scheduler = (SRC / "services" / "paper_market_scheduler.py").read_text(encoding="utf-8")

    assert "refresh_trader_live_market_data" in paper
    assert "premarket_plan_refresh" in paper
    assert "paper_scheduler_live_refresh" in scheduler


def test_temporal_ui_displays_certified_live_and_frozen_cutoffs() -> None:
    panel = (FRONT / "features" / "TemporalIntelligencePanel.jsx").read_text(encoding="utf-8")
    assert "Certified through" in panel
    assert "Live market data" in panel
    assert "Research snapshot" in panel
    assert "/admin/strategies/control" in panel
