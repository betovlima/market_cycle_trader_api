from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = ROOT / 'src/market_cycle_trader_api/services/paper_market_scheduler.py'
ROUTER = ROOT / 'src/market_cycle_trader_api/api/routers/admin_trader.py'


def test_trader_control_modes_are_explicit():
    source = SCHEDULER.read_text(encoding='utf-8')
    assert 'TRADER_CONTROL_MODES = frozenset({"active", "paused", "exit_only", "stopped"})' in source


def test_exit_only_never_allows_entries_or_rotations():
    source = SCHEDULER.read_text(encoding='utf-8')
    assert 'EXIT_ONLY_ALLOWED_ACTIONS = frozenset({"sell_to_cash", "stay_in_cash", "hold"})' in source
    assert 'Blocked by exit-only Trader mode.' in source


def test_admin_control_routes_exist():
    source = ROUTER.read_text(encoding='utf-8')
    assert '@router.get("/status")' in source
    assert '@router.post("/mode")' in source
    assert '@router.get("/history")' in source
