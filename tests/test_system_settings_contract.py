from pathlib import Path

from market_cycle_trader_api.schemas.system_settings import (
    SystemSettingsUpdateRequest,
    TrainingSettingsPatch,
)

ROOT = Path(__file__).resolve().parents[1]
ROUTER = ROOT / "src/market_cycle_trader_api/api/routers/system_settings.py"
SERVICE = ROOT / "src/market_cycle_trader_api/services/system_settings.py"
JOBS = ROOT / "src/market_cycle_trader_api/api/routers/jobs.py"
SCHEDULER = ROOT / "src/market_cycle_trader_api/services/paper_market_scheduler.py"
FRONTEND_ROUTE = "/api/admin/system-settings"


def test_system_settings_schema_accepts_valid_operational_values() -> None:
    request = SystemSettingsUpdateRequest(
        expected_revision=3,
        reason="Use the Railway CPU allocation",
        training=TrainingSettingsPatch(
            enabled=True,
            automatic_training_enabled=True,
            model_threads=8,
            numeric_threads=4,
            max_concurrent_jobs=1,
            timeout_seconds=21_600,
        ),
    )
    assert request.training.model_threads == 8
    assert request.training.max_concurrent_jobs == 1


def test_admin_system_settings_routes_exist() -> None:
    source = ROUTER.read_text(encoding="utf-8")
    assert 'router = APIRouter(prefix="/api/admin/system-settings"' in source
    assert '@router.get("")' in source
    assert '@router.patch("")' in source
    assert '@router.get("/history")' in source


def test_system_settings_use_revisioned_mongodb_storage() -> None:
    source = SERVICE.read_text(encoding="utf-8")
    assert 'SYSTEM_SETTINGS_ID = "market-cycle-runtime"' in source
    assert '"revision": 1' in source
    assert '"$inc": {"revision": 1}' in source
    assert 'SystemSettingsConflict' in source
    assert 'SYSTEM_SETTINGS_HISTORY_COLLECTION' in source


def test_training_limits_are_applied_to_manual_and_automatic_training() -> None:
    jobs_source = JOBS.read_text(encoding="utf-8")
    scheduler_source = SCHEDULER.read_text(encoding="utf-8")
    assert 'active_jobs >= 1' in jobs_source
    assert 'Wait for the active backtest to finish' in jobs_source
    assert 'training_timeout_seconds' in jobs_source
    assert 'apply_training_runtime_settings' in jobs_source
    assert 'automatic_training_enabled' in scheduler_source
    assert 'Waiting for training to be enabled in System Settings.' in scheduler_source



def test_runtime_settings_never_override_locked_winner_compute_fields() -> None:
    source = SERVICE.read_text(encoding="utf-8")
    function_source = source.split("def apply_training_runtime_settings", 1)[1]
    assert 'configuration.model_dump(mode="python")' in function_source
    assert 'payload["xgb_n_jobs"]' not in function_source
    assert 'payload["numeric_thread_limit"]' not in function_source
    assert 'settings["model_threads"]' not in function_source
    assert 'settings["numeric_threads"]' not in function_source
