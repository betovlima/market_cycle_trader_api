from __future__ import annotations

import hashlib
from pathlib import Path

from market_cycle_trader_api.tcc_v106_reference import config as tcc_config
from market_cycle_trader_api.services import jobs as jobs_service


ROOT = Path(__file__).resolve().parents[1]
VENDORED = ROOT / "src" / "market_cycle_trader_api" / "tcc_v106_reference"

EXPECTED_GIT_BLOBS = {
    "__init__.py": "c678201b82ddc4339a364aef4fc14541ee3ed0d1",
    "absolute_utility_cash_gate.py": "cafda67e1bb25893ae7128cb07b605665a9eafc4",
    "capital_rotation.py": "b8fb12c87c23d77fda17dac08457c46f00d5c236",
    "compound_risk_overlay.py": "0a729a228bcb04182863fb113ae95f8b0efa66c8",
    "concentrated_allocation.py": "dd37cbed077c61c52a5e3983ceda818b68cc9c12",
    "config.py": "22fd03894463579dc8a5757ae4c14d21c21610ee",
    "execution.py": "165d5d98c59e2bfc443fbce7af10c8d9ac963971",
    "optimized_allocation.py": "21332d6902271894939972e01f308e0e37e61b67",
    "research_challengers.py": "95c242bd29233095a7da8cc71da733d7d7592546",
    "rotation_diagnostics.py": "aa16531943ca0fd7413b5896fe9a78c507bf3342",
    "selective_opportunity.py": "5d8b4335ef106ac0acd2dadec35f5a6a024ff1ed",
}


def _git_blob_sha(path: Path) -> str:
    payload = path.read_bytes()
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def test_tcc_v106_engine_is_vendored_verbatim() -> None:
    actual = {
        name: _git_blob_sha(VENDORED / name)
        for name in EXPECTED_GIT_BLOBS
    }
    assert actual == EXPECTED_GIT_BLOBS


def test_tcc_v106_reference_config_is_frozen() -> None:
    assert tcc_config.EXPERIMENT_VERSION == "1.0.6"
    assert tcc_config.CONFIG.strategy_mode == "COMPOUND_ROTATION_SWING_LIGHTGBM"
    assert tcc_config.CONFIG.rotation_accelerator == "cpu"
    assert tcc_config.CONFIG.deterministic_execution is False
    assert tcc_config.CONFIG.rotation_switch_margin == 0.0005
    assert tcc_config.CONFIG.rotation_switch_margin_candidates == (
        0.0,
        0.0025,
        0.005,
        0.01,
    )
    assert tcc_config.CONFIG.rotation_target_horizons == (5, 10, 20, 40, 60)
    assert tcc_config.CONFIG.rotation_target_horizon_weights == (
        0.10,
        0.15,
        0.20,
        0.30,
        0.25,
    )
    assert tcc_config.CONFIG.analysis_end_date == "2026-09-17"
    assert len(tcc_config.ASSETS) == 56


def test_tcc_v106_reference_engine_module_is_allowlisted() -> None:
    assert (
        jobs_service.TCC_V106_REFERENCE_ENGINE_MODULE
        in jobs_service._ALLOWED_ENGINE_MODULES
    )
