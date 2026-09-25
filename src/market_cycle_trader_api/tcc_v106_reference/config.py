"""Configuracao congelada da reproducao oficial do TCC.

Este modulo e independente de banco de dados e do Market Cycle Trader em tempo
de execucao. Os dados entram somente por CSVs locais gerados pela etapa de
snapshot da Alpaca.

Versao cientifica: 1.0.6
Backend oficial: CPU
Comparacao experimental: Control vs Soft Horizon Consensus
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any

EXPERIMENT_VERSION = "1.0.6"
START_DATE = "2016-01-01"
ANALYSIS_END_DATE = "2026-09-17"
BAR_SNAPSHOT_AS_OF_END = "2026-09-17"

ASSETS = (
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AMD", "JPM",
    "SPY", "AVGO", "NFLX", "CRM", "ORCL", "COST", "LLY", "XOM", "CAT",
    "WMT", "V", "HD", "ADC", "ADEA", "ADI", "ADM", "DDS", "CNQ", "VRTS",
    "XSD", "CXW", "ENS", "CORT", "CCK", "CEF", "GKOS", "RACE", "TX", "UNF",
    "BXMT", "PXLW", "KKR", "SCSC", "LKFT", "DNN", "VNCE", "UNFI", "DOC",
    "CLMT", "APD", "MGM", "MAN", "MYE", "YANG", "MKSI", "MCS", "ECC",
)

REFERENCE_ASSETS = (
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AMD", "JPM",
    "SPY", "AVGO", "NFLX", "CRM", "ORCL", "COST", "LLY", "XOM", "CAT",
    "WMT", "V", "HD", "ADC", "ADEA", "ADI", "ADM",
)

CANDIDATE_ASSETS = tuple(asset for asset in ASSETS if asset not in REFERENCE_ASSETS)

SOFT_HORIZON_CONSENSUS_PENALTY = 1.0


def _lightgbm_settings() -> dict[str, Any]:
    return {
        "schema_version": 3,
        "settings_revision": 2,
        "profile_id": "strategy",
        "lightgbm": {
            "n_estimators": 329,
            "learning_rate": 0.020731,
            "max_depth": 3,
            "num_leaves": 6,
            "min_child_samples": 18,
            "min_child_weight": 5.0,
            "subsample": 0.85,
            "subsample_freq": 0,
            "colsample_bytree": 0.88067,
            "reg_alpha": 0.050837,
            "reg_lambda": 3.596305,
            "max_bin": 255,
            "n_jobs": -1,
            "repetitions": 1,
            "seed_step": 1000,
            "random_state": 42,
        },
    }


@dataclass(frozen=True)
class StandaloneBacktestConfig:
    assets: tuple[str, ...] = ASSETS
    strategy_mode: str = "COMPOUND_ROTATION_SWING_LIGHTGBM"
    start_date: str = START_DATE
    end_date: str | None = None
    timeframe: str = "1Day"

    # Estes campos descrevem a semantica do dado entregue ao motor depois da
    # normalizacao local dos splits. O download fisico e sempre Alpaca RAW/SIP.
    market_data_provider: str = "alpaca"
    alpaca_historical_feed: str = "sip"
    alpaca_live_feed: str = "iex"
    alpaca_adjustment: str = "split"
    market_data_history_backfill_enabled: bool = False
    market_data_history_backfill_provider: str = "alpaca"
    market_data_history_start_tolerance_days: int = 10
    market_data_require_complete_history: bool = True

    rotation_horizon_days: int = 40
    rotation_target_horizons: tuple[int, ...] = (5, 10, 20, 40, 60)
    rotation_target_horizon_weights: tuple[float, ...] = (
        0.10, 0.15, 0.20, 0.30, 0.25
    )
    rotation_movement_capture_weight: float = 0.35
    rotation_trend_persistence_weight: float = 0.20
    rotation_minimum_training_rows: int = 700
    rotation_walk_forward_enabled: bool = True
    rotation_walk_forward_calibration_days: int = 126
    rotation_walk_forward_test_days: int = 504
    rotation_walk_forward_min_test_days: int = 126
    rotation_purge_days: int = 60

    rotation_downside_penalty: float = 0.20
    rotation_drawdown_penalty: float = 0.35
    rotation_min_holding_days: int = 2
    rotation_min_expected_edge: float = 0.001
    rotation_cash_threshold: float = 0.0
    rotation_switch_margin: float = 0.0005
    rotation_switch_margin_candidates: tuple[float, ...] = (
        0.0, 0.0025, 0.005, 0.01
    )

    opportunity_utility_entry_threshold: float = 0.28
    opportunity_utility_exit_threshold: float = 0.27
    allocation_lookback_days: int = 126
    allocation_max_asset_weight: float = 1.0
    allocation_cvar_confidence: float = 0.95
    allocation_cvar_penalty: float = 1.0
    allocation_turnover_penalty: float = 0.0025
    allocation_minimum_utility: float = 0.0
    allocation_signal_scale: float = 1.0

    rotation_accelerator: str = "cpu"
    rotation_allow_cpu_fallback: bool = False
    rotation_model_repetitions: int = 1
    rotation_seed_step: int = 1000

    initial_capital: float = 10_000.0
    whole_shares: bool = False
    slippage_bps: float = 0.0
    commission_rate: float = 0.0
    sec_fee_rate: float = 0.0000206
    taf_fee_per_share: float = 0.000195
    taf_fee_cap: float = 9.79
    cat_fee_per_share: float = 0.000003

    deterministic_execution: bool = False
    numeric_thread_limit: int = 1
    random_state: int = 42

    analysis_start_date: str = START_DATE
    analysis_end_date: str | None = ANALYSIS_END_DATE
    calendar_anchor_assets: tuple[str, ...] = REFERENCE_ASSETS
    research_reference_assets: tuple[str, ...] = REFERENCE_ASSETS
    research_candidate_assets: tuple[str, ...] = CANDIDATE_ASSETS
    research_model_family: str = "lightgbm_utility"
    research_model_settings: dict[str, Any] = field(default_factory=_lightgbm_settings)
    walk_forward_fold_count_override: int | None = None

    @property
    def fractional_shares(self) -> bool:
        return not self.whole_shares

    def model_copy(
        self,
        *,
        update: dict[str, Any] | None = None,
    ) -> "StandaloneBacktestConfig":
        return replace(self, **dict(update or {}))


def build_control_config(
    base: StandaloneBacktestConfig,
    *,
    assets: tuple[str, ...] | list[str] | None = None,
) -> StandaloneBacktestConfig:
    """Control = LightGBM + politica-base, sem consenso Soft."""
    settings = deepcopy(base.research_model_settings)
    lightgbm = deepcopy(settings.get("lightgbm") or {})
    lightgbm["early_stopping_enabled"] = False
    settings["lightgbm"] = lightgbm
    settings["horizon_voting"] = {"enabled": False}
    settings["soft_horizon_consensus"] = {"enabled": False}
    update: dict[str, Any] = {"research_model_settings": settings}
    if assets is not None:
        update["assets"] = tuple(assets)
    return base.model_copy(update=update)


def build_soft_config(
    base: StandaloneBacktestConfig,
    *,
    assets: tuple[str, ...] | list[str] | None = None,
    penalty_strength: float = SOFT_HORIZON_CONSENSUS_PENALTY,
) -> StandaloneBacktestConfig:
    """Soft = mesmo Control + modificador continuo multi-horizonte."""
    control = build_control_config(base, assets=assets)
    settings = deepcopy(control.research_model_settings)
    settings["soft_horizon_consensus"] = {
        "enabled": True,
        "penalty_strength": float(penalty_strength),
    }
    return control.model_copy(update={"research_model_settings": settings})


CONFIG = StandaloneBacktestConfig()
