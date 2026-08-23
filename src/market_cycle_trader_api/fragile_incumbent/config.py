from __future__ import annotations

SCHEMA_VERSION = 1
ANALYSIS_VERSION = "6.2.0"
FOCUS_MONTHS = ("2022-12", "2022-04")
SEVERE_MONTH_THRESHOLD = -0.05
RANDOM_STATE = 42
INNER_VALIDATION_SHARE = 0.20
MIN_TRAIN_MONTHS = 12

FEATURES = (
    "position_drawdown_from_peak",
    "incumbent_risk_health",
    "position_return_since_entry",
    "score_change_from_entry",
    "best_vs_second_gap",
    "best_vs_current_gap",
    "all_horizon_risk_safety",
    "best_score_zscore",
    "short_profit_consensus",
    "long_profit_confirmation",
    "horizon_agreement",
    "current_asset_rank",
    "recent_rotations_10",
)
