from __future__ import annotations

SCHEMA_VERSION = 1
ANALYSIS_VERSION = "6.1.0"
FOCUS_MONTH = "2026-06"
SEVERE_MONTH_THRESHOLD = -0.05
RANDOM_STATE = 42
INNER_VALIDATION_SHARE = 0.20
MIN_TRAIN_MONTHS = 12

FEATURES = (
    "universe_breadth_5",
    "universe_breadth_20",
    "breadth_impulse",
    "positive_score_share",
    "best_score_zscore",
    "all_horizon_risk_safety",
    "short_profit_consensus",
    "long_profit_confirmation",
    "horizon_agreement",
)
