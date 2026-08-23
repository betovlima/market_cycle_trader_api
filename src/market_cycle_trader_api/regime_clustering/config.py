from __future__ import annotations

SCHEMA_VERSION = 2
ANALYSIS_VERSION = "6.3.0"
SEVERE_MONTH_THRESHOLD = -0.05
RANDOM_STATE = 42
MAX_CLUSTERS = 6
MIN_CLUSTERS = 2

FEATURES = (
    "universe_breadth_5",
    "universe_breadth_20",
    "breadth_impulse",
    "spy_realized_volatility_20",
    "spy_return_5",
    "spy_return_20",
    "best_vs_second_gap",
    "position_drawdown_from_peak",
    "position_return_since_entry",
    "score_change_from_entry",
    "incumbent_risk_health",
    "all_horizon_risk_safety",
    "positive_score_share",
    "best_score_zscore",
    "short_profit_consensus",
    "long_profit_confirmation",
    "horizon_agreement",
    "recent_rotations_10",
    "healthy_leader_share",
    "weak_relative_leader_share",
    "whipsaw_leadership_share",
    "no_good_opportunity_share",
)
