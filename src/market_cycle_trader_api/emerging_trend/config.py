from __future__ import annotations

SCHEMA_VERSION = 1
ANALYSIS_VERSION = "6.4.0"
RANDOM_STATE = 42
INNER_VALIDATION_SHARE = 0.20
MIN_TRAIN_ROWS = 120
LEADER_PERSISTENCE_HORIZON = 5
FORWARD_RETURN_HORIZON = 10
MIN_LEADER_PERSISTENCE_SHARE = 0.60
MIN_FORWARD_RETURN = 0.03
FOCUS_MONTHS = ("2021-05", "2021-06", "2021-07")

FEATURES = (
    "best_score",
    "best_vs_second_gap",
    "best_vs_current_gap",
    "current_score",
    "current_asset_rank",
    "universe_score_mean",
    "universe_score_std",
    "best_score_zscore",
    "best_vs_second_zscore",
    "universe_breadth_5",
    "universe_breadth_20",
    "spy_return_5",
    "spy_return_20",
    "spy_realized_volatility_20",
    "position_return_since_entry",
    "position_drawdown_from_peak",
    "score_change_from_entry",
    "days_current_not_top1",
    "consecutive_days_current_not_top1",
    "entry_rank_score",
    "risk_adjusted_asset_rank_score",
    "entry_rank_percentile",
    "opportunity_gate_score",
    "risk_adjusted_entry_score",
    "entry_risk_multiplier",
    "hold_score",
    "incumbent_persistence_score",
    "incumbent_risk_health",
    "short_profit_consensus",
    "short_risk_safety",
    "long_profit_confirmation",
    "long_risk_safety",
    "cross_horizon_agreement",
    "horizon_agreement",
    "all_horizon_risk_safety",
    "predicted_drawdown",
    "entry_separation_strength",
    "entry_top_gap_strength",
    "profit_before_loss_probability_h5",
    "profit_before_loss_probability_h10",
    "profit_before_loss_probability_h20",
    "trend_persistence_probability_h5",
    "trend_persistence_probability_h10",
    "trend_persistence_probability_h20",
)
