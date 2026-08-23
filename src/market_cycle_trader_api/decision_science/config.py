from __future__ import annotations

SCHEMA_VERSION = 2
ANALYSIS_VERSION = "6.0.0"
TARGET_HORIZON = 5
TARGET_NAME = "realized_profit_before_loss_h5"
TARGET_DESCRIPTION = (
    "Significant growth: the asset reaches the Temporal Intelligence 5-session profit barrier "
    "before the loss barrier. At h=5 the configured barrier is approximately +4% before -2.5%."
)
MIN_TRAIN_DATES = 120
INNER_VALIDATION_SHARE = 0.20
RANDOM_STATE = 42
PROBABILITY_BINS = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.000001)

OPPORTUNITY_FEATURES = (
    "profit_before_loss_probability_h5",
    "profit_before_loss_probability_h10",
    "profit_before_loss_probability_h20",
    "profit_percentile_h5",
    "profit_percentile_h10",
    "profit_percentile_h20",
    "short_profit_consensus",
    "long_profit_confirmation",
    "horizon_agreement",
    "all_horizon_risk_safety",
    "predicted_drawdown",
    "predicted_drawdown_h5",
    "predicted_drawdown_h10",
    "predicted_drawdown_h20",
    "risk_safety_percentile_h5",
    "risk_safety_percentile_h10",
    "risk_safety_percentile_h20",
    "bottom_probability_h5",
    "top_probability_h5",
    "trend_persistence_probability_h5",
    "entry_rank_score",
    "risk_adjusted_asset_rank_score",
    "opportunity_gate_score",
)

TRANSITION_BASE_FEATURES = (
    "profit_before_loss_probability_h5",
    "profit_before_loss_probability_h10",
    "short_profit_consensus",
    "long_profit_confirmation",
    "horizon_agreement",
    "all_horizon_risk_safety",
    "predicted_drawdown",
    "risk_safety_percentile_h5",
    "entry_rank_score",
    "risk_adjusted_asset_rank_score",
    "opportunity_gate_score",
)

TRANSITION_CONTEXT_FEATURES = (
    "best_vs_current_gap",
    "best_vs_second_gap",
    "universe_breadth_5",
    "universe_breadth_20",
    "spy_return_5",
    "spy_return_20",
    "spy_realized_volatility_20",
)
