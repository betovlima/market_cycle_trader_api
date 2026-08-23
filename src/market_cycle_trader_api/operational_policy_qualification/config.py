SCHEMA_VERSION = 1
ANALYSIS_VERSION = "6.6.0"
HORIZONS = (1, 3, 5, 10)
UTILITY_WEIGHTS = {1: 0.10, 3: 0.20, 5: 0.30, 10: 0.40}
DOWNSIDE_PENALTY = 0.20
ONE_SIDE_COST_BPS = 2.0
MIN_UTILITY_EDGE = 0.005
ACTION_PROBABILITY_THRESHOLD = 0.70
MIN_INTERVENTIONS = 5
MIN_CAPITAL_LIFT = 0.02
MAX_SHARPE_DEGRADATION = 0.02
MAX_DRAWDOWN_DEGRADATION = 0.01
MAX_WORST_MONTH_DEGRADATION = 0.02
MIN_POSITIVE_OOS_YEARS = 3
RANDOM_STATE = 42

FEATURES = (
    "top1_top2_gap",
    "target_rank",
    "incumbent_rank",
    "target_score",
    "incumbent_score",
    "target_minus_incumbent_score",
    "universe_score_mean",
    "universe_score_std",
    "positive_score_share",
    "target_entry_rank_percentile",
    "target_opportunity_gate_score",
    "target_risk_adjusted_entry_score",
    "target_hold_score",
    "target_incumbent_persistence_score",
    "target_incumbent_risk_health",
    "target_short_profit_consensus",
    "target_long_profit_confirmation",
    "target_horizon_agreement",
    "target_all_horizon_risk_safety",
    "target_predicted_drawdown",
    "incumbent_entry_rank_percentile",
    "incumbent_opportunity_gate_score",
    "incumbent_hold_score",
    "incumbent_persistence_score",
    "incumbent_risk_health",
    "incumbent_short_profit_consensus",
    "incumbent_long_profit_confirmation",
    "incumbent_horizon_agreement",
    "incumbent_all_horizon_risk_safety",
    "incumbent_predicted_drawdown",
    "delta_opportunity_gate_score",
    "delta_hold_score",
    "delta_incumbent_risk_health",
    "delta_short_profit_consensus",
    "delta_long_profit_confirmation",
    "delta_horizon_agreement",
    "delta_all_horizon_risk_safety",
    "delta_predicted_drawdown",
)
