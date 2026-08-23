from __future__ import annotations

SCHEMA_VERSION = 1
TRAILING_WINDOW = 126
MIN_HISTORY = 30
RECENT_ROTATION_WINDOW = 10

STATE_HEALTHY = "healthy_leader"
STATE_WEAK = "weak_relative_leader"
STATE_WHIPSAW = "whipsaw_leadership"
STATE_NO_OPPORTUNITY = "no_good_opportunity"
STATES = (STATE_HEALTHY, STATE_WEAK, STATE_WHIPSAW, STATE_NO_OPPORTUNITY)

FALLBACK_THRESHOLDS = {
    "universe_breadth_20:q35": 0.45,
    "breadth_impulse:q25": -0.10,
    "spy_realized_volatility_20:q75": 0.25,
    "best_vs_second_gap:q35": 0.04,
    "position_drawdown_from_peak:q25": -0.07,
    "score_change_from_entry:q35": -0.02,
    "incumbent_risk_health:q35": 0.25,
    "all_horizon_risk_safety:q35": 0.20,
    "positive_score_share:q35": 0.45,
    "best_score_zscore:q35": 0.50,
}

QUALITY_PENALTIES = {
    "breadth_low": 8,
    "breadth_impulse_low": 14,
    "volatility_high": 8,
    "leader_gap_low": 12,
    "position_drawdown_low": 16,
    "score_change_low": 8,
    "risk_health_low": 10,
    "risk_safety_low": 10,
    "positive_share_low": 5,
    "best_score_weak": 4,
    "rotation_pressure": 5,
}
