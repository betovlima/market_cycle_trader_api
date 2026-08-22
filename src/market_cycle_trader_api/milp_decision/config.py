from __future__ import annotations

COLLECTION = "temporal_decision_optimization"
PROCESSING_PREFIX = "strategy-milp:"
COST_STRESS_BPS = (0.0, 1.0, 2.0, 5.0, 10.0)

DEFAULT_CONFIGURATION = {
    "candidate_rank_limit": 3,
    "opportunity_weight": 0.50,
    "temporal_weight": 0.40,
    "persistence_weight": 0.10,
    "reference_anchor_weight": 0.70,
    "risk_penalty": 0.30,
    "tail_risk_penalty": 0.20,
    "turnover_penalty": 0.04,
    "switch_penalty": 0.04,
    "cash_objective": 0.55,
}
