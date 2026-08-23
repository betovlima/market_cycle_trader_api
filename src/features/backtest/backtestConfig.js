export const ROTATION_PAGE_SIZE = 12

export const HISTORY_PAGE_SIZE = 10

export const ZOOM_STEP = 0.84


export const DAY_MS = 24 * 60 * 60 * 1000

export const METRIC_HINTS = {
  ending_capital: 'Portfolio capital at the end of the selected simulation after the modeled trades and transaction costs.',
  reference_ending_capital: 'Ending value of the reference Buy & Hold comparison over the same historical period.',
  cagr: 'Compound annual growth rate. It converts the total simulated growth into an annualized rate for easier comparison.',
  sharpe: 'Risk-adjusted return based on the variability of the simulated returns. Higher values indicate more return per unit of volatility.',
  maximum_drawdown: 'Largest peak-to-trough decline in portfolio equity during the simulation.',
  session_win_rate: 'Share of evaluated sessions in which the strategy produced a positive portfolio return.',
  average_cash_weight: 'Average fraction of portfolio equity kept in CASH across the out-of-sample execution period.',
  average_assets_held: 'Average number of risky assets simultaneously held by the optimized allocation engine.',
  maximum_assets_held: 'Largest number of risky assets held simultaneously at any evaluated session.',
  allocation_rebalances: 'Number of execution sessions in which the optimized target weights caused capital to move between assets or CASH.',
  risk_overlay_decisions: 'Number of out-of-sample sessions evaluated by the Compound Risk Overlay after the original rotation policy selected HOLD, ROTATE, ENTER or CASH.',
  risk_overlay_technical_fallbacks: 'Technical cases in which the risk estimate was unavailable and the engine explicitly preserved the original base-policy allocation instead of disguising the problem as a CASH decision.',
  risk_overlay_full_exposure_decisions: 'Sessions in which the risk overlay preserved essentially 100% of compounded capital in the asset selected by the original rotation policy.',
  risk_overlay_reduced_exposure_decisions: 'Sessions in which the base policy selected an asset but the risk overlay moved part of compounded capital to CASH.',
  average_primary_weight: 'Average fraction of total portfolio equity allocated to the Top-1 asset selected by Ranking Utility.',
  average_primary_share_of_risk: 'Average share of risky capital concentrated in the Top-1 asset, excluding CASH from the denominator.',
  average_secondary_weight: 'Average fraction of total portfolio equity allocated to optional Top-2/Top-3 positions.',
  absolute_utility_gate_decisions: 'Out-of-sample sessions evaluated by the Absolute Utility Cash Gate using the Champion Top-1 utility directly, without fitting a second opportunity classifier.',
  absolute_utility_gate_accepted: 'Sessions in which the Top-1 absolute Utility met the active hysteresis floor and the protected base rotation policy was allowed to remain in the market.',
  absolute_utility_gate_rejected: 'Sessions in which the Top-1 absolute Utility fell below the active hysteresis floor and the portfolio moved or stayed in CASH.',
  absolute_utility_entry_threshold: 'Absolute Top-1 Utility required to move from CASH back into the market. Model Tuning can search this threshold while the LightGBM Champion remains frozen.',
  absolute_utility_exit_threshold: 'Absolute Top-1 Utility floor used while already invested. It is lower than or equal to the entry threshold to reduce unnecessary CASH churn.',
  opportunity_gate_decisions: 'Out-of-sample sessions evaluated by the Opportunity Cash Gate. The gate does not choose the asset; it decides whether the protected base rotation policy may keep capital exposed or must hold CASH.',
  opportunity_gate_accepted: 'Sessions in which the calibrated Opportunity Cash Gate allowed the protected base rotation policy to remain active.',
  opportunity_gate_rejected: 'Sessions in which the calibrated Opportunity Cash Gate rejected risky exposure and moved or kept the portfolio in CASH.',
  opportunity_entry_threshold_mean: 'Average adaptive absolute probability threshold required to enter the market from CASH. Opportunity Cash Gate v2 starts from pre-test calibration and may refresh every 21 matured out-of-sample sessions without using future labels.',
  opportunity_exit_threshold_mean: 'Average adaptive absolute probability threshold below which an exposed portfolio exits to CASH. It is lower than or equal to the entry threshold to create hysteresis, and v2 falls back to the protected B0 policy when CASH has no conservative validation alpha.',
  opportunity_gate_adaptive_refreshes: 'Number of no-look-ahead Opportunity Cash Gate v2 recalibrations triggered after 21 newly matured out-of-sample B0 outcomes.',
  opportunity_gate_regularized_sessions: 'Out-of-sample sessions in which the current calibration found insufficient conservative evidence for CASH and therefore regularized back to the protected B0 market policy.',
  opportunity_target_horizon_sessions: 'Cash Gate v2 target horizon. A value of 1 means the gate learns whether the protected B0 risky action produced positive net open-to-close growth in the next execution session.',
  cash_gate_changed_base_action_sessions: 'Sessions in which the Cash Gate actually overrode a non-CASH action that the protected base rotation policy would otherwise have taken.',
  cash_gate_counterfactual_negative_sessions: 'Gate overrides whose protected base action had a negative next-session open-to-close return. This is a diagnostic counterfactual, not an exact portfolio P/L attribution.',
  cash_gate_counterfactual_positive_sessions: 'Gate overrides whose protected base action had a positive next-session open-to-close return. This is a diagnostic counterfactual, not an exact portfolio P/L attribution.',
  cash_gate_avoided_loss_return_sum: 'Sum of the absolute negative next-session open-to-close returns that the protected base action would have experienced on Cash Gate override sessions. Diagnostic only; it is not exact portfolio P/L.',
  cash_gate_missed_gain_return_sum: 'Sum of the positive next-session open-to-close returns forgone on Cash Gate override sessions. Diagnostic only; it is not exact portfolio P/L.',
  cash_gate_net_avoided_return_sum: 'Avoided-loss diagnostic return minus missed-gain diagnostic return across Cash Gate overrides. Positive is favorable, but this is not a compounded portfolio return.',
  total_rotations: 'Number of executed capital movements: asset-to-asset rotations, exits to CASH, and entries from CASH into the market.',
  profitable_rotations: 'Completed rotations whose closed position produced positive realized profit.',
  total_realized_pnl: 'Sum of realized profit and loss from positions closed during capital rotations.',
  average_holding_days: 'Average number of trading sessions the portfolio remained in a position before rotating out of it.',
}

export const HISTORY_HINTS = {
  created_at: 'When this backtest execution was created.',
  strategy_profile_name: 'Public display name of the strategy profile used by the backtest. Protected parameters remain server-side.',
  status: 'Current or terminal execution state for the backtest.',
  simulation_return: 'Total return produced by the simulated strategy over the test period.',
  sharpe: 'Risk-adjusted return of the simulation.',
  maximum_drawdown: 'Largest peak-to-trough decline during the simulation.',
  position_changes: 'Number of capital rotations recorded by the simulation.',
  duration_seconds: 'Wall-clock execution time of the backtest job.',
}

export const ROTATION_HINTS = {
  executed_at: 'Timestamp when the simulated capital switch was executed.',
  from_asset: 'Asset that was exited. This is the sell side of the rotation.',
  to_asset: 'Asset entered after the exit. This is the buy side of the rotation.',
  holding_days: 'Number of trading sessions the exited position was held.',
  position_return: 'Return of the position that was closed by this rotation.',
  realized_pnl: 'Profit or loss realized when the previous position was closed.',
  transaction_fees: 'Transaction costs attributed to the completed rotation.',
}
