import { tr } from '../../../i18n/runtime'
import { useEffect, useMemo, useState } from 'react'
import { apiFetch, downloadFile } from '../../../api/http'
import { hasCapability } from '../../../auth/capabilities'
import { API } from '../../../config/env'
import {
  BacktestIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  ListFilterIcon,
  PlayIcon,
  SearchIcon,
  SortIcon,
  TrendDownIcon,
  TrendUpIcon,
} from '../../../shared/components/Icons'
import { ParameterHint } from '../../../shared/components/ParameterHint'
import { durationLabel, money, percent, shortDateTime } from '../../../shared/formatters'
import { ExecutionStatus } from './ExecutionStatus'
import { ModelTuningPanel } from '../../ModelTuningPanel'
import { TemporalIntelligencePanel } from '../../TemporalIntelligencePanel'
import { HISTORY_HINTS, HISTORY_PAGE_SIZE, METRIC_HINTS } from '../backtestConfig'
import { sortRows, toggleSort } from '../backtestUtils'
import { FilterButton, ListToolbar, Metric, MetricLabel, Pagination, SortableHeader, StatusBadge } from './BacktestPrimitives'
import { RotationPanel } from './RotationPanel'

export function BacktestPage({ workspace, capabilities = {}, onSessionExpired }) {
  const canExportResults = hasCapability(capabilities, 'backtest.export')
  const canViewResearchModels = hasCapability(capabilities, 'research_models.view')
  const canStartBacktest = hasCapability(capabilities, 'backtest.start')
  const canViewTuning = hasCapability(capabilities, 'tuning.view')
  const canViewTemporalIntelligence = hasCapability(capabilities, 'temporal_intelligence.view')
  const {
    job,
    dashboard,
    detail,
    loadingDetail,
    running,
    restoringExecution,
    startingBacktest,
    startDisabled,
    runBacktest,
    refreshDashboard,
  } = workspace
  const metrics = detail?.metrics || {}
  const [rotationPayload, setRotationPayload] = useState(null)
  const [rotationLoading, setRotationLoading] = useState(false)
  const [rotationError, setRotationError] = useState('')
  const [exporting, setExporting] = useState(false)
  const [exportError, setExportError] = useState('')
  const [historyQuery, setHistoryQuery] = useState('')
  const [historyStatus, setHistoryStatus] = useState('all')
  const [historySort, setHistorySort] = useState({ key: 'created_at', direction: 'desc' })
  const [historyPage, setHistoryPage] = useState(1)
  const [selectedStrategyModel, setSelectedStrategyModel] = useState(null)
  const [researchWorkspaceMode, setResearchWorkspaceMode] = useState('simulation')
  const [researchLabMode, setResearchLabMode] = useState(canViewTuning ? 'tuning' : 'temporal')
  const [temporalTuningStrategy, setTemporalTuningStrategy] = useState(null)
  const [strategyContextError, setStrategyContextError] = useState('')
  const [researchExecutionModels, setResearchExecutionModels] = useState({})
  const researchWorkspaceActive = researchWorkspaceMode === 'research'
  const selectedStrategyName = dashboard?.selected_strategy_research_name || dashboard?.selected_backtest_strategy_name || tr('Not selected')
  const activeStrategyName = (running ? job?.strategy_profile_name : null) || selectedStrategyName

  useEffect(() => {
    if (researchLabMode === 'tuning' && !canViewTuning && canViewTemporalIntelligence) setResearchLabMode('temporal')
    if (researchLabMode === 'temporal' && !canViewTemporalIntelligence && canViewTuning) setResearchLabMode('tuning')
  }, [canViewTemporalIntelligence, canViewTuning, researchLabMode])

  useEffect(() => {
    let active = true
    if (!canViewResearchModels) {
      setSelectedStrategyModel(null)
      setStrategyContextError('')
      return () => { active = false }
    }

    apiFetch(`${API}/admin/strategies/control`)
      .then((value) => {
        if (!active) return
        setSelectedStrategyModel(value?.research_strategy?.research_model || null)
        setStrategyContextError('')
      })
      .catch((requestError) => {
        if (!active) return
        setSelectedStrategyModel(null)
        if (requestError?.status === 403) {
          setStrategyContextError('')
          return
        }
        setStrategyContextError(requestError.message || 'Unable to load the model saved with the selected Strategy.')
      })
    return () => { active = false }
  }, [canViewResearchModels, dashboard?.selected_strategy_research_name, dashboard?.selected_backtest_strategy_name])

  useEffect(() => {
    let active = true
    if (!canViewResearchModels) {
      setResearchExecutionModels({})
      return () => { active = false }
    }

    apiFetch(`${API}/admin/model-research/executions?limit=50`)
      .then((value) => {
        if (!active) return
        const items = Array.isArray(value?.items) ? value.items : []
        setResearchExecutionModels(Object.fromEntries(items.filter((item) => item?.id).map((item) => [item.id, item])))
      })
      .catch(() => {
        if (active) setResearchExecutionModels({})
      })
    return () => { active = false }
  }, [canViewResearchModels, job?.id, detail?.id, dashboard?.recent_backtests?.[0]?.id])

  useEffect(() => {
    let active = true
    const jobId = detail?.id
    if (!jobId) {
      setRotationPayload(null)
      setRotationError('')
      setRotationLoading(false)
      return () => { active = false }
    }

    setRotationLoading(true)
    setRotationPayload(null)
    setRotationError('')
    apiFetch(`${API}/analytics/backtests/${encodeURIComponent(jobId)}`)
      .then((value) => {
        if (active) setRotationPayload(value)
      })
      .catch((requestError) => {
        if (active) {
          setRotationPayload(null)
          setRotationError(tr(requestError.message || 'Unable to load capital rotations.'))
        }
      })
      .finally(() => {
        if (active) setRotationLoading(false)
      })

    return () => { active = false }
  }, [detail?.id])

  const historyRows = useMemo(() => {
    const normalizedQuery = historyQuery.trim().toLowerCase()
    const rows = (dashboard?.recent_backtests || []).map((item) => ({
      ...item,
      research_model_label: canViewResearchModels ? (researchExecutionModels[item.id]?.model_label || 'Baseline') : '',
    })).filter((item) => {
      if (historyStatus !== 'all' && String(item.status || '').toLowerCase() !== historyStatus) return false
      if (!normalizedQuery) return true
      const haystack = `${item.strategy_profile_name || 'Unknown test'} ${canViewResearchModels ? item.research_model_label : ''}`.toLowerCase()
      return haystack.includes(normalizedQuery)
    })
    return sortRows(rows, historySort, {
      created_at: (item) => Date.parse(item.created_at || '') || 0,
      strategy_profile_name: (item) => String(item.strategy_profile_name || 'Unknown test'),
      research_model_label: (item) => String(item.research_model_label || 'Baseline'),
      status: (item) => String(item.status || ''),
      simulation_return: (item) => item.metrics?.simulation_return == null ? null : Number(item.metrics.simulation_return),
      sharpe: (item) => item.metrics?.sharpe == null ? null : Number(item.metrics.sharpe),
      maximum_drawdown: (item) => item.metrics?.maximum_drawdown == null ? null : Number(item.metrics.maximum_drawdown),
      position_changes: (item) => item.metrics?.position_changes == null ? null : Number(item.metrics.position_changes),
      duration_seconds: (item) => item.duration_seconds == null ? null : Number(item.duration_seconds),
    })
  }, [canViewResearchModels, dashboard, historyQuery, historySort, historyStatus, researchExecutionModels])

  const historyPages = Math.max(1, Math.ceil(historyRows.length / HISTORY_PAGE_SIZE))
  const currentHistoryPage = Math.min(historyPage, historyPages)
  const paginatedHistoryRows = historyRows.slice((currentHistoryPage - 1) * HISTORY_PAGE_SIZE, currentHistoryPage * HISTORY_PAGE_SIZE)

  useEffect(() => {
    refreshDashboard()
  }, [refreshDashboard])

  useEffect(() => { setHistoryPage(1) }, [historyQuery, historySort, historyStatus])

  const savedResearchModelLabel = canViewResearchModels ? (selectedStrategyModel?.label || '') : ''
  const activeResearchModelLabel = canViewResearchModels && job?.id ? (researchExecutionModels[job.id]?.model_label || savedResearchModelLabel) : ''
  const displayedResearchModelLabel = canViewResearchModels && detail?.id ? (researchExecutionModels[detail.id]?.model_label || '') : ''
  const historyColumnCount = canViewResearchModels ? 9 : 8

  async function exportResults() {
    if (!canExportResults || !detail?.id || exporting) return
    setExporting(true)
    setExportError('')
    try {
      await downloadFile(
        `${API}/jobs/${encodeURIComponent(detail.id)}/export.zip`,
        `market_cycle_trader_${detail.id}.zip`,
      )
    } catch (requestError) {
      setExportError(requestError.message || 'Unable to export the result.')
    } finally {
      setExporting(false)
    }
  }

  return (
    <section className="page-stack backtest-page backtest-single-workspace">
      <section className="data-panel backtest-workspace-panel">
        <header className="backtest-workspace-header">
          <div className="backtest-workspace-title">
            <div className="page-title-icon"><BacktestIcon size={20} /></div>
            <div>
              <h2>{tr("Backtest")}</h2>
              <div className="backtest-context-line" aria-live="polite">
                {researchWorkspaceActive && temporalTuningStrategy ? <>
                  <span>{tr('Tuning target')}:</span>
                  <strong title={temporalTuningStrategy.name || ''}>{temporalTuningStrategy.name || '—'}</strong>
                </> : <>
                  <span>{tr(running ? 'Evaluating' : 'Selected test')}:</span>
                  <strong title={activeStrategyName}>{activeStrategyName}</strong>
                  {!running && savedResearchModelLabel ? <><i>·</i><span>{tr('Saved model')}:</span><strong>{savedResearchModelLabel}</strong></> : null}
                  {running && activeResearchModelLabel ? <><i>·</i><span>{tr('Model')}:</span><strong>{activeResearchModelLabel}</strong></> : null}
                  {!running && detail?.strategy_profile_name ? <><i>·</i><span>{tr("Displayed:")}</span><strong title={detail.strategy_profile_name}>{detail.strategy_profile_name}</strong>{displayedResearchModelLabel ? <><i>·</i><span>{tr('Model')}:</span><strong>{displayedResearchModelLabel}</strong></> : null}</> : null}
                </>}
              </div>
            </div>
          </div>
          <div className="backtest-workspace-actions">
            {(canViewResearchModels || canViewTuning || canViewTemporalIntelligence) ? (
              <div className="backtest-research-mode-switch" role="tablist" aria-label={tr('Research workspace')}>
                <button type="button" role="tab" aria-selected={researchWorkspaceMode === 'simulation'} className={researchWorkspaceMode === 'simulation' ? 'active' : ''} onClick={() => setResearchWorkspaceMode('simulation')}>{tr('Simulation Backtest')}</button>
                {(canViewTuning || canViewTemporalIntelligence) ? <button type="button" role="tab" aria-selected={researchWorkspaceMode === 'research'} className={researchWorkspaceMode === 'research' ? 'active' : ''} onClick={() => setResearchWorkspaceMode('research')}>{tr('Research Lab')}</button> : null}
              </div>
            ) : null}
            {canViewResearchModels && researchWorkspaceMode === 'simulation' ? (
              <div className="research-model-control research-model-readonly" aria-label={tr('Model saved with selected Strategy')}>
                <span>{tr('Saved model')}</span>
                <strong>{savedResearchModelLabel || tr('Unavailable')}</strong>
                <small>{tr('Defined in Selected Strategy')}</small>
              </div>
            ) : null}
            {researchWorkspaceMode === 'simulation' && canExportResults && detail?.metrics ? (
              <button type="button" className="secondary-action compact" onClick={exportResults} disabled={exporting}>
                {tr(exporting ? 'Exporting…' : 'Export Results')}
              </button>
            ) : null}
            {researchWorkspaceMode === 'simulation' && canStartBacktest ? <button type="button" className="primary-action compact" onClick={() => runBacktest()} disabled={startDisabled}>
              <PlayIcon /> {tr(restoringExecution ? 'Checking Execution' : startingBacktest ? 'Starting…' : running ? 'Simulation Running' : 'Start New Backtest')}
            </button> : null}
          </div>
        </header>

        {researchWorkspaceActive ? (
          <div className="backtest-research-lab-stack">
            {canViewTuning && canViewTemporalIntelligence ? (
              <div className="backtest-research-lab-mode-switch backtest-research-mode-switch" role="tablist" aria-label={tr('Research Lab')}>
                <button type="button" role="tab" aria-selected={researchLabMode === 'tuning'} className={researchLabMode === 'tuning' ? 'active' : ''} onClick={() => setResearchLabMode('tuning')}>{tr('Model Tuning')}</button>
                <button type="button" role="tab" aria-selected={researchLabMode === 'temporal'} className={researchLabMode === 'temporal' ? 'active' : ''} onClick={() => setResearchLabMode('temporal')}>{tr('Temporal Intelligence')}</button>
              </div>
            ) : null}
            {canViewTuning && researchLabMode === 'tuning' ? (
              <ModelTuningPanel
                capabilities={capabilities}
                onSessionExpired={onSessionExpired}
                onStrategyModelSaved={(updated) => setSelectedStrategyModel(updated?.research_model_configuration || updated?.research_model || null)}
                onTuningContextChange={setTemporalTuningStrategy}
              />
            ) : null}
            {canViewTemporalIntelligence && researchLabMode === 'temporal' ? <TemporalIntelligencePanel capabilities={capabilities} onSessionExpired={onSessionExpired} tuningStrategy={temporalTuningStrategy} /> : null}
          </div>
        ) : <>
        <ExecutionStatus workspace={workspace} modelLabel={activeResearchModelLabel} />
        {strategyContextError ? <div className="global-inline-message error-inline backtest-workspace-message">{tr(strategyContextError)}</div> : null}
        {exportError ? <div className="global-inline-message error-inline backtest-workspace-message">{tr(exportError)}</div> : null}
        {loadingDetail ? <div className="backtest-loading-row">{tr("Loading simulation result…")}</div> : null}

        {detail?.metrics ? (
          <>
            <section className="backtest-workspace-metrics">
              <Metric id="hint-final-capital" label={tr("Final Capital")} value={money(metrics.ending_capital)} note={tr('Initial {value}', { value: money(metrics.starting_capital) })} tone="green" hint={METRIC_HINTS.ending_capital} />
              <Metric id="hint-reference-capital" label={tr("Reference Capital")} value={money(metrics.reference_ending_capital)} note={tr('{value} total return', { value: percent(metrics.reference_return) })} tone="blue" hint={METRIC_HINTS.reference_ending_capital} />
              <Metric id="hint-cagr" label={tr("CAGR")} value={percent(metrics.cagr)} note={tr('Reference {value}', { value: percent(metrics.reference_cagr) })} tone="purple" hint={METRIC_HINTS.cagr} />
              <Metric id="hint-sharpe" label={tr("Sharpe Ratio")} value={metrics.sharpe == null ? '—' : Number(metrics.sharpe).toFixed(3)} note={tr('Reference {value}', { value: metrics.reference_sharpe == null ? '—' : Number(metrics.reference_sharpe).toFixed(3) })} tone="green" hint={METRIC_HINTS.sharpe} />
              <Metric id="hint-max-drawdown" label={tr("Max Drawdown")} value={percent(metrics.maximum_drawdown)} note={tr('Reference {value}', { value: percent(metrics.reference_maximum_drawdown) })} tone="red" hint={METRIC_HINTS.maximum_drawdown} />
              <Metric id="hint-session-win-rate" label={tr("Session Win Rate")} value={percent(metrics.session_win_rate)} note={tr('{value} market exposure', { value: percent(metrics.market_exposure) })} tone="blue" hint={METRIC_HINTS.session_win_rate} />
            </section>
            {metrics.average_cash_weight != null ? (
              <section className="backtest-workspace-metrics">
                <Metric id="hint-average-cash-weight" label={tr("Average CASH Weight")} value={percent(metrics.average_cash_weight)} note={tr('{value} market exposure', { value: percent(metrics.market_exposure) })} tone="blue" hint={METRIC_HINTS.average_cash_weight} />
                <Metric id="hint-average-assets-held" label={tr("Average Assets Held")} value={metrics.average_assets_held == null ? '—' : Number(metrics.average_assets_held).toFixed(2)} note={tr('Simultaneous risky positions')} tone="green" hint={METRIC_HINTS.average_assets_held} />
                <Metric id="hint-maximum-assets-held" label={tr("Maximum Assets Held")} value={metrics.maximum_assets_held == null ? '—' : Number(metrics.maximum_assets_held).toFixed(0)} note={tr('Largest simultaneous allocation')} tone="purple" hint={METRIC_HINTS.maximum_assets_held} />
                <Metric id="hint-allocation-rebalances" label={tr("Allocation Rebalances")} value={metrics.allocation_rebalances == null ? '—' : Number(metrics.allocation_rebalances).toFixed(0)} note={tr('Capital movement sessions')} tone="blue" hint={METRIC_HINTS.allocation_rebalances} />
                {metrics.compound_risk_overlay_enabled ? <Metric id="hint-risk-overlay-decisions" label={tr("Risk Overlay Decisions")} value={Number(metrics.risk_overlay_decisions || 0).toFixed(0)} note={tr('Top-1/CASH risk sizing')} tone="blue" hint={METRIC_HINTS.risk_overlay_decisions} /> : null}
                {metrics.compound_risk_overlay_enabled ? <Metric id="hint-risk-overlay-full" label={tr("Risk Overlay Full Exposure")} value={Number(metrics.risk_overlay_full_exposure_decisions || 0).toFixed(0)} note={tr('Capital compound preserved')} tone="green" hint={METRIC_HINTS.risk_overlay_full_exposure_decisions} /> : null}
                {metrics.compound_risk_overlay_enabled ? <Metric id="hint-risk-overlay-reduced" label={tr("Risk Overlay Reduced Exposure")} value={Number(metrics.risk_overlay_reduced_exposure_decisions || 0).toFixed(0)} note={tr('Part of capital moved to CASH')} tone="blue" hint={METRIC_HINTS.risk_overlay_reduced_exposure_decisions} /> : null}
                {metrics.compound_risk_overlay_enabled ? <Metric id="hint-risk-overlay-fallbacks" label={tr("Risk Overlay Technical Fallbacks")} value={Number(metrics.risk_overlay_technical_fallbacks || 0).toFixed(0)} note={tr('Must be zero in a clean run')} tone={Number(metrics.risk_overlay_technical_fallbacks || 0) > 0 ? "red" : "green"} hint={METRIC_HINTS.risk_overlay_technical_fallbacks} /> : null}
                {metrics.average_primary_weight != null ? <Metric id="hint-average-primary-weight" label={tr("Average Top-1 Weight")} value={percent(metrics.average_primary_weight)} note={tr('Share of total compounded capital')} tone="green" hint={METRIC_HINTS.average_primary_weight} /> : null}
                {metrics.average_primary_share_of_risk != null ? <Metric id="hint-average-primary-share" label={tr("Top-1 Share of Risk")} value={percent(metrics.average_primary_share_of_risk)} note={tr('Share of risky capital excluding CASH')} tone="purple" hint={METRIC_HINTS.average_primary_share_of_risk} /> : null}
                {metrics.average_secondary_weight != null ? <Metric id="hint-average-secondary-weight" label={tr("Average Secondary Weight")} value={percent(metrics.average_secondary_weight)} note={tr('Optional Top-2/Top-3 capital')} tone="blue" hint={METRIC_HINTS.average_secondary_weight} /> : null}
              </section>
            ) : null}
            {metrics.absolute_utility_cash_gate_enabled ? (
              <section className="backtest-workspace-metrics">
                <Metric id="hint-absolute-utility-decisions" label={tr("Absolute Utility Decisions")} value={Number(metrics.absolute_utility_gate_decisions || 0).toFixed(0)} note={tr('{value} market exposure', { value: percent(metrics.market_exposure) })} tone="blue" hint={METRIC_HINTS.absolute_utility_gate_decisions} />
                <Metric id="hint-absolute-utility-accepted" label={tr("Market Accepted")} value={Number(metrics.absolute_utility_gate_accepted || 0).toFixed(0)} note={tr('{value} acceptance rate', { value: percent(metrics.absolute_utility_gate_acceptance_rate) })} tone="green" hint={METRIC_HINTS.absolute_utility_gate_accepted} />
                <Metric id="hint-absolute-utility-rejected" label={tr("CASH Rejected")} value={Number(metrics.absolute_utility_gate_rejected || 0).toFixed(0)} note={tr('{count} CASH sessions', { count: Number(metrics.cash_days || 0).toFixed(0) })} tone="blue" hint={METRIC_HINTS.absolute_utility_gate_rejected} />
                <Metric id="hint-absolute-utility-entry" label={tr("Utility Entry Floor")} value={metrics.absolute_utility_entry_threshold == null ? '—' : Number(metrics.absolute_utility_entry_threshold).toFixed(6)} note={tr('CASH → market threshold')} tone="green" hint={METRIC_HINTS.absolute_utility_entry_threshold} />
                <Metric id="hint-absolute-utility-exit" label={tr("Utility Exit Floor")} value={metrics.absolute_utility_exit_threshold == null ? '—' : Number(metrics.absolute_utility_exit_threshold).toFixed(6)} note={tr('Market → CASH threshold')} tone="blue" hint={METRIC_HINTS.absolute_utility_exit_threshold} />
                <Metric id="hint-absolute-cash-overrides" label={tr("Cash Gate Overrides")} value={Number(metrics.cash_gate_changed_base_action_sessions || 0).toFixed(0)} note={tr('{entries} entries · {exits} exits', { entries: Number(metrics.cash_gate_entries || 0).toFixed(0), exits: Number(metrics.cash_gate_exits || 0).toFixed(0) })} tone="purple" hint={METRIC_HINTS.cash_gate_changed_base_action_sessions} />
                <Metric id="hint-absolute-cash-avoided" label={tr("Avoided-Loss Sessions")} value={Number(metrics.cash_gate_counterfactual_negative_sessions || 0).toFixed(0)} note={tr('Base action fell next session')} tone="green" hint={METRIC_HINTS.cash_gate_counterfactual_negative_sessions} />
                <Metric id="hint-absolute-cash-missed" label={tr("Missed-Gain Sessions")} value={Number(metrics.cash_gate_counterfactual_positive_sessions || 0).toFixed(0)} note={tr('Base action rose next session')} tone="red" hint={METRIC_HINTS.cash_gate_counterfactual_positive_sessions} />
                <Metric id="hint-absolute-cash-net" label={tr("Net Cash-Gate Diagnostic")} value={percent(metrics.cash_gate_net_avoided_return_sum)} note={tr('Avoided loss minus missed gain')} tone={Number(metrics.cash_gate_net_avoided_return_sum || 0) >= 0 ? "green" : "red"} hint={METRIC_HINTS.cash_gate_net_avoided_return_sum} />
              </section>
            ) : null}
            {metrics.opportunity_cash_gate_enabled ? (
              <section className="backtest-workspace-metrics">
                <Metric id="hint-cash-gate-decisions" label={tr("Cash Gate Decisions")} value={Number(metrics.opportunity_gate_decisions || 0).toFixed(0)} note={tr('{value} market exposure', { value: percent(metrics.market_exposure) })} tone="blue" hint={METRIC_HINTS.opportunity_gate_decisions} />
                <Metric id="hint-cash-gate-accepted" label={tr("Cash Gate Accepted")} value={Number(metrics.opportunity_gate_accepted || 0).toFixed(0)} note={tr('{value} acceptance rate', { value: percent(metrics.opportunity_gate_acceptance_rate) })} tone="green" hint={METRIC_HINTS.opportunity_gate_accepted} />
                <Metric id="hint-cash-gate-rejected" label={tr("Cash Gate Rejected")} value={Number(metrics.opportunity_gate_rejected || 0).toFixed(0)} note={tr('{count} CASH sessions', { count: Number(metrics.cash_days || 0).toFixed(0) })} tone="blue" hint={METRIC_HINTS.opportunity_gate_rejected} />
                <Metric id="hint-cash-gate-entry-threshold" label={tr("Entry Growth Probability")} value={percent(metrics.opportunity_entry_threshold_mean)} note={tr('CASH → market threshold')} tone="green" hint={METRIC_HINTS.opportunity_entry_threshold_mean} />
                <Metric id="hint-cash-gate-exit-threshold" label={tr("Exit Growth Probability")} value={percent(metrics.opportunity_exit_threshold_mean)} note={tr('Market → CASH threshold')} tone="blue" hint={METRIC_HINTS.opportunity_exit_threshold_mean} />
                <Metric id="hint-cash-gate-refreshes" label={tr("Adaptive Gate Refreshes")} value={Number(metrics.opportunity_gate_adaptive_refreshes || 0).toFixed(0)} note={tr('Every 21 matured OOS sessions')} tone="purple" hint={METRIC_HINTS.opportunity_gate_adaptive_refreshes} />
                <Metric id="hint-cash-gate-b0-prior" label={tr("B0 Prior Sessions")} value={Number(metrics.opportunity_gate_regularized_sessions || 0).toFixed(0)} note={tr('CASH lacked conservative alpha')} tone="green" hint={METRIC_HINTS.opportunity_gate_regularized_sessions} />
                <Metric id="hint-cash-gate-target-horizon" label={tr("Gate Target Horizon")} value={metrics.opportunity_target_horizon_sessions == null ? '—' : Number(metrics.opportunity_target_horizon_sessions).toFixed(0)} note={tr('Next execution sessions')} tone="blue" hint={METRIC_HINTS.opportunity_target_horizon_sessions} />
                <Metric id="hint-cash-gate-overrides" label={tr("Cash Gate Overrides")} value={Number(metrics.cash_gate_changed_base_action_sessions || 0).toFixed(0)} note={tr('{entries} entries · {exits} exits', { entries: Number(metrics.cash_gate_entries || 0).toFixed(0), exits: Number(metrics.cash_gate_exits || 0).toFixed(0) })} tone="purple" hint={METRIC_HINTS.cash_gate_changed_base_action_sessions} />
                <Metric id="hint-cash-gate-avoided" label={tr("Avoided-Loss Sessions")} value={Number(metrics.cash_gate_counterfactual_negative_sessions || 0).toFixed(0)} note={tr('Base action fell next session')} tone="green" hint={METRIC_HINTS.cash_gate_counterfactual_negative_sessions} />
                <Metric id="hint-cash-gate-missed" label={tr("Missed-Gain Sessions")} value={Number(metrics.cash_gate_counterfactual_positive_sessions || 0).toFixed(0)} note={tr('Base action rose next session')} tone="red" hint={METRIC_HINTS.cash_gate_counterfactual_positive_sessions} />
                <Metric id="hint-cash-gate-avoided-return" label={tr("Avoided-Loss Diagnostic")} value={percent(metrics.cash_gate_avoided_loss_return_sum)} note={tr('Non-compounded counterfactual sum')} tone="green" hint={METRIC_HINTS.cash_gate_avoided_loss_return_sum} />
                <Metric id="hint-cash-gate-missed-return" label={tr("Missed-Gain Diagnostic")} value={percent(metrics.cash_gate_missed_gain_return_sum)} note={tr('Non-compounded counterfactual sum')} tone="red" hint={METRIC_HINTS.cash_gate_missed_gain_return_sum} />
                <Metric id="hint-cash-gate-net-return" label={tr("Net Cash-Gate Diagnostic")} value={percent(metrics.cash_gate_net_avoided_return_sum)} note={tr('Avoided loss minus missed gain')} tone={Number(metrics.cash_gate_net_avoided_return_sum || 0) >= 0 ? "green" : "red"} hint={METRIC_HINTS.cash_gate_net_avoided_return_sum} />
              </section>
            ) : null}

            <section className="backtest-workspace-main">
              <article className="backtest-results-section">
                <div className="backtest-section-heading compact">
                  <div><span className="panel-kicker">{tr("Summary")}</span><h2>{tr("Backtest Results")}</h2></div>
                </div>
                <div className="backtest-result-columns"><span>{tr("Metric")}</span><span>{tr("Simulation")}</span><span>{tr("Reference")}</span></div>
                <dl className="result-comparison-list backtest-result-list">
                  <div><dt><MetricLabel id="hint-total-return" label={tr("Total return")} hint="Total percentage change over the complete test period." /></dt><dd>{percent(metrics.simulation_return)}</dd><dd>{percent(metrics.reference_return)}</dd></div>
                  <div><dt><MetricLabel id="hint-result-cagr" label={tr("CAGR")} hint={METRIC_HINTS.cagr} /></dt><dd>{percent(metrics.cagr)}</dd><dd>{percent(metrics.reference_cagr)}</dd></div>
                  <div><dt><MetricLabel id="hint-result-sharpe" label={tr("Sharpe ratio")} hint={METRIC_HINTS.sharpe} /></dt><dd>{metrics.sharpe == null ? '—' : Number(metrics.sharpe).toFixed(3)}</dd><dd>{metrics.reference_sharpe == null ? '—' : Number(metrics.reference_sharpe).toFixed(3)}</dd></div>
                  <div><dt><MetricLabel id="hint-result-drawdown" label={tr("Max drawdown")} hint={METRIC_HINTS.maximum_drawdown} /></dt><dd>{percent(metrics.maximum_drawdown)}</dd><dd>{percent(metrics.reference_maximum_drawdown)}</dd></div>
                  <div><dt><MetricLabel id="hint-result-rotations" label={tr("Capital rotations")} hint={METRIC_HINTS.total_rotations} /></dt><dd>{metrics.position_changes == null ? '—' : Math.round(metrics.position_changes)}</dd><dd>—</dd></div>
                  <div><dt><MetricLabel id="hint-result-holding" label={tr("Avg. holding")} hint={METRIC_HINTS.average_holding_days} /></dt><dd>{metrics.average_holding_days == null ? '—' : tr('{count} days', { count: Number(metrics.average_holding_days).toFixed(1) })}</dd><dd>—</dd></div>
                </dl>
              </article>
            </section>

            <RotationPanel jobId={detail.id} payload={rotationPayload} loading={rotationLoading} error={rotationError} />
          </>
        ) : (
          <section className="backtest-workspace-section empty-result backtest-empty-result">
            <BacktestIcon size={32} />
            <h2>{tr("No completed result selected")}</h2>
          </section>
        )}

        <section className="backtest-workspace-section backtest-history-section">
          <div className="backtest-section-heading">
            <div><span className="panel-kicker">{tr("History")}</span><h2>{tr("Backtest History")}</h2></div>
            <span className="backtest-section-meta">{tr("Latest")}{' '}{dashboard?.recent_backtests?.length || 0} {tr("executions")}</span>
          </div>

          <ListToolbar
            query={historyQuery}
            onQueryChange={setHistoryQuery}
            placeholder={tr("Filter by test or model")}
            resultCount={historyRows.length}
            resultLabel={historyRows.length === 1 ? 'execution' : 'executions'}
          >
            <FilterButton active={historyStatus === 'all'} label={tr("All")} onClick={() => setHistoryStatus('all')}><ListFilterIcon size={14} /></FilterButton>
            <FilterButton active={historyStatus === 'completed'} label={tr("Completed")} tone="positive" onClick={() => setHistoryStatus('completed')} />
            <FilterButton active={historyStatus === 'failed'} label={tr("Failed")} tone="negative" onClick={() => setHistoryStatus('failed')} />
            <FilterButton active={historyStatus === 'interrupted'} label={tr("Interrupted")} onClick={() => setHistoryStatus('interrupted')} />
          </ListToolbar>

          <div className="table-wrap backtest-table-wrap">
            <table className="dashboard-table backtest-sortable-table">
              <thead>
                <tr>
                  <SortableHeader label={tr("Date")} field="created_at" sort={historySort} onSort={(key) => setHistorySort((current) => toggleSort(current, key))} hint={HISTORY_HINTS.created_at} />
                  <SortableHeader label={tr("Test")} field="strategy_profile_name" sort={historySort} onSort={(key) => setHistorySort((current) => toggleSort(current, key))} hint={HISTORY_HINTS.strategy_profile_name} />
                  {canViewResearchModels ? <SortableHeader label={tr("Model")} field="research_model_label" sort={historySort} onSort={(key) => setHistorySort((current) => toggleSort(current, key))} /> : null}
                  <SortableHeader label={tr("Status")} field="status" sort={historySort} onSort={(key) => setHistorySort((current) => toggleSort(current, key))} hint={HISTORY_HINTS.status} />
                  <SortableHeader label={tr("Total Return")} field="simulation_return" sort={historySort} onSort={(key) => setHistorySort((current) => toggleSort(current, key))} hint={HISTORY_HINTS.simulation_return} />
                  <SortableHeader label={tr("Sharpe Ratio")} field="sharpe" sort={historySort} onSort={(key) => setHistorySort((current) => toggleSort(current, key))} hint={HISTORY_HINTS.sharpe} />
                  <SortableHeader label={tr("Max Drawdown")} field="maximum_drawdown" sort={historySort} onSort={(key) => setHistorySort((current) => toggleSort(current, key))} hint={HISTORY_HINTS.maximum_drawdown} />
                  <SortableHeader label={tr("Rotations")} field="position_changes" sort={historySort} onSort={(key) => setHistorySort((current) => toggleSort(current, key))} hint={HISTORY_HINTS.position_changes} />
                  <SortableHeader label={tr("Duration")} field="duration_seconds" sort={historySort} onSort={(key) => setHistorySort((current) => toggleSort(current, key))} hint={HISTORY_HINTS.duration_seconds} />
                </tr>
              </thead>
              <tbody>
                {paginatedHistoryRows.length ? paginatedHistoryRows.map((item) => (
                  <tr key={item.id} className={detail?.id === item.id ? 'selected-row' : ''}>
                    <td>{shortDateTime(item.created_at)}</td>
                    <td className="backtest-name-cell" title={item.strategy_profile_name || tr('Unknown test')}>{item.strategy_profile_name || tr('Unknown test')}</td>
                    {canViewResearchModels ? <td>{item.research_model_label || tr('Baseline')}</td> : null}
                    <td><StatusBadge status={item.status} /></td>
                    <td className={item.metrics?.simulation_return == null ? '' : Number(item.metrics.simulation_return) >= 0 ? 'positive' : 'negative'}>{percent(item.metrics?.simulation_return)}</td>
                    <td>{item.metrics?.sharpe == null ? '—' : Number(item.metrics.sharpe).toFixed(3)}</td>
                    <td className="negative">{percent(item.metrics?.maximum_drawdown)}</td>
                    <td>{item.metrics?.position_changes == null ? '—' : Math.round(item.metrics.position_changes)}</td>
                    <td>{durationLabel(item.duration_seconds)}</td>
                  </tr>
                )) : <tr><td colSpan={historyColumnCount} className="empty-cell">{tr("No backtest history matches the selected filters.")}</td></tr>}
              </tbody>
            </table>
          </div>
          <Pagination page={currentHistoryPage} pages={historyPages} total={historyRows.length} pageSize={HISTORY_PAGE_SIZE} onPageChange={setHistoryPage} />
        </section>
        </>}
      </section>
    </section>
  )
}
