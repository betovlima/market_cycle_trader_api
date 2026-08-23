import { useEffect, useMemo, useState } from 'react'

import { apiFetch } from '../../api/http'
import { API } from '../../config/env'
import { tr } from '../../i18n/runtime'
import { PlayIcon } from '../../shared/components/Icons'
import { number, percent, shortDateTime } from '../../shared/formatters'
import { MonthlyCapitalMovementHeatmap } from '../backtest/components/RotationPanel'
import { TemporalPolicySearchPanel } from './TemporalPolicySearchPanel'
import { WinnerTransitionAttributionPanel } from './WinnerTransitionAttributionPanel'
import { WinnerTransitionRiskSearchPanel } from './WinnerTransitionRiskSearchPanel'
import { WinnerTransitionInterventionSearchPanel } from './WinnerTransitionInterventionSearchPanel'
import { WinnerTransitionConfidenceCalibrationPanel } from './WinnerTransitionConfidenceCalibrationPanel'
import { WinnerTransitionStatefulReplayPanel } from './WinnerTransitionStatefulReplayPanel'
import {
  attachDecisionContexts,
  attachWinnerTransitionAttributions,
  attachMonthlyDecisionContexts,
  attachMonthlyWinnerTransitionAttributions,
  filterAnalyticsByPeriod,
  monthsWithData,
  periodFromSnapshot,
  periodIsValid,
  saveJsonZip,
  temporalStudyParameters,
} from './temporalStudyUtils'
import './temporalStudy.css'

function Parameter({ label, value }) {
  return <div className="temporal-study-parameter"><span>{label}</span><strong title={String(value || '')}>{value || '—'}</strong></div>
}

function pctValue(value) {
  return value == null ? '—' : percent(Number(value), 2)
}

function storedPeriodKey(runId, processingId) {
  if (!runId || !processingId) return ''
  return `mct.temporal-study.period.${processingId}.${runId}`
}

function readStoredPeriod(runId, processingId, fallback) {
  const key = storedPeriodKey(runId, processingId)
  if (!key || typeof window === 'undefined') return fallback
  try {
    const value = JSON.parse(window.localStorage.getItem(key) || 'null')
    if (periodIsValid(value?.start, value?.end)) return value
  } catch {
    return fallback
  }
  return fallback
}

function writeStoredPeriod(runId, processingId, value) {
  const key = storedPeriodKey(runId, processingId)
  if (!key || typeof window === 'undefined') return
  try {
    window.localStorage.setItem(key, JSON.stringify(value))
  } catch {
  }
}

export function TemporalStudyPanel({ run = null, processing = null, canRun = false, canExport = false, canMaterializeStrategy = false }) {
  const effectiveRun = run
  const defaults = useMemo(() => periodFromSnapshot(effectiveRun?.research_snapshot_cutoff || effectiveRun?.analysis_end_date), [effectiveRun?.analysis_end_date, effectiveRun?.research_snapshot_cutoff])
  const [startMonth, setStartMonth] = useState(defaults.start)
  const [endMonth, setEndMonth] = useState(defaults.end)
  const [study, setStudy] = useState(null)
  const [running, setRunning] = useState(false)
  const [loadingLatest, setLoadingLatest] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [error, setError] = useState('')
  const [policySearch, setPolicySearch] = useState(null)
  const [transitionRiskSearch, setTransitionRiskSearch] = useState(null)
  const [transitionInterventionSearch, setTransitionInterventionSearch] = useState(null)
  const [transitionConfidenceSearch, setTransitionConfidenceSearch] = useState(null)
  const [transitionStatefulReplay, setTransitionStatefulReplay] = useState(null)
  const [pipelineStage, setPipelineStage] = useState('')
  const [pipelineRefresh, setPipelineRefresh] = useState(0)

  async function fetchStudyData(periodStart, periodEnd, executedAt, { includeTransition = false } = {}) {
    const requests = [apiFetch(`${API}/analytics/processings/${encodeURIComponent(processing.id)}`)]
    if (effectiveRun?.id) {
      requests.push(apiFetch(`${API}/temporal-intelligence/${encodeURIComponent(effectiveRun.id)}/decision-context?start_month=${encodeURIComponent(periodStart)}&end_month=${encodeURIComponent(periodEnd)}`))
      if (includeTransition) requests.push(apiFetch(`${API}/temporal-intelligence/${encodeURIComponent(effectiveRun.id)}/winner-transition-attribution?start_month=${encodeURIComponent(periodStart)}&end_month=${encodeURIComponent(periodEnd)}`))
    }
    const [payload, decisionPayload, transitionPayload] = await Promise.all(requests)
    const decisionContexts = decisionPayload?.items || []
    const transitionAttributions = transitionPayload?.items || []
    const analytics = filterAnalyticsByPeriod(payload, periodStart, periodEnd)
    analytics.rotations = attachDecisionContexts(analytics.rotations || [], decisionContexts)
    analytics.rotations = attachWinnerTransitionAttributions(analytics.rotations || [], transitionAttributions)
    return {
      executed_at: executedAt || processing?.finished_at || effectiveRun?.finished_at || null,
      start_month: periodStart,
      end_month: periodEnd,
      parameters: temporalStudyParameters(effectiveRun, processing, periodStart, periodEnd),
      decision_context: decisionPayload || null,
      winner_transition_attribution: transitionPayload || null,
      analytics,
    }
  }

  useEffect(() => {
    let active = true
    if (!processing?.id || !effectiveRun?.id || String(effectiveRun?.status || '').toLowerCase() !== 'completed' || !effectiveRun?.result || !periodIsValid(defaults.start, defaults.end)) return () => { active = false }
    const stored = readStoredPeriod(effectiveRun.id, processing.id, defaults)
    setStartMonth(stored.start)
    setEndMonth(stored.end)
    setPolicySearch(null)
    setTransitionRiskSearch(null)
    setTransitionInterventionSearch(null)
    setTransitionConfidenceSearch(null)
    setTransitionStatefulReplay(null)
    setError('')
    setLoadingLatest(true)
    fetchStudyData(stored.start, stored.end, stored.executed_at)
      .then((value) => { if (active) setStudy(value) })
      .catch((requestError) => { if (active) setError(tr(requestError?.message || 'Unable to load the latest study result.')) })
      .finally(() => { if (active) setLoadingLatest(false) })
    return () => { active = false }
  }, [defaults.end, defaults.start, effectiveRun?.id, effectiveRun?.result, effectiveRun?.status, processing?.id])

  const validPeriod = periodIsValid(startMonth, endMonth)
  const multi = effectiveRun?.result?.multi_horizon_metrics || {}
  const capital = multi?.shadow_capital || {}
  const technique = effectiveRun?.result?.experiment === 'temporal_decision_intelligence_v8_winner_anchored_timing'
    ? tr('Winner-Anchored Temporal Timing')
    : (effectiveRun?.result?.model_label || effectiveRun?.experiment || tr('Temporal Intelligence'))
  const horizons = (effectiveRun?.horizons || effectiveRun?.result?.horizons || []).map((item) => `${item}d`).join(' · ')
  const displayedSettings = study?.policy_candidate?.settings || capital
  const timingThresholds = [
    displayedSettings?.timing_base_weak_threshold == null ? null : `base ${number(displayedSettings.timing_base_weak_threshold, 3)}`,
    displayedSettings?.timing_challenger_minimum == null ? null : `challenger ${number(displayedSettings.timing_challenger_minimum, 3)}`,
    displayedSettings?.timing_minimum_advantage == null ? null : `gap ${number(displayedSettings.timing_minimum_advantage, 3)}`,
  ].filter(Boolean).join(' · ')

  async function runStudyStep() {
    const value = await fetchStudyData(startMonth, endMonth, new Date().toISOString(), { includeTransition: true })
    setStudy(value)
    writeStoredPeriod(effectiveRun.id, processing.id, { start: startMonth, end: endMonth, executed_at: value?.executed_at || new Date().toISOString() })
    return value
  }


  async function executeFullAnalysis() {
    if (!canRun || running || !processing?.id || !effectiveRun?.id || String(effectiveRun?.status || '').toLowerCase() !== 'completed' || !effectiveRun?.result || !validPeriod) return
    setRunning(true)
    setError('')
    setTransitionRiskSearch(null)
    setTransitionInterventionSearch(null)
    setTransitionConfidenceSearch(null)
    setTransitionStatefulReplay(null)
    try {
      setPipelineStage('study')
      const value = await runStudyStep()
      const body = { processing_id: processing.id, start_month: startMonth, end_month: endMonth }

      setPipelineStage('risk')
      const risk = await apiFetch(`${API}/temporal-intelligence/${encodeURIComponent(effectiveRun.id)}/winner-transition-risk-search`, {
        method: 'POST',
        body: { ...body, seed: 42 },
      })
      setTransitionRiskSearch(risk)

      setPipelineStage('intervention')
      const intervention = await apiFetch(`${API}/temporal-intelligence/${encodeURIComponent(effectiveRun.id)}/winner-transition-intervention-search`, {
        method: 'POST',
        body: { ...body, seed: 42 },
      })
      setTransitionInterventionSearch(intervention)

      setPipelineStage('confidence')
      const confidence = await apiFetch(`${API}/temporal-intelligence/${encodeURIComponent(effectiveRun.id)}/winner-transition-confidence-calibration`, {
        method: 'POST',
        body,
      })
      setTransitionConfidenceSearch(confidence)

      setPipelineStage('stateful')
      const stateful = await apiFetch(`${API}/temporal-intelligence/${encodeURIComponent(effectiveRun.id)}/winner-transition-stateful-replay`, {
        method: 'POST',
        body,
      })
      setTransitionStatefulReplay(stateful)
      setPipelineRefresh((current) => current + 1)
    } catch (requestError) {
      setError(tr(requestError?.message || 'Unable to run the temporal analysis pipeline.'))
      setPipelineRefresh((current) => current + 1)
    } finally {
      setPipelineStage('')
      setRunning(false)
    }
  }


  async function exportStudy() {
    if (!canExport || exporting || !study || !processing?.id) return
    setExporting(true)
    setError('')
    try {
      let details = study.monthly_details || null
      if (!details) {
        details = {}
        for (const month of monthsWithData(study.analytics)) {
          const [year, monthNumber] = month.split('-')
          details[month] = await apiFetch(`${API}/analytics/processings/${encodeURIComponent(processing.id)}/rotation-period?year=${encodeURIComponent(year)}&month=${encodeURIComponent(Number(monthNumber))}`)
        }
      }
      const decisionContexts = study.decision_context?.items || []
      await saveJsonZip(
        `temporal_study_${study.policy_search_id || processing.id}_${study.start_month}_${study.end_month}.json`,
        {
          schema_version: 12,
          study: {
            executed_at: study.executed_at,
            source_kind: study.source_kind || 'temporal_control',
            parameters: study.parameters,
            policy_candidate: study.policy_candidate || null,
          },
          analytics: study.analytics,
          decision_context: study.decision_context,
          winner_transition_attribution: study.winner_transition_attribution || null,
          winner_transition_risk_search: transitionRiskSearch || null,
          winner_transition_intervention_search: transitionInterventionSearch || null,
          winner_transition_confidence_calibration: transitionConfidenceSearch || null,
          winner_transition_stateful_replay: transitionStatefulReplay || null,
          monthly_details: attachMonthlyWinnerTransitionAttributions(
            attachMonthlyDecisionContexts(details, decisionContexts),
            study.winner_transition_attribution?.items || [],
          ),
          policy_search: study.policy_search || policySearch,
        },
      )
    } catch (requestError) {
      setError(tr(requestError?.message || 'Unable to export the temporal study data.'))
    } finally {
      setExporting(false)
    }
  }

  return <section className="temporal-study-panel">
    <div className="temporal-study-heading">
      <div>
        <span className="panel-kicker">{tr('CURRENT STUDY')}</span>
        <h3>{tr('Temporal Timing Study')}</h3>
      </div>
      {study?.executed_at ? <span className="temporal-study-last-run">{tr('Study data')} · {shortDateTime(study.executed_at)}</span> : null}
    </div>

    <div className="temporal-study-parameters">
      <Parameter label={tr('Technique')} value={technique} />
      <Parameter label={tr('Model')} value={effectiveRun?.model_label || effectiveRun?.result?.model_label || 'LightGBM'} />
      <Parameter label={tr('Horizons')} value={horizons} />
      <Parameter label={tr('Timing thresholds')} value={timingThresholds} />
      <Parameter label={tr('One-side cost')} value={pctValue(capital?.one_side_cost_rate)} />
      <Parameter label={tr('Research snapshot')} value={effectiveRun?.research_snapshot_cutoff || effectiveRun?.analysis_end_date || '—'} />
      <Parameter label={tr('Temporal source')} value={effectiveRun?.id || '—'} />
      <Parameter label={tr('Source processing')} value={processing?.id || '—'} />
    </div>

    <div className="temporal-study-controls">
      <label>
        <span>{tr('Period from')}</span>
        <input type="month" value={startMonth} onChange={(event) => setStartMonth(event.target.value)} />
      </label>
      <label>
        <span>{tr('Period to')}</span>
        <input type="month" value={endMonth} onChange={(event) => setEndMonth(event.target.value)} />
      </label>
      <div className="temporal-study-actions">
        {canRun ? <button type="button" className="primary-action compact" onClick={executeFullAnalysis} disabled={running || !processing?.id || !effectiveRun?.id || !validPeriod}><PlayIcon />{tr(running ? 'Running full analysis…' : 'Run full analysis')}</button> : null}
      </div>
    </div>

    <div className="temporal-study-run-help">{running && pipelineStage ? `${tr('Current stage')}: ${tr({ study: 'Study', risk: 'Risk search', intervention: 'Intervention search', confidence: 'Confidence calibration', stateful: 'Stateful replay' }[pipelineStage] || pipelineStage)}` : tr('Changing the period only prepares the analysis. Execution starts only with Run full analysis.')}</div>
    {!validPeriod ? <div className="global-inline-message error-inline">{tr('Select a valid period.')}</div> : null}
    {!processing?.id ? <div className="global-inline-message error-inline">{tr('A completed Backtest processing is required for this study.')}</div> : null}
    {error ? <div className="global-inline-message error-inline">{error}</div> : null}

    <div className="temporal-study-result always-visible">
      {study?.analytics ? <>
        <div className="temporal-study-result-toolbar">
          <div className="temporal-study-result-context">
            <span>{tr('Displayed result')}</span><strong>{study.start_month} → {study.end_month}</strong>
            <i>·</i><span>{tr('Processing')}</span><strong>{processing.id}</strong>
            {study.source_kind === 'policy_search_candidate' ? <><i>·</i><span>{tr('Result')}</span><strong>{tr('Policy Search Candidate')}</strong></> : null}
            {processing.strategy_profile_name ? <><i>·</i><span>{tr('Strategy')}</span><strong>{processing.strategy_profile_name}</strong></> : null}
          </div>
          {canExport ? <button type="button" className="secondary-action compact temporal-study-export" onClick={exportStudy} disabled={exporting}>{tr(exporting ? 'Exporting…' : 'Export analysis data')}</button> : null}
        </div>
        <MonthlyCapitalMovementHeatmap
          jobId={processing.id}
          processingId={processing.id}
          rotations={study.analytics.rotations || []}
          equity={study.analytics.equity || []}
          allowDrilldown={study.source_kind !== 'policy_search_candidate'}
        />
        <WinnerTransitionAttributionPanel rotations={study.analytics.rotations || []} />
        {study.source_kind !== 'policy_search_candidate' ? <WinnerTransitionRiskSearchPanel
          study={study}
          runId={effectiveRun?.id || null}
          processingId={processing?.id || null}
          canRun={canRun}
          showRunButton={false}
          refreshToken={pipelineRefresh}
          onChange={setTransitionRiskSearch}
        /> : null}
        {study.source_kind !== 'policy_search_candidate' ? <WinnerTransitionInterventionSearchPanel
          study={study}
          runId={effectiveRun?.id || null}
          processingId={processing?.id || null}
          riskSearch={transitionRiskSearch}
          confidenceSearch={transitionConfidenceSearch}
          canRun={canRun}
          showRunButton={false}
          refreshToken={pipelineRefresh}
          onChange={setTransitionInterventionSearch}
        /> : null}
        {study.source_kind !== 'policy_search_candidate' ? <WinnerTransitionConfidenceCalibrationPanel
          study={study}
          runId={effectiveRun?.id || null}
          processingId={processing?.id || null}
          interventionSearch={transitionInterventionSearch}
          canRun={canRun}
          showRunButton={false}
          refreshToken={pipelineRefresh}
          onChange={setTransitionConfidenceSearch}
        /> : null}
        {study.source_kind !== 'policy_search_candidate' ? <WinnerTransitionStatefulReplayPanel
          study={study}
          runId={effectiveRun?.id || null}
          processingId={processing?.id || null}
          confidenceSearch={transitionConfidenceSearch}
          canRun={canRun}
          canMaterializeStrategy={canMaterializeStrategy}
          showRunButton={false}
          refreshToken={pipelineRefresh}
          onChange={setTransitionStatefulReplay}
        /> : null}
        <TemporalPolicySearchPanel
          study={study}
          runId={effectiveRun?.id || null}
          processingId={processing?.id || null}
          canRun={false}
          showActions={false}
          onChange={setPolicySearch}
        />
      </> : <div className="temporal-study-loading-result">{tr(loadingLatest ? 'Loading latest study result…' : 'No completed study result is available yet.')}</div>}
    </div>
  </section>
}
