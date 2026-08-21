import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { ApiError, apiFetch, downloadFile } from '../../api/http'
import { hasCapability } from '../../auth/capabilities'
import { API } from '../../config/env'
import { tr } from '../../i18n/runtime'
import { AnalyticsIcon, PlayIcon } from '../../shared/components/Icons'
import { StrategyResearchPipeline, STRATEGY_RESEARCH_STAGES } from './StrategyResearchPipeline'
import { StrategyResearchVisuals } from './StrategyResearchVisuals'
import './strategyResearch.css'

const ACTIVE_TEMPORAL = new Set(['queued', 'running', 'stop_requested'])
const ACTIVE_JOB = new Set(['queued', 'running'])
const FAILED = new Set(['failed', 'interrupted', 'cancelled'])

function currentMonth() {
  const date = new Date()
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`
}

function monthValue(value, fallback = '') {
  const text = String(value || '')
  return /^\d{4}-\d{2}/.test(text) ? text.slice(0, 7) : fallback
}

function validPeriod(start, end) {
  return /^\d{4}-\d{2}$/.test(start || '') && /^\d{4}-\d{2}$/.test(end || '') && start <= end
}

function wait(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms))
}

function defaultStageState() {
  return Object.fromEntries(STRATEGY_RESEARCH_STAGES.map((stage) => [stage.id, 'waiting']))
}

function strategyNeedsReferenceBacktest(strategy) {
  return String(strategy?.strategy_kind || 'standard') === 'standard'
}

function runMatchesStrategy(run, strategy) {
  if (!run?.id || !strategy?.id) return false
  if (String(run.strategy_profile_id || '') !== String(strategy.id)) return false
  if (run.strategy_profile_revision != null && strategy.revision != null && Number(run.strategy_profile_revision) !== Number(strategy.revision)) return false
  const runHash = String(run.strategy_configuration_hash || '').trim()
  const strategyHash = String(strategy.configuration_hash || '').trim()
  return !runHash || !strategyHash || runHash === strategyHash
}

export function StrategyResearchPage({ workspace, capabilities = {}, onSessionExpired }) {
  const canRun = hasCapability(capabilities, 'temporal_intelligence.start')
  const canStop = hasCapability(capabilities, 'temporal_intelligence.stop')
  const canExport = hasCapability(capabilities, 'temporal_intelligence.export')
  const canMaterialize = hasCapability(capabilities, 'temporal_intelligence.materialize_strategy')
  const canStartBacktest = hasCapability(capabilities, 'backtest.start')
  const [control, setControl] = useState(null)
  const [run, setRun] = useState(null)
  const [blockingRun, setBlockingRun] = useState(null)
  const [analytics, setAnalytics] = useState(null)
  const [risk, setRisk] = useState(null)
  const [intervention, setIntervention] = useState(null)
  const [confidence, setConfidence] = useState(null)
  const [stateful, setStateful] = useState(null)
  const [selectedStage, setSelectedStage] = useState('reference')
  const [stageState, setStageState] = useState(defaultStageState)
  const [startMonth, setStartMonth] = useState('2020-01')
  const [endMonth, setEndMonth] = useState(currentMonth)
  const [running, setRunning] = useState(false)
  const [pausing, setPausing] = useState(false)
  const [restarting, setRestarting] = useState(false)
  const [loading, setLoading] = useState(true)
  const [exporting, setExporting] = useState(false)
  const [materializing, setMaterializing] = useState(false)
  const [notice, setNotice] = useState('')
  const [error, setError] = useState('')
  const pauseRequestedRef = useRef(false)
  const activeTemporalRunRef = useRef(null)
  const activeStageRef = useRef(null)

  const strategy = control?.strategy_research_strategy || control?.research_strategy || null
  const temporalActive = ACTIVE_TEMPORAL.has(String(run?.status || '').toLowerCase())
  const blockingTemporalActive = ACTIVE_TEMPORAL.has(String(blockingRun?.status || '').toLowerCase())
  const pipelineProgress = useMemo(() => {
    const completed = STRATEGY_RESEARCH_STAGES.filter((stage) => ['completed', 'skipped'].includes(stageState[stage.id])).length
    let partial = 0
    const runningStage = STRATEGY_RESEARCH_STAGES.find((stage) => stageState[stage.id] === 'running')
    if (runningStage?.id === 'temporal') {
      const temporalProgress = Number(run?.progress)
      if (Number.isFinite(temporalProgress)) partial = Math.max(0, Math.min(100, temporalProgress)) / 100
    }
    return Math.round(((completed + partial) / STRATEGY_RESEARCH_STAGES.length) * 100)
  }, [run?.progress, stageState])

  const currentStage = useMemo(() => STRATEGY_RESEARCH_STAGES.find((stage) => stageState[stage.id] === 'running') || null, [stageState])

  const handleError = useCallback((requestError) => {
    if (requestError instanceof ApiError && requestError.status === 401) {
      onSessionExpired?.()
      return
    }
    if (requestError?.status === 403) return
    setError(tr(requestError?.message || 'Unable to load Strategy Research.'))
  }, [onSessionExpired])

  const loadExistingPipelineData = useCallback(async (loadedRun, periodStart, periodEnd) => {
    if (!loadedRun?.id || String(loadedRun?.status || '').toLowerCase() !== 'completed') return
    const processingId = String(loadedRun?.research_processing_id || '').trim()
    const nextState = defaultStageState()
    nextState.temporal = 'completed'
    let loadedAnalytics = null
    if (processingId) {
      try {
        loadedAnalytics = await apiFetch(`${API}/analytics/processings/${encodeURIComponent(processingId)}`)
        setAnalytics(loadedAnalytics)
        nextState.reference = 'completed'
      } catch {
        setAnalytics(null)
        nextState.reference = 'waiting'
      }
    }
    const query = processingId && periodStart && periodEnd ? new URLSearchParams({ processing_id: processingId, start_month: periodStart, end_month: periodEnd }) : null
    if (query) {
      const fetchLatest = async (endpoint) => {
        try {
          const value = await apiFetch(`${API}/temporal-intelligence/${encodeURIComponent(loadedRun.id)}/${endpoint}?${query.toString()}`)
          return value?.id ? value : null
        } catch {
          return null
        }
      }
      const [riskValue, interventionValue, confidenceValue, statefulValue] = await Promise.all([
        fetchLatest('winner-transition-risk-search/latest'),
        fetchLatest('winner-transition-intervention-search/latest'),
        fetchLatest('winner-transition-confidence-calibration/latest'),
        fetchLatest('winner-transition-stateful-replay/latest'),
      ])
      setRisk(riskValue)
      setIntervention(interventionValue)
      setConfidence(confidenceValue)
      setStateful(statefulValue)
      if (riskValue?.id && interventionValue?.id) nextState.risk = 'completed'
      else if (riskValue?.id || interventionValue?.id) nextState.risk = 'paused'
      if (confidenceValue?.id) nextState.confidence = 'completed'
      if (statefulValue?.id) nextState.stateful = 'completed'
    }
    if (nextState.stateful === 'completed') nextState.validation = 'completed'
    setStageState(nextState)
    const lastCompleted = [...STRATEGY_RESEARCH_STAGES].reverse().find((stage) => nextState[stage.id] === 'completed')
    if (lastCompleted) setSelectedStage(lastCompleted.id)
  }, [])

  const loadActivePipelineData = useCallback(async (loadedRun) => {
    if (!loadedRun?.id) return
    const nextState = defaultStageState()
    const processingId = String(loadedRun?.research_processing_id || '').trim()
    if (processingId) {
      try {
        const loadedAnalytics = await apiFetch(`${API}/analytics/processings/${encodeURIComponent(processingId)}`)
        setAnalytics(loadedAnalytics)
        nextState.reference = Array.isArray(loadedAnalytics?.equity) && loadedAnalytics.equity.length ? 'completed' : 'prepared'
      } catch {
        setAnalytics(null)
        nextState.reference = 'prepared'
      }
    }
    nextState.temporal = 'running'
    setStageState(nextState)
    setSelectedStage('temporal')
    activeTemporalRunRef.current = loadedRun.id
  }, [])

  const load = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      let nextControl = null
      try {
        nextControl = await apiFetch(`${API}/admin/strategies/control`)
        setControl(nextControl)
      } catch (requestError) {
        if (!(requestError instanceof ApiError) || requestError.status !== 403) throw requestError
      }
      const latest = await apiFetch(`${API}/temporal-intelligence/latest`)
      const selectedStrategy = nextControl?.strategy_research_strategy || nextControl?.research_strategy || null
      const latestIsActive = ACTIVE_TEMPORAL.has(String(latest?.status || '').toLowerCase())
      const matchesSelected = !selectedStrategy || runMatchesStrategy(latest, selectedStrategy)
      const currentRun = matchesSelected ? latest : null
      setBlockingRun(latestIsActive && !matchesSelected ? latest : null)
      setRun(currentRun)
      const start = monthValue(currentRun?.result?.oos_start, '2020-01')
      const end = monthValue(currentRun?.result?.oos_end || currentRun?.research_snapshot_cutoff || currentRun?.analysis_end_date, currentMonth())
      setStartMonth(start)
      setEndMonth(end)
      if (currentRun && ACTIVE_TEMPORAL.has(String(currentRun?.status || '').toLowerCase())) await loadActivePipelineData(currentRun)
      else if (currentRun) await loadExistingPipelineData(currentRun, start, end)
      else setStageState(defaultStageState())
      if (!nextControl && latest?.strategy_profile_name) {
        setControl({ strategy_research_strategy: { id: latest.strategy_profile_id, name: latest.strategy_profile_name, revision: latest.strategy_profile_revision, strategy_kind: latest.strategy_kind, temporal_strategy_variant: latest.temporal_strategy_variant, research_model: { label: latest.model_label } } })
      }
    } catch (requestError) {
      handleError(requestError)
    } finally {
      setLoading(false)
    }
  }, [handleError, loadActivePipelineData, loadExistingPipelineData])

  useEffect(() => { load() }, [load])

  const setStage = useCallback((id, state) => {
    if (state === 'running') activeStageRef.current = id
    if (['completed', 'failed', 'paused', 'stopped', 'skipped'].includes(state) && activeStageRef.current === id) activeStageRef.current = null
    setStageState((current) => ({ ...current, [id]: state }))
    if (state === 'running' || state === 'failed') setSelectedStage(id)
  }, [])

  useEffect(() => {
    if (!temporalActive || running || !run?.id) return undefined
    let disposed = false
    let timer = null

    const syncActiveRun = async () => {
      try {
        const value = await apiFetch(`${API}/temporal-intelligence/${encodeURIComponent(run.id)}`)
        if (disposed) return
        setRun(value)
        setBlockingRun(null)
        const status = String(value?.status || '').toLowerCase()
        if (ACTIVE_TEMPORAL.has(status)) {
          activeTemporalRunRef.current = value.id
          setStageState((current) => ({ ...current, temporal: 'running' }))
          const referenceReady = Array.isArray(analytics?.equity) && analytics.equity.length > 0
          const processingId = String(value?.research_processing_id || '').trim()
          if (!referenceReady && processingId) {
            try {
              const loadedAnalytics = await apiFetch(`${API}/analytics/processings/${encodeURIComponent(processingId)}`)
              if (!disposed) {
                setAnalytics(loadedAnalytics)
                setStageState((current) => ({ ...current, reference: Array.isArray(loadedAnalytics?.equity) && loadedAnalytics.equity.length ? 'completed' : 'prepared', temporal: 'running' }))
              }
            } catch {
              if (!disposed) setStageState((current) => ({ ...current, reference: current.reference === 'completed' ? 'completed' : 'prepared', temporal: 'running' }))
            }
          }
          return
        }

        activeTemporalRunRef.current = null
        if (status === 'completed') {
          await loadExistingPipelineData(value, startMonth, endMonth)
          return
        }
        if (FAILED.has(status)) {
          const stopped = status === 'cancelled'
          setStageState((current) => ({ ...current, temporal: stopped ? 'stopped' : 'failed' }))
          setSelectedStage('temporal')
          if (!stopped) setError(tr(value?.failure_message || 'Temporal Intelligence failed.'))
        }
      } catch (requestError) {
        if (!disposed) handleError(requestError)
      }
    }

    syncActiveRun()
    timer = window.setInterval(syncActiveRun, 2500)
    return () => {
      disposed = true
      if (timer) window.clearInterval(timer)
    }
  }, [analytics?.equity, endMonth, handleError, loadExistingPipelineData, run?.id, running, startMonth, temporalActive])

  async function waitForBacktest(jobId) {
    while (true) {
      const job = await apiFetch(`${API}/jobs/${encodeURIComponent(jobId)}`)
      if (job?.status === 'completed') return job
      if (FAILED.has(String(job?.status || '').toLowerCase())) throw new Error(job?.failure_message || 'Reference Backtest failed.')
      await wait(2500)
    }
  }

  async function waitForTemporal(runId) {
    activeTemporalRunRef.current = runId
    while (true) {
      const value = await apiFetch(`${API}/temporal-intelligence/${encodeURIComponent(runId)}`)
      setRun(value)
      if (String(value?.status || '').toLowerCase() === 'completed') {
        activeTemporalRunRef.current = null
        return value
      }
      if (FAILED.has(String(value?.status || '').toLowerCase())) {
        activeTemporalRunRef.current = null
        throw new Error(value?.failure_message || 'Temporal Intelligence failed.')
      }
      await wait(2500)
    }
  }

  async function runReferenceReplay(selectedStrategy) {
    if (!strategyNeedsReferenceBacktest(selectedStrategy)) return
    if (!canStartBacktest) throw new Error('A reference Backtest is required for this Strategy, but this profile cannot start Backtests.')
    const created = await workspace.runBacktest()
    if (!created?.id) throw new Error(workspace.error || 'Unable to start the reference Backtest.')
    if (ACTIVE_JOB.has(String(created.status || '').toLowerCase())) await waitForBacktest(created.id)
  }

  async function hydrateReferenceReplay(createdRun) {
    const processingId = String(createdRun?.research_processing_id || '').trim()
    if (!processingId) throw new Error('The selected Strategy did not produce a compatible Reference Replay source.')
    const processingAnalytics = await apiFetch(`${API}/analytics/processings/${encodeURIComponent(processingId)}`)
    setAnalytics(processingAnalytics)
    return processingId
  }

  async function hydrateStudyData(completedRun) {
    const processingId = String(completedRun?.research_processing_id || '').trim()
    if (!processingId) throw new Error('The selected Strategy did not produce a compatible Research result source.')
    const [processingAnalytics] = await Promise.all([
      apiFetch(`${API}/analytics/processings/${encodeURIComponent(processingId)}`),
      apiFetch(`${API}/temporal-intelligence/${encodeURIComponent(completedRun.id)}/decision-context?start_month=${encodeURIComponent(startMonth)}&end_month=${encodeURIComponent(endMonth)}`),
      apiFetch(`${API}/temporal-intelligence/${encodeURIComponent(completedRun.id)}/winner-transition-attribution?start_month=${encodeURIComponent(startMonth)}&end_month=${encodeURIComponent(endMonth)}`),
    ])
    setAnalytics(processingAnalytics)
    return processingId
  }

  async function runPipeline({ forceNew = false } = {}) {
    if (!canRun || running || temporalActive || blockingTemporalActive || !validPeriod(startMonth, endMonth)) return
    pauseRequestedRef.current = false
    setRunning(true)
    setPausing(false)
    setError('')
    setNotice('')
    try {
      let latestControl = control
      if (!latestControl) latestControl = await apiFetch(`${API}/admin/strategies/control`)
      setControl(latestControl)
      const selectedStrategy = latestControl?.strategy_research_strategy || latestControl?.research_strategy || strategy
      const continueCurrent = !forceNew && Boolean(
        run?.id
        && String(run?.status || '').toLowerCase() === 'completed'
        && runMatchesStrategy(run, selectedStrategy)
        && stageState.temporal === 'completed'
        && stageState.validation !== 'completed'
      )

      let completedRun = continueCurrent ? run : null
      let processingId = continueCurrent ? String(run?.research_processing_id || '').trim() : ''
      let riskValue = continueCurrent ? risk : null
      let interventionValue = continueCurrent ? intervention : null
      let confidenceValue = continueCurrent ? confidence : null
      let statefulValue = continueCurrent ? stateful : null

      if (continueCurrent) {
        if (!(Array.isArray(analytics?.equity) && analytics.equity.length) || !processingId) {
          setStage('reference', 'running')
          processingId = await hydrateStudyData(completedRun)
          setStage('reference', 'completed')
        }
        setStageState((current) => ({ ...current, temporal: 'completed' }))
      } else {
        setAnalytics(null)
        setRisk(null)
        setIntervention(null)
        setConfidence(null)
        setStateful(null)
        setStageState(defaultStageState())

        setStage('reference', 'running')
        await runReferenceReplay(selectedStrategy)
        if (pauseRequestedRef.current) throw new Error('Pipeline paused by user.')

        const created = await apiFetch(`${API}/temporal-intelligence`, { method: 'POST' })
        setBlockingRun(null)
        setRun(created)
        processingId = await hydrateReferenceReplay(created)
        setStage('reference', 'completed')
        setStage('temporal', 'running')
        completedRun = await waitForTemporal(created.id)
        processingId = await hydrateStudyData(completedRun)
        setStage('temporal', 'completed')
        if (pauseRequestedRef.current) throw new Error('Pipeline paused by user.')
      }

      const body = { processing_id: processingId, start_month: startMonth, end_month: endMonth }
      if (!riskValue?.id || !interventionValue?.id) {
        setStage('risk', 'running')
        if (!riskValue?.id) {
          riskValue = await apiFetch(`${API}/temporal-intelligence/${encodeURIComponent(completedRun.id)}/winner-transition-risk-search`, { method: 'POST', body: { ...body, seed: 42 } })
          setRisk(riskValue)
        }
        if (pauseRequestedRef.current) throw new Error('Pipeline paused by user.')
        if (!interventionValue?.id) {
          interventionValue = await apiFetch(`${API}/temporal-intelligence/${encodeURIComponent(completedRun.id)}/winner-transition-intervention-search`, { method: 'POST', body: { ...body, seed: 42 } })
          setIntervention(interventionValue)
        }
      }
      setStage('risk', 'completed')

      if (pauseRequestedRef.current) throw new Error('Pipeline paused by user.')
      if (!confidenceValue?.id) {
        setStage('confidence', 'running')
        confidenceValue = await apiFetch(`${API}/temporal-intelligence/${encodeURIComponent(completedRun.id)}/winner-transition-confidence-calibration`, { method: 'POST', body })
        setConfidence(confidenceValue)
      }
      setStage('confidence', 'completed')

      if (pauseRequestedRef.current) throw new Error('Pipeline paused by user.')
      if (!statefulValue?.id) {
        setStage('stateful', 'running')
        statefulValue = await apiFetch(`${API}/temporal-intelligence/${encodeURIComponent(completedRun.id)}/winner-transition-stateful-replay`, { method: 'POST', body })
        setStateful(statefulValue)
      }
      setStage('stateful', 'completed')

      if (pauseRequestedRef.current) throw new Error('Pipeline paused by user.')
      setStage('validation', 'running')
      await wait(150)
      setStage('validation', 'completed')
      setSelectedStage('validation')
      setNotice(tr('Research Pipeline completed.'))
      await workspace.refreshDashboard()
    } catch (requestError) {
      const paused = pauseRequestedRef.current || String(requestError?.message || '').includes('paused by user')
      const activeStageId = activeStageRef.current
      if (activeStageId) setStage(activeStageId, paused ? 'paused' : 'failed')
      if (!paused) handleError(requestError)
      else setNotice(tr('Research Pipeline paused. Continue resumes from the first unfinished stage.'))
    } finally {
      setRunning(false)
      setPausing(false)
      activeTemporalRunRef.current = null
      activeStageRef.current = null
    }
  }

  function pausePipeline() {
    if (!running || pausing) return
    pauseRequestedRef.current = true
    setPausing(true)
    setNotice(tr('Pause requested. The current stage will finish safely before the pipeline pauses.'))
  }

  async function restartPipeline() {
    if (!canRun || restarting || running || temporalActive || blockingTemporalActive || !validPeriod(startMonth, endMonth)) return
    if (!window.confirm(tr('Restart the Research Pipeline? Current derived research results will be cleared and processing will start again from the beginning.'))) return
    setRestarting(true)
    setError('')
    setNotice('')
    try {
      if (run?.id) {
        await apiFetch(`${API}/temporal-intelligence/${encodeURIComponent(run.id)}/strategy-research/reset`, { method: 'POST' })
      }
      setAnalytics(null)
      setRisk(null)
      setIntervention(null)
      setConfidence(null)
      setStateful(null)
      setStageState(defaultStageState())
      setSelectedStage('reference')
      await runPipeline({ forceNew: true })
    } catch (requestError) {
      handleError(requestError)
    } finally {
      setRestarting(false)
    }
  }

  async function exportPipeline() {
    if (!canExport || exporting || !run?.id) return
    setExporting(true)
    setError('')
    try {
      const query = new URLSearchParams({ start_month: startMonth, end_month: endMonth })
      await downloadFile(`${API}/temporal-intelligence/${encodeURIComponent(run.id)}/export.zip?${query.toString()}`, `strategy_research_${run.id}.zip`)
    } catch (requestError) {
      handleError(requestError)
    } finally {
      setExporting(false)
    }
  }

  async function materializeStrategy() {
    if (!canMaterialize || materializing || !run?.id) return
    setMaterializing(true)
    setError('')
    setNotice('')
    try {
      const canCreateStateful = stateful?.id && stateful?.candidate_a && stateful?.control_parity?.status === 'passed'
      const endpoint = canCreateStateful
        ? `${API}/temporal-intelligence/${encodeURIComponent(run.id)}/winner-transition-stateful-replay/${encodeURIComponent(stateful.id)}/candidate-a/strategy`
        : `${API}/temporal-intelligence/${encodeURIComponent(run.id)}/strategy`
      const response = await apiFetch(endpoint, { method: 'POST' })
      setNotice(tr(response?.created ? 'Strategy created in Strategy catalog.' : 'Strategy already exists in Strategy catalog.'))
    } catch (requestError) {
      handleError(requestError)
    } finally {
      setMaterializing(false)
    }
  }

  if (loading) return <div className="strategy-research-loading"><span className="loading-ring" />{tr('Loading Strategy Research…')}</div>

  const strategyName = strategy?.name || run?.strategy_profile_name || tr('Not selected')
  const strategyType = strategy?.temporal_strategy_variant || strategy?.strategy_kind || run?.temporal_strategy_variant || run?.strategy_kind || '—'
  const model = strategy?.research_model?.label || strategy?.winner_model?.label || run?.model_label || '—'
  const canContinueCurrent = Boolean(
    run?.id
    && String(run?.status || '').toLowerCase() === 'completed'
    && runMatchesStrategy(run, strategy)
    && stageState.temporal === 'completed'
    && stageState.validation !== 'completed'
  )
  const runButtonLabel = running
    ? 'Research Pipeline Running'
    : temporalActive || blockingTemporalActive
      ? 'Temporal Intelligence Running'
      : canContinueCurrent
        ? 'Continue Research Pipeline'
        : 'Run Research Pipeline'
  const temporalProgress = Number(run?.progress)
  const progressDetail = currentStage
    ? `${tr('Current stage')}: ${tr(currentStage.label)}${currentStage.id === 'temporal' && Number.isFinite(temporalProgress) ? ` · ${Math.round(Math.max(0, Math.min(100, temporalProgress)))}%${run?.stage ? ` · ${run.stage}` : ''}` : ''}`
    : running
      ? tr('Preparing pipeline…')
      : tr('Changing inputs only prepares the pipeline. Processing starts with Run Research Pipeline.')

  return <section className="strategy-research-page page-stack">
    <section className="strategy-research-header data-panel">
      <div className="strategy-research-title-block">
        <div className="page-title-icon"><AnalyticsIcon size={20} /></div>
        <div><span className="panel-kicker">{tr('STRATEGY RESEARCH')}</span><h2>{tr('Research Pipeline')}</h2><div className="strategy-research-context"><strong>{strategyName}</strong><span>·</span><span>{String(strategyType).replaceAll('_', ' ')}</span><span>·</span><span>{model}</span></div></div>
      </div>
      <div className="strategy-research-actions">
        {running && canStop ? <button type="button" className="secondary-action compact" onClick={pausePipeline} disabled={pausing}>{tr(pausing ? 'Pausing…' : 'Pause Pipeline')}</button> : null}
        {!running && !temporalActive && run?.id && canRun ? <button type="button" className="secondary-action compact" onClick={restartPipeline} disabled={restarting || blockingTemporalActive}>{tr(restarting ? 'Restarting…' : 'Restart Pipeline')}</button> : null}
        {canExport && run?.result ? <button type="button" className="secondary-action compact" onClick={exportPipeline} disabled={exporting}>{tr(exporting ? 'Exporting…' : 'Export Results')}</button> : null}
        {!running && !temporalActive && canMaterialize && stateful?.id && stateful?.control_parity?.status === 'passed' ? <button type="button" className="secondary-action compact" onClick={materializeStrategy} disabled={materializing}>{tr(materializing ? 'Creating Strategy…' : 'Create Strategy')}</button> : null}
        {canRun ? <button type="button" className="primary-action compact" onClick={() => runPipeline()} disabled={running || temporalActive || blockingTemporalActive || !validPeriod(startMonth, endMonth)}><PlayIcon />{tr(runButtonLabel)}</button> : null}
      </div>
    </section>

    <section className="strategy-research-controls data-panel">
      <label><span>{tr('Period from')}</span><input type="month" value={startMonth} onChange={(event) => setStartMonth(event.target.value)} disabled={running || temporalActive} /></label>
      <label><span>{tr('Period to')}</span><input type="month" value={endMonth} onChange={(event) => setEndMonth(event.target.value)} disabled={running || temporalActive} /></label>
      <div className="strategy-research-progress"><div><span>{tr('Pipeline progress')}</span><strong>{pipelineProgress}%</strong></div><div className="strategy-research-progress-track"><span style={{ width: `${pipelineProgress}%` }} /></div><small>{progressDetail}</small></div>
    </section>

    {!validPeriod(startMonth, endMonth) ? <div className="global-inline-message error-inline">{tr('Select a valid period.')}</div> : null}
    {blockingTemporalActive ? <div className="global-inline-message error-inline">{tr('Another Temporal Intelligence run is active for a different Strategy Research baseline.')} {blockingRun?.id || ''}</div> : null}
    {error ? <div className="global-inline-message error-inline">{error}</div> : null}
    {notice ? <div className="global-inline-message success-inline">{notice}</div> : null}

    <StrategyResearchPipeline stageState={stageState} selectedStage={selectedStage} onSelect={setSelectedStage} runProgress={run?.progress} />

    <section className="strategy-research-stage-content data-panel">
      <div className="strategy-research-stage-content-heading"><div><span className="panel-kicker">{tr('SELECTED STAGE')}</span><h3>{tr(STRATEGY_RESEARCH_STAGES.find((stage) => stage.id === selectedStage)?.label || 'Research Pipeline')}</h3></div><span>{tr('Select a pipeline stage to inspect its visual result.')}</span></div>
      <StrategyResearchVisuals selectedStage={selectedStage} stageState={stageState} pipelineProgress={pipelineProgress} run={run} analytics={analytics} risk={risk} intervention={intervention} confidence={confidence} stateful={stateful} pipelineError={error} />
    </section>
  </section>
}
