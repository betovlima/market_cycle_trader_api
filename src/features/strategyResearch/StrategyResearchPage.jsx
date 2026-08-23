import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { ApiError, apiFetch, downloadFile } from '../../api/http'
import { hasCapability } from '../../auth/capabilities'
import { API } from '../../config/env'
import { tr } from '../../i18n/runtime'
import { AnalyticsIcon, PlayIcon } from '../../shared/components/Icons'
import { StrategyResearchPipeline, STRATEGY_RESEARCH_STAGES } from './StrategyResearchPipeline'
import { StrategyResearchVisuals } from './StrategyResearchVisuals'
import { DecisionCandidates, FinalValidation, useMilpDecision } from '../milpDecision'
import '../milpDecision/milpDecision.css'
import './strategyResearch.css'

const ACTIVE_TEMPORAL = new Set(['queued', 'running', 'stop_requested'])
const ACTIVE_JOB = new Set(['queued', 'running'])
const FAILED = new Set(['failed', 'interrupted', 'cancelled'])
const ACTIVE_PIPELINE = new Set(['running', 'stop_requested'])
const RESTORE_TIMEOUT_MS = 20_000

async function apiFetchTimed(url, options = {}, timeoutMs = RESTORE_TIMEOUT_MS) {
  const controller = new AbortController()
  const timer = window.setTimeout(() => controller.abort(), timeoutMs)
  try {
    return await apiFetch(url, { ...options, signal: controller.signal })
  } catch (requestError) {
    if (requestError?.name === 'AbortError') throw new Error('Strategy Research restore request timed out.')
    throw requestError
  } finally {
    window.clearTimeout(timer)
  }
}

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

function strategyTypeLabel(strategy, run) {
  const variant = String(strategy?.temporal_strategy_variant || run?.temporal_strategy_variant || '').trim()
  const kind = String(strategy?.strategy_kind || run?.strategy_kind || 'standard').trim()
  if (variant === 'winner_transition_stateful') return tr('Conservative Decision Policy')
  if (variant === 'milp_decision_overlay') return tr('MILP Decision Strategy')
  if (kind === 'temporal_intelligence') return tr('Temporal Intelligence Strategy')
  if (kind === 'standard') return tr('Standard Strategy')
  return String(variant || kind || '—').replaceAll('_', ' ')
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
  const [leadershipRegime, setLeadershipRegime] = useState(null)
  const [clustering, setClustering] = useState(null)
  const [fragileIncumbent, setFragileIncumbent] = useState(null)
  const [emergingTrend, setEmergingTrend] = useState(null)
  const {
    result: milpResult,
    selectedCandidate,
    setSelectedCandidate,
    clear: clearMilp,
    loadLatest: loadLatestMilp,
    materialize: materializeMilp,
  } = useMilpDecision()
  const [selectedStage, setSelectedStage] = useState('reference')
  const [stageState, setStageState] = useState(defaultStageState)
  const [startMonth, setStartMonth] = useState('2020-01')
  const [endMonth, setEndMonth] = useState(currentMonth)
  const [running, setRunning] = useState(false)
  const [stopping, setStopping] = useState(false)
  const [restarting, setRestarting] = useState(false)
  const [pipelineControl, setPipelineControl] = useState(null)
  const [loading, setLoading] = useState(true)
  const [exporting, setExporting] = useState(false)
  const [materializing, setMaterializing] = useState(false)
  const [notice, setNotice] = useState('')
  const [error, setError] = useState('')
  const stopRequestedRef = useRef(false)
  const activeTemporalRunRef = useRef(null)
  const activeStageRef = useRef(null)
  const stageSelectionPinnedRef = useRef(false)

  const selectStageAutomatically = useCallback((stageId) => {
    if (!stageId || stageSelectionPinnedRef.current) return
    setSelectedStage(stageId)
  }, [])

  const handleStageSelect = useCallback((stageId) => {
    stageSelectionPinnedRef.current = true
    setSelectedStage(stageId)
  }, [])

  const strategy = control?.strategy_research_strategy || control?.research_strategy || null
  const temporalActive = ACTIVE_TEMPORAL.has(String(run?.status || '').toLowerCase())
  const blockingTemporalActive = ACTIVE_TEMPORAL.has(String(blockingRun?.status || '').toLowerCase())
  const blockingPipelineStatus = String(blockingRun?.strategy_research_pipeline?.status || 'idle').toLowerCase()
  const pipelineStatus = String(pipelineControl?.status || run?.strategy_research_pipeline?.status || 'idle').toLowerCase()
  const persistedPipelineActive = ACTIVE_PIPELINE.has(pipelineStatus)
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

  const applyPipelineControl = useCallback((value, { selectCurrent = true } = {}) => {
    if (!value || typeof value !== 'object') return
    setPipelineControl(value)
    if (value.stage_states && typeof value.stage_states === 'object') {
      setStageState((current) => ({ ...current, ...value.stage_states }))
    }
    if (selectCurrent && value.current_stage) selectStageAutomatically(value.current_stage)
    const status = String(value.status || '').toLowerCase()
    if (!ACTIVE_PIPELINE.has(status)) {
      setStopping(false)
    }
  }, [selectStageAutomatically])

  const pipelineControlAction = useCallback(async (runId, payload) => {
    if (!runId) return null
    const value = await apiFetch(`${API}/temporal-intelligence/${encodeURIComponent(runId)}/strategy-research/pipeline/control`, { method: 'POST', body: payload })
    applyPipelineControl(value)
    return value
  }, [applyPipelineControl])


  const hydratePipelineStageResults = useCallback(async (loadedRun, periodStart, periodEnd, controlValue = null) => {
    if (!loadedRun?.id) return null
    let snapshot = null
    try {
      snapshot = await apiFetchTimed(`${API}/temporal-intelligence/${encodeURIComponent(loadedRun.id)}/strategy-research/pipeline/snapshot`)
    } catch {
      return null
    }

    if (!snapshot || typeof snapshot !== 'object') return null

    const riskValue = snapshot?.risk?.id ? snapshot.risk : null
    const interventionValue = snapshot?.intervention?.id ? snapshot.intervention : null
    const confidenceValue = snapshot?.confidence?.id ? snapshot.confidence : null
    const statefulValue = snapshot?.stateful?.id ? snapshot.stateful : null
    const leadershipRegimeValue = snapshot?.leadership_regime || null
    const clusteringValue = snapshot?.clustering || null
    const fragileIncumbentValue = snapshot?.fragile_incumbent || null
    const emergingTrendValue = snapshot?.emerging_trend || null

    if (riskValue) setRisk(riskValue)
    if (interventionValue) setIntervention(interventionValue)
    if (confidenceValue) setConfidence(confidenceValue)
    if (statefulValue) setStateful(statefulValue)
    if (leadershipRegimeValue) setLeadershipRegime(leadershipRegimeValue)
    if (clusteringValue) setClustering(clusteringValue)
    if (fragileIncumbentValue) setFragileIncumbent(fragileIncumbentValue)
    if (emergingTrendValue) setEmergingTrend(emergingTrendValue)

    const processingId = String(loadedRun?.research_processing_id || '').trim()
    const snapshotStart = monthValue(snapshot?.period_start, periodStart)
    const snapshotEnd = monthValue(snapshot?.period_end, periodEnd)
    const states = controlValue?.stage_states || snapshot?.pipeline?.stage_states || {}
    const milpReady = states?.milp === 'completed' || states?.validation === 'running' || states?.validation === 'completed'
    if (processingId && snapshotStart && snapshotEnd && milpReady) {
      try {
        await loadLatestMilp(loadedRun.id, processingId, snapshotStart, snapshotEnd)
      } catch {
      }
    }

    return snapshot
  }, [loadLatestMilp])

  const loadExistingPipelineData = useCallback(async (loadedRun, periodStart, periodEnd) => {
    if (!loadedRun?.id || String(loadedRun?.status || '').toLowerCase() !== 'completed') return
    const processingId = String(loadedRun?.research_processing_id || '').trim()
    const nextState = defaultStageState()
    nextState.temporal = 'completed'

    if (processingId) {
      try {
        const loadedAnalytics = await apiFetchTimed(`${API}/analytics/processings/${encodeURIComponent(processingId)}`)
        setAnalytics(loadedAnalytics)
        nextState.reference = Array.isArray(loadedAnalytics?.equity) && loadedAnalytics.equity.length ? 'completed' : 'prepared'
      } catch {
        setAnalytics(null)
        nextState.reference = 'prepared'
      }
    }

    let snapshot = null
    try {
      snapshot = await apiFetchTimed(`${API}/temporal-intelligence/${encodeURIComponent(loadedRun.id)}/strategy-research/pipeline/snapshot`)
    } catch {
      snapshot = null
    }

    const riskValue = snapshot?.risk?.id ? snapshot.risk : null
    const interventionValue = snapshot?.intervention?.id ? snapshot.intervention : null
    const confidenceValue = snapshot?.confidence?.id ? snapshot.confidence : null
    const statefulValue = snapshot?.stateful?.id ? snapshot.stateful : null
    const leadershipRegimeValue = snapshot?.leadership_regime || null
    const clusteringValue = snapshot?.clustering || null
    const fragileIncumbentValue = snapshot?.fragile_incumbent || null
    const emergingTrendValue = snapshot?.emerging_trend || null
    const snapshotStart = monthValue(snapshot?.period_start, periodStart)
    const snapshotEnd = monthValue(snapshot?.period_end, periodEnd)
    let milpValue = null
    if (processingId && snapshotStart && snapshotEnd) {
      try {
        milpValue = await loadLatestMilp(loadedRun.id, processingId, snapshotStart, snapshotEnd)
      } catch {
        milpValue = null
      }
    }
    setRisk(riskValue)
    setIntervention(interventionValue)
    setConfidence(confidenceValue)
    setStateful(statefulValue)
    setLeadershipRegime(leadershipRegimeValue)
    setClustering(clusteringValue)
    setFragileIncumbent(fragileIncumbentValue)
    setEmergingTrend(emergingTrendValue)

    if (String(clusteringValue?.status || '').toLowerCase() === 'completed') nextState.clustering = 'completed'
    if (String(fragileIncumbentValue?.status || '').toLowerCase() === 'completed') nextState.fragile_incumbent = 'completed'
    if (String(emergingTrendValue?.status || '').toLowerCase() === 'completed') nextState.emerging_trend = 'completed'
    if (riskValue?.id && interventionValue?.id) nextState.risk = 'completed'
    else if (riskValue?.id || interventionValue?.id) nextState.risk = 'running'
    if (confidenceValue?.id) nextState.confidence = 'completed'
    if (statefulValue?.id) nextState.stateful = 'completed'
    if (milpValue?.id) nextState.milp = 'completed'
    if (String(snapshot?.validation?.status || '').toLowerCase() === 'completed') nextState.validation = 'completed'

    if (snapshotStart) setStartMonth(snapshotStart)
    if (snapshotEnd) setEndMonth(snapshotEnd)

    const persisted = snapshot?.pipeline || loadedRun?.strategy_research_pipeline
    const persistedStatus = String(persisted?.status || '').toLowerCase()
    const resolvedState = persisted?.stage_states && persistedStatus && persistedStatus !== 'idle'
      ? { ...nextState, ...persisted.stage_states }
      : nextState
    if (String(clusteringValue?.status || '').toLowerCase() === 'completed') resolvedState.clustering = 'completed'
    if (String(fragileIncumbentValue?.status || '').toLowerCase() === 'completed') resolvedState.fragile_incumbent = 'completed'
    if (String(emergingTrendValue?.status || '').toLowerCase() === 'completed') resolvedState.emerging_trend = 'completed'
    if (statefulValue?.id) resolvedState.stateful = 'completed'
    if (milpValue?.id) resolvedState.milp = 'completed'
    else resolvedState.milp = 'waiting'
    if (String(snapshot?.validation?.status || '').toLowerCase() === 'completed') resolvedState.validation = 'completed'
    setStageState(resolvedState)
    if (persisted && persistedStatus && persistedStatus !== 'idle') applyPipelineControl(persisted, { selectCurrent: false })
    const selected = persisted?.current_stage
      ? STRATEGY_RESEARCH_STAGES.find((stage) => stage.id === persisted.current_stage)
      : [...STRATEGY_RESEARCH_STAGES].reverse().find((stage) => resolvedState[stage.id] === 'completed')
    if (selected) selectStageAutomatically(selected.id)
  }, [applyPipelineControl, loadLatestMilp, selectStageAutomatically])

  const loadActivePipelineData = useCallback(async (loadedRun) => {
    if (!loadedRun?.id) return
    const nextState = defaultStageState()
    const processingId = String(loadedRun?.research_processing_id || '').trim()
    if (processingId) {
      try {
        const loadedAnalytics = await apiFetchTimed(`${API}/analytics/processings/${encodeURIComponent(processingId)}`)
        setAnalytics(loadedAnalytics)
        nextState.reference = Array.isArray(loadedAnalytics?.equity) && loadedAnalytics.equity.length ? 'completed' : 'prepared'
      } catch {
        setAnalytics(null)
        nextState.reference = 'prepared'
      }
    }
    nextState.temporal = 'running'
    const persisted = loadedRun?.strategy_research_pipeline
    const persistedStatus = String(persisted?.status || '').toLowerCase()
    const resolvedState = persisted?.stage_states && persistedStatus && persistedStatus !== 'idle'
      ? { ...nextState, ...persisted.stage_states, temporal: 'running' }
      : nextState
    setStageState(resolvedState)
    if (persisted && persistedStatus && persistedStatus !== 'idle') applyPipelineControl(persisted, { selectCurrent: false })
    selectStageAutomatically('temporal')
    activeTemporalRunRef.current = loadedRun.id
  }, [applyPipelineControl, selectStageAutomatically])

  const restorePipeline = useCallback(async (summaryRun) => {
    if (!summaryRun?.id) return
    if (ACTIVE_TEMPORAL.has(String(summaryRun?.status || '').toLowerCase())) {
      await loadActivePipelineData(summaryRun)
      return
    }
    let detailedRun = summaryRun
    try {
      detailedRun = await apiFetchTimed(`${API}/temporal-intelligence/${encodeURIComponent(summaryRun.id)}`)
      setRun(detailedRun)
    } catch (requestError) {
      handleError(requestError)
      return
    }
    const persistedPeriod = detailedRun?.strategy_research_pipeline || {}
    const start = monthValue(persistedPeriod?.start_month || detailedRun?.result?.oos_start, '2020-01')
    const end = monthValue(persistedPeriod?.end_month || detailedRun?.result?.oos_end || detailedRun?.research_snapshot_cutoff || detailedRun?.analysis_end_date, currentMonth())
    setStartMonth(start)
    setEndMonth(end)
    await loadExistingPipelineData(detailedRun, start, end)
    const persistedStatus = String(detailedRun?.strategy_research_pipeline?.status || '').toLowerCase()
    if (persistedStatus === 'paused' && canRun) {
      try {
        const recovered = await pipelineControlAction(detailedRun.id, { action: 'resume' })
        applyPipelineControl(recovered)
      } catch (requestError) {
        handleError(requestError)
      }
    }
  }, [applyPipelineControl, canRun, handleError, loadActivePipelineData, loadExistingPipelineData, pipelineControlAction])

  const load = useCallback(async () => {
    setLoading(true)
    setError('')
    let currentRun = null
    try {
      let nextControl = null
      try {
        nextControl = await apiFetchTimed(`${API}/admin/strategies/control`)
        setControl(nextControl)
      } catch (requestError) {
        if (!(requestError instanceof ApiError) || requestError.status !== 403) throw requestError
      }
      const history = await apiFetchTimed(`${API}/temporal-intelligence/history?limit=1`)
      const latest = Array.isArray(history?.items) ? history.items[0] || null : null
      const selectedStrategy = nextControl?.strategy_research_strategy || nextControl?.research_strategy || null
      const latestIsActive = ACTIVE_TEMPORAL.has(String(latest?.status || '').toLowerCase())
      const matchesSelected = !selectedStrategy || runMatchesStrategy(latest, selectedStrategy)
      currentRun = matchesSelected ? latest : null
      setBlockingRun(latestIsActive && !matchesSelected ? latest : null)
      setRun(currentRun)
      setPipelineControl(currentRun?.strategy_research_pipeline || null)
      const end = monthValue(currentRun?.research_snapshot_cutoff || currentRun?.analysis_end_date, currentMonth())
      setEndMonth(end)
      if (!currentRun) setStageState(defaultStageState())
      if (!nextControl && latest?.strategy_profile_name) {
        setControl({ strategy_research_strategy: { id: latest.strategy_profile_id, name: latest.strategy_profile_name, revision: latest.strategy_profile_revision, strategy_kind: latest.strategy_kind, temporal_strategy_variant: latest.temporal_strategy_variant, research_model: { label: latest.model_label } } })
      }
    } catch (requestError) {
      handleError(requestError)
    } finally {
      setLoading(false)
    }
    if (currentRun) {
      window.setTimeout(() => {
        restorePipeline(currentRun).catch(handleError)
      }, 0)
    }
  }, [handleError, restorePipeline])

  useEffect(() => { load() }, [load])

  useEffect(() => {
    if (!blockingTemporalActive || !blockingRun?.id) return undefined
    let disposed = false
    let timer = null
    const blockingRunId = String(blockingRun.id)

    const syncBlockingRun = async () => {
      try {
        const value = await apiFetch(`${API}/temporal-intelligence/${encodeURIComponent(blockingRunId)}`)
        if (disposed) return
        if (ACTIVE_TEMPORAL.has(String(value?.status || '').toLowerCase())) {
          setBlockingRun(value)
          timer = window.setTimeout(syncBlockingRun, 2000)
        } else {
          setBlockingRun(null)
          setStopping(false)
          stopRequestedRef.current = false
        }
      } catch (requestError) {
        if (disposed) return
        handleError(requestError)
        timer = window.setTimeout(syncBlockingRun, 3000)
      }
    }

    timer = window.setTimeout(syncBlockingRun, 1000)
    return () => {
      disposed = true
      if (timer) window.clearTimeout(timer)
    }
  }, [blockingRun?.id, blockingTemporalActive, handleError])

  const setStage = useCallback((id, state) => {
    if (state === 'running') activeStageRef.current = id
    if (['completed', 'failed', 'stopped', 'skipped'].includes(state) && activeStageRef.current === id) activeStageRef.current = null
    setStageState((current) => ({ ...current, [id]: state }))
    if (state === 'running' || state === 'failed') selectStageAutomatically(id)
  }, [selectStageAutomatically])

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
        if (value?.strategy_research_pipeline) applyPipelineControl(value.strategy_research_pipeline, { selectCurrent: false })
        const status = String(value?.status || '').toLowerCase()
        if (ACTIVE_TEMPORAL.has(status)) {
          activeTemporalRunRef.current = value.id
          setStageState((current) => ({ ...current, temporal: 'running' }))
          const referenceReady = Array.isArray(analytics?.equity) && analytics.equity.length > 0
          const processingId = String(value?.research_processing_id || '').trim()
          if (!referenceReady && processingId) {
            try {
              const loadedAnalytics = await apiFetchTimed(`${API}/analytics/processings/${encodeURIComponent(processingId)}`)
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
          selectStageAutomatically('temporal')
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
  }, [analytics?.equity, applyPipelineControl, endMonth, handleError, loadExistingPipelineData, run?.id, running, selectStageAutomatically, startMonth, temporalActive])

  useEffect(() => {
    if (!run?.id || running || !persistedPipelineActive) return undefined
    let disposed = false
    let timer = null

    const syncPipelineControl = async () => {
      try {
        const controlValue = await apiFetch(`${API}/temporal-intelligence/${encodeURIComponent(run.id)}/strategy-research/pipeline`)
        if (disposed) return
        if (controlValue) applyPipelineControl(controlValue)
        await hydratePipelineStageResults(run, startMonth, endMonth, controlValue)
        const statusValue = String(controlValue?.status || '').toLowerCase()
        if (!ACTIVE_PIPELINE.has(statusValue)) {
          if (timer) window.clearInterval(timer)
          try {
            const value = await apiFetch(`${API}/temporal-intelligence/${encodeURIComponent(run.id)}`)
            if (disposed) return
            setRun(value)
            if (String(value?.status || '').toLowerCase() === 'completed') {
              await loadExistingPipelineData(value, startMonth, endMonth)
            }
          } catch (requestError) {
            if (!disposed) handleError(requestError)
          }
        }
      } catch (requestError) {
        if (!disposed) handleError(requestError)
      }
    }

    syncPipelineControl()
    timer = window.setInterval(syncPipelineControl, 2500)
    return () => {
      disposed = true
      if (timer) window.clearInterval(timer)
    }
  }, [applyPipelineControl, endMonth, handleError, hydratePipelineStageResults, loadExistingPipelineData, persistedPipelineActive, run, run?.id, running, startMonth])

  async function waitForBacktest(jobId) {
    while (true) {
      const job = await apiFetch(`${API}/jobs/${encodeURIComponent(jobId)}`)
      if (job?.status === 'completed') return job
      if (FAILED.has(String(job?.status || '').toLowerCase())) throw new Error(job?.failure_message || 'Strategy Replay Backtest failed.')
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
    if (!processingId) throw new Error('The selected Strategy did not produce a compatible Strategy Replay source.')
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

  async function waitForResearchPipeline(runId) {
    while (true) {
      const [runValue, controlValue] = await Promise.all([
        apiFetch(`${API}/temporal-intelligence/${encodeURIComponent(runId)}`),
        apiFetch(`${API}/temporal-intelligence/${encodeURIComponent(runId)}/strategy-research/pipeline`),
      ])
      setRun(runValue)
      applyPipelineControl(controlValue)
      await hydratePipelineStageResults(runValue, startMonth, endMonth, controlValue)
      const status = String(controlValue?.status || '').toLowerCase()
      if (status === 'completed') {
        await loadExistingPipelineData(runValue, startMonth, endMonth)
        selectStageAutomatically('validation')
        setNotice(tr('Research Pipeline completed.'))
        await workspace.refreshDashboard()
        return controlValue
      }
      if (status === 'failed') {
        const failure = new Error(controlValue?.failure_message || 'Strategy Research pipeline failed.')
        failure.pipelineState = 'failed'
        throw failure
      }
      if (status === 'stopped') {
        const stopped = new Error('Pipeline stopped by user.')
        stopped.pipelineState = 'stopped'
        throw stopped
      }
      await wait(2000)
    }
  }

  async function runPipeline({ forceNew = false } = {}) {
    if (!canRun || running || temporalActive || blockingTemporalActive || !validPeriod(startMonth, endMonth)) return
    if (!forceNew && persistedPipelineActive) return
    stopRequestedRef.current = false
    stageSelectionPinnedRef.current = false
    setRunning(true)
    setStopping(false)
    setError('')
    setNotice('')
    try {
      let latestControl = control
      if (!latestControl) latestControl = await apiFetch(`${API}/admin/strategies/control`)
      setControl(latestControl)
      const selectedStrategy = latestControl?.strategy_research_strategy || latestControl?.research_strategy || strategy

      setAnalytics(null)
      setRisk(null)
      setIntervention(null)
      setConfidence(null)
      setStateful(null)
      setLeadershipRegime(null)
      setClustering(null)
      setFragileIncumbent(null)
      setEmergingTrend(null)
      clearMilp()
      setPipelineControl(null)
      setStageState(defaultStageState())

      setStage('reference', 'running')
      await runReferenceReplay(selectedStrategy)
      if (stopRequestedRef.current) {
        const stopError = new Error('Pipeline stopped by user.')
        stopError.pipelineState = 'stopped'
        throw stopError
      }

      const created = await apiFetch(`${API}/temporal-intelligence`, { method: 'POST' })
      setBlockingRun(null)
      setRun(created)
      await hydrateReferenceReplay(created)
      const started = await pipelineControlAction(created.id, { action: 'start', start_month: startMonth, end_month: endMonth })
      applyPipelineControl(started)
      await waitForResearchPipeline(created.id)
    } catch (requestError) {
      const pipelineState = String(requestError?.pipelineState || '').toLowerCase()
      const stopped = pipelineState === 'stopped' || stopRequestedRef.current || String(requestError?.message || '').includes('stopped by user')
      if (stopped) {
        setNotice(tr('Research Pipeline stopped. Restart begins a new pipeline.'))
      } else {
        handleError(requestError)
      }
    } finally {
      setRunning(false)
      setStopping(false)
      activeTemporalRunRef.current = null
      activeStageRef.current = null
    }
  }


  async function stopPipeline() {
    if (running && activeStageRef.current === 'reference' && !temporalActive) {
      if (stopping) return
      stopRequestedRef.current = true
      setStopping(true)
      setError('')
      setNotice(tr('Stop requested. The pipeline will not advance after Strategy Replay.'))
      return
    }
    const currentControllable = running || temporalActive || persistedPipelineActive
    const targetRun = currentControllable ? run : blockingTemporalActive ? blockingRun : null
    const stoppable = currentControllable || blockingTemporalActive
    if (!stoppable || stopping || !targetRun?.id) return
    stopRequestedRef.current = true
    setStopping(true)
    setError('')
    setNotice('')
    try {
      const value = await apiFetch(`${API}/temporal-intelligence/${encodeURIComponent(targetRun.id)}/strategy-research/pipeline/stop`, { method: 'POST' })
      if (String(targetRun.id) === String(run?.id || '')) {
        applyPipelineControl(value)
      } else {
        setBlockingRun((current) => current && String(current.id) === String(targetRun.id)
          ? { ...current, strategy_research_pipeline: value }
          : current)
      }
      const status = String(value?.status || '').toLowerCase()
      setNotice(tr(status === 'stopped'
        ? 'Research Pipeline stopped. Restart begins a new pipeline.'
        : 'Stop requested. The active processing is being stopped.'))
    } catch (requestError) {
      stopRequestedRef.current = false
      setStopping(false)
      setNotice('')
      handleError(requestError)
    }
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
      setLeadershipRegime(null)
      setClustering(null)
      setFragileIncumbent(null)
      setEmergingTrend(null)
      setPipelineControl(null)
      setRun(null)
      setBlockingRun(null)
      setStageState(defaultStageState())
      stageSelectionPinnedRef.current = false
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
    if (!canMaterialize || materializing || !run?.id || !selectedCandidate) return
    setMaterializing(true)
    setError('')
    setNotice('')
    try {
      let response
      if (selectedCandidate === 'milp') {
        response = await materializeMilp(run.id)
      } else if (selectedCandidate === 'stateful' && stateful?.id) {
        response = await apiFetch(`${API}/temporal-intelligence/${encodeURIComponent(run.id)}/winner-transition-stateful-replay/${encodeURIComponent(stateful.id)}/candidate-a/strategy`, { method: 'POST' })
      } else {
        throw new Error('Select a candidate in Final Validation before creating a Strategy.')
      }
      setNotice(tr(response?.created ? 'Strategy created in Strategy catalog.' : 'Strategy already exists in Strategy catalog.'))
    } catch (requestError) {
      handleError(requestError)
    } finally {
      setMaterializing(false)
    }
  }

  const canCreateCatalogStrategy = Boolean(
    canMaterialize
    && run?.id
    && run?.result
    && String(run?.status || '').toLowerCase() === 'completed'
    && stageState.validation === 'completed'
    && ((selectedCandidate === 'milp' && Boolean(milpResult?.id)) || (selectedCandidate === 'stateful' && Boolean(stateful?.candidate_a)))
  )
  const showCreateCatalogStrategy = Boolean(
    canMaterialize
    && run?.id
    && run?.result
    && String(run?.status || '').toLowerCase() === 'completed'
    && stageState.validation === 'completed'
  )


  if (loading) return <div className="strategy-research-loading"><span className="loading-ring" />{tr('Loading Strategy Research…')}</div>

  const strategyName = strategy?.name || run?.strategy_profile_name || tr('Not selected')
  const strategyType = strategyTypeLabel(strategy, run)
  const model = strategy?.research_model?.label || strategy?.winner_model?.label || run?.model_label || '—'
  const runButtonLabel = running
    ? 'Research Pipeline Running'
    : temporalActive || blockingTemporalActive
      ? 'Temporal Intelligence Running'
      : pipelineStatus === 'stop_requested'
        ? 'Research Pipeline Stopping'
        : pipelineStatus === 'stopped'
          ? 'Restart Research Pipeline'
          : persistedPipelineActive
            ? 'Research Pipeline Running'
            : 'Run Research Pipeline'
  const effectivePipelineBusy = running || temporalActive || persistedPipelineActive
  const canStopPipeline = canStop && (effectivePipelineBusy || blockingTemporalActive)
  const stopControlPending = stopping || pipelineStatus === 'stop_requested' || (blockingTemporalActive && blockingPipelineStatus === 'stop_requested')
  const temporalProgress = Number(run?.progress)
  const progressDetail = currentStage
    ? `${tr('Current stage')}: ${tr(currentStage.label)}${currentStage.id === 'temporal' && Number.isFinite(temporalProgress) ? ` · ${Math.round(Math.max(0, Math.min(100, temporalProgress)))}%${run?.stage ? ` · ${run.stage}` : ''}` : ''}`
    : pipelineStatus === 'stop_requested'
      ? tr('Stop requested. The active processing is being stopped.')
      : pipelineStatus === 'stopped'
        ? tr('Research Pipeline stopped. Restart begins a new pipeline.')
        : running || persistedPipelineActive
          ? tr('Preparing pipeline…')
          : tr('Changing inputs only prepares the pipeline. Processing starts with Run Research Pipeline.')


  return <section className="strategy-research-page page-stack">
    <section className="strategy-research-header data-panel">
      <div className="strategy-research-title-block">
        <div className="page-title-icon"><AnalyticsIcon size={20} /></div>
        <div><span className="panel-kicker">{tr('STRATEGY RESEARCH')}</span><h2>{tr('Research Pipeline')}</h2><div className="strategy-research-context"><strong>{strategyName}</strong><span>·</span><span>{strategyType}</span><span>·</span><span>{model}</span></div></div>
      </div>
      <div className="strategy-research-actions">
        {canStopPipeline ? <button type="button" className="secondary-action compact" onClick={stopPipeline} disabled={stopControlPending}>{tr(stopControlPending ? 'Stopping…' : 'Stop Pipeline')}</button> : null}
        {!effectivePipelineBusy && !temporalActive && pipelineStatus !== 'stopped' && run?.id && canRun ? <button type="button" className="secondary-action compact" onClick={restartPipeline} disabled={restarting || blockingTemporalActive}>{tr(restarting ? 'Restarting…' : 'Restart Pipeline')}</button> : null}
        {canExport && run?.result ? <button type="button" className="secondary-action compact" onClick={exportPipeline} disabled={exporting}>{tr(exporting ? 'Exporting…' : 'Export Results')}</button> : null}
        {!effectivePipelineBusy && !temporalActive && showCreateCatalogStrategy ? <button type="button" className="secondary-action compact" onClick={materializeStrategy} disabled={materializing || !canCreateCatalogStrategy} title={!selectedCandidate ? tr('Select a candidate in Final Validation before creating a Strategy.') : undefined}>{tr(materializing ? 'Creating Strategy…' : 'Create Strategy')}</button> : null}
        {canRun ? <button type="button" className="primary-action compact" onClick={pipelineStatus === 'stopped' ? restartPipeline : () => runPipeline()} disabled={restarting || effectivePipelineBusy || temporalActive || blockingTemporalActive || !validPeriod(startMonth, endMonth)}><PlayIcon />{tr(runButtonLabel)}</button> : null}
      </div>
    </section>

    <section className="strategy-research-controls data-panel">
      <label><span>{tr('Period from')}</span><input type="month" value={startMonth} onChange={(event) => setStartMonth(event.target.value)} disabled={effectivePipelineBusy || temporalActive} /></label>
      <label><span>{tr('Period to')}</span><input type="month" value={endMonth} onChange={(event) => setEndMonth(event.target.value)} disabled={effectivePipelineBusy || temporalActive} /></label>
      <div className="strategy-research-progress"><div><span>{tr('Pipeline progress')}</span><strong>{pipelineProgress}%</strong></div><div className="strategy-research-progress-track"><span style={{ width: `${pipelineProgress}%` }} /></div><small>{progressDetail}</small></div>
    </section>

    {!validPeriod(startMonth, endMonth) ? <div className="global-inline-message error-inline">{tr('Select a valid period.')}</div> : null}
    {blockingTemporalActive ? <div className="global-inline-message error-inline">{tr('Another Temporal Intelligence run is active for a different Strategy Research baseline.')} {blockingRun?.id || ''}</div> : null}
    {error ? <div className="global-inline-message error-inline">{error}</div> : null}
    {notice ? <div className="global-inline-message success-inline">{notice}</div> : null}

    <StrategyResearchPipeline stageState={stageState} selectedStage={selectedStage} onSelect={handleStageSelect} pipelineProgress={pipelineProgress} />

    <section className="strategy-research-stage-content data-panel">
      <div className="strategy-research-stage-content-heading"><div><span className="panel-kicker">{tr('SELECTED STAGE')}</span><h3>{tr(STRATEGY_RESEARCH_STAGES.find((stage) => stage.id === selectedStage)?.label || 'Research Pipeline')}</h3></div><span>{tr('Select a pipeline stage to inspect its visual result.')}</span></div>
      {selectedStage !== 'milp' ? <StrategyResearchVisuals selectedStage={selectedStage} stageState={stageState} pipelineProgress={pipelineProgress} run={run} analytics={analytics} risk={risk} intervention={intervention} confidence={confidence} stateful={stateful} leadershipRegime={leadershipRegime} clustering={clustering} fragileIncumbent={fragileIncumbent} emergingTrend={emergingTrend} pipelineError={error} /> : null}
      {selectedStage === 'milp' ? <DecisionCandidates stateful={stateful} milp={milpResult} selectedCandidate={selectedCandidate} onCandidateSelect={setSelectedCandidate} /> : null}
      {selectedStage === 'validation' ? <FinalValidation control={analytics} stateful={stateful} milp={milpResult} selectedCandidate={selectedCandidate} onCandidateSelect={setSelectedCandidate} /> : null}
    </section>
  </section>
}
