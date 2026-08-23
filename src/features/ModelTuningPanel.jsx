import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { ApiError, apiFetch, downloadFile } from '../api/http'
import { hasCapability } from '../auth/capabilities'
import { API } from '../config/env'
import { tr } from '../i18n/runtime'
import { ParameterHint } from '../shared/components/ParameterHint'
import { CANDIDATE_RANKING_HINTS } from './modelTuning/modelTuningCandidateHints'
import { ACTIVE, PROBABILITY_METHOD } from './modelTuning/modelTuningConfig'
import { candidateLabel, decimal, money, numberOr, pct } from './modelTuning/modelTuningUtils'

const MODEL_TUNING_START_CONTRACT_VERSION = 1

function CandidateCardMetric({ candidateId, label, value, tone = '' }) {
  const hint = CANDIDATE_RANKING_HINTS[label]
  return (
    <div className={`model-tuning-candidate-metric ${tone}`}>
      <span className="model-tuning-candidate-metric-label">
        <span>{tr(label)}</span>
        {hint ? <ParameterHint id={`model-tuning-card-${candidateId}-${label.toLowerCase().replace(/[^a-z0-9]+/g, '-')}`} title={tr(label)} {...hint} /> : null}
      </span>
      <strong>{value}</strong>
    </div>
  )
}

function tuningSettingValue(value) {
  if (typeof value === 'number') {
    if (Number.isInteger(value)) return String(value)
    return Number(value).toPrecision(7).replace(/0+$/, '').replace(/\.$/, '')
  }
  if (typeof value === 'boolean') return value ? tr('Yes') : tr('No')
  return value == null ? '—' : String(value)
}

function CandidateParametersGrid({ settings }) {
  const entries = Object.entries(settings || {})
  if (!entries.length) return null
  return (
    <div className="model-tuning-parameters-dialog-grid">
      {entries.map(([name, value]) => (
        <div key={name} className="model-tuning-parameters-dialog-row">
          <span title={name}>{name}</span>
          <strong>{tuningSettingValue(value)}</strong>
        </div>
      ))}
    </div>
  )
}

function signedMetricTone(value) {
  const parsed = Number(value)
  if (!Number.isFinite(parsed) || parsed === 0) return 'neutral'
  return parsed > 0 ? 'positive' : 'negative'
}

function probabilityMetricTone(value) {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return 'neutral'
  if (parsed >= 0.65) return 'positive'
  if (parsed >= 0.35) return 'warning'
  return 'negative'
}

function TuningContextLabel({ id, label, description, align = 'left' }) {
  return (
    <span className="model-tuning-context-label">
      <span>{tr(label)}</span>
      {description ? <ParameterHint id={id} title={tr(label)} description={description} align={align} /> : null}
    </span>
  )
}

export function ModelTuningPanel({ capabilities = {}, onSessionExpired, onStrategyModelSaved, onTuningContextChange }) {
  const canStartTuning = hasCapability(capabilities, 'tuning.start')
  const canStopTuning = hasCapability(capabilities, 'tuning.stop')
  const canExportTuning = hasCapability(capabilities, 'tuning.export')
  const canViewTuningLogs = hasCapability(capabilities, 'tuning.logs.view')
  const canPromoteTuning = hasCapability(capabilities, 'tuning.promote')
  const [catalog, setCatalog] = useState(null)
  const [strategy, setStrategy] = useState(null)
  const [strategyCatalogItems, setStrategyCatalogItems] = useState([])
  const [officialWinnerId, setOfficialWinnerId] = useState(null)
  const [modelFamily, setModelFamily] = useState('')
  const [baselines, setBaselines] = useState([])
  const [run, setRun] = useState(null)
  const [method, setMethod] = useState(PROBABILITY_METHOD)
  const [temporalTuningTarget, setTemporalTuningTarget] = useState('temporal_model')
  const [candidateCount, setCandidateCount] = useState(20)
  const [researchFolds, setResearchFolds] = useState(3)
  const [validationFolds, setValidationFolds] = useState(5)
  const [certificationFolds, setCertificationFolds] = useState(7)
  const [seed, setSeed] = useState(42)
  const [minimumCapitalImprovementPct, setMinimumCapitalImprovementPct] = useState('')
  const [sharpeTolerance, setSharpeTolerance] = useState('')
  const [drawdownTolerancePct, setDrawdownTolerancePct] = useState('')
  const [minimumWorstFoldPct, setMinimumWorstFoldPct] = useState('')
  const [adaptiveStoppingEnabled, setAdaptiveStoppingEnabled] = useState(true)
  const [noImprovementTrialLimit, setNoImprovementTrialLimit] = useState(100)
  const [minimumMeaningfulImprovementPct, setMinimumMeaningfulImprovementPct] = useState('0.25')
  const [busy, setBusy] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [selectedCandidateId, setSelectedCandidateId] = useState(null)
  const [parameterCandidateId, setParameterCandidateId] = useState(null)
  const [logView, setLogView] = useState(null)
  const [logLoading, setLogLoading] = useState(false)
  const [logError, setLogError] = useState('')
  const timerRef = useRef(null)
  const validationTimerRef = useRef(null)

  const handleError = useCallback((requestError) => {
    if (requestError instanceof ApiError && requestError.status === 401) {
      onSessionExpired?.()
      return
    }
    if (requestError instanceof ApiError && requestError.status === 403) {
      setError('')
      return
    }
    setError(tr(requestError.message || 'Unable to manage model tuning.'))
  }, [onSessionExpired])

  const loadLatest = useCallback(async () => {
    const latest = await apiFetch(`${API}/admin/model-tuning/latest`)
    setRun(latest || null)
    return latest || null
  }, [])

  const loadWorkspace = useCallback(async () => {
    try {
      const [nextCatalog, strategyCatalog] = await Promise.all([
        apiFetch(`${API}/admin/model-tuning/catalog`),
        apiFetch(`${API}/admin/strategies`),
      ])
      const control = strategyCatalog?.control || {}
      const strategyId = control?.strategy_research_strategy_id || control?.research_strategy_id
      const detail = strategyId ? await apiFetch(`${API}/admin/strategies/${encodeURIComponent(strategyId)}`) : null
      const [baselinePayload] = await Promise.all([
        apiFetch(`${API}/admin/model-tuning/baselines?limit=20`),
        loadLatest(),
      ])
      const items = Array.isArray(baselinePayload?.items) ? baselinePayload.items : []
      const savedModel = detail?.research_model_configuration || detail?.research_model || null
      const probability = nextCatalog?.probability || {}
      setCatalog(nextCatalog)
      setStrategyCatalogItems(Array.isArray(strategyCatalog?.items) ? strategyCatalog.items : [])
      setOfficialWinnerId(control?.trader_winner_strategy_id || null)
      setStrategy(detail)
      const temporalModes = Array.isArray(nextCatalog?.temporal_tuning_modes) ? nextCatalog.temporal_tuning_modes : []
      const temporalDefault = nextCatalog?.default_temporal_tuning_target || temporalModes[0]?.id || 'temporal_model'
      if (detail?.strategy_kind === 'temporal_intelligence') {
        setTemporalTuningTarget((current) => {
          const target = temporalModes.some((item) => item.id === current) ? current : temporalDefault
          setCandidateCount(target === 'temporal_model' ? Math.min(8, nextCatalog.default_candidate_count || 20) : (nextCatalog.default_candidate_count || 20))
          return target
        })
      } else {
        setCandidateCount(nextCatalog.default_candidate_count || 20)
      }
      setModelFamily(detail?.strategy_kind === 'temporal_intelligence' ? 'lightgbm_utility' : (savedModel?.family || ''))
      setBaselines(items)
      const foldProtocol = nextCatalog?.fold_protocol || {}
      setResearchFolds(Number(foldProtocol.research_default || 3))
      setValidationFolds(Number(foldProtocol.validation_default || 5))
      setCertificationFolds(Number(foldProtocol.certification_default || 7))
      setSeed(nextCatalog.default_seed ?? 42)
      setMinimumCapitalImprovementPct(String(numberOr(probability.default_min_capital_improvement) * 100))
      setSharpeTolerance(String(probability.default_sharpe_tolerance ?? ''))
      setDrawdownTolerancePct(String(numberOr(probability.default_drawdown_tolerance) * 100))
      setMinimumWorstFoldPct(String(numberOr(probability.default_min_worst_fold_return) * 100))
      setAdaptiveStoppingEnabled(probability.default_adaptive_stopping_enabled !== false)
      setNoImprovementTrialLimit(Number(probability.default_no_improvement_trial_limit || 100))
      setMinimumMeaningfulImprovementPct(String(numberOr(probability.default_minimum_meaningful_improvement ?? 0.0025) * 100))
      setError('')
    } catch (requestError) {
      handleError(requestError)
    }
  }, [handleError, loadLatest])

  useEffect(() => {
    loadWorkspace()
  }, [loadWorkspace])

  useEffect(() => {
    onTuningContextChange?.(strategy ? {
      id: strategy.id,
      name: strategy.name,
      revision: strategy.revision,
      strategy_kind: strategy.strategy_kind,
      status: strategy.catalog_status || strategy.status,
      source_temporal_run_id: strategy.source_temporal_run_id,
    } : null)
  }, [onTuningContextChange, strategy?.id, strategy?.name, strategy?.revision, strategy?.strategy_kind, strategy?.status, strategy?.source_temporal_run_id])

  useEffect(() => {
    if (!run?.id || !run?.fold_protocol) return
    setResearchFolds(Number(run.fold_protocol.research_folds || 3))
    setValidationFolds(Number(run.fold_protocol.validation_folds || 5))
    setCertificationFolds(Number(run.fold_protocol.certification_folds || 7))
  }, [run?.id])

  useEffect(() => {
    if (timerRef.current) window.clearInterval(timerRef.current)
    timerRef.current = null
    if (!run?.id || !ACTIVE.has(run.status)) return undefined
    timerRef.current = window.setInterval(async () => {
      try {
        const updated = await apiFetch(`${API}/admin/model-tuning/${encodeURIComponent(run.id)}`)
        setRun(updated)
      } catch (requestError) {
        handleError(requestError)
      }
    }, 2500)
    return () => {
      if (timerRef.current) window.clearInterval(timerRef.current)
      timerRef.current = null
    }
  }, [handleError, run?.id, run?.status])

  const activeValidation = useMemo(() => (
    (run?.candidates || []).find((candidate) => ['queued', 'running'].includes(String(candidate?.validation?.status || '').toLowerCase())) || null
  ), [run?.candidates])

  useEffect(() => {
    if (validationTimerRef.current) window.clearInterval(validationTimerRef.current)
    validationTimerRef.current = null
    if (!run?.id || !activeValidation) return undefined
    validationTimerRef.current = window.setInterval(async () => {
      try {
        const updated = await apiFetch(`${API}/admin/model-tuning/${encodeURIComponent(run.id)}`)
        setRun(updated)
      } catch (requestError) {
        handleError(requestError)
      }
    }, 2000)
    return () => {
      if (validationTimerRef.current) window.clearInterval(validationTimerRef.current)
      validationTimerRef.current = null
    }
  }, [activeValidation?.candidate_id, handleError, run?.id])

  const sortedCandidates = useMemo(() => {
    const items = [...(run?.candidates || [])]
    items.sort((left, right) => {
      if (Boolean(left.is_control) !== Boolean(right.is_control)) return left.is_control ? -1 : 1
      const leftRank = left.rank ?? Number.MAX_SAFE_INTEGER
      const rightRank = right.rank ?? Number.MAX_SAFE_INTEGER
      if (leftRank !== rightRank) return leftRank - rightRank
      return left.candidate_id - right.candidate_id
    })
    return items
  }, [run?.candidates])

  const selectedBaseline = baselines[0] || null
  const activeRun = Boolean(run && ACTIVE.has(run.status))

  const baselineControlCandidate = useMemo(() => {
    if (!selectedBaseline) return null
    return {
      candidate_id: 0,
      kind: 'control',
      is_control: true,
      status: 'completed',
      rank: null,
      metrics: selectedBaseline.metrics || {},
      proposal: null,
      job_id: selectedBaseline.job_id,
      source_job_id: selectedBaseline.job_id,
      baseline_reused: true,
      baseline_preview: true,
      settings_hash: selectedBaseline.model_settings_hash || '',
    }
  }, [selectedBaseline])

  const runControlCandidate = useMemo(
    () => sortedCandidates.find((candidate) => candidate.is_control) || null,
    [sortedCandidates],
  )
  const runBaselineJobId = String(run?.baseline_execution?.job_id || runControlCandidate?.source_job_id || runControlCandidate?.job_id || '')
  const currentBaselineJobId = String(selectedBaseline?.job_id || '')
  const runMatchesCurrentBaseline = !currentBaselineJobId || !runBaselineJobId || currentBaselineJobId === runBaselineJobId

  const displayedCandidates = useMemo(() => {
    const control = activeRun ? (runControlCandidate || baselineControlCandidate) : (baselineControlCandidate || runControlCandidate)
    const challengers = (activeRun || runMatchesCurrentBaseline)
      ? sortedCandidates.filter((candidate) => !candidate.is_control)
      : []
    return control ? [control, ...challengers] : challengers
  }, [activeRun, baselineControlCandidate, runControlCandidate, runMatchesCurrentBaseline, sortedCandidates])

  const currentChampionCandidateId = run?.probability_state?.last_champion_candidate_id ?? run?.probability_anchor?.candidate_id ?? 0
  const bestCompletedChallenger = useMemo(() => {
    const items = displayedCandidates
      .filter((candidate) => !candidate.is_control && candidate.status === 'completed')
      .sort((left, right) => {
        const leftRank = Number(left.rank ?? Number.MAX_SAFE_INTEGER)
        const rightRank = Number(right.rank ?? Number.MAX_SAFE_INTEGER)
        if (leftRank !== rightRank) return leftRank - rightRank
        return Number(right.metrics?.ending_capital || 0) - Number(left.metrics?.ending_capital || 0)
      })
    return items[0] || null
  }, [displayedCandidates])
  const currentBestCandidate = useMemo(() => {
    const champion = run?.method === PROBABILITY_METHOD
      ? displayedCandidates.find((candidate) => Number(candidate.candidate_id) === Number(currentChampionCandidateId))
      : null
    return champion || bestCompletedChallenger || baselineControlCandidate || null
  }, [baselineControlCandidate, bestCompletedChallenger, currentChampionCandidateId, displayedCandidates, run?.method])
  const currentBestImprovement = useMemo(() => {
    const baselineCapital = Number(selectedBaseline?.metrics?.ending_capital)
    const bestCapital = Number(currentBestCandidate?.metrics?.ending_capital)
    if (!Number.isFinite(baselineCapital) || baselineCapital === 0 || !Number.isFinite(bestCapital)) return null
    return (bestCapital - baselineCapital) / baselineCapital
  }, [currentBestCandidate, selectedBaseline])

  const visibleCandidates = useMemo(() => {
    if (!activeRun) {
      return displayedCandidates.filter((candidate) => {
        if (candidate.is_control || candidate.status !== 'pending') return true
        return Number(candidate.candidate_id) === Number(run?.current_candidate_id)
      })
    }
    const importantIds = new Set([
      Number(currentBestCandidate?.candidate_id),
      Number(run?.current_candidate_id),
    ].filter(Number.isFinite))
    return displayedCandidates.filter((candidate) => candidate.is_control || importantIds.has(Number(candidate.candidate_id)))
  }, [activeRun, currentBestCandidate?.candidate_id, displayedCandidates, run?.current_candidate_id])

  const selectedCandidate = useMemo(
    () => displayedCandidates.find((item) => item.candidate_id === selectedCandidateId) || null,
    [displayedCandidates, selectedCandidateId],
  )

  const parameterCandidate = useMemo(
    () => displayedCandidates.find((item) => item.candidate_id === parameterCandidateId) || null,
    [displayedCandidates, parameterCandidateId],
  )

  const selectedMethod = useMemo(
    () => (catalog?.methods || []).find((item) => item.id === method) || null,
    [catalog?.methods, method],
  )

  const active = activeRun
  const candidateCardMethod = run?.id && runMatchesCurrentBaseline ? run.method : method
  const probabilityMode = method === PROBABILITY_METHOD
  const adaptiveMode = probabilityMode
  const temporalStrategy = strategy?.strategy_kind === 'temporal_intelligence'
  const temporalModes = Array.isArray(catalog?.temporal_tuning_modes) ? catalog.temporal_tuning_modes : []
  const selectedTemporalMode = temporalStrategy
    ? (temporalModes.find((item) => item.id === temporalTuningTarget) || temporalModes[0] || null)
    : null
  const effectivePlan = selectedTemporalMode || catalog
  const temporalModelMode = temporalStrategy && temporalTuningTarget === 'temporal_model'
  const temporalPolicyMode = temporalStrategy && temporalTuningTarget === 'temporal_policy'
  const temporalTarget = temporalStrategy
  const gateTuning = ['absolute_utility_cash_gate', 'joint_model_absolute_utility_cash_gate'].includes(String(run?.tuning_scope || effectivePlan?.tuning_scope || catalog?.tuning_scope || ''))
  const startActionLabel = temporalModelMode
    ? tr(probabilityMode ? 'Start Temporal Model CARO' : 'Start Temporal Model LHS')
    : temporalPolicyMode
      ? tr(probabilityMode ? 'Start Temporal Policy CARO' : 'Start Temporal Policy LHS')
      : probabilityMode ? tr('Start Unified CARO') : tr('Start Latin Hypercube')
  const officialWinner = useMemo(
    () => strategyCatalogItems.find((item) => item.id === officialWinnerId) || null,
    [officialWinnerId, strategyCatalogItems],
  )
  const tuningStartContractCompatible = Number(catalog?.start_request_contract_version || 0) === MODEL_TUNING_START_CONTRACT_VERSION
  const strategyTuningCompatible = catalog?.strategy_compatibility?.eligible !== false
  const canTune = Boolean(
    canStartTuning
    && strategy
    && strategyTuningCompatible
    && tuningStartContractCompatible
    && (temporalStrategy || modelFamily === catalog?.model_family)
    && selectedBaseline
  )
  const workflowStepIndex = active
    ? 2
    : run?.status === 'completed' && runMatchesCurrentBaseline
      ? 3
      : canTune
        ? 1
        : 0
  const foldMinimum = Number(catalog?.fold_protocol?.minimum || 2)
  const foldProtocolValid = Number(researchFolds) >= foldMinimum
    && Number(validationFolds) >= Number(researchFolds)
    && Number(certificationFolds) >= Number(validationFolds)
  const continuationResearchFoldsCompatible = !run?.fold_protocol
    || Number(researchFolds) === Number(run.fold_protocol.research_folds || 3)
  const foldInputDisabled = !canStartTuning || !canTune || active || busy


  async function start() {
    if (!canTune || busy || (temporalStrategy && !foldProtocolValid)) return
    if (temporalModelMode && !window.confirm(tr('Temporal Model Tuning retrains LightGBM for every candidate. Start this campaign now?'))) return
    setBusy(true)
    setError('')
    setNotice('')
    try {
      const body = {
        method,
        candidate_count: Number(candidateCount),
        seed: Number(seed),
      }
      if (temporalStrategy) {
        body.tuning_target = temporalTuningTarget
        body.explicit_start_confirmation = temporalModelMode
        body.fold_protocol = {
          research_folds: Number(researchFolds),
          validation_folds: Number(validationFolds),
          certification_folds: Number(certificationFolds),
        }
      }
      if (adaptiveMode) {
        body.probability = {
          min_capital_improvement: numberOr(minimumCapitalImprovementPct) / 100,
          sharpe_tolerance: numberOr(sharpeTolerance),
          drawdown_tolerance: numberOr(drawdownTolerancePct) / 100,
          min_worst_fold_return: numberOr(minimumWorstFoldPct) / 100,
          adaptive_stopping_enabled: adaptiveStoppingEnabled,
          no_improvement_trial_limit: Number(noImprovementTrialLimit),
          minimum_meaningful_improvement: numberOr(minimumMeaningfulImprovementPct) / 100,
        }
      }
      const created = await apiFetch(`${API}/admin/model-tuning`, { method: 'POST', body })
      setRun(created)
      setNotice(temporalModelMode
        ? tr('Temporal Model Tuning started. Every challenger retrains the Temporal LightGBM models on the same frozen market snapshot and walk-forward protocol.')
        : temporalPolicyMode
          ? tr('Temporal Policy Tuning started. Candidates reuse the frozen Temporal predictions and replay only the Winner-Anchored timing policy.')
          : probabilityMode
            ? tr('Unified CARO started from the certified Candidate Backtest. Exploration and probabilistic refinement are selected automatically throughout the campaign.')
            : tr('Latin Hypercube tuning started from the certified Candidate Backtest.'))
    } catch (requestError) {
      handleError(requestError)
    } finally {
      setBusy(false)
    }
  }


  async function continueResearch() {
    if (!run?.id || run.status !== 'completed' || run.method !== PROBABILITY_METHOD || busy || !tuningStartContractCompatible || !foldProtocolValid || !continuationResearchFoldsCompatible) return
    if (run?.tuning_scope === 'temporal_model' && !window.confirm(tr('Temporal Model Tuning retrains LightGBM for every candidate. Continue this campaign now?'))) return
    setBusy(true)
    setError('')
    setNotice('')
    try {
      const body = {
        method: PROBABILITY_METHOD,
        candidate_count: Number(candidateCount),
        seed: Number(seed),
        source_tuning_run_id: run.id,
        tuning_target: run.tuning_scope,
        explicit_start_confirmation: run.tuning_scope === 'temporal_model',
        fold_protocol: {
          research_folds: Number(researchFolds),
          validation_folds: Number(validationFolds),
          certification_folds: Number(certificationFolds),
        },
        probability: {
          min_capital_improvement: numberOr(minimumCapitalImprovementPct) / 100,
          sharpe_tolerance: numberOr(sharpeTolerance),
          drawdown_tolerance: numberOr(drawdownTolerancePct) / 100,
          min_worst_fold_return: numberOr(minimumWorstFoldPct) / 100,
          adaptive_stopping_enabled: adaptiveStoppingEnabled,
          no_improvement_trial_limit: Number(noImprovementTrialLimit),
          minimum_meaningful_improvement: numberOr(minimumMeaningfulImprovementPct) / 100,
        },
      }
      const created = await apiFetch(`${API}/admin/model-tuning`, { method: 'POST', body })
      setRun(created)
      setNotice(tr('Research continued from the completed CARO campaign. Prior observations were imported and the new budget adds only new trials.'))
    } catch (requestError) {
      handleError(requestError)
    } finally {
      setBusy(false)
    }
  }

  async function validateFinalist(candidate) {
    if (!run?.id || !candidate || busy) return
    setBusy(true)
    setError('')
    setNotice('')
    try {
      await apiFetch(`${API}/admin/model-tuning/${encodeURIComponent(run.id)}/candidates/${candidate.candidate_id}/validate-champion`, {
        method: 'POST',
      })
      const updated = await apiFetch(`${API}/admin/model-tuning/${encodeURIComponent(run.id)}`)
      setRun(updated)
      setNotice(tr('Validation started. The full Temporal LightGBM walk-forward is running in the background. Progress is shown on this candidate.'))
    } catch (requestError) {
      handleError(requestError)
    } finally {
      setBusy(false)
    }
  }

  async function certifyCandidate(candidate) {
    if (!run?.id || !candidate || busy) return
    setBusy(true)
    setError('')
    setNotice('')
    try {
      const certification = await apiFetch(`${API}/admin/model-tuning/${encodeURIComponent(run.id)}/candidates/${candidate.candidate_id}/certify`, {
        method: 'POST',
      })
      const updated = await apiFetch(`${API}/admin/model-tuning/${encodeURIComponent(run.id)}`)
      setRun(updated)
      setNotice(tr('CARO candidate certification completed with the configured certification folds. Trader Winner promotion remains protected until the Temporal live execution engine is installed.'))
      if (certification?.certification_processing_id) viewProcessing(certification.certification_processing_id)
    } catch (requestError) {
      handleError(requestError)
    } finally {
      setBusy(false)
    }
  }

  function viewProcessing(processingId) {
    if (!processingId) return
    window.dispatchEvent(new CustomEvent('mct:open-dashboard-processing', {
      detail: { processingId },
    }))
  }

  function viewChampionAnalytics() {
    viewProcessing(run?.certification_processing_id || run?.validation_processing_id)
  }

  async function stop() {
    if (!run?.id || !active || busy) return
    setBusy(true)
    setError('')
    try {
      const updated = await apiFetch(`${API}/admin/model-tuning/${encodeURIComponent(run.id)}/stop`, { method: 'POST' })
      setRun(updated)
      setNotice(tr(run?.tuning_scope === 'temporal_model' ? 'Stop requested. The active Temporal LightGBM candidate will stop at the next model checkpoint and no new candidate will start.' : 'Stop requested. The active tuning candidate is being cancelled and no new candidate will start.'))
    } catch (requestError) {
      handleError(requestError)
    } finally {
      setBusy(false)
    }
  }

  async function adopt(candidate) {
    if (!run?.id || busy) return
    setBusy(true)
    setError('')
    setNotice('')
    try {
      const response = await apiFetch(`${API}/admin/model-tuning/${encodeURIComponent(run.id)}/candidates/${candidate.candidate_id}/adopt`, {
        method: 'POST',
        body: {},
      })
      if (response?.recommended_tuning_target) setTemporalTuningTarget(response.recommended_tuning_target)
      setNotice(tr(response?.ready_for_model_tuning
        ? (run?.tuning_scope === 'temporal_model'
          ? 'A new TEMPORAL Strategy was created with the tuned LightGBM model and selected for the next Policy Tuning stage.'
          : 'A new TEMPORAL Strategy was created from the tuned policy and selected for Model Tuning.')
        : 'A new Strategy was created from the frozen tuning result and selected as BACKTEST. After a successful Backtest it becomes the active CANDIDATE automatically.'))
      await onStrategyModelSaved?.(response.strategy)
      await loadWorkspace()
    } catch (requestError) {
      handleError(requestError)
    } finally {
      setBusy(false)
    }
  }

  async function exportCampaign() {
    if (!run?.id || exporting) return
    setExporting(true)
    setError('')
    try {
      await downloadFile(
        `${API}/admin/model-tuning/${encodeURIComponent(run.id)}/export.zip`,
        `model_tuning_${run.id}.zip`,
      )
    } catch (requestError) {
      handleError(requestError)
    } finally {
      setExporting(false)
    }
  }

  async function openCampaignLog() {
    if (!run?.id || logLoading) return
    setLogLoading(true)
    setLogError('')
    try {
      const payload = await apiFetch(`${API}/admin/model-tuning/${encodeURIComponent(run.id)}/log`)
      setLogView({ ...payload, title: tr('Campaign log') })
    } catch (requestError) {
      if (requestError instanceof ApiError && requestError.status === 401) onSessionExpired?.()
      else setLogError(tr(requestError.message || 'Unable to load diagnostic log.'))
    } finally {
      setLogLoading(false)
    }
  }

  async function openCandidateLog(candidate) {
    if (!run?.id || candidate?.candidate_id === undefined || logLoading) return
    setLogLoading(true)
    setLogError('')
    try {
      const payload = await apiFetch(`${API}/admin/model-tuning/${encodeURIComponent(run.id)}/candidates/${candidate.candidate_id}/log`)
      setLogView({ ...payload, title: `${tr('Execution log')} · ${candidateLabel(candidate)}` })
    } catch (requestError) {
      if (requestError instanceof ApiError && requestError.status === 401) onSessionExpired?.()
      else setLogError(tr(requestError.message || 'Unable to load diagnostic log.'))
    } finally {
      setLogLoading(false)
    }
  }

  async function copyDiagnosticLog() {
    const text = String(logView?.log_text || '')
    if (!text) return
    try {
      await navigator.clipboard.writeText(text)
      setNotice(tr('Log copied to clipboard.'))
    } catch {
      setLogError(tr('Unable to copy diagnostic log.'))
    }
  }

  function downloadDiagnosticLog() {
    const text = String(logView?.log_text || '')
    if (!text) return
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    const candidatePart = logView?.candidate_id !== undefined ? `_candidate_${logView.candidate_id}` : '_campaign'
    anchor.href = url
    anchor.download = `model_tuning_${run?.id || 'log'}${candidatePart}.txt`
    document.body.appendChild(anchor)
    anchor.click()
    anchor.remove()
    URL.revokeObjectURL(url)
  }

  if (!catalog) return <div className="backtest-loading-row">{tr('Loading model tuning…')}</div>

  return (
    <section className={`model-tuning-panel model-tuning-workspace ${active ? 'is-running' : ''}`}>
      <div className="model-tuning-heading model-tuning-heading-compact">
        <div>
          <span className="panel-kicker">{tr('MODEL TUNING')}</span>
          <h3>{tr('Probabilistic parameter research')}</h3>
        </div>
        <div className="model-tuning-method-badge">{selectedMethod?.label || tr('Model Tuning')}</div>
      </div>

      {error ? <div className="global-inline-message error-inline">{error}</div> : null}
      {notice ? <div className="global-inline-message success-inline">{notice}</div> : null}
      {!strategy ? <div className="global-inline-message warning-inline">{tr('Select a Strategy from the catalog to begin research.')}</div> : null}
      {catalog && !tuningStartContractCompatible ? <div className="global-inline-message warning-inline">{tr('Model Tuning API/Front contract mismatch. Refresh the application after both API and Front are deployed from the same release.')}</div> : null}
      {strategy && !strategyTuningCompatible ? <div className="global-inline-message warning-inline">{tr(catalog?.strategy_compatibility?.reason || 'The selected Strategy is not compatible with the current Model Tuning engine.')}</div> : null}
      {strategy && !temporalTarget && modelFamily !== catalog.model_family ? <div className="global-inline-message warning-inline">{tr('The current tuning target must use LightGBM.')}</div> : null}
      {strategy && !temporalTarget && modelFamily === catalog.model_family && !baselines.length ? <div className="global-inline-message warning-inline">{tr('A compatible completed Backtest is required for this Strategy before tuning can start.')}</div> : null}
      {strategy && temporalTarget && !baselines.length ? <div className="global-inline-message warning-inline">{tr('The TEMPORAL Strategy source run is not available as a completed frozen replay.')}</div> : null}

      <div className="model-tuning-workflow-steps" aria-label={tr('Research workflow')}>
        {['1. BASELINE', '2. RESEARCH', '3. TUNING', '4. RESULTS'].map((label, index) => <span key={label} className={workflowStepIndex === index ? 'active' : ''}>{tr(label)}</span>)}
      </div>

      <section className="model-tuning-step model-tuning-step-baseline">
        <div className="model-tuning-step-heading"><span>1</span><div><strong>{tr('Baseline')}</strong><small>{tr('Model Tuning always uses the Strategy selected for Strategy Research.')}</small></div></div>
        {strategy ? <div className="model-tuning-selected-strategy"><div><span>{tr('Selected Strategy')}</span><strong>{strategy.name}</strong></div><div><span>{tr('Status')}</span><strong>{tr(strategy.catalog_status || strategy.status)}</strong></div><div><span>{tr('Kind')}</span><strong>{strategy.strategy_kind || 'standard'}</strong></div><div><span>{tr('Revision')}</span><strong>{strategy.revision}</strong></div>{officialWinner ? <div className="model-tuning-winner-reference"><span>{tr('Official Winner')}</span><strong>{officialWinner.name}</strong><small>{officialWinner.tuning_result_metrics?.ending_capital != null ? money(officialWinner.tuning_result_metrics.ending_capital) : tr(officialWinner.status || 'winner')}</small></div> : null}</div> : null}
      </section>

      {temporalStrategy && temporalModes.length ? (
        <div className="model-tuning-temporal-mode model-tuning-idle-only">
          <div className="model-tuning-temporal-mode-head">
            <div><span className="panel-kicker">{tr('TEMPORAL TUNING')}</span><strong>{tr('Choose what to optimize')}</strong></div>
            <small>{tr('Model first, then policy. Both use the same materialized TEMPORAL Strategy workflow.')}</small>
          </div>
          <div className="research-mode-switch">
            {temporalModes.map((item) => (
              <button
                key={item.id}
                type="button"
                className={temporalTuningTarget === item.id ? 'active' : ''}
                disabled={active || busy}
                onClick={() => {
                  setTemporalTuningTarget(item.id)
                  setCandidateCount(item.id === 'temporal_model' ? Math.min(8, catalog.default_candidate_count || 20) : (catalog.default_candidate_count || 20))
                }}
              >
                <strong>{tr(item.id === 'temporal_model' ? 'Model Tuning · LightGBM' : 'Policy Tuning · Replay')}</strong>
                <span>{tr(item.description || '')}</span>
              </button>
            ))}
          </div>
          {temporalModelMode ? <div className="global-inline-message warning-inline">{tr('Full Temporal Model Tuning retrains LightGBM for every candidate and is intentionally much slower than Policy Tuning.')}</div> : null}
        </div>
      ) : null}

      {temporalStrategy && catalog?.fold_protocol?.supported ? (
        <section className="model-tuning-fold-protocol model-tuning-idle-only">
          <div className="model-tuning-fold-protocol-heading">
            <div>
              <span className="panel-kicker">{tr('WALK-FORWARD PROTOCOL')}</span>
              <strong>{tr('Research → Validation → Certification')}</strong>
            </div>
            <small>{tr('Folds are part of the experimental protocol and are never optimized by CARO.')}</small>
          </div>
          <div className="model-tuning-fold-protocol-grid">
            <label>
              <span>{tr('Research folds')}</span>
              <input type="number" min={foldMinimum} step="1" value={researchFolds} disabled={foldInputDisabled} onChange={(event) => setResearchFolds(event.target.value)} />
              <small>{tr('Used by CARO candidate search. For Policy Tuning, changing this value builds one new frozen Temporal LightGBM prediction cache before the fast policy replays begin.')}</small>
            </label>
            <label>
              <span>{tr('Validation folds')}</span>
              <input type="number" min={Math.max(foldMinimum, Number(researchFolds) || foldMinimum)} step="1" value={validationFolds} disabled={foldInputDisabled} onChange={(event) => setValidationFolds(event.target.value)} />
              <small>{tr('Used only when you validate a selected CARO finalist. The Temporal LightGBM models are fully retrained under the new walk-forward split.')}</small>
            </label>
            <label>
              <span>{tr('Certification folds')}</span>
              <input type="number" min={Math.max(foldMinimum, Number(validationFolds) || foldMinimum)} step="1" value={certificationFolds} disabled={foldInputDisabled} onChange={(event) => setCertificationFolds(event.target.value)} />
              <small>{tr('Used after a finalist passes validation. Certification performs another full Temporal LightGBM walk-forward rerun before Winner eligibility can be considered.')}</small>
            </label>
          </div>
          {!foldProtocolValid ? <small className="model-tuning-fold-protocol-error">{tr('Fold protocol must satisfy Research ≤ Validation ≤ Certification, with at least 2 folds at every stage.')}</small> : null}
          {run?.id && run?.status === 'completed' && run?.method === PROBABILITY_METHOD ? <small>{tr('Continue Research must keep the same Research fold count because the imported CARO observations belong to that protocol. Start a new campaign to change Research folds.')}</small> : null}
        </section>
      ) : null}

      <div className="model-tuning-context-grid model-tuning-context-grid-wide model-tuning-idle-only">
        <div className="model-tuning-context-card model-tuning-target-card">
          <TuningContextLabel
            id="model-tuning-hint-target"
            label="Tuning target"
            description={temporalTarget ? tr('The selected materialized TEMPORAL Strategy is the immutable baseline for both LightGBM Model Tuning and fast Policy Tuning.') : tr('The Strategy selected as RESEARCH is used as the starting point for this research campaign.')}
          />
          <strong title={strategy?.name || ''}>{strategy?.name || '—'}</strong>
          <small>{strategy ? `${tr(strategy.catalog_status || strategy.status)} · ${tr('Revision')} ${strategy.revision}` : '—'}</small>
        </div>
        <div className="model-tuning-context-card model-tuning-scope-card">
          <TuningContextLabel
            id="model-tuning-hint-scope"
            label="Tuning scope"
            description={effectivePlan?.description || catalog?.tuning_scope_description || 'Defines exactly which parameters Adaptive CARO may change while the remaining experiment stays frozen.'}
          />
          <strong title={tr(effectivePlan?.label || effectivePlan?.tuning_scope_label || 'LightGBM model parameters')}>{tr(effectivePlan?.label || effectivePlan?.tuning_scope_label || 'LightGBM model parameters')}</strong>
          <small>{catalog?.joint_optimization ? `${(effectivePlan?.tuned_parameters || catalog.tuned_parameters || []).length} ${tr('parameters')} · ${(catalog.tuned_model_parameters || []).length} LightGBM · ${(catalog.tuned_strategy_parameters || []).length} MARKET/CASH` : `${(effectivePlan?.tuned_parameters || effectivePlan?.search_space || catalog?.tuned_parameters || catalog?.search_space || []).length} ${tr('parameters')}`}</small>
        </div>
        <div className="model-tuning-context-card">
          <TuningContextLabel
            id="model-tuning-hint-saved-model"
            label="Saved model"
            description={temporalModelMode ? tr('The Temporal LightGBM classifiers and regressors are retrained for every challenger. Winner allocation and Temporal policy thresholds remain frozen.') : temporalPolicyMode ? tr('The Temporal LightGBM predictions and underlying Winner remain frozen. Only the Winner-Anchored timing thresholds are replayed.') : tr('Model family currently saved with the RESEARCH Strategy. Joint CARO may tune the supported LightGBM hyperparameters, while fixed model settings remain unchanged.')}
          />
          <strong>{temporalModelMode ? tr('LightGBM Temporal Intelligence') : temporalPolicyMode ? tr('Temporal Policy') : (strategy?.research_model_configuration?.label || strategy?.research_model?.label || '—')}</strong>
          <small>{temporalModelMode ? tr('Retrained per candidate') : temporalPolicyMode ? tr('LightGBM + Winner frozen') : (modelFamily || '—')}</small>
        </div>
        <div className="model-tuning-context-card worker-online">
          <TuningContextLabel
            id="model-tuning-hint-execution"
            label="Execution"
            description={temporalModelMode ? tr('Candidates retrain Temporal LightGBM sequentially using only the frozen MongoDB market snapshot. No Alpaca request occurs.') : temporalPolicyMode ? tr('Candidates replay the frozen Temporal observations and immutable Winner decisions. No LightGBM retraining, Alpaca request or new market-data load occurs.') : tr('Candidates run sequentially through the integrated API worker. Each LightGBM training still uses the CPU thread configuration saved in the model/runtime.')}
            align="right"
          />
          <strong>{tr(temporalModelMode ? 'Full Temporal LightGBM retrain' : temporalPolicyMode ? 'Frozen Temporal replay' : 'Integrated API worker')}</strong>
          <small>{tr('One candidate at a time')}</small>
        </div>
      </div>

      <div className="model-tuning-baseline model-tuning-baseline-compact">
        <div className="model-tuning-baseline-head">
          <div className="model-tuning-baseline-title">
            <span className="model-tuning-context-label">
              <span>{tr('Selected Strategy baseline')}</span>
              <ParameterHint
                id="model-tuning-hint-baseline"
                title={tr('Selected Strategy baseline')}
                description={tr(temporalModelMode ? 'The API reuses the completed Temporal run as Control, retrains challenger LightGBM models against the same frozen market snapshot, and never downloads new market data.' : temporalPolicyMode ? 'The API reuses the completed Temporal Intelligence source run and frozen replay stored with this TEMPORAL Strategy.' : 'The API uses the latest compatible completed Backtest for the selected Strategy and freezes its execution context for this campaign.')}
              />
            </span>
            <strong title={selectedBaseline?.job_id || ''}>{selectedBaseline?.job_id || tr('No compatible completed baseline execution was found.')}</strong>
          </div>
          <button type="button" className="secondary-action compact model-tuning-refresh" onClick={loadWorkspace} disabled={busy || active}>{tr('Refresh')}</button>
        </div>
        {selectedBaseline ? (
          <div className="model-tuning-baseline-metrics model-tuning-baseline-metrics-compact">
            <div><span>{tr('Capital')}</span><strong>{money(selectedBaseline.metrics?.ending_capital)}</strong></div>
            <div><span>{tr('CAGR')}</span><strong>{pct(selectedBaseline.metrics?.cagr)}</strong></div>
            <div><span>{tr('Sharpe')}</span><strong>{decimal(selectedBaseline.metrics?.sharpe)}</strong></div>
            <div><span>{tr('Max DD')}</span><strong>{pct(selectedBaseline.metrics?.maximum_drawdown)}</strong></div>
            <div><span>{tr('Worst fold')}</span><strong>{pct(selectedBaseline.metrics?.worst_fold_return)}</strong></div>
          </div>
        ) : null}
      </div>

      <div className="model-tuning-step-heading model-tuning-idle-only"><span>2</span><div><strong>{tr('Research')}</strong><small>{tr('Choose the research method and protocol. Advanced settings stay optional.')}</small></div></div>
      <div className="model-tuning-method-selector model-tuning-method-selector-compact model-tuning-idle-only">
        <label>
          <span className="model-tuning-context-label">
            <span>{tr('Research method')}</span>
            <ParameterHint
              id="model-tuning-hint-method"
              title={selectedMethod?.label || tr('Research method')}
              description={selectedMethod?.description || ''}
            />
          </span>
          <select
            value={method}
            onChange={(event) => setMethod(event.target.value)}
            disabled={!canStartTuning || active || busy}
          >
            {(catalog.methods || []).map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}
          </select>
        </label>
        <div className="model-tuning-method-summary"><strong>{selectedMethod?.label || '—'}</strong></div>
      </div>

      <div className="model-tuning-step-heading"><span>3</span><div><strong>{tr('Tuning')}</strong><small>{tr(active ? 'Campaign is running. No additional action is required unless you want to stop it.' : 'Set the research budget and start the campaign.')}</small></div></div>
      <div className="model-tuning-controls">
        <label>
          <span>{tr(probabilityMode ? 'Research budget (trials)' : 'Exploration candidates')}</span>
          <input type="number" min={catalog.candidate_count_min} max={catalog.research_budget_technical_segment_max || catalog.candidate_count_max} step="1" value={candidateCount} disabled={!canStartTuning || !canTune || active || busy} onChange={(event) => setCandidateCount(event.target.value)} />
          {probabilityMode ? <small>{tr('No fixed research ceiling. Continue Research adds another compatible budget segment without discarding prior observations.')}</small> : null}
        </label>
        <label>
          <span>{tr('Sampling seed')}</span>
          <input type="number" min="0" step="1" value={seed} disabled={!canStartTuning || !canTune || active || busy} onChange={(event) => setSeed(event.target.value)} />
        </label>
        <div className="model-tuning-control-note">
          <span>{tr('Validation')}</span>
          <strong>{tr(temporalModelMode ? 'Walk-forward · frozen market snapshot' : temporalPolicyMode ? 'Frozen Temporal replay' : 'Chronological walk-forward')}</strong>
        </div>
        {(canStartTuning || canStopTuning) ? <div className="model-tuning-actions">
          {canStartTuning ? <button type="button" className="primary-action" onClick={start} disabled={!canTune || active || busy || (temporalStrategy && !foldProtocolValid)}>{busy && !active ? tr('Starting…') : startActionLabel}</button> : null}
          {canStopTuning ? <button type="button" className="secondary-action" onClick={stop} disabled={!active || busy || run?.status === 'stop_requested'}>{run?.status === 'stop_requested' ? tr('Stopping…') : tr('Stop')}</button> : null}
        </div> : null}
      </div>

      {adaptiveMode ? (
        <details className="model-tuning-space model-tuning-advanced model-tuning-idle-only">
          <summary>{tr('Advanced CARO settings')}<span>{tr('Optional')}</span></summary>
          <div className="model-tuning-probability-config">
            <div className="model-tuning-probability-grid">
              <label><span>{tr('Minimum capital improvement (%)')}</span><input type="number" min="0" step="0.1" value={minimumCapitalImprovementPct} disabled={!canStartTuning || active || busy} onChange={(event) => setMinimumCapitalImprovementPct(event.target.value)} /></label>
              <label><span>{tr('Sharpe tolerance')}</span><input type="number" min="0" step="0.01" value={sharpeTolerance} disabled={!canStartTuning || active || busy} onChange={(event) => setSharpeTolerance(event.target.value)} /></label>
              <label><span>{tr('Drawdown tolerance (pp)')}</span><input type="number" min="0" step="0.1" value={drawdownTolerancePct} disabled={!canStartTuning || active || busy} onChange={(event) => setDrawdownTolerancePct(event.target.value)} /></label>
              <label><span>{tr('Minimum worst fold (%)')}</span><input type="number" step="0.1" value={minimumWorstFoldPct} disabled={!canStartTuning || active || busy} onChange={(event) => setMinimumWorstFoldPct(event.target.value)} /></label>
              <label className="model-tuning-checkbox-control"><span>{tr('Adaptive early stopping')}</span><input type="checkbox" checked={adaptiveStoppingEnabled} disabled={!canStartTuning || active || busy} onChange={(event) => setAdaptiveStoppingEnabled(event.target.checked)} /></label>
              <label><span>{tr('No-improvement trials')}</span><input type="number" min="10" step="10" value={noImprovementTrialLimit} disabled={!canStartTuning || active || busy || !adaptiveStoppingEnabled} onChange={(event) => setNoImprovementTrialLimit(event.target.value)} /></label>
              <label><span>{tr('Meaningful improvement (%)')}</span><input type="number" min="0" step="0.05" value={minimumMeaningfulImprovementPct} disabled={!canStartTuning || active || busy || !adaptiveStoppingEnabled} onChange={(event) => setMinimumMeaningfulImprovementPct(event.target.value)} /></label>
            </div>
          </div>
        </details>
      ) : null}

      <details className="model-tuning-space model-tuning-idle-only">
        <summary>{tr('Search space')}<span>{(effectivePlan?.search_space || catalog.search_space || []).length} {tr('parameters')}</span></summary>
        <div className="model-tuning-space-grid">
          {(effectivePlan?.search_space || catalog.search_space || []).map((field) => (
            <div key={field.name}>
              <strong>{field.name}</strong>
              <span>{field.min} → {field.max}</span>
            </div>
          ))}
        </div>
      </details>

      <div className="model-tuning-step-heading model-tuning-results-step-heading"><span>4</span><div><strong>{tr('Results')}</strong><small>{tr(active ? 'Live campaign status and current challengers.' : 'Compare challengers with the selected baseline Strategy.')}</small></div></div>

      {run ? (
        <div className="model-tuning-run">
          <div className="model-tuning-progress-row">
            <div>
              <strong>{tr('Campaign')} {run.id}</strong>
              <span>{tr(run.status)} · {run.research_completed_candidates ?? run.completed_candidates}/{run.research_total_candidates ?? run.total_candidates} {tr('completed')}{run.cancelled_candidates ? ` · ${run.cancelled_candidates} ${tr('cancelled')}` : ''} · {tr(run.tuning_scope_label || catalog?.tuning_scope_label || '')} · {(catalog.methods || []).find((item) => item.id === run.method)?.label || run.method}</span>
              <span>{tr('Created at')} {run.created_at ? new Date(run.created_at).toLocaleString() : '—'}{run.created_by ? ` · ${tr('Started by')} ${run.created_by}` : ''}</span>
            </div>
            <div className="model-tuning-run-actions">
              {canViewTuningLogs ? <button type="button" className="secondary-action compact" onClick={openCampaignLog} disabled={logLoading}>{tr('Campaign log')}</button> : null}
              {canExportTuning && !active ? <button type="button" className="secondary-action compact" onClick={exportCampaign} disabled={exporting}>{tr(exporting ? 'Exporting…' : 'Export Campaign')}</button> : null}
              {canStartTuning && temporalStrategy && run.status === 'completed' && run.method === PROBABILITY_METHOD ? <button type="button" className="secondary-action compact" onClick={continueResearch} disabled={busy || !tuningStartContractCompatible || !foldProtocolValid || !continuationResearchFoldsCompatible}>{tr('Continue Research')}</button> : null}
              {run.validation_processing_id ? <button type="button" className="secondary-action compact" onClick={viewChampionAnalytics}>{tr('View Analytics')}</button> : null}
              <strong>{Number(run.progress || 0).toFixed(1)}%</strong>
            </div>
          </div>
          <div className="model-tuning-progress-track"><i style={{ width: `${Math.max(0, Math.min(100, Number(run.progress || 0)))}%` }} /></div>
          {active && currentBestCandidate ? (
            <div className="model-tuning-current-best">
              <div><span>{tr('Current best')}</span><strong>{candidateLabel(currentBestCandidate)}</strong></div>
              <div><span>{tr('Capital')}</span><strong>{money(currentBestCandidate.metrics?.ending_capital)}</strong></div>
              <div><span>{tr('vs selected baseline')}</span><strong className={signedMetricTone(currentBestImprovement)}>{currentBestImprovement == null ? '—' : pct(currentBestImprovement)}</strong></div>
            </div>
          ) : null}
          {run.tuning_scope === 'temporal_model' && run.current_candidate_id != null ? (
            <div className="model-tuning-current-candidate-progress">
              <span>{tr('Candidate')} #{run.current_candidate_id} · {tr(run.current_candidate_stage || 'Training Temporal LightGBM')}</span>
              <strong>{Number(run.current_candidate_progress || 0).toFixed(1)}%</strong>
            </div>
          ) : null}
          {run.baseline_execution ? <small>{tr('Baseline')} · {run.baseline_execution.job_id} · {money(run.baseline_execution.metrics?.ending_capital)}</small> : null}
          {run.probability_anchor ? <small>{tr('Champion anchor')} · {run.probability_anchor.candidate_id !== undefined ? `#${run.probability_anchor.candidate_id} · ` : ''}{money(run.probability_anchor.metrics?.ending_capital)} · {run.imported_observation_count || 0} {tr('imported observations')}</small> : null}
          {run.method === PROBABILITY_METHOD && run.probability_state ? <small>{tr('Unified state')} · {tr('Champion')} #{run.probability_state.last_champion_candidate_id ?? run.probability_anchor?.candidate_id ?? 0} · {tr('Exploration trials')} {run.probability_state.exploration_trials_completed || 0} · {tr('Adaptive trials')} {run.probability_state.adaptive_trials_completed || 0} · {tr('Trust region')} {(Number(run.probability_state.trust_region_radius || 0) * 100).toFixed(1)}% · {tr('No-improvement streak')} {run.probability_state.no_improvement_streak || 0}</small> : null}
          {run.market_data_cutoff_date ? <small>{tr('Frozen market-data cutoff')} · {run.market_data_cutoff_date}</small> : null}
          {run.adaptive_early_stopped ? <small>{tr(run.adaptive_early_stop_reason || 'Adaptive early stopping completed the campaign after convergence.')}</small> : null}
          {run.status === 'stop_requested' ? <small>{tr(run?.tuning_scope === 'temporal_model' ? 'Cancelling the active Temporal LightGBM candidate at the next model checkpoint. Partial research artifacts will be discarded.' : 'Cancelling the active tuning candidate now. Partial research artifacts will be discarded.')}</small> : null}
          {run.active_candidate_ids?.length ? <small>{tr(run.status === 'stop_requested' ? 'Cancelling candidate' : 'Active candidates')} · {run.active_candidate_ids.map((id) => `#${id}`).join(', ')}</small> : null}
        </div>
      ) : null}

      {visibleCandidates.length ? (
        <div className="model-tuning-results-wrap">
          <div className="model-tuning-results-heading">
            <div><strong>{tr('Candidate ranking')}</strong></div>
            {runMatchesCurrentBaseline && run?.best_candidate_id !== null && run?.best_candidate_id !== undefined ? <small>{tr('Best')} #{run.best_candidate_id}</small> : null}
          </div>
          <div className="model-tuning-candidate-grid" aria-label={tr('Candidate ranking')}>
            {visibleCandidates.map((candidate) => {
              const metrics = candidate.metrics || {}
              const proposal = candidate.proposal || {}
              const adoptable = !active && candidate.status === 'completed' && !candidate.is_control
              const previouslyPromoted = (run?.adoption_history || []).some((item) => Number(item.candidate_id) === Number(candidate.candidate_id))
              const isFinalChampion = !active && candidate.status === 'completed' && !candidate.is_control && Number(run?.best_candidate_id) === Number(candidate.candidate_id)
              const candidateValidation = candidate.validation || null
              const candidateCertification = candidate.certification || null
              const validationStatus = String(candidateValidation?.status || '').toLowerCase()
              const validationRunning = ['queued', 'running'].includes(validationStatus)
              const validationCompleted = validationStatus === 'completed'
              const validationTechnicalFailure = validationStatus === 'failed'
              const canValidateFinalist = temporalPolicyMode && canPromoteTuning && !active && candidate.status === 'completed' && !candidate.is_control && (!candidateValidation || validationTechnicalFailure)
              const canCertifyCandidate = temporalPolicyMode && canPromoteTuning && !active && validationCompleted && Boolean(candidateValidation?.passed) && !candidateCertification
              const typeLabel = candidate.is_control
                ? tr('Control')
                : candidate.kind === 'champion_probability'
                  ? tr('Adaptive')
                  : candidate.kind === 'unified_exploration'
                    ? tr('Exploration')
                    : tr('Candidate')
              const tone = candidate.is_control
                ? 'control'
                : candidate.rank === 1
                  ? 'best'
                  : candidate.kind === 'champion_probability'
                    ? 'adaptive'
                    : candidate.kind === 'unified_exploration'
                      ? 'exploration'
                      : ''
              const status = candidate.status || 'unknown'
              const hasExecutionStatusIndicator = !candidate.is_control && ['probability_startup', 'unified_exploration', 'champion_probability'].includes(candidate.kind)
              const statusLabelKey = {
                running: 'Running',
                queued: 'Queued',
                pending: 'Pending',
                completed: 'Completed',
                failed: 'Failed',
                cancelled: 'Cancelled',
              }[status] || 'Status'
              const persistedJobProgress = Math.max(0, Math.min(100, Number(candidate.job_progress || 0)))
              const liveCurrentCandidate = candidate.status === 'running'
                && run?.current_candidate_id != null
                && Number(run.current_candidate_id) === Number(candidate.candidate_id)
              const liveCurrentCandidateProgress = Math.max(0, Math.min(100, Number(run?.current_candidate_progress || 0)))
              const jobProgress = liveCurrentCandidate
                ? Math.max(persistedJobProgress, liveCurrentCandidateProgress)
                : persistedJobProgress
              const jobProgressLabel = `${jobProgress.toFixed(0)}%`
              const statusTitle = candidate.status === 'running'
                ? `${tr(statusLabelKey)} · ${jobProgressLabel}`
                : tr(statusLabelKey)
              const hasParameters = Object.keys(candidate.settings || {}).length > 0
              const showResults = candidate.is_control || ['completed', 'failed', 'cancelled'].includes(status)
              return (
                <article key={candidate.candidate_id} className={`model-tuning-candidate-card ${tone} ${status}`}>
                  <header className="model-tuning-candidate-card-header">
                    <div className="model-tuning-candidate-card-title model-tuning-candidate-card-title-header">
                      <div className="model-tuning-candidate-name-row">
                        {hasExecutionStatusIndicator && status === 'running' ? <span className="model-tuning-caro-status-loader" aria-hidden="true" /> : null}
                        {hasExecutionStatusIndicator && status === 'completed' ? <span className="model-tuning-caro-status-complete" aria-hidden="true" /> : null}
                        <strong className={`model-tuning-candidate-name ${status}`} title={candidateLabel(candidate)}>{candidateLabel(candidate)}</strong>
                      </div>
                      <small>
                        {candidate.is_control && candidate.baseline_reused
                          ? `${tr('Certified Backtest reused')} · ${candidate.source_job_id || candidate.job_id || '—'}`
                          : `#${candidate.candidate_id} · ${typeLabel}`}
                      </small>
                    </div>
                    {candidate.status === 'running' ? (
                      <span
                        className="loader"
                        role="status"
                        aria-label={statusTitle}
                        title={statusTitle}
                      >
                        <span className="loader-percent" aria-hidden="true">{jobProgressLabel}</span>
                      </span>
                    ) : null}
                  </header>


                  {!candidate.is_control && ['pending', 'queued', 'running'].includes(status) && candidateCardMethod === PROBABILITY_METHOD && candidate.kind === 'champion_probability' ? (
                    <div className="model-tuning-candidate-preflight">
                      <CandidateCardMetric candidateId={candidate.candidate_id} label="P(beat)" value={pct(proposal.estimated_probability_beats_champion)} tone={probabilityMetricTone(proposal.estimated_probability_beats_champion)} />
                      <CandidateCardMetric candidateId={candidate.candidate_id} label="Expected improvement" value={pct(proposal.estimated_expected_improvement)} tone={signedMetricTone(proposal.estimated_expected_improvement)} />
                    </div>
                  ) : null}

                  {showResults ? (
                    <div className="model-tuning-candidate-metrics">
                      <CandidateCardMetric candidateId={candidate.candidate_id} label="Capital" value={money(metrics.ending_capital)} tone="capital" />
                      <CandidateCardMetric candidateId={candidate.candidate_id} label="CAGR" value={pct(metrics.cagr)} tone={signedMetricTone(metrics.cagr)} />
                      <CandidateCardMetric candidateId={candidate.candidate_id} label="Sharpe" value={decimal(metrics.sharpe)} tone={signedMetricTone(metrics.sharpe)} />
                      <CandidateCardMetric candidateId={candidate.candidate_id} label="Max DD" value={pct(metrics.maximum_drawdown)} tone={signedMetricTone(metrics.maximum_drawdown)} />
                      <CandidateCardMetric candidateId={candidate.candidate_id} label="Worst fold" value={pct(metrics.worst_fold_return)} tone={signedMetricTone(metrics.worst_fold_return)} />
                      <CandidateCardMetric candidateId={candidate.candidate_id} label="Market Exposure" value={pct(metrics.market_exposure)} tone="info" />
                      <CandidateCardMetric candidateId={candidate.candidate_id} label="CASH Days" value={metrics.cash_days == null ? '—' : Number(metrics.cash_days).toFixed(0)} tone="cash" />
                      {candidateCardMethod === PROBABILITY_METHOD && !candidate.is_control ? <CandidateCardMetric candidateId={candidate.candidate_id} label="P(beat)" value={pct(proposal.estimated_probability_beats_champion)} tone={probabilityMetricTone(proposal.estimated_probability_beats_champion)} /> : null}
                      {candidateCardMethod === PROBABILITY_METHOD && !candidate.is_control ? <CandidateCardMetric candidateId={candidate.candidate_id} label="Expected improvement" value={pct(proposal.estimated_expected_improvement)} tone={signedMetricTone(proposal.estimated_expected_improvement)} /> : null}
                      <CandidateCardMetric candidateId={candidate.candidate_id} label="Score" value={decimal(metrics.risk_adjusted_compound_score, 4)} tone={signedMetricTone(metrics.risk_adjusted_compound_score)} />
                    </div>
                  ) : null}

                  {candidate.status === 'failed' ? <small className="model-tuning-candidate-card-error">{candidate.failure_type || candidate.failure_message || tr('See log')}</small> : null}
                  {validationRunning ? (
                    <div className="model-tuning-validation-progress">
                      <div>
                        <span>{tr('Validating')} · {candidateValidation.fold_count || run?.fold_protocol?.validation_folds || validationFolds} {tr('folds')} · {tr(candidateValidation.stage || 'Starting validation')}</span>
                        <strong>{Number(candidateValidation.progress || 0).toFixed(1)}%</strong>
                      </div>
                      <div className="model-tuning-progress-track"><i style={{ width: `${Math.max(0, Math.min(100, Number(candidateValidation.progress || 0)))}%` }} /></div>
                    </div>
                  ) : null}
                  {validationTechnicalFailure ? <small className="model-tuning-candidate-card-error">{tr('Validation processing failed')} · {candidateValidation.failure_message || tr('See log')}</small> : null}

                  <footer className="model-tuning-candidate-card-actions">
                    {hasParameters ? <button type="button" onClick={() => setParameterCandidateId(candidate.candidate_id)}>{tr('Parameters')}</button> : null}
                    <button type="button" onClick={() => setSelectedCandidateId(candidate.candidate_id)}>{tr('View')}</button>
                    {canViewTuningLogs && !candidate.baseline_preview ? <button type="button" onClick={() => openCandidateLog(candidate)} disabled={logLoading}>{tr('Log')}</button> : null}
                    {previouslyPromoted && !temporalTarget ? <span className="model-tuning-adopted">{tr('Promoted')}</span> : null}
                    {temporalModelMode && canPromoteTuning && isFinalChampion ? <button type="button" onClick={() => adopt(candidate)} disabled={busy}>{tr('Continue to Policy Tuning')}</button> : null}
                    {canValidateFinalist ? <button type="button" onClick={() => validateFinalist(candidate)} disabled={busy}>{tr(validationTechnicalFailure ? 'Retry Validation' : 'Validate')} · {run?.fold_protocol?.validation_folds || validationFolds} {tr('folds')}</button> : null}
                    {validationCompleted ? <span className={`model-tuning-adopted ${candidateValidation.passed ? '' : 'failed'}`}>{tr(candidateValidation.passed ? 'Validation passed' : 'Validation failed')} · {candidateValidation.fold_count || run?.fold_protocol?.validation_folds || validationFolds}</span> : null}
                    {validationCompleted && candidateValidation?.processing_id ? <button type="button" onClick={() => viewProcessing(candidateValidation.processing_id)}>{tr('Validation Analytics')}</button> : null}
                    {canCertifyCandidate ? <button type="button" onClick={() => certifyCandidate(candidate)} disabled={busy}>{tr('Certify')} · {run?.fold_protocol?.certification_folds || certificationFolds} {tr('folds')}</button> : null}
                    {candidateCertification ? <span className={`model-tuning-adopted ${candidateCertification.passed ? '' : 'failed'}`}>{tr(candidateCertification.passed ? 'Certification passed' : 'Certification failed')} · {candidateCertification.fold_count || run?.fold_protocol?.certification_folds || certificationFolds}</span> : null}
                    {candidateCertification?.processing_id ? <button type="button" onClick={() => viewProcessing(candidateCertification.processing_id)}>{tr('Certification Analytics')}</button> : null}
                    {!temporalTarget && canPromoteTuning && adoptable ? <button type="button" onClick={() => adopt(candidate)} disabled={busy}>{tr(previouslyPromoted ? 'Promote again' : 'Promote to Backtest')}</button> : null}
                  </footer>
                </article>
              )
            })}
          </div>



        </div>
      ) : null}

      {selectedCandidate ? (
        <div
          className="model-tuning-candidate-detail-overlay"
          role="presentation"
          onMouseDown={(event) => { if (event.target === event.currentTarget) setSelectedCandidateId(null) }}
        >
          <div className="model-tuning-candidate-detail-dialog" role="dialog" aria-modal="true" aria-label={`${tr('View')} · ${candidateLabel(selectedCandidate)}`}>
            <div className="model-tuning-candidate-detail-body">
              <div className="model-tuning-results-heading">
                <div><strong>{candidateLabel(selectedCandidate)}</strong><span>{selectedCandidate.settings_hash}</span></div>
                <button type="button" className="secondary-action compact" onClick={() => setSelectedCandidateId(null)}>{tr('Close')}</button>
              </div>
              {selectedCandidate.proposal ? (
                <div className="model-tuning-proposal-grid">
                  <div><span>{tr('Estimated P(beat Champion)')}</span><strong>{pct(selectedCandidate.proposal.estimated_probability_beats_champion)}</strong></div>
                  <div><span>{tr('Expected improvement')}</span><strong>{pct(selectedCandidate.proposal.estimated_expected_improvement)}</strong></div>
                  <div><span>{tr('Predicted capital')}</span><strong>{money(selectedCandidate.proposal.estimated_ending_capital_mean)}</strong></div>
                  <div><span>{tr('Prediction spread')}</span><strong>{money(selectedCandidate.proposal.estimated_ending_capital_std)}</strong></div>
                  <div><span>{tr('Predicted Sharpe')}</span><strong>{decimal(selectedCandidate.proposal.estimated_sharpe_mean)}</strong></div>
                  <div><span>{tr('Predicted max DD')}</span><strong>{pct(selectedCandidate.proposal.estimated_maximum_drawdown_mean)}</strong></div>
                  <div><span>{tr('Predicted worst fold')}</span><strong>{pct(selectedCandidate.proposal.estimated_worst_fold_mean)}</strong></div>
                  <div><span>{tr('Observations used')}</span><strong>{selectedCandidate.proposal.observation_count ?? '—'}</strong></div>
                  <div><span>{tr('Acquisition score')}</span><strong>{decimal(selectedCandidate.proposal.acquisition_score, 5)}</strong></div>
                </div>
              ) : null}
              {selectedCandidate.proposal?.promising_region ? (
                <div className="model-tuning-promising-region">
                  <strong>{tr('Promising region')}</strong>
                  <div className="model-tuning-settings-grid">
                    {Object.entries(selectedCandidate.proposal.promising_region).map(([name, bounds]) => (
                      <div key={name}><span>{name}</span><strong>{String(bounds?.low ?? '—')} → {String(bounds?.high ?? '—')}</strong></div>
                    ))}
                  </div>
                </div>
              ) : null}
              {selectedCandidate.status === 'failed' ? (
                <div className="model-tuning-candidate-failure">
                  <strong>{selectedCandidate.failure_type || tr('Candidate failed')}</strong>
                  <span>{selectedCandidate.failure_message || selectedCandidate.error || tr('Open the execution log for the technical details.')}</span>
                  {canViewTuningLogs && !selectedCandidate.baseline_preview ? <button type="button" className="secondary-action compact" onClick={() => openCandidateLog(selectedCandidate)} disabled={logLoading}>{tr('Open log')}</button> : null}
                </div>
              ) : null}
              {gateTuning && selectedCandidate.metrics ? (
                <div className="model-tuning-proposal-grid">
                  <div><span>{tr('Market Exposure')}</span><strong>{pct(selectedCandidate.metrics.market_exposure)}</strong></div>
                  <div><span>{tr('CASH Days')}</span><strong>{selectedCandidate.metrics.cash_days == null ? '—' : Number(selectedCandidate.metrics.cash_days).toFixed(0)}</strong></div>
                  <div><span>{tr('Cash Gate Overrides')}</span><strong>{selectedCandidate.metrics.cash_gate_changed_base_action_sessions ?? '—'}</strong></div>
                  <div><span>{tr('Net Cash-Gate Diagnostic')}</span><strong>{pct(selectedCandidate.metrics.cash_gate_net_avoided_return_sum)}</strong></div>
                </div>
              ) : null}
              {selectedCandidate.metrics?.folds?.length ? (
                <div className="model-tuning-fold-grid">
                  {selectedCandidate.metrics.folds.map((fold) => <div key={fold.fold_id}><span>{tr('Fold')} {fold.fold_id}</span><strong>{pct(fold.strategy_return)}</strong><small>{tr('Max DD')} {pct(fold.maximum_drawdown)}</small></div>)}
                </div>
              ) : null}
            
            </div>
          </div>
        </div>
      ) : null}

      {parameterCandidate ? (
        <div
          className="model-tuning-parameters-overlay"
          role="presentation"
          onMouseDown={(event) => { if (event.target === event.currentTarget) setParameterCandidateId(null) }}
        >
          <div className="model-tuning-parameters-dialog" role="dialog" aria-modal="true" aria-label={`${tr('Parameters')} · ${candidateLabel(parameterCandidate)}`}>
            <div className="model-tuning-parameters-dialog-heading">
              <div>
                <span>{tr('Parameters')}</span>
                <strong>{candidateLabel(parameterCandidate)}</strong>
                {parameterCandidate.settings_hash ? <small>{parameterCandidate.settings_hash}</small> : null}
              </div>
              <button type="button" className="secondary-action compact" onClick={() => setParameterCandidateId(null)}>{tr('Close')}</button>
            </div>
            <CandidateParametersGrid settings={parameterCandidate.settings} />
          </div>
        </div>
      ) : null}

      {(logView || logLoading || logError) ? (
        <div className="model-tuning-log-overlay" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) { setLogView(null); setLogError('') } }}>
          <div className="model-tuning-log-dialog" role="dialog" aria-modal="true" aria-label={logView?.title || tr('Diagnostic log')}>
            <div className="model-tuning-log-heading">
              <div><strong>{logView?.title || tr('Diagnostic log')}</strong><span>{logView?.run_id || run?.id || '—'}</span></div>
              <button type="button" className="secondary-action compact" onClick={() => { setLogView(null); setLogError('') }}>{tr('Close')}</button>
            </div>
            {logLoading ? <div className="backtest-loading-row">{tr('Loading diagnostic log…')}</div> : null}
            {logError ? <div className="global-inline-message error-inline">{logError}</div> : null}
            {logView ? (
              <>
                <div className="model-tuning-log-meta">
                  <span>{tr('Status')} <strong>{tr(logView.status || 'unknown')}</strong></span>
                  {logView.candidate_id !== undefined ? <span>{tr('Candidate')} <strong>#{logView.candidate_id}</strong></span> : null}
                  {logView.job_id ? <span>Job <strong>{logView.job_id}</strong></span> : null}
                  {logView.failure_type ? <span>{tr('Failure')} <strong>{logView.failure_type}</strong></span> : null}
                </div>
                <pre className="model-tuning-log-pre">{logView.log_text || tr('No diagnostic lines were recorded.')}</pre>
                <div className="model-tuning-log-actions">
                  <button type="button" onClick={copyDiagnosticLog}>{tr('Copy log')}</button>
                  <button type="button" className="secondary-action" onClick={downloadDiagnosticLog}>{tr('Download .txt')}</button>
                </div>
              </>
            ) : null}
          </div>
        </div>
      ) : null}
    </section>
  )
}
