import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { ApiError, apiFetch, downloadFile } from '../api/http'
import { hasCapability } from '../auth/capabilities'
import { API } from '../config/env'
import { tr } from '../i18n/runtime'
import { money, number, percent, shortDateTime } from '../shared/formatters'

const ACTIVE = new Set(['queued', 'running'])
const DIAGNOSTIC_ACTIVE = new Set(['queued', 'running', 'stop_requested'])

function toInput(value) {
  return value === null || value === undefined ? '' : String(value)
}

function numberValue(value, fallback = 0) {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : fallback
}

function optionalNumber(value) {
  if (value === '' || value === null || value === undefined) return null
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

function numberListText(values) {
  return Array.isArray(values) ? values.join(', ') : ''
}

function parseNumberList(value) {
  return String(value || '')
    .split(/[\s,;]+/)
    .map((item) => item.trim())
    .filter(Boolean)
    .map(Number)
    .filter(Number.isFinite)
}

function manualCandidateText(values) {
  if (!Array.isArray(values)) return ''
  return values.map((item) => [item.drawdown_trigger, item.rotation_score_tolerance, item.challenger_quality_floor].filter((value) => value !== null && value !== undefined).join(', ')).join('\n')
}

function parseManualCandidates(value) {
  return String(value || '')
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const [drawdown, tolerance, qualityFloor] = line.split(/[\s,;]+/).filter(Boolean).map(Number)
      return { drawdown_trigger: drawdown, rotation_score_tolerance: tolerance, ...(Number.isFinite(qualityFloor) ? { challenger_quality_floor: qualityFloor } : {}) }
    })
    .filter((item) => Number.isFinite(item.drawdown_trigger) && Number.isFinite(item.rotation_score_tolerance))
}

function statusLabel(value) {
  const normalized = String(value || '').toLowerCase()
  if (normalized === 'completed') return tr('Completed')
  if (normalized === 'running') return tr('Running')
  if (normalized === 'queued') return tr('Queued')
  if (normalized === 'failed') return tr('Failed')
  if (normalized === 'stopped') return tr('Stopped')
  if (normalized === 'stop_requested') return tr('Stop requested')
  return value || '—'
}

function preferredDiagnosticCandidateId(evidence) {
  const bestId = evidence?.best_validated_candidate?.candidate_id
  if (bestId) return bestId
  const candidates = Array.isArray(evidence?.candidates) ? [...evidence.candidates] : []
  candidates.sort((a, b) => Number(b.ending_capital || 0) - Number(a.ending_capital || 0))
  return candidates[0]?.candidate_id || ''
}

function evidenceWorkflowState(item) {
  if (!item) return '—'
  const status = String(item.status || '').toLowerCase()
  if (status !== 'completed') return statusLabel(item.status)
  if (item.passing_candidate_count === null || item.passing_candidate_count === undefined) return 'Completed'
  return Number(item.passing_candidate_count) > 0 ? 'PASS' : 'FAIL'
}

function EvidenceBadge({ passed }) {
  if (passed === null || passed === undefined) return null
  return <span className={`rotation-quality-badge ${passed ? 'pass' : 'fail'}`}>{tr(passed ? 'PASS' : 'FAIL')}</span>
}

function NumericField({ label, value, onChange, disabled, min, max, step = 'any' }) {
  return (
    <label className="rotation-quality-field">
      <span>{tr(label)}</span>
      <input type="number" value={value} min={min} max={max} step={step} disabled={disabled} onChange={(event) => onChange(event.target.value)} />
    </label>
  )
}

function Metric({ label, value, tone = '' }) {
  return <div className={`rotation-quality-metric ${tone}`}><span>{tr(label)}</span><strong>{value}</strong></div>
}

function WorkflowStep({ label, state, tone = '' }) {
  return (
    <div className={`rotation-quality-workflow-step ${tone}`}>
      <span>{tr(label)}</span>
      <strong>{tr(state || '—')}</strong>
    </div>
  )
}

function ResearchResult({ research }) {
  const best = research?.best_candidate
  const control = research?.control
  if (!best || !control) return null
  return (
    <div className="rotation-quality-result-grid">
      <Metric label="Control capital" value={money(control.replayed_ending_capital)} />
      <Metric label="Best candidate" value={best.candidate_id || '—'} />
      <Metric label="Candidate capital" value={money(best.ending_capital)} tone={Number(best.capital_lift_vs_control) >= 0 ? 'positive' : 'negative'} />
      <Metric label="Capital lift" value={percent(best.capital_lift_vs_control, 2)} tone={Number(best.capital_lift_vs_control) >= 0 ? 'positive' : 'negative'} />
      <Metric label="Sharpe" value={number(best.sharpe, 4)} />
      <Metric label="Max Drawdown" value={percent(best.max_drawdown, 2)} />
      <Metric label="Robust candidates" value={number(research.robust_candidate_count, 0)} />
      <Metric label="Research folds" value={number(research.source_fold_count, 0)} />
      <Metric label="Switches" value={number(best.switch_count, 0)} />
      {best.challenger_quality_floor == null ? null : <Metric label="Challenger quality floor" value={number(best.challenger_quality_floor, 4)} />}
      {best.strong_challenger_overrides == null ? null : <Metric label="Strong challenger overrides" value={number(best.strong_challenger_overrides, 0)} />}
    </div>
  )
}

function EvidenceResult({ validation }) {
  if (!validation?.control || !Array.isArray(validation?.candidates)) return null
  return (
    <div className="rotation-quality-evidence-result">
      <div className="rotation-quality-result-grid">
        <Metric label="Control capital" value={money(validation.control.ending_capital)} />
        <Metric label="Passing candidates" value={number(validation.passing_candidate_count, 0)} />
        <Metric label="Folds" value={number(validation.fold_count, 0)} />
        <Metric label="Required fold wins" value={number(validation.validation_policy?.required_fold_wins ?? validation.required_fold_wins, 0)} />
      </div>
      <div className="rotation-quality-table-shell">
        <table className="rotation-quality-table">
          <thead><tr><th>{tr('Candidate')}</th><th>{tr('Capital')}</th><th>{tr('Lift')}</th><th>{tr('Sharpe')}</th><th>{tr('Max Drawdown')}</th><th>{tr('Fold wins')}</th><th>{tr('Result')}</th></tr></thead>
          <tbody>
            {validation.candidates.map((candidate) => (
              <tr key={candidate.candidate_id}>
                <td><strong>{candidate.candidate_id}</strong></td>
                <td>{money(candidate.ending_capital)}</td>
                <td className={Number(candidate.capital_lift_vs_control) >= 0 ? 'positive' : 'negative'}>{percent(candidate.capital_lift_vs_control, 2)}</td>
                <td>{number(candidate.sharpe, 4)}</td>
                <td>{percent(candidate.max_drawdown, 2)}</td>
                <td>{candidate.folds_beating_control}/{candidate.fold_count}</td>
                <td><EvidenceBadge passed={candidate.validation_pass} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function DiagnosticResult({ diagnostic }) {
  if (!diagnostic || String(diagnostic.status || '').toLowerCase() !== 'completed') return null
  const rows = Array.isArray(diagnostic.top_feature_separation) ? diagnostic.top_feature_separation : []
  const folds = Array.isArray(diagnostic.fold_summary) ? diagnostic.fold_summary : []
  return (
    <div className="rotation-quality-diagnostic-result">
      <div className="rotation-quality-result-grid">
        <Metric label="Blocked rotations" value={number(diagnostic.blocked_rotations, 0)} />
        <Metric label="Helpful blocks" value={number(diagnostic.helpful_blocks, 0)} tone="positive" />
        <Metric label="Harmful blocks" value={number(diagnostic.harmful_blocks, 0)} tone="negative" />
        <Metric label="Neutral blocks" value={number(diagnostic.neutral_blocks, 0)} />
        <Metric label="Helpful rate" value={percent(diagnostic.helpful_rate_excluding_neutral, 2)} />
        <Metric label="Immediate net benefit" value={money(diagnostic.immediate_net_rotation_benefit_dollars)} tone={Number(diagnostic.immediate_net_rotation_benefit_dollars) >= 0 ? 'positive' : 'negative'} />
      </div>
      {folds.length ? <div className="rotation-quality-table-shell">
        <table className="rotation-quality-table">
          <thead><tr><th>{tr('Fold')}</th><th>{tr('Blocked')}</th><th>{tr('Helpful')}</th><th>{tr('Harmful')}</th><th>{tr('Neutral')}</th><th>{tr('Immediate net benefit')}</th></tr></thead>
          <tbody>{folds.map((item) => <tr key={item.fold_id}><td>{item.fold_id}</td><td>{item.blocked_rotations}</td><td>{item.helpful}</td><td>{item.harmful}</td><td>{item.neutral}</td><td className={Number(item.immediate_net_rotation_benefit_dollars) >= 0 ? 'positive' : 'negative'}>{money(item.immediate_net_rotation_benefit_dollars)}</td></tr>)}</tbody>
        </table>
      </div> : null}
      {rows.length ? <div className="rotation-quality-table-shell">
        <table className="rotation-quality-table diagnostic-features">
          <thead><tr><th>{tr('Feature')}</th><th>{tr('Metric')}</th><th>{tr('Helpful mean')}</th><th>{tr('Harmful mean')}</th><th>{tr('Std. separation')}</th><th>{tr('Helpful direction')}</th></tr></thead>
          <tbody>{rows.map((item) => <tr key={item.engineered_metric}><td>{item.feature}</td><td>{item.engineered_metric}</td><td>{number(item.helpful_mean, 5)}</td><td>{number(item.harmful_mean, 5)}</td><td>{number(item.standardized_separation, 4)}</td><td>{tr(item.helpful_direction)}</td></tr>)}</tbody>
        </table>
      </div> : null}
    </div>
  )
}

export function TemporalRotationQualityPanel({ capabilities = {}, onSessionExpired, sourceRun = null }) {
  const canManage = hasCapability(capabilities, 'research.manage')
  const canExport = hasCapability(capabilities, 'temporal_intelligence.export')
  const [config, setConfig] = useState(null)
  const [researchForm, setResearchForm] = useState(null)
  const [researches, setResearches] = useState([])
  const [research, setResearch] = useState(null)
  const [candidates, setCandidates] = useState([])
  const [selectedCandidates, setSelectedCandidates] = useState([])
  const [validations, setValidations] = useState([])
  const [validation, setValidation] = useState(null)
  const [validationForm, setValidationForm] = useState(null)
  const [certificationForm, setCertificationForm] = useState(null)
  const [diagnosticForm, setDiagnosticForm] = useState(null)
  const [diagnostics, setDiagnostics] = useState([])
  const [diagnostic, setDiagnostic] = useState(null)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [error, setError] = useState('')
  const timerRef = useRef(null)

  const handleError = useCallback((requestError) => {
    if (requestError instanceof ApiError && requestError.status === 401) {
      onSessionExpired?.()
      return
    }
    if (requestError instanceof ApiError && requestError.status === 403) {
      setError('')
      return
    }
    setError(tr(requestError?.message || 'Unable to manage Rotation Quality research.'))
  }, [onSessionExpired])

  const buildFormsFromConfig = useCallback((nextConfig) => {
    const defaults = nextConfig?.defaults || {}
    const caro = defaults.caro || {}
    const grid = defaults.grid || {}
    const strong = defaults.strong_challenger_override || {}
    const gate = defaults.research_gate || {}
    setResearchForm((current) => current || {
      source_run_id: sourceRun?.id || '',
      search_method: defaults.search_method || nextConfig?.search_methods?.[0]?.id || '',
      focus_month: defaults.focus_month || '',
      control_tolerance_usd: toInput(defaults.control_tolerance_usd),
      strong_challenger_override: Boolean(strong.enabled),
      baseline_drawdown_trigger: toInput(strong.baseline_drawdown_trigger),
      baseline_rotation_score_tolerance: toInput(strong.baseline_rotation_score_tolerance),
      challenger_quality_floors: numberListText(strong.challenger_quality_floors),
      drawdown_triggers: numberListText(grid.drawdown_triggers),
      rotation_score_tolerances: numberListText(grid.rotation_score_tolerances),
      manual_candidates: manualCandidateText(defaults.manual_candidates),
      caro: Object.fromEntries(Object.entries(caro).map(([key, value]) => [key, toInput(value)])),
      research_gate: Object.fromEntries(Object.entries(gate).map(([key, value]) => [key, toInput(value)])),
    })
    const validationDefaults = defaults.validation || {}
    const certificationDefaults = defaults.certification || {}
    setValidationForm((current) => current || Object.fromEntries(Object.entries(validationDefaults).map(([key, value]) => [key, toInput(value)])))
    setCertificationForm((current) => current || Object.fromEntries(Object.entries(certificationDefaults).map(([key, value]) => [key, toInput(value)])))
    const diagnosticDefaults = nextConfig?.diagnostics?.defaults || {}
    setDiagnosticForm((current) => current || {
      candidate_id: '',
      lookback_sessions: toInput(diagnosticDefaults.lookback_sessions),
      feature_names: Array.isArray(diagnosticDefaults.feature_names) ? [...diagnosticDefaults.feature_names] : [],
      minimum_group_samples: toInput(diagnosticDefaults.minimum_group_samples),
      outcome_neutral_band: toInput(diagnosticDefaults.outcome_neutral_band),
      top_feature_count: toInput(diagnosticDefaults.top_feature_count),
    })
  }, [sourceRun?.id])

  const loadResearchHistory = useCallback(async () => {
    const payload = await apiFetch(`${API}/temporal-rotation-quality-research?limit=30`)
    const items = Array.isArray(payload?.items) ? payload.items : []
    setResearches(items)
    return items
  }, [])

  const loadWorkspace = useCallback(async () => {
    setLoading(true)
    try {
      const [nextConfig, items] = await Promise.all([
        apiFetch(`${API}/temporal-rotation-quality-research/config`),
        loadResearchHistory(),
      ])
      setConfig(nextConfig)
      buildFormsFromConfig(nextConfig)
      const active = items.find((item) => ACTIVE.has(String(item.status || '').toLowerCase()))
      const latest = active || items[0] || null
      if (latest?.id) {
        const detail = await apiFetch(`${API}/temporal-rotation-quality-research/${encodeURIComponent(latest.id)}`)
        setResearch(detail)
      }
      setError('')
    } catch (requestError) {
      handleError(requestError)
    } finally {
      setLoading(false)
    }
  }, [buildFormsFromConfig, handleError, loadResearchHistory])

  useEffect(() => { loadWorkspace() }, [loadWorkspace])

  useEffect(() => {
    if (!sourceRun?.id) return
    setResearchForm((current) => current && !current.source_run_id ? { ...current, source_run_id: sourceRun.id } : current)
  }, [sourceRun?.id])

  const loadDiagnostics = useCallback(async (researchId, validationId, { selectLatest = true } = {}) => {
    if (!researchId || !validationId) {
      setDiagnostics([])
      setDiagnostic(null)
      return []
    }
    const payload = await apiFetch(`${API}/temporal-rotation-quality-research/${encodeURIComponent(researchId)}/validations/${encodeURIComponent(validationId)}/diagnostics?limit=30`).catch(() => ({ items: [] }))
    const items = Array.isArray(payload?.items) ? payload.items : []
    setDiagnostics(items)
    if (selectLatest) {
      const active = items.find((item) => DIAGNOSTIC_ACTIVE.has(String(item.status || '').toLowerCase()))
      const latest = active || items[0] || null
      if (latest?.id) {
        const detail = await apiFetch(`${API}/temporal-rotation-quality-research/${encodeURIComponent(researchId)}/validations/${encodeURIComponent(validationId)}/diagnostics/${encodeURIComponent(latest.id)}`)
        setDiagnostic(detail)
      } else {
        setDiagnostic(null)
      }
    }
    return items
  }, [])

  const loadResearchDetail = useCallback(async (researchId, { selectDefaults = false } = {}) => {
    if (!researchId) return null
    const [detail, candidatePayload, validationPayload] = await Promise.all([
      apiFetch(`${API}/temporal-rotation-quality-research/${encodeURIComponent(researchId)}`),
      apiFetch(`${API}/temporal-rotation-quality-research/${encodeURIComponent(researchId)}/candidates?limit=2000`).catch(() => ({ items: [] })),
      apiFetch(`${API}/temporal-rotation-quality-research/${encodeURIComponent(researchId)}/validations?limit=30`).catch(() => ({ items: [] })),
    ])
    const nextCandidates = Array.isArray(candidatePayload?.items) ? candidatePayload.items.filter((item) => item.candidate_id !== 'CONTROL') : []
    const nextValidations = Array.isArray(validationPayload?.items) ? validationPayload.items : []
    setResearch(detail)
    setCandidates(nextCandidates)
    setValidations(nextValidations)
    if (selectDefaults) {
      const robustIds = nextCandidates.filter((item) => item.robust_vs_control).map((item) => item.candidate_id)
      setSelectedCandidates(robustIds.length ? robustIds : detail?.best_candidate?.candidate_id ? [detail.best_candidate.candidate_id] : [])
    }
    const activeEvidence = nextValidations.find((item) => ACTIVE.has(String(item.status || '').toLowerCase()))
    const latestCertification = nextValidations.find((item) => item.kind === 'certification' && String(item.status || '').toLowerCase() === 'completed')
    const latestValidation = nextValidations.find((item) => item.kind !== 'certification' && String(item.status || '').toLowerCase() === 'completed')
    const latestEvidence = activeEvidence || latestCertification || latestValidation || nextValidations[0] || null
    if (latestEvidence?.id) {
      const evidenceDetail = await apiFetch(`${API}/temporal-rotation-quality-research/${encodeURIComponent(researchId)}/validations/${encodeURIComponent(latestEvidence.id)}`)
      setValidation(evidenceDetail)
      setDiagnosticForm((current) => current ? { ...current, candidate_id: preferredDiagnosticCandidateId(evidenceDetail) || current.candidate_id || '' } : current)
      await loadDiagnostics(researchId, latestEvidence.id)
    } else {
      setValidation(null)
      setDiagnostics([])
      setDiagnostic(null)
    }
    return detail
  }, [loadDiagnostics])

  useEffect(() => {
    if (!research?.id || String(research.status || '').toLowerCase() !== 'completed') return
    loadResearchDetail(research.id, { selectDefaults: candidates.length === 0 }).catch(handleError)
  }, [candidates.length, handleError, loadResearchDetail, research?.id, research?.status])

  useEffect(() => {
    if (timerRef.current) window.clearInterval(timerRef.current)
    timerRef.current = null
    const researchActive = research?.id && ACTIVE.has(String(research.status || '').toLowerCase())
    const validationActive = validation?.id && ACTIVE.has(String(validation.status || '').toLowerCase())
    const diagnosticActive = diagnostic?.id && DIAGNOSTIC_ACTIVE.has(String(diagnostic.status || '').toLowerCase())
    if (!researchActive && !validationActive && !diagnosticActive) return undefined
    timerRef.current = window.setInterval(async () => {
      try {
        if (researchActive) {
          const updated = await apiFetch(`${API}/temporal-rotation-quality-research/${encodeURIComponent(research.id)}`)
          setResearch(updated)
          if (String(updated.status || '').toLowerCase() === 'completed') {
            await loadResearchHistory()
            await loadResearchDetail(updated.id, { selectDefaults: true })
          }
        }
        if (validationActive) {
          const updated = await apiFetch(`${API}/temporal-rotation-quality-research/${encodeURIComponent(validation.research_id)}/validations/${encodeURIComponent(validation.id)}`)
          setValidation(updated)
          if (String(updated.status || '').toLowerCase() === 'completed') {
            const payload = await apiFetch(`${API}/temporal-rotation-quality-research/${encodeURIComponent(updated.research_id)}/validations?limit=30`)
            setValidations(Array.isArray(payload?.items) ? payload.items : [])
          }
        }
        if (diagnosticActive) {
          const updated = await apiFetch(`${API}/temporal-rotation-quality-research/${encodeURIComponent(diagnostic.research_id)}/validations/${encodeURIComponent(diagnostic.validation_id)}/diagnostics/${encodeURIComponent(diagnostic.id)}`)
          setDiagnostic(updated)
          if (!DIAGNOSTIC_ACTIVE.has(String(updated.status || '').toLowerCase())) {
            await loadDiagnostics(updated.research_id, updated.validation_id, { selectLatest: false })
          }
        }
      } catch (requestError) {
        handleError(requestError)
      }
    }, 2500)
    return () => {
      if (timerRef.current) window.clearInterval(timerRef.current)
      timerRef.current = null
    }
  }, [diagnostic?.id, diagnostic?.research_id, diagnostic?.status, diagnostic?.validation_id, handleError, loadDiagnostics, loadResearchDetail, loadResearchHistory, research?.id, research?.status, validation?.id, validation?.research_id, validation?.status])

  const updateResearch = (key, value) => setResearchForm((current) => ({ ...current, [key]: value }))
  const updateCaro = (key, value) => setResearchForm((current) => ({ ...current, caro: { ...current.caro, [key]: value } }))
  const updateGate = (key, value) => setResearchForm((current) => ({ ...current, research_gate: { ...current.research_gate, [key]: value } }))

  async function startResearch() {
    if (!canManage || busy || !researchForm?.source_run_id) return
    setBusy(true)
    setError('')
    try {
      const body = {
        source_run_id: researchForm.source_run_id.trim(),
        search_method: researchForm.search_method,
        focus_month: researchForm.focus_month || null,
        control_tolerance_usd: numberValue(researchForm.control_tolerance_usd),
        strong_challenger_override: Boolean(researchForm.strong_challenger_override),
        baseline_drawdown_trigger: researchForm.strong_challenger_override ? optionalNumber(researchForm.baseline_drawdown_trigger) : null,
        baseline_rotation_score_tolerance: researchForm.strong_challenger_override ? optionalNumber(researchForm.baseline_rotation_score_tolerance) : null,
        research_gate: {
          minimum_capital_lift: numberValue(researchForm.research_gate.minimum_capital_lift),
          minimum_sharpe_delta: numberValue(researchForm.research_gate.minimum_sharpe_delta),
          minimum_max_drawdown_delta: numberValue(researchForm.research_gate.minimum_max_drawdown_delta),
          required_fold_wins: optionalNumber(researchForm.research_gate.required_fold_wins),
        },
      }
      if (researchForm.search_method === 'grid') {
        if (researchForm.strong_challenger_override) {
          body.challenger_quality_floors = parseNumberList(researchForm.challenger_quality_floors)
        } else {
          body.drawdown_triggers = parseNumberList(researchForm.drawdown_triggers)
          body.rotation_score_tolerances = parseNumberList(researchForm.rotation_score_tolerances)
        }
      } else if (researchForm.search_method === 'manual') {
        body.manual_candidates = parseManualCandidates(researchForm.manual_candidates)
      } else if (researchForm.search_method === 'caro') {
        body.caro = Object.fromEntries(Object.entries(researchForm.caro).map(([key, value]) => {
          if (key === 'minimum_exploration_trials') return [key, optionalNumber(value)]
          return [key, numberValue(value)]
        }))
      }
      const created = await apiFetch(`${API}/temporal-rotation-quality-research/runs`, { method: 'POST', body })
      setResearch(created)
      setCandidates([])
      setSelectedCandidates([])
      setValidation(null)
      setValidations([])
      await loadResearchHistory()
    } catch (requestError) {
      handleError(requestError)
    } finally {
      setBusy(false)
    }
  }

  async function selectResearch(researchId) {
    setBusy(true)
    setError('')
    try {
      await loadResearchDetail(researchId, { selectDefaults: true })
    } catch (requestError) {
      handleError(requestError)
    } finally {
      setBusy(false)
    }
  }

  function toggleCandidate(candidateId) {
    setSelectedCandidates((current) => current.includes(candidateId) ? current.filter((id) => id !== candidateId) : [...current, candidateId])
  }

  async function startEvidence(kind) {
    if (!canManage || !research?.id || !selectedCandidates.length || busy) return
    const form = kind === 'certification' ? certificationForm : validationForm
    if (!form) return
    setBusy(true)
    setError('')
    try {
      const body = {
        kind,
        fold_count: numberValue(form.fold_count),
        required_fold_wins: optionalNumber(form.required_fold_wins),
        candidate_ids: selectedCandidates,
        minimum_capital_lift: numberValue(form.minimum_capital_lift),
        minimum_sharpe_delta: numberValue(form.minimum_sharpe_delta),
        minimum_max_drawdown_delta: numberValue(form.minimum_max_drawdown_delta),
      }
      const created = await apiFetch(`${API}/temporal-rotation-quality-research/${encodeURIComponent(research.id)}/validate`, { method: 'POST', body })
      setValidation(created)
      setValidations((current) => [created, ...current.filter((item) => item.id !== created.id)])
    } catch (requestError) {
      handleError(requestError)
    } finally {
      setBusy(false)
    }
  }

  async function selectValidation(validationId) {
    if (!research?.id || !validationId) return
    setBusy(true)
    try {
      const detail = await apiFetch(`${API}/temporal-rotation-quality-research/${encodeURIComponent(research.id)}/validations/${encodeURIComponent(validationId)}`)
      setValidation(detail)
      setDiagnosticForm((current) => current ? { ...current, candidate_id: preferredDiagnosticCandidateId(detail) || current.candidate_id || '' } : current)
      await loadDiagnostics(research.id, validationId)
    } catch (requestError) {
      handleError(requestError)
    } finally {
      setBusy(false)
    }
  }

  async function exportResearch() {
    if (!canExport || !research?.id || research.status !== 'completed' || exporting) return
    setExporting(true)
    try {
      await downloadFile(`${API}/temporal-rotation-quality-research/${encodeURIComponent(research.id)}/export.zip`, `temporal_rotation_quality_${research.id}.zip`)
    } catch (requestError) {
      handleError(requestError)
    } finally {
      setExporting(false)
    }
  }

  async function exportValidation() {
    if (!canExport || !validation?.id || validation.status !== 'completed' || exporting) return
    setExporting(true)
    try {
      await downloadFile(`${API}/temporal-rotation-quality-research/${encodeURIComponent(validation.research_id)}/validations/${encodeURIComponent(validation.id)}/export.zip`, `temporal_rotation_quality_${validation.id}.zip`)
    } catch (requestError) {
      handleError(requestError)
    } finally {
      setExporting(false)
    }
  }

  function toggleDiagnosticFeature(feature) {
    setDiagnosticForm((current) => {
      if (!current) return current
      const names = Array.isArray(current.feature_names) ? current.feature_names : []
      return { ...current, feature_names: names.includes(feature) ? names.filter((item) => item !== feature) : [...names, feature] }
    })
  }

  async function startDiagnostic() {
    if (!canManage || !research?.id || !validation?.id || validation.status !== 'completed' || !diagnosticForm?.candidate_id || busy) return
    setBusy(true)
    setError('')
    try {
      const body = {
        candidate_id: diagnosticForm.candidate_id,
        lookback_sessions: numberValue(diagnosticForm.lookback_sessions),
        feature_names: diagnosticForm.feature_names,
        minimum_group_samples: numberValue(diagnosticForm.minimum_group_samples),
        outcome_neutral_band: numberValue(diagnosticForm.outcome_neutral_band),
        top_feature_count: numberValue(diagnosticForm.top_feature_count),
      }
      const created = await apiFetch(`${API}/temporal-rotation-quality-research/${encodeURIComponent(research.id)}/validations/${encodeURIComponent(validation.id)}/diagnostics`, { method: 'POST', body })
      setDiagnostic(created)
      setDiagnostics((current) => [created, ...current.filter((item) => item.id !== created.id)])
    } catch (requestError) {
      handleError(requestError)
    } finally {
      setBusy(false)
    }
  }

  async function stopDiagnostic() {
    if (!canManage || !diagnostic?.id || !DIAGNOSTIC_ACTIVE.has(String(diagnostic.status || '').toLowerCase()) || busy) return
    setBusy(true)
    try {
      const updated = await apiFetch(`${API}/temporal-rotation-quality-research/${encodeURIComponent(diagnostic.research_id)}/validations/${encodeURIComponent(diagnostic.validation_id)}/diagnostics/${encodeURIComponent(diagnostic.id)}/stop`, { method: 'POST' })
      setDiagnostic(updated)
    } catch (requestError) {
      handleError(requestError)
    } finally {
      setBusy(false)
    }
  }

  async function selectDiagnostic(diagnosticId) {
    if (!research?.id || !validation?.id || !diagnosticId) return
    setBusy(true)
    try {
      const detail = await apiFetch(`${API}/temporal-rotation-quality-research/${encodeURIComponent(research.id)}/validations/${encodeURIComponent(validation.id)}/diagnostics/${encodeURIComponent(diagnosticId)}`)
      setDiagnostic(detail)
    } catch (requestError) {
      handleError(requestError)
    } finally {
      setBusy(false)
    }
  }

  async function exportDiagnostic() {
    if (!canExport || !diagnostic?.id || diagnostic.status !== 'completed' || exporting) return
    setExporting(true)
    try {
      await downloadFile(`${API}/temporal-rotation-quality-research/${encodeURIComponent(diagnostic.research_id)}/validations/${encodeURIComponent(diagnostic.validation_id)}/diagnostics/${encodeURIComponent(diagnostic.id)}/export.zip`, `temporal_rotation_quality_diagnostic_${diagnostic.id}.zip`)
    } catch (requestError) {
      handleError(requestError)
    } finally {
      setExporting(false)
    }
  }

  const method = researchForm?.search_method || ''
  const limits = config?.limits || {}
  const researchActive = ACTIVE.has(String(research?.status || '').toLowerCase())
  const evidenceActive = ACTIVE.has(String(validation?.status || '').toLowerCase())
  const diagnosticActive = DIAGNOSTIC_ACTIVE.has(String(diagnostic?.status || '').toLowerCase())
  const disabled = busy || researchActive || evidenceActive || diagnosticActive

  const candidateRows = useMemo(() => [...candidates].sort((a, b) => Number(b.ending_capital || 0) - Number(a.ending_capital || 0)), [candidates])
  const completedEvidence = useMemo(
    () => validations.filter((item) => String(item.status || '').toLowerCase() === 'completed'),
    [validations],
  )
  const latestValidationSummary = useMemo(
    () => validations.find((item) => item.kind !== 'certification' && String(item.status || '').toLowerCase() === 'completed') || null,
    [validations],
  )
  const latestCertificationSummary = useMemo(
    () => validations.find((item) => item.kind === 'certification' && String(item.status || '').toLowerCase() === 'completed') || null,
    [validations],
  )

  const researchWorkflowState = researchActive ? statusLabel(research?.status) : research?.status === 'completed' ? 'Completed' : '—'
  const validationWorkflowState = evidenceWorkflowState(latestValidationSummary)
  const certificationWorkflowState = evidenceWorkflowState(latestCertificationSummary)
  const diagnosticWorkflowState = diagnosticActive
    ? statusLabel(diagnostic?.status)
    : diagnostic?.status === 'completed'
      ? 'Completed'
      : latestCertificationSummary
        ? 'Next'
        : '—'

  if (loading || !researchForm || !config) return <div className="temporal-loading"><span className="loading-ring" />{tr('Loading Rotation Quality Research…')}</div>

  return (
    <section className="temporal-section rotation-quality-console rotation-quality-console-refined">
      <div className="temporal-section-heading">
        <h3>{tr('Rotation Quality Research')}</h3>
      </div>

      {error ? <div className="global-inline-message error-inline">{error}</div> : null}

      <div className="rotation-quality-workflow" aria-label={tr('Current workflow')}>
        <WorkflowStep label="Research" state={researchWorkflowState} tone={research?.status === 'completed' ? 'complete' : researchActive ? 'active' : ''} />
        <WorkflowStep label="Validation" state={validationWorkflowState} tone={validationWorkflowState === 'PASS' ? 'pass' : validationWorkflowState === 'FAIL' ? 'fail' : ''} />
        <WorkflowStep label="Certification" state={certificationWorkflowState} tone={certificationWorkflowState === 'PASS' ? 'pass' : certificationWorkflowState === 'FAIL' ? 'fail' : ''} />
        <WorkflowStep label="Diagnostics" state={diagnosticWorkflowState} tone={diagnosticActive ? 'active' : diagnostic?.status === 'completed' ? 'complete' : latestCertificationSummary ? 'next' : ''} />
      </div>

      {validation?.status === 'completed' && diagnosticForm ? <div className="rotation-quality-card rotation-quality-current-action">
        <div className="rotation-quality-current-action-heading">
          <div>
            <span>{tr('Current action')}</span>
            <strong>{tr('Diagnostics')}</strong>
          </div>
          <span className="rotation-quality-current-action-state">{statusLabel(diagnostic?.status || '—')}</span>
        </div>

        <div className="rotation-quality-form-grid current-action-grid">
          <label className="rotation-quality-field wide">
            <span>{tr('Source execution')}</span>
            <select value={validation?.id || ''} disabled={disabled} onChange={(event) => selectValidation(event.target.value)}>
              {completedEvidence.map((item) => <option key={item.id} value={item.id}>{tr(item.kind === 'certification' ? 'Certification' : 'Validation')} · {item.fold_count} {tr('folds')} · {shortDateTime(item.created_at)}</option>)}
            </select>
          </label>
          <label className="rotation-quality-field">
            <span>{tr('Candidate')}</span>
            <select value={diagnosticForm.candidate_id} disabled={disabled} onChange={(event) => setDiagnosticForm((current) => ({ ...current, candidate_id: event.target.value }))}>
              <option value="">—</option>
              {(validation.candidates || []).map((item) => <option key={item.candidate_id} value={item.candidate_id}>{item.candidate_id}</option>)}
            </select>
          </label>
        </div>

        <details className="rotation-quality-advanced">
          <summary>{tr('Advanced parameters')}</summary>
          <div className="rotation-quality-advanced-body">
            <div className="rotation-quality-form-grid">
              <NumericField label="Lookback sessions" value={diagnosticForm.lookback_sessions} min={config.diagnostics?.limits?.lookback_sessions_min} max={config.diagnostics?.limits?.lookback_sessions_max} step="1" disabled={disabled} onChange={(value) => setDiagnosticForm((current) => ({ ...current, lookback_sessions: value }))} />
              <NumericField label="Minimum group samples" value={diagnosticForm.minimum_group_samples} min={config.diagnostics?.limits?.minimum_group_samples_min} max={config.diagnostics?.limits?.minimum_group_samples_max} step="1" disabled={disabled} onChange={(value) => setDiagnosticForm((current) => ({ ...current, minimum_group_samples: value }))} />
              <NumericField label="Outcome neutral band" value={diagnosticForm.outcome_neutral_band} min={config.diagnostics?.limits?.outcome_neutral_band_min} max={config.diagnostics?.limits?.outcome_neutral_band_max} disabled={disabled} onChange={(value) => setDiagnosticForm((current) => ({ ...current, outcome_neutral_band: value }))} />
              <NumericField label="Top feature count" value={diagnosticForm.top_feature_count} min={config.diagnostics?.limits?.top_feature_count_min} max={config.diagnostics?.limits?.top_feature_count_max} step="1" disabled={disabled} onChange={(value) => setDiagnosticForm((current) => ({ ...current, top_feature_count: value }))} />
            </div>
            <div className="rotation-quality-subsection-title">{tr('Decision-time features')}</div>
            <div className="rotation-quality-feature-grid">{(config.diagnostics?.features || []).map((item) => <label key={item.id} className="rotation-quality-feature-option"><input type="checkbox" checked={diagnosticForm.feature_names.includes(item.id)} disabled={disabled} onChange={() => toggleDiagnosticFeature(item.id)} /><span>{tr(item.label)}</span></label>)}</div>
          </div>
        </details>

        <div className="rotation-quality-actions current-action-actions">
          {canManage && !diagnosticActive ? <button type="button" className="primary-action compact" disabled={disabled || !diagnosticForm.candidate_id || !diagnosticForm.feature_names.length} onClick={startDiagnostic}>{tr('Run Diagnostic')}</button> : null}
          {canManage && diagnosticActive ? <button type="button" className="secondary-action compact danger" disabled={busy} onClick={stopDiagnostic}>{tr('Stop Diagnostic')}</button> : null}
          {canExport && diagnostic?.status === 'completed' ? <button type="button" className="secondary-action compact" disabled={exporting} onClick={exportDiagnostic}>{tr('Export Diagnostic')}</button> : null}
        </div>

        {diagnostic ? <div className="rotation-quality-diagnostic-current">
          <div className="rotation-quality-card-title"><strong>{diagnostic.candidate_id || tr('Diagnostic')}</strong><span>{diagnostic.id}</span><span>{statusLabel(diagnostic.status)}</span></div>
          {diagnosticActive ? <><div className="temporal-status-line"><strong>{statusLabel(diagnostic.status)}</strong><span>{diagnostic.stage || '—'}</span><span>{number(diagnostic.progress, 1)}%</span></div><div className="temporal-progress"><span style={{ width: `${Math.max(0, Math.min(100, Number(diagnostic.progress || 0)))}%` }} /></div></> : null}
          {diagnostic.failure_message ? <div className="global-inline-message error-inline">{diagnostic.failure_message}</div> : null}
          <DiagnosticResult diagnostic={diagnostic} />
        </div> : null}
      </div> : null}

      <details className="rotation-quality-disclosure">
        <summary><strong>{tr('Research & candidates')}</strong><span>{research ? statusLabel(research.status) : '—'}</span></summary>
        <div className="rotation-quality-disclosure-body">
          <div className="rotation-quality-card embedded-card">
            <div className="rotation-quality-card-title"><strong>{tr('Research')}</strong></div>
            <div className="rotation-quality-form-grid">
              <label className="rotation-quality-field wide"><span>{tr('Source Temporal run')}</span><input value={researchForm.source_run_id} disabled={disabled} onChange={(event) => updateResearch('source_run_id', event.target.value)} /></label>
              <label className="rotation-quality-field"><span>{tr('Search method')}</span><select value={method} disabled={disabled} onChange={(event) => updateResearch('search_method', event.target.value)}>{(config.search_methods || []).map((item) => <option key={item.id} value={item.id}>{tr(item.label)}</option>)}</select></label>
              <label className="rotation-quality-field"><span>{tr('Focus month')}</span><input type="month" value={researchForm.focus_month} disabled={disabled} onChange={(event) => updateResearch('focus_month', event.target.value)} /></label>
              <NumericField label="Control tolerance (USD)" value={researchForm.control_tolerance_usd} disabled={disabled} onChange={(value) => updateResearch('control_tolerance_usd', value)} />
              <label className="rotation-quality-feature-option rotation-quality-strong-toggle">
                <input type="checkbox" checked={researchForm.strong_challenger_override} disabled={disabled} onChange={(event) => updateResearch('strong_challenger_override', event.target.checked)} />
                <span>{tr('Strong Challenger Override')}</span>
              </label>
            </div>

            {researchForm.strong_challenger_override ? <div className="rotation-quality-form-grid method-grid rotation-quality-baseline-grid">
              <NumericField label="Baseline drawdown trigger" value={researchForm.baseline_drawdown_trigger} disabled={disabled} onChange={(value) => updateResearch('baseline_drawdown_trigger', value)} />
              <NumericField label="Baseline rotation score tolerance" value={researchForm.baseline_rotation_score_tolerance} disabled={disabled} onChange={(value) => updateResearch('baseline_rotation_score_tolerance', value)} />
            </div> : null}

            {method === 'grid' ? <div className="rotation-quality-form-grid method-grid">
              {researchForm.strong_challenger_override ? <label className="rotation-quality-field full"><span>{tr('Challenger quality floors')}</span><textarea rows="3" value={researchForm.challenger_quality_floors} disabled={disabled} onChange={(event) => updateResearch('challenger_quality_floors', event.target.value)} /></label> : <>
                <label className="rotation-quality-field wide"><span>{tr('Drawdown triggers')}</span><textarea rows="3" value={researchForm.drawdown_triggers} disabled={disabled} onChange={(event) => updateResearch('drawdown_triggers', event.target.value)} /></label>
                <label className="rotation-quality-field wide"><span>{tr('Rotation score tolerances')}</span><textarea rows="3" value={researchForm.rotation_score_tolerances} disabled={disabled} onChange={(event) => updateResearch('rotation_score_tolerances', event.target.value)} /></label>
              </>}
            </div> : null}

            {method === 'manual' ? <div className="rotation-quality-form-grid method-grid">
              <label className="rotation-quality-field full"><span>{tr(researchForm.strong_challenger_override ? 'Manual candidates · drawdown, tolerance, challenger floor' : 'Manual candidates · drawdown, tolerance')}</span><textarea rows="5" value={researchForm.manual_candidates} disabled={disabled} onChange={(event) => updateResearch('manual_candidates', event.target.value)} /></label>
            </div> : null}

            {method === 'caro' ? <details className="rotation-quality-advanced nested">
              <summary>{tr('CARO parameters')}</summary>
              <div className="rotation-quality-advanced-body">
                <div className="rotation-quality-form-grid method-grid">
                  {researchForm.strong_challenger_override ? <>
                    <NumericField label="Challenger quality floor min" value={researchForm.caro.challenger_quality_floor_min} min="0" max="1" disabled={disabled} onChange={(value) => updateCaro('challenger_quality_floor_min', value)} />
                    <NumericField label="Challenger quality floor max" value={researchForm.caro.challenger_quality_floor_max} min="0" max="1" disabled={disabled} onChange={(value) => updateCaro('challenger_quality_floor_max', value)} />
                  </> : <>
                    <NumericField label="Drawdown min" value={researchForm.caro.drawdown_trigger_min} disabled={disabled} onChange={(value) => updateCaro('drawdown_trigger_min', value)} />
                    <NumericField label="Drawdown max" value={researchForm.caro.drawdown_trigger_max} disabled={disabled} onChange={(value) => updateCaro('drawdown_trigger_max', value)} />
                    <NumericField label="Tolerance min" value={researchForm.caro.rotation_score_tolerance_min} disabled={disabled} onChange={(value) => updateCaro('rotation_score_tolerance_min', value)} />
                    <NumericField label="Tolerance max" value={researchForm.caro.rotation_score_tolerance_max} disabled={disabled} onChange={(value) => updateCaro('rotation_score_tolerance_max', value)} />
                  </>}
                  <NumericField label="Trials" value={researchForm.caro.trials} min={limits.caro_trials_min} max={limits.caro_trials_max} step="1" disabled={disabled} onChange={(value) => updateCaro('trials', value)} />
                  <NumericField label="Seed" value={researchForm.caro.seed} step="1" disabled={disabled} onChange={(value) => updateCaro('seed', value)} />
                  <NumericField label="Candidate pool size" value={researchForm.caro.candidate_pool_size} step="1" disabled={disabled} onChange={(value) => updateCaro('candidate_pool_size', value)} />
                  <NumericField label="Space-filling pool size" value={researchForm.caro.space_filling_pool_size} step="1" disabled={disabled} onChange={(value) => updateCaro('space_filling_pool_size', value)} />
                  <NumericField label="Exploration weight" value={researchForm.caro.exploration_weight} disabled={disabled} onChange={(value) => updateCaro('exploration_weight', value)} />
                  <NumericField label="Minimum exploration trials" value={researchForm.caro.minimum_exploration_trials} step="1" disabled={disabled} onChange={(value) => updateCaro('minimum_exploration_trials', value)} />
                  <NumericField label="Initial exploration fraction" value={researchForm.caro.initial_exploration_fraction} disabled={disabled} onChange={(value) => updateCaro('initial_exploration_fraction', value)} />
                  <NumericField label="Minimum exploration fraction" value={researchForm.caro.minimum_exploration_fraction} disabled={disabled} onChange={(value) => updateCaro('minimum_exploration_fraction', value)} />
                  <NumericField label="Stagnation recovery trials" value={researchForm.caro.stagnation_recovery_trials} step="1" disabled={disabled} onChange={(value) => updateCaro('stagnation_recovery_trials', value)} />
                  <NumericField label="CARO minimum capital improvement" value={researchForm.caro.minimum_capital_improvement} disabled={disabled} onChange={(value) => updateCaro('minimum_capital_improvement', value)} />
                  <NumericField label="CARO Sharpe tolerance" value={researchForm.caro.sharpe_tolerance} disabled={disabled} onChange={(value) => updateCaro('sharpe_tolerance', value)} />
                  <NumericField label="CARO drawdown tolerance" value={researchForm.caro.drawdown_tolerance} disabled={disabled} onChange={(value) => updateCaro('drawdown_tolerance', value)} />
                  <NumericField label="CARO minimum worst fold return" value={researchForm.caro.minimum_worst_fold_return} disabled={disabled} onChange={(value) => updateCaro('minimum_worst_fold_return', value)} />
                </div>
              </div>
            </details> : null}

            <details className="rotation-quality-advanced nested">
              <summary>{tr('Research gate')}</summary>
              <div className="rotation-quality-advanced-body">
                <div className="rotation-quality-form-grid">
                  <NumericField label="Minimum capital lift" value={researchForm.research_gate.minimum_capital_lift} disabled={disabled} onChange={(value) => updateGate('minimum_capital_lift', value)} />
                  <NumericField label="Minimum Sharpe delta" value={researchForm.research_gate.minimum_sharpe_delta} disabled={disabled} onChange={(value) => updateGate('minimum_sharpe_delta', value)} />
                  <NumericField label="Minimum MaxDD delta" value={researchForm.research_gate.minimum_max_drawdown_delta} disabled={disabled} onChange={(value) => updateGate('minimum_max_drawdown_delta', value)} />
                  <NumericField label="Required fold wins" value={researchForm.research_gate.required_fold_wins} step="1" disabled={disabled} onChange={(value) => updateGate('required_fold_wins', value)} />
                </div>
              </div>
            </details>

            <div className="rotation-quality-actions">
              {canManage ? <button type="button" className="secondary-action compact" onClick={startResearch} disabled={disabled || !researchForm.source_run_id}>{tr(busy ? 'Starting…' : method === 'caro' ? 'Start CARO Research' : 'Start Research')}</button> : null}
              {canExport && research?.status === 'completed' ? <button type="button" className="secondary-action compact" onClick={exportResearch} disabled={exporting}>{tr('Export Research')}</button> : null}
            </div>
          </div>

          {research ? <div className="rotation-quality-card embedded-card">
            <div className="rotation-quality-card-title"><strong>{tr('Current research')}</strong><span>{research.id}</span><span>{statusLabel(research.status)}</span></div>
            {ACTIVE.has(String(research.status || '').toLowerCase()) ? <><div className="temporal-status-line"><strong>{statusLabel(research.status)}</strong><span>{research.stage || '—'}</span><span>{number(research.progress, 1)}%</span></div><div className="temporal-progress"><span style={{ width: `${Math.max(0, Math.min(100, Number(research.progress || 0)))}%` }} /></div></> : null}
            {research.failure_message ? <div className="global-inline-message error-inline">{research.failure_message}</div> : null}
            <ResearchResult research={research} />
          </div> : null}

          {candidateRows.length ? <div className="rotation-quality-card embedded-card">
            <div className="rotation-quality-card-title"><strong>{tr('Candidates')}</strong><span>{selectedCandidates.length} {tr('selected')}</span></div>
            <div className="rotation-quality-table-shell">
              <table className="rotation-quality-table candidates">
                <thead><tr><th></th><th>{tr('Candidate')}</th><th>{tr('DD trigger')}</th><th>{tr('Tolerance')}</th><th>{tr('Quality floor')}</th><th>{tr('Overrides')}</th><th>{tr('Capital')}</th><th>{tr('Lift')}</th><th>{tr('Sharpe')}</th><th>{tr('Max Drawdown')}</th><th>{tr('Fold wins')}</th><th>{tr('Robust')}</th></tr></thead>
                <tbody>{candidateRows.map((candidate) => <tr key={candidate.candidate_id}>
                  <td><input type="checkbox" checked={selectedCandidates.includes(candidate.candidate_id)} disabled={disabled} onChange={() => toggleCandidate(candidate.candidate_id)} /></td>
                  <td><strong>{candidate.candidate_id}</strong></td><td>{percent(candidate.drawdown_trigger, 2)}</td><td>{number(candidate.rotation_score_tolerance, 4)}</td><td>{candidate.challenger_quality_floor == null ? '—' : number(candidate.challenger_quality_floor, 4)}</td><td>{candidate.strong_challenger_overrides == null ? '—' : number(candidate.strong_challenger_overrides, 0)}</td><td>{money(candidate.ending_capital)}</td><td className={Number(candidate.capital_lift_vs_control) >= 0 ? 'positive' : 'negative'}>{percent(candidate.capital_lift_vs_control, 2)}</td><td>{number(candidate.sharpe, 4)}</td><td>{percent(candidate.max_drawdown, 2)}</td><td>{candidate.folds_beating_control ?? '—'}/{research.source_fold_count ?? '—'}</td><td><EvidenceBadge passed={candidate.robust_vs_control} /></td>
                </tr>)}</tbody>
              </table>
            </div>
          </div> : null}
        </div>
      </details>

      {research?.status === 'completed' ? <details className="rotation-quality-disclosure">
        <summary><strong>{tr('Validation & Certification')}</strong><span>{validation ? statusLabel(validation.status) : '—'}</span></summary>
        <div className="rotation-quality-disclosure-body">
          <div className="rotation-quality-evidence-grid">
            {[
              ['validation', validationForm, setValidationForm, 'Validation'],
              ['certification', certificationForm, setCertificationForm, 'Certification'],
            ].map(([kind, form, setForm, title]) => form ? <div className="rotation-quality-card embedded-card" key={kind}>
              <div className="rotation-quality-card-title"><strong>{tr(title)}</strong></div>
              <div className="rotation-quality-form-grid evidence">
                <NumericField label="Fold count" value={form.fold_count} min={limits.fold_count_min} max={limits.fold_count_max} step="1" disabled={disabled} onChange={(value) => setForm((current) => ({ ...current, fold_count: value }))} />
                <NumericField label="Required fold wins" value={form.required_fold_wins} min="0" max={form.fold_count || limits.fold_count_max} step="1" disabled={disabled} onChange={(value) => setForm((current) => ({ ...current, required_fold_wins: value }))} />
                <NumericField label="Minimum capital lift" value={form.minimum_capital_lift} disabled={disabled} onChange={(value) => setForm((current) => ({ ...current, minimum_capital_lift: value }))} />
                <NumericField label="Minimum Sharpe delta" value={form.minimum_sharpe_delta} disabled={disabled} onChange={(value) => setForm((current) => ({ ...current, minimum_sharpe_delta: value }))} />
                <NumericField label="Minimum MaxDD delta" value={form.minimum_max_drawdown_delta} disabled={disabled} onChange={(value) => setForm((current) => ({ ...current, minimum_max_drawdown_delta: value }))} />
              </div>
              <div className="rotation-quality-actions">{canManage ? <button type="button" className="secondary-action compact" disabled={disabled || !selectedCandidates.length} onClick={() => startEvidence(kind)}>{tr(kind === 'certification' ? 'Start Certification' : 'Start Validation')}</button> : null}</div>
            </div> : null)}
          </div>

          {validation ? <div className="rotation-quality-card embedded-card">
            <div className="rotation-quality-card-title"><strong>{tr(validation.kind === 'certification' ? 'Certification result' : 'Validation result')}</strong><span>{validation.id}</span><span>{statusLabel(validation.status)}</span></div>
            {ACTIVE.has(String(validation.status || '').toLowerCase()) ? <><div className="temporal-status-line"><strong>{statusLabel(validation.status)}</strong><span>{validation.stage || '—'}</span><span>{number(validation.progress, 1)}%</span></div><div className="temporal-progress"><span style={{ width: `${Math.max(0, Math.min(100, Number(validation.progress || 0)))}%` }} /></div></> : null}
            {validation.failure_message ? <div className="global-inline-message error-inline">{validation.failure_message}</div> : null}
            <EvidenceResult validation={validation} />
            {canExport && validation?.status === 'completed' ? <div className="rotation-quality-actions"><button type="button" className="secondary-action compact" onClick={exportValidation} disabled={exporting}>{tr(validation.kind === 'certification' ? 'Export Certification' : 'Export Validation')}</button></div> : null}
          </div> : null}
        </div>
      </details> : null}

      <details className="rotation-quality-disclosure compact-history-disclosure">
        <summary><strong>{tr('Execution history')}</strong><span>{validations.length + researches.length + diagnostics.length}</span></summary>
        <div className="rotation-quality-disclosure-body">
          {diagnostics.length ? <div className="rotation-quality-history-group">
            <div className="rotation-quality-subsection-title">{tr('Diagnostics')}</div>
            <div className="rotation-quality-history-row-wrap diagnostic-history">{diagnostics.map((item) => <button key={item.id} type="button" className={diagnostic?.id === item.id ? 'active' : ''} onClick={() => selectDiagnostic(item.id)} disabled={busy}><strong>{item.candidate_id}</strong><span>{item.lookback_sessions || item.request?.lookback_sessions || '—'} {tr('sessions')}</span><span>{statusLabel(item.status)}</span><small>{shortDateTime(item.created_at)}</small></button>)}</div>
          </div> : null}
          {validations.length ? <div className="rotation-quality-history-group">
            <div className="rotation-quality-subsection-title">{tr('Validation / Certification history')}</div>
            <div className="rotation-quality-history-row-wrap">{validations.map((item) => <button key={item.id} type="button" className={validation?.id === item.id ? 'active' : ''} onClick={() => selectValidation(item.id)} disabled={busy}><strong>{item.kind === 'certification' ? tr('Certification') : tr('Validation')}</strong><span>{item.fold_count} {tr('folds')}</span><span>{statusLabel(item.status)}</span><small>{shortDateTime(item.created_at)}</small></button>)}</div>
          </div> : null}
          {researches.length ? <div className="rotation-quality-history-group">
            <div className="rotation-quality-subsection-title">{tr('Research history')}</div>
            <div className="rotation-quality-history-row-wrap">{researches.map((item) => <button key={item.id} type="button" className={research?.id === item.id ? 'active' : ''} onClick={() => selectResearch(item.id)} disabled={busy}><strong>{tr(item.search?.method === 'caro' ? 'Unified Adaptive CARO' : item.search?.method === 'manual' ? 'Manual' : 'Grid Search')}</strong><span>{statusLabel(item.status)}</span><span>{item.best_candidate?.candidate_id || '—'}</span><small>{shortDateTime(item.created_at)}</small></button>)}</div>
          </div> : null}
        </div>
      </details>
    </section>
  )
}
