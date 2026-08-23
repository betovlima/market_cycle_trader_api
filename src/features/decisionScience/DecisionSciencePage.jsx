import { useEffect, useMemo, useState } from 'react'
import { createPortal } from 'react-dom'

import { apiFetch, downloadFile } from '../../api/http'
import { hasCapability } from '../../auth/capabilities'
import { API } from '../../config/env'
import { getIntlLocale, tr } from '../../i18n/runtime'
import { CoffeeProgress } from '../../shared/CoffeeProgress'
import { MonthlyReturnHeatmap } from '../analytics/components/MonthlyReturnHeatmap'
import './decisionScience.css'

function pct(value, digits = 1) {
  const number = Number(value)
  return Number.isFinite(number) ? `${(number * 100).toFixed(digits)}%` : '—'
}

function num(value, digits = 3) {
  const number = Number(value)
  return Number.isFinite(number) ? number.toFixed(digits) : '—'
}

function ModelMetric({ title, metrics }) {
  const unavailable = metrics?.status === 'unavailable'
  return <div className="decision-science-model-card">
    <div className="decision-science-model-title">{title}</div>
    {unavailable ? <div className="decision-science-muted">{tr('Unavailable')}</div> : <>
      <div className="decision-science-metric-grid">
        <span><small>AUC</small><strong>{num(metrics?.auc)}</strong></span>
        <span><small>Brier</small><strong>{num(metrics?.brier)}</strong></span>
        <span><small>Calibration</small><strong>{num(metrics?.calibration_error)}</strong></span>
        <span><small>Positive</small><strong>{pct(metrics?.positive_rate)}</strong></span>
      </div>
    </>}
  </div>
}

function FoldCard({ fold }) {
  return <article className="decision-science-fold-card">
    <header>
      <div><small>{tr('Fold')}</small><strong>{fold.fold_id}</strong></div>
      <div className="decision-science-selected-model">{String(fold.selected_model || '').replaceAll('_', ' ')}</div>
    </header>
    <div className="decision-science-model-row">
      <ModelMetric title="Logistic Regression" metrics={fold.models?.logistic_regression} />
      <ModelMetric title="LightGBM" metrics={fold.models?.lightgbm} />
    </div>
    <footer>
      <span>{tr('Training rows')}: {Number(fold.train_rows || 0).toLocaleString()}</span>
      <span>{tr('Test rows')}: {Number(fold.test_rows || 0).toLocaleString()}</span>
      <span>{tr('Threshold')}: {num(fold.selection?.threshold, 2)}</span>
    </footer>
  </article>
}

function fullMonthLabel(value) {
  const [year, month] = String(value || '').split('-').map(Number)
  if (!year || !month) return String(value || '')
  const label = new Intl.DateTimeFormat(getIntlLocale(), { month: 'long', year: 'numeric', timeZone: 'UTC' })
    .format(new Date(Date.UTC(year, month - 1, 1)))
  return label ? `${label.charAt(0).toUpperCase()}${label.slice(1)}` : String(value || '')
}

function displayList(values = [], formatter = (value) => String(value)) {
  const items = Array.isArray(values) ? values : []
  return items.length ? items.map(formatter).join(' · ') : '—'
}

function CashMonthDetailDialog({ row, onClose }) {
  useEffect(() => {
    if (!row) return undefined
    const handleKeyDown = (event) => {
      if (event.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [row, onClose])

  if (!row || typeof document === 'undefined') return null

  const sessions = Number(row.sessions || 0)
  const cashSessions = Number(row.cash_sessions || 0)
  const investSessions = Number.isFinite(Number(row.invest_sessions)) ? Number(row.invest_sessions) : Math.max(0, sessions - cashSessions)
  const cashShare = Number(row.cash_share || 0)
  const averageProbability = Number(row.average_best_probability)
  const averageThreshold = Number(row.average_threshold)
  const averageMargin = Number(row.average_probability_margin)
  const labeledCashSessions = Number(row.labeled_cash_sessions || 0)
  const missedSessions = Number(row.missed_opportunity_sessions || 0)
  const avoidedSessions = Number(row.avoided_non_opportunity_sessions || 0)
  const hasOutcomeDetail = labeledCashSessions > 0 || Number.isFinite(Number(row.missed_opportunity_rate)) || Number.isFinite(Number(row.avoided_non_opportunity_rate))

  return createPortal(<div className="decision-science-dialog-backdrop" role="presentation" onMouseDown={onClose}>
    <section className="decision-science-dialog" role="dialog" aria-modal="true" aria-label={fullMonthLabel(row.month)} onMouseDown={(event) => event.stopPropagation()}>
      <header className="decision-science-dialog-header">
        <div><span className="decision-science-dialog-kicker">CASH SHADOW</span><h3>{fullMonthLabel(row.month)}</h3><p>{tr('Research-only monthly decision summary. No portfolio action was changed.')}</p></div>
        <button type="button" onClick={onClose} aria-label={tr('Close')}>×</button>
      </header>

      <div className="decision-science-dialog-metrics">
        <div><span>{tr('CASH share')}</span><strong>{pct(cashShare, 1)}</strong></div>
        <div><span>{tr('CASH sessions')}</span><strong>{cashSessions.toLocaleString()}</strong></div>
        <div><span>{tr('INVEST sessions')}</span><strong>{investSessions.toLocaleString()}</strong></div>
        <div><span>{tr('OOS sessions')}</span><strong>{sessions.toLocaleString()}</strong></div>
      </div>

      <div className="decision-science-dialog-context">
        <div><span>{tr('Average best opportunity probability')}</span><strong>{Number.isFinite(averageProbability) ? pct(averageProbability, 1) : '—'}</strong></div>
        <div><span>{tr('CASH-session average probability')}</span><strong>{Number.isFinite(Number(row.cash_average_best_probability)) ? pct(row.cash_average_best_probability, 1) : '—'}</strong></div>
        <div><span>{tr('INVEST-session average probability')}</span><strong>{Number.isFinite(Number(row.invest_average_best_probability)) ? pct(row.invest_average_best_probability, 1) : '—'}</strong></div>
        <div><span>{tr('Decision threshold')}</span><strong>{Array.isArray(row.thresholds) && row.thresholds.length ? displayList(row.thresholds, (value) => num(value, 2)) : Number.isFinite(averageThreshold) ? num(averageThreshold, 2) : '—'}</strong></div>
        <div><span>{tr('Average probability margin')}</span><strong>{Number.isFinite(averageMargin) ? `${averageMargin >= 0 ? '+' : ''}${num(averageMargin, 3)}` : '—'}</strong></div>
        <div><span>{tr('Fold / model')}</span><strong>{`${displayList(row.fold_ids, (value) => `Fold ${value}`)} / ${displayList(row.models, (value) => String(value).replaceAll('_', ' '))}`}</strong></div>
      </div>

      {hasOutcomeDetail ? <div className="decision-science-dialog-outcomes">
        <div><span>{tr('Selected-asset missed opportunity')}</span><strong>{pct(row.missed_opportunity_rate, 1)}</strong><small>{missedSessions} / {labeledCashSessions} {tr('labeled CASH sessions')}</small></div>
        <div><span>{tr('Selected-asset non-opportunity avoided')}</span><strong>{pct(row.avoided_non_opportunity_rate, 1)}</strong><small>{avoidedSessions} / {labeledCashSessions} {tr('labeled CASH sessions')}</small></div>
      </div> : null}

      <div className="decision-science-dialog-notes">
        <div><strong>{tr('How to read this cell')}</strong><p>{tr('A value of {cashShare} means the shadow gate would choose CASH in {cashSessions} of {sessions} out-of-sample sessions in this month.').replace('{cashShare}', pct(cashShare, 0)).replace('{cashSessions}', String(cashSessions)).replace('{sessions}', String(sessions))}</p></div>
        <div><strong>{tr('What this chart is not')}</strong><p>{tr('This is a hypothetical research decision frequency. It is not the actual portfolio CASH allocation, monthly return, or realized performance.')}</p></div>
        {hasOutcomeDetail ? <div><strong>{tr('Outcome metric scope')}</strong><p>{tr('Missed and avoided rates refer only to the asset selected by the Strategy on CASH-shadow sessions; they do not prove whether another asset in the universe had an opportunity.')}</p></div> : null}
      </div>
    </section>
  </div>, document.body)
}

function CashMonthMap({ monthly = [] }) {
  const [selectedMonth, setSelectedMonth] = useState(null)
  const byYear = useMemo(() => {
    const result = new Map()
    monthly.forEach((row) => {
      const [year, month] = String(row.month || '').split('-')
      if (!year || !month) return
      if (!result.has(year)) result.set(year, new Map())
      result.get(year).set(Number(month), row)
    })
    return [...result.entries()].sort(([a], [b]) => Number(a) - Number(b))
  }, [monthly])
  if (!byYear.length) return null
  return <div className="decision-science-map-wrap">
    <div className="decision-science-map-head"><span />{Array.from({ length: 12 }, (_, i) => <span key={i}>{String(i + 1).padStart(2, '0')}</span>)}</div>
    {byYear.map(([year, months]) => <div className="decision-science-map-row" key={year}>
      <strong>{year}</strong>
      {Array.from({ length: 12 }, (_, i) => {
        const row = months.get(i + 1)
        if (!row) return <span className="decision-science-month-cell empty" key={i} />
        const share = Number(row.cash_share || 0)
        const level = share >= 0.75 ? 4 : share >= 0.5 ? 3 : share >= 0.25 ? 2 : share > 0 ? 1 : 0
        return <button
          type="button"
          className={`decision-science-month-cell level-${level}`}
          key={i}
          aria-label={`${fullMonthLabel(row.month)} · CASH ${pct(share, 0)} · ${tr('Open details')}`}
          onClick={() => setSelectedMonth(row)}
        >{share > 0 ? `${Math.round(share * 100)}` : '0'}</button>
      })}
    </div>)}
    <CashMonthDetailDialog row={selectedMonth} onClose={() => setSelectedMonth(null)} />
  </div>
}

function Coefficients({ rows = [] }) {
  if (!rows.length) return null
  return <div className="decision-science-coefficients">
    {rows.map((row) => <div key={row.feature}>
      <span>{String(row.feature).replaceAll('_', ' ')}</span>
      <strong>{Number(row.coefficient || 0).toFixed(3)}</strong>
    </div>)}
  </div>
}

export function DecisionSciencePage({ capabilities = {} }) {
  const [history, setHistory] = useState([])
  const [runId, setRunId] = useState('')
  const [analysis, setAnalysis] = useState(null)
  const [analysisHistory, setAnalysisHistory] = useState([])
  const [monthlyMode, setMonthlyMode] = useState('simulation')
  const [running, setRunning] = useState(false)
  const [runElapsed, setRunElapsed] = useState(0)
  const [exporting, setExporting] = useState(false)
  const [error, setError] = useState('')

  const canRun = hasCapability(capabilities, 'temporal_intelligence.start')
  const canExport = hasCapability(capabilities, 'temporal_intelligence.export')

  useEffect(() => {
    if (!running) {
      setRunElapsed(0)
      return undefined
    }
    const startedAt = Date.now()
    const updateElapsed = () => setRunElapsed(Math.max(0, Math.floor((Date.now() - startedAt) / 1000)))
    updateElapsed()
    const timer = window.setInterval(updateElapsed, 1000)
    return () => window.clearInterval(timer)
  }, [running])

  useEffect(() => {
    let disposed = false
    Promise.all([
      apiFetch(`${API}/temporal-intelligence/history?limit=30`),
      apiFetch(`${API}/decision-science/history?limit=30`),
    ]).then(([runHistory, savedHistory]) => {
      if (disposed) return
      const items = (runHistory?.items || []).filter((row) => String(row?.status || '').toLowerCase() === 'completed')
      const saved = (savedHistory?.items || []).filter((row) => String(row?.status || '').toLowerCase() === 'completed')
      setHistory(items)
      setAnalysisHistory(saved)
      const latestSaved = saved[0] || null
      const latestRunId = latestSaved?.run_id && items.some((row) => String(row.id) === String(latestSaved.run_id))
        ? String(latestSaved.run_id)
        : String(items[0]?.id || '')
      setRunId(latestRunId)
      setAnalysis(latestSaved && String(latestSaved.run_id) === latestRunId ? latestSaved : saved.find((row) => String(row.run_id) === latestRunId) || null)
    }).catch((requestError) => { if (!disposed) setError(requestError.message) })
    return () => { disposed = true }
  }, [])

  const selectedRun = useMemo(() => history.find((row) => String(row.id) === String(runId)) || null, [history, runId])

  const runAnalysis = async () => {
    if (!runId || !canRun) return
    setRunning(true); setMonthlyMode('simulation'); setError('')
    try {
      const value = await apiFetch(`${API}/decision-science/${encodeURIComponent(runId)}/analyze`, { method: 'POST' })
      setAnalysis(value)
      setAnalysisHistory((current) => [value, ...current.filter((row) => String(row.id) !== String(value.id))])
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setRunning(false)
    }
  }

  const exportResults = async () => {
    if (!canExport || !analysis?.id || exporting) return
    setExporting(true); setError('')
    try {
      await downloadFile(
        `${API}/decision-science/${encodeURIComponent(analysis.id)}/export.zip`,
        `decision_science_${analysis.id}.zip`,
      )
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setExporting(false)
    }
  }

  const opportunity = analysis?.absolute_opportunity
  const shadow = opportunity?.shadow_cash
  const transition = analysis?.leader_transition
  const folds = opportunity?.walk_forward?.folds || []
  const monthlyPerformance = analysis?.strategy_monthly_performance || {}
  const monthlyPerformanceRows = monthlyPerformance?.rows || []

  return <section className="decision-science-page">
    <header className="decision-science-toolbar">
      <div>
        <h2>{tr('Decision Science')}</h2>
        <div className="decision-science-run-meta">{selectedRun?.strategy_profile_name || '—'} · {selectedRun?.analysis_end_date || '—'}</div>
      </div>
      <div className="decision-science-actions">
        <select value={runId} disabled={running || exporting} onChange={(event) => {
          const nextRunId = event.target.value
          setRunId(nextRunId)
          setAnalysis(analysisHistory.find((row) => String(row.run_id) === String(nextRunId)) || null)
        }}>
          {history.map((row) => <option value={row.id} key={row.id}>{row.strategy_profile_name || row.id} · {row.analysis_end_date || ''}</option>)}
        </select>
        {canExport && analysis?.id && !running ? <button type="button" className="secondary" disabled={running || exporting} onClick={exportResults}>{tr(exporting ? 'Exporting…' : 'Export Results')}</button> : null}
        {canRun ? <button type="button" className="primary" disabled={!runId || running || exporting} onClick={runAnalysis}>{running ? tr('Running…') : tr('Run Decision Science')}</button> : null}
      </div>
    </header>

    {error ? <div className="decision-science-error">{error}</div> : null}
    {running ? <section className="decision-science-running-panel">
      <CoffeeProgress counter={`${runElapsed}s`} size="md" label={tr('Decision Science processing')} />
    </section> : !analysis ? <div className="decision-science-empty">{tr('No saved Decision Science analysis exists for this research run. Run Decision Science to create one.')}</div> : <>
      <div className="decision-science-summary-grid">
        <article><small>{tr('Target')}</small><strong>5d</strong><span>{tr('Significant growth')}</span></article>
        <article><small>{tr('CASH shadow')}</small><strong>{pct(shadow?.cash_share)}</strong><span>{Number(shadow?.cash_sessions || 0).toLocaleString()} {tr('sessions')}</span></article>
        <article><small>{tr('Missed opportunity')}</small><strong>{pct(shadow?.missed_opportunity_rate)}</strong><span>{tr('within CASH shadow')}</span></article>
        <article><small>{tr('Avoided non-opportunity')}</small><strong>{pct(shadow?.avoided_non_opportunity_rate)}</strong><span>{tr('within CASH shadow')}</span></article>
      </div>

      <section className="decision-science-section">
        <div className="decision-science-section-title"><h3>{tr('Absolute Opportunity')}</h3><span>{tr('INVEST vs CASH')}</span></div>
        <div className="decision-science-folds">{folds.map((fold) => <FoldCard fold={fold} key={fold.fold_id} />)}</div>
      </section>

      {monthlyPerformanceRows.length ? <section className="decision-science-section decision-science-performance-section">
        <div className="decision-science-section-title decision-science-performance-heading">
          <div><h3>{tr('Monthly Return Heatmap')}</h3><span>{tr('Strategy Research final validation')} · {tr('Click a cell for details')}</span></div>
          <div className="decision-science-performance-modes" role="group" aria-label={tr('Monthly return view')}>
            {[['simulation', 'Strategy'], ['reference', 'Control'], ['excess', 'S − C']].map(([key, label]) => <button key={key} type="button" className={monthlyMode === key ? 'active' : ''} onClick={() => setMonthlyMode(key)}>{tr(label)}</button>)}
          </div>
        </div>
        <MonthlyReturnHeatmap
          rows={monthlyPerformanceRows}
          mode={monthlyMode}
          simulationLabel={tr('Strategy')}
          referenceLabel={tr('Control')}
          excessLabel="S − C"
        />
      </section> : null}

      <section className="decision-science-section">
        <div className="decision-science-section-title"><h3>{tr('CASH Shadow')}</h3><span>{tr('Monthly share of sessions')} · {tr('Click a cell for details')}</span></div>
        <CashMonthMap monthly={shadow?.monthly || []} />
      </section>

      <section className="decision-science-two-column">
        <article className="decision-science-section">
          <div className="decision-science-section-title"><h3>{tr('Logistic Regression')}</h3><span>{tr('Largest standardized coefficients')}</span></div>
          <Coefficients rows={opportunity?.logistic_interpretability || []} />
        </article>
        <article className="decision-science-section">
          <div className="decision-science-section-title"><h3>{tr('Leader Transition')}</h3><span>{tr('HOLD vs ROTATE research')}</span></div>
          <div className="decision-science-transition-summary">
            <strong>{Number(transition?.rows || 0).toLocaleString()}</strong><span>{tr('eligible transitions')}</span>
          </div>
          {(transition?.walk_forward_folds || []).map((fold) => <div className="decision-science-transition-fold" key={fold.fold_id}>
            <span>Fold {fold.fold_id}</span>
            <span>Logistic AUC <strong>{num(fold.models?.logistic_regression?.auc)}</strong></span>
            <span>LightGBM AUC <strong>{num(fold.models?.lightgbm?.auc)}</strong></span>
          </div>)}
        </article>
      </section>
    </>}
  </section>
}
