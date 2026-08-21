import { useEffect, useMemo, useState } from 'react'
import { createPortal } from 'react-dom'
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import { apiFetch } from '../../../api/http'
import { API } from '../../../config/env'
import { tr } from '../../../i18n/runtime'
import { money, number, percent, shortDateTime } from '../../../shared/formatters'
import { AnalyticsResponsiveContainer } from '../../analytics/components/AnalyticsPrimitives'
import { MonthlyReturnHeatmap } from '../../analytics/components/MonthlyReturnHeatmap'
import { MonthlyAssetAnalysis } from '../../backtest/components/MonthlyAssetAnalysis'

function kindLabel(value) {
  if (value === 'certification') return tr('Certification')
  if (value === 'validation') return tr('Validation')
  return tr('Research')
}

function candidateRows(execution) {
  return [...(execution?.candidates || [])].sort((a, b) => Number(b?.candidate_metrics?.ending_capital || 0) - Number(a?.candidate_metrics?.ending_capital || 0))
}

function experimentSequence(candidateId, fallbackIndex) {
  const match = String(candidateId || '').match(/(\d+)(?!.*\d)/)
  return match ? Number(match[1]) : fallbackIndex + 1
}

function normalizedExperimentRows(rows, kind) {
  return (Array.isArray(rows) ? rows : [])
    .filter((item) => String(item?.candidate_id || '').toUpperCase() !== 'CONTROL')
    .map((item, index) => {
      const passed = kind === 'research'
        ? item?.robust_vs_control === true
        : item?.validation_pass === true
      return {
        ...item,
        sequence: experimentSequence(item?.candidate_id, index),
        status: passed ? 'pass' : 'fail',
      }
    })
    .sort((left, right) => left.sequence - right.sequence || String(left.candidate_id).localeCompare(String(right.candidate_id)))
}

function SummaryMetric({ label, value, tone = '' }) {
  return <div className={`rq-dashboard-metric ${tone}`}><span>{tr(label)}</span><strong>{value}</strong></div>
}

function ExperimentTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const point = payload.find((item) => item?.payload?.candidate_id)?.payload
  if (!point) return null
  const foldWins = point.folds_beating_control
  const foldCount = point.fold_count
  return <div className="analytics-performance-tooltip rq-experiment-tooltip">
    <strong>{point.candidate_id}</strong>
    <div><span>{tr('Status')}</span><b className={point.status === 'pass' ? 'positive' : 'negative'}>{point.status === 'pass' ? 'PASS' : 'FAIL'}</b></div>
    <div><span>{tr('Ending capital')}</span><b>{money(point.ending_capital)}</b></div>
    <div><span>{tr('Capital lift')}</span><b className={Number(point.capital_lift_vs_control) >= 0 ? 'positive' : 'negative'}>{percent(point.capital_lift_vs_control, 2)}</b></div>
    <div><span>{tr('Sharpe')}</span><b>{number(point.sharpe, 4)}</b></div>
    <div><span>{tr('Max Drawdown')}</span><b>{percent(point.max_drawdown, 2)}</b></div>
    {foldWins == null ? null : <div><span>{tr('Fold wins')}</span><b>{foldWins}/{foldCount || '—'}</b></div>}
  </div>
}

function ExperimentResultsChart({ rows, controlCapital, filter, onFilterChange, loading }) {
  const visibleRows = useMemo(() => rows.filter((row) => filter === 'all' || row.status === filter), [filter, rows])
  const values = visibleRows.map((row) => Number(row.ending_capital)).filter(Number.isFinite)
  const control = Number(controlCapital)
  if (Number.isFinite(control)) values.push(control)
  const domain = useMemo(() => {
    if (!values.length) return ['auto', 'auto']
    const min = Math.min(...values)
    const max = Math.max(...values)
    const spread = Math.max(max - min, Math.abs(max) * .02, 1)
    return [Math.max(0, min - spread * .08), max + spread * .08]
  }, [values.join('|')])

  return <article className="rq-dashboard-experiment-card">
    <div className="rq-dashboard-chart-head">
      <div>
        <span className="panel-kicker">{tr('ROTATION QUALITY')}</span>
        <strong>{tr('Experiment results')}</strong>
      </div>
      <div className="rotation-monthly-heatmap-modes rq-experiment-filter" role="group" aria-label={tr('Experiment result filter')}>
        {[['pass', 'PASS'], ['fail', 'FAIL'], ['all', 'PASS & FAIL']].map(([key, label]) => <button key={key} type="button" className={filter === key ? 'active' : ''} onClick={() => onFilterChange(key)}>{label}</button>)}
      </div>
    </div>

    {loading ? <div className="backtest-loading-row">{tr('Loading experiments…')}</div> : visibleRows.length ? <div className="rq-dashboard-experiment-chart">
      <AnalyticsResponsiveContainer fallbackHeight={300}>
        <LineChart data={visibleRows} margin={{ top: 12, right: 18, bottom: 8, left: 12 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="candidate_id" minTickGap={34} interval="preserveStartEnd" />
          <YAxis domain={domain} tickFormatter={(value) => `$${Math.round(Number(value) / 1000)}k`} width={62} />
          <Tooltip content={<ExperimentTooltip />} cursor={{ stroke: 'rgba(147, 177, 210, .45)', strokeDasharray: '4 4' }} />
          {Number.isFinite(control) ? <ReferenceLine y={control} label={{ value: tr('Control'), position: 'insideTopRight' }} stroke="var(--accent)" strokeDasharray="5 4" /> : null}
          <Line
            type="linear"
            dataKey="ending_capital"
            name={tr('Ending capital')}
            stroke="var(--positive)"
            strokeWidth={2.2}
            dot={(props) => {
              const status = props?.payload?.status
              return <circle cx={props.cx} cy={props.cy} r={3.2} fill={status === 'pass' ? 'var(--positive)' : 'var(--negative)'} stroke="none" />
            }}
            activeDot={{ r: 5 }}
            isAnimationActive={false}
          />
        </LineChart>
      </AnalyticsResponsiveContainer>
    </div> : <div className="global-inline-message">{tr(filter === 'pass' ? 'No PASS experiments are available for the selected execution.' : filter === 'fail' ? 'No FAIL experiments are available for the selected execution.' : 'No experiments are available for the selected execution.')}</div>}
  </article>
}

function RotationQualityMonthDialog({ month, processingId, candidateId, onClose }) {
  if (!month || typeof document === 'undefined') return null
  const detailUrl = `${API}/temporal-rotation-quality-research/analytics/processings/${encodeURIComponent(processingId)}/rotation-period?candidate_id=${encodeURIComponent(candidateId)}&year=${encodeURIComponent(month.year)}&month=${encodeURIComponent(month.monthNumber)}`
  return createPortal(
    <div className="monthly-return-dialog-backdrop" role="presentation" onMouseDown={onClose}>
      <section className="monthly-return-dialog rq-month-dialog" role="dialog" aria-modal="true" onMouseDown={(event) => event.stopPropagation()}>
        <header className="monthly-return-dialog-header">
          <div>
            <span className="panel-kicker">{tr('ROTATION QUALITY')}</span>
            <h3>{month.month} {month.year} · {candidateId}</h3>
          </div>
          <button type="button" className="monthly-return-dialog-close" onClick={onClose} aria-label={tr('Close')}>×</button>
        </header>
        <div className="monthly-return-dialog-metrics">
          <div><span>{tr('Strategy')}</span><strong className={month.simulation >= 0 ? 'positive' : 'negative'}>{percent(month.simulation)}</strong></div>
          <div><span>{tr('Control')}</span><strong className={month.reference >= 0 ? 'positive' : 'negative'}>{percent(month.reference)}</strong></div>
          <div><span>S − C</span><strong className={month.excess >= 0 ? 'positive' : 'negative'}>{percent(month.excess)}</strong></div>
        </div>
        <MonthlyAssetAnalysis
          jobId={processingId}
          month={{ year: month.year, month: month.monthNumber }}
          detailUrl={detailUrl}
        />
      </section>
    </div>,
    document.body,
  )
}

export function RotationQualityPerformanceSection() {
  const [executions, setExecutions] = useState([])
  const [processingId, setProcessingId] = useState('')
  const [candidateId, setCandidateId] = useState('')
  const [data, setData] = useState(null)
  const [experimentRows, setExperimentRows] = useState([])
  const [experimentFilter, setExperimentFilter] = useState('pass')
  const [experimentLoading, setExperimentLoading] = useState(false)
  const [experimentError, setExperimentError] = useState('')
  const [mode, setMode] = useState('simulation')
  const [selectedMonth, setSelectedMonth] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    let active = true
    apiFetch(`${API}/temporal-rotation-quality-research/analytics/processings?limit=50`)
      .then((payload) => {
        if (!active) return
        const items = Array.isArray(payload?.items) ? payload.items : []
        setExecutions(items)
        if (items.length) setProcessingId((current) => current || String(items[0].id || ''))
      })
      .catch((requestError) => { if (active) setError(requestError.message || tr('Unable to load Rotation Quality analytics.')) })
    return () => { active = false }
  }, [])

  const execution = useMemo(() => executions.find((item) => String(item.id) === String(processingId)) || null, [executions, processingId])
  const availableCandidates = useMemo(() => candidateRows(execution), [execution])

  useEffect(() => {
    if (!availableCandidates.length) {
      setCandidateId('')
      return
    }
    if (!availableCandidates.some((item) => String(item.candidate_id) === String(candidateId))) {
      setCandidateId(String(availableCandidates[0].candidate_id || ''))
    }
  }, [availableCandidates, candidateId])

  useEffect(() => {
    if (!processingId || !execution?.research_id) {
      setExperimentRows([])
      return undefined
    }
    let active = true
    setExperimentLoading(true)
    setExperimentError('')
    const kind = String(execution.kind || 'research')
    const request = kind === 'research'
      ? apiFetch(`${API}/temporal-rotation-quality-research/${encodeURIComponent(execution.research_id)}/candidates?limit=2000`)
          .then((payload) => payload?.items || [])
      : apiFetch(`${API}/temporal-rotation-quality-research/${encodeURIComponent(execution.research_id)}/validations/${encodeURIComponent(processingId)}`)
          .then((payload) => payload?.candidates || [])
    request
      .then((rows) => { if (active) setExperimentRows(normalizedExperimentRows(rows, kind)) })
      .catch((requestError) => {
        if (!active) return
        setExperimentRows([])
        setExperimentError(requestError.message || tr('Unable to load experiment results.'))
      })
      .finally(() => { if (active) setExperimentLoading(false) })
    return () => { active = false }
  }, [execution, processingId])

  useEffect(() => {
    if (!processingId || !candidateId) {
      setData(null)
      return undefined
    }
    let active = true
    setLoading(true)
    setError('')
    apiFetch(`${API}/temporal-rotation-quality-research/analytics/processings/${encodeURIComponent(processingId)}?candidate_id=${encodeURIComponent(candidateId)}`)
      .then((payload) => { if (active) setData(payload) })
      .catch((requestError) => {
        if (!active) return
        setData(null)
        setError(requestError.message || tr('Unable to load Rotation Quality analytics.'))
      })
      .finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [candidateId, processingId])

  const candidate = data?.candidate_metrics || {}
  const control = data?.control_metrics || availableCandidates[0]?.control_metrics || {}
  const lift = Number(candidate.capital_lift_vs_control)
  const foldWins = candidate.folds_beating_control
  const foldCount = candidate.fold_count || data?.candidate_metrics?.fold_count

  return (
    <section className="dashboard-analytics-hub rq-dashboard-performance">
      <div className="dashboard-analytics-history-head rq-dashboard-performance-head">
        <div>
          <span className="panel-kicker">{tr('ROTATION QUALITY')}</span>
          <h2>{tr('Monthly strategy performance')}</h2>
        </div>
        <div className="rq-dashboard-selectors">
          <label className="dashboard-analytics-run-select">
            <span>{tr('Execution')}</span>
            <select value={processingId} onChange={(event) => setProcessingId(event.target.value)} disabled={!executions.length || loading}>
              {executions.map((item) => <option key={item.id} value={item.id}>{kindLabel(item.kind)} · {shortDateTime(item.finished_at || item.created_at)}</option>)}
            </select>
          </label>
          <label className="dashboard-analytics-run-select">
            <span>{tr('Candidate')}</span>
            <select value={candidateId} onChange={(event) => setCandidateId(event.target.value)} disabled={!availableCandidates.length || loading}>
              {availableCandidates.map((item) => <option key={item.candidate_id} value={item.candidate_id}>{item.candidate_id}</option>)}
            </select>
          </label>
        </div>
      </div>

      {!executions.length && !error ? <div className="global-inline-message">{tr('No Rotation Quality monthly analytics are available yet. New completed Research, Validation and Certification runs will appear here.')}</div> : null}
      {loading ? <div className="backtest-loading-row">{tr('Loading Rotation Quality analytics…')}</div> : null}
      {error ? <div className="global-inline-message error-inline">{tr(error)}</div> : null}

      {processingId ? <ExperimentResultsChart
        rows={experimentRows}
        controlCapital={control?.ending_capital}
        filter={experimentFilter}
        onFilterChange={setExperimentFilter}
        loading={experimentLoading}
      /> : null}
      {experimentError ? <div className="global-inline-message error-inline">{tr(experimentError)}</div> : null}

      {data ? <>
        <div className="rq-dashboard-summary-grid">
          <SummaryMetric label="Strategy capital" value={money(candidate.ending_capital)} tone={lift >= 0 ? 'positive' : 'negative'} />
          <SummaryMetric label="Control capital" value={money(control.ending_capital)} />
          <SummaryMetric label="Capital lift" value={percent(lift, 2)} tone={lift >= 0 ? 'positive' : 'negative'} />
          <SummaryMetric label="Sharpe" value={number(candidate.sharpe, 4)} />
          <SummaryMetric label="Max Drawdown" value={percent(candidate.max_drawdown, 2)} />
          <SummaryMetric label="Fold wins" value={foldWins == null ? '—' : `${foldWins}/${foldCount || '—'}`} />
        </div>

        <article className="rq-dashboard-heatmap-card">
          <div className="rq-dashboard-heatmap-head">
            <div><strong>{tr('Monthly Return Heatmap')}</strong></div>
            <div className="rotation-monthly-heatmap-modes" role="group" aria-label={tr('Monthly return view')}>
              {[['simulation', 'Strategy'], ['reference', 'Control'], ['excess', 'S − C']].map(([key, label]) => <button key={key} type="button" className={mode === key ? 'active' : ''} onClick={() => setMode(key)}>{tr(label)}</button>)}
            </div>
          </div>
          <MonthlyReturnHeatmap rows={data.monthly_returns || []} mode={mode} simulationLabel={tr('Strategy')} referenceLabel={tr('Control')} excessLabel="S − C" onMonthSelect={setSelectedMonth} />
        </article>
      </> : null}

      <RotationQualityMonthDialog month={selectedMonth} processingId={processingId} candidateId={candidateId} onClose={() => setSelectedMonth(null)} />
    </section>
  )
}
