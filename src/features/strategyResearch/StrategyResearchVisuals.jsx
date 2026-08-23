import { useMemo, useState } from 'react'
import { createPortal } from 'react-dom'

import { getIntlLocale, tr } from '../../i18n/runtime'
import { CoffeeProgress } from '../../shared/CoffeeProgress'
import { money, number, percent } from '../../shared/formatters'
import { LeadershipRegimePanel } from '../leadershipRegime'
import { FragileIncumbentPanel } from '../fragileIncumbent/FragileIncumbentPanel'
import { RegimeClusteringPanel } from '../regimeClustering/RegimeClusteringPanel'
import { EmergingTrendPanel } from '../emergingTrend/EmergingTrendPanel'

function MetricCard({ label, value, note, tone = '' }) {
  return <div className={`strategy-research-metric-card ${tone}`}><span>{label}</span><strong>{value}</strong>{note ? <small>{note}</small> : null}</div>
}

function EmptyVisual({ title, detail, loading = false, progress = null }) {
  return <div className={`strategy-research-empty-visual ${loading ? 'loading' : ''}`}>
    {loading ? <CoffeeProgress progress={progress} label={tr('Processing')} /> : <span className="strategy-research-empty-icon" aria-hidden="true">◇</span>}
    <strong>{tr(loading ? 'Processing' : title)}</strong>
    {loading ? <small>{tr(title)}</small> : detail ? <small>{tr(detail)}</small> : null}
  </div>
}

function monthKey(value) {
  const text = String(value || '')
  if (/^\d{4}-\d{2}/.test(text)) return text.slice(0, 7)
  const parsed = Date.parse(text)
  if (!Number.isFinite(parsed)) return ''
  const date = new Date(parsed)
  return `${date.getUTCFullYear()}-${String(date.getUTCMonth() + 1).padStart(2, '0')}`
}

function monthNames() {
  const formatter = new Intl.DateTimeFormat(getIntlLocale(), { month: 'short', timeZone: 'UTC' })
  return Array.from({ length: 12 }, (_, index) => formatter.format(new Date(Date.UTC(2024, index, 1))).replace('.', ''))
}

function fullMonthLabel(year, monthNumber) {
  return new Intl.DateTimeFormat(getIntlLocale(), { month: 'long', year: 'numeric', timeZone: 'UTC' })
    .format(new Date(Date.UTC(Number(year), Number(monthNumber) - 1, 1)))
}

function equityValue(row) {
  const values = [row?.simulation_equity, row?.equity, row?.portfolio_value, row?.value, row?.strategy_equity]
  for (const value of values) {
    const parsed = Number(value)
    if (Number.isFinite(parsed) && parsed > 0) return parsed
  }
  return null
}

function monthlyReturns(equity = []) {
  const grouped = new Map()
  for (const row of equity || []) {
    const month = monthKey(row?.timestamp || row?.recorded_at || row?.date)
    const value = equityValue(row)
    if (!month || value == null) continue
    const current = grouped.get(month) || { first: value, last: value, observations: 0 }
    current.last = value
    current.observations += 1
    grouped.set(month, current)
  }
  return [...grouped.entries()].sort(([left], [right]) => left.localeCompare(right)).map(([month, values]) => ({
    month,
    first: values.first,
    last: values.last,
    observations: values.observations,
    capitalDelta: values.last - values.first,
    value: values.first > 0 ? values.last / values.first - 1 : 0,
  }))
}

function yearlyReturns(rows = []) {
  const grouped = new Map()
  for (const row of rows) {
    const year = row.month.slice(0, 4)
    const bucket = grouped.get(year) || []
    bucket.push(row)
    grouped.set(year, bucket)
  }
  return new Map([...grouped.entries()].map(([year, values]) => {
    const ordered = [...values].sort((left, right) => left.month.localeCompare(right.month))
    const first = ordered[0]?.first
    const last = ordered[ordered.length - 1]?.last
    const ranked = [...ordered].sort((left, right) => right.value - left.value)
    return [year, {
      year,
      first,
      last,
      capitalDelta: Number.isFinite(first) && Number.isFinite(last) ? last - first : null,
      value: Number.isFinite(first) && first > 0 && Number.isFinite(last) ? last / first - 1 : null,
      months: ordered.length,
      positiveMonths: ordered.filter((item) => item.value > 0).length,
      negativeMonths: ordered.filter((item) => item.value < 0).length,
      best: ranked[0] || null,
      worst: ranked[ranked.length - 1] || null,
    }]
  }))
}

function heatTone(value, maxAbs) {
  if (!Number.isFinite(value)) return 'neutral'
  const normalized = maxAbs > 0 ? Math.min(1, Math.abs(value) / maxAbs) : 0
  if (value > 0) return normalized > 0.66 ? 'positive strong' : normalized > 0.33 ? 'positive medium' : 'positive soft'
  if (value < 0) return normalized > 0.66 ? 'negative strong' : normalized > 0.33 ? 'negative medium' : 'negative soft'
  return 'neutral'
}

function HeatmapDetailDialog({ detail, yearSummary, onClose }) {
  if (!detail || typeof document === 'undefined') return null
  const isYear = detail.kind === 'year'
  const summary = isYear ? detail : yearSummary
  const title = isYear ? String(detail.year) : fullMonthLabel(detail.year, detail.monthNumber)
  const tone = Number(detail.value) > 0 ? 'positive' : Number(detail.value) < 0 ? 'negative' : ''
  return createPortal(<div className="strategy-research-heatmap-dialog-backdrop" role="presentation" onMouseDown={onClose}>
    <section className="strategy-research-heatmap-dialog" role="dialog" aria-modal="true" aria-label={title} onMouseDown={(event) => event.stopPropagation()}>
      <header className="strategy-research-heatmap-dialog-header">
        <div><span className="panel-kicker">{tr('STRATEGY REPLAY')}</span><h3>{title}</h3><p>{tr(isYear ? 'Annual return detail for the Strategy replay.' : 'Monthly return detail for the Strategy replay.')}</p></div>
        <button type="button" onClick={onClose} aria-label={tr('Close')}>×</button>
      </header>
      <div className="strategy-research-heatmap-dialog-metrics">
        <div><span>{tr(isYear ? 'Annual return' : 'Monthly return')}</span><strong className={tone}>{percent(detail.value, 2)}</strong></div>
        <div><span>{tr('Starting capital')}</span><strong>{money(detail.first)}</strong></div>
        <div><span>{tr('Ending capital')}</span><strong>{money(detail.last)}</strong></div>
        <div><span>{tr('Capital movement')}</span><strong className={Number(detail.capitalDelta) > 0 ? 'positive' : Number(detail.capitalDelta) < 0 ? 'negative' : ''}>{money(detail.capitalDelta)}</strong></div>
      </div>
      {!isYear ? <div className="strategy-research-heatmap-dialog-context">
        <div><span>{tr('Year')}</span><strong>{detail.year}</strong></div>
        <div><span>{tr('Month')}</span><strong>{fullMonthLabel(detail.year, detail.monthNumber)}</strong></div>
        <div><span>{tr('Observations')}</span><strong>{number(detail.observations, 0)}</strong></div>
        <div><span>{tr('Year return')}</span><strong className={Number(summary?.value) > 0 ? 'positive' : Number(summary?.value) < 0 ? 'negative' : ''}>{summary ? percent(summary.value, 2) : '—'}</strong></div>
      </div> : <div className="strategy-research-heatmap-dialog-context">
        <div><span>{tr('Months observed')}</span><strong>{number(detail.months, 0)}</strong></div>
        <div><span>{tr('Positive months')}</span><strong className="positive">{number(detail.positiveMonths, 0)}</strong></div>
        <div><span>{tr('Negative months')}</span><strong className="negative">{number(detail.negativeMonths, 0)}</strong></div>
        <div><span>{tr('Best / worst month')}</span><strong>{detail.best ? `${fullMonthLabel(detail.year, Number(detail.best.month.slice(5, 7)))} ${percent(detail.best.value, 1)}` : '—'} · {detail.worst ? `${percent(detail.worst.value, 1)}` : '—'}</strong></div>
      </div>}
    </section>
  </div>, document.body)
}


function ResearchDetailDialog({ detail, onClose }) {
  if (!detail || typeof document === 'undefined') return null
  return createPortal(<div className="strategy-research-heatmap-dialog-backdrop" role="presentation" onMouseDown={onClose}>
    <section className="strategy-research-heatmap-dialog" role="dialog" aria-modal="true" aria-label={detail.title} onMouseDown={(event) => event.stopPropagation()}>
      <header className="strategy-research-heatmap-dialog-header">
        <div><span className="panel-kicker">{tr(detail.kicker || 'RESEARCH DETAIL')}</span><h3>{detail.title}</h3>{detail.description ? <p>{detail.description}</p> : null}</div>
        <button type="button" onClick={onClose} aria-label={tr('Close')}>×</button>
      </header>
      {detail.metrics?.length ? <div className="strategy-research-heatmap-dialog-metrics">{detail.metrics.map((item, index) => <div key={`${item.label}-${index}`}><span>{item.label}</span><strong className={item.tone || ''}>{item.value}</strong></div>)}</div> : null}
      {detail.notes?.length ? <div className="strategy-research-detail-dialog-notes">{detail.notes.map((note, index) => <div key={index}><span>{note.label}</span><p>{note.text}</p></div>)}</div> : null}
    </section>
  </div>, document.body)
}

function MonthlyHeatmap({ analytics }) {
  const [selectedDetail, setSelectedDetail] = useState(null)
  const rows = useMemo(() => monthlyReturns(analytics?.equity || []), [analytics?.equity])
  const yearMap = useMemo(() => yearlyReturns(rows), [rows])
  if (!rows.length) return <EmptyVisual title="Strategy replay visualization will appear here." />
  const years = [...new Set(rows.map((item) => item.month.slice(0, 4)))]
  const byMonth = new Map(rows.map((item) => [item.month, item]))
  const maxAbs = Math.max(...rows.map((item) => Math.abs(Number(item.value || 0))), ...[...yearMap.values()].map((item) => Math.abs(Number(item.value || 0))), 0.0001)
  const months = monthNames()
  return <div className="strategy-research-calendar-heatmap-wrap">
    <div className="strategy-research-heatmap-heading">
      <div><strong>{tr('Monthly Return Heatmap')}</strong></div>
      <span>{tr('Monthly change in Strategy equity')}</span>
    </div>
    <div className="strategy-research-calendar-heatmap" role="grid" aria-label={tr('Monthly Return Heatmap')}>
      <div className="strategy-research-month-labels"><span />{months.map((name) => <span key={name}>{name}</span>)}<strong>{tr('Year total')}</strong></div>
      {years.map((year) => {
        const yearSummary = yearMap.get(year)
        return <div className="strategy-research-heatmap-row" key={year}>
          <strong>{year}</strong>
          {Array.from({ length: 12 }, (_, index) => {
            const key = `${year}-${String(index + 1).padStart(2, '0')}`
            const detail = byMonth.get(key)
            return <button type="button" className={`strategy-research-heat-cell ${detail == null ? 'missing' : heatTone(detail.value, maxAbs)}`} key={key} aria-label={detail == null ? `${months[index]} ${year}` : `${months[index]} ${year} ${percent(detail.value, 2)}`} disabled={!detail} onClick={() => detail && setSelectedDetail({ ...detail, kind: 'month', year, monthNumber: index + 1 })}><span>{detail == null ? '—' : percent(detail.value, 0)}</span></button>
          })}
          <button type="button" className={`strategy-research-heat-cell year-total ${yearSummary ? heatTone(yearSummary.value, maxAbs) : 'missing'}`} disabled={!yearSummary} onClick={() => yearSummary && setSelectedDetail({ ...yearSummary, kind: 'year' })}><span>{yearSummary ? percent(yearSummary.value, 0) : '—'}</span></button>
        </div>
      })}
    </div>
    <div className="strategy-research-heatmap-footer">
      <div className="strategy-research-heatmap-legend" aria-label={tr('Heatmap color legend')}>
        <span>{tr('Higher loss')}</span><i className="negative strong"/><i className="negative soft"/><span>{tr('Near zero')}</span><i className="positive soft"/><i className="positive strong"/><span>{tr('Higher gain')}</span>
      </div>
      <small>{tr('Color intensity represents the magnitude of monthly return in the displayed period.')}</small>
      <small>{tr('Click a month or year total for detailed analysis')}</small>
    </div>
    <HeatmapDetailDialog detail={selectedDetail} yearSummary={selectedDetail?.year ? yearMap.get(String(selectedDetail.year)) : null} onClose={() => setSelectedDetail(null)} />
  </div>
}

function horizonCellValue(row, metric) {
  const profitSignal = (row?.signal_metrics || []).find((item) => item?.signal === 'profit_before_loss') || {}
  if (metric === 'auc') return Number(row?.profit_before_loss_auc ?? profitSignal?.auc)
  if (metric === 'brier_skill') return Number(row?.profit_before_loss_brier_skill ?? profitSignal?.brier_skill)
  if (metric === 'confidence') return Number(profitSignal?.high_confidence_positive_rate)
  if (metric === 'lift') return Number(row?.profit_before_loss_high_confidence_lift ?? profitSignal?.high_confidence_lift)
  if (metric === 'drawdown') return Number(row?.drawdown_mae_skill)
  return NaN
}

function TemporalHeatmap({ run }) {
  const [selectedDetail, setSelectedDetail] = useState(null)
  const rows = run?.result?.horizon_metrics || []
  if (!rows.length) return <EmptyVisual title="Temporal horizon analysis will appear here." />
  const metrics = [
    { key: 'auc', label: 'AUC', description: 'Measures how well the profit-before-loss model separates positive from negative outcomes for this horizon.', relationship: '0.50 ≈ random · higher is better · 1.00 = perfect ranking', example: 'AUC 0.628 at 5d means the 5-session model has useful, but not strong, discrimination.' },
    { key: 'brier_skill', label: 'Brier Skill', description: 'Measures probability accuracy relative to the baseline probability model. Positive values improve on the baseline; negative values are worse.', relationship: '> 0 = better than baseline · 0 = same as baseline · < 0 = worse', example: '4.3% means the calibrated probability error improved by about 4.3% versus the baseline.' },
    { key: 'confidence', label: 'High Confidence Hit Rate', description: 'Among profit-before-loss predictions with probability at or above 70%, this is the share whose realized outcome was positive.', relationship: 'Only high-confidence predictions are counted · higher is better', example: '57.1% means 57.1% of qualifying high-confidence predictions were correct.' },
    { key: 'lift', label: 'High Confidence Lift', description: 'Shows how much the high-confidence hit rate exceeds the overall positive-outcome rate for the same horizon.', relationship: 'High-confidence hit rate − overall positive rate', example: '20.1% means high-confidence signals beat the horizon base positive rate by 20.1 percentage points.' },
    { key: 'drawdown', label: 'Drawdown MAE Skill', description: 'Measures improvement in predicted drawdown error versus the baseline drawdown predictor.', relationship: '> 0 = lower MAE than baseline · higher is better', example: '12.8% means the model reduced drawdown MAE by about 12.8% versus the baseline.' },
  ]
  return <div className="strategy-research-matrix">
    <div className="strategy-research-matrix-head"><span />{rows.map((row) => <strong key={row.horizon}>{row.horizon}d</strong>)}</div>
    {metrics.map(({ key, label, description, relationship, example }) => {
      const values = rows.map((row) => horizonCellValue(row, key)).filter(Number.isFinite)
      const maxAbs = Math.max(...values.map(Math.abs), 0.0001)
      return <div className="strategy-research-matrix-row" key={key}><strong className="strategy-research-metric-label"><span>{tr(label)}</span></strong>{rows.map((row) => {
        const value = horizonCellValue(row, key)
        const display = Number.isFinite(value) ? (key === 'auc' ? number(value, 3) : percent(value, 1)) : '—'
        const tone = Number(value) > 0 ? 'positive' : Number(value) < 0 ? 'negative' : ''
        return <button type="button" className={`strategy-research-matrix-cell ${Number.isFinite(value) ? heatTone(value, maxAbs) : 'missing'}`} key={row.horizon} disabled={!Number.isFinite(value)} aria-label={`${row.horizon}d · ${tr(label)} · ${display}`} onClick={() => setSelectedDetail({
          kicker: 'TEMPORAL INTELLIGENCE',
          title: `${tr(label)} · ${row.horizon}d`,
          description: tr(description),
          metrics: [
            { label: tr('Value'), value: display, tone },
            { label: tr('Forecast horizon'), value: `${row.horizon}d` },
          ],
          notes: [
            { label: tr('How to read it'), text: tr(relationship) },
            { label: tr('Example'), text: tr(example) },
            { label: tr('Horizon'), text: tr('The horizon is measured in trading sessions.') },
          ],
        })}>{display}</button>
      })}</div>
    })}
    <ResearchDetailDialog detail={selectedDetail} onClose={() => setSelectedDetail(null)} />
  </div>
}

function BubbleQuadrant({ risk, intervention }) {
  const [selectedDetail, setSelectedDetail] = useState(null)
  const rows = risk?.oos?.scored_transitions || []
  const severeThreshold = Number(risk?.research_settings?.settings?.risk?.severe_threshold ?? -0.05)
  const metrics = risk?.oos?.metrics || {}
  const points = rows.map((row, index) => {
    const riskScore = Number(row?.risk_score)
    const riskThreshold = Number(row?.risk_threshold)
    const valueAdded = Number(row?.rotation_value_added)
    return {
      id: `${row?.transition_key || index}`,
      x: riskScore - riskThreshold,
      riskScore,
      riskThreshold,
      y: valueAdded,
      severe: Boolean(row?.severe),
      highRisk: Boolean(row?.high_risk),
      label: `${row?.from_asset || '—'} → ${row?.to_asset || '—'}`,
      row,
    }
  }).filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y))
  if (!points.length) return <EmptyVisual title="Risk and intervention bubbles will appear here." />

  const xAbs = Math.max(...points.map((point) => Math.abs(point.x)), 0.04) * 1.08
  const yMinRaw = Math.min(...points.map((point) => point.y), severeThreshold, -0.01)
  const yMaxRaw = Math.max(...points.map((point) => point.y), 0.01)
  const yPadding = Math.max((yMaxRaw - yMinRaw) * 0.08, 0.01)
  const minX = -xAbs
  const maxX = xAbs
  const minY = yMinRaw - yPadding
  const maxY = yMaxRaw + yPadding
  const plot = { left: 72, right: 734, top: 54, bottom: 304 }
  const mapX = (value) => plot.left + ((value - minX) / Math.max(1e-9, maxX - minX)) * (plot.right - plot.left)
  const mapY = (value) => plot.bottom - ((value - minY) / Math.max(1e-9, maxY - minY)) * (plot.bottom - plot.top)
  const zeroX = mapX(0)
  const zeroY = mapY(0)
  const xTicks = Array.from({ length: 5 }, (_, index) => minX + ((maxX - minX) * index) / 4)
  const yTicks = Array.from({ length: 5 }, (_, index) => minY + ((maxY - minY) * index) / 4)
  const interventionMetrics = intervention?.walk_forward_selected_shadow || risk?.shadow_replay || null
  const total = Number(metrics?.count ?? points.length)
  const flagged = Number(metrics?.flagged_count ?? points.filter((point) => point.highRisk).length)
  const severe = Number(metrics?.severe_count ?? points.filter((point) => point.severe).length)
  const captured = Number(metrics?.captured_severe_count ?? points.filter((point) => point.severe && point.highRisk).length)
  const precision = Number(metrics?.precision)
  const recall = Number(metrics?.recall)

  const quadrantLabel = (point) => {
    if (point.highRisk && point.y < 0) return tr('Harmful transition captured by the risk alert')
    if (point.highRisk && point.y >= 0) return tr('Risk alert, but the rotation added value')
    if (!point.highRisk && point.y < 0) return tr('Harmful transition not flagged by the risk model')
    return tr('Beneficial transition left unchanged')
  }

  const openPoint = (point) => {
    const targetReturn = Number(point.row?.one_interval_target_return)
    const incumbentReturn = Number(point.row?.one_interval_incumbent_return)
    const decisionDate = point.row?.decision_at || point.row?.execution_at || '—'
    setSelectedDetail({
      kicker: 'RISK & INTERVENTION',
      title: point.label,
      description: quadrantLabel(point),
      metrics: [
        { label: tr('Decision date'), value: String(decisionDate).replace('T', ' ').slice(0, 16) },
        { label: tr('Risk score'), value: number(point.riskScore, 3) },
        { label: tr('Risk threshold'), value: number(point.riskThreshold, 3) },
        { label: tr('Risk margin'), value: `${point.x >= 0 ? '+' : ''}${number(point.x, 3)}`, tone: point.x >= 0 ? 'negative' : '' },
        { label: tr('Risk alert'), value: point.highRisk ? tr('Yes') : tr('No'), tone: point.highRisk ? 'negative' : '' },
        { label: tr('Realized value added'), value: percent(point.y, 2), tone: point.y > 0 ? 'positive' : point.y < 0 ? 'negative' : '' },
        { label: tr('Severity'), value: point.severe ? tr('Severe') : tr('Standard'), tone: point.severe ? 'negative' : '' },
        { label: tr('Transition'), value: point.label },
        ...(Number.isFinite(targetReturn) ? [{ label: tr('Target interval return'), value: percent(targetReturn, 2), tone: targetReturn > 0 ? 'positive' : targetReturn < 0 ? 'negative' : '' }] : []),
        ...(Number.isFinite(incumbentReturn) ? [{ label: tr('Incumbent interval return'), value: percent(incumbentReturn, 2), tone: incumbentReturn > 0 ? 'positive' : incumbentReturn < 0 ? 'negative' : '' }] : []),
      ],
      notes: [
        { label: tr('Classification'), text: quadrantLabel(point) },
        { label: tr('Risk margin'), text: tr('Risk margin is risk score minus the chronological threshold selected for that out-of-sample year. Values at or above zero generated a risk alert.') },
        { label: tr('Realized value added'), text: tr('Positive realized value added means the rotation outperformed the counterfactual hold. Negative value means the rotation destroyed value relative to holding.') },
        { label: tr('Severe transition'), text: `${tr('The configured severe-loss boundary for this research run is')} ${percent(severeThreshold, 1)}.` },
      ],
    })
  }

  return <div className="strategy-research-bubble-wrap strategy-research-risk-map">
    <div className="strategy-research-risk-intro">
      <div>
        <strong>{tr('Risk detection map')}</strong>
        <span>{tr('Each point is an out-of-sample rotation. Horizontal position is the risk margin relative to the threshold selected for that year; vertical position is the realized economic value added by the rotation.')}</span>
      </div>
      <div className="strategy-research-risk-legend" aria-label={tr('Chart legend')}>
        <span><i className="standard" />{tr('Standard transition')}</span>
        <span><i className="severe" />{tr('Severe loss')}</span>
        <span><i className="alert" />{tr('Risk alert')}</span>
      </div>
    </div>

    <div className="strategy-research-risk-kpis">
      <MetricCard label={tr('OOS transitions')} value={number(total, 0)} />
      <MetricCard label={tr('Risk alerts')} value={number(flagged, 0)} note={total > 0 ? percent(flagged / total, 1) : '—'} />
      <MetricCard label={tr('Severe losses')} value={number(severe, 0)} />
      <MetricCard label={tr('Captured severe')} value={number(captured, 0)} />
      <MetricCard label={tr('Precision')} value={Number.isFinite(precision) ? percent(precision, 1) : '—'} />
      <MetricCard label={tr('Recall')} value={Number.isFinite(recall) ? percent(recall, 1) : '—'} />
    </div>

    <svg viewBox="0 0 780 350" role="img" aria-label={tr('Risk margin versus realized value added')}>
      <rect x={plot.left} y={plot.top} width={zeroX - plot.left} height={zeroY - plot.top} className="strategy-research-quadrant beneficial" />
      <rect x={zeroX} y={plot.top} width={plot.right - zeroX} height={zeroY - plot.top} className="strategy-research-quadrant cautioned" />
      <rect x={plot.left} y={zeroY} width={zeroX - plot.left} height={plot.bottom - zeroY} className="strategy-research-quadrant missed" />
      <rect x={zeroX} y={zeroY} width={plot.right - zeroX} height={plot.bottom - zeroY} className="strategy-research-quadrant captured" />

      {yTicks.map((tick) => <g key={`y-${tick}`}><line x1={plot.left} y1={mapY(tick)} x2={plot.right} y2={mapY(tick)} className="strategy-research-grid-line" /><text x={plot.left - 10} y={mapY(tick) + 4} textAnchor="end" className="strategy-research-axis-tick">{percent(tick, 0)}</text></g>)}
      {xTicks.map((tick) => <g key={`x-${tick}`}><line x1={mapX(tick)} y1={plot.top} x2={mapX(tick)} y2={plot.bottom} className="strategy-research-grid-line" /><text x={mapX(tick)} y={plot.bottom + 18} textAnchor="middle" className="strategy-research-axis-tick">{`${tick >= 0 ? '+' : ''}${number(tick, 2)}`}</text></g>)}

      <line x1={plot.left} y1={zeroY} x2={plot.right} y2={zeroY} className="strategy-research-axis zero" />
      <line x1={zeroX} y1={plot.top} x2={zeroX} y2={plot.bottom} className="strategy-research-axis threshold" />
      <line x1={plot.left} y1={mapY(severeThreshold)} x2={plot.right} y2={mapY(severeThreshold)} className="strategy-research-axis severe-boundary" />

      <text x={plot.left} y="25" className="strategy-research-svg-label axis-title">{tr('Realized rotation value added (%)')}</text>
      <text x={(plot.left + plot.right) / 2} y="342" textAnchor="middle" className="strategy-research-svg-label axis-title">{tr('Risk margin vs threshold (score − threshold)')}</text>
      <text x={zeroX + 7} y={plot.top + 14} className="strategy-research-threshold-label">{tr('Alert threshold')}</text>
      <text x={plot.right - 5} y={mapY(severeThreshold) - 6} textAnchor="end" className="strategy-research-severe-label">{`${tr('Severe')} ${percent(severeThreshold, 0)}`}</text>

      <text x={plot.left + 12} y={plot.top + 24} className="strategy-research-quadrant-label positive">{tr('Beneficial · no alert')}</text>
      <text x={plot.right - 12} y={plot.top + 24} textAnchor="end" className="strategy-research-quadrant-label warning">{tr('Alerted · rotation helped')}</text>
      <text x={plot.left + 12} y={plot.bottom - 14} className="strategy-research-quadrant-label negative">{tr('Harmful · missed risk')}</text>
      <text x={plot.right - 12} y={plot.bottom - 14} textAnchor="end" className="strategy-research-quadrant-label captured">{tr('Harmful · alert captured')}</text>

      {points.map((point) => <circle key={point.id} cx={mapX(point.x)} cy={mapY(point.y)} r={point.severe ? 6.5 : 3.6} className={`strategy-research-bubble ${point.severe ? 'severe' : ''} ${point.highRisk ? 'high-risk' : ''}`} role="button" tabIndex="0" aria-label={`${point.label} · ${tr('Risk score')} ${number(point.riskScore, 3)} · ${tr('Risk margin')} ${number(point.x, 3)} · ${tr('Value added')} ${percent(point.y, 2)}`} onClick={() => openPoint(point)} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') openPoint(point) }} />)}
    </svg>

    <div className="strategy-research-risk-reading">
      <strong>{tr('How to read')}</strong>
      <span>{tr('Right of the vertical line means the detector issued a high-risk alert. Below the horizontal zero line means the realized rotation destroyed value versus holding. The lower-right quadrant is therefore where the detector correctly identified economically harmful transitions.')}</span>
    </div>

    {interventionMetrics?.shadow ? <div className="strategy-research-compact-metrics"><MetricCard label={tr('Intervention capital')} value={money(interventionMetrics.shadow.ending_capital)} tone={Number(interventionMetrics.shadow.ending_capital_delta_rate || 0) >= 0 ? 'positive' : 'negative'} /><MetricCard label={tr('Capital delta')} value={percent(interventionMetrics.shadow.ending_capital_delta_rate, 2)} /><MetricCard label={tr('Interventions')} value={number(interventionMetrics.interventions, 0)} /></div> : null}
    <ResearchDetailDialog detail={selectedDetail} onClose={() => setSelectedDetail(null)} />
  </div>
}

function confidenceReason(reason) {
  if (reason === 'warmup_no_prior_oos_year') return tr('Warm-up year: no prior out-of-sample year is available to calibrate a confidence threshold.')
  if (reason === 'insufficient_prior_oos_alerts') return tr('Control kept: prior out-of-sample years do not contain enough risk alerts for a reliable confidence threshold.')
  if (reason === 'no_positive_tail_safe_confidence_gate') return tr('Control kept: no tested confidence threshold improved capital while preserving tail safety consistently across prior out-of-sample years.')
  if (reason === 'best_positive_tail_safe_prior_oos_confidence_gate') return tr('Confidence gate activated: the selected threshold was positive, tail-safe and chronologically consistent on prior out-of-sample years.')
  return tr('The calibration result did not activate a confidence intervention for this year.')
}

function ConfidenceTiles({ confidence }) {
  const [selectedDetail, setSelectedDetail] = useState(null)
  const rows = confidence?.outer_results || []
  if (!rows.length) return <EmptyVisual title="Confidence calibration tiles will appear here." />
  const values = rows.map((row) => row?.test_result?.ending_capital_delta_rate).filter((value) => value != null).map(Number).filter(Number.isFinite)
  const maxAbs = Math.max(...values.map(Math.abs), 0.0001)
  return <div className="strategy-research-confidence-tiles">{rows.map((row, index) => {
    const selectedMode = String(row?.selected_mode || '')
    const active = selectedMode === 'confidence_calibrated_one_session'
    const rawDelta = row?.test_result?.ending_capital_delta_rate
    const delta = rawDelta == null ? null : Number(rawDelta)
    const safe = Boolean(row?.test_result?.tail_safe)
    const rawThreshold = row?.selected_margin_threshold
    const threshold = rawThreshold == null ? null : Number(rawThreshold)
    const endingCapital = Number(row?.test_result?.ending_capital)
    const reason = confidenceReason(row?.reason)
    const tileTone = active && Number.isFinite(delta) ? heatTone(delta, maxAbs) : 'control'
    return <button type="button" className={`strategy-research-confidence-tile ${tileTone}`} key={`${row?.test_year || index}`} onClick={() => setSelectedDetail({
      kicker: 'CONFIDENCE CALIBRATION',
      title: `${tr('Test year')} ${row?.test_year || '—'}`,
      description: active
        ? tr('Shows the out-of-sample economic result obtained with the confidence threshold selected for this chronological test year.')
        : tr('No confidence intervention was activated for this chronological test year. The original Strategy decision was preserved.'),
      metrics: [
        { label: tr('Mode'), value: active ? tr('Active') : tr('Control') },
        { label: tr('Capital delta'), value: Number.isFinite(delta) ? percent(delta, 2) : active ? '—' : percent(0, 2), tone: Number(delta) > 0 ? 'positive' : Number(delta) < 0 ? 'negative' : '' },
        { label: tr('Margin threshold'), value: Number.isFinite(threshold) ? number(threshold, 3) : '—' },
        { label: tr('Tail safety'), value: active ? (safe ? tr('Tail safe') : tr('Tail warning')) : tr('Original Strategy preserved') },
      ],
      notes: [
        { label: tr('Selection reason'), text: reason },
        { label: tr('Interpretation'), text: tr('The calibration chooses how much confidence margin is required before a risk signal is allowed to change the original Strategy decision.') },
        { label: tr('Validation'), text: tr('Each tile represents an out-of-sample chronological test year, not an in-sample fit result.') },
        ...(Number.isFinite(endingCapital) ? [{ label: tr('Ending capital'), text: money(endingCapital) }] : []),
      ],
    })}>
      <span>{row?.test_year || '—'}</span>
      <strong>{active ? (Number.isFinite(delta) ? percent(delta, 1) : '—') : tr('Control')}</strong>
      <small>{active
        ? `${tr('Margin')} ${Number.isFinite(threshold) ? number(threshold, 3) : '—'} · ${safe ? tr('Tail safe') : tr('Tail warning')}`
        : tr('No calibrated intervention')}</small>
    </button>
  })}<ResearchDetailDialog detail={selectedDetail} onClose={() => setSelectedDetail(null)} /></div>
}

function decisionAction(previousAsset, currentAsset) {
  const previous = String(previousAsset || '').toUpperCase()
  const current = String(currentAsset || '').toUpperCase()
  if (!current) return ''
  if (current === 'CASH') return 'CASH'
  if (!previous) return 'HOLD'
  if (previous === 'CASH') return 'BUY'
  if (previous === current) return 'HOLD'
  return 'ROTATE'
}

function SankeyVisual({ stateful }) {
  const [selectedDetail, setSelectedDetail] = useState(null)
  const candidateEquity = stateful?.candidate_a?.analytics?.equity || []
  const referenceEquity = stateful?.control_replay?.analytics?.equity || []
  if (!candidateEquity.length || !referenceEquity.length) return <EmptyVisual title="Decision policy flow will appear here." />
  const referenceByTimestamp = new Map(referenceEquity.map((row, index) => [String(row?.timestamp || ''), { row, index }]))
  const counts = new Map()
  let aligned = 0
  candidateEquity.forEach((candidateRow, candidateIndex) => {
    const timestamp = String(candidateRow?.timestamp || '')
    const referenceMatch = referenceByTimestamp.get(timestamp)
    if (!referenceMatch) return
    const { row: referenceRow, index: referenceIndex } = referenceMatch
    const previousReference = referenceIndex > 0 ? referenceEquity[referenceIndex - 1]?.selected_asset : referenceRow?.selected_asset
    const previousCandidate = candidateIndex > 0 ? candidateEquity[candidateIndex - 1]?.selected_asset : candidateRow?.selected_asset
    const referenceAction = decisionAction(previousReference, referenceRow?.selected_asset)
    const statefulAction = decisionAction(previousCandidate, candidateRow?.selected_asset)
    if (!referenceAction || !statefulAction) return
    aligned += 1
    const key = `${referenceAction}|${statefulAction}`
    const current = counts.get(key) || { from: referenceAction, to: statefulAction, value: 0, interventions: 0 }
    current.value += 1
    if (candidateRow?.stateful_intervention) current.interventions += 1
    counts.set(key, current)
  })
  const orderedNodes = ['HOLD', 'ROTATE', 'BUY', 'CASH']
  const links = [...counts.values()].sort((a, b) => b.value - a.value)
  const nodes = orderedNodes.filter((name) => links.some((link) => link.from === name || link.to === name))
  if (!links.length || !nodes.length) return <EmptyVisual title="Decision policy flow will appear here." />
  const yMap = Object.fromEntries(nodes.map((name, index) => [name, 54 + index * (210 / Math.max(1, nodes.length - 1))]))
  const max = Math.max(...links.map((link) => link.value), 1)
  const total = Math.max(1, links.reduce((sum, link) => sum + link.value, 0))
  const changed = links.filter((link) => link.from !== link.to).reduce((sum, link) => sum + link.value, 0)
  const openLink = (link) => {
    const sameDecision = link.from === link.to
    setSelectedDetail({
      kicker: 'DECISION POLICY REPLAY',
      title: `${link.from} → ${link.to}`,
      description: tr('Compares the original Strategy decision with the decision after the Decision Policy is applied for the same market session.'),
      metrics: [
        { label: tr('Original Strategy decision'), value: link.from },
        { label: tr('Decision after policy'), value: link.to },
        { label: tr('Sessions'), value: number(link.value, 0) },
        { label: tr('Share of analyzed sessions'), value: percent(link.value / total, 1) },
        { label: tr('Policy interventions'), value: number(link.interventions, 0) },
        { label: tr('Decision effect'), value: sameDecision ? tr('Decision preserved') : tr('Decision changed') },
      ],
      notes: [
        { label: tr('How to read the flow'), text: tr('The left side is the original Strategy decision. The right side is the decision after the Decision Policy is applied. Thicker flows represent more sessions.') },
        { label: tr('Why it matters'), text: sameDecision ? tr('This flow shows sessions where the Decision Policy preserved the original Strategy decision.') : tr('This flow shows sessions where the Decision Policy changed the original Strategy decision. These are the interventions that must be evaluated economically.') },
      ],
    })
  }
  return <div className="strategy-research-sankey-wrap">
    <div className="strategy-research-sankey-heading"><div><strong>{tr('Decision Flow')}</strong><span>{tr('Original Strategy versus Decision Policy')}</span></div><small>{tr('Thickness represents the number of aligned market sessions. Click a flow for details.')}</small></div>
    <div className="strategy-research-sankey-side-labels"><strong>{tr('ORIGINAL STRATEGY DECISION')}</strong><strong>{tr('DECISION AFTER POLICY')}</strong></div>
    <svg viewBox="0 0 760 340" role="img" aria-label={tr('Decision policy action transitions')}>
      {links.map((link, index) => {
        const y1 = yMap[link.from]
        const y2 = yMap[link.to]
        const width = 2 + (link.value / max) * 16
        return <path key={`${link.from}-${link.to}-${index}`} d={`M 160 ${y1} C 320 ${y1}, 440 ${y2}, 600 ${y2}`} className={`strategy-research-sankey-link ${link.from === link.to ? 'preserved' : 'changed'}`} style={{ strokeWidth: width }} role="button" tabIndex="0" aria-label={`${link.from} → ${link.to} · ${link.value}`} onClick={() => openLink(link)} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') openLink(link) }} />
      })}
      {nodes.map((name) => <g key={`left-${name}`}><rect x="56" y={yMap[name] - 22} width="104" height="44" rx="12" className="strategy-research-sankey-node"/><text x="108" y={yMap[name] + 5} textAnchor="middle" className="strategy-research-sankey-text">{name}</text></g>)}
      {nodes.map((name) => <g key={`right-${name}`}><rect x="600" y={yMap[name] - 22} width="104" height="44" rx="12" className="strategy-research-sankey-node target"/><text x="652" y={yMap[name] + 5} textAnchor="middle" className="strategy-research-sankey-text">{name}</text></g>)}
    </svg>
    <div className="strategy-research-compact-metrics">
      <MetricCard label={tr('Analyzed sessions')} value={number(aligned, 0)} />
      <MetricCard label={tr('Changed decisions')} value={number(changed, 0)} />
      <MetricCard label={tr('Policy interventions')} value={number(stateful?.candidate_a?.analytics?.metrics?.interventions, 0)} />
      <MetricCard label={tr('Candidate A capital')} value={money(stateful?.candidate_a?.analytics?.metrics?.ending_capital)} />
    </div>
    <ResearchDetailDialog detail={selectedDetail} onClose={() => setSelectedDetail(null)} />
  </div>
}

function FoldHeatmap({ run, stateful }) {
  const [selectedDetail, setSelectedDetail] = useState(null)
  const folds = run?.result?.multi_horizon_fold_metrics || []
  const statefulMetrics = stateful?.candidate_a?.analytics?.metrics || {}
  if (!folds.length) return <div className="strategy-research-final-wrap"><div className="strategy-research-compact-metrics"><MetricCard label={tr('Ending capital')} value={money(statefulMetrics.ending_capital)} /><MetricCard label="CAGR" value={percent(statefulMetrics.cagr, 2)} /><MetricCard label="Sharpe" value={number(statefulMetrics.sharpe, 3)} /><MetricCard label="MaxDD" value={percent(statefulMetrics.maximum_drawdown, 2)} /></div><EmptyVisual title="Fold validation heatmap will appear here." /></div>
  const metrics = [
    { key: 'return_gap_vs_winner', label: 'Vs Winner', description: 'Return difference between this fold and the Winner reference. Positive values favor the researched result.' },
    { key: 'return_gap_vs_benchmark', label: 'Vs Benchmark', description: 'Return difference between this fold and the benchmark. Positive values favor the researched result.' },
    { key: 'capital_vs_winner_anchor_replay', label: 'Capital vs anchor', description: 'Relative capital difference against the Winner anchor replay for the same fold.' },
  ]
  return <div className="strategy-research-final-wrap">
    <div className="strategy-research-compact-metrics">
      <MetricCard label={tr('Ending capital')} value={money(statefulMetrics.ending_capital || run?.result?.multi_horizon_metrics?.shadow_capital?.ending_capital)} />
      <MetricCard label="CAGR" value={percent(statefulMetrics.cagr || run?.result?.multi_horizon_metrics?.shadow_capital?.cagr, 2)} />
      <MetricCard label="Sharpe" value={number(statefulMetrics.sharpe || run?.result?.multi_horizon_metrics?.shadow_capital?.sharpe, 3)} />
      <MetricCard label="MaxDD" value={percent(statefulMetrics.maximum_drawdown || run?.result?.multi_horizon_metrics?.shadow_capital?.max_drawdown, 2)} />
    </div>
    <div className="strategy-research-matrix fold-matrix" style={{ '--strategy-research-fold-count': Math.max(1, folds.length) }}>
      <div className="strategy-research-matrix-head"><span />{folds.map((fold) => <strong key={fold.fold_id}>F{fold.fold_id}</strong>)}</div>
      {metrics.map(({ key, label, description }) => {
        const values = folds.map((fold) => Number(fold?.[key])).filter(Number.isFinite)
        const maxAbs = Math.max(...values.map(Math.abs), 0.0001)
        return <div className="strategy-research-matrix-row" key={key}><strong>{tr(label)}</strong>{folds.map((fold) => {
          const value = Number(fold?.[key])
          const display = Number.isFinite(value) ? percent(value, 1) : '—'
          return <button type="button" className={`strategy-research-matrix-cell ${Number.isFinite(value) ? heatTone(value, maxAbs) : 'missing'}`} key={fold.fold_id} disabled={!Number.isFinite(value)} aria-label={`Fold ${fold.fold_id} · ${tr(label)} · ${display}`} onClick={() => setSelectedDetail({
            kicker: 'FINAL VALIDATION',
            title: `Fold ${fold.fold_id} · ${tr(label)}`,
            description: tr(description),
            metrics: [
              { label: tr('Value'), value: display, tone: value > 0 ? 'positive' : value < 0 ? 'negative' : '' },
              { label: tr('Fold'), value: `F${fold.fold_id}` },
              { label: tr('Period from'), value: fold?.test_start || fold?.start_date || '—' },
              { label: tr('Period to'), value: fold?.test_end || fold?.end_date || '—' },
            ],
            notes: [{ label: tr('Interpretation'), text: tr('Positive values favor the researched result. Negative values identify folds that deserve additional investigation.') }],
          })}>{display}</button>
        })}</div>
      })}
    </div>
    <ResearchDetailDialog detail={selectedDetail} onClose={() => setSelectedDetail(null)} />
  </div>
}

export function StrategyResearchVisuals({ selectedStage, stageState = {}, pipelineProgress = 0, run, analytics, risk, intervention, confidence, stateful, leadershipRegime, clustering, fragileIncumbent, emergingTrend, pipelineError = '' }) {
  const selectedStageRunning = stageState[selectedStage] === 'running'
  const temporalProgress = Number(run?.progress)
  const selectedProgress = selectedStage === 'temporal' && Number.isFinite(temporalProgress) ? temporalProgress : pipelineProgress
  const empty = (title, detail) => <EmptyVisual title={title} detail={detail} loading={selectedStageRunning} progress={selectedProgress} />
  const content = {
    reference: analytics?.equity?.length ? <MonthlyHeatmap analytics={analytics} /> : empty('Strategy replay visualization will appear here.'),
    temporal: run?.result?.horizon_metrics?.length ? <TemporalHeatmap run={run} /> : empty('Temporal horizon analysis will appear here.'),
    clustering: clustering?.id ? <RegimeClusteringPanel analysis={clustering} /> : empty('Regime Clustering will appear here.'),
    fragile_incumbent: fragileIncumbent?.id ? <FragileIncumbentPanel analysis={fragileIncumbent} /> : empty('Fragile Incumbent Research will appear here.'),
    emerging_trend: emergingTrend?.id ? <EmergingTrendPanel analysis={emergingTrend} /> : empty('Emerging Trend Research will appear here.'),
    risk: (risk?.oos?.high_risk_transitions || risk?.oos?.scored_transitions || []).length
      ? <BubbleQuadrant risk={risk} intervention={intervention} />
      : stageState.risk === 'failed' && pipelineError
        ? <EmptyVisual title={pipelineError} detail={tr('The pipeline stopped before Risk & Intervention produced a valid research dataset. Export Results includes the available partial processing data.')} />
        : empty('Risk and intervention bubbles will appear here.'),
    confidence: confidence?.outer_results?.length ? <ConfidenceTiles confidence={confidence} /> : empty('Confidence calibration tiles will appear here.'),
    stateful: stateful?.candidate_a ? <><SankeyVisual stateful={stateful} /><LeadershipRegimePanel analysis={leadershipRegime} /></> : empty('Decision policy flow will appear here.'),
    validation: stageState.validation === 'completed' || stateful?.candidate_a ? <FoldHeatmap run={run} stateful={stateful} /> : empty('Fold validation heatmap will appear here.'),
  }[selectedStage]

  return <section className="strategy-research-visual-panel">{content}</section>
}
