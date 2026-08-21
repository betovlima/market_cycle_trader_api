import { useMemo, useState } from 'react'
import { createPortal } from 'react-dom'

import { getIntlLocale, tr } from '../../i18n/runtime'
import { money, number, percent } from '../../shared/formatters'
import { ParameterHint } from '../../shared/components/ParameterHint'

function MetricCard({ label, value, note, tone = '' }) {
  return <div className={`strategy-research-metric-card ${tone}`}><span>{label}</span><strong>{value}</strong>{note ? <small>{note}</small> : null}</div>
}

function CoffeeSpinner({ progress }) {
  return <span className="strategy-research-coffee-loader" aria-hidden="true">
    {Number.isFinite(progress) ? <span className="strategy-research-coffee-percent">{Math.max(0, Math.min(100, Math.round(progress)))}%</span> : null}
  </span>
}

function EmptyVisual({ title, detail, loading = false, progress = null }) {
  return <div className={`strategy-research-empty-visual ${loading ? 'loading' : ''}`}>
    {loading ? <CoffeeSpinner progress={progress} /> : <span className="strategy-research-empty-icon" aria-hidden="true">◇</span>}
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
        <div><span className="panel-kicker">{tr('REFERENCE REPLAY')}</span><h3>{title}</h3><p>{tr(isYear ? 'Annual return detail for the reference replay.' : 'Monthly return detail for the reference replay.')}</p></div>
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

function MonthlyHeatmap({ analytics }) {
  const [selectedDetail, setSelectedDetail] = useState(null)
  const rows = useMemo(() => monthlyReturns(analytics?.equity || []), [analytics?.equity])
  const yearMap = useMemo(() => yearlyReturns(rows), [rows])
  if (!rows.length) return <EmptyVisual title="Reference replay visualization will appear here." />
  const years = [...new Set(rows.map((item) => item.month.slice(0, 4)))]
  const byMonth = new Map(rows.map((item) => [item.month, item]))
  const maxAbs = Math.max(...rows.map((item) => Math.abs(Number(item.value || 0))), ...[...yearMap.values()].map((item) => Math.abs(Number(item.value || 0))), 0.0001)
  const months = monthNames()
  return <div className="strategy-research-calendar-heatmap-wrap">
    <div className="strategy-research-heatmap-heading">
      <div><strong>{tr('Monthly Return Heatmap')}</strong><span className="strategy-research-heatmap-help" title={tr('Each cell shows the monthly percentage change in reference equity. Color intensity represents magnitude.')}>?</span></div>
      <span>{tr('Monthly change in reference equity')}</span>
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
            return <button type="button" className={`strategy-research-heat-cell ${detail == null ? 'missing' : heatTone(detail.value, maxAbs)}`} key={key} title={detail == null ? `${months[index]} ${year}` : `${months[index]} ${year} · ${percent(detail.value, 2)}`} aria-label={detail == null ? `${months[index]} ${year}` : `${months[index]} ${year} ${percent(detail.value, 2)}`} disabled={!detail} onClick={() => detail && setSelectedDetail({ ...detail, kind: 'month', year, monthNumber: index + 1 })}><span>{detail == null ? '—' : percent(detail.value, 0)}</span></button>
          })}
          <button type="button" className={`strategy-research-heat-cell year-total ${yearSummary ? heatTone(yearSummary.value, maxAbs) : 'missing'}`} disabled={!yearSummary} title={yearSummary ? `${year} · ${percent(yearSummary.value, 2)}` : year} onClick={() => yearSummary && setSelectedDetail({ ...yearSummary, kind: 'year' })}><span>{yearSummary ? percent(yearSummary.value, 0) : '—'}</span></button>
        </div>
      })}
    </div>
    <div className="strategy-research-heatmap-footer">
      <div className="strategy-research-heatmap-legend" aria-label={tr('Heatmap color legend')}>
        <span>{tr('Higher loss')}</span><i className="negative strong"/><i className="negative soft"/><span>{tr('Near zero')}</span><i className="positive soft"/><i className="positive strong"/><span>{tr('Higher gain')}</span>
      </div>
      <small>{tr('Color intensity represents the magnitude of monthly return in the displayed period.')}</small>
      <small>{tr('Hover for summary · Click a month or year total for detailed analysis')}</small>
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
      return <div className="strategy-research-matrix-row" key={key}><strong className="strategy-research-metric-label"><span>{tr(label)}</span><ParameterHint id={`strategy-research-hint-${key}`} title={tr(label)} description={tr(description)} relationship={tr(relationship)} example={tr(example)} details={[{ label: tr('Columns'), value: tr('5d · 10d · 20d · 40d · 60d'), description: tr('Each column is a forecast horizon measured in trading sessions.') }]} /></strong>{rows.map((row) => {
        const value = horizonCellValue(row, key)
        const display = Number.isFinite(value) ? (key === 'auc' ? number(value, 3) : percent(value, 1)) : '—'
        return <button type="button" className={`strategy-research-matrix-cell ${Number.isFinite(value) ? heatTone(value, maxAbs) : 'missing'}`} key={row.horizon} title={`${row.horizon}d · ${tr(label)} · ${display} · ${tr(description)}`}>{display}</button>
      })}</div>
    })}
  </div>
}

function BubbleQuadrant({ risk, intervention }) {
  const highRiskRows = risk?.oos?.high_risk_transitions || []
  const riskRows = highRiskRows.length ? highRiskRows : (risk?.oos?.scored_transitions || [])
  const points = riskRows.slice(0, 80).map((row, index) => ({
    id: `${row?.transition_key || index}`,
    x: Number(row?.risk_score),
    y: Number(row?.rotation_value_added),
    severe: Boolean(row?.severe),
    label: `${row?.from_asset || '—'} → ${row?.to_asset || '—'}`,
  })).filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y))
  if (!points.length) return <EmptyVisual title="Risk and intervention bubbles will appear here." />
  const xValues = points.map((point) => point.x)
  const yValues = points.map((point) => point.y)
  const minX = Math.min(...xValues)
  const maxX = Math.max(...xValues)
  const minY = Math.min(...yValues, -0.01)
  const maxY = Math.max(...yValues, 0.01)
  const mapX = (value) => 38 + ((value - minX) / Math.max(1e-9, maxX - minX)) * 624
  const mapY = (value) => 250 - ((value - minY) / Math.max(1e-9, maxY - minY)) * 210
  const zeroY = mapY(0)
  const metrics = intervention?.walk_forward_selected_shadow || risk?.shadow_replay || null
  return <div className="strategy-research-bubble-wrap">
    <svg viewBox="0 0 700 285" role="img" aria-label={tr('Risk versus realized value added')}>
      <line x1="38" y1={zeroY} x2="662" y2={zeroY} className="strategy-research-axis" />
      <line x1="350" y1="30" x2="350" y2="250" className="strategy-research-axis faint" />
      <text x="44" y="22" className="strategy-research-svg-label">{tr('Realized value added')}</text>
      <text x="566" y="278" className="strategy-research-svg-label">{tr('Risk score')}</text>
      {points.map((point) => <circle key={point.id} cx={mapX(point.x)} cy={mapY(point.y)} r={point.severe ? 8 : 5} className={`strategy-research-bubble ${point.severe ? 'severe' : ''}`}><title>{point.label} · {tr('Risk')} {number(point.x, 3)} · {tr('Value added')} {percent(point.y, 2)}</title></circle>)}
    </svg>
    {metrics?.shadow ? <div className="strategy-research-compact-metrics"><MetricCard label={tr('Intervention capital')} value={money(metrics.shadow.ending_capital)} tone={Number(metrics.shadow.ending_capital_delta_rate || 0) >= 0 ? 'positive' : 'negative'} /><MetricCard label={tr('Capital delta')} value={percent(metrics.shadow.ending_capital_delta_rate, 2)} /><MetricCard label={tr('Interventions')} value={number(metrics.interventions, 0)} /></div> : null}
  </div>
}

function ConfidenceTiles({ confidence }) {
  const rows = confidence?.outer_results || []
  if (!rows.length) return <EmptyVisual title="Confidence calibration tiles will appear here." />
  const values = rows.map((row) => Number(row?.test_result?.ending_capital_delta_rate)).filter(Number.isFinite)
  const maxAbs = Math.max(...values.map(Math.abs), 0.0001)
  return <div className="strategy-research-confidence-tiles">{rows.map((row, index) => {
    const delta = Number(row?.test_result?.ending_capital_delta_rate)
    const safe = Boolean(row?.test_result?.tail_safe)
    const threshold = Number(row?.selected_margin_threshold)
    return <button type="button" className={`strategy-research-confidence-tile ${Number.isFinite(delta) ? heatTone(delta, maxAbs) : 'neutral'}`} key={`${row?.test_year || index}`} title={`${tr('Test year')} ${row?.test_year || '—'} · ${tr('Capital delta')} ${Number.isFinite(delta) ? percent(delta, 2) : '—'}`}>
      <span>{row?.test_year || '—'}</span>
      <strong>{Number.isFinite(delta) ? percent(delta, 1) : '—'}</strong>
      <small>{tr('Margin')} {Number.isFinite(threshold) ? number(threshold, 3) : '—'} · {safe ? tr('Tail safe') : tr('Tail warning')}</small>
    </button>
  })}</div>
}

function SankeyVisual({ stateful }) {
  const equity = stateful?.candidate_a?.analytics?.equity || []
  if (!equity.length) return <EmptyVisual title="Stateful transition flow will appear here." />
  const actions = equity.map((row) => String(row?.trade_action || '').toUpperCase()).filter(Boolean)
  const counts = new Map()
  for (let index = 1; index < actions.length; index += 1) {
    const key = `${actions[index - 1]}|${actions[index]}`
    counts.set(key, (counts.get(key) || 0) + 1)
  }
  const nodes = ['HOLD', 'ROTATE', 'CASH'].filter((name) => actions.includes(name))
  const links = [...counts.entries()].map(([key, value]) => {
    const [from, to] = key.split('|')
    return { from, to, value }
  }).filter((link) => nodes.includes(link.from) && nodes.includes(link.to)).sort((a, b) => b.value - a.value).slice(0, 8)
  if (!links.length) return <EmptyVisual title="Stateful transition flow will appear here." />
  const yMap = Object.fromEntries(nodes.map((name, index) => [name, 60 + index * (180 / Math.max(1, nodes.length - 1))]))
  const max = Math.max(...links.map((link) => link.value), 1)
  return <div className="strategy-research-sankey-wrap">
    <svg viewBox="0 0 760 320" role="img" aria-label={tr('Stateful action transitions')}>
      {links.map((link, index) => {
        const y1 = yMap[link.from]
        const y2 = yMap[link.to]
        const width = 2 + (link.value / max) * 16
        return <path key={`${link.from}-${link.to}-${index}`} d={`M 160 ${y1} C 320 ${y1}, 440 ${y2}, 600 ${y2}`} className="strategy-research-sankey-link" style={{ strokeWidth: width }}><title>{link.from} → {link.to} · {link.value}</title></path>
      })}
      {nodes.map((name) => <g key={`left-${name}`}><rect x="56" y={yMap[name] - 22} width="104" height="44" rx="12" className="strategy-research-sankey-node"/><text x="108" y={yMap[name] + 5} textAnchor="middle" className="strategy-research-sankey-text">{name}</text></g>)}
      {nodes.map((name) => <g key={`right-${name}`}><rect x="600" y={yMap[name] - 22} width="104" height="44" rx="12" className="strategy-research-sankey-node target"/><text x="652" y={yMap[name] + 5} textAnchor="middle" className="strategy-research-sankey-text">{name}</text></g>)}
    </svg>
    <div className="strategy-research-compact-metrics">
      <MetricCard label={tr('Candidate A capital')} value={money(stateful?.candidate_a?.analytics?.metrics?.ending_capital)} />
      <MetricCard label={tr('Interventions')} value={number(stateful?.candidate_a?.analytics?.metrics?.interventions, 0)} />
      <MetricCard label={tr('Deferred sessions')} value={number(stateful?.candidate_a?.analytics?.metrics?.deferred_sessions, 0)} />
    </div>
  </div>
}

function FoldHeatmap({ run, stateful }) {
  const folds = run?.result?.multi_horizon_fold_metrics || []
  const statefulMetrics = stateful?.candidate_a?.analytics?.metrics || {}
  if (!folds.length) return <div className="strategy-research-final-wrap"><div className="strategy-research-compact-metrics"><MetricCard label={tr('Ending capital')} value={money(statefulMetrics.ending_capital)} /><MetricCard label="CAGR" value={percent(statefulMetrics.cagr, 2)} /><MetricCard label="Sharpe" value={number(statefulMetrics.sharpe, 3)} /><MetricCard label="MaxDD" value={percent(statefulMetrics.maximum_drawdown, 2)} /></div><EmptyVisual title="Fold validation heatmap will appear here." /></div>
  const metrics = [
    ['return_gap_vs_winner', 'Vs Winner'],
    ['return_gap_vs_benchmark', 'Vs Benchmark'],
    ['capital_vs_winner_anchor_replay', 'Capital vs anchor'],
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
      {metrics.map(([key, label]) => {
        const values = folds.map((fold) => Number(fold?.[key])).filter(Number.isFinite)
        const maxAbs = Math.max(...values.map(Math.abs), 0.0001)
        return <div className="strategy-research-matrix-row" key={key}><strong>{tr(label)}</strong>{folds.map((fold) => {
          const value = Number(fold?.[key])
          return <button type="button" className={`strategy-research-matrix-cell ${Number.isFinite(value) ? heatTone(value, maxAbs) : 'missing'}`} key={fold.fold_id} title={`Fold ${fold.fold_id} · ${tr(label)} · ${Number.isFinite(value) ? percent(value, 2) : '—'}`}>{Number.isFinite(value) ? percent(value, 1) : '—'}</button>
        })}</div>
      })}
    </div>
  </div>
}

export function StrategyResearchVisuals({ selectedStage, stageState = {}, pipelineProgress = 0, run, analytics, risk, intervention, confidence, stateful, pipelineError = '' }) {
  const selectedStageRunning = stageState[selectedStage] === 'running'
  const temporalProgress = Number(run?.progress)
  const selectedProgress = selectedStage === 'temporal' && Number.isFinite(temporalProgress) ? temporalProgress : pipelineProgress
  const empty = (title, detail) => <EmptyVisual title={title} detail={detail} loading={selectedStageRunning} progress={selectedProgress} />
  const content = {
    reference: analytics?.equity?.length ? <MonthlyHeatmap analytics={analytics} /> : empty('Reference replay visualization will appear here.'),
    temporal: run?.result?.horizon_metrics?.length ? <TemporalHeatmap run={run} /> : empty('Temporal horizon analysis will appear here.'),
    risk: (risk?.oos?.high_risk_transitions || risk?.oos?.scored_transitions || []).length
      ? <BubbleQuadrant risk={risk} intervention={intervention} />
      : stageState.risk === 'failed' && pipelineError
        ? <EmptyVisual title={pipelineError} detail={tr('The pipeline stopped before Risk & Intervention produced a valid research dataset. Export Results includes the available partial processing data.')} />
        : empty('Risk and intervention bubbles will appear here.'),
    confidence: confidence?.outer_results?.length ? <ConfidenceTiles confidence={confidence} /> : empty('Confidence calibration tiles will appear here.'),
    stateful: stateful?.candidate_a ? <SankeyVisual stateful={stateful} /> : empty('Stateful transition flow will appear here.'),
    validation: stageState.validation === 'completed' || stateful?.candidate_a ? <FoldHeatmap run={run} stateful={stateful} /> : empty('Fold validation heatmap will appear here.'),
  }[selectedStage]

  return <section className="strategy-research-visual-panel">{content}</section>
}
