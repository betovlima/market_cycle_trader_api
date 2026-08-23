import { useMemo, useState } from 'react'
import { createPortal } from 'react-dom'

import { getIntlLocale, tr } from '../../../i18n/runtime'
import { number, percent } from '../../../shared/formatters'
import '../leadershipRegime.css'

const STATES = [
  { id: 'healthy_leader', short: 'H', label: 'Healthy Leader' },
  { id: 'weak_relative_leader', short: 'W', label: 'Weak Relative Leader' },
  { id: 'whipsaw_leadership', short: 'X', label: 'Whipsaw Leadership' },
  { id: 'no_good_opportunity', short: 'N', label: 'No Good Opportunity' },
]

function monthNames() {
  const formatter = new Intl.DateTimeFormat(getIntlLocale(), { month: 'short', timeZone: 'UTC' })
  return Array.from({ length: 12 }, (_, index) => formatter.format(new Date(Date.UTC(2024, index, 1))).replace('.', ''))
}

function fullMonthLabel(month) {
  const [year, monthNumber] = String(month || '').split('-').map(Number)
  if (!year || !monthNumber) return month || '—'
  return new Intl.DateTimeFormat(getIntlLocale(), { month: 'long', year: 'numeric', timeZone: 'UTC' })
    .format(new Date(Date.UTC(year, monthNumber - 1, 1)))
}

function stateMeta(value) {
  return STATES.find((state) => state.id === value) || STATES[0]
}

function stateShare(row, state) {
  return Number(row?.state_shares?.[state] || 0)
}

function Metric({ label, value, tone = '' }) {
  return <div className="leadership-regime-metric"><span>{label}</span><strong className={tone}>{value}</strong></div>
}

function MonthDialog({ row, onClose }) {
  if (!row || typeof document === 'undefined') return null
  const dominant = stateMeta(row.dominant_state)
  const monthlyReturn = Number(row.monthly_return)
  return createPortal(<div className="strategy-research-heatmap-dialog-backdrop" role="presentation" onMouseDown={onClose}>
    <section className="strategy-research-heatmap-dialog leadership-regime-dialog" role="dialog" aria-modal="true" aria-label={fullMonthLabel(row.month)} onMouseDown={(event) => event.stopPropagation()}>
      <header className="strategy-research-heatmap-dialog-header">
        <div><span className="panel-kicker">{tr('LEADERSHIP REGIME')}</span><h3>{fullMonthLabel(row.month)}</h3><p>{tr('Causal leadership state observed during the month. Realized return is shown only for post-hoc validation.')}</p></div>
        <button type="button" onClick={onClose} aria-label={tr('Close')}>×</button>
      </header>
      <div className="strategy-research-heatmap-dialog-metrics">
        <Metric label={tr('Monthly return')} value={Number.isFinite(monthlyReturn) ? percent(monthlyReturn, 2) : '—'} tone={monthlyReturn > 0 ? 'positive' : monthlyReturn < 0 ? 'negative' : ''} />
        <Metric label={tr('Dominant state')} value={tr(dominant.label)} />
        <Metric label={tr('Dominant share')} value={percent(Number(row.dominant_share || 0), 1)} />
        <Metric label={tr('Leadership quality')} value={`${number(row.average_quality_score, 0)}/100`} />
      </div>
      <div className="leadership-regime-dialog-state-grid">
        {STATES.map((state) => <div key={state.id} className={`leadership-regime-state-share ${state.id}`}><span>{tr(state.label)}</span><strong>{percent(stateShare(row, state.id), 1)}</strong></div>)}
      </div>
      <div className="strategy-research-heatmap-dialog-context">
        <Metric label={tr('Breadth 5d')} value={percent(Number(row.average_breadth_5), 1)} />
        <Metric label={tr('Breadth 20d')} value={percent(Number(row.average_breadth_20), 1)} />
        <Metric label={tr('Breadth impulse')} value={percent(Number(row.average_breadth_impulse), 1)} tone={Number(row.average_breadth_impulse) < 0 ? 'negative' : 'positive'} />
        <Metric label={tr('SPY volatility 20d')} value={percent(Number(row.average_volatility_20), 1)} />
        <Metric label={tr('Leader gap')} value={number(row.average_leader_gap, 3)} />
        <Metric label={tr('Position drawdown')} value={percent(Number(row.average_position_drawdown), 1)} tone={Number(row.average_position_drawdown) < 0 ? 'negative' : ''} />
        <Metric label={tr('Incumbent risk health')} value={percent(Number(row.average_risk_health), 1)} />
        <Metric label={tr('Rotation-pressure sessions')} value={number(row.rotation_pressure_sessions, 0)} />
      </div>
    </section>
  </div>, document.body)
}

function StateCard({ row }) {
  const state = stateMeta(row?.state)
  const avg5 = Number(row?.average_forward_return_5)
  const severe = Number(row?.severe_forward_5_rate)
  return <div className={`leadership-regime-state-card ${state.id}`}>
    <div><span className="leadership-regime-state-code">{state.short}</span><strong>{tr(state.label)}</strong></div>
    <dl>
      <div><dt>{tr('Sessions')}</dt><dd>{number(row?.sessions, 0)}</dd></div>
      <div><dt>{tr('Share')}</dt><dd>{percent(Number(row?.share || 0), 1)}</dd></div>
      <div><dt>{tr('Forward 5d return')}</dt><dd className={avg5 > 0 ? 'positive' : avg5 < 0 ? 'negative' : ''}>{Number.isFinite(avg5) ? percent(avg5, 2) : '—'}</dd></div>
      <div><dt>{tr('Severe 5d loss rate')}</dt><dd className={severe > 0 ? 'negative' : ''}>{Number.isFinite(severe) ? percent(severe, 1) : '—'}</dd></div>
    </dl>
    <small>{tr('Forward outcomes are post-hoc validation metrics and are never used to classify the session.')}</small>
  </div>
}

function CohortCard({ title, cohort, tone }) {
  const shares = cohort?.state_shares || {}
  const stressed = Number(shares.weak_relative_leader || 0) + Number(shares.whipsaw_leadership || 0) + Number(shares.no_good_opportunity || 0)
  return <div className={`leadership-regime-cohort ${tone}`}>
    <div><span>{tr(title)}</span><strong>{number(cohort?.months, 0)}</strong></div>
    <dl>
      <div><dt>{tr('Average monthly return')}</dt><dd>{percent(Number(cohort?.average_monthly_return), 1)}</dd></div>
      <div><dt>{tr('Healthy Leader share')}</dt><dd>{percent(Number(shares.healthy_leader || 0), 1)}</dd></div>
      <div><dt>{tr('Stressed leadership share')}</dt><dd>{percent(stressed, 1)}</dd></div>
      <div><dt>{tr('Breadth impulse')}</dt><dd>{percent(Number(cohort?.average_breadth_impulse), 1)}</dd></div>
      <div><dt>{tr('Leader gap')}</dt><dd>{number(cohort?.average_leader_gap, 3)}</dd></div>
      <div><dt>{tr('Position drawdown')}</dt><dd>{percent(Number(cohort?.average_position_drawdown), 1)}</dd></div>
    </dl>
  </div>
}

export function LeadershipRegimePanel({ analysis }) {
  const [selectedMonth, setSelectedMonth] = useState(null)
  const monthly = analysis?.monthly || []
  const states = analysis?.summary?.states || []
  const byMonth = useMemo(() => new Map(monthly.map((row) => [row.month, row])), [monthly])
  const years = useMemo(() => [...new Set(monthly.map((row) => String(row.month || '').slice(0, 4)))].filter(Boolean), [monthly])
  if (!analysis || String(analysis.status || '').toLowerCase() !== 'completed' || !monthly.length) {
    return <section className="leadership-regime-panel"><div className="leadership-regime-heading"><div><span className="panel-kicker">{tr('LEADERSHIP REGIME')}</span><h4>{tr('Leadership Regime Analysis')}</h4></div></div><div className="leadership-regime-empty">{tr(analysis?.failure_message || 'Leadership regime diagnostics will appear after Decision Policy Replay completes.')}</div></section>
  }
  const months = monthNames()
  return <section className="leadership-regime-panel">
    <div className="leadership-regime-heading">
      <div><span className="panel-kicker">{tr('LEADERSHIP REGIME')}</span><h4>{tr('Leadership Regime Analysis')}</h4><p>{tr('Classifies the quality of relative leadership using only information available at decision time. Realized returns are used only afterward to validate the states.')}</p></div>
      <span className="leadership-regime-causal-badge">{tr('Causal classification')}</span>
    </div>

    <div className="leadership-regime-state-cards">
      {STATES.map((state) => <StateCard key={state.id} row={states.find((row) => row.state === state.id) || { state: state.id }} />)}
    </div>

    <div className="leadership-regime-map-heading"><strong>{tr('Leadership state map')}</strong><span>{tr('Dominant causal state and its share of sessions in each month')}</span></div>
    <div className="leadership-regime-calendar" role="grid" aria-label={tr('Leadership state map')}>
      <div className="leadership-regime-month-head"><span />{months.map((month) => <strong key={month}>{month}</strong>)}</div>
      {years.map((year) => <div className="leadership-regime-row" key={year}>
        <strong>{year}</strong>
        {Array.from({ length: 12 }, (_, index) => {
          const key = `${year}-${String(index + 1).padStart(2, '0')}`
          const row = byMonth.get(key)
          const state = row ? stateMeta(row.dominant_state) : null
          return <button type="button" key={key} disabled={!row} className={`leadership-regime-cell ${state?.id || 'missing'}`} onClick={() => row && setSelectedMonth(row)} aria-label={row ? `${fullMonthLabel(key)} · ${tr(state.label)} · ${percent(Number(row.dominant_share || 0), 0)}` : key}>
            {row ? <><span>{state.short}</span><small>{percent(Number(row.dominant_share || 0), 0)}</small></> : <span>—</span>}
          </button>
        })}
      </div>)}
    </div>
    <div className="leadership-regime-legend">
      {STATES.map((state) => <span key={state.id}><i className={state.id} />{state.short} · {tr(state.label)}</span>)}
    </div>

    <div className="leadership-regime-cohorts">
      <CohortCard title="Strong months ≥ +10%" cohort={analysis?.summary?.strong_months || {}} tone="positive" />
      <CohortCard title="Severe-loss months ≤ -10%" cohort={analysis?.summary?.severe_loss_months || {}} tone="negative" />
    </div>
    <MonthDialog row={selectedMonth} onClose={() => setSelectedMonth(null)} />
  </section>
}
