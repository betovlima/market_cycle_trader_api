import { useMemo, useState } from 'react'
import { createPortal } from 'react-dom'

import { getIntlLocale, tr } from '../../i18n/runtime'
import { number, percent } from '../../shared/formatters'
import './emergingTrend.css'

const FEATURE_LABELS = {
  best_score: 'Best score',
  best_vs_second_gap: 'Leader gap',
  best_vs_current_gap: 'Best vs current gap',
  best_score_zscore: 'Best score z-score',
  best_vs_second_zscore: 'Leader gap z-score',
  universe_breadth_5: 'Breadth 5d',
  universe_breadth_20: 'Breadth 20d',
  spy_return_5: 'SPY return 5d',
  spy_return_20: 'SPY return 20d',
  spy_realized_volatility_20: 'SPY volatility 20d',
  position_return_since_entry: 'Position return since entry',
  position_drawdown_from_peak: 'Position drawdown',
  score_change_from_entry: 'Score change from entry',
  entry_rank_score: 'Entry rank score',
  opportunity_gate_score: 'Opportunity gate score',
  hold_score: 'Hold score',
  incumbent_persistence_score: 'Incumbent persistence',
  incumbent_risk_health: 'Incumbent risk health',
  short_profit_consensus: 'Short profit consensus',
  long_profit_confirmation: 'Long profit confirmation',
  horizon_agreement: 'Horizon agreement',
  all_horizon_risk_safety: 'Risk safety',
  predicted_drawdown: 'Predicted drawdown',
  entry_separation_strength: 'Entry separation',
  entry_top_gap_strength: 'Entry top gap',
  trend_persistence_probability_h5: 'Trend persistence 5d',
  trend_persistence_probability_h10: 'Trend persistence 10d',
  trend_persistence_probability_h20: 'Trend persistence 20d',
}

function fullMonthLabel(value) {
  if (!/^\d{4}-\d{2}$/.test(String(value || ''))) return String(value || '—')
  const [year, month] = String(value).split('-').map(Number)
  return new Intl.DateTimeFormat(getIntlLocale(), { month: 'long', year: 'numeric', timeZone: 'UTC' }).format(new Date(Date.UTC(year, month - 1, 1)))
}

function monthNames() {
  const formatter = new Intl.DateTimeFormat(getIntlLocale(), { month: 'short', timeZone: 'UTC' })
  return Array.from({ length: 12 }, (_, index) => formatter.format(new Date(Date.UTC(2024, index, 1))).replace('.', ''))
}

function readinessLabel(value) {
  if (value === 'consistent_research_signal') return tr('Consistent research signal')
  if (value === 'promising_but_not_consistent') return tr('Promising, not yet consistent')
  if (value === 'not_stable_oos') return tr('Not stable OOS')
  return tr('Insufficient evidence')
}

function Metric({ label, value, tone = '' }) {
  return <div className="emerging-trend-metric"><span>{tr(label)}</span><strong className={tone}>{value}</strong></div>
}

function MonthDialog({ row, onClose }) {
  if (!row || typeof document === 'undefined') return null
  return createPortal(<div className="emerging-trend-dialog-backdrop" role="presentation" onMouseDown={onClose}>
    <section className="emerging-trend-dialog" role="dialog" aria-modal="true" aria-label={fullMonthLabel(row.month)} onMouseDown={(event) => event.stopPropagation()}>
      <header><div><span className="panel-kicker">{tr('EMERGING TREND / DELAYED CONFIRMATION')}</span><h3>{fullMonthLabel(row.month)}</h3></div><button type="button" onClick={onClose} aria-label={tr('Close')}>×</button></header>
      <div className="emerging-trend-dialog-grid">
        <Metric label="Emerging leader sessions" value={String(row.emerging_leader_sessions ?? '—')} />
        <Metric label="Emerging leader share" value={percent(row.emerging_leader_share, 1)} />
        <Metric label="Delayed confirmation sessions" value={String(row.delayed_confirmation_sessions ?? '—')} />
        <Metric label="Delayed confirmation share" value={percent(row.delayed_confirmation_share, 1)} />
        <Metric label="Leader forward 10d" value={row.average_best_forward_return_10 == null ? '—' : percent(row.average_best_forward_return_10, 2)} tone="positive" />
        <Metric label="Strategy forward 10d" value={row.average_strategy_forward_return_10 == null ? '—' : percent(row.average_strategy_forward_return_10, 2)} />
        <Metric label="Missed edge 10d" value={row.average_missed_edge_10 == null ? '—' : percent(row.average_missed_edge_10, 2)} tone={Number(row.average_missed_edge_10) > 0 ? 'negative' : 'positive'} />
        <Metric label="OOS probability" value={row.average_oos_probability == null ? '—' : percent(row.average_oos_probability, 1)} />
      </div>
      <article className="emerging-trend-assets"><h4>{tr('Persistent leaders in this month')}</h4><div>{(row.top_assets || []).length ? (row.top_assets || []).map((item) => <span key={item.asset}><strong>{item.asset}</strong><small>{item.sessions} {tr('sessions')}</small></span>) : <p>{tr('No persistent emerging leader was identified in this month.')}</p>}</div></article>
    </section>
  </div>, document.body)
}

function FocusCard({ row }) {
  return <article className="emerging-trend-focus-card">
    <header><strong>{fullMonthLabel(row.month)}</strong><span>{row.top_assets?.[0]?.asset || '—'}</span></header>
    <div className="emerging-trend-focus-grid">
      <Metric label="Emerging sessions" value={String(row.emerging_leader_sessions ?? '—')} />
      <Metric label="Leader forward 10d" value={row.average_best_forward_return_10 == null ? '—' : percent(row.average_best_forward_return_10, 2)} tone="positive" />
      <Metric label="Strategy forward 10d" value={row.average_strategy_forward_return_10 == null ? '—' : percent(row.average_strategy_forward_return_10, 2)} />
      <Metric label="Missed edge" value={row.average_missed_edge_10 == null ? '—' : percent(row.average_missed_edge_10, 2)} tone={Number(row.average_missed_edge_10) > 0 ? 'negative' : 'positive'} />
    </div>
  </article>
}

export function EmergingTrendPanel({ analysis }) {
  const [selectedMonth, setSelectedMonth] = useState(null)
  const monthly = analysis?.monthly || []
  const years = useMemo(() => [...new Set(monthly.map((row) => String(row.month || '').slice(0, 4)))].filter(Boolean), [monthly])
  const byMonth = useMemo(() => new Map(monthly.map((row) => [row.month, row])), [monthly])
  if (!analysis || String(analysis.status || '').toLowerCase() !== 'completed') {
    return <div className="emerging-trend-empty">{tr(analysis?.failure_message || 'Emerging Trend Research will appear after Temporal Intelligence completes.')}</div>
  }

  const summary = analysis.summary || {}
  const readiness = analysis.readiness || {}
  const folds = analysis.folds || []
  const focus = analysis.focus_months || []

  return <section className="emerging-trend-panel">
    <div className="emerging-trend-heading">
      <div><span className="panel-kicker">{tr('FAILURE FAMILY — TREND CAPTURE')}</span><h4>{tr('Emerging Trend / Delayed Confirmation')}</h4></div>
      <span className={`emerging-trend-readiness ${readiness.status || 'insufficient'}`}>{readinessLabel(readiness.status)}</span>
    </div>

    <div className="emerging-trend-summary-grid">
      <Metric label="Emerging leader sessions" value={String(summary.emerging_leader_sessions ?? '—')} />
      <Metric label="Emerging leader share" value={percent(summary.emerging_leader_share, 1)} />
      <Metric label="Delayed confirmation sessions" value={String(summary.delayed_confirmation_sessions ?? '—')} />
      <Metric label="Delayed confirmation share" value={percent(summary.delayed_confirmation_share, 1)} />
      <Metric label="Average missed edge 10d" value={summary.average_missed_edge_10 == null ? '—' : percent(summary.average_missed_edge_10, 2)} tone={Number(summary.average_missed_edge_10) > 0 ? 'negative' : 'positive'} />
      <Metric label="Policy ready" value={readiness.policy_ready ? tr('Yes') : tr('No')} tone={readiness.policy_ready ? 'positive' : 'negative'} />
    </div>

    <div className="emerging-trend-section-heading"><strong>{tr('Focus: May to July 2021')}</strong><span>{tr('May tests under-capture, June tests confirmation, July is the mature-trend reference.')}</span></div>
    <div className="emerging-trend-focus-row">{focus.map((row) => <FocusCard key={row.month} row={row} />)}</div>

    <div className="emerging-trend-section-heading"><strong>{tr('Trend capture heatmap')}</strong><span>{tr('Cells show average 10-session edge left on the table during retrospectively confirmed emerging-leader sessions. Click a month for details.')}</span></div>
    <div className="emerging-trend-calendar" role="grid" aria-label={tr('Trend capture heatmap')}>
      <div className="emerging-trend-month-head"><span />{monthNames().map((month) => <strong key={month}>{month}</strong>)}</div>
      {years.map((year) => <div className="emerging-trend-row" key={year}><strong>{year}</strong>{Array.from({ length: 12 }, (_, index) => {
        const key = `${year}-${String(index + 1).padStart(2, '0')}`
        const row = byMonth.get(key)
        const edge = Number(row?.average_missed_edge_10)
        const tone = !row || !row.emerging_leader_sessions ? 'missing' : edge > 0.03 ? 'miss-strong' : edge > 0.01 ? 'miss-soft' : edge < 0 ? 'captured' : 'neutral'
        return <button key={key} type="button" disabled={!row} className={`emerging-trend-cell ${tone}`} onClick={() => row && setSelectedMonth(row)}>{row ? <><strong>{row.emerging_leader_sessions ? percent(row.average_missed_edge_10, 1) : '—'}</strong><small>{row.emerging_leader_sessions ? `${row.emerging_leader_sessions}×` : '0'}</small></> : <span>—</span>}</button>
      })}</div>)}
    </div>

    <div className="emerging-trend-section-heading"><strong>{tr('Walk-forward evidence')}</strong></div>
    <div className="emerging-trend-folds">{folds.map((fold) => <article key={fold.fold_id}><header><span>{tr('Fold')} {fold.fold_id}</span><strong>{number(fold.auc, 3)}</strong></header><div><Metric label="Test sessions" value={String(fold.test_rows ?? '—')} /><Metric label="Positive rate" value={percent(fold.positive_rate, 1)} /><Metric label="Threshold" value={number(fold.threshold, 2)} /><Metric label="Recall" value={percent(fold.recall, 1)} /></div></article>)}</div>

    <div className="emerging-trend-card"><div className="emerging-trend-section-heading"><strong>{tr('LightGBM early-trend drivers')}</strong><span>{tr('Importance is explanatory only; the model does not alter Strategy decisions.')}</span></div><div className="emerging-trend-drivers">{(analysis.feature_importance || []).slice(0, 10).map((row) => <div key={row.feature}><span>{tr(FEATURE_LABELS[row.feature] || String(row.feature || '').replaceAll('_', ' '))}</span><strong>{number(row.mean_abs_contribution, 3)}</strong></div>)}</div></div>

    <MonthDialog row={selectedMonth} onClose={() => setSelectedMonth(null)} />
  </section>
}
