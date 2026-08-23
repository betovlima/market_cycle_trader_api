import { useMemo, useState } from 'react'
import { createPortal } from 'react-dom'

import { getIntlLocale, tr } from '../../i18n/runtime'
import { number, percent } from '../../shared/formatters'
import './fragileIncumbent.css'

const FEATURE_LABELS = {
  position_drawdown_from_peak: 'Position drawdown',
  incumbent_risk_health: 'Incumbent risk health',
  position_return_since_entry: 'Position return since entry',
  score_change_from_entry: 'Score change from entry',
  best_vs_second_gap: 'Leader gap',
  best_vs_current_gap: 'Best vs incumbent gap',
  all_horizon_risk_safety: 'All-horizon risk safety',
  best_score_zscore: 'Best score z-score',
  short_profit_consensus: 'Short profit consensus',
  long_profit_confirmation: 'Long profit confirmation',
  horizon_agreement: 'Horizon agreement',
  current_asset_rank: 'Incumbent rank',
  recent_rotations_10: 'Recent rotations 10d',
}

const RAW_NUMBER_FEATURES = new Set(['best_score_zscore', 'current_asset_rank', 'recent_rotations_10'])

function featureLabel(value) {
  return tr(FEATURE_LABELS[value] || String(value || '').replaceAll('_', ' '))
}

function featureValue(key, value) {
  return RAW_NUMBER_FEATURES.has(key) ? number(value, 2) : percent(value, 1)
}

function monthNames() {
  const formatter = new Intl.DateTimeFormat(getIntlLocale(), { month: 'short', timeZone: 'UTC' })
  return Array.from({ length: 12 }, (_, index) => formatter.format(new Date(Date.UTC(2024, index, 1))).replace('.', ''))
}

function fullMonthLabel(value) {
  if (!/^\d{4}-\d{2}$/.test(String(value || ''))) return String(value || '—')
  const [year, month] = String(value).split('-').map(Number)
  return new Intl.DateTimeFormat(getIntlLocale(), { month: 'long', year: 'numeric', timeZone: 'UTC' })
    .format(new Date(Date.UTC(year, month - 1, 1)))
}

function readinessLabel(value) {
  if (value === 'consistent_research_signal') return tr('Consistent research signal')
  if (value === 'promising_but_not_consistent') return tr('Promising, not yet consistent')
  if (value === 'not_stable_oos') return tr('Not stable OOS')
  return tr('Insufficient evidence')
}

function outcomeLabel(value) {
  if (value === 'severe_negative') return tr('Severe negative')
  if (value === 'negative') return tr('Negative')
  return tr('Positive')
}

function Metric({ label, value, tone = '' }) {
  return <div className="fragile-incumbent-metric"><span>{tr(label)}</span><strong className={tone}>{value}</strong></div>
}

function Cohort({ title, data, tone }) {
  const features = data?.features || {}
  return <article className={`fragile-incumbent-cohort ${tone}`}>
    <header><span>{tr(title)}</span><strong>{Number(data?.months || 0)}</strong></header>
    <div className="fragile-incumbent-cohort-grid">
      <Metric label="Average return" value={percent(data?.average_return, 1)} tone={tone} />
      <Metric label="Fragility probability" value={data?.average_fragility_probability == null ? '—' : percent(data.average_fragility_probability, 1)} />
      <Metric label="Position drawdown" value={percent(features.position_drawdown_from_peak, 1)} />
      <Metric label="Risk health" value={percent(features.incumbent_risk_health, 1)} />
      <Metric label="Weak leader share" value={percent(data?.average_weak_leader_share, 1)} />
      <Metric label="Rotation share" value={percent(data?.average_rotation_share, 1)} />
    </div>
  </article>
}

function MonthDialog({ row, onClose }) {
  if (!row || typeof document === 'undefined') return null
  const features = row.features || {}
  const drivers = row.top_drivers || []
  return createPortal(<div className="fragile-incumbent-dialog-backdrop" onMouseDown={onClose} role="presentation">
    <section className="fragile-incumbent-dialog" role="dialog" aria-modal="true" aria-label={fullMonthLabel(row.month)} onMouseDown={(event) => event.stopPropagation()}>
      <header>
        <div><span className="panel-kicker">{tr('FRAGILE INCUMBENT / WEAK LEADER')}</span><h3>{fullMonthLabel(row.month)}</h3></div>
        <button type="button" onClick={onClose} aria-label={tr('Close')}>×</button>
      </header>
      <div className="fragile-incumbent-dialog-summary">
        <Metric label="Official monthly return" value={percent(row.official_return, 2)} tone={Number(row.official_return) < 0 ? 'negative' : 'positive'} />
        <Metric label="Outcome" value={outcomeLabel(row.outcome)} />
        <Metric label="OOS fragility probability" value={row.oos_fragility_probability == null ? '—' : percent(row.oos_fragility_probability, 1)} />
        <Metric label="Weak leader share" value={percent(row.weak_leader_share, 1)} />
        <Metric label="HOLD sessions" value={String(row.hold_sessions ?? '—')} />
        <Metric label="ROTATE sessions" value={String(row.rotation_sessions ?? '—')} />
      </div>
      <div className="fragile-incumbent-dialog-columns">
        <article><h4>{tr('Incumbent and leader conditions')}</h4><div className="fragile-incumbent-dialog-feature-grid">
          {Object.entries(features).map(([key, value]) => <Metric key={key} label={FEATURE_LABELS[key] || key} value={featureValue(key, value)} />)}
        </div></article>
        <article><h4>{tr('LightGBM drivers')}</h4>{drivers.length ? <div className="fragile-incumbent-driver-list">{drivers.map((driver) => <div key={driver.feature}><span>{featureLabel(driver.feature)}</span><strong className={Number(driver.contribution) >= 0 ? 'negative' : 'positive'}>{number(driver.contribution, 3)}</strong></div>)}</div> : <p className="fragile-incumbent-muted">{tr('No OOS LightGBM explanation is available for this month.')}</p>}</article>
      </div>
    </section>
  </div>, document.body)
}

export function FragileIncumbentPanel({ analysis }) {
  const [selectedMonth, setSelectedMonth] = useState(null)
  const monthly = analysis?.monthly || []
  const years = useMemo(() => [...new Set(monthly.map((row) => String(row.month || '').slice(0, 4)))].filter(Boolean), [monthly])
  const byMonth = useMemo(() => new Map(monthly.map((row) => [row.month, row])), [monthly])
  if (!analysis || String(analysis.status || '').toLowerCase() !== 'completed') {
    return <div className="fragile-incumbent-empty">{tr(analysis?.failure_message || 'Fragile Incumbent Research will appear after Temporal Intelligence completes.')}</div>
  }

  const summary = analysis.summary || {}
  const readiness = analysis.readiness || {}
  const folds = analysis.folds || []
  const focus = new Map((analysis.focus_months || []).map((row) => [row.month, row]))
  const december = focus.get('2022-12') || {}
  const behavior = analysis.behavior_attribution || {}
  const hold = behavior.hold || {}
  const rotate = behavior.rotate || {}

  return <section className="fragile-incumbent-panel">
    <div className="fragile-incumbent-heading">
      <div><span className="panel-kicker">{tr('FAILURE FAMILY 02')}</span><h4>{tr('Fragile Incumbent / Weak Leader')}</h4></div>
      <span className={`fragile-incumbent-readiness ${readiness.status || 'insufficient'}`}>{readinessLabel(readiness.status)}</span>
    </div>

    <div className="fragile-incumbent-summary-grid">
      <Metric label="Fold 2 OOS AUC" value={number(folds.find((row) => Number(row.fold_id) === 2)?.monthly_metrics?.auc, 3)} />
      <Metric label="Fold 3 OOS AUC" value={number(folds.find((row) => Number(row.fold_id) === 3)?.monthly_metrics?.auc, 3)} />
      <Metric label="Negative months" value={String(summary.negative_months ?? '—')} />
      <Metric label="Severe months ≤ -5%" value={String(summary.severe_negative_months ?? '—')} />
      <Metric label="Dec/2022 return" value={december.official_return == null ? '—' : percent(december.official_return, 2)} tone="negative" />
      <Metric label="Policy ready" value={readiness.policy_ready ? tr('Yes') : tr('No')} tone={readiness.policy_ready ? 'positive' : 'negative'} />
    </div>

    <div className="fragile-incumbent-section-heading"><strong>{tr('Official monthly outcomes')}</strong><span>{tr('Click a month for incumbent, leader and LightGBM details.')}</span></div>
    <div className="fragile-incumbent-calendar" role="grid" aria-label={tr('Fragile Incumbent monthly outcomes')}>
      <div className="fragile-incumbent-month-head"><span />{monthNames().map((month) => <strong key={month}>{month}</strong>)}</div>
      {years.map((year) => <div className="fragile-incumbent-row" key={year}><strong>{year}</strong>{Array.from({ length: 12 }, (_, index) => {
        const key = `${year}-${String(index + 1).padStart(2, '0')}`
        const row = byMonth.get(key)
        return <button key={key} type="button" disabled={!row} className={`fragile-incumbent-cell ${row?.outcome || 'missing'}`} onClick={() => row && setSelectedMonth(row)}>
          {row ? <><strong>{percent(row.official_return, 0)}</strong><small>{row.oos_fragility_probability == null ? '—' : `P ${percent(row.oos_fragility_probability, 0)}`}</small></> : <span>—</span>}
        </button>
      })}</div>)}
    </div>

    <div className="fragile-incumbent-section-heading"><strong>{tr('Positive versus negative months')}</strong></div>
    <div className="fragile-incumbent-cohorts">
      <Cohort title="Positive months" data={analysis?.cohorts?.positive || {}} tone="positive" />
      <Cohort title="Negative months" data={analysis?.cohorts?.negative || {}} tone="negative" />
      <Cohort title="Severe negative months" data={analysis?.cohorts?.severe_negative || {}} tone="negative" />
    </div>

    <div className="fragile-incumbent-section-heading"><strong>{tr('Walk-forward evidence')}</strong><span>{tr('Fold 1 is training history; folds 2 and 3 are scored OOS.')}</span></div>
    <div className="fragile-incumbent-folds">{folds.map((fold) => <article key={fold.fold_id}>
      <header><span>{tr('Fold')} {fold.fold_id}</span><strong>{number(fold?.monthly_metrics?.auc, 3)}</strong></header>
      <div><Metric label="Test months" value={String(fold.test_months ?? '—')} /><Metric label="Negative test months" value={String(fold.negative_test_months ?? '—')} /><Metric label="Threshold" value={number(fold.threshold, 2)} /><Metric label="Best iteration" value={String(fold.best_iteration ?? '—')} /></div>
    </article>)}</div>

    <div className="fragile-incumbent-lower-grid">
      <article className="fragile-incumbent-card"><h4>{tr('LightGBM fragility drivers')}</h4><div className="fragile-incumbent-driver-list">{(analysis.feature_importance || []).slice(0, 7).map((row) => <div key={row.feature}><span>{featureLabel(row.feature)}</span><strong>{number(row.mean_abs_contribution, 3)}</strong></div>)}</div></article>
      <article className="fragile-incumbent-card"><h4>{tr('Flagged behavior')}</h4><div className="fragile-incumbent-dialog-feature-grid"><Metric label="Flagged OOS sessions" value={String(behavior.flagged_sessions ?? '—')} /><Metric label="Flagged share" value={percent(behavior.flagged_share, 1)} /><Metric label="HOLD sessions" value={String(hold.sessions ?? '—')} /><Metric label="HOLD forward 5d" value={percent(hold.average_forward_return_5, 2)} /><Metric label="ROTATE sessions" value={String(rotate.sessions ?? '—')} /><Metric label="ROTATE forward 5d" value={percent(rotate.average_forward_return_5, 2)} /></div></article>
    </div>
    <MonthDialog row={selectedMonth} onClose={() => setSelectedMonth(null)} />
  </section>
}
