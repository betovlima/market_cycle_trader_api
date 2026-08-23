import { useMemo, useState } from 'react'
import { createPortal } from 'react-dom'

import { getIntlLocale, tr } from '../../i18n/runtime'
import { number, percent } from '../../shared/formatters'
import './opportunityDrought.css'

const FEATURE_LABELS = {
  universe_breadth_5: 'Breadth 5d',
  universe_breadth_20: 'Breadth 20d',
  breadth_impulse: 'Breadth impulse',
  positive_score_share: 'Positive score share',
  best_score_zscore: 'Best score z-score',
  all_horizon_risk_safety: 'All-horizon risk safety',
  short_profit_consensus: 'Short profit consensus',
  long_profit_confirmation: 'Long profit confirmation',
  horizon_agreement: 'Horizon agreement',
}

function featureLabel(value) {
  return tr(FEATURE_LABELS[value] || String(value || '').replaceAll('_', ' '))
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
  return <div className="opportunity-drought-metric"><span>{tr(label)}</span><strong className={tone}>{value}</strong></div>
}

function Cohort({ title, data, tone }) {
  const features = data?.features || {}
  return <article className={`opportunity-drought-cohort ${tone}`}>
    <header><span>{tr(title)}</span><strong>{Number(data?.months || 0)}</strong></header>
    <div className="opportunity-drought-cohort-grid">
      <Metric label="Average return" value={percent(data?.average_return, 1)} tone={tone} />
      <Metric label="OOS drought probability" value={percent(data?.average_drought_probability, 1)} />
      <Metric label="Breadth 5d" value={percent(features.universe_breadth_5, 1)} />
      <Metric label="Breadth 20d" value={percent(features.universe_breadth_20, 1)} />
      <Metric label="Positive score share" value={percent(features.positive_score_share, 1)} />
      <Metric label="Horizon agreement" value={percent(features.horizon_agreement, 1)} />
    </div>
  </article>
}

function MonthDialog({ row, onClose }) {
  if (!row || typeof document === 'undefined') return null
  const features = row.features || {}
  const drivers = row.top_drivers || []
  return createPortal(<div className="opportunity-drought-dialog-backdrop" onMouseDown={onClose} role="presentation">
    <section className="opportunity-drought-dialog" role="dialog" aria-modal="true" aria-label={fullMonthLabel(row.month)} onMouseDown={(event) => event.stopPropagation()}>
      <header>
        <div><span className="panel-kicker">{tr('OPPORTUNITY DROUGHT')}</span><h3>{fullMonthLabel(row.month)}</h3><p>{tr('Official Strategy outcome compared with the causal opportunity signals available during the month.')}</p></div>
        <button type="button" onClick={onClose} aria-label={tr('Close')}>×</button>
      </header>
      <div className="opportunity-drought-dialog-summary">
        <Metric label="Official monthly return" value={percent(row.official_return, 2)} tone={Number(row.official_return) < 0 ? 'negative' : 'positive'} />
        <Metric label="Outcome" value={outcomeLabel(row.outcome)} />
        <Metric label="OOS drought probability" value={row.oos_drought_probability == null ? '—' : percent(row.oos_drought_probability, 1)} />
        <Metric label="Drought signal share" value={row.drought_signal_share == null ? '—' : percent(row.drought_signal_share, 1)} />
        <Metric label="Legacy no-good-opportunity share" value={percent(row.no_good_opportunity_share, 1)} />
        <Metric label="Fold" value={String(row.fold_id || '—')} />
      </div>
      <div className="opportunity-drought-dialog-columns">
        <article><h4>{tr('Opportunity conditions')}</h4><div className="opportunity-drought-dialog-feature-grid">
          {Object.entries(features).map(([key, value]) => <Metric key={key} label={FEATURE_LABELS[key] || key} value={key.includes('zscore') ? number(value, 2) : percent(value, 1)} />)}
        </div></article>
        <article><h4>{tr('LightGBM drivers')}</h4>{drivers.length ? <div className="opportunity-drought-driver-list">{drivers.map((driver) => <div key={driver.feature}><span>{featureLabel(driver.feature)}</span><strong className={Number(driver.contribution) >= 0 ? 'negative' : 'positive'}>{number(driver.contribution, 3)}</strong></div>)}</div> : <p className="opportunity-drought-muted">{tr('No OOS LightGBM explanation is available for this month.')}</p>}</article>
      </div>
    </section>
  </div>, document.body)
}

export function OpportunityDroughtPanel({ analysis }) {
  const [selectedMonth, setSelectedMonth] = useState(null)
  const monthly = analysis?.monthly || []
  const years = useMemo(() => [...new Set(monthly.map((row) => String(row.month || '').slice(0, 4)))].filter(Boolean), [monthly])
  const byMonth = useMemo(() => new Map(monthly.map((row) => [row.month, row])), [monthly])
  if (!analysis || String(analysis.status || '').toLowerCase() !== 'completed') {
    return <div className="opportunity-drought-empty">{tr(analysis?.failure_message || 'Opportunity Drought Research will appear after Temporal Intelligence completes.')}</div>
  }

  const summary = analysis.summary || {}
  const readiness = analysis.readiness || {}
  const aggregate = summary.aggregate_oos_month_metrics || {}
  const focus = analysis.focus_month || {}
  const shadow = analysis.shadow_evidence || {}
  const flagged = shadow.flagged || {}
  const unflagged = shadow.unflagged || {}

  return <section className="opportunity-drought-panel">
    <div className="opportunity-drought-heading">
      <div><span className="panel-kicker">{tr('FAILURE FAMILY 01')}</span><h4>{tr('Opportunity Drought')}</h4><p>{tr('LightGBM tests whether weak universe-level opportunity conditions consistently distinguish negative Strategy months. This study is diagnostic only and does not change the Strategy.')}</p></div>
      <span className={`opportunity-drought-readiness ${readiness.status || 'insufficient'}`}>{readinessLabel(readiness.status)}</span>
    </div>

    <div className="opportunity-drought-summary-grid">
      <Metric label="OOS month AUC" value={number(aggregate.auc, 3)} />
      <Metric label="Negative months" value={String(summary.negative_months ?? '—')} />
      <Metric label="Severe months ≤ -5%" value={String(summary.severe_negative_months ?? '—')} />
      <Metric label="Jun/2026 official return" value={focus.official_return == null ? '—' : percent(focus.official_return, 2)} tone="negative" />
      <Metric label="Jun/2026 drought probability" value={focus.oos_drought_probability == null ? '—' : percent(focus.oos_drought_probability, 1)} />
      <Metric label="Policy ready" value={readiness.policy_ready ? tr('Yes') : tr('No')} tone={readiness.policy_ready ? 'positive' : 'negative'} />
    </div>

    <div className="opportunity-drought-section-heading"><strong>{tr('Official monthly outcomes')}</strong><span>{tr('Click a cell for the month-level opportunity diagnostics and LightGBM explanation.')}</span></div>
    <div className="opportunity-drought-calendar" role="grid" aria-label={tr('Opportunity Drought monthly outcomes')}>
      <div className="opportunity-drought-month-head"><span />{monthNames().map((month) => <strong key={month}>{month}</strong>)}</div>
      {years.map((year) => <div className="opportunity-drought-row" key={year}><strong>{year}</strong>{Array.from({ length: 12 }, (_, index) => {
        const key = `${year}-${String(index + 1).padStart(2, '0')}`
        const row = byMonth.get(key)
        return <button key={key} type="button" disabled={!row} className={`opportunity-drought-cell ${row?.outcome || 'missing'}`} onClick={() => row && setSelectedMonth(row)} aria-label={row ? `${fullMonthLabel(key)} · ${percent(row.official_return, 1)}` : key}>
          {row ? <><strong>{percent(row.official_return, 0)}</strong><small>{row.oos_drought_probability == null ? '—' : `P ${percent(row.oos_drought_probability, 0)}`}</small></> : <span>—</span>}
        </button>
      })}</div>)}
    </div>

    <div className="opportunity-drought-section-heading"><strong>{tr('Positive versus negative months')}</strong><span>{tr('Every month has equal weight in the cohort comparison.')}</span></div>
    <div className="opportunity-drought-cohorts">
      <Cohort title="Positive months" data={analysis?.cohorts?.positive || {}} tone="positive" />
      <Cohort title="Negative months" data={analysis?.cohorts?.negative || {}} tone="negative" />
      <Cohort title="Severe negative months" data={analysis?.cohorts?.severe_negative || {}} tone="negative" />
    </div>

    <div className="opportunity-drought-section-heading"><strong>{tr('Walk-forward evidence')}</strong><span>{tr('Fold 1 is training history; the cards below are the scored OOS folds.')}</span></div>
    <div className="opportunity-drought-folds">{(analysis.folds || []).map((fold) => <article key={fold.fold_id}>
      <header><span>{tr('Fold')} {fold.fold_id}</span><strong>{number(fold?.monthly_metrics?.auc, 3)}</strong></header>
      <div><Metric label="Test months" value={String(fold.test_months ?? '—')} /><Metric label="Negative test months" value={String(fold.negative_test_months ?? '—')} /><Metric label="Threshold" value={number(fold.threshold, 2)} /><Metric label="Best iteration" value={String(fold.best_iteration ?? '—')} /></div>
    </article>)}</div>

    <div className="opportunity-drought-lower-grid">
      <article className="opportunity-drought-card"><h4>{tr('LightGBM opportunity drivers')}</h4><div className="opportunity-drought-driver-list">{(analysis.feature_importance || []).slice(0, 6).map((row) => <div key={row.feature}><span>{featureLabel(row.feature)}</span><strong>{number(row.mean_abs_contribution, 3)}</strong></div>)}</div></article>
      <article className="opportunity-drought-card"><h4>{tr('CASH intervention evidence')}</h4><div className="opportunity-drought-dialog-feature-grid"><Metric label="Flagged OOS sessions" value={String(shadow.flagged_sessions ?? '—')} /><Metric label="Flagged share" value={percent(shadow.flagged_share, 1)} /><Metric label="Flagged forward 5d" value={percent(flagged.average_forward_return_5, 2)} /><Metric label="Unflagged forward 5d" value={percent(unflagged.average_forward_return_5, 2)} /></div><p className="opportunity-drought-muted">{tr('No capital is changed in this version. A CASH replay is allowed only after the Opportunity Drought signal is stable OOS.')}</p></article>
    </div>
    <MonthDialog row={selectedMonth} onClose={() => setSelectedMonth(null)} />
  </section>
}
