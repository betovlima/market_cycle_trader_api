import { useEffect, useMemo, useState } from 'react'

import { apiFetch } from '../../api/http'
import { API } from '../../config/env'
import { tr } from '../../i18n/runtime'
import { money, number, percent } from '../../shared/formatters'

function familyLabel(value) {
  const labels = {
    temporal_rejection: 'Temporal rejection',
    fragile_leader: 'Fragile leader',
    combined: 'Combined',
  }
  return tr(labels[value] || value || '—')
}

function Metric({ label, value, tone = '' }) {
  return <div className={`winner-risk-metric ${tone}`}><span>{label}</span><strong>{value}</strong></div>
}

export function WinnerTransitionRiskSearchPanel({ study, runId, processingId, canRun = false, showRunButton = true, refreshToken = 0, onChange }) {
  const [search, setSearch] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const scopeKey = `${runId || ''}:${processingId || ''}:${study?.start_month || ''}:${study?.end_month || ''}`

  useEffect(() => {
    let active = true
    setSearch(null)
    setError('')
    if (!runId || !processingId || !study?.start_month || !study?.end_month) return () => { active = false }
    const query = new URLSearchParams({
      processing_id: processingId,
      start_month: study.start_month,
      end_month: study.end_month,
    })
    apiFetch(`${API}/temporal-intelligence/${encodeURIComponent(runId)}/winner-transition-risk-search/latest?${query.toString()}`)
      .then((payload) => { if (active && payload?.id) setSearch(payload) })
      .catch(() => { if (active) setSearch(null) })
    return () => { active = false }
  }, [scopeKey, refreshToken])

  useEffect(() => { onChange?.(search) }, [onChange, search])

  async function runSearch() {
    if (!canRun || busy || !runId || !processingId || !study?.start_month || !study?.end_month) return
    setBusy(true)
    setError('')
    try {
      const payload = await apiFetch(`${API}/temporal-intelligence/${encodeURIComponent(runId)}/winner-transition-risk-search`, {
        method: 'POST',
        body: {
          processing_id: processingId,
          start_month: study.start_month,
          end_month: study.end_month,
          seed: 42,
        },
      })
      setSearch(payload)
    } catch (requestError) {
      setError(tr(requestError?.message || 'Unable to run Winner transition risk search.'))
    } finally {
      setBusy(false)
    }
  }

  const model = useMemo(() => {
    const metrics = search?.oos?.metrics || {}
    const shadow = search?.shadow_replay || {}
    const familyRows = search?.family_comparison || []
    const outerRows = search?.outer_results || []
    const riskRows = search?.oos?.high_risk_transitions || []
    const baselineMonths = shadow?.monthly_returns?.baseline || []
    const shadowMonths = shadow?.monthly_returns?.shadow || []
    const baselineMonthMap = new Map(baselineMonths.map((row) => [row.month, row]))
    const shadowMonthMap = new Map(shadowMonths.map((row) => [row.month, row]))
    const worstBaseline = [...baselineMonths].sort((left, right) => Number(left.return ?? 0) - Number(right.return ?? 0)).slice(0, 6)
    const worstShadow = [...shadowMonths].sort((left, right) => Number(left.return ?? 0) - Number(right.return ?? 0)).slice(0, 6)
    const monthKeys = [...new Set([...worstBaseline, ...worstShadow].map((row) => row.month))]
    const monthRows = monthKeys
      .map((month) => ({
        month,
        baselineReturn: baselineMonthMap.get(month)?.return ?? null,
        shadowReturn: shadowMonthMap.get(month)?.return ?? null,
      }))
      .map((row) => ({ ...row, delta: row.shadowReturn == null || row.baselineReturn == null ? null : Number(row.shadowReturn) - Number(row.baselineReturn) }))
      .sort((left, right) => Math.min(Number(left.baselineReturn ?? 0), Number(left.shadowReturn ?? 0)) - Math.min(Number(right.baselineReturn ?? 0), Number(right.shadowReturn ?? 0)))
      .slice(0, 10)
    return { metrics, shadow, familyRows, outerRows, riskRows, monthRows }
  }, [search])

  if (!study?.winner_transition_attribution?.count) return null

  return <section className="winner-transition-risk-search">
    <div className="temporal-policy-heading winner-risk-heading">
      <div>
        <span className="panel-kicker">{tr('WINNER TRANSITION RISK')}</span>
        <h4>{tr('Winner Anchor Transition Risk Search')}</h4>
      </div>
      {canRun && showRunButton ? <button type="button" className="secondary-action compact" disabled={busy} onClick={runSearch}>{tr(busy ? 'Running risk search…' : 'Run risk search')}</button> : null}
    </div>

    <div className="winner-risk-protocol">
      <span>{tr('Target')} <strong>≤ -5.00%</strong></span>
      <span>{tr('Validation')} <strong>{tr('Expanding chronological')}</strong></span>
      <span>{tr('Intervention')} <strong>{tr('Keep incumbent until next transition')}</strong></span>
    </div>

    {error ? <div className="global-inline-message error-inline">{error}</div> : null}

    {search ? <>
      <div className="winner-risk-metrics">
        <Metric label={tr('OOS transitions')} value={number(model.metrics.count, 0)} />
        <Metric label={tr('Severe transitions')} value={number(model.metrics.severe_count, 0)} tone="negative" />
        <Metric label={tr('Severe captured')} value={number(model.metrics.captured_severe_count, 0)} />
        <Metric label={tr('Recall')} value={model.metrics.recall == null ? '—' : percent(model.metrics.recall, 1)} />
        <Metric label={tr('Precision')} value={model.metrics.precision == null ? '—' : percent(model.metrics.precision, 1)} />
        <Metric label="AUC" value={model.metrics.auc == null ? '—' : number(model.metrics.auc, 3)} />
        <Metric label={tr('Alert rate')} value={model.metrics.alert_rate == null ? '—' : percent(model.metrics.alert_rate, 1)} />
        <Metric label={tr('Value-added separation')} value={model.metrics.rotation_value_added_separation == null ? '—' : percent(model.metrics.rotation_value_added_separation, 2)} />
      </div>

      {model.shadow?.baseline && model.shadow?.shadow ? <div className="winner-risk-metrics winner-risk-shadow-metrics">
        <Metric label={tr('Baseline capital')} value={money(model.shadow.baseline.ending_capital)} />
        <Metric label={tr('Shadow capital')} value={money(model.shadow.shadow.ending_capital)} tone={Number(model.shadow.shadow.ending_capital_delta || 0) >= 0 ? 'positive' : 'negative'} />
        <Metric label={tr('Capital delta')} value={model.shadow.shadow.ending_capital_delta_rate == null ? '—' : percent(model.shadow.shadow.ending_capital_delta_rate, 2)} tone={Number(model.shadow.shadow.ending_capital_delta_rate || 0) >= 0 ? 'positive' : 'negative'} />
        <Metric label={tr('Baseline MaxDD')} value={model.shadow.baseline.maximum_drawdown == null ? '—' : percent(model.shadow.baseline.maximum_drawdown, 2)} />
        <Metric label={tr('Shadow MaxDD')} value={model.shadow.shadow.maximum_drawdown == null ? '—' : percent(model.shadow.shadow.maximum_drawdown, 2)} />
        <Metric label={tr('Shadow interventions')} value={number(model.shadow.interventions, 0)} />
      </div> : null}

      {model.monthRows.length ? <div className="temporal-table-shell winner-risk-table-shell">
        <table className="temporal-table winner-risk-table winner-risk-month-table">
          <thead><tr><th>{tr('Month')}</th><th>{tr('Baseline return')}</th><th>{tr('Shadow return')}</th><th>{tr('Return delta')}</th></tr></thead>
          <tbody>{model.monthRows.map((row) => <tr key={row.month}>
            <td><strong>{row.month}</strong></td><td className={Number(row.baselineReturn || 0) < 0 ? 'negative' : 'positive'}>{row.baselineReturn == null ? '—' : percent(row.baselineReturn, 2)}</td><td className={Number(row.shadowReturn || 0) < 0 ? 'negative' : 'positive'}>{row.shadowReturn == null ? '—' : percent(row.shadowReturn, 2)}</td><td className={Number(row.delta || 0) < 0 ? 'negative' : 'positive'}>{row.delta == null ? '—' : percent(row.delta, 2)}</td>
          </tr>)}</tbody>
        </table>
      </div> : null}

      {model.familyRows.length ? <div className="temporal-table-shell winner-risk-table-shell">
        <table className="temporal-table winner-risk-table">
          <thead><tr><th>{tr('Risk family')}</th><th>AUC</th><th>{tr('Recall')}</th><th>{tr('Precision')}</th><th>{tr('Alert rate')}</th><th>{tr('Flagged value added')}</th><th>{tr('Unflagged value added')}</th></tr></thead>
          <tbody>{model.familyRows.map((row) => <tr key={row.family}>
            <td><strong>{familyLabel(row.family)}</strong></td>
            <td>{row.metrics?.auc == null ? '—' : number(row.metrics.auc, 3)}</td>
            <td>{row.metrics?.recall == null ? '—' : percent(row.metrics.recall, 1)}</td>
            <td>{row.metrics?.precision == null ? '—' : percent(row.metrics.precision, 1)}</td>
            <td>{row.metrics?.alert_rate == null ? '—' : percent(row.metrics.alert_rate, 1)}</td>
            <td className={Number(row.metrics?.mean_rotation_value_added_flagged || 0) < 0 ? 'negative' : 'positive'}>{row.metrics?.mean_rotation_value_added_flagged == null ? '—' : percent(row.metrics.mean_rotation_value_added_flagged, 2)}</td>
            <td className={Number(row.metrics?.mean_rotation_value_added_unflagged || 0) < 0 ? 'negative' : 'positive'}>{row.metrics?.mean_rotation_value_added_unflagged == null ? '—' : percent(row.metrics.mean_rotation_value_added_unflagged, 2)}</td>
          </tr>)}</tbody>
        </table>
      </div> : null}

      {model.outerRows.length ? <div className="temporal-table-shell winner-risk-table-shell">
        <table className="temporal-table winner-risk-table winner-risk-outer-table">
          <thead><tr><th>{tr('Test year')}</th><th>{tr('Selected family')}</th><th>{tr('Risk quantile')}</th><th>{tr('Transitions')}</th><th>{tr('Severe')}</th><th>{tr('Captured')}</th><th>AUC</th><th>{tr('Alert rate')}</th></tr></thead>
          <tbody>{model.outerRows.map((row) => <tr key={row.test_year}>
            <td>{row.test_year}</td><td><strong>{familyLabel(row.selected_family)}</strong></td><td>{percent(row.risk_quantile, 0)}</td><td>{number(row.metrics?.count, 0)}</td><td>{number(row.metrics?.severe_count, 0)}</td><td>{number(row.metrics?.captured_severe_count, 0)}</td><td>{row.metrics?.auc == null ? '—' : number(row.metrics.auc, 3)}</td><td>{row.metrics?.alert_rate == null ? '—' : percent(row.metrics.alert_rate, 1)}</td>
          </tr>)}</tbody>
        </table>
      </div> : null}

      {model.riskRows.length ? <div className="temporal-table-shell winner-risk-table-shell">
        <table className="temporal-table winner-risk-table winner-risk-events-table">
          <thead><tr><th>{tr('Executed')}</th><th>{tr('Transition')}</th><th>{tr('Risk score')}</th><th>{tr('Rotation value added')}</th><th>{tr('Severe')}</th><th>{tr('Selected family')}</th></tr></thead>
          <tbody>{model.riskRows.slice(0, 15).map((row, index) => <tr key={`${row.transition_key}-${index}`}>
            <td>{String(row.execution_at || '').slice(0, 10)}</td><td><strong>{row.from_asset} → {row.to_asset}</strong></td><td>{percent(row.risk_score, 1)}</td><td className={Number(row.rotation_value_added || 0) < 0 ? 'negative' : 'positive'}>{percent(row.rotation_value_added, 2)}</td><td>{row.severe ? tr('Yes') : tr('No')}</td><td>{familyLabel(row.selected_family)}</td>
          </tr>)}</tbody>
        </table>
      </div> : null}
    </> : null}
  </section>
}
