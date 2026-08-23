import { useEffect, useMemo, useState } from 'react'

import { apiFetch } from '../../api/http'
import { API } from '../../config/env'
import { tr } from '../../i18n/runtime'
import { money, number, percent } from '../../shared/formatters'

function Metric({ label, value, tone = '' }) {
  return <div className={`winner-risk-metric ${tone}`}><span>{label}</span><strong>{value}</strong></div>
}

function modeLabel(value) {
  if (value === 'confidence_calibrated_one_session') return tr('Confidence calibrated')
  return tr('Control')
}

export function WinnerTransitionConfidenceCalibrationPanel({ study, runId, processingId, interventionSearch, canRun = false, showRunButton = true, refreshToken = 0, onChange }) {
  const [search, setSearch] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const scopeKey = `${runId || ''}:${processingId || ''}:${study?.start_month || ''}:${study?.end_month || ''}`

  useEffect(() => {
    let active = true
    setSearch(null)
    setError('')
    if (!runId || !processingId || !study?.start_month || !study?.end_month) return () => { active = false }
    const query = new URLSearchParams({ processing_id: processingId, start_month: study.start_month, end_month: study.end_month })
    apiFetch(`${API}/temporal-intelligence/${encodeURIComponent(runId)}/winner-transition-confidence-calibration/latest?${query.toString()}`)
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
      const payload = await apiFetch(`${API}/temporal-intelligence/${encodeURIComponent(runId)}/winner-transition-confidence-calibration`, {
        method: 'POST',
        body: { processing_id: processingId, start_month: study.start_month, end_month: study.end_month },
      })
      setSearch(payload)
    } catch (requestError) {
      setError(tr(requestError?.message || 'Unable to run transition confidence calibration.'))
    } finally {
      setBusy(false)
    }
  }

  const model = useMemo(() => {
    const selected = search?.walk_forward_calibrated_shadow || {}
    return { selected, outerRows: search?.outer_results || [], june: search?.june_2026 || {} }
  }, [search])

  if (!interventionSearch?.id && !search?.id) return null

  return <section className="winner-transition-risk-search winner-transition-confidence-search">
    <div className="temporal-policy-heading winner-risk-heading">
      <div><span className="panel-kicker">{tr('INTERVENTION CONFIDENCE')}</span><h4>{tr('Intervention Confidence Calibration')}</h4></div>
      {canRun && showRunButton ? <button type="button" className="secondary-action compact" disabled={busy || !interventionSearch?.id} onClick={runSearch}>{tr(busy ? 'Running confidence calibration…' : 'Run confidence calibration')}</button> : null}
    </div>
    <div className="winner-risk-protocol">
      <span>{tr('Detector')} <strong>{tr('Temporal rejection')}</strong></span>
      <span>{tr('Intervention')} <strong>{tr('Defer 1 session')}</strong></span>
      <span>{tr('Confidence')} <strong>{tr('Risk margin')}</strong></span>
      <span>{tr('Selection')} <strong>{tr('Prior OOS only')}</strong></span>
    </div>
    {error ? <div className="global-inline-message error-inline">{error}</div> : null}
    {search ? <>
      <div className="winner-risk-metrics winner-risk-shadow-metrics">
        <Metric label={tr('Control capital')} value={money(model.selected?.baseline?.ending_capital)} />
        <Metric label={tr('Calibrated capital')} value={money(model.selected?.shadow?.ending_capital)} tone={Number(model.selected?.shadow?.ending_capital_delta_rate || 0) >= 0 ? 'positive' : 'negative'} />
        <Metric label={tr('Calibrated delta')} value={model.selected?.shadow?.ending_capital_delta_rate == null ? '—' : percent(model.selected.shadow.ending_capital_delta_rate, 2)} tone={Number(model.selected?.shadow?.ending_capital_delta_rate || 0) >= 0 ? 'positive' : 'negative'} />
        <Metric label={tr('Interventions')} value={number(model.selected?.interventions, 0)} />
        <Metric label={tr('Max drawdown')} value={model.selected?.shadow?.maximum_drawdown == null ? '—' : percent(model.selected.shadow.maximum_drawdown, 2)} />
        <Metric label={tr('Jun 2026 calibrated')} value={model.june?.confidence_calibrated_return == null ? '—' : percent(model.june.confidence_calibrated_return, 2)} />
      </div>
      {model.outerRows.length ? <div className="temporal-table-shell winner-risk-table-shell"><table className="temporal-table winner-risk-table winner-confidence-outer-table">
        <thead><tr><th>{tr('Test year')}</th><th>{tr('Prior OOS years')}</th><th>{tr('Selected intervention')}</th><th>{tr('Margin quantile')}</th><th>{tr('Margin threshold')}</th><th>{tr('Training capital delta')}</th><th>{tr('Test capital delta')}</th><th>{tr('Tail safe')}</th><th>{tr('Test interventions')}</th></tr></thead>
        <tbody>{model.outerRows.map((row) => <tr key={row.test_year}>
          <td>{row.test_year}</td><td>{(row.prior_oos_years || []).join(', ') || '—'}</td><td><strong>{modeLabel(row.selected_mode)}</strong></td>
          <td>{row.selected_margin_quantile == null ? '—' : `${Math.round(Number(row.selected_margin_quantile) * 100)}%`}</td>
          <td>{row.selected_margin_threshold == null ? '—' : Number(row.selected_margin_threshold).toFixed(4)}</td>
          <td className={Number(row.selected_candidate?.ending_capital_delta_rate || 0) < 0 ? 'negative' : 'positive'}>{row.selected_candidate?.ending_capital_delta_rate == null ? '—' : percent(row.selected_candidate.ending_capital_delta_rate, 2)}</td>
          <td className={Number(row.test_result?.ending_capital_delta_rate || 0) < 0 ? 'negative' : 'positive'}>{row.test_result?.ending_capital_delta_rate == null ? '—' : percent(row.test_result.ending_capital_delta_rate, 2)}</td>
          <td>{row.test_result?.tail_safe ? tr('Yes') : tr('No')}</td><td>{number(row.test_result?.interventions, 0)}</td>
        </tr>)}</tbody>
      </table></div> : null}
    </> : null}
  </section>
}
