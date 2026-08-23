import { useEffect, useMemo, useState } from 'react'

import { apiFetch } from '../../api/http'
import { API } from '../../config/env'
import { tr } from '../../i18n/runtime'
import { money, number, percent } from '../../shared/formatters'
import { TransitionShadowCapitalChart } from './TransitionShadowCapitalChart'
import { MonthlyCapitalMovementHeatmap } from '../backtest/components/RotationPanel'

function Metric({ label, value, tone = '' }) {
  return <div className={`winner-risk-metric ${tone}`}><span>{label}</span><strong>{value}</strong></div>
}


function shadowEquity(replay) {
  return (((replay?.equity || {}).shadow) || []).map((row) => ({
    timestamp: row.timestamp,
    simulation_equity: row.value,
  }))
}

function shadowSeriesOption(key, label, replay) {
  const rotations = replay?.movement_heatmap?.shadow_rotations
  const equity = shadowEquity(replay)
  if (!Array.isArray(rotations) || !rotations.length || !equity.length) return null
  return {
    key,
    label,
    rotations,
    equity,
    allowDrilldown: true,
    allowAssetAnalysis: false,
  }
}

function modeLabel(value) {
  if (value === 'one_session_recheck') return tr('Defer 1 session')
  return tr('Control')
}

export function WinnerTransitionInterventionSearchPanel({ study, runId, processingId, riskSearch, confidenceSearch = null, canRun = false, showRunButton = true, refreshToken = 0, onChange }) {
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
    apiFetch(`${API}/temporal-intelligence/${encodeURIComponent(runId)}/winner-transition-intervention-search/latest?${query.toString()}`)
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
      const payload = await apiFetch(`${API}/temporal-intelligence/${encodeURIComponent(runId)}/winner-transition-intervention-search`, {
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
      setError(tr(requestError?.message || 'Unable to run Winner transition intervention search.'))
    } finally {
      setBusy(false)
    }
  }

  const model = useMemo(() => {
    const selected = search?.walk_forward_selected_shadow || {}
    const oneSession = search?.one_session_all_oos_shadow || {}
    const legacy = search?.legacy_long_shadow_reference || {}
    return {
      selected,
      oneSession,
      legacy,
      outerRows: search?.outer_results || [],
      june: search?.june_2026 || {},
    }
  }, [search])

  if (!riskSearch?.id && !search?.id) return null

  return <section className="winner-transition-risk-search winner-transition-intervention-search">
    <div className="temporal-policy-heading winner-risk-heading">
      <div>
        <span className="panel-kicker">{tr('TRANSITION INTERVENTION')}</span>
        <h4>{tr('Winner Anchor Transition Intervention Search')}</h4>
      </div>
      {canRun && showRunButton ? <button type="button" className="secondary-action compact" disabled={busy} onClick={runSearch}>{tr(busy ? 'Running intervention search…' : 'Run intervention search')}</button> : null}
    </div>

    <div className="winner-risk-protocol">
      <span>{tr('Detector')} <strong>{tr('Temporal rejection')}</strong></span>
      <span>{tr('Candidate')} <strong>{tr('Defer 1 session')}</strong></span>
      <span>{tr('Selection')} <strong>{tr('Prior OOS only')}</strong></span>
    </div>

    {error ? <div className="global-inline-message error-inline">{error}</div> : null}

    {search ? <>
      <div className="winner-risk-metrics winner-risk-shadow-metrics">
        <Metric label={tr('Control capital')} value={money(model.selected?.baseline?.ending_capital)} />
        <Metric label={tr('Walk-forward capital')} value={money(model.selected?.shadow?.ending_capital)} tone={Number(model.selected?.shadow?.ending_capital_delta_rate || 0) >= 0 ? 'positive' : 'negative'} />
        <Metric label={tr('Walk-forward delta')} value={model.selected?.shadow?.ending_capital_delta_rate == null ? '—' : percent(model.selected.shadow.ending_capital_delta_rate, 2)} tone={Number(model.selected?.shadow?.ending_capital_delta_rate || 0) >= 0 ? 'positive' : 'negative'} />
        <Metric label={tr('One-session all-OOS capital')} value={money(model.oneSession?.shadow?.ending_capital)} />
        <Metric label={tr('Long shadow reference capital')} value={money(model.legacy?.shadow?.ending_capital)} />
        <Metric label={tr('Walk-forward interventions')} value={number(model.selected?.interventions, 0)} />
      </div>

      <TransitionShadowCapitalChart result={search} confidenceResult={confidenceSearch} chartKey={`${search.id}:${confidenceSearch?.id || ''}:${processingId || ''}`} />

      {(() => {
        const series = [
          {
            key: 'control',
            label: tr('Control'),
            rotations: study?.analytics?.rotations || [],
            equity: study?.analytics?.equity || [],
            allowDrilldown: true,
            allowAssetAnalysis: true,
          },
          shadowSeriesOption('confidence_calibrated', tr('Confidence calibrated'), confidenceSearch?.walk_forward_calibrated_shadow),
          shadowSeriesOption('walk_forward', tr('Walk-forward intervention'), model.selected),
          shadowSeriesOption('one_session', tr('One-session shadow'), model.oneSession),
          shadowSeriesOption('long_shadow', tr('Long research shadow'), model.legacy),
        ].filter(Boolean)
        const preferred = series.some((item) => item.key === 'confidence_calibrated')
          ? 'confidence_calibrated'
          : (series.some((item) => item.key === 'walk_forward') ? 'walk_forward' : 'control')
        return series.length > 1 ? <MonthlyCapitalMovementHeatmap
          jobId={processingId}
          processingId={processingId}
          rotations={study?.analytics?.rotations || []}
          equity={study?.analytics?.equity || []}
          seriesOptions={series}
          defaultSeriesKey={preferred}
        /> : null
      })()}

      {model.outerRows.length ? <div className="temporal-table-shell winner-risk-table-shell">
        <table className="temporal-table winner-risk-table winner-intervention-outer-table">
          <thead><tr><th>{tr('Test year')}</th><th>{tr('Prior OOS years')}</th><th>{tr('Selected intervention')}</th><th>{tr('Training capital delta')}</th><th>{tr('Tail safe')}</th><th>{tr('Test capital delta')}</th><th>{tr('Test interventions')}</th></tr></thead>
          <tbody>{model.outerRows.map((row) => <tr key={row.test_year}>
            <td>{row.test_year}</td>
            <td>{(row.prior_oos_years || []).join(', ') || '—'}</td>
            <td><strong>{modeLabel(row.selected_mode)}</strong></td>
            <td className={Number(row.candidate?.ending_capital_delta_rate || 0) < 0 ? 'negative' : 'positive'}>{row.candidate?.ending_capital_delta_rate == null ? '—' : percent(row.candidate.ending_capital_delta_rate, 2)}</td>
            <td>{row.candidate == null ? '—' : (row.candidate.tail_safe ? tr('Yes') : tr('No'))}</td>
            <td className={Number(row.test_result?.ending_capital_delta_rate || 0) < 0 ? 'negative' : 'positive'}>{row.test_result?.ending_capital_delta_rate == null ? '—' : percent(row.test_result.ending_capital_delta_rate, 2)}</td>
            <td>{number(row.test_result?.interventions, 0)}</td>
          </tr>)}</tbody>
        </table>
      </div> : null}

      {model.june?.month ? <div className="winner-risk-metrics winner-intervention-june-metrics">
        <Metric label="Jun 2026 Control" value={model.june.baseline_return == null ? '—' : percent(model.june.baseline_return, 2)} tone="negative" />
        <Metric label={tr('Jun 2026 Walk-forward')} value={model.june.walk_forward_intervention_return == null ? '—' : percent(model.june.walk_forward_intervention_return, 2)} tone={Number(model.june.walk_forward_intervention_return || 0) >= Number(model.june.baseline_return || 0) ? 'positive' : 'negative'} />
        <Metric label={tr('Jun 2026 One-session')} value={model.june.one_session_all_oos_return == null ? '—' : percent(model.june.one_session_all_oos_return, 2)} />
        <Metric label={tr('Jun 2026 Long shadow')} value={model.june.legacy_long_shadow_return == null ? '—' : percent(model.june.legacy_long_shadow_return, 2)} />
      </div> : null}
    </> : null}
  </section>
}
