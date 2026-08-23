import { useEffect, useMemo, useState } from 'react'

import { apiFetch } from '../../api/http'
import { API } from '../../config/env'
import { tr } from '../../i18n/runtime'
import { money, number, percent } from '../../shared/formatters'
import { MonthlyCapitalMovementHeatmap } from '../backtest/components/RotationPanel'

function Metric({ label, value, tone = '' }) {
  return <div className={`winner-risk-metric ${tone}`}><span>{label}</span><strong>{value}</strong></div>
}

function candidateSeries(key, label, candidate) {
  const analytics = candidate?.analytics || {}
  if (!Array.isArray(analytics?.equity) || !analytics.equity.length) return null
  return {
    key,
    label,
    rotations: analytics.rotations || [],
    equity: analytics.equity || [],
    allowDrilldown: true,
    allowAssetAnalysis: false,
  }
}

function worstMonth(metrics) {
  const row = metrics?.worst_month
  if (!row?.month || row?.return == null) return '—'
  return `${row.month} · ${percent(row.return, 2)}`
}

function CandidateRow({ label, candidate, controlCapital, action = null }) {
  const metrics = candidate?.analytics?.metrics || {}
  const capital = Number(metrics.ending_capital || 0)
  const delta = controlCapital > 0 ? capital / controlCapital - 1 : null
  return <tr>
    <td><strong>{label}</strong></td>
    <td>{money(capital)}</td>
    <td className={Number(delta || 0) >= 0 ? 'positive' : 'negative'}>{delta == null ? '—' : percent(delta, 2)}</td>
    <td>{metrics.cagr == null ? '—' : percent(metrics.cagr, 2)}</td>
    <td>{metrics.sharpe == null ? '—' : number(metrics.sharpe, 3)}</td>
    <td>{metrics.maximum_drawdown == null ? '—' : percent(metrics.maximum_drawdown, 2)}</td>
    <td>{worstMonth(metrics)}</td>
    <td>{number(metrics.interventions, 0)}</td>
    <td>{number(metrics.deferred_sessions, 0)}</td>
    {action ? <td>{action}</td> : null}
  </tr>
}

export function WinnerTransitionStatefulReplayPanel({ study, runId, processingId, confidenceSearch, canRun = false, canMaterializeStrategy = false, showRunButton = true, refreshToken = 0, onChange }) {
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [materializingA, setMaterializingA] = useState(false)
  const [error, setError] = useState('')
  const [strategyNotice, setStrategyNotice] = useState('')
  const scopeKey = `${runId || ''}:${processingId || ''}:${study?.start_month || ''}:${study?.end_month || ''}`

  useEffect(() => {
    let active = true
    setResult(null)
    setError('')
    setStrategyNotice('')
    if (!runId || !processingId || !study?.start_month || !study?.end_month) return () => { active = false }
    const query = new URLSearchParams({ processing_id: processingId, start_month: study.start_month, end_month: study.end_month })
    apiFetch(`${API}/temporal-intelligence/${encodeURIComponent(runId)}/winner-transition-stateful-replay/latest?${query.toString()}`)
      .then((payload) => { if (active && payload?.id) setResult(payload) })
      .catch(() => { if (active) setResult(null) })
    return () => { active = false }
  }, [scopeKey, refreshToken])

  useEffect(() => { onChange?.(result) }, [onChange, result])

  async function runReplay() {
    if (!canRun || busy || !runId || !processingId || !study?.start_month || !study?.end_month) return
    setBusy(true)
    setError('')
    try {
      const payload = await apiFetch(`${API}/temporal-intelligence/${encodeURIComponent(runId)}/winner-transition-stateful-replay`, {
        method: 'POST',
        body: { processing_id: processingId, start_month: study.start_month, end_month: study.end_month },
      })
      setResult(payload)
    } catch (requestError) {
      setError(tr(requestError?.message || 'Unable to run stateful transition replay.'))
    } finally {
      setBusy(false)
    }
  }

  async function materializeCandidateA() {
    if (!canMaterializeStrategy || materializingA || !runId || !result?.id || result?.control_parity?.status !== 'passed' || !result?.candidate_a) return
    setMaterializingA(true)
    setError('')
    setStrategyNotice('')
    try {
      const response = await apiFetch(`${API}/temporal-intelligence/${encodeURIComponent(runId)}/winner-transition-stateful-replay/${encodeURIComponent(result.id)}/candidate-a/strategy`, { method: 'POST' })
      const strategy = response?.strategy || null
      if (strategy) {
        setResult((current) => current ? { ...current, candidate_a_materialized_strategy_id: strategy.id, candidate_a_materialized_strategy_name: strategy.name } : current)
        setStrategyNotice(tr(response?.created ? 'Candidate A Strategy created in Strategy catalog.' : 'Candidate A Strategy already exists in Strategy catalog.'))
      }
    } catch (requestError) {
      setError(tr(requestError?.message || 'Unable to create Candidate A Strategy.'))
    } finally {
      setMaterializingA(false)
    }
  }

  const model = useMemo(() => {
    const control = result?.control_replay?.analytics || {}
    const a = result?.candidate_a || {}
    const b = result?.candidate_b || {}
    return { control, a, b, drift: result?.control_replay?.replay_vs_processing_delta_rate, parity: result?.control_parity || null }
  }, [result])

  if (!confidenceSearch?.id && !result?.id) return null

  const controlCapital = Number(model.control?.metrics?.ending_capital || study?.analytics?.metrics?.ending_capital || 0)

  return <section className="winner-transition-risk-search winner-transition-stateful-search">
    <div className="temporal-policy-heading winner-risk-heading">
      <div><span className="panel-kicker">{tr('STATEFUL REPLAY')}</span><h4>{tr('Confidence-Calibrated Stateful Transition Replay')}</h4></div>
      {canRun && showRunButton ? <button type="button" className="secondary-action compact" disabled={busy || !confidenceSearch?.id} onClick={runReplay}>{tr(busy ? 'Running stateful replay…' : 'Run stateful replay')}</button> : null}
    </div>

    <div className="winner-risk-protocol">
      <span>{tr('Candidate A')} <strong>{tr('One-session stateful')}</strong></span>
      <span>{tr('Candidate B')} <strong>{tr('Adaptive long stateful')}</strong></span>
      <span>{tr('Re-evaluation')} <strong>{tr('Every session')}</strong></span>
      <span>{tr('Future Control path')} <strong>{tr('Not used')}</strong></span>
    </div>

    {error ? <div className="global-inline-message error-inline">{error}</div> : null}
    {strategyNotice ? <div className="global-inline-message success-inline">{strategyNotice}</div> : null}

    {result ? <>
      <div className="winner-risk-metrics winner-risk-shadow-metrics">
        <Metric label={tr('Stateful Control')} value={money(controlCapital)} />
        <Metric label={tr('Candidate A capital')} value={money(model.a?.analytics?.metrics?.ending_capital)} tone={Number(model.a?.analytics?.metrics?.ending_capital || 0) >= controlCapital ? 'positive' : 'negative'} />
        <Metric label={tr('Candidate B capital')} value={money(model.b?.analytics?.metrics?.ending_capital)} tone={Number(model.b?.analytics?.metrics?.ending_capital || 0) >= controlCapital ? 'positive' : 'negative'} />
        <Metric label={tr('Control parity')} value={model.parity?.status ? tr(String(model.parity.status).toUpperCase()) : '—'} tone={model.parity?.status === 'passed' ? 'positive' : 'negative'} />
        <Metric label={tr('Control replay drift')} value={model.drift == null ? '—' : percent(model.drift, 3)} tone={Math.abs(Number(model.drift || 0)) <= 0.000001 ? 'positive' : 'negative'} />
      </div>

      {model.parity?.status === 'passed' ? <div className="temporal-table-shell winner-risk-table-shell"><table className="temporal-table winner-risk-table">
        <thead><tr><th>{tr('Replay')}</th><th>{tr('Ending capital')}</th><th>{tr('Delta vs control')}</th><th>CAGR</th><th>Sharpe</th><th>MaxDD</th><th>{tr('Worst month')}</th><th>{tr('Interventions')}</th><th>{tr('Deferred sessions')}</th>{canMaterializeStrategy ? <th>{tr('Strategy')}</th> : null}</tr></thead>
        <tbody>
          <CandidateRow label={tr('Candidate A — Conservative Stateful')} candidate={model.a} controlCapital={controlCapital} action={canMaterializeStrategy ? <button type="button" className="secondary-action compact" onClick={materializeCandidateA} disabled={materializingA || Boolean(result?.candidate_a_materialized_strategy_id)}>{tr(materializingA ? 'Creating Strategy…' : result?.candidate_a_materialized_strategy_id ? 'Strategy created' : 'Create Strategy')}</button> : null} />
          <CandidateRow label={tr('Candidate B — Adaptive Long Stateful')} candidate={model.b} controlCapital={controlCapital} action={canMaterializeStrategy ? <span>—</span> : null} />
        </tbody>
      </table></div> : <div className="global-inline-message error-inline">{tr('Stateful candidates are blocked until Control parity passes.')}</div>}

      {model.parity?.status === 'passed' ? (() => {
        const series = [
          {
            key: 'control',
            label: tr('Control'),
            rotations: study?.analytics?.rotations || [],
            equity: study?.analytics?.equity || [],
            allowDrilldown: true,
            allowAssetAnalysis: true,
          },
          candidateSeries('stateful_a', tr('Candidate A — Conservative Stateful'), model.a),
          candidateSeries('stateful_b', tr('Candidate B — Adaptive Long Stateful'), model.b),
        ].filter(Boolean)
        return series.length > 1 ? <MonthlyCapitalMovementHeatmap
          jobId={processingId}
          processingId={processingId}
          rotations={study?.analytics?.rotations || []}
          equity={study?.analytics?.equity || []}
          seriesOptions={series}
          defaultSeriesKey={series.some((item) => item.key === 'stateful_b') ? 'stateful_b' : 'stateful_a'}
        /> : null
      })() : null}
    </> : null}
  </section>
}
