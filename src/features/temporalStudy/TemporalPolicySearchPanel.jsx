import { useEffect, useMemo, useState } from 'react'

import { apiFetch } from '../../api/http'
import { API } from '../../config/env'
import { tr } from '../../i18n/runtime'
import { money, number, percent, shortDateTime } from '../../shared/formatters'

function metric(candidate, baseline, key, format, digits = 2) {
  const labels = {
    ending_capital: 'Ending capital',
    cagr: 'CAGR',
    sharpe: 'Sharpe',
    maximum_drawdown: 'Max Drawdown',
    capital_rotations: 'Capital rotations',
    cash_days: 'CASH days',
  }
  const candidateValue = candidate?.[key]
  const baselineValue = baseline?.[key]
  const delta = candidateValue == null || baselineValue == null ? null : Number(candidateValue) - Number(baselineValue)
  return <div className="temporal-policy-metric">
    <span>{tr(labels[key] || key)}</span>
    <strong>{candidateValue == null ? '—' : format(candidateValue, digits)}</strong>
    <small>{tr('Baseline')} {baselineValue == null ? '—' : format(baselineValue, digits)}{delta == null ? '' : ` · Δ ${delta >= 0 ? '+' : ''}${format(delta, digits)}`}</small>
  </div>
}

function dimensionLabel(name) {
  const labels = {
    timing_base_weak_threshold: 'Base weak threshold',
    timing_challenger_minimum: 'Challenger minimum',
    timing_minimum_advantage: 'Minimum advantage',
    trajectory_lookback_sessions: 'Trajectory lookback',
    trajectory_deterioration_quantile: 'Deterioration quantile',
    trajectory_min_signals: 'Minimum deteriorating signals',
    late_exit_min_challenger_advantage: 'Late-exit challenger advantage',
    late_exit_cash_guard: 'Temporary CASH guard',
  }
  return tr(labels[name] || name)
}

function roleLabel(role) {
  const labels = {
    winner_anchor_weakness: 'Winner-anchor weakness',
    challenger_confirmation: 'Challenger confirmation',
    short_horizon_advantage: 'Short-horizon advantage',
    incumbent_trajectory_window: 'Incumbent trajectory window',
    fold_relative_deterioration_threshold: 'Fold-relative deterioration',
    trajectory_agreement: 'Trajectory agreement',
    late_exit_challenger_quality: 'Late-exit challenger quality',
    temporary_cash_response: 'Temporary CASH response',
  }
  return tr(labels[role] || role || '—')
}

function stageDone(search, stage) {
  const status = search?.[stage]?.status
  if (stage !== 'controlled_comparison') return status === 'completed'
  if (status === 'completed' || status === 'rejected') return true
  return status === 'supported' && Boolean(search?.final_candidate?.settings)
}

export function TemporalPolicySearchPanel({ study, runId, processingId = null, canRun = false, showActions = true, onChange }) {
  const [search, setSearch] = useState(null)
  const [lhsTrials, setLhsTrials] = useState(24)
  const [caroTrials, setCaroTrials] = useState(12)
  const [seed, setSeed] = useState(42)
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const periodKey = `${study?.start_month || ''}:${study?.end_month || ''}:${runId || ''}`

  useEffect(() => {
    let active = true
    setSearch(null)
    setError('')
    if (!study || !runId) return () => { active = false }
    apiFetch(`${API}/temporal-intelligence/${encodeURIComponent(runId)}/policy-search/latest?start_month=${encodeURIComponent(study.start_month)}&end_month=${encodeURIComponent(study.end_month)}`)
      .then((payload) => {
        if (!active || !payload?.id) return
        setSearch(payload)
        setLhsTrials(Number(payload?.budgets?.latin_hypercube_trials || 24))
        setCaroTrials(Number(payload?.budgets?.caro_trials_per_outer_fold || 12))
        setSeed(Number(payload?.seed ?? 42))
      })
      .catch(() => { if (active) setSearch(null) })
    return () => { active = false }
  }, [periodKey])

  useEffect(() => { onChange?.(search) }, [onChange, search])

  async function runStage(stage) {
    if (!canRun || !study || !runId || busy) return
    setBusy(stage)
    setError('')
    try {
      let payload
      if (stage === 'prepare') {
        payload = await apiFetch(`${API}/temporal-intelligence/${encodeURIComponent(runId)}/policy-search`, {
          method: 'POST',
          body: {
            start_month: study.start_month,
            end_month: study.end_month,
            processing_id: processingId,
            lhs_trials: Math.max(4, Number(lhsTrials) || 24),
            caro_trials: Math.max(1, Number(caroTrials) || 12),
            seed: Math.max(0, Number(seed) || 0),
          },
        })
      } else {
        if (!search?.id) return
        payload = await apiFetch(`${API}/temporal-intelligence/${encodeURIComponent(runId)}/policy-search/${encodeURIComponent(search.id)}/${stage}`, { method: 'POST' })
      }
      setSearch(payload)
    } catch (requestError) {
      setError(tr(requestError?.message || 'Unable to execute Temporal Policy Search.'))
    } finally {
      setBusy('')
    }
  }

  if (!study) return null

  const samplingOuter = search?.sampling?.outer_folds || []
  const caroOuter = search?.caro?.outer_folds || []
  const validation = search?.validation || null
  const outerResults = validation?.outer_results || []
  const comparison = search?.controlled_comparison || null
  const dimensions = search?.search_space?.dimensions || []
  const log = search?.decision_log || []
  const prepared = Boolean(search?.id)
  const sampled = stageDone(search, 'sampling')
  const caroDone = stageDone(search, 'caro')
  const validated = stageDone(search, 'validation')
  const compared = stageDone(search, 'controlled_comparison')
  const comparisonCandidate = comparison?.candidate || {}
  const comparisonBaseline = comparison?.baseline || {}
  const decisionSupported = comparison?.decision === 'candidate_search_procedure_supported'

  return <section className="temporal-policy-search">
    <div className="temporal-policy-heading">
      <div>
        <span className="panel-kicker">{tr('TEMPORAL POLICY SEARCH')}</span>
        <h4>{tr('Search space → Latin Hypercube → CARO → nested validation → controlled comparison')}</h4>
      </div>
      {showActions ? <span>{tr('Each stage is explicit. Held-out outer folds are never used to select their candidate settings.')}</span> : null}
    </div>

    {showActions ? <>
    <div className="temporal-policy-budget">
      <label><span>{tr('Hypercube trials')}</span><input type="number" min="4" step="1" value={lhsTrials} onChange={(event) => setLhsTrials(event.target.value)} disabled={Boolean(busy)} /></label>
      <label><span>{tr('CARO trials / outer fold')}</span><input type="number" min="1" step="1" value={caroTrials} onChange={(event) => setCaroTrials(event.target.value)} disabled={Boolean(busy)} /></label>
      <label><span>{tr('Seed')}</span><input type="number" min="0" step="1" value={seed} onChange={(event) => setSeed(event.target.value)} disabled={Boolean(busy)} /></label>
      <small>{tr('Preparing a new search discards only the current search campaign result; the chart and frozen Temporal run remain unchanged.')}</small>
    </div>

    <div className="temporal-policy-steps">
      <button type="button" className={prepared ? 'done' : ''} disabled={!canRun || Boolean(busy)} onClick={() => runStage('prepare')}>{tr(busy === 'prepare' ? 'Preparing…' : '1. Define search space')}</button>
      <button type="button" className={sampled ? 'done' : ''} disabled={!canRun || Boolean(busy) || !prepared} onClick={() => runStage('sampling')}>{tr(busy === 'sampling' ? 'Sampling…' : '2. Run Hypercube')}</button>
      <button type="button" className={caroDone ? 'done' : ''} disabled={!canRun || Boolean(busy) || !sampled} onClick={() => runStage('caro')}>{tr(busy === 'caro' ? 'Running CARO…' : '3. Run CARO')}</button>
      <button type="button" className={validated ? 'done' : ''} disabled={!canRun || Boolean(busy) || !caroDone} onClick={() => runStage('validation')}>{tr(busy === 'validation' ? 'Validating…' : '4. Nested validation')}</button>
      <button type="button" className={compared ? 'done' : ''} disabled={!canRun || Boolean(busy) || !validated} onClick={() => runStage('comparison')}>{tr(busy === 'comparison' ? 'Comparing…' : '5. Controlled comparison')}</button>
    </div>
    </> : null}


    {error ? <div className="global-inline-message error-inline">{error}</div> : null}

    {prepared ? <div className="temporal-policy-card">
      <div className="temporal-policy-card-title"><strong>{tr('Search space')}</strong><span>{search.period_start} → {search.period_end} · {dimensions.length} {tr('dimensions')}</span></div>
      <div className="temporal-policy-table-wrap"><table className="temporal-policy-table"><thead><tr><th>{tr('Variable')}</th><th>{tr('Range')}</th><th>{tr('Role')}</th></tr></thead><tbody>{dimensions.map((item) => <tr key={item.name}><td>{dimensionLabel(item.name)}</td><td>{item.type === 'integer' ? `${item.min} → ${item.max}` : `${number(item.min, 2)} → ${number(item.max, 2)}`}</td><td>{roleLabel(item.role)}</td></tr>)}</tbody></table></div>
      <small className="temporal-policy-note">{tr('Known failed HOLD/defer gates are not part of this search space. The only new response being researched is temporary CASH under a fold-relative late-exit trajectory signal.')}</small>
    </div> : null}

    {sampled ? <div className="temporal-policy-card">
      <div className="temporal-policy-card-title"><strong>{tr('Latin Hypercube exploration')}</strong><span>{tr('Independent inner-fold search for every held-out outer fold')}</span></div>
      <div className="temporal-policy-table-wrap"><table className="temporal-policy-table"><thead><tr><th>{tr('Outer fold')}</th><th>{tr('Research folds')}</th><th>{tr('Candidates')}</th><th>{tr('Champion capital')}</th><th>{tr('Champion Sharpe')}</th><th>{tr('Search utility')}</th></tr></thead><tbody>{samplingOuter.map((item) => <tr key={item.outer_fold_id}><td>{item.outer_fold_id}</td><td>{(item.inner_fold_ids || []).join(', ')}</td><td>{number(item.evaluated_count, 0)}</td><td>{money(item.champion?.metrics?.ending_capital)}</td><td>{number(item.champion?.metrics?.sharpe, 3)}</td><td>{number(item.champion?.metrics?.search_utility, 4)}</td></tr>)}</tbody></table></div>
    </div> : null}

    {caroDone ? <div className="temporal-policy-card">
      <div className="temporal-policy-card-title"><strong>{tr('CARO adaptive refinement')}</strong><span>{tr('Gaussian-process refinement uses only completed inner-fold observations')}</span></div>
      <div className="temporal-policy-table-wrap"><table className="temporal-policy-table"><thead><tr><th>{tr('Outer fold')}</th><th>{tr('Adaptive trials')}</th><th>{tr('Champion capital')}</th><th>{tr('Champion Sharpe')}</th><th>{tr('Search utility')}</th><th>{tr('CASH guard')}</th></tr></thead><tbody>{caroOuter.map((item) => <tr key={item.outer_fold_id}><td>{item.outer_fold_id}</td><td>{number(item.completed_count, 0)}</td><td>{money(item.champion?.metrics?.ending_capital)}</td><td>{number(item.champion?.metrics?.sharpe, 3)}</td><td>{number(item.champion?.metrics?.search_utility, 4)}</td><td>{Number(item.champion?.settings?.late_exit_cash_guard) === 1 ? tr('Enabled') : tr('Off')}</td></tr>)}</tbody></table></div>
    </div> : null}

    {validated ? <div className="temporal-policy-card">
      <div className="temporal-policy-card-title"><strong>{tr('Nested temporal validation')}</strong><span>{validation.supported ? tr('Robustness criteria satisfied') : tr('Robustness criteria not satisfied')}</span></div>
      <div className="temporal-policy-table-wrap"><table className="temporal-policy-table"><thead><tr><th>{tr('Outer fold')}</th><th>{tr('Baseline return')}</th><th>{tr('Candidate return')}</th><th>{tr('Return delta')}</th><th>{tr('Sharpe delta')}</th><th>{tr('MaxDD delta')}</th><th>{tr('CASH guards')}</th></tr></thead><tbody>{outerResults.map((item) => <tr key={item.outer_fold_id}><td>{item.outer_fold_id}</td><td>{percent(item.baseline_outer_metrics?.strategy_return, 2)}</td><td>{percent(item.candidate_outer_metrics?.strategy_return, 2)}</td><td>{percent(item.return_delta, 2)}</td><td>{number(item.sharpe_delta, 3)}</td><td>{percent(item.maximum_drawdown_delta, 2)}</td><td>{number(item.late_exit_cash_guard_count, 0)}</td></tr>)}</tbody></table></div>
      <small className="temporal-policy-note">{tr('A fold is evaluated only with settings selected while that fold was excluded from Hypercube and CARO research.')}</small>
    </div> : null}

    {compared ? <div className={`temporal-policy-card temporal-policy-decision ${decisionSupported ? 'supported' : 'rejected'}`}>
      <div className="temporal-policy-card-title"><strong>{tr('Controlled comparison')}</strong><span>{tr(decisionSupported ? 'Search procedure supported' : 'No robust candidate')}</span></div>
      <div className="temporal-policy-metrics">
        {metric(comparisonCandidate, comparisonBaseline, 'ending_capital', money)}
        {metric(comparisonCandidate, comparisonBaseline, 'cagr', percent)}
        {metric(comparisonCandidate, comparisonBaseline, 'sharpe', number, 3)}
        {metric(comparisonCandidate, comparisonBaseline, 'maximum_drawdown', percent)}
        {metric(comparisonCandidate, comparisonBaseline, 'capital_rotations', number, 0)}
        {metric(comparisonCandidate, comparisonBaseline, 'cash_days', number, 0)}
      </div>
      <div className="temporal-policy-decision-text"><strong>{tr(decisionSupported ? 'SUPPORTED' : 'REJECTED')}</strong><span>{comparison.rejection_reasons?.length ? comparison.rejection_reasons.join(' · ') : tr('Nested outer-fold evidence passed the configured robustness criteria.')}</span></div>
    </div> : null}

    {log.length ? <div className="temporal-policy-card">
      <div className="temporal-policy-card-title"><strong>{tr('Decision log')}</strong><span>{tr('Every stage records what was accepted or rejected and why.')}</span></div>
      <div className="temporal-policy-log">{log.map((item, index) => <div key={`${item.at}-${index}`}><span>{shortDateTime(item.at)}</span><strong>{tr(item.stage)}</strong><em>{tr(item.outcome)}</em><p>{tr(item.message)}</p></div>)}</div>
    </div> : null}
  </section>
}
