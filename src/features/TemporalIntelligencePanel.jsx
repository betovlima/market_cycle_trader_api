import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { ApiError, apiFetch, downloadFile } from '../api/http'
import { hasCapability } from '../auth/capabilities'
import { API } from '../config/env'
import { tr } from '../i18n/runtime'
import { PlayIcon } from '../shared/components/Icons'
import { TemporalStudyPanel } from './temporalStudy/TemporalStudyPanel'
import { money, number, percent, shortDateTime } from '../shared/formatters'

function statusLabel(value) {
  const normalized = String(value || '').toLowerCase()
  if (normalized === 'completed') return tr('Completed')
  if (normalized === 'running') return tr('Running')
  if (normalized === 'queued') return tr('Queued')
  if (normalized === 'stop_requested') return tr('Stopping')
  if (normalized === 'cancelled') return tr('Stopped')
  if (normalized === 'interrupted') return tr('Interrupted')
  if (normalized === 'failed') return tr('Failed')
  return value || '—'
}

function Metric({ label, value, note, tone = '' }) {
  return (
    <div className={`temporal-metric ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      {note ? <small>{note}</small> : null}
    </div>
  )
}

function LegacyHorizonTable({ items = [], selectedHorizon, onSelect }) {
  return (
    <div className="temporal-table-shell">
      <table className="temporal-table">
        <thead>
          <tr>
            <th>{tr('Horizon')}</th><th>{tr('OOS Samples')}</th><th>{tr('Brier')}</th><th>{tr('Brier Skill')}</th>
            <th>{tr('Calibration Error')}</th><th>{tr('AUC')}</th><th>{tr('Alpha Rank Correlation')}</th>
            <th>{tr('Alpha MAE')}</th><th>{tr('Alpha MAE Skill')}</th><th>{tr('Drawdown MAE')}</th>
            <th>{tr('Drawdown MAE Skill')}</th><th>{tr('High Confidence Hit Rate')}</th><th>{tr('High Confidence Lift')}</th>
          </tr>
        </thead>
        <tbody>
          {items.map((item) => (
            <tr key={item.horizon} className={Number(selectedHorizon) === Number(item.horizon) ? 'selected' : ''} onClick={() => onSelect(Number(item.horizon))}>
              <td><strong>{item.horizon}d</strong></td><td>{number(item.samples, 0)}</td><td>{number(item.brier, 4)}</td>
              <td>{percent(item.brier_skill, 2)}</td><td>{percent(item.calibration_error, 2)}</td><td>{number(item.auc, 3)}</td>
              <td>{number(item.alpha_rank_correlation, 3)}</td><td>{percent(item.alpha_mae, 2)}</td><td>{percent(item.alpha_mae_skill, 2)}</td>
              <td>{percent(item.drawdown_mae, 2)}</td><td>{percent(item.drawdown_mae_skill, 2)}</td>
              <td>{percent(item.high_confidence_positive_rate, 2)}</td><td>{percent(item.high_confidence_lift, 2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function ConfidenceTable({ items = [] }) {
  if (!items.length) return null
  return (
    <div className="temporal-table-shell compact">
      <table className="temporal-table">
        <thead><tr><th>{tr('Probability Band')}</th><th>{tr('Samples')}</th><th>{tr('Mean Probability')}</th><th>{tr('Realized Hit Rate')}</th><th>{tr('Realized Alpha')}</th><th>{tr('Predicted Alpha')}</th><th>{tr('Realized Drawdown')}</th></tr></thead>
        <tbody>{items.map((item) => <tr key={`${item.from_probability}-${item.to_probability}`}><td>{percent(item.from_probability, 0)}–{percent(item.to_probability, 0)}</td><td>{number(item.samples, 0)}</td><td>{percent(item.mean_probability, 2)}</td><td>{percent(item.realized_positive_rate, 2)}</td><td>{percent(item.mean_realized_alpha, 2)}</td><td>{percent(item.mean_predicted_alpha, 2)}</td><td>{percent(item.mean_realized_drawdown, 2)}</td></tr>)}</tbody>
      </table>
    </div>
  )
}

function LegacyForecastTable({ items = [] }) {
  if (!items.length) return <div className="temporal-empty">{tr('No latest forecast is available for this horizon.')}</div>
  return (
    <div className="temporal-table-shell">
      <table className="temporal-table temporal-forecast-table">
        <thead><tr><th>{tr('Asset')}</th><th>{tr('Expected Alpha')}</th><th>{tr('P(Alpha > 0)')}</th><th>{tr('Expected Max Drawdown')}</th></tr></thead>
        <tbody>{items.map((item) => <tr key={`${item.horizon}-${item.symbol}`}><td><strong>{item.symbol}</strong></td><td className={Number(item.expected_alpha) >= 0 ? 'positive' : 'negative'}>{percent(item.expected_alpha, 2)}</td><td>{percent(item.probability_positive_alpha, 2)}</td><td>{percent(item.expected_max_drawdown, 2)}</td></tr>)}</tbody>
      </table>
    </div>
  )
}

export function TemporalIntelligencePanel({ capabilities = {}, onSessionExpired, tuningStrategy = null }) {
  const canStart = hasCapability(capabilities, 'temporal_intelligence.start')
  const canStop = hasCapability(capabilities, 'temporal_intelligence.stop')
  const canExport = hasCapability(capabilities, 'temporal_intelligence.export')
  const canMaterializeStrategy = hasCapability(capabilities, 'temporal_intelligence.materialize_strategy')
  const [run, setRun] = useState(null)
  const [marketContext, setMarketContext] = useState(null)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [materializingStrategy, setMaterializingStrategy] = useState(false)
  const [strategyNotice, setStrategyNotice] = useState('')
  const [error, setError] = useState('')
  const [selectedHorizon, setSelectedHorizon] = useState(null)
  const pollRef = useRef(null)

  const handleError = useCallback((requestError) => {
    if (requestError instanceof ApiError && requestError.status === 401) {
      onSessionExpired?.()
      return
    }
    if (requestError?.status === 403) return
    setError(tr(requestError?.message || 'Unable to load Temporal Intelligence.'))
  }, [onSessionExpired])

  const load = useCallback(async ({ silent = false } = {}) => {
    if (!silent) setLoading(true)
    try {
      const latest = await apiFetch(`${API}/temporal-intelligence/latest`)
      let control = null
      try {
        control = await apiFetch(`${API}/admin/strategies/control`)
      } catch (contextError) {
        if (!(contextError instanceof ApiError) || contextError.status !== 403) throw contextError
      }
      setRun(latest)
      setMarketContext(control)
      setError('')
    } catch (requestError) {
      handleError(requestError)
    } finally {
      if (!silent) setLoading(false)
    }
  }, [handleError])

  useEffect(() => { load() }, [load])

  const active = ['queued', 'running', 'stop_requested'].includes(String(run?.status || '').toLowerCase())
  useEffect(() => {
    if (!active) {
      if (pollRef.current) window.clearInterval(pollRef.current)
      pollRef.current = null
      return undefined
    }
    pollRef.current = window.setInterval(() => load({ silent: true }), 2500)
    return () => {
      if (pollRef.current) window.clearInterval(pollRef.current)
      pollRef.current = null
    }
  }, [active, load])

  const researchProcessing = useMemo(() => {
    const id = String(run?.research_processing_id || '').trim()
    if (!id) return null
    return {
      id,
      strategy_profile_id: run?.strategy_profile_id || null,
      strategy_profile_name: run?.strategy_profile_name || null,
      strategy_profile_revision: run?.strategy_profile_revision ?? null,
      strategy_configuration_hash: run?.strategy_configuration_hash || null,
      processing_kind: run?.research_processing_kind || null,
      processing_label: run?.research_processing_label || null,
      created_at: run?.created_at || null,
      finished_at: run?.finished_at || null,
    }
  }, [run?.created_at, run?.finished_at, run?.research_processing_id, run?.research_processing_kind, run?.research_processing_label, run?.strategy_configuration_hash, run?.strategy_profile_id, run?.strategy_profile_name, run?.strategy_profile_revision])

  const result = run?.result || null
  const horizonMetrics = result?.horizon_metrics || []
  const trendCapturePolicy = ['temporal_decision_intelligence_v5_trend_capture_hysteresis', 'temporal_decision_intelligence_v6_adaptive_trend_capture', 'temporal_decision_intelligence_v7_rotation_before_cash'].includes(result?.experiment)
  const winnerAnchoredPolicy = result?.experiment === 'temporal_decision_intelligence_v8_winner_anchored_timing'
  const multiHorizonPolicy = ['temporal_decision_intelligence_v4_multi_horizon', 'temporal_decision_intelligence_v5_trend_capture_hysteresis', 'temporal_decision_intelligence_v6_adaptive_trend_capture', 'temporal_decision_intelligence_v7_rotation_before_cash', 'temporal_decision_intelligence_v8_winner_anchored_timing'].includes(result?.experiment)
  const decisionExperiment = ['temporal_decision_intelligence_v1', 'temporal_decision_intelligence_v2', 'temporal_decision_intelligence_v3', 'temporal_decision_intelligence_v4_multi_horizon', 'temporal_decision_intelligence_v5_trend_capture_hysteresis', 'temporal_decision_intelligence_v6_adaptive_trend_capture', 'temporal_decision_intelligence_v7_rotation_before_cash', 'temporal_decision_intelligence_v8_winner_anchored_timing'].includes(result?.experiment)
  const capitalPolicyV2 = ['temporal_decision_intelligence_v2', 'temporal_decision_intelligence_v3', 'temporal_decision_intelligence_v4_multi_horizon', 'temporal_decision_intelligence_v5_trend_capture_hysteresis', 'temporal_decision_intelligence_v6_adaptive_trend_capture', 'temporal_decision_intelligence_v7_rotation_before_cash', 'temporal_decision_intelligence_v8_winner_anchored_timing'].includes(result?.experiment)
  const capitalPolicyV3 = ['temporal_decision_intelligence_v3', 'temporal_decision_intelligence_v4_multi_horizon', 'temporal_decision_intelligence_v5_trend_capture_hysteresis', 'temporal_decision_intelligence_v6_adaptive_trend_capture', 'temporal_decision_intelligence_v7_rotation_before_cash', 'temporal_decision_intelligence_v8_winner_anchored_timing'].includes(result?.experiment)
  useEffect(() => {
    if (!horizonMetrics.length) {
      setSelectedHorizon(null)
      return
    }
    if (!horizonMetrics.some((item) => Number(item.horizon) === Number(selectedHorizon))) {
      setSelectedHorizon(Number(horizonMetrics[0].horizon))
    }
  }, [horizonMetrics, selectedHorizon])

  const selectedMetrics = useMemo(() => horizonMetrics.find((item) => Number(item.horizon) === Number(selectedHorizon)) || null, [horizonMetrics, selectedHorizon])
  const forecasts = useMemo(() => (result?.latest_forecasts || []).filter((item) => Number(item.horizon) === Number(selectedHorizon)), [result?.latest_forecasts, selectedHorizon])
  const multiHorizonMetrics = result?.multi_horizon_metrics || null
  const multiHorizonCapital = multiHorizonMetrics?.shadow_capital || {}
  const multiHorizonForecasts = result?.multi_horizon_latest_forecasts || []
  const multiHorizonBestForecast = multiHorizonForecasts.find((item) => item.shadow_target) || null

  async function start() {
    if (!canStart || busy || active) return
    setBusy(true)
    setError('')
    try {
      const created = await apiFetch(`${API}/temporal-intelligence`, { method: 'POST' })
      setRun(created)
      await load({ silent: true })
    } catch (requestError) {
      handleError(requestError)
    } finally {
      setBusy(false)
    }
  }

  async function stop() {
    if (!canStop || busy || !active || !run?.id) return
    setBusy(true)
    setError('')
    try {
      const updated = await apiFetch(`${API}/temporal-intelligence/${encodeURIComponent(run.id)}/stop`, { method: 'POST' })
      setRun(updated)
    } catch (requestError) {
      handleError(requestError)
    } finally {
      setBusy(false)
    }
  }

  async function materializeStrategy() {
    if (!canMaterializeStrategy || materializingStrategy || !run?.id || !run?.result || run?.status !== 'completed') return
    setMaterializingStrategy(true)
    setError('')
    setStrategyNotice('')
    try {
      const response = await apiFetch(`${API}/temporal-intelligence/${encodeURIComponent(run.id)}/strategy`, { method: 'POST' })
      const strategy = response?.strategy || null
      if (strategy) {
        setRun((current) => current ? { ...current, materialized_strategy_id: strategy.id, materialized_strategy_name: strategy.name } : current)
        setStrategyNotice(tr(response?.created ? 'Temporal Strategy created in Strategy catalog.' : 'Temporal Strategy already exists in Strategy catalog.'))
      }
    } catch (requestError) {
      handleError(requestError)
    } finally {
      setMaterializingStrategy(false)
    }
  }

  async function exportResults() {
    if (!canExport || exporting || !run?.id || !run?.result) return
    setExporting(true)
    setError('')
    try {
      await downloadFile(`${API}/temporal-intelligence/${encodeURIComponent(run.id)}/export.zip`, `temporal_intelligence_${run.id}.zip`)
    } catch (requestError) {
      handleError(requestError)
    } finally {
      setExporting(false)
    }
  }

  if (loading) return <div className="temporal-loading"><span className="loading-ring" />{tr('Loading Temporal Intelligence…')}</div>

  const shadowCapital = selectedMetrics?.shadow_capital || {}
  const bestForecast = decisionExperiment ? forecasts.find((item) => item.shadow_target) : null
  const currentResearchStrategy = marketContext?.strategy_research_strategy || marketContext?.research_strategy || null
  const winnerAnchorName = run?.strategy_profile_name || currentResearchStrategy?.name || '—'
  const certifiedBacktestCutoff = run?.certified_backtest_cutoff || currentResearchStrategy?.certified_backtest_cutoff || null
  const liveMarketCutoff = marketContext?.live_market_cutoff || run?.live_market_cutoff || null
  const researchSnapshotCutoff = run?.research_snapshot_cutoff || run?.analysis_end_date || null

  return (
    <div className="temporal-intelligence-panel">
      <div className="temporal-toolbar">
        <div className="temporal-run-meta temporal-source-context">
          <span>{tr('Strategy anchor')}</span><strong title={winnerAnchorName}>{winnerAnchorName}</strong>
          {tuningStrategy ? <><i>·</i><span>{tr('Selected for tuning')}</span><strong title={tuningStrategy.name || ''}>{tuningStrategy.name || '—'}</strong></> : null}
          <i>·</i><span>{tr('Model')}</span><strong>{run?.model_label || currentResearchStrategy?.research_model?.label || currentResearchStrategy?.winner_model?.label || 'LightGBM'}</strong>
          {certifiedBacktestCutoff ? <><i>·</i><span>{tr('Certified through')}</span><strong>{certifiedBacktestCutoff}</strong></> : null}
          {liveMarketCutoff ? <><i>·</i><span>{tr('Live market data')}</span><strong>{liveMarketCutoff}</strong></> : null}
          {researchSnapshotCutoff ? <><i>·</i><span>{tr('Research snapshot')}</span><strong>{researchSnapshotCutoff}</strong></> : null}
        </div>
        <div className="temporal-actions">
          {canMaterializeStrategy && run?.status === 'completed' && run?.result ? <button type="button" className="secondary-action compact" onClick={materializeStrategy} disabled={materializingStrategy || Boolean(run?.materialized_strategy_id)}>{tr(materializingStrategy ? 'Creating Strategy…' : run?.materialized_strategy_id ? 'Strategy created' : 'Create Strategy')}</button> : null}
          {canExport && run?.result ? <button type="button" className="secondary-action compact" onClick={exportResults} disabled={exporting}>{tr(exporting ? 'Exporting…' : 'Export Results')}</button> : null}
          {canStop && active ? <button type="button" className="secondary-action compact" onClick={stop} disabled={busy}>{tr(busy ? 'Stopping…' : 'Stop')}</button> : null}
          {canStart ? <button type="button" className="primary-action compact" onClick={start} disabled={busy || active}><PlayIcon />{tr(busy ? 'Starting…' : active ? 'Running' : 'Start Temporal Intelligence')}</button> : null}
        </div>
      </div>

      {error ? <div className="global-inline-message error-inline">{error}</div> : null}
      {strategyNotice ? <div className="global-inline-message success-inline">{strategyNotice}</div> : null}
      {run?.materialized_strategy_name ? <div className="temporal-strategy-materialized"><span>{tr('Catalog Strategy')}</span><strong>{run.materialized_strategy_name}</strong></div> : null}
      {run?.failure_message ? <div className="global-inline-message error-inline">{tr(run.failure_message)}</div> : null}

      {run ? <section className="temporal-status-panel"><div className="temporal-status-line"><strong>{statusLabel(run.status)}</strong><span>{run.stage || '—'}</span><span>{number(run.progress, 0)}%</span></div><div className="temporal-progress"><span style={{ width: `${Math.max(0, Math.min(100, Number(run.progress || 0)))}%` }} /></div><div className="temporal-status-meta"><span>{tr('Run')} {run.id}</span><span>{tr('Created')} {shortDateTime(run.created_at)}</span>{run.finished_at ? <span>{tr('Finished')} {shortDateTime(run.finished_at)}</span> : null}</div></section> : <div className="temporal-empty">{tr('No Temporal Intelligence execution yet.')}</div>}

      {result ? <>
        <section className="temporal-summary-grid"><Metric label={tr('Assets')} value={number(result.asset_count, 0)} note={`${number(result.feature_count, 0)} ${tr('features')}`} /><Metric label={tr('Walk-forward Folds')} value={number(result.walk_forward_fold_count, 0)} note={`${number(result.purge_sessions, 0)} ${tr('purge sessions')}`} /><Metric label={tr('Horizons')} value={(result.horizons || []).map((item) => `${item}d`).join(' · ')} /><Metric label={tr('OOS Window')} value={`${String(result.oos_start || '').slice(0, 10)} → ${String(result.oos_end || '').slice(0, 10)}`} /></section>
        {capitalPolicyV3 && result.winner_reference ? <section className="temporal-summary-grid selected-horizon"><Metric label={tr('Winner Replay Capital')} value={money(result.winner_reference.ending_capital)} tone="positive" /><Metric label={tr('Winner Replay CAGR')} value={percent(result.winner_reference.cagr, 2)} /><Metric label={tr('Winner Replay Sharpe')} value={number(result.winner_reference.sharpe, 3)} /><Metric label={tr('Winner Replay Max Drawdown')} value={percent(result.winner_reference.max_drawdown, 2)} tone={Number(result.winner_reference.max_drawdown || 0) < 0 ? 'negative' : ''} /><Metric label={tr('Benchmark Capital')} value={money(result.winner_reference.benchmark_ending_capital)} /><Metric label={tr('Same Frozen Snapshot')} value={result.winner_reference.same_frozen_market_snapshot ? tr('Yes') : tr('No')} /></section> : null}

        {multiHorizonPolicy && multiHorizonMetrics ? <>
          <section className="temporal-section">
            <div className="temporal-section-heading"><h3>{tr('Multi-Horizon Decision Engine')}</h3><span>{tr(winnerAnchoredPolicy ? 'Winner-Anchored Temporal Timing' : result?.experiment === 'temporal_decision_intelligence_v7_rotation_before_cash' ? 'Adaptive Trend Capture + Rotation Before CASH' : result?.experiment === 'temporal_decision_intelligence_v6_adaptive_trend_capture' ? 'Adaptive Trend Capture + Risk-Conditioned Hysteresis' : trendCapturePolicy ? 'Trend Capture + Decision Hysteresis' : 'Single BUY / HOLD / SELL / CASH policy')}</span></div>
            <div className="temporal-summary-grid selected-horizon">
              <Metric label={tr('Multi-Horizon Capital')} value={money(multiHorizonCapital.ending_capital)} tone={Number(multiHorizonCapital.total_return || 0) >= 0 ? 'positive' : 'negative'} />
              <Metric label={tr('CAGR')} value={percent(multiHorizonCapital.cagr, 2)} tone={Number(multiHorizonCapital.cagr || 0) >= 0 ? 'positive' : 'negative'} />
              <Metric label={tr('Sharpe')} value={number(multiHorizonCapital.sharpe, 3)} />
              <Metric label={tr('Max Drawdown')} value={percent(multiHorizonCapital.max_drawdown, 2)} tone={Number(multiHorizonCapital.max_drawdown || 0) < 0 ? 'negative' : ''} />
              <Metric label={tr('Exposure')} value={percent(multiHorizonCapital.exposure, 2)} />
              {(trendCapturePolicy || winnerAnchoredPolicy) ? <Metric label={tr('Switches')} value={number(multiHorizonCapital.switch_count, 0)} note={`${tr('Median hold')} ${number(multiHorizonCapital.median_holding_days, 1)}d`} /> : null}
              {(trendCapturePolicy || winnerAnchoredPolicy) ? <Metric label={tr('Short Holds ≤ 2d')} value={percent(multiHorizonCapital.short_holding_ratio_2d, 1)} note={`${tr('Median CASH')} ${number(multiHorizonCapital.median_cash_days, 1)}d`} /> : null}
              {trendCapturePolicy ? <Metric label={tr('Re-entries')} value={number(multiHorizonCapital.reentry_count, 0)} note={`${number(multiHorizonCapital.next_day_reentry_count, 0)} ${tr('next-day')}`} /> : null}
              {result?.experiment === 'temporal_decision_intelligence_v7_rotation_before_cash' ? <Metric label={tr('Rotation Before CASH')} value={number(multiHorizonCapital.rotation_before_cash_count, 0)} note={`${tr('Incumbent recovery')} ${number(multiHorizonCapital.incumbent_entry_recovery_hold_count, 0)}`} /> : null}
              {result?.experiment === 'temporal_decision_intelligence_v7_rotation_before_cash' ? <Metric label={tr('Defensive CASH')} value={number(multiHorizonCapital.defensive_exit_cash_count, 0)} note={`${tr('Opportunity CASH')} ${number(multiHorizonCapital.opportunity_exit_cash_count, 0)}`} /> : null}
              {winnerAnchoredPolicy ? <Metric label={tr('Temporal Timing Overrides')} value={number(multiHorizonCapital.timing_override_count, 0)} note={`${tr('Top-2 challenger')} · ${percent(multiHorizonMetrics.capital_lift_vs_winner_anchor_replay, 2)}`} /> : null}
              {winnerAnchoredPolicy ? <Metric label={tr('Winner Anchor Replay')} value={money(multiHorizonMetrics.winner_anchor_replay?.ending_capital)} note={tr('Same open-to-open evaluation basis')} /> : null}
              <Metric label={tr('Vs Winner')} value={percent(multiHorizonMetrics.capital_vs_winner, 2)} tone={Number(multiHorizonMetrics.capital_vs_winner || 0) >= 0 ? 'positive' : 'negative'} />
              <Metric label={tr('Vs Benchmark')} value={percent(multiHorizonMetrics.capital_vs_benchmark, 2)} tone={Number(multiHorizonMetrics.capital_vs_benchmark || 0) >= 0 ? 'positive' : 'negative'} />
              <Metric label={tr('Latest Shadow Target')} value={multiHorizonBestForecast?.symbol || tr('CASH')} note={multiHorizonBestForecast ? `${tr('Opportunity Gate')} ${number(multiHorizonBestForecast.opportunity_gate_score, 4)}` : ''} />
            </div>
            <div className="temporal-run-meta temporal-multi-horizon-roles">
              <span>{tr('Entry Horizons')}</span><strong>{(multiHorizonMetrics.entry_horizons || []).map((item) => `${item}d`).join(' · ')}</strong><i>·</i>
              <span>{tr('Hold Horizons')}</span><strong>{(multiHorizonMetrics.hold_horizons || []).map((item) => `${item}d`).join(' · ')}</strong><i>·</i>
              <span>{tr('Risk Horizons')}</span><strong>{(multiHorizonMetrics.risk_horizons || []).map((item) => `${item}d`).join(' · ')}</strong>
            </div>
          </section>
        </> : null}

        {!decisionExperiment ? <section className="temporal-section"><div className="temporal-section-heading"><h3>{tr('Out-of-Sample Metrics by Horizon')}</h3></div><LegacyHorizonTable items={horizonMetrics} selectedHorizon={selectedHorizon} onSelect={setSelectedHorizon} /></section> : null}

        {selectedMetrics && decisionExperiment ? <section className="temporal-summary-grid selected-horizon"><Metric label={tr('Shadow Capital')} value={money(shadowCapital.ending_capital)} tone={Number(shadowCapital.total_return || 0) >= 0 ? 'positive' : 'negative'} /><Metric label={tr('CAGR')} value={percent(shadowCapital.cagr, 2)} tone={Number(shadowCapital.cagr || 0) >= 0 ? 'positive' : 'negative'} /><Metric label={tr('Sharpe')} value={number(shadowCapital.sharpe, 3)} /><Metric label={tr('Max Drawdown')} value={percent(shadowCapital.max_drawdown, 2)} tone={Number(shadowCapital.max_drawdown || 0) < 0 ? 'negative' : ''} /><Metric label={tr('Exposure')} value={percent(shadowCapital.exposure, 2)} />{capitalPolicyV3 ? <Metric label={tr('Vs Winner')} value={percent(selectedMetrics.capital_vs_winner, 2)} tone={Number(selectedMetrics.capital_vs_winner || 0) >= 0 ? 'positive' : 'negative'} /> : null}<Metric label={tr('Latest Shadow Target')} value={bestForecast?.symbol || tr('CASH')} note={bestForecast ? `${tr(capitalPolicyV3 ? 'Opportunity Gate' : capitalPolicyV2 ? 'Entry Score' : 'Decision Score')} ${number(capitalPolicyV3 ? bestForecast.opportunity_gate_score : capitalPolicyV2 ? bestForecast.entry_score : bestForecast.decision_score, 4)}` : ''} /></section> : null}

        {selectedMetrics && !decisionExperiment ? <section className="temporal-summary-grid selected-horizon"><Metric label={tr('P(Alpha > 0) Brier')} value={number(selectedMetrics.brier, 4)} /><Metric label={tr('Brier Skill')} value={percent(selectedMetrics.brier_skill, 2)} tone={Number(selectedMetrics.brier_skill || 0) >= 0 ? 'positive' : 'negative'} /><Metric label={tr('Calibration Error')} value={percent(selectedMetrics.calibration_error, 2)} /><Metric label={tr('Alpha MAE Skill')} value={percent(selectedMetrics.alpha_mae_skill, 2)} tone={Number(selectedMetrics.alpha_mae_skill || 0) >= 0 ? 'positive' : 'negative'} /><Metric label={tr('Drawdown MAE Skill')} value={percent(selectedMetrics.drawdown_mae_skill, 2)} tone={Number(selectedMetrics.drawdown_mae_skill || 0) >= 0 ? 'positive' : 'negative'} /><Metric label={tr('High Confidence Lift')} value={percent(selectedMetrics.high_confidence_lift, 2)} tone={Number(selectedMetrics.high_confidence_lift || 0) >= 0 ? 'positive' : 'negative'} /></section> : null}

        {!decisionExperiment && selectedMetrics?.confidence_bins?.length ? <section className="temporal-section"><div className="temporal-section-heading"><h3>{tr('Probability Calibration')}</h3><span>{selectedHorizon}d</span></div><ConfidenceTable items={selectedMetrics.confidence_bins} /></section> : null}

        {!decisionExperiment ? <section className="temporal-section"><div className="temporal-section-heading"><h3>{tr('Latest Shadow Forecast')}</h3><div className="temporal-horizon-buttons">{horizonMetrics.map((item) => <button type="button" key={item.horizon} className={Number(selectedHorizon) === Number(item.horizon) ? 'active' : ''} onClick={() => setSelectedHorizon(Number(item.horizon))}>{item.horizon}d</button>)}</div></div><LegacyForecastTable items={forecasts} /></section> : null}
      </> : null}

      <TemporalStudyPanel run={run} processing={researchProcessing} canRun={canStart} canExport={canExport} canMaterializeStrategy={canMaterializeStrategy} />
    </div>
  )
}
