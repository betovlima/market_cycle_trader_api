import { useMemo } from 'react'

import { tr } from '../../i18n/runtime'
import { number, percent, shortDateTime } from '../../shared/formatters'

function finite(value) {
  if (value == null || value === '') return null
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

function rotationValueAdded(row) {
  return finite(row?.rotation_value_added)
}

function windowValue(row, key, field) {
  return row?.winner_transition?.trajectory?.windows?.[String(key)]?.[field] ?? null
}

function temporalWindowValue(row, key, field, measure = 'latest') {
  return row?.winner_transition?.trajectory?.windows?.[String(key)]?.temporal_target_minus_incumbent?.[field]?.[measure] ?? null
}

function Metric({ label, value, tone = '' }) {
  return <div className={`winner-transition-metric ${tone}`}><span>{label}</span><strong>{value}</strong></div>
}

export function WinnerTransitionAttributionPanel({ rotations = [] }) {
  const model = useMemo(() => {
    const attributed = (rotations || []).filter((row) => row?.winner_transition)
    const diagnostic = attributed.filter((row) => rotationValueAdded(row) != null)
    const helpful = diagnostic.filter((row) => rotationValueAdded(row) > 0)
    const harmful = diagnostic.filter((row) => rotationValueAdded(row) < 0)
    const flat = diagnostic.filter((row) => rotationValueAdded(row) === 0)
    const sortedHarmful = [...harmful].sort((left, right) => rotationValueAdded(left) - rotationValueAdded(right))
    const meanHarmful = harmful.length
      ? harmful.reduce((total, row) => total + rotationValueAdded(row), 0) / harmful.length
      : null
    return {
      attributed,
      diagnostic,
      helpful,
      harmful,
      flat,
      helpfulRate: diagnostic.length ? helpful.length / diagnostic.length : null,
      meanHarmful,
      worst: sortedHarmful[0] || null,
      rows: sortedHarmful.slice(0, 12),
    }
  }, [rotations])

  if (!model.attributed.length) return null

  return <section className="winner-transition-attribution">
    <div className="temporal-section-heading winner-transition-heading">
      <div><h3>{tr('Winner Anchor Transition Attribution')}</h3><span>{tr('Winner-anchor asset replacements')}</span></div>
    </div>

    <div className="winner-transition-metrics">
      <Metric label={tr('Attributed transitions')} value={number(model.diagnostic.length, 0)} />
      <Metric label={tr('Helpful')} value={number(model.helpful.length, 0)} tone="positive" />
      <Metric label={tr('Harmful')} value={number(model.harmful.length, 0)} tone="negative" />
      <Metric label={tr('Helpful rate')} value={model.helpfulRate == null ? '—' : percent(model.helpfulRate, 1)} />
      <Metric label={tr('Mean harmful value added')} value={model.meanHarmful == null ? '—' : percent(model.meanHarmful, 2)} tone="negative" />
      <Metric label={tr('Worst transition')} value={model.worst ? `${model.worst.from_asset || '—'} → ${model.worst.to_asset || '—'} · ${percent(rotationValueAdded(model.worst), 2)}` : '—'} tone="negative" />
    </div>

    {model.rows.length ? <div className="temporal-table-shell winner-transition-table-shell">
      <table className="temporal-table winner-transition-table">
        <thead><tr>
          <th>{tr('Executed')}</th>
          <th>{tr('Transition')}</th>
          <th>{tr('Rotation value added')}</th>
          <th>{tr('Winner gap')}</th>
          <th>{tr('Leader changes · 5d')}</th>
          <th>{tr('Target Top-1 · 5d')}</th>
          <th>{tr('Score Δ · 5d')}</th>
          <th>{tr('Profit Δ')}</th>
          <th>{tr('Risk safety Δ')}</th>
          <th>{tr('Predicted DD Δ')}</th>
        </tr></thead>
        <tbody>{model.rows.map((row, index) => {
          const valueAdded = rotationValueAdded(row)
          const profitDelta = temporalWindowValue(row, 5, 'short_profit_consensus')
          const riskDelta = temporalWindowValue(row, 5, 'all_horizon_risk_safety')
          const drawdownDelta = temporalWindowValue(row, 5, 'predicted_drawdown')
          return <tr key={`${row.executed_at}-${row.from_asset}-${row.to_asset}-${index}`}>
            <td>{shortDateTime(row.executed_at)}</td>
            <td><strong>{row.from_asset || 'CASH'} → {row.to_asset || 'CASH'}</strong></td>
            <td className={valueAdded >= 0 ? 'positive' : 'negative'}>{percent(valueAdded, 2)}</td>
            <td>{row?.winner_transition?.winner_top1_top2_score_gap == null ? '—' : number(row.winner_transition.winner_top1_top2_score_gap, 4)}</td>
            <td>{number(windowValue(row, 5, 'leader_change_count'), 0)}</td>
            <td>{windowValue(row, 5, 'target_top1_rate') == null ? '—' : percent(windowValue(row, 5, 'target_top1_rate'), 0)}</td>
            <td>{windowValue(row, 5, 'target_minus_incumbent_score_delta') == null ? '—' : number(windowValue(row, 5, 'target_minus_incumbent_score_delta'), 4)}</td>
            <td className={profitDelta == null ? '' : profitDelta >= 0 ? 'positive' : 'negative'}>{profitDelta == null ? '—' : number(profitDelta, 4)}</td>
            <td className={riskDelta == null ? '' : riskDelta >= 0 ? 'positive' : 'negative'}>{riskDelta == null ? '—' : number(riskDelta, 4)}</td>
            <td className={drawdownDelta == null ? '' : drawdownDelta <= 0 ? 'positive' : 'negative'}>{drawdownDelta == null ? '—' : number(drawdownDelta, 4)}</td>
          </tr>
        })}</tbody>
      </table>
    </div> : null}
  </section>
}
