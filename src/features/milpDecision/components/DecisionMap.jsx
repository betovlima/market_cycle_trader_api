import { useMemo, useState } from 'react'

import { tr } from '../../../i18n/runtime'
import { MilpDialog } from './MilpDialog'
import { number, percent } from '../utils/formatters'

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

function monthKey(row) {
  return String(row?.execution_at || row?.decision_at || '').slice(0, 7)
}

function isDifferent(row) {
  return String(row?.target_symbol || '') !== String(row?.control_target_symbol || '')
}

function monthDetail(title, rows, mode) {
  const holds = rows.filter((row) => row?.action === 'HOLD').length
  const rotates = rows.filter((row) => row?.action === 'ROTATE').length
  const buys = rows.filter((row) => row?.action === 'BUY').length
  const cash = rows.filter((row) => row?.action === 'CASH').length
  const differences = rows.filter(isDifferent)
  const metrics = mode === 'activity'
    ? [
        { label: tr('Decisions'), value: number(rows.length, 0) },
        { label: 'HOLD', value: number(holds, 0) },
        { label: 'ROTATE', value: number(rotates, 0) },
        { label: 'BUY', value: number(buys, 0) },
        { label: 'CASH', value: number(cash, 0) },
        { label: tr('Rotation rate'), value: percent(rows.length ? rotates / rows.length : 0, 1) },
      ]
    : [
        { label: tr('Decisions'), value: number(rows.length, 0) },
        { label: tr('Same decision'), value: number(rows.length - differences.length, 0) },
        { label: tr('Different decision'), value: number(differences.length, 0) },
      ]
  const listedRows = mode === 'activity' ? rows.filter((row) => row?.action === 'ROTATE') : differences
  return {
    kicker: mode === 'activity' ? tr('Policy Activity') : tr('MILP vs Control'),
    title,
    metrics,
    itemsTitle: mode === 'activity' ? tr('Rotations in month') : tr('Different decisions in month'),
    alternatives: listedRows.slice(0, 30).map((row) => ({
      symbol: mode === 'activity'
        ? `${String(row.execution_at || row.decision_at || '').slice(0, 10)} · ${row.current_symbol || '—'} → ${row.target_symbol || '—'}`
        : `${String(row.execution_at || row.decision_at || '').slice(0, 10)} · ${row.control_target_symbol || '—'} → ${row.target_symbol || '—'}`,
      objective: mode === 'activity' ? 'ROTATE' : percent(row.decision_value_added_vs_control, 2),
    })),
  }
}

function buildMonths(decisions) {
  const grouped = new Map()
  for (const row of decisions) {
    const key = monthKey(row)
    if (!/^\d{4}-\d{2}$/.test(key)) continue
    if (!grouped.has(key)) grouped.set(key, [])
    grouped.get(key).push(row)
  }
  return grouped
}

function yearsFromMonths(months) {
  const grouped = new Map()
  for (const key of months.keys()) {
    const [year, month] = key.split('-')
    if (!grouped.has(year)) grouped.set(year, {})
    grouped.get(year)[Number(month)] = key
  }
  return [...grouped.entries()].sort(([a], [b]) => a.localeCompare(b))
}

function MonthlyMap({ title, subtitle, decisions, mode, onDetail }) {
  const months = useMemo(() => buildMonths(decisions), [decisions])
  const years = useMemo(() => yearsFromMonths(months), [months])
  if (!years.length) return null

  return <section className="milp-decision-map">
    <div className="milp-section-heading"><div><strong>{tr(title)}</strong><span>{tr(subtitle)}</span></div></div>
    <div className="milp-month-grid milp-month-head"><span />{MONTHS.map((month) => <strong key={month}>{month}</strong>)}</div>
    {years.map(([year, rowMonths]) => <div className="milp-month-grid" key={year}><strong>{year}</strong>{MONTHS.map((_, index) => {
      const key = rowMonths[index + 1]
      const rows = key ? months.get(key) || [] : []
      const rotations = rows.filter((row) => row?.action === 'ROTATE').length
      const differences = rows.filter(isDifferent).length
      const value = mode === 'activity' ? rotations : differences
      const label = !key ? '·' : mode === 'activity' ? (rotations ? `R${rotations}` : 'R0') : (differences ? `Δ${differences}` : '=')
      return <button
        key={`${year}-${index + 1}`}
        type="button"
        className={`milp-month-cell milp-${mode} ${value ? 'active' : 'zero'}`}
        disabled={!key}
        onClick={() => onDetail(monthDetail(key, rows, mode))}
      >{label}</button>
    })}</div>)}
  </section>
}

export function DecisionMap({ result }) {
  const [detail, setDetail] = useState(null)
  const decisions = Array.isArray(result?.decisions) ? result.decisions : []
  if (!decisions.length) return null
  return <div className="milp-decision-maps">
    <MonthlyMap
      title="Policy Activity"
      subtitle="Rotations per month"
      decisions={decisions}
      mode="activity"
      onDetail={setDetail}
    />
    <MonthlyMap
      title="MILP vs Control"
      subtitle="Decision differences per month"
      decisions={decisions}
      mode="difference"
      onDetail={setDetail}
    />
    <MilpDialog detail={detail} onClose={() => setDetail(null)} />
  </div>
}
