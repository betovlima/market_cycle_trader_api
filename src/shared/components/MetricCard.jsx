import { tr } from '../../i18n/runtime'
export function MetricCard({ label, value, note = null }) {
  return (
    <article className="metric-card">
      <span>{tr(label)}</span>
      <strong>{value}</strong>
      {note && <small>{tr(note)}</small>}
    </article>
  )
}

