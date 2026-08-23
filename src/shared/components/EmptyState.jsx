import { tr } from '../../i18n/runtime'
export function EmptyState() {
  return (
    <section className="empty-state">
      <div className="empty-icon">◎</div>
      <h2>{tr("No completed backtest yet")}</h2>
      <p>{tr("Run the protected simulation to generate the first result.")}</p>
    </section>
  )
}
