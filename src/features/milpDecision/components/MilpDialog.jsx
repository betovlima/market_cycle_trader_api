import { tr } from '../../../i18n/runtime'

export function MilpDialog({ detail, onClose }) {
  if (!detail) return null
  return <div className="milp-dialog-backdrop" role="presentation" onMouseDown={onClose}>
    <section className="milp-dialog" role="dialog" aria-modal="true" aria-label={detail.title || tr('MILP decision detail')} onMouseDown={(event) => event.stopPropagation()}>
      <header><div><span>{detail.kicker || 'MILP'}</span><h4>{detail.title}</h4></div><button type="button" onClick={onClose} aria-label={tr('Close')}>×</button></header>
      {detail.description ? <p>{detail.description}</p> : null}
      {detail.metrics?.length ? <div className="milp-dialog-metrics">{detail.metrics.map((item) => <div key={`${item.label}-${item.value}`}><span>{item.label}</span><strong className={item.tone || ''}>{item.value}</strong></div>)}</div> : null}
      {detail.components?.length ? <div className="milp-objective-list">{detail.components.map((item) => <div key={item.label}><span>{item.label}</span><strong>{item.value}</strong></div>)}</div> : null}
      {detail.alternatives?.length ? <div className="milp-alternatives"><strong>{detail.itemsTitle || tr('Alternatives evaluated')}</strong>{detail.alternatives.map((item) => <div key={`${item.symbol}-${item.objective}`}><span>{item.symbol}</span><strong>{item.objective}</strong></div>)}</div> : null}
    </section>
  </div>
}
