import { tr } from '../../../i18n/runtime'

import { lifecycleSummary, statusLabel } from '../strategySettingsUtils'

export function StrategyCatalog({
  catalog,
  orderedStrategies,
  selected,
  busy,
  researchId,
  winnerId,
  latestSavedId,
  onCloneWinner,
  onSelectDetail,
}) {
  return (
    <aside className="strategy-list-panel">
      <div className="strategy-list-heading">
        <strong>{tr("Strategy Catalog")}</strong>
        <button type="button" onClick={() => onCloneWinner(catalog.control.trader_winner)} disabled={Boolean(busy)}>{tr("Clone Winner")}</button>
      </div>
      <div className="strategy-list">
        {orderedStrategies.map((item) => {
          const isResearch = item.id === researchId && item.id !== winnerId
          const isWinner = item.id === winnerId
          const isLatestSaved = item.id === latestSavedId
          const visibleStatus = isWinner ? 'winner' : isResearch ? 'research' : 'saved'
          return (
            <article key={item.id} className={`strategy-list-item ${selected.id === item.id ? 'selected' : ''}`}>
              <button type="button" className="strategy-list-select" onClick={() => onSelectDetail(item.id)} disabled={Boolean(busy)}>
                <span className="strategy-list-title-row">
                  <strong>{item.name}</strong>
                  <small className={`strategy-status status-${visibleStatus}`}>{statusLabel(visibleStatus)}</small>
                </span>
                <span>{tr("Revision")}{' '}{item.revision} · {tr(item.locked ? 'Protected' : 'Editable')}</span>
                <span>{lifecycleSummary(item, isWinner, isResearch)}</span>
                {item.description ? <small className="strategy-list-description">{item.description}</small> : null}
              </button>
              <div className="strategy-list-markers">
                {isWinner ? <span className="winner">{tr("WINNER")}</span> : null}
                {isResearch ? <span>{tr("RESEARCH")}</span> : null}
                {isLatestSaved && !isWinner ? <span className="latest">{tr("LATEST")}</span> : null}
              </div>
            </article>
          )
        })}
      </div>
    </aside>
  )
}
