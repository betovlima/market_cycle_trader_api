import { tr } from '../../../i18n/runtime'

import { lifecycleSummary, statusLabel } from '../strategySettingsUtils'

export function StrategyCatalog({
  catalog,
  orderedStrategies,
  selected,
  busy,
  researchId,
  winnerId,
  candidateId,
  promotedCandidateId,
  onCloneWinner,
  onSelectDetail,
}) {
  return (
<aside className="strategy-list-panel">
          <div className="strategy-list-heading">
            <strong>{tr("Strategy catalog")}</strong>
            <button type="button" onClick={() => onCloneWinner(catalog.control.trader_winner)} disabled={Boolean(busy)}>{tr("Clone winner")}</button>
          </div>
          <div className="strategy-list">
            {orderedStrategies.map((item) => {
              const isResearch = item.id === researchId
              const isWinner = item.id === winnerId
              const isCandidate = item.id === candidateId
              const isPromotedCandidate = item.id === promotedCandidateId
              return (
                <article key={item.id} className={`strategy-list-item ${selected.id === item.id ? 'selected' : ''}`}>
                  <button type="button" className="strategy-list-select" onClick={() => onSelectDetail(item.id)} disabled={Boolean(busy)}>
                    <span className="strategy-list-title-row">
                      <strong>{item.name}</strong>
                      <small className={`strategy-status status-${item.status}`}>{statusLabel(item.status)}</small>
                    </span>
                    <span>{tr("Revision")}{' '}{item.revision} · {tr(item.locked ? 'Protected' : 'Editable')}</span>
                    <span>{lifecycleSummary(item, isWinner, isCandidate, isPromotedCandidate)}</span>
                  </button>
                  <div className="strategy-list-markers">
                    {isResearch ? <span>{tr("RESEARCH")}</span> : null}
                    {isCandidate ? <span className="candidate">{tr("CANDIDATE")}</span> : null}
                    {isPromotedCandidate ? <span className="promoted">{tr("PROMOTED")}</span> : null}
                    {isWinner ? <span className="winner">{tr("TRADER")}</span> : null}
                  </div>
                </article>
              )
            })}
          </div>
        </aside>
  )
}
