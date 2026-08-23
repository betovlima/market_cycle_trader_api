import { tr } from '../../../i18n/runtime'

import { ActivityIcon, ShieldIcon, TrophyIcon } from '../../../shared/components/Icons'
import { ParameterHint } from '../../../shared/components/ParameterHint'
import { BOUNDARY_HINTS } from '../strategySettingsConfig'

export function StrategyBoundaryGrid({ catalog }) {
  const winner = catalog.control.trader_winner
  const research = catalog.control.research_strategy_id !== catalog.control.trader_winner_strategy_id
    ? catalog.control.research_strategy
    : null
  return (
    <div className="strategy-boundary-grid">
      <article className="winner-boundary-card">
        <TrophyIcon size={20} />
        <div>
          <span className="strategy-boundary-label">{tr("WINNER")}{' '}<ParameterHint id="hint-boundary-winner" title={tr("WINNER")} {...BOUNDARY_HINTS.winner} /></span>
          <strong>{winner?.name}</strong>
          <small>{winner?.description || '—'}</small>
        </div>
      </article>
      <article>
        <ActivityIcon size={20} />
        <div>
          <span className="strategy-boundary-label">{tr("RESEARCH")}{' '}<ParameterHint id="hint-boundary-strategy-research" title={tr("RESEARCH")} {...BOUNDARY_HINTS.backtest} /></span>
          <strong>{research?.name || tr('No RESEARCH selected')}</strong>
          <small>{research?.description || '—'}</small>
        </div>
      </article>
      <article>
        <ShieldIcon size={20} />
        <div>
          <span className="strategy-boundary-label">{tr("Catalog rule")}{' '}<ParameterHint id="hint-boundary-lifecycle" title={tr("Catalog rule")} align="right" {...BOUNDARY_HINTS.lifecycle} /></span>
          <strong>{tr("One RESEARCH · one WINNER")}</strong>
          <small>{tr("Promotion changes the role, never the Strategy identity.")}</small>
        </div>
      </article>
    </div>
  )
}
