import { tr } from '../../../i18n/runtime'

import { ActivityIcon, ShieldIcon, StarIcon, TrophyIcon } from '../../../shared/components/Icons'
import { ParameterHint } from '../../../shared/components/ParameterHint'
import { BOUNDARY_HINTS } from '../strategySettingsConfig'

export function StrategyBoundaryGrid({ catalog }) {
  return (
<div className="strategy-boundary-grid">
        <article className="winner-boundary-card">
          <TrophyIcon size={20} />
          <div>
            <span className="strategy-boundary-label">{tr("Trader winner")}{' '}<ParameterHint id="hint-boundary-winner" title={tr("Trader winner")} {...BOUNDARY_HINTS.winner} /></span>
            <strong>{catalog.control.trader_winner?.name}</strong>
            <small>{catalog.control.trader_winner?.winner_model?.label || tr('XGBoost Utility')}</small>
          </div>
        </article>
        <article>
          <ActivityIcon size={20} />
          <div>
            <span className="strategy-boundary-label">{tr("Strategy Research")}{' '}<ParameterHint id="hint-boundary-strategy-research" title={tr("Strategy Research")} {...BOUNDARY_HINTS.backtest} /></span>
            <strong>{catalog.control.research_strategy?.name}</strong>
            {catalog.control.research_strategy?.research_model?.label ? <small>{catalog.control.research_strategy.research_model.label}</small> : null}
          </div>
        </article>
        <article className="candidate-boundary-card">
          <StarIcon size={20} />
          <div>
            <span className="strategy-boundary-label">{tr("Current candidate")}{' '}<ParameterHint id="hint-boundary-candidate" title={tr("Current candidate")} {...BOUNDARY_HINTS.candidate} /></span>
            <strong>{catalog.control.candidate_strategy?.name || tr('No active candidate')}</strong>
            {catalog.control.candidate_strategy?.candidate_model?.label ? <small>{catalog.control.candidate_strategy.candidate_model.label}</small> : null}
          </div>
        </article>
        <article className="promoted-candidate-boundary-card">
          <TrophyIcon size={20} />
          <div>
            <span className="strategy-boundary-label">{tr("Promoted candidate")}</span>
            <strong>{catalog.control.promoted_candidate_strategy?.name || tr('No promoted candidate')}</strong>
            {catalog.control.promoted_candidate_strategy?.candidate_model?.label ? <small>{catalog.control.promoted_candidate_strategy.candidate_model.label}</small> : null}
          </div>
        </article>
        <article>
          <ShieldIcon size={20} />
          <div>
            <span className="strategy-boundary-label">{tr("Lifecycle rule")}{' '}<ParameterHint id="hint-boundary-lifecycle" title={tr("Lifecycle rule")} align="right" {...BOUNDARY_HINTS.lifecycle} /></span>
            <strong>{tr("One Candidate · one Promoted Candidate · one Winner")}</strong>
          </div>
        </article>
      </div>
  )
}
