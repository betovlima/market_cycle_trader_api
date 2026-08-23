import { tr } from '../../../i18n/runtime'
import { money, number, percent } from '../utils/formatters'

export function ControlParity({ parity }) {
  if (!parity) return null
  const passed = parity.status === 'passed'
  const reference = parity.reference || {}
  const replay = parity.replay || {}
  return <section className={`milp-parity ${passed ? 'passed' : 'failed'}`}>
    <div className="milp-parity-head">
      <div><strong>{tr('Control replay parity')}</strong><span>{tr('MILP must reproduce the selected Strategy exactly before optimization.')}</span></div>
      <b>{passed ? tr('Passed') : tr('Failed')}</b>
    </div>
    <div className="milp-parity-grid">
      <div><span>{tr('Ending capital')}</span><strong>{money(replay.ending_capital)}</strong><small>{tr('Reference')} {money(reference.ending_capital)}</small></div>
      <div><span>{tr('Switches')}</span><strong>{number(replay.switches, 0)}</strong><small>{tr('Reference')} {number(reference.switches, 0)}</small></div>
      <div><span>{tr('Cash days')}</span><strong>{number(replay.cash_days, 0)}</strong><small>{tr('Reference')} {number(reference.cash_days, 0)}</small></div>
      <div><span>{tr('Market exposure')}</span><strong>{percent(replay.market_exposure, 2)}</strong><small>{tr('Reference')} {percent(reference.market_exposure, 2)}</small></div>
      <div><span>{tr('Equity sessions')}</span><strong>{number(replay.equity_sessions, 0)}</strong><small>{tr('Reference')} {number(reference.equity_sessions, 0)}</small></div>
    </div>
  </section>
}
