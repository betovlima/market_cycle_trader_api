import { tr } from '../../../i18n/runtime'
import { money, number, percent } from '../utils/formatters'

export function CandidateCard({ title, subtitle, metrics = {}, selectable = false, selected = false, onSelect, extra = null }) {
  return <button type="button" className={`milp-candidate-card ${selectable ? 'selectable' : ''} ${selected ? 'selected' : ''}`} onClick={selectable ? onSelect : undefined} disabled={!selectable}>
    <div className="milp-candidate-card-head"><div><strong>{tr(title)}</strong><span>{tr(subtitle)}</span></div>{selectable ? <span className="milp-candidate-select">{selected ? '✓' : '○'}</span> : null}</div>
    <div className="milp-candidate-metrics">
      <div><span>{tr('Ending capital')}</span><strong>{money(metrics.ending_capital)}</strong></div>
      <div><span>CAGR</span><strong>{percent(metrics.cagr, 2)}</strong></div>
      <div><span>Sharpe</span><strong>{number(metrics.sharpe, 3)}</strong></div>
      <div><span>MaxDD</span><strong>{percent(metrics.maximum_drawdown ?? metrics.max_drawdown, 2)}</strong></div>
      <div><span>{tr('Switches')}</span><strong>{number(metrics.capital_rotations ?? metrics.switch_count, 0)}</strong></div>
      <div><span>{tr('Cash days')}</span><strong>{number(metrics.cash_days, 0)}</strong></div>
    </div>
    {extra}
  </button>
}
