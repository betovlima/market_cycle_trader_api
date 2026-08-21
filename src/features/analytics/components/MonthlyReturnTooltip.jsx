import { tr } from '../../../i18n/runtime'
import { createPortal } from 'react-dom'

import { percent } from '../../../shared/formatters'
import { returnTone } from '../utils/performance'

export function MonthlyReturnTooltip({ tooltip }) {
  if (!tooltip || typeof document === 'undefined') return null

  return createPortal(
    <div
      className={`analytics-heatmap-tooltip ${tooltip.placement}`}
      style={{ left: `${tooltip.left}px`, top: `${tooltip.top}px` }}
      role="tooltip"
    >
      <div className="analytics-heatmap-tooltip-header">
        <div>
          <strong>{tooltip.month} {tooltip.year}</strong>
          <span>{tr("Monthly performance")}</span>
        </div>
        <b className={returnTone(tooltip.selectedValue)}>
          {percent(tooltip.selectedValue)}
        </b>
      </div>

      <div className="analytics-heatmap-tooltip-selected">
        <span>{tr("Current view")}</span>
        <strong>{tr(tooltip.selectedModeLabel)}</strong>
      </div>

      <div className="analytics-heatmap-tooltip-grid">
        <Metric label={tooltip.simulationLabel || tr("Simulation")} value={tooltip.simulation} />
        <Metric label={tooltip.referenceLabel || tr("Reference")} value={tooltip.reference} />
        <Metric label={tooltip.excessLabel || "S − R"} value={tooltip.excess} signed />
      </div>

      <div className="analytics-heatmap-tooltip-result">
        <span className={`analytics-heatmap-tooltip-dot ${returnTone(tooltip.excess)}`} />
        <span>{tr(tooltip.relativeResult)} · {tr('Click for monthly consistency details')}</span>
      </div>
    </div>,
    document.body,
  )
}

function Metric({ label, value, signed = false }) {
  return <div>
    <span>{tr(label)}</span>
    <strong className={returnTone(value)}>
      {signed && value > 0 ? '+' : ''}{percent(value)}
    </strong>
  </div>
}
