import { useState } from 'react'

import { tr } from '../../../i18n/runtime'
import { ParameterHint } from '../../../shared/components/ParameterHint'
import {
  AnalyticsModeTabs,
  ChartCell,
  SectionHeading,
} from '../../analytics/components/AnalyticsPrimitives'
import { MonthlyReturnHeatmap } from '../../analytics/components/MonthlyReturnHeatmap'
import { PerformanceDifferenceHint } from '../../analytics/components/PerformanceDifferenceHint'

export function DashboardMonthlyReturnConsistencySection({ data }) {
  const [monthlyMode, setMonthlyMode] = useState('simulation')
  const referenceLabel = data?.reference_label ? tr(data.reference_label) : tr('Reference')
  const rows = data?.monthly_returns || []

  if (!rows.length) return null

  return <section className="analytics-workspace-section analytics-performance-explorer-section dashboard-monthly-consistency-section">
    <SectionHeading
      kicker={tr('PERFORMANCE')}
      title={tr('Return and consistency')}
      hintId="dashboard-return-consistency-hint"
      hint={{
        description: tr('Shows monthly Simulation, Reference or S − R returns to reveal persistent gains, weak periods and concentration of performance.'),
        details: [
          { label: tr('Simulation'), value: tr('Strategy monthly return') },
          { label: tr('Reference'), value: tr('Benchmark monthly return') },
          { label: 'S − R', value: tr('Monthly excess return') },
          { label: tr('Monthly details'), value: tr('Click a populated month to inspect its consistency analysis') },
        ],
      }}
    />

    <div className="analytics-performance-explorer dashboard-monthly-consistency-grid">
      <ChartCell
        kicker={tr('CONSISTENCY')}
        title={tr('Monthly return heatmap')}
        className="analytics-heatmap-feature-cell"
        action={<div className="analytics-card-heading-actions">
          <ParameterHint
            id="dashboard-monthly-return-heatmap-hint"
            title={tr('Monthly return heatmap')}
            description={tr('Each month compares Simulation and Reference returns for the selected processing. Click any populated month to open its consistency analysis.')}
            details={[
              { label: tr('Simulation'), value: tr('Strategy monthly return') },
              { label: tr('Reference'), value: tr('Benchmark monthly return') },
              { label: 'S − R', value: tr('Monthly excess return') },
            ]}
          />
          <AnalyticsModeTabs
            value={monthlyMode}
            onChange={setMonthlyMode}
            label={tr('Monthly heatmap view')}
            items={[
              { value: 'simulation', label: 'Simulation' },
              { value: 'reference', label: referenceLabel },
              { value: 'excess', label: 'S − R' },
            ]}
          />
          <PerformanceDifferenceHint id="dashboard-heatmap-sr-hint" />
        </div>}
      >
        <MonthlyReturnHeatmap rows={rows} mode={monthlyMode} referenceLabel={referenceLabel} />
      </ChartCell>
    </div>
  </section>
}
