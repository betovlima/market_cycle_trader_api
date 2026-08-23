import { tr } from '../../../i18n/runtime'
import { useMemo, useState } from 'react'
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import { money, number, percent, shortDateTime } from '../../../shared/formatters'
import { ParameterHint } from '../../../shared/components/ParameterHint'
import { usePerformanceZoom } from '../hooks/usePerformanceZoom'
import { useReorderableCards } from '../hooks/useReorderableCards'
import { analyticsAxisLabel, analyticsTimestamp, monthParts, returnTone } from '../utils/performance'
import {
  AnalyticsDragHandle,
  AnalyticsResponsiveContainer,
  AnalyticsModeTabs,
  ChartCell,
  ChartEmpty,
  SectionHeading,
} from './AnalyticsPrimitives'
import { MonthlyReturnHeatmap } from './MonthlyReturnHeatmap'
import { PerformanceDifferenceHint } from './PerformanceDifferenceHint'

const PERFORMANCE_LAYOUT_STORAGE_KEY = 'market-cycle-trader.analytics.performance-layout.v1'
const DEFAULT_PERFORMANCE_LAYOUT = ['performance', 'heatmap']


function PerformanceTooltip({ active, payload, referenceLabel = null }) {
  if (!active || !payload?.length) return null
  const point = payload.find((item) => item?.payload?.timestamp)?.payload
  if (!point) return null

  return <div className="analytics-performance-tooltip">
    <strong>{shortDateTime(point.timestamp)}</strong>
    <div><span>{tr("Simulation")}</span><b>{money(point.simulation_equity)}</b></div>
    <div><span>{referenceLabel || tr("Reference")}</span><b>{money(point.reference_equity)}</b></div>
    <div><span>{tr("Simulation change")}</span><b className={returnTone(point.simulation_change)}>{percent(point.simulation_change)}</b></div>
    <div><span>{tr("Reference change")}</span><b className={returnTone(point.reference_change)}>{percent(point.reference_change)}</b></div>
    <div><span>S − R</span><b className={returnTone(point.excess_change)}>{percent(point.excess_change)}</b></div>
  </div>
}

export function BacktestPerformanceExplorer({ data, jobId }) {
  const referenceLabel = data?.reference_label ? tr(data.reference_label) : tr('Reference')
  const [performanceMode, setPerformanceMode] = useState('value')
  const [monthlyMode, setMonthlyMode] = useState('simulation')
  const reorderable = useReorderableCards({
    storageKey: PERFORMANCE_LAYOUT_STORAGE_KEY,
    defaultLayout: DEFAULT_PERFORMANCE_LAYOUT,
  })

  const equityRows = useMemo(() => (data?.equity || [])
    .map((row) => ({ ...row, timestamp_value: analyticsTimestamp(row.timestamp) }))
    .filter((row) => row.timestamp_value !== null)
    .sort((left, right) => left.timestamp_value - right.timestamp_value), [data?.equity])

  const {
    chartInteractionRef,
    effectiveZoomDomain,
    isPanning,
    visibleEquityRows,
    visibleSpan,
    zoomActive,
    zoomLevel,
    beginPan,
    movePan,
    endPan,
    resetZoom,
  } = usePerformanceZoom({ equityRows, jobId })

  const performanceRows = useMemo(() => {
    if (!visibleEquityRows.length) return []
    const firstSimulation = visibleEquityRows
      .map((row) => Number(row.simulation_equity))
      .find(Number.isFinite)
    const firstReference = visibleEquityRows
      .map((row) => Number(row.reference_equity))
      .find(Number.isFinite)

    return visibleEquityRows.map((row) => {
      const simulation = Number(row.simulation_equity)
      const reference = Number(row.reference_equity)
      const simulationChange = Number.isFinite(simulation) && Number.isFinite(firstSimulation) && firstSimulation !== 0
        ? (simulation / firstSimulation) - 1
        : null
      const referenceChange = Number.isFinite(reference) && Number.isFinite(firstReference) && firstReference !== 0
        ? (reference / firstReference) - 1
        : null

      return {
        ...row,
        simulation_change: simulationChange,
        reference_change: referenceChange,
        excess_change: Number.isFinite(simulationChange) && Number.isFinite(referenceChange)
          ? simulationChange - referenceChange
          : null,
        simulation_index: Number.isFinite(simulationChange) ? (1 + simulationChange) * 100 : null,
        reference_index: Number.isFinite(referenceChange) ? (1 + referenceChange) * 100 : null,
      }
    })
  }, [visibleEquityRows])

  const performanceDomain = useMemo(() => {
    const keys = performanceMode === 'value'
      ? ['simulation_equity', 'reference_equity']
      : performanceMode === 'indexed'
        ? ['simulation_index', 'reference_index']
        : ['excess_change']
    const values = performanceRows
      .flatMap((row) => keys.map((key) => Number(row[key])))
      .filter(Number.isFinite)
    if (!values.length) return ['auto', 'auto']

    const minimum = Math.min(...values)
    const maximum = Math.max(...values)
    const spread = maximum - minimum
    const magnitude = Math.max(Math.abs(minimum), Math.abs(maximum), 1)
    const padding = spread > 0
      ? Math.max(spread * .08, magnitude * .0005)
      : Math.max(magnitude * .002, .01)

    return [minimum - padding, maximum + padding]
  }, [performanceMode, performanceRows])

  const visibleMonthlyRows = useMemo(() => {
    if (!effectiveZoomDomain) return data?.monthly_returns || []
    return (data?.monthly_returns || []).filter((row) => {
      const parts = monthParts(row.month)
      if (!parts) return false
      const monthStart = new Date(parts.year, parts.month - 1, 1).getTime()
      const monthEnd = new Date(parts.year, parts.month, 1).getTime() - 1
      return monthEnd >= effectiveZoomDomain.start && monthStart <= effectiveZoomDomain.end
    })
  }, [data?.monthly_returns, effectiveZoomDomain])

  const cardDragHandle = (cardId, label) => <AnalyticsDragHandle
    label={label}
    {...reorderable.dragHandleProps(cardId)}
  />

  const performanceAction = <div className="analytics-chart-controls">
    <AnalyticsModeTabs
      value={performanceMode}
      onChange={setPerformanceMode}
      label={tr("Performance chart view")}
      items={[
        { value: 'value', label: 'Value' },
        { value: 'indexed', label: 'Indexed 100' },
        { value: 'excess', label: 'S − R' },
      ]}
    />
    <PerformanceDifferenceHint id="analytics-performance-sr-hint" />
    <span className="analytics-zoom-status">
      {zoomActive
        ? tr('{level}× · drag to pan', { level: zoomLevel >= 10 ? zoomLevel.toFixed(0) : zoomLevel.toFixed(1) })
        : tr('Wheel to zoom')}
    </span>
    {zoomActive ? <button type="button" className="analytics-reset-zoom" onClick={resetZoom}>{tr("Reset")}</button> : null}
  </div>

  const performanceCards = {
    performance: <ChartCell
      kicker={tr("PERFORMANCE EXPLORER")}
      title={tr("Simulation versus reference")}
      className="analytics-chart-primary analytics-performance-main"
      action={<div className="analytics-card-heading-actions">
        {performanceAction}
        {cardDragHandle('performance', 'Simulation versus reference')}
      </div>}
    >
      {performanceRows.length ? <div
        ref={chartInteractionRef}
        className={`analytics-chart analytics-chart-explorer analytics-interactive-chart ${zoomActive ? 'is-zoomed' : ''} ${isPanning ? 'is-panning' : ''}`}
        onPointerDown={beginPan}
        onPointerMove={movePan}
        onPointerUp={endPan}
        onPointerCancel={endPan}
      >
        <AnalyticsResponsiveContainer fallbackHeight={350}>
          <LineChart data={performanceRows}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="timestamp_value"
              type="number"
              domain={['dataMin', 'dataMax']}
              tickFormatter={(value) => analyticsAxisLabel(value, visibleSpan)}
              minTickGap={38}
            />
            <YAxis
              domain={performanceDomain}
              tickFormatter={(value) => performanceMode === 'value'
                ? `$${Math.round(value / 1000)}k`
                : performanceMode === 'indexed'
                  ? number(value, 0)
                  : `${(value * 100).toFixed(0)}%`}
            />
            <Tooltip
              content={<PerformanceTooltip referenceLabel={referenceLabel} />}
              cursor={{ stroke: 'rgba(147, 177, 210, .45)', strokeDasharray: '4 4' }}
            />
            {performanceMode === 'value' ? <>
              <Line type="monotone" dataKey="simulation_equity" name={tr("Simulation")} dot={false} strokeWidth={2.3} stroke="var(--positive)" isAnimationActive={false} />
              <Line type="monotone" dataKey="reference_equity" name={referenceLabel} dot={false} strokeWidth={2} stroke="var(--accent)" isAnimationActive={false} />
            </> : performanceMode === 'indexed' ? <>
              <ReferenceLine y={100} stroke="rgba(145, 169, 198, .35)" strokeDasharray="3 4" />
              <Line type="monotone" dataKey="simulation_index" name={tr("Simulation")} dot={false} strokeWidth={2.3} stroke="var(--positive)" isAnimationActive={false} />
              <Line type="monotone" dataKey="reference_index" name={referenceLabel} dot={false} strokeWidth={2} stroke="var(--accent)" isAnimationActive={false} />
            </> : <>
              <ReferenceLine y={0} stroke="rgba(145, 169, 198, .45)" strokeDasharray="3 4" />
              <Line type="monotone" dataKey="excess_change" name="S − R" dot={false} strokeWidth={2.3} stroke="var(--positive)" isAnimationActive={false} />
            </>}
          </LineChart>
        </AnalyticsResponsiveContainer>
      </div> : <ChartEmpty />}
    </ChartCell>,

    heatmap: <ChartCell
      kicker={tr("CONSISTENCY")}
      title={tr("Monthly return heatmap")}
      className="analytics-heatmap-feature-cell"
      action={<div className="analytics-card-heading-actions">
        <ParameterHint
          id="analytics-monthly-return-heatmap-hint"
          title={tr('Monthly return heatmap')}
          description={tr('Each month compares Simulation and Reference returns for the selected processing. Click any populated month to open its consistency analysis.')}
          details={[
            { label: tr('Simulation'), value: tr('Strategy monthly return') },
            { label: tr('Reference'), value: tr('Benchmark monthly return') },
            { label: 'S − R', value: tr('Monthly excess return') },
            { label: tr('Monthly details'), value: tr('Rank, rolling 3-month return, YTD, streak, average, median and volatility') },
          ]}
        />
        <AnalyticsModeTabs
          value={monthlyMode}
          onChange={setMonthlyMode}
          label={tr("Monthly heatmap view")}
          items={[
            { value: 'simulation', label: 'Simulation' },
            { value: 'reference', label: referenceLabel },
            { value: 'excess', label: 'S − R' },
          ]}
        />
        <PerformanceDifferenceHint id="analytics-heatmap-sr-hint" />
        {cardDragHandle('heatmap', 'Monthly return heatmap')}
      </div>}
    >
      <MonthlyReturnHeatmap rows={visibleMonthlyRows} mode={monthlyMode} referenceLabel={referenceLabel} />
    </ChartCell>,
  }

  return <section className="analytics-workspace-section analytics-performance-explorer-section">
    <SectionHeading
      kicker={tr("PERFORMANCE")}
      title={tr("Return and consistency")}
      hintId="analytics-return-consistency-hint"
      hint={{
        description: tr("Combines capital growth and monthly consistency so the selected processing can be evaluated by both total performance and how that performance was built through time."),
        details: [
          { label: tr("Value"), value: tr("Capital in USD"), description: tr("Shows the historical Simulation and Reference equity values in the selected processing.") },
          { label: tr("Indexed 100"), value: tr("Relative growth"), description: tr("Normalizes both series to 100 at the beginning of the visible window so their relative growth can be compared directly.") },
          { label: tr("S − R"), value: tr("Excess return"), description: tr("Shows the cumulative performance difference between Simulation and Reference. Values above zero indicate relative outperformance.") },
          { label: tr("Monthly heatmap"), value: tr("Consistency"), description: tr("Shows monthly Simulation, Reference or S − R returns to reveal persistent gains, weak periods and concentration of performance.") },
          { label: tr("Zoom / pan"), value: tr("Synchronized window"), description: tr("Changing the visible period in the capital chart applies the same time window to the monthly return heatmap.") },
        ],
        relationship: tr("A stronger historical profile combines high cumulative return with repeatable monthly performance. A large final return concentrated in only a few periods should be interpreted as less consistent than a similar return distributed across many months."),
      }}
      action={<div className="analytics-layout-toolbar">
        <span><b aria-hidden="true">⋮⋮</b> {tr("Drag chart headers to reorder")}</span>
        {reorderable.customized ? <button type="button" onClick={reorderable.reset}>{tr("Reset layout")}</button> : null}
      </div>}
    />

    <div className="analytics-performance-explorer analytics-reorderable-performance">
      {reorderable.layout.map((cardId) => <div
        key={cardId}
        className={`analytics-reorderable-card ${reorderable.draggedId === cardId ? 'is-dragging' : ''} ${reorderable.dropTargetId === cardId ? 'is-drop-target' : ''} ${reorderable.justDroppedId === cardId ? 'just-dropped' : ''}`}
        {...reorderable.dropZoneProps(cardId)}
      >{performanceCards[cardId]}</div>)}
    </div>
  </section>
}
