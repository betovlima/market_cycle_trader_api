import { useMemo, useState } from 'react'
import {
  CartesianGrid,
  Line,
  LineChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import { tr } from '../../i18n/runtime'
import { money, shortDateTime } from '../../shared/formatters'
import { AnalyticsModeTabs, AnalyticsResponsiveContainer, ChartCell, ChartEmpty } from '../analytics/components/AnalyticsPrimitives'
import { usePerformanceZoom } from '../analytics/hooks/usePerformanceZoom'
import { analyticsAxisLabel, analyticsTimestamp } from '../analytics/utils/performance'

function ShadowTooltip({ active, payload, mode, hasCalibrated }) {
  if (!active || !payload?.length) return null
  const point = payload.find((item) => item?.payload?.timestamp)?.payload
  if (!point) return null
  return <div className="analytics-performance-tooltip">
    <strong>{shortDateTime(point.timestamp)}</strong>
    <div><span>{tr('Control')}</span><b>{money(point.baseline)}</b></div>
    {mode === 'selected' ? <>{hasCalibrated ? <div><span>{tr('Confidence calibrated')}</span><b>{money(point.calibrated)}</b></div> : null}<div><span>{tr('Walk-forward intervention')}</span><b>{money(point.selected)}</b></div></> : <>
      <div><span>{tr('Long risk shadow')}</span><b>{money(point.legacy)}</b></div>
      <div><span>{tr('One-session shadow')}</span><b>{money(point.oneSession)}</b></div>
    </>}
  </div>
}

function pathMap(rows) {
  return new Map((rows || []).map((row) => [String(row.timestamp || ''), Number(row.value)]))
}

export function TransitionShadowCapitalChart({ result, confidenceResult = null, chartKey = '' }) {
  const [mode, setMode] = useState('selected')
  const selected = result?.walk_forward_selected_shadow?.equity || {}
  const oneSession = result?.one_session_all_oos_shadow?.equity || {}
  const legacy = result?.legacy_long_shadow_reference?.equity || {}
  const calibrated = confidenceResult?.walk_forward_calibrated_shadow?.equity || {}

  const equityRows = useMemo(() => {
    const baselineRows = selected?.baseline?.length ? selected.baseline : (legacy?.baseline || [])
    const selectedMap = pathMap(selected?.shadow)
    const oneSessionMap = pathMap(oneSession?.shadow)
    const legacyMap = pathMap(legacy?.shadow)
    const calibratedMap = pathMap(calibrated?.shadow)
    return (baselineRows || [])
      .map((row) => ({
        timestamp: row.timestamp,
        timestamp_value: analyticsTimestamp(row.timestamp),
        baseline: Number(row.value),
        selected: selectedMap.get(String(row.timestamp || '')),
        oneSession: oneSessionMap.get(String(row.timestamp || '')),
        legacy: legacyMap.get(String(row.timestamp || '')),
        calibrated: calibratedMap.get(String(row.timestamp || '')),
      }))
      .filter((row) => row.timestamp_value !== null && Number.isFinite(row.baseline))
      .sort((left, right) => left.timestamp_value - right.timestamp_value)
  }, [calibrated?.shadow, legacy?.baseline, legacy?.shadow, oneSession?.shadow, selected?.baseline, selected?.shadow])

  const {
    chartInteractionRef,
    isPanning,
    visibleEquityRows,
    visibleSpan,
    zoomActive,
    zoomLevel,
    beginPan,
    movePan,
    endPan,
    resetZoom,
  } = usePerformanceZoom({ equityRows, jobId: chartKey })

  const domain = useMemo(() => {
    const keys = mode === 'selected' ? ['baseline', 'selected', ...(calibrated?.shadow?.length ? ['calibrated'] : [])] : ['baseline', 'legacy', 'oneSession']
    const values = visibleEquityRows.flatMap((row) => keys.map((key) => Number(row[key]))).filter(Number.isFinite)
    if (!values.length) return ['auto', 'auto']
    const minimum = Math.min(...values)
    const maximum = Math.max(...values)
    const spread = maximum - minimum
    const padding = Math.max(spread * .08, Math.max(Math.abs(maximum), 1) * .002)
    return [minimum - padding, maximum + padding]
  }, [calibrated?.shadow?.length, mode, visibleEquityRows])

  const action = <div className="analytics-chart-controls">
    <AnalyticsModeTabs
      value={mode}
      onChange={setMode}
      label={tr('Transition shadow chart view')}
      items={[
        { value: 'selected', label: 'Walk-forward' },
        { value: 'research', label: 'Research shadows' },
      ]}
    />
    <span className="analytics-zoom-status">
      {zoomActive ? tr('{level}× · drag to pan', { level: zoomLevel >= 10 ? zoomLevel.toFixed(0) : zoomLevel.toFixed(1) }) : tr('Wheel to zoom')}
    </span>
    {zoomActive ? <button type="button" className="analytics-reset-zoom" onClick={resetZoom}>{tr('Reset')}</button> : null}
  </div>

  return <ChartCell
    kicker={tr('TRANSITION SHADOW')}
    title={tr('Control versus intervention capital')}
    className="winner-intervention-chart-cell"
    action={action}
  >
    {visibleEquityRows.length ? <div
      ref={chartInteractionRef}
      className={`analytics-chart analytics-chart-explorer analytics-interactive-chart ${zoomActive ? 'is-zoomed' : ''} ${isPanning ? 'is-panning' : ''}`}
      onPointerDown={beginPan}
      onPointerMove={movePan}
      onPointerUp={endPan}
      onPointerCancel={endPan}
    >
      <AnalyticsResponsiveContainer fallbackHeight={350}>
        <LineChart data={visibleEquityRows}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="timestamp_value" type="number" domain={['dataMin', 'dataMax']} tickFormatter={(value) => analyticsAxisLabel(value, visibleSpan)} minTickGap={38} />
          <YAxis domain={domain} tickFormatter={(value) => `$${Math.round(value / 1000)}k`} />
          <Tooltip content={<ShadowTooltip mode={mode} hasCalibrated={Boolean(calibrated?.shadow?.length)} />} cursor={{ stroke: 'rgba(147, 177, 210, .45)', strokeDasharray: '4 4' }} />
          <Line type="monotone" dataKey="baseline" name={tr('Control')} dot={false} strokeWidth={2} stroke="var(--accent)" isAnimationActive={false} />
          {mode === 'selected' ? <>{calibrated?.shadow?.length ? <Line type="monotone" dataKey="calibrated" name={tr('Confidence calibrated')} dot={false} strokeWidth={2.7} stroke="var(--positive)" isAnimationActive={false} /> : null}<Line type="monotone" dataKey="selected" name={tr('Walk-forward intervention')} dot={false} strokeWidth={2.1} stroke="var(--warning, #e6b06b)" isAnimationActive={false} /></> : <>
            <Line type="monotone" dataKey="legacy" name={tr('Long risk shadow')} dot={false} strokeWidth={2.4} stroke="var(--positive)" isAnimationActive={false} />
            <Line type="monotone" dataKey="oneSession" name={tr('One-session shadow')} dot={false} strokeWidth={2.1} stroke="var(--warning, #e6b06b)" isAnimationActive={false} />
          </>}
        </LineChart>
      </AnalyticsResponsiveContainer>
    </div> : <ChartEmpty />}
  </ChartCell>
}
