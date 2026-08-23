import { getIntlLocale, tr } from '../../../i18n/runtime'
import { useCallback, useEffect, useMemo, useState } from 'react'

import { ListFilterIcon, TrendDownIcon, TrendUpIcon } from '../../../shared/components/Icons'
import { compactDate, money, percent, shortDate, shortDateTime } from '../../../shared/formatters'
import { METRIC_HINTS, ROTATION_HINTS, ROTATION_PAGE_SIZE } from '../backtestConfig'
import { sortRows, toggleSort } from '../backtestUtils'
import { FilterButton, ListToolbar, Metric, MetricLabel, Pagination, SortableHeader } from './BacktestPrimitives'
import { MonthlyAssetAnalysis } from './MonthlyAssetAnalysis'


const MONTH_TOOLTIP_WIDTH = 318
const MONTH_TOOLTIP_PADDING = 12

function movementTimestamp(value) {
  const timestamp = Date.parse(value || '')
  return Number.isFinite(timestamp) ? timestamp : null
}

function normalizedMovementAsset(value) {
  const asset = String(value || 'CASH').trim().toUpperCase()
  return asset || 'CASH'
}

function monthKeyFromTimestamp(timestamp) {
  const date = new Date(timestamp)
  return `${date.getUTCFullYear()}-${date.getUTCMonth() + 1}`
}

function monthNames() {
  const formatter = new Intl.DateTimeFormat(getIntlLocale(), { month: 'short', timeZone: 'UTC' })
  return Array.from({ length: 12 }, (_, index) => formatter.format(new Date(Date.UTC(2024, index, 1))).replace('.', ''))
}

function fullMonthName(year, month) {
  return new Intl.DateTimeFormat(getIntlLocale(), { month: 'long', year: 'numeric', timeZone: 'UTC' })
    .format(new Date(Date.UTC(year, month - 1, 1)))
}

function compactMoney(value) {
  const number = Number(value)
  if (!Number.isFinite(number)) return '—'
  return new Intl.NumberFormat(getIntlLocale(), {
    style: 'currency',
    currency: 'USD',
    notation: 'compact',
    maximumFractionDigits: 1,
  }).format(number)
}

function monthMovementModel(rotations, equity) {
  const movements = (rotations || [])
    .map((item) => ({ ...item, executedAtValue: movementTimestamp(item.executed_at) }))
    .filter((item) => item.executedAtValue !== null)
    .sort((left, right) => left.executedAtValue - right.executedAtValue)

  const equityRows = (equity || [])
    .map((item) => ({
      ...item,
      timestampValue: movementTimestamp(item.timestamp || item.recorded_at),
      equityValue: Number(item.simulation_equity),
    }))
    .filter((item) => item.timestampValue !== null && Number.isFinite(item.equityValue))
    .sort((left, right) => left.timestampValue - right.timestampValue)

  const months = new Map()
  const ensureMonth = (year, month) => {
    const key = `${year}-${month}`
    if (!months.has(key)) {
      months.set(key, {
        key,
        year,
        month,
        movements: [],
        equityPoints: [],
        totalRealizedPnl: 0,
        totalFees: 0,
        profitableExits: 0,
        losingExits: 0,
        flatExits: 0,
        assetToAsset: 0,
        marketToCash: 0,
        cashToMarket: 0,
        holdingValues: [],
        sessionCount: 0,
        cashSessions: 0,
        boughtCounts: new Map(),
        soldCounts: new Map(),
        bestExit: null,
        worstExit: null,
      })
    }
    return months.get(key)
  }

  for (const movement of movements) {
    const date = new Date(movement.executedAtValue)
    const month = ensureMonth(date.getUTCFullYear(), date.getUTCMonth() + 1)
    month.movements.push(movement)
    const fromAsset = normalizedMovementAsset(movement.from_asset)
    const toAsset = normalizedMovementAsset(movement.to_asset)
    if (fromAsset === 'CASH' && toAsset !== 'CASH') month.cashToMarket += 1
    else if (fromAsset !== 'CASH' && toAsset === 'CASH') month.marketToCash += 1
    else if (fromAsset !== 'CASH' && toAsset !== 'CASH') month.assetToAsset += 1

    if (fromAsset !== 'CASH') month.soldCounts.set(fromAsset, (month.soldCounts.get(fromAsset) || 0) + 1)
    if (toAsset !== 'CASH') month.boughtCounts.set(toAsset, (month.boughtCounts.get(toAsset) || 0) + 1)

    const holding = Number(movement.holding_days)
    if (Number.isFinite(holding)) month.holdingValues.push(holding)

    const pnl = Number(movement.realized_pnl)
    if (movement.realized_pnl !== null && movement.realized_pnl !== undefined && Number.isFinite(pnl)) {
      month.totalRealizedPnl += pnl
      if (pnl > 0) month.profitableExits += 1
      else if (pnl < 0) month.losingExits += 1
      else month.flatExits += 1
      if (!month.bestExit || pnl > Number(month.bestExit.realized_pnl)) month.bestExit = movement
      if (!month.worstExit || pnl < Number(month.worstExit.realized_pnl)) month.worstExit = movement
    }

    const fees = Number(movement.transaction_fees)
    if (Number.isFinite(fees)) month.totalFees += fees
  }

  let currentAsset = normalizedMovementAsset(movements[0]?.from_asset || 'CASH')
  let movementIndex = 0
  for (const row of equityRows) {
    while (movementIndex < movements.length && movements[movementIndex].executedAtValue <= row.timestampValue) {
      currentAsset = normalizedMovementAsset(movements[movementIndex].to_asset)
      movementIndex += 1
    }
    const date = new Date(row.timestampValue)
    const month = ensureMonth(date.getUTCFullYear(), date.getUTCMonth() + 1)
    month.sessionCount += 1
    if (currentAsset === 'CASH') month.cashSessions += 1
    month.equityPoints.push({ timestamp: row.timestampValue, value: row.equityValue })
  }

  const topAsset = (counts) => [...counts.entries()].sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))[0]?.[0] || null
  let maxAbsPnl = 0
  let maxMovements = 0
  let maxCashSessions = 0
  let maxHolding = 0

  for (const month of months.values()) {
    month.movementCount = month.movements.length
    month.averageHolding = month.holdingValues.length
      ? month.holdingValues.reduce((total, value) => total + value, 0) / month.holdingValues.length
      : null
    month.marketExposure = month.sessionCount > 0 ? (month.sessionCount - month.cashSessions) / month.sessionCount : null
    month.topBoughtAsset = topAsset(month.boughtCounts)
    month.topSoldAsset = topAsset(month.soldCounts)
    month.firstEquity = month.equityPoints[0]?.value ?? null
    month.lastEquity = month.equityPoints[month.equityPoints.length - 1]?.value ?? null
    month.equityReturn = month.firstEquity && month.lastEquity != null ? month.lastEquity / month.firstEquity - 1 : null
    maxAbsPnl = Math.max(maxAbsPnl, Math.abs(month.totalRealizedPnl))
    maxMovements = Math.max(maxMovements, month.movementCount)
    maxCashSessions = Math.max(maxCashSessions, month.cashSessions)
    if (month.averageHolding != null) maxHolding = Math.max(maxHolding, month.averageHolding)
  }

  const years = [...new Set([...months.values()].map((item) => item.year))].sort((left, right) => left - right)
  return { months, years, maxAbsPnl, maxMovements, maxCashSessions, maxHolding }
}

function heatmapMetric(month, mode, model) {
  if (mode === 'movements') {
    const ratio = model.maxMovements ? month.movementCount / model.maxMovements : 0
    return { label: String(month.movementCount), tone: 'movements', alpha: .12 + ratio * .64 }
  }
  if (mode === 'cash') {
    const ratio = model.maxCashSessions ? month.cashSessions / model.maxCashSessions : 0
    return { label: `${month.cashSessions}d`, tone: 'cash', alpha: .12 + ratio * .68 }
  }
  if (mode === 'holding') {
    if (month.averageHolding == null) return { label: '—', tone: 'empty', alpha: .08 }
    const ratio = model.maxHolding ? month.averageHolding / model.maxHolding : 0
    return { label: `${month.averageHolding.toFixed(1)}d`, tone: 'holding', alpha: .12 + ratio * .62 }
  }
  const ratio = model.maxAbsPnl ? Math.abs(month.totalRealizedPnl) / model.maxAbsPnl : 0
  return {
    label: compactMoney(month.totalRealizedPnl),
    tone: month.totalRealizedPnl > 0 ? 'positive' : month.totalRealizedPnl < 0 ? 'negative' : 'flat',
    alpha: .12 + ratio * .66,
  }
}


function aggregateHeatmapMetric(items, mode, model) {
  const months = (items || []).filter(Boolean)
  if (!months.length) return { label: '—', tone: 'empty', alpha: .08 }
  if (mode === 'movements') {
    const value = months.reduce((total, month) => total + Number(month.movementCount || 0), 0)
    return { label: String(value), tone: 'movements', alpha: .42 }
  }
  if (mode === 'cash') {
    const value = months.reduce((total, month) => total + Number(month.cashSessions || 0), 0)
    return { label: `${value}d`, tone: 'cash', alpha: .42 }
  }
  if (mode === 'holding') {
    const values = months.flatMap((month) => month.holdingValues || []).map(Number).filter(Number.isFinite)
    if (!values.length) return { label: '—', tone: 'empty', alpha: .08 }
    const average = values.reduce((total, value) => total + value, 0) / values.length
    return { label: `${average.toFixed(1)}d`, tone: 'holding', alpha: .42 }
  }
  const total = months.reduce((sum, month) => sum + Number(month.totalRealizedPnl || 0), 0)
  const ratio = model.maxAbsPnl ? Math.min(1, Math.abs(total) / model.maxAbsPnl) : 0
  return {
    label: compactMoney(total),
    tone: total > 0 ? 'positive' : total < 0 ? 'negative' : 'flat',
    alpha: .18 + ratio * .56,
  }
}

function HeatmapLegend({ mode }) {
  if (mode !== 'pnl') return <div className="rotation-monthly-heatmap-legend compact-legend">
    <span>{tr('Lower intensity')}</span>
    <i className={`rotation-heatmap-swatch ${mode}`} style={{ '--movement-heat-alpha': .18 }} />
    <i className={`rotation-heatmap-swatch ${mode}`} style={{ '--movement-heat-alpha': .42 }} />
    <i className={`rotation-heatmap-swatch ${mode}`} style={{ '--movement-heat-alpha': .74 }} />
    <span>{tr('Higher intensity')}</span>
  </div>
  return <div className="rotation-monthly-heatmap-legend">
    <span>{tr('Higher loss')}</span>
    <i className="rotation-heatmap-swatch negative" style={{ '--movement-heat-alpha': .74 }} />
    <i className="rotation-heatmap-swatch negative" style={{ '--movement-heat-alpha': .24 }} />
    <span>{tr('Near zero')}</span>
    <i className="rotation-heatmap-swatch positive" style={{ '--movement-heat-alpha': .24 }} />
    <i className="rotation-heatmap-swatch positive" style={{ '--movement-heat-alpha': .74 }} />
    <span>{tr('Higher gain')}</span>
    <em>{tr('Color intensity represents the magnitude of realized P/L in the displayed period.')}</em>
  </div>
}

function monthEquityPath(points) {
  if (!points?.length) return null
  const width = 760
  const height = 176
  const paddingX = 12
  const paddingY = 16
  const values = points.map((point) => Number(point.value)).filter(Number.isFinite)
  if (!values.length) return null
  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = max - min || Math.max(1, Math.abs(max) * .01)
  const firstTs = points[0].timestamp
  const lastTs = points[points.length - 1].timestamp
  const span = lastTs - firstTs || 1
  const coordinates = points.map((point) => ({
    timestamp: point.timestamp,
    value: point.value,
    x: paddingX + ((point.timestamp - firstTs) / span) * (width - paddingX * 2),
    y: paddingY + (1 - (point.value - min) / range) * (height - paddingY * 2),
  }))
  const path = coordinates.map((point, index) => `${index ? 'L' : 'M'} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`).join(' ')
  const area = `${path} L ${coordinates[coordinates.length - 1].x.toFixed(2)} ${(height - paddingY).toFixed(2)} L ${coordinates[0].x.toFixed(2)} ${(height - paddingY).toFixed(2)} Z`
  return { width, height, min, max, path, area, coordinates }
}

function MonthlyMovementTooltip({ tooltip }) {
  if (!tooltip) return null
  const style = { left: `${tooltip.left}px`, top: `${tooltip.top}px` }
  return <div className={`rotation-month-tooltip ${tooltip.placement}`} style={style} role="tooltip">
    <div className="rotation-month-tooltip-title">
      <strong>{fullMonthName(tooltip.year, tooltip.month)}</strong>
      <span>{tr('Click for monthly details')}</span>
    </div>
    <div className="rotation-month-tooltip-grid">
      <span>{tr('Realized P/L')}</span><strong className={tooltip.totalRealizedPnl >= 0 ? 'positive' : 'negative'}>{money(tooltip.totalRealizedPnl)}</strong>
      <span>{tr('Capital Movements')}</span><strong>{tooltip.movementCount}</strong>
      <span>{tr('Asset → CASH')}</span><strong className="cash">{tooltip.marketToCash}</strong>
      <span>{tr('CASH → Market')}</span><strong className="positive">{tooltip.cashToMarket}</strong>
      <span>{tr('CASH Sessions')}</span><strong className="cash">{tooltip.cashSessions}</strong>
      <span>{tr('Market Exposure')}</span><strong>{tooltip.marketExposure == null ? '—' : percent(tooltip.marketExposure)}</strong>
      <span>{tr('Profitable Exits')}</span><strong className="positive">{tooltip.profitableExits}</strong>
      <span>{tr('Average Holding')}</span><strong>{tooltip.averageHolding == null ? '—' : tr('{count} days', { count: tooltip.averageHolding.toFixed(1) })}</strong>
    </div>
  </div>
}

function MonthlyMovementDialog({ jobId, processingId = null, month, onClose, allowAssetAnalysis = true }) {
  const chart = useMemo(() => monthEquityPath(month?.equityPoints || []), [month])
  const capitalMarkers = useMemo(() => {
    if (!chart?.coordinates?.length || !month?.movements?.length) return []
    return month.movements.flatMap((movement, index) => {
      const timestamp = movementTimestamp(movement.executed_at)
      if (timestamp === null) return []
      let nearest = null
      let nearestDistance = Number.POSITIVE_INFINITY
      for (const point of chart.coordinates) {
        const distance = Math.abs(point.timestamp - timestamp)
        if (distance < nearestDistance) { nearest = point; nearestDistance = distance }
      }
      if (!nearest) return []
      const fromAsset = normalizedMovementAsset(movement.from_asset)
      const toAsset = normalizedMovementAsset(movement.to_asset)
      return [{ ...nearest, index, timestamp: movement.executed_at, fromAsset, toAsset, label: toAsset === 'CASH' ? 'CASH' : toAsset }]
    })
  }, [chart, month])
  if (!month) return null
  const exits = month.profitableExits + month.losingExits + month.flatExits
  const profitableRate = exits ? month.profitableExits / exits : null
  const hasDecisionContext = month.movements.some((item) => item?.decision_context)

  return <div className="rotation-month-dialog-backdrop" role="presentation" onMouseDown={onClose}>
    <section className="rotation-month-dialog" role="dialog" aria-modal="true" aria-label={fullMonthName(month.year, month.month)} onMouseDown={(event) => event.stopPropagation()}>
      <header className="rotation-month-dialog-header">
        <div>
          <span className="panel-kicker">{tr('Monthly capital movements')}</span>
          <h3>{fullMonthName(month.year, month.month)}</h3>
        </div>
        <button type="button" className="rotation-month-dialog-close" onClick={onClose} aria-label={tr('Close')}>×</button>
      </header>

      <div className="rotation-month-dialog-metrics">
        <div><span>{tr('Realized P/L')}</span><strong className={month.totalRealizedPnl >= 0 ? 'positive' : 'negative'}>{money(month.totalRealizedPnl)}</strong></div>
        <div><span>{tr('Capital Movements')}</span><strong>{month.movementCount}</strong></div>
        <div><span>{tr('Profitable exit rate')}</span><strong className="positive">{profitableRate == null ? '—' : percent(profitableRate)}</strong></div>
        <div><span>{tr('CASH Sessions')}</span><strong className="cash">{month.cashSessions}</strong></div>
        <div><span>{tr('Market Exposure')}</span><strong>{month.marketExposure == null ? '—' : percent(month.marketExposure)}</strong></div>
        <div><span>{tr('Average Holding')}</span><strong>{month.averageHolding == null ? '—' : tr('{count} days', { count: month.averageHolding.toFixed(1) })}</strong></div>
      </div>

      <div className="rotation-month-dialog-main">
        <article className="rotation-month-equity-card">
          <div className="rotation-month-dialog-section-title">
            <div><span>{tr('Capital during the month')}</span><strong>{month.firstEquity == null ? '—' : money(month.firstEquity)} → {month.lastEquity == null ? '—' : money(month.lastEquity)}</strong></div>
            <span className={month.equityReturn == null ? '' : month.equityReturn >= 0 ? 'positive' : 'negative'}>{month.equityReturn == null ? '—' : percent(month.equityReturn)}</span>
          </div>
          {chart ? <svg className="rotation-month-equity-chart" viewBox={`0 0 ${chart.width} ${chart.height}`} role="img" aria-label={tr('Monthly capital curve')}>
            <line x1="12" x2="748" y1="44" y2="44" />
            <line x1="12" x2="748" y1="88" y2="88" />
            <line x1="12" x2="748" y1="132" y2="132" />
            <path className="area" d={chart.area} />
            <path className="line" d={chart.path} />
            {capitalMarkers.map((marker) => <g key={`${marker.timestamp}-${marker.index}`} className={`capital-movement-marker ${marker.toAsset === 'CASH' ? 'cash' : 'market'}`}>
              <circle cx={marker.x} cy={marker.y} r="5.5" />
              <text x={marker.x} y={marker.y - 9}>{marker.label}</text>
              <title>{`${shortDateTime(marker.timestamp)} · ${marker.fromAsset} → ${marker.toAsset}`}</title>
            </g>)}
          </svg> : <div className="rotation-month-chart-empty">{tr('No equity observations for this month.')}</div>}
          <div className="rotation-month-equity-range"><span>{compactMoney(chart?.min)}</span><span>{compactMoney(chart?.max)}</span></div>
        </article>

        <aside className="rotation-month-insights">
          <div><span>{tr('Asset → Asset')}</span><strong>{month.assetToAsset}</strong></div>
          <div><span>{tr('Asset → CASH')}</span><strong className="cash">{month.marketToCash}</strong></div>
          <div><span>{tr('CASH → Market')}</span><strong className="positive">{month.cashToMarket}</strong></div>
          <div><span>{tr('Transaction fees')}</span><strong>{money(month.totalFees)}</strong></div>
          <div><span>{tr('Top bought asset')}</span><strong>{month.topBoughtAsset || '—'}</strong></div>
          <div><span>{tr('Top sold asset')}</span><strong>{month.topSoldAsset || '—'}</strong></div>
          <div><span>{tr('Best exit')}</span><strong className="positive">{month.bestExit ? `${normalizedMovementAsset(month.bestExit.from_asset)} · ${money(month.bestExit.realized_pnl)}` : '—'}</strong></div>
          <div><span>{tr('Worst exit')}</span><strong className="negative">{month.worstExit ? `${normalizedMovementAsset(month.worstExit.from_asset)} · ${money(month.worstExit.realized_pnl)}` : '—'}</strong></div>
        </aside>
      </div>

      {allowAssetAnalysis ? <MonthlyAssetAnalysis jobId={jobId} processingId={processingId} month={month} /> : null}

      <div className="rotation-month-dialog-table-wrap">
        <div className="rotation-month-dialog-section-title"><div><span>{tr('Capital movements')}</span><strong>{tr('{count} movements', { count: month.movementCount })}</strong></div></div>
        <div className="table-wrap">
          <table className="dashboard-table rotation-month-dialog-table">
            <thead><tr><th>{tr('Executed')}</th><th>{tr('Sold')}</th><th>{tr('Bought')}</th><th>{tr('Holding')}</th><th>{tr('Position Return')}</th><th>{tr('Realized P/L')}</th><th>{tr('Fees')}</th>{hasDecisionContext ? <th>{tr('Decision context')}</th> : null}</tr></thead>
            <tbody>{month.movements.length ? month.movements.map((item, index) => {
              const context = item?.decision_context
              const gap = context?.winner_top1_top2_score_gap ?? context?.top1_top2_asset_rank_gap
              return <tr key={`${item.executed_at}-${index}`}>
              <td>{shortDateTime(item.executed_at)}</td>
              <td><span className={`rotation-asset from ${normalizedMovementAsset(item.from_asset) === 'CASH' ? 'cash' : ''}`}>{item.from_asset || 'CASH'}</span></td>
              <td><span className={`rotation-asset to ${normalizedMovementAsset(item.to_asset) === 'CASH' ? 'cash' : ''}`}>{item.to_asset || 'CASH'}</span></td>
              <td>{item.holding_days == null ? '—' : tr('{count} days', { count: Number(item.holding_days).toFixed(0) })}</td>
              <td className={item.position_return == null ? '' : Number(item.position_return) >= 0 ? 'positive' : 'negative'}>{percent(item.position_return)}</td>
              <td className={item.realized_pnl == null ? '' : Number(item.realized_pnl) >= 0 ? 'positive' : 'negative'}>{money(item.realized_pnl)}</td>
              <td>{money(item.transaction_fees)}</td>
              {hasDecisionContext ? <td className="rotation-decision-context-cell">{context ? <>
                <strong>{context.action || '—'} · {context.reason || '—'}</strong>
                <span>Top-1 {context.top1?.symbol || '—'} · Top-2 {context.top2?.symbol || '—'}</span>
                <small>{tr('Top-1 / Top-2 gap')}: {gap == null ? '—' : Number(gap).toFixed(4)}</small>
              </> : '—'}</td> : null}
            </tr>}) : <tr><td colSpan={hasDecisionContext ? 8 : 7} className="empty-cell">{tr('No capital movements in this month.')}</td></tr>}</tbody>
          </table>
        </div>
      </div>
    </section>
  </div>
}

export function MonthlyCapitalMovementHeatmap({
  jobId,
  processingId = null,
  rotations,
  equity,
  allowDrilldown = true,
  allowAssetAnalysis = true,
  seriesOptions = null,
  defaultSeriesKey = null,
}) {
  const [mode, setMode] = useState('pnl')
  const [tooltip, setTooltip] = useState(null)
  const [selectedMonth, setSelectedMonth] = useState(null)
  const availableSeries = useMemo(() => {
    const supplied = Array.isArray(seriesOptions)
      ? seriesOptions.filter((item) => item && Array.isArray(item.rotations) && Array.isArray(item.equity) && item.equity.length)
      : []
    if (supplied.length) return supplied
    return [{
      key: 'default',
      label: null,
      rotations: Array.isArray(rotations) ? rotations : [],
      equity: Array.isArray(equity) ? equity : [],
      allowDrilldown,
      allowAssetAnalysis,
    }]
  }, [allowAssetAnalysis, allowDrilldown, equity, rotations, seriesOptions])
  const preferredSeriesKey = defaultSeriesKey && availableSeries.some((item) => item.key === defaultSeriesKey)
    ? defaultSeriesKey
    : (availableSeries[0]?.key || '')
  const [seriesKey, setSeriesKey] = useState(preferredSeriesKey)

  useEffect(() => {
    if (!availableSeries.some((item) => item.key === seriesKey)) setSeriesKey(preferredSeriesKey)
  }, [availableSeries, preferredSeriesKey, seriesKey])

  const selectedSeries = availableSeries.find((item) => item.key === seriesKey) || availableSeries[0]
  const selectedRotations = selectedSeries?.rotations || []
  const selectedEquity = selectedSeries?.equity || []
  const selectedAllowDrilldown = selectedSeries?.allowDrilldown ?? allowDrilldown
  const selectedAllowAssetAnalysis = selectedSeries?.allowAssetAnalysis ?? allowAssetAnalysis
  const model = useMemo(() => monthMovementModel(selectedRotations, selectedEquity), [selectedEquity, selectedRotations])
  const months = useMemo(monthNames, [])

  useEffect(() => {
    setTooltip(null)
    setSelectedMonth(null)
  }, [seriesKey])

  const hideTooltip = useCallback(() => setTooltip(null), [])
  const showTooltip = useCallback((event, month) => {
    if (!month || typeof window === 'undefined') return
    const rect = event.currentTarget.getBoundingClientRect()
    const preferredLeft = rect.left + rect.width / 2 - MONTH_TOOLTIP_WIDTH / 2
    const left = Math.min(window.innerWidth - MONTH_TOOLTIP_WIDTH - MONTH_TOOLTIP_PADDING, Math.max(MONTH_TOOLTIP_PADDING, preferredLeft))
    const showAbove = rect.top > 290
    setTooltip({
      ...month,
      left,
      top: showAbove ? rect.top - 10 : rect.bottom + 10,
      placement: showAbove ? 'above' : 'below',
    })
  }, [])

  useEffect(() => {
    if (!selectedMonth) return undefined
    const handler = (event) => { if (event.key === 'Escape') setSelectedMonth(null) }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [selectedMonth])

  if (!model.years.length) return null

  const options = [
    ['pnl', tr('Realized P/L')],
    ['movements', tr('Movements')],
    ['cash', 'CASH'],
    ['holding', tr('Holding')],
  ]

  return <>
    <article className="rotation-monthly-heatmap-panel">
      <div className="rotation-monthly-heatmap-heading">
        <div>
          <MetricLabel
            id="hint-monthly-capital-movement-heatmap"
            label={tr('Monthly Capital Movement Heatmap')}
            hint={tr('Summarizes the operational behavior of this backtest by month. Change the view to compare realized P/L, movement frequency, CASH sessions or average holding. Hover for a detailed summary and click a month to inspect its capital curve and movements.')}
          />
        </div>
        <div className="rotation-monthly-heatmap-heading-actions">
          {availableSeries.length > 1 ? <label className="rotation-monthly-series-selector">
            <span>{tr('Result series')}</span>
            <select value={seriesKey} onChange={(event) => setSeriesKey(event.target.value)}>
              {availableSeries.map((item) => <option key={item.key} value={item.key}>{item.label || item.key}</option>)}
            </select>
          </label> : null}
          <div className="rotation-monthly-heatmap-modes" role="group" aria-label={tr('Heatmap metric')}>
            {options.map(([key, label]) => <button key={key} type="button" className={mode === key ? 'active' : ''} onClick={() => setMode(key)}>{label}</button>)}
          </div>
        </div>
      </div>

      <div className="rotation-monthly-heatmap" role="grid" aria-label={tr('Monthly Capital Movement Heatmap')} onMouseLeave={hideTooltip}>
        <div className="rotation-monthly-heatmap-head" aria-hidden="true"><span />{months.map((name) => <span key={name}>{name}</span>)}<span className="summary-heading">{tr('Year total')}</span></div>
        {model.years.map((year) => {
          const yearMonths = months.map((monthName, index) => model.months.get(`${year}-${index + 1}`)).filter(Boolean)
          const yearMetric = aggregateHeatmapMetric(yearMonths, mode, model)
          return <div className="rotation-monthly-heatmap-row" key={year}>
          <strong>{year}</strong>
          {months.map((monthName, index) => {
            const month = model.months.get(`${year}-${index + 1}`)
            if (!month || !month.sessionCount && !month.movementCount) return <span key={`${year}-${index}`} className="rotation-monthly-heatmap-cell empty" role="gridcell">—</span>
            const metric = heatmapMetric(month, mode, model)
            return <button
              key={`${year}-${index}`}
              type="button"
              role="gridcell"
              className={`rotation-monthly-heatmap-cell ${metric.tone}`}
              style={{ '--movement-heat-alpha': Math.min(.82, metric.alpha) }}
              onMouseEnter={(event) => showTooltip(event, month)}
              onFocus={(event) => showTooltip(event, month)}
              onBlur={hideTooltip}
              onClick={selectedAllowDrilldown ? () => { hideTooltip(); setSelectedMonth(month) } : undefined}
              aria-label={`${fullMonthName(year, index + 1)}. ${metric.label}`}
            >{metric.label}</button>
          })}
          <span className={`rotation-monthly-heatmap-cell summary ${yearMetric.tone}`} style={{ '--movement-heat-alpha': Math.min(.82, yearMetric.alpha) }} role="gridcell">{yearMetric.label}</span>
        </div>})}
        <div className="rotation-monthly-heatmap-row totals-row">
          <strong>{tr('Total')}</strong>
          {months.map((monthName, index) => {
            const monthItems = model.years.map((year) => model.months.get(`${year}-${index + 1}`)).filter(Boolean)
            const metric = aggregateHeatmapMetric(monthItems, mode, model)
            return <span key={`total-${index}`} className={`rotation-monthly-heatmap-cell summary ${metric.tone}`} style={{ '--movement-heat-alpha': Math.min(.82, metric.alpha) }} role="gridcell">{metric.label}</span>
          })}
          {(() => {
            const metric = aggregateHeatmapMetric([...model.months.values()], mode, model)
            return <span className={`rotation-monthly-heatmap-cell summary grand-total ${metric.tone}`} style={{ '--movement-heat-alpha': Math.min(.82, metric.alpha) }} role="gridcell">{metric.label}</span>
          })()}
        </div>
      </div>
      <HeatmapLegend mode={mode} />
      <div className="rotation-monthly-heatmap-footer"><span>{tr('Hover for summary')}</span>{selectedAllowDrilldown ? <><span>·</span><span>{tr('Click a month for detailed analysis')}</span></> : null}</div>
    </article>
    <MonthlyMovementTooltip tooltip={tooltip} />
    {selectedAllowDrilldown ? <MonthlyMovementDialog jobId={jobId} processingId={processingId} month={selectedMonth} onClose={() => setSelectedMonth(null)} allowAssetAnalysis={selectedAllowAssetAnalysis} /> : null}
  </>
}

export function RotationPanel({ jobId, payload, loading, error }) {
  const [query, setQuery] = useState('')
  const [outcome, setOutcome] = useState('all')
  const [sort, setSort] = useState({ key: 'executed_at', direction: 'desc' })
  const [page, setPage] = useState(1)

  const summary = payload?.rotation_summary || {}
  const metrics = payload?.metrics || {}
  const rotations = payload?.rotations || []

  const exitCount = Number(summary.profitable_rotations || 0) + Number(summary.losing_rotations || 0) + Number(summary.flat_rotations || 0)
  const profitableExitRate = exitCount > 0 ? Number(summary.profitable_rotations || 0) / exitCount : null

  const realizedPnlBreakdown = useMemo(() => {
    const realized = rotations
      .map((item) => item.realized_pnl)
      .filter((value) => value !== null && value !== undefined && Number.isFinite(Number(value)))
      .map(Number)
    const grossProfits = realized.filter((value) => value > 0).reduce((total, value) => total + value, 0)
    const grossLosses = realized.filter((value) => value < 0).reduce((total, value) => total + value, 0)
    return {
      grossProfits,
      grossLosses,
      averageRealizedPnl: realized.length ? realized.reduce((total, value) => total + value, 0) / realized.length : null,
    }
  }, [rotations])

  const holdingStats = useMemo(() => {
    const values = rotations
      .map((item) => item.holding_days)
      .filter((value) => value !== null && value !== undefined && Number.isFinite(Number(value)))
      .map(Number)
      .sort((a, b) => a - b)
    if (!values.length) return { median: null, minimum: null, maximum: null }
    const middle = Math.floor(values.length / 2)
    const median = values.length % 2
      ? values[middle]
      : (values[middle - 1] + values[middle]) / 2
    return { median, minimum: values[0], maximum: values[values.length - 1] }
  }, [rotations])

  const filteredRows = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase()
    const rows = rotations.filter((item) => {
      if (normalizedQuery) {
        const haystack = `${item.from_asset || 'CASH'} ${item.to_asset || 'CASH'}`.toLowerCase()
        if (!haystack.includes(normalizedQuery)) return false
      }
      const hasRealizedPnl = item.realized_pnl !== null && item.realized_pnl !== undefined
      const pnl = hasRealizedPnl ? Number(item.realized_pnl) : null
      if (outcome === 'profit' && (!hasRealizedPnl || pnl <= 0)) return false
      if (outcome === 'loss' && (!hasRealizedPnl || pnl >= 0)) return false
      if (outcome === 'flat' && (!hasRealizedPnl || pnl !== 0)) return false
      return true
    })

    return sortRows(rows, sort, {
      executed_at: (item) => Date.parse(item.executed_at || '') || 0,
      from_asset: (item) => String(item.from_asset || 'CASH'),
      to_asset: (item) => String(item.to_asset || 'CASH'),
      holding_days: (item) => item.holding_days == null ? null : Number(item.holding_days),
      position_return: (item) => item.position_return == null ? null : Number(item.position_return),
      realized_pnl: (item) => item.realized_pnl == null ? null : Number(item.realized_pnl),
      transaction_fees: (item) => item.transaction_fees == null ? null : Number(item.transaction_fees),
    })
  }, [outcome, query, rotations, sort])

  const pages = Math.max(1, Math.ceil(filteredRows.length / ROTATION_PAGE_SIZE))
  const currentPage = Math.min(page, pages)
  const paginatedRows = filteredRows.slice((currentPage - 1) * ROTATION_PAGE_SIZE, currentPage * ROTATION_PAGE_SIZE)

  useEffect(() => { setPage(1) }, [jobId, outcome, query, sort])

  if (loading) {
    return <section className="backtest-workspace-section backtest-loading-row">{tr("Loading capital rotations…")}</section>
  }

  if (error) {
    return <section className="backtest-workspace-section rotation-error"><strong>{tr("Unable to load capital rotations")}</strong><span>{tr(error)}</span></section>
  }

  return (
    <section className="backtest-workspace-section rotation-workspace-section">
      <div className="backtest-section-heading">
        <div><span className="panel-kicker">{tr("Backtest analytics")}</span><h2>{tr("Capital Rotations")}</h2></div>
        <span className="backtest-section-meta">{tr("Executed capital movements · includes CASH")}</span>
      </div>

      <div className="backtest-rotation-summary">
        <Metric
          id="hint-rotation-count"
          label={tr("Capital Movements")}
          value={String(summary.total_rotations ?? 0)}
          tone="blue"
          hint={METRIC_HINTS.total_rotations}
          hintDetails={[
            { label: 'Total movements', value: String(summary.total_rotations ?? 0), tone: 'blue', description: 'All executed capital movements shown in this table.' },
            { label: 'Asset → Asset', value: String(summary.asset_to_asset_rotations ?? 0), tone: 'amber', description: 'Direct rotations from one risky asset into another risky asset.' },
            { label: 'Asset → CASH', value: String(summary.market_to_cash_moves ?? 0), tone: 'purple', description: 'Times the strategy exited a risky asset and left the capital in CASH.' },
            { label: 'CASH → Market', value: String(summary.cash_to_market_moves ?? 0), tone: 'green', description: 'Times the strategy redeployed CASH into a risky asset.' },
            { label: 'CASH Sessions', value: metrics.cash_days == null ? '—' : String(Math.round(Number(metrics.cash_days))), tone: 'purple', description: 'Out-of-sample sessions in which the portfolio remained fully in CASH.' },
            { label: 'Market Exposure', value: metrics.market_exposure == null ? '—' : percent(metrics.market_exposure), tone: 'blue', description: 'Average fraction of portfolio capital exposed to risky assets during the out-of-sample period.' },
          ]}
        />
        <Metric
          id="hint-profitable-rotations"
          label={tr("Profitable Exits")}
          value={String(summary.profitable_rotations ?? 0)}
          tone="green"
          hint={METRIC_HINTS.profitable_rotations}
          hintDetails={[
            { label: 'Profitable exits', value: String(summary.profitable_rotations ?? 0), tone: 'green', description: 'Closed positions with realized P/L greater than zero.' },
            { label: 'Losing exits', value: String(summary.losing_rotations ?? 0), tone: 'red', description: 'Closed positions with realized P/L below zero.' },
            { label: 'Flat exits', value: String(summary.flat_rotations ?? 0), tone: 'blue', description: 'Closed positions whose realized P/L was exactly zero.' },
            { label: 'Profitable exit rate', value: profitableExitRate == null ? '—' : percent(profitableExitRate), tone: 'green', description: 'Profitable exits divided by all exits with a realized outcome.' },
          ]}
        />
        <Metric
          id="hint-realized-pnl"
          label={tr("Realized P/L")}
          value={money(summary.total_realized_pnl)}
          tone={Number(summary.total_realized_pnl || 0) >= 0 ? 'green' : 'red'}
          hint={METRIC_HINTS.total_realized_pnl}
          hintDetails={[
            { label: 'Realized P/L', value: money(summary.total_realized_pnl), tone: Number(summary.total_realized_pnl || 0) >= 0 ? 'green' : 'red', description: 'Total realized result recorded when positions were closed by capital movements.' },
            { label: 'Gross profitable exits', value: money(realizedPnlBreakdown.grossProfits), tone: 'green', description: 'Sum of positive realized P/L across closed positions.' },
            { label: 'Gross losing exits', value: money(realizedPnlBreakdown.grossLosses), tone: 'red', description: 'Sum of negative realized P/L across closed positions.' },
            { label: 'Average P/L per exit', value: realizedPnlBreakdown.averageRealizedPnl == null ? '—' : money(realizedPnlBreakdown.averageRealizedPnl), tone: realizedPnlBreakdown.averageRealizedPnl == null ? 'blue' : realizedPnlBreakdown.averageRealizedPnl >= 0 ? 'green' : 'red', description: 'Average realized P/L for movements that closed an existing position.' },
            { label: 'Transaction fees', value: money(summary.total_transaction_fees), tone: 'amber', description: 'Total modeled transaction costs attributed to the executed capital movements.' },
          ]}
        />
        <Metric
          id="hint-average-holding"
          label={tr("Average Holding")}
          value={summary.average_holding_days == null ? '—' : tr('{count} days', { count: Number(summary.average_holding_days).toFixed(1) })}
          tone="purple"
          hint={METRIC_HINTS.average_holding_days}
          hintDetails={[
            { label: 'Average holding', value: summary.average_holding_days == null ? '—' : tr('{count} days', { count: Number(summary.average_holding_days).toFixed(1) }), tone: 'purple', description: 'Mean holding period for positions that were subsequently exited.' },
            { label: 'Median holding', value: holdingStats.median == null ? '—' : tr('{count} days', { count: Number(holdingStats.median).toFixed(1) }), tone: 'blue', description: 'Middle holding period, which is less sensitive to unusually long positions.' },
            { label: 'Shortest holding', value: holdingStats.minimum == null ? '—' : tr('{count} days', { count: Number(holdingStats.minimum).toFixed(0) }), tone: 'amber', description: 'Shortest completed holding period in the movement history.' },
            { label: 'Longest holding', value: holdingStats.maximum == null ? '—' : tr('{count} days', { count: Number(holdingStats.maximum).toFixed(0) }), tone: 'purple', description: 'Longest completed holding period in the movement history.' },
            { label: 'Last capital movement', value: summary.last_rotation_at ? shortDateTime(summary.last_rotation_at) : '—', tone: 'blue', description: 'Timestamp of the most recent executed capital movement in this backtest.' },
          ]}
        />
      </div>


      <ListToolbar
        query={query}
        onQueryChange={setQuery}
        placeholder={tr("Filter by sold or bought asset")}
        resultCount={filteredRows.length}
        resultLabel={filteredRows.length === 1 ? 'movement' : 'movements'}
      >
        <FilterButton active={outcome === 'all'} label={tr("All")} onClick={() => setOutcome('all')}><ListFilterIcon size={14} /></FilterButton>
        <FilterButton active={outcome === 'profit'} label={tr("Profit")} tone="positive" onClick={() => setOutcome('profit')}><TrendUpIcon size={14} /></FilterButton>
        <FilterButton active={outcome === 'loss'} label={tr("Loss")} tone="negative" onClick={() => setOutcome('loss')}><TrendDownIcon size={14} /></FilterButton>
        <FilterButton active={outcome === 'flat'} label={tr("Flat")} onClick={() => setOutcome('flat')} />
      </ListToolbar>

      <div className="table-wrap backtest-table-wrap">
        <table className="dashboard-table rotation-table backtest-sortable-table">
          <thead>
            <tr>
              <SortableHeader label={tr("Executed")} field="executed_at" sort={sort} onSort={(key) => setSort((current) => toggleSort(current, key))} hint={ROTATION_HINTS.executed_at} />
              <SortableHeader label={tr("Sold")} field="from_asset" sort={sort} onSort={(key) => setSort((current) => toggleSort(current, key))} hint={ROTATION_HINTS.from_asset} />
              <SortableHeader label={tr("Bought")} field="to_asset" sort={sort} onSort={(key) => setSort((current) => toggleSort(current, key))} hint={ROTATION_HINTS.to_asset} />
              <SortableHeader label={tr("Holding")} field="holding_days" sort={sort} onSort={(key) => setSort((current) => toggleSort(current, key))} hint={ROTATION_HINTS.holding_days} />
              <SortableHeader label={tr("Position Return")} field="position_return" sort={sort} onSort={(key) => setSort((current) => toggleSort(current, key))} hint={ROTATION_HINTS.position_return} />
              <SortableHeader label={tr("Realized P/L")} field="realized_pnl" sort={sort} onSort={(key) => setSort((current) => toggleSort(current, key))} hint={ROTATION_HINTS.realized_pnl} />
              <SortableHeader label={tr("Fees")} field="transaction_fees" sort={sort} onSort={(key) => setSort((current) => toggleSort(current, key))} hint={ROTATION_HINTS.transaction_fees} />
            </tr>
          </thead>
          <tbody>
            {paginatedRows.length ? paginatedRows.map((item, index) => (
              <tr key={`${item.executed_at || 'rotation'}-${item.from_asset}-${item.to_asset}-${index}`}>
                <td>{shortDateTime(item.executed_at)}</td>
                <td><span className={`rotation-asset from ${String(item.from_asset || 'CASH').toUpperCase() === 'CASH' ? 'cash' : ''}`}>{item.from_asset || 'CASH'}</span></td>
                <td><span className={`rotation-asset to ${String(item.to_asset || 'CASH').toUpperCase() === 'CASH' ? 'cash' : ''}`}>{item.to_asset || 'CASH'}</span></td>
                <td>{item.holding_days == null ? '—' : tr('{count} days', { count: Number(item.holding_days).toFixed(0) })}</td>
                <td className={item.position_return == null ? '' : Number(item.position_return) >= 0 ? 'positive' : 'negative'}>{percent(item.position_return)}</td>
                <td className={item.realized_pnl == null ? '' : Number(item.realized_pnl) >= 0 ? 'positive' : 'negative'}>{money(item.realized_pnl)}</td>
                <td>{money(item.transaction_fees)}</td>
              </tr>
            )) : <tr><td colSpan="7" className="empty-cell">{tr("No capital rotations match the selected filters.")}</td></tr>}
          </tbody>
        </table>
      </div>
      <Pagination page={currentPage} pages={pages} total={filteredRows.length} pageSize={ROTATION_PAGE_SIZE} onPageChange={setPage} />
    </section>
  )
}
