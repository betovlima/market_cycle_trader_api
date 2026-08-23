import { useEffect, useMemo, useState } from 'react'

import { apiFetch } from '../../../api/http'
import { API } from '../../../config/env'
import { tr } from '../../../i18n/runtime'
import { money, percent, shortDate } from '../../../shared/formatters'

const CHART_WIDTH = 1000
const CHART_HEIGHT = 250
const CHART_PADDING = { left: 54, right: 18, top: 18, bottom: 30 }

function timestampValue(value) {
  const timestamp = Date.parse(value || '')
  return Number.isFinite(timestamp) ? timestamp : null
}

function finiteNumber(value) {
  const number = Number(value)
  return Number.isFinite(number) ? number : null
}

function normalizedAsset(value) {
  const asset = String(value || 'CASH').trim().toUpperCase()
  return asset || 'CASH'
}

function extent(values) {
  const finite = values.map(finiteNumber).filter((value) => value !== null)
  if (!finite.length) return null
  const minimum = Math.min(...finite)
  const maximum = Math.max(...finite)
  const range = maximum - minimum || Math.max(Math.abs(maximum) * .02, 1)
  return { minimum: minimum - range * .08, maximum: maximum + range * .08 }
}

function chartCoordinates(points, valueKey, domain) {
  if (!points?.length || !domain) return []
  const firstTimestamp = domain.firstTimestamp
  const lastTimestamp = domain.lastTimestamp
  const timeSpan = lastTimestamp - firstTimestamp || 1
  const valueSpan = domain.maximum - domain.minimum || 1
  const innerWidth = CHART_WIDTH - CHART_PADDING.left - CHART_PADDING.right
  const innerHeight = CHART_HEIGHT - CHART_PADDING.top - CHART_PADDING.bottom
  return points.flatMap((point) => {
    const timestamp = timestampValue(point.timestamp)
    const value = finiteNumber(point[valueKey])
    if (timestamp === null || value === null) return []
    return [{
      ...point,
      timestampValue: timestamp,
      chartValue: value,
      x: CHART_PADDING.left + ((timestamp - firstTimestamp) / timeSpan) * innerWidth,
      y: CHART_PADDING.top + (1 - (value - domain.minimum) / valueSpan) * innerHeight,
    }]
  })
}

function svgPath(points) {
  return points.map((point, index) => `${index ? 'L' : 'M'} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`).join(' ')
}

function nearestPrice(prices, timestamp) {
  if (!prices?.length || timestamp === null) return null
  let best = null
  let bestDistance = Number.POSITIVE_INFINITY
  for (const row of prices) {
    const currentTimestamp = timestampValue(row.timestamp)
    const close = finiteNumber(row.close)
    if (currentTimestamp === null || close === null) continue
    const distance = Math.abs(currentTimestamp - timestamp)
    if (distance < bestDistance) {
      best = close
      bestDistance = distance
    }
  }
  return best
}

function indexedSeries(rows, valueKey) {
  const clean = (rows || []).flatMap((row) => {
    const value = finiteNumber(row[valueKey])
    const timestamp = timestampValue(row.timestamp)
    return value === null || timestamp === null ? [] : [{ timestamp: row.timestamp, value }]
  })
  const base = clean.find((row) => row.value !== 0)?.value
  if (!base) return []
  return clean.map((row) => ({ timestamp: row.timestamp, value: row.value / base * 100 }))
}

function formatPrice(value) {
  const number = finiteNumber(value)
  if (number === null) return '—'
  return new Intl.NumberFormat(undefined, { style: 'currency', currency: 'USD', maximumFractionDigits: 2 }).format(number)
}

function eventRows(movements, asset, prices) {
  const rows = []
  for (const movement of movements || []) {
    const fromAsset = normalizedAsset(movement.from_asset)
    const toAsset = normalizedAsset(movement.to_asset)
    const timestamp = timestampValue(movement.executed_at)
    if (timestamp === null) continue
    if (toAsset === asset) {
      rows.push({
        timestamp: movement.executed_at,
        kind: fromAsset === 'CASH' ? 'BUY' : 'ROTATE_IN',
        price: finiteNumber(movement.buy_execution_price) ?? nearestPrice(prices, timestamp),
        fromAsset,
        toAsset,
      })
    }
    if (fromAsset === asset) {
      rows.push({
        timestamp: movement.executed_at,
        kind: toAsset === 'CASH' ? 'SELL' : 'ROTATE_OUT',
        price: finiteNumber(movement.sell_execution_price) ?? nearestPrice(prices, timestamp),
        fromAsset,
        toAsset,
      })
    }
  }
  return rows
}

function PriceAndTradeChart({ detail, asset }) {
  const prices = asset?.prices || []
  const events = useMemo(() => eventRows(detail?.movements, asset?.symbol, prices), [asset?.symbol, detail?.movements, prices])
  const model = useMemo(() => {
    const timestamps = prices.map((row) => timestampValue(row.timestamp)).filter((value) => value !== null)
    if (!timestamps.length) return null
    const values = [...prices.map((row) => row.close), ...events.map((row) => row.price)]
    const valueExtent = extent(values)
    if (!valueExtent) return null
    const domain = {
      firstTimestamp: Math.min(...timestamps),
      lastTimestamp: Math.max(...timestamps),
      ...valueExtent,
    }
    const points = chartCoordinates(prices, 'close', domain)
    const eventPoints = chartCoordinates(events.map((event) => ({ ...event, value: event.price })), 'value', domain)
    return { domain, points, eventPoints }
  }, [events, prices])

  if (!model?.points?.length) return <div className="rotation-asset-analysis-empty">{tr('No stored market prices are available for this asset in the selected month.')}</div>

  const firstPoint = model.points[0]
  const lastPoint = model.points[model.points.length - 1]
  const timeSpan = model.domain.lastTimestamp - model.domain.firstTimestamp || 1
  const innerWidth = CHART_WIDTH - CHART_PADDING.left - CHART_PADDING.right
  const heldSegments = (detail?.position_segments || []).filter((segment) => normalizedAsset(segment.asset) === asset.symbol)

  return <div className="rotation-asset-price-chart-wrap">
    <svg className="rotation-asset-price-chart" viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`} role="img" aria-label={`${asset.symbol} · ${tr('Price and trade events')}`}>
      {[.25, .5, .75].map((ratio) => {
        const y = CHART_PADDING.top + ratio * (CHART_HEIGHT - CHART_PADDING.top - CHART_PADDING.bottom)
        return <line key={ratio} className="grid" x1={CHART_PADDING.left} x2={CHART_WIDTH - CHART_PADDING.right} y1={y} y2={y} />
      })}
      {heldSegments.map((segment, index) => {
        const start = timestampValue(segment.start_at)
        const end = timestampValue(segment.end_at)
        if (start === null || end === null) return null
        const x = CHART_PADDING.left + ((Math.max(start, model.domain.firstTimestamp) - model.domain.firstTimestamp) / timeSpan) * innerWidth
        const x2 = CHART_PADDING.left + ((Math.min(end, model.domain.lastTimestamp) - model.domain.firstTimestamp) / timeSpan) * innerWidth
        return x2 > x ? <rect key={`${segment.start_at}-${index}`} className="held-period" x={x} y={CHART_PADDING.top} width={x2 - x} height={CHART_HEIGHT - CHART_PADDING.top - CHART_PADDING.bottom} /> : null
      })}
      <path className="price-line" d={svgPath(model.points)} />
      {model.eventPoints.map((point, index) => {
        const entry = point.kind === 'BUY' || point.kind === 'ROTATE_IN'
        const label = point.kind === 'BUY' ? tr('BUY') : point.kind === 'SELL' ? tr('SELL') : point.kind === 'ROTATE_IN' ? tr('ROTATE IN') : tr('ROTATE OUT')
        return <g key={`${point.timestamp}-${point.kind}-${index}`} className={`trade-marker ${entry ? 'entry' : 'exit'}`}>
          <circle cx={point.x} cy={point.y} r="8" />
          <text x={point.x} y={point.y + 3}>{entry ? 'B' : 'S'}</text>
          <title>{`${shortDate(point.timestamp)} · ${label} · ${formatPrice(point.price)} · ${point.fromAsset} → ${point.toAsset}`}</title>
        </g>
      })}
      <text className="axis-label y top" x="4" y={CHART_PADDING.top + 5}>{formatPrice(model.domain.maximum)}</text>
      <text className="axis-label y bottom" x="4" y={CHART_HEIGHT - CHART_PADDING.bottom}>{formatPrice(model.domain.minimum)}</text>
      <text className="axis-label x" x={CHART_PADDING.left} y={CHART_HEIGHT - 8}>{shortDate(firstPoint.timestamp)}</text>
      <text className="axis-label x end" x={CHART_WIDTH - CHART_PADDING.right} y={CHART_HEIGHT - 8}>{shortDate(lastPoint.timestamp)}</text>
    </svg>
    <div className="rotation-asset-chart-legend">
      <span className="held">{tr('Held position')}</span>
      <span className="entry">{tr('BUY / ROTATE IN')}</span>
      <span className="exit">{tr('SELL / ROTATE OUT')}</span>
    </div>
  </div>
}

function StrategyAssetComparisonChart({ detail, asset }) {
  const assetSeries = useMemo(() => indexedSeries(asset?.prices || [], 'close'), [asset?.prices])
  const strategySeries = useMemo(() => indexedSeries(detail?.strategy_equity || [], 'value'), [detail?.strategy_equity])
  const model = useMemo(() => {
    const allRows = [...assetSeries, ...strategySeries]
    const timestamps = allRows.map((row) => timestampValue(row.timestamp)).filter((value) => value !== null)
    const valueExtent = extent(allRows.map((row) => row.value))
    if (!timestamps.length || !valueExtent) return null
    const domain = { firstTimestamp: Math.min(...timestamps), lastTimestamp: Math.max(...timestamps), ...valueExtent }
    return {
      assetPoints: chartCoordinates(assetSeries, 'value', domain),
      strategyPoints: chartCoordinates(strategySeries, 'value', domain),
      domain,
    }
  }, [assetSeries, strategySeries])

  if (!model?.assetPoints?.length || !model?.strategyPoints?.length) return <div className="rotation-asset-analysis-empty">{tr('Not enough observations to compare strategy and asset in this month.')}</div>

  return <div className="rotation-asset-price-chart-wrap">
    <svg className="rotation-asset-price-chart comparison" viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`} role="img" aria-label={`${asset.symbol} · ${tr('Strategy versus asset')}`}>
      {[.25, .5, .75].map((ratio) => {
        const y = CHART_PADDING.top + ratio * (CHART_HEIGHT - CHART_PADDING.top - CHART_PADDING.bottom)
        return <line key={ratio} className="grid" x1={CHART_PADDING.left} x2={CHART_WIDTH - CHART_PADDING.right} y1={y} y2={y} />
      })}
      <path className="strategy-line" d={svgPath(model.strategyPoints)} />
      <path className="asset-line" d={svgPath(model.assetPoints)} />
      <text className="axis-label y top" x="4" y={CHART_PADDING.top + 5}>{model.domain.maximum.toFixed(1)}</text>
      <text className="axis-label y bottom" x="4" y={CHART_HEIGHT - CHART_PADDING.bottom}>{model.domain.minimum.toFixed(1)}</text>
      <text className="axis-label x" x={CHART_PADDING.left} y={CHART_HEIGHT - 8}>{shortDate(model.assetPoints[0].timestamp)}</text>
      <text className="axis-label x end" x={CHART_WIDTH - CHART_PADDING.right} y={CHART_HEIGHT - 8}>{shortDate(model.assetPoints[model.assetPoints.length - 1].timestamp)}</text>
    </svg>
    <div className="rotation-asset-chart-legend">
      <span className="strategy">{tr('Strategy')} · {detail?.strategy_return == null ? '—' : percent(detail.strategy_return)}</span>
      <span className="asset">{asset.symbol} · {asset.period_return == null ? '—' : percent(asset.period_return)}</span>
    </div>
  </div>
}

function AllocationTimeline({ detail }) {
  const start = timestampValue(detail?.period_start)
  const end = timestampValue(detail?.period_end)
  const span = start === null || end === null ? 0 : end - start
  if (!span || !(detail?.position_segments || []).length) return null
  return <div className="rotation-allocation-timeline">
    <div className="rotation-month-dialog-section-title"><div><span>{tr('Monthly allocation timeline')}</span></div></div>
    <div className="rotation-allocation-track">
      {detail.position_segments.map((segment, index) => {
        const segmentStart = timestampValue(segment.start_at)
        const segmentEnd = timestampValue(segment.end_at)
        if (segmentStart === null || segmentEnd === null) return null
        const left = Math.max(0, Math.min(100, ((segmentStart - start) / span) * 100))
        const right = Math.max(0, Math.min(100, ((segmentEnd - start) / span) * 100))
        const width = Math.max(0, right - left)
        const asset = normalizedAsset(segment.asset)
        return <div
          key={`${segment.start_at}-${asset}-${index}`}
          className={`rotation-allocation-segment ${asset === 'CASH' ? 'cash' : `market tone-${index % 5}`}`}
          style={{ left: `${left}%`, width: `${width}%` }}
          title={`${asset} · ${shortDate(segment.start_at)} → ${shortDate(segment.end_at)}`}
        >{width >= 7 ? asset : ''}</div>
      })}
    </div>
    <div className="rotation-allocation-range"><span>{shortDate(detail.period_start)}</span><span>{shortDate(detail.period_end)}</span></div>
  </div>
}

export function MonthlyAssetAnalysis({ jobId, processingId = null, month, detailUrl = null }) {
  const [detail, setDetail] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [selectedAsset, setSelectedAsset] = useState('')
  const [mode, setMode] = useState('price')

  useEffect(() => {
    let active = true
    if (!jobId || !month?.year || !month?.month) return () => { active = false }
    setLoading(true)
    setError('')
    setDetail(null)
    const url = detailUrl || `${API}/analytics/${processingId ? `processings/${encodeURIComponent(processingId)}` : `backtests/${encodeURIComponent(jobId)}`}/rotation-period?year=${encodeURIComponent(month.year)}&month=${encodeURIComponent(month.month)}`
    apiFetch(url)
      .then((value) => {
        if (!active) return
        setDetail(value)
        setSelectedAsset(value?.default_asset || value?.assets?.[0]?.symbol || '')
      })
      .catch((requestError) => {
        if (!active) return
        setError(tr(requestError.message || 'Unable to load monthly asset analysis.'))
      })
      .finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [detailUrl, jobId, processingId, month?.month, month?.year])

  const asset = useMemo(() => (detail?.assets || []).find((item) => item.symbol === selectedAsset) || detail?.assets?.[0] || null, [detail?.assets, selectedAsset])

  return <section className="rotation-month-asset-analysis">
    <div className="rotation-month-asset-analysis-heading">
      <div>
        <span>{tr('Asset trade analysis')}</span>
      </div>
      {detail?.assets?.length ? <div className="rotation-asset-tabs" role="tablist" aria-label={tr('Operated assets')}>
        {detail.assets.map((item) => <button key={item.symbol} type="button" className={item.symbol === asset?.symbol ? 'active' : ''} onClick={() => setSelectedAsset(item.symbol)}>{item.symbol}</button>)}
      </div> : null}
    </div>

    {loading ? <div className="rotation-asset-analysis-empty">{tr('Loading monthly asset analysis…')}</div> : null}
    {error ? <div className="rotation-asset-analysis-empty error">{error}</div> : null}
    {!loading && !error && !asset ? <div className="rotation-asset-analysis-empty">{tr('No operated assets were found for this month.')}</div> : null}

    {!loading && !error && asset ? <>
      <div className="rotation-asset-analysis-metrics">
        <div><span>{tr('Asset')}</span><strong>{asset.symbol}</strong></div>
        <div><span>{tr('Asset return in month')}</span><strong className={asset.period_return == null ? '' : asset.period_return >= 0 ? 'positive' : 'negative'}>{asset.period_return == null ? '—' : percent(asset.period_return)}</strong></div>
        <div><span>{tr('Bought')}</span><strong className="positive">{asset.buy_count}</strong></div>
        <div><span>{tr('Sold')}</span><strong className="negative">{asset.sell_count}</strong></div>
        <div><span>{tr('Held sessions')}</span><strong>{asset.held_sessions}</strong></div>
        <div><span>{tr('Realized P/L')}</span><strong className={Number(asset.realized_pnl || 0) >= 0 ? 'positive' : 'negative'}>{money(asset.realized_pnl)}</strong></div>
      </div>

      <div className="rotation-asset-analysis-toolbar">
        <button type="button" className={mode === 'price' ? 'active' : ''} onClick={() => setMode('price')}>{tr('Price + trades')}</button>
        <button type="button" className={mode === 'comparison' ? 'active' : ''} onClick={() => setMode('comparison')}>{tr('Strategy vs asset')}</button>
      </div>

      {mode === 'price' ? <PriceAndTradeChart detail={detail} asset={asset} /> : <StrategyAssetComparisonChart detail={detail} asset={asset} />}
      <AllocationTimeline detail={detail} />
    </> : null}
  </section>
}
