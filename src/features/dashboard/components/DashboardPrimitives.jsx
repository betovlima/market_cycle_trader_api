import { tr } from '../../../i18n/runtime'
import { useEffect, useState } from 'react'

import { ChevronLeftIcon, ChevronRightIcon, SortIcon } from '../../../shared/components/Icons'
import { ParameterHint } from '../../../shared/components/ParameterHint'
import { money, percent, shortDateTime } from '../../../shared/formatters'
import { DASHBOARD_HINTS, DASHBOARD_PAGE_SIZE } from '../dashboardConfig'
import { decimal, strategyValue } from '../dashboardUtils'

function nextWholeHourTimestamp(now = new Date()) {
  const next = new Date(now)
  next.setMinutes(60, 0, 0)
  return next.getTime()
}

function secondsUntil(timestamp) {
  return Math.max(0, Math.ceil((timestamp - Date.now()) / 1000))
}

function countdownLabel(totalSeconds) {
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return [minutes, seconds].map((value) => String(value).padStart(2, '0')).join(':')
}

function nextUpdateLabel(timestamp) {
  return new Intl.DateTimeFormat(undefined, { hour: '2-digit', minute: '2-digit' }).format(new Date(timestamp))
}

export function DashboardMetric({ id, label, value, note, tone = '', hint }) {
  return (
    <div className={`dashboard-workspace-metric ${tone}`}>
      <div className="dashboard-metric-label">
        <span>{tr(label)}</span>
        <ParameterHint id={id} title={label} description={hint?.description || ''} relationship={hint?.relationship || ''} />
      </div>
      <strong>{value}</strong>
      <small>{typeof note === 'string' ? tr(note) : note}</small>
    </div>
  )
}

export function MarketUpdateMetric() {
  const [nextUpdateAt, setNextUpdateAt] = useState(() => nextWholeHourTimestamp())
  const [remaining, setRemaining] = useState(() => secondsUntil(nextWholeHourTimestamp()))

  useEffect(() => {
    const timer = window.setInterval(() => {
      const seconds = secondsUntil(nextUpdateAt)
      if (seconds <= 0) {
        const next = nextWholeHourTimestamp()
        setNextUpdateAt(next)
        setRemaining(secondsUntil(next))
        return
      }
      setRemaining(seconds)
    }, 1000)
    return () => window.clearInterval(timer)
  }, [nextUpdateAt])

  const progress = Math.max(0, Math.min(1, remaining / 3600))

  return (
    <div className="dashboard-workspace-metric dashboard-market-metric">
      <div className="dashboard-market-dial" style={{ '--clock-progress': `${progress * 360}deg` }} aria-hidden="true"><span>{countdownLabel(remaining)}</span></div>
      <div className="dashboard-market-copy">
        <div className="dashboard-metric-label">
          <span>{tr("Next Market Update")}</span>
          <ParameterHint id="dashboard-hint-market-update" title={tr("Next Market Update")} description={DASHBOARD_HINTS.nextMarketUpdate.description} relationship={DASHBOARD_HINTS.nextMarketUpdate.relationship} />
        </div>
        <strong>{countdownLabel(remaining)}</strong>
        <small>{tr("Scheduled for")}{' '}{nextUpdateLabel(nextUpdateAt)}</small>
      </div>
    </div>
  )
}

export function StatusBadge({ status }) {
  return <span className={`table-status ${status || 'unknown'}`}>{tr(String(status || 'unknown').replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase()))}</span>
}

export function DashboardSortHeader({ label, sortKey, sort, onSort, hint }) {
  const active = sort.key === sortKey
  return (
    <th>
      <div className="dashboard-sort-header">
        <button type="button" className={active ? 'active' : ''} onClick={() => onSort(sortKey)} title={tr("Sort by {label}", { label: tr(label) })}>
          <span>{tr(label)}</span><SortIcon size={14} descending={active ? sort.direction === 'desc' : true} />
        </button>
        <ParameterHint id={`dashboard-column-${sortKey}`} title={label} description={hint} />
      </div>
    </th>
  )
}

export function DashboardPagination({ page, pages, total, onPageChange }) {
  const from = total ? ((page - 1) * DASHBOARD_PAGE_SIZE) + 1 : 0
  const to = Math.min(page * DASHBOARD_PAGE_SIZE, total)
  return (
    <div className="dashboard-pagination">
      <span>{total ? tr("{from}–{to} of {total}", { from, to, total }) : tr("0 results")}</span>
      <div>
        <button type="button" onClick={() => onPageChange(page - 1)} disabled={page <= 1} aria-label={tr("Previous page")} title={tr("Previous page")}><ChevronLeftIcon size={16} /></button>
        <strong>{tr("Page")}{' '}{page} {tr("of")}{' '}{pages}</strong>
        <button type="button" onClick={() => onPageChange(page + 1)} disabled={page >= pages} aria-label={tr("Next page")} title={tr("Next page")}><ChevronRightIcon size={16} /></button>
      </div>
    </div>
  )
}

export function StoryTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const point = payload.find((item) => item?.payload?.timestamp_value !== undefined)?.payload
  if (!point) return null
  const events = Array.isArray(point.tradeEvents) ? point.tradeEvents : []
  return (
    <div className="dashboard-story-tooltip">
      <strong>{shortDateTime(point.timestamp)}</strong>
      <div><span>{tr("Simulation")}</span><b>{money(point.simulation_equity)}</b></div>
      <div><span>{tr("Reference")}</span><b>{money(point.reference_equity)}</b></div>
      {events.length ? <div className="dashboard-story-tooltip-events">
        {events.slice(0, 6).map((event) => event.grouped ? (
          <p key={event.key}><b>{event.count} {tr("executions")}</b><span>{event.buyCount} {tr("buy ·")}{' '}{event.sellCount} {tr("sell")}</span></p>
        ) : (
          <p key={event.key} className={event.side}><b>{event.side.toUpperCase()} · {event.asset}</b><span>{tr("Trade #")}{event.tradeNumber}</span></p>
        ))}
      </div> : null}
    </div>
  )
}

export function StoryTradeDot({ cx, cy, payload }) {
  const events = Array.isArray(payload?.tradeEvents) ? payload.tradeEvents : []
  if (!Number.isFinite(cx) || !Number.isFinite(cy) || !events.length) return null
  const grouped = events.find((event) => event.grouped)
  if (grouped) {
    return <g className="dashboard-story-group-marker" transform={`translate(${cx}, ${cy})`}>
      <circle r="11" /><text textAnchor="middle" dominantBaseline="central">{grouped.count > 99 ? '99+' : grouped.count}</text>
    </g>
  }
  let buyOffset = 0
  let sellOffset = 0
  return <g transform={`translate(${cx}, ${cy})`}>
    {events.slice(0, 6).map((event) => {
      const isBuy = event.side === 'buy'
      const offset = isBuy ? ++buyOffset : ++sellOffset
      const y = isBuy ? 8 + ((offset - 1) * 10) : -8 - ((offset - 1) * 10)
      return <circle key={event.key} cy={y} r="4.5" className={`dashboard-story-execution-dot ${event.side}`} />
    })}
  </g>
}

export function SelectedTradeTooltip({ active, payload, selectedTrade, viewMode }) {
  if (!active || !payload?.length) return null
  const point = payload[0]?.payload
  if (!point) return null
  const strategyDisplay = viewMode === 'indexed' ? (Number.isFinite(Number(point.strategy_index)) ? Number(point.strategy_index).toFixed(2) : '—') : money(point.simulation_equity)
  const referenceDisplay = viewMode === 'indexed' ? (Number.isFinite(Number(point.reference_index)) ? Number(point.reference_index).toFixed(2) : '—') : money(point.reference_equity)
  return <div className="dashboard-story-tooltip dashboard-trade-comparison-tooltip">
    <strong>{selectedTrade?.asset || tr('Position')} · {shortDateTime(point.timestamp)}</strong>
    <div><span>{tr("Strategy")}</span><b>{strategyDisplay}</b></div>
    <div><span>{tr("Buy & Hold")}</span><b>{referenceDisplay}</b></div>
    <div className="dashboard-tooltip-divider"><span>{tr("Strategy change")}</span><b className={Number(point.strategy_change) >= 0 ? 'positive' : 'negative'}>{percent(point.strategy_change)}</b></div>
    <div><span>{tr("Buy & Hold change")}</span><b className={Number(point.reference_change) >= 0 ? 'positive' : 'negative'}>{percent(point.reference_change)}</b></div>
    <div><span>{tr("Excess")}</span><b className={Number(point.excess_change) >= 0 ? 'positive' : 'negative'}>{Number.isFinite(Number(point.excess_change)) ? `${Number(point.excess_change).toFixed(2)} pp` : '—'}</b></div>
  </div>
}

export function StrategyForecastTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const row = payload[0]?.payload
  if (!row) return null
  return <div className="dashboard-story-tooltip dashboard-intelligence-tooltip">
    <strong>{row.asset}</strong>
    <div><span>{tr('Rank')}</span><b>#{row.rank ?? '—'}</b></div>
    <div><span>{tr('Ranking Utility')}</span><b>{decimal(row.ranking_utility)}</b></div>
    <div><span>{tr('Cash Edge')}</span><b>{decimal(row.cash_edge)}</b></div>
    <div><span>{tr('State')}</span><b>{row.is_target ? tr('TARGET') : row.is_current ? tr('CURRENT') : row.is_raw_best ? tr('BEST') : tr('WATCH')}</b></div>
  </div>
}

export function DecisionTimelineTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const row = payload[0]?.payload
  if (!row) return null
  return <div className="dashboard-story-tooltip dashboard-intelligence-tooltip wide">
    <strong>{shortDateTime(row.decision_date || row.timestamp)}</strong>
    <div><span>{tr('Current')}</span><b>{row.current_asset || row.previous_asset || 'CASH'}</b></div>
    <div><span>{tr('Decision')}</span><b>{row.final_action_asset || row.selected_asset || 'CASH'}</b></div>
    <div><span>{tr('Reason')}</span><b>{row.decision_reason || '—'}</b></div>
    <div><span>{tr('Best asset')}</span><b>{row.best_asset || row.raw_best_asset || '—'}</b></div>
    <div><span>{tr('Best Utility')}</span><b>{decimal(row.best_utility ?? row.absolute_utility_best_score ?? row.best_score)}</b></div>
    {Number.isFinite(Number(row.current_utility ?? row.current_score)) ? <div><span>{tr('Current Utility')}</span><b>{decimal(row.current_utility ?? row.current_score)}</b></div> : null}
    {Number.isFinite(Number(row.best_cash_edge)) ? <div><span>{tr('Best Cash Edge')}</span><b>{decimal(row.best_cash_edge)}</b></div> : null}
    {Number.isFinite(Number(row.current_cash_edge)) ? <div><span>{tr('Current Cash Edge')}</span><b>{decimal(row.current_cash_edge)}</b></div> : null}
    {Number.isFinite(Number(row.opportunity_probability)) ? <div><span>{tr('Opportunity Probability')}</span><b>{percent(row.opportunity_probability)}</b></div> : null}
    {Number.isFinite(Number(row.opportunity_confidence)) ? <div><span>{tr('Opportunity Confidence')}</span><b>{percent(row.opportunity_confidence)}</b></div> : null}
    {Number.isFinite(Number(row.opportunity_threshold)) ? <div><span>{tr(Number.isFinite(Number(row.opportunity_confidence)) ? 'Opportunity confidence threshold' : 'Opportunity threshold')}</span><b>{percent(row.opportunity_threshold)}</b></div> : null}
  </div>
}

export function TuningTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const row = payload[0]?.payload
  if (!row) return null
  return <div className="dashboard-story-tooltip dashboard-intelligence-tooltip">
    <strong>{row.label}</strong>
    <div><span>{tr('Status')}</span><b>{tr(row.status || 'unknown')}</b></div>
    <div><span>{tr('Ending capital')}</span><b>{money(row.ending_capital)}</b></div>
    <div><span>{tr('Sharpe')}</span><b>{decimal(row.sharpe, 3)}</b></div>
    <div><span>{tr('Max Drawdown')}</span><b>{percent(row.maximum_drawdown)}</b></div>
  </div>
}

export function StrategyConfigurationGrid({ configuration, modelConfiguration }) {
  const rows = Object.entries(configuration || {}).sort(([left], [right]) => left.localeCompare(right))
  const modelRows = Object.entries(modelConfiguration || {}).sort(([left], [right]) => left.localeCompare(right))
  if (!rows.length && !modelRows.length) return null
  return <details className="dashboard-strategy-config">
    <summary>{tr('Full Strategy Configuration')}</summary>
    <div className="dashboard-strategy-config-body">
      {rows.length ? <section><h4>{tr('Strategy')}</h4><dl>{rows.map(([key, value]) => <div key={key}><dt>{key}</dt><dd>{strategyValue(value)}</dd></div>)}</dl></section> : null}
      {modelRows.length ? <section><h4>{tr('Research Model')}</h4><dl>{modelRows.map(([key, value]) => <div key={key}><dt>{key}</dt><dd>{strategyValue(value)}</dd></div>)}</dl></section> : null}
    </div>
  </details>
}
