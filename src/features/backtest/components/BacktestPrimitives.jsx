import { tr } from '../../../i18n/runtime'

import { ChevronLeftIcon, ChevronRightIcon, SearchIcon, SortIcon } from '../../../shared/components/Icons'
import { ParameterHint } from '../../../shared/components/ParameterHint'
import { money, percent, shortDateTime } from '../../../shared/formatters'

export function BacktestTradeEventDot({ cx, cy, payload }) {
  const events = Array.isArray(payload?.tradeEvents) ? payload.tradeEvents : []
  if (!Number.isFinite(cx) || !Number.isFinite(cy) || !events.length) return null

  let buyLevel = 0
  let sellLevel = 0
  return (
    <g className="backtest-trade-event-group" transform={`translate(${cx}, ${cy})`}>
      {events.map((event) => {
        const isBuy = event.tradeSide === 'buy'
        const level = isBuy ? buyLevel++ : sellLevel++
        const markerY = (isBuy ? 9 : -9) + (isBuy ? 1 : -1) * level * 13
        return (
          <g key={event.markerKey} className={`backtest-trade-marker ${isBuy ? 'buy' : 'sell'}`} transform={`translate(0, ${markerY})`}>
            <circle
              r="13"
              className="backtest-trade-marker-hit"
              onPointerDown={(pointerEvent) => pointerEvent.stopPropagation()}
              aria-label={`${tr(event.tradeSide === 'buy' ? 'Buy' : event.tradeSide === 'sell' ? 'Sell' : event.tradeSide)} ${event.asset || ''}`}
            />
            <circle r="5.5" className="backtest-trade-marker-dot" />
          </g>
        )
      })}
    </g>
  )
}

export function BacktestChartTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const point = payload.find((item) => item?.payload?.timestamp_value !== undefined)?.payload
  if (!point) return null
  const tradeEvents = Array.isArray(point.tradeEvents) ? point.tradeEvents : []

  return (
    <div className={`backtest-chart-tooltip ${tradeEvents.length ? 'trade' : ''}`}>
      <div className="backtest-chart-tooltip-title">
        <strong>{tradeEvents.length ? tr(tradeEvents.length === 1 ? '{count} EXECUTION' : '{count} EXECUTIONS', { count: tradeEvents.length }) : tr('EQUITY')}</strong>
        <span>{shortDateTime(point.timestamp)}</span>
      </div>
      {tradeEvents.length ? (
        <div className="backtest-chart-tooltip-events">
          {tradeEvents.map((trade) => (
            <div key={trade.markerKey} className={`backtest-chart-tooltip-event ${trade.tradeSide}`}>
              <div className="backtest-chart-tooltip-event-title">
                <strong>{tr(trade.tradeSide === 'buy' ? 'BUY' : trade.tradeSide === 'sell' ? 'SELL' : trade.tradeSide.toUpperCase())} · {trade.asset || 'CASH'}</strong>
                <span>{trade.fromAsset || 'CASH'} → {trade.toAsset || 'CASH'}</span>
              </div>
              <div className="backtest-chart-tooltip-grid">
                <span>{tr("Executed")}</span><strong>{shortDateTime(trade.executedAt)}</strong>
                {trade.tradeSide === 'sell' ? <><span>{tr("Position return")}</span><strong>{percent(trade.positionReturn)}</strong></> : null}
                {trade.tradeSide === 'sell' ? <><span>{tr("Realized P/L")}</span><strong>{money(trade.realizedPnl)}</strong></> : null}
                <span>{tr("Fees")}</span><strong>{money(trade.transactionFees)}</strong>
              </div>
            </div>
          ))}
        </div>
      ) : null}
      <div className="backtest-chart-tooltip-equity">
        <span>{tr("Simulation")}</span><strong>{money(point.simulation_equity)}</strong>
        <span>{tr("Reference")}</span><strong>{money(point.reference_equity)}</strong>
      </div>
    </div>
  )
}

export function MetricLabel({ id, label, hint, hintDetails = [] }) {
  return (
    <span className="backtest-field-label">
      <span>{tr(label)}</span>
      {hint || hintDetails.length ? <ParameterHint id={id} title={tr(label)} description={hint} details={hintDetails} /> : null}
    </span>
  )
}

export function Metric({ id, label, value, note, tone = '', hint = '', hintDetails = [] }) {
  return (
    <article className={`result-metric ${tone}`}>
      <MetricLabel id={id} label={label} hint={hint} hintDetails={hintDetails} />
      <strong>{value}</strong>
      <small>{typeof note === 'string' ? tr(note) : note}</small>
    </article>
  )
}

export function StatusBadge({ status }) {
  return <span className={`table-status ${status || 'unknown'}`}>{tr(String(status || 'unknown').replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase()))}</span>
}

export function SortableHeader({ label, field, sort, onSort, hint = '' }) {
  const active = sort.key === field
  const description = active ? `Sorted ${sort.direction === 'desc' ? 'descending' : 'ascending'}.` : 'Click to sort this column.'
  return (
    <th aria-sort={active ? (sort.direction === 'desc' ? 'descending' : 'ascending') : 'none'}>
      <div className="backtest-sort-header">
        <button type="button" onClick={() => onSort(field)} title={`${tr(description)} ${tr(label)}`}>
          <span>{tr(label)}</span>
          <SortIcon size={14} descending={!active || sort.direction === 'desc'} />
        </button>
        {hint ? <ParameterHint id={`hint-${field}`} title={tr(label)} description={hint} align="right" /> : null}
      </div>
    </th>
  )
}

export function FilterButton({ active, label, onClick, tone = '', children = null }) {
  return (
    <button
      type="button"
      className={`backtest-filter-button ${tone} ${active ? 'active' : ''}`}
      onClick={onClick}
      aria-pressed={active}
      title={tr(label)}
    >
      {children}
      <span>{tr(label)}</span>
    </button>
  )
}

export function ListToolbar({
  query,
  onQueryChange,
  placeholder,
  children,
  resultCount,
  resultLabel = 'records',
}) {
  return (
    <div className="backtest-list-toolbar">
      <label className="backtest-list-search">
        <SearchIcon size={15} />
        <input
          type="search"
          value={query}
          onChange={(event) => onQueryChange(event.target.value)}
          placeholder={tr(placeholder)}
          aria-label={tr(placeholder)}
        />
      </label>
      <div className="backtest-list-filters">{children}</div>
      <span className="backtest-list-count">{resultCount} {tr(resultLabel)}</span>
    </div>
  )
}

export function Pagination({ page, pages, total, pageSize, onPageChange }) {
  const safePages = Math.max(1, pages)
  const safePage = Math.min(Math.max(1, page), safePages)
  const from = total ? ((safePage - 1) * pageSize) + 1 : 0
  const to = Math.min(safePage * pageSize, total)
  return (
    <div className="backtest-pagination">
      <span>{total ? tr("{from}–{to} of {total}", { from, to, total }) : tr("0 results")}</span>
      <div>
        <button type="button" onClick={() => onPageChange(safePage - 1)} disabled={safePage <= 1} aria-label={tr("Previous page")} title={tr("Previous page")}><ChevronLeftIcon size={16} /></button>
        <strong>{tr("Page")}{' '}{safePage} {tr("of")}{' '}{safePages}</strong>
        <button type="button" onClick={() => onPageChange(safePage + 1)} disabled={safePage >= safePages} aria-label={tr("Next page")} title={tr("Next page")}><ChevronRightIcon size={16} /></button>
      </div>
    </div>
  )
}
