import { getIntlLocale, tr } from '../../../i18n/runtime'

import { ActivityIcon } from '../../../shared/components/Icons'
import { money, number, percent, shortDateTime } from '../../../shared/formatters'
import { scheduleValue } from '../portfolioUtils'

export function TradingSessionStrip({ connection, marketClock, robot, now, refreshing, nextRefreshAt }) {
  const firstCheckPending = refreshing && !connection.checkedAt
  const connectionStatus = firstCheckPending ? 'checking' : connection.status
  const connectionReady = connectionStatus === 'ready'

  const enabled = Boolean(robot?.enabled)
  const schedulerAlive = Boolean(robot?.scheduler_alive)
  const blocked = robot?.status === 'blocked'
  const unavailable = robot?.status === 'unavailable'
  const robotLoaded = Boolean(robot)
  const robotReady = robotLoaded && enabled && schedulerAlive && !blocked && !unavailable
  const robotTone = robotReady ? 'ready' : blocked || unavailable || (enabled && !schedulerAlive) ? 'unavailable' : 'checking'
  const robotLabel = tr(!robotLoaded ? 'Checking' : robotReady ? 'Active' : blocked ? 'Review' : enabled ? 'Degraded' : 'Stopped')

  const marketLoaded = Boolean(marketClock)
  const marketOpen = marketLoaded && Boolean(marketClock.is_open)
  const marketTone = !marketLoaded ? 'checking' : marketOpen ? 'ready' : 'closed'
  const marketLabel = tr(!marketLoaded ? 'Market checking' : marketOpen ? 'Market open' : 'Market closed')

  const phaseRaw = String(robot?.active_run?.phase || robot?.phase || 'stopped')
  const phase = phaseRaw.replaceAll('_', ' ')
  const phaseLower = phaseRaw.toLowerCase()
  const analysisRunning = phaseLower.includes('training') || phaseLower.includes('refreshing_market_data') || phaseLower.includes('preparing_premarket_plan')
  const executionRunning = phaseLower.includes('submitting_alpaca_paper_orders') || phaseLower === 'executing'
  const analysisAt = robot?.next_premarket_analysis_at || robot?.active_run?.premarket_analysis_at
  const nextOpenAt = robot?.next_market_open || robot?.active_run?.expected_market_open
  const nextCloseAt = marketClock?.next_close
  const session = robot?.next_execution_session || tr('No session scheduled')
  const checkedLabel = connection.checkedAt
    ? tr('Broker {time}', { time: connection.checkedAt.toLocaleTimeString(getIntlLocale()) })
    : tr('Broker check pending')

  const schedule = [
    { label: 'Analysis', value: scheduleValue(analysisAt, now, analysisRunning), tone: analysisRunning ? 'green' : 'blue' },
    { label: 'Execution', value: scheduleValue(nextOpenAt, now, executionRunning), tone: executionRunning ? 'green' : 'purple' },
    { label: 'Daily close', value: scheduleValue(nextCloseAt, now), tone: 'gold' },
    { label: 'Portfolio update', value: refreshing ? tr('Running now') : scheduleValue(nextRefreshAt, now), tone: refreshing ? 'green' : 'cyan' },
  ]

  return (
    <div className="portfolio-session-strip" aria-label={tr("Trading session status")} aria-live="polite">
      <div className="portfolio-session-main">
        <div className="portfolio-session-title" title={tr('Robot phase: {phase}. Next execution: {session}. {broker}.', { phase: tr(phase.replace(/\b\w/g, (letter) => letter.toUpperCase())), session, broker: checkedLabel })}>
          <span className="portfolio-session-icon"><ActivityIcon size={16} /></span>
          <span>{tr("Trading Session")}</span>
        </div>
        <div className="trading-session-statuses portfolio-session-statuses">
          <span className={`session-status-chip ${connectionReady ? 'ready' : connectionStatus === 'checking' ? 'checking' : 'unavailable'}`}>
            <span className="connection-dot" />{tr("Alpaca")}{' '}{tr(connectionReady ? 'Connected' : connectionStatus === 'checking' ? 'Checking' : 'Unavailable')}
          </span>
          <span className={`session-status-chip ${robotTone}`}><span className="connection-dot" />{tr("Robot")}{' '}{robotLabel}</span>
          <span className={`session-status-chip ${marketTone}`}><span className="connection-dot" />{marketLabel}</span>
        </div>
      </div>
      <div className="portfolio-session-schedule" aria-label={tr('Robot phase {phase}. Next execution {session}.', { phase: tr(phase.replace(/\b\w/g, (letter) => letter.toUpperCase())), session })}>
        {schedule.map((item) => (
          <div key={item.label} className={`portfolio-session-step ${item.tone}`}>
            <span>{tr(item.label)}</span>
            <strong>{item.value}</strong>
          </div>
        ))}
      </div>
    </div>
  )
}

export function PortfolioMetric({ label, value, detail, tone = '' }) {
  return (
    <div className={`portfolio-workspace-metric ${tone}`}>
      <span>{tr(label)}</span>
      <strong>{value}</strong>
      {detail ? <small>{tr(detail)}</small> : null}
    </div>
  )
}

export function PortfolioMetricsStrip({ data, position }) {
  const activePositions = position ? 1 : 0
  const returnTone = Number(data.total_return) >= 0 ? 'positive' : 'negative'

  return (
    <div className="portfolio-workspace-metrics" aria-label={tr("Portfolio summary")}>
      <PortfolioMetric label={tr("Starting Capital")} value={money(data.initial_capital)} tone="blue" />
      <PortfolioMetric label={tr("Portfolio Value")} value={money(data.portfolio_value)} tone="blue" />
      <PortfolioMetric label={tr("Total P/L")} value={money(data.total_pnl)} detail={percent(data.total_return)} tone={returnTone} />
      <PortfolioMetric label={tr("Cash")} value={money(data.strategy_cash)} tone="purple" />
      <PortfolioMetric label={tr("Position")} value={String(activePositions)} detail={position ? position.symbol : tr('Cash')} tone="gold" />
    </div>
  )
}

export function CurrentPosition({ position, cash }) {
  if (!position) {
    return (
      <aside className="current-position-section">
        <div className="portfolio-section-heading compact">
          <div><span className="panel-kicker">{tr("Position")}</span><h2>{tr("Current Position")}</h2></div>
        </div>
        <div className="cash-state current-position-cash"><strong>{money(cash)}</strong><span>{tr("Cash")}</span><p>{tr("No open position.")}</p></div>
      </aside>
    )
  }

  const quantity = Number(position.quantity)
  const entryPrice = Number(position.average_entry_price)
  const marketValue = Number(position.market_value)
  const costBasis = Number.isFinite(quantity) && Number.isFinite(entryPrice) ? quantity * entryPrice : null
  const unrealizedPnl = Number.isFinite(marketValue) && Number.isFinite(costBasis) ? marketValue - costBasis : null
  const returnPositive = Number(position.unrealized_return) >= 0

  return (
    <aside className="current-position-section">
      <div className="portfolio-section-heading compact current-position-heading">
        <div><span className="panel-kicker">{tr("Position")}</span><h2>{tr("Current Position")}</h2></div>
        <span className="current-trade-open">{tr("Open")}</span>
      </div>

      <div className="current-trade-asset current-position-asset">
        <strong>{position.symbol}</strong>
        <span>{number(position.quantity, 6)} {tr("shares")}</span>
      </div>

      <div className="current-position-stats">
        <div><span>{tr("Entry")}</span><strong>{money(position.average_entry_price)}</strong></div>
        <div><span>{tr("Current")}</span><strong>{money(position.current_price)}</strong></div>
        <div><span>{tr("Market value")}</span><strong>{money(position.market_value)}</strong></div>
        <div><span>{tr("Trade P/L")}</span><strong className={returnPositive ? 'positive' : 'negative'}>{unrealizedPnl === null ? '—' : money(unrealizedPnl)}</strong></div>
      </div>

      <div className={`current-trade-return current-position-return ${returnPositive ? 'positive' : 'negative'}`}>
        <span>{tr("Unrealized return")}</span>
        <strong>{percent(position.unrealized_return)}</strong>
      </div>
    </aside>
  )
}

export function TradeEventDot({ cx, cy, payload }) {
  const events = Array.isArray(payload?.tradeEvents) ? payload.tradeEvents : []
  if (!Number.isFinite(cx) || !Number.isFinite(cy) || !events.length) return null

  let buyLevel = 0
  let sellLevel = 0

  return (
    <g className="portfolio-trade-event-group" transform={`translate(${cx}, ${cy})`}>
      {events.map((event) => {
        const isBuy = event.tradeSide === 'buy'
        const level = isBuy ? buyLevel++ : sellLevel++
        const markerY = (isBuy ? 8 : -8) + (isBuy ? 1 : -1) * level * 13
        const markerClass = isBuy ? 'buy' : 'sell'
        return (
          <g
            key={event.markerKey}
            className={`portfolio-trade-marker ${markerClass}`}
            transform={`translate(0, ${markerY})`}
          >
            <circle r="11" className="portfolio-trade-marker-hit" />
            <circle r="6" className="portfolio-trade-marker-dot" />
          </g>
        )
      })}
    </g>
  )
}

export function PortfolioChartTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const pointPayload = payload.find((item) => item?.payload?.portfolio_value !== undefined)?.payload
  if (!pointPayload) return null

  const tradeEvents = Array.isArray(pointPayload.tradeEvents) ? pointPayload.tradeEvents : []
  if (tradeEvents.length) {
    const singleTrade = tradeEvents.length === 1 ? tradeEvents[0] : null
    return (
      <div className={`portfolio-chart-tooltip trade ${singleTrade?.tradeSide || 'multiple'}`}>
        <div className="portfolio-tooltip-title">
          <strong>{singleTrade ? singleTrade.tradeSide.toUpperCase() : `${tradeEvents.length} EXECUTIONS`}</strong>
          <span>{singleTrade?.symbol || shortDateTime(pointPayload.recorded_at)}</span>
        </div>
        <div className="portfolio-tooltip-trades">
          {tradeEvents.map((trade) => (
            <div key={trade.markerKey} className={`portfolio-tooltip-trade ${trade.tradeSide}`}>
              {tradeEvents.length > 1 ? (
                <div className="portfolio-tooltip-trade-header">
                  <strong>{trade.tradeSide.toUpperCase()}</strong><span>{trade.symbol || '—'}</span>
                </div>
              ) : null}
              <div className="portfolio-tooltip-grid">
                <span>{tr("Executed")}</span><strong>{shortDateTime(trade.orderTime)}</strong>
                <span>{tr("Quantity")}</span><strong>{trade.quantity ?? '—'}</strong>
                <span>{tr("Average fill")}</span><strong>{trade.price == null ? '—' : money(trade.price)}</strong>
                <span>{tr("Portfolio")}</span><strong>{money(pointPayload.portfolio_value)}</strong>
              </div>
            </div>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="portfolio-chart-tooltip">
      <span>{shortDateTime(pointPayload.recorded_at)}</span>
      <strong>{tr("Portfolio ·")}{' '}{money(pointPayload.portfolio_value)}</strong>
    </div>
  )
}
