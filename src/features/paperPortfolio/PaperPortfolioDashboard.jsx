import { getIntlLocale, tr } from '../../i18n/runtime'
import { useCallback, useEffect, useRef, useState } from 'react'
import { apiFetch } from '../../api/http'
import { API } from '../../config/env'
import { PortfolioIcon } from '../../shared/components/Icons'
import { money, number, percent, shortDateTime } from '../../shared/formatters'
import { POLL_MS, ROBOT_POLL_MS } from './portfolioConfig'
import { CurrentPosition, PortfolioMetricsStrip, TradingSessionStrip } from './components/PortfolioPrimitives'

export function PaperPortfolioDashboard() {
  const [data, setData] = useState(null)
  const [error, setError] = useState('')
  const [refreshing, setRefreshing] = useState(false)
  const [lastUpdated, setLastUpdated] = useState(null)
  const [connection, setConnection] = useState({ status: 'checking', checkedAt: null })
  const [robot, setRobot] = useState(null)
  const [nextRefreshAt, setNextRefreshAt] = useState(null)
  const [clockNow, setClockNow] = useState(() => Date.now())
  const mountedRef = useRef(false)
  const portfolioTimerRef = useRef(null)
  const portfolioRequestRef = useRef(false)

  const loadRobotStatus = useCallback(async ({ silent = false } = {}) => {
    try {
      const response = await apiFetch(`${API}/paper-market/public-robot-status`)
      if (mountedRef.current) setRobot(response)
    } catch {
      if (mountedRef.current) {
        setRobot((current) => current ? { ...current, scheduler_alive: false, status: 'unavailable' } : { enabled: false, scheduler_alive: false, status: 'unavailable' })
      }
    }
  }, [])

  const loadPortfolio = useCallback(async ({ silent = false } = {}) => {
    if (portfolioRequestRef.current) return
    portfolioRequestRef.current = true
    if (mountedRef.current) setRefreshing(true)
    try {
      const response = await apiFetch(`${API}/paper-market/public-portfolio`)
      if (!mountedRef.current) return
      const checkedAt = new Date()
      setData(response)
      setError('')
      setLastUpdated(checkedAt)
      setConnection({ status: response?.status === 'ready' ? 'ready' : 'unavailable', checkedAt })
    } catch (requestError) {
      if (!mountedRef.current) return
      setConnection({ status: 'unavailable', checkedAt: new Date() })
      if (!silent) setError(requestError.message)
    } finally {
      portfolioRequestRef.current = false
      if (mountedRef.current) setRefreshing(false)
    }
  }, [])

  const scheduleNextPortfolioRefresh = useCallback(function scheduleNextPortfolioRefresh() {
    if (!mountedRef.current) return
    if (portfolioTimerRef.current) window.clearTimeout(portfolioTimerRef.current)
    const nextAt = Date.now() + POLL_MS
    setNextRefreshAt(nextAt)
    portfolioTimerRef.current = window.setTimeout(async () => {
      if (!mountedRef.current) return
      setNextRefreshAt(null)
      await loadPortfolio({ silent: true })
      if (mountedRef.current) scheduleNextPortfolioRefresh()
    }, POLL_MS)
  }, [loadPortfolio])

  const refreshPortfolio = useCallback(async ({ silent = false, includeRobot = false } = {}) => {
    if (portfolioTimerRef.current) window.clearTimeout(portfolioTimerRef.current)
    if (mountedRef.current) setNextRefreshAt(null)
    const tasks = [loadPortfolio({ silent })]
    if (includeRobot) tasks.push(loadRobotStatus({ silent: true }))
    await Promise.all(tasks)
    if (mountedRef.current) scheduleNextPortfolioRefresh()
  }, [loadPortfolio, loadRobotStatus, scheduleNextPortfolioRefresh])

  useEffect(() => {
    mountedRef.current = true
    refreshPortfolio({ includeRobot: true })
    const robotTimer = window.setInterval(() => loadRobotStatus({ silent: true }), ROBOT_POLL_MS)
    const clockTimer = window.setInterval(() => setClockNow(Date.now()), 1000)
    return () => {
      mountedRef.current = false
      if (portfolioTimerRef.current) window.clearTimeout(portfolioTimerRef.current)
      window.clearInterval(robotTimer)
      window.clearInterval(clockTimer)
    }
  }, [loadRobotStatus, refreshPortfolio])

  const position = data?.position

  return (
    <section className="page-stack portfolio-page portfolio-single-workspace" aria-busy={refreshing}>
      {error ? <div className="inline-error"><strong>{tr("Portfolio unavailable")}</strong><span>{tr(error)}</span><button type="button" onClick={() => setError('')}>×</button></div> : null}

      {!data ? (
        <section className="data-panel portfolio-locked portfolio-tab-loader" role="status" aria-live="polite">
          <div className="portfolio-tab-loader-visual"><span className="loading-ring" aria-hidden="true" /></div>
          <h2>{tr("Loading simulated portfolio")}</h2>
          <p>{tr("Connecting to Alpaca Paper and requesting the latest read-only portfolio snapshot.")}</p>
        </section>
      ) : (
        <section className="data-panel portfolio-workspace-panel">
          <header className="portfolio-workspace-header">
            <div className="portfolio-workspace-title">
              <div className="page-title-icon"><PortfolioIcon size={18} /></div>
              <div><h2>{tr("Portfolio")}</h2></div>
            </div>
            <div className="portfolio-workspace-actions">
              <span>{lastUpdated ? tr('Updated {time}', { time: lastUpdated.toLocaleTimeString(getIntlLocale()) }) : tr('Read-only snapshot')}</span>
              <button type="button" className="secondary-action portfolio-refresh-button compact" disabled={refreshing} onClick={() => refreshPortfolio({ includeRobot: true })}>
                {refreshing ? <span className="portfolio-button-spinner" aria-hidden="true" /> : null}
                {tr(refreshing ? 'Refreshing…' : 'Refresh')}
              </button>
            </div>
          </header>

          <PortfolioMetricsStrip data={data} position={position} />

          <TradingSessionStrip
            connection={connection}
            marketClock={data?.market_clock}
            robot={robot}
            now={clockNow}
            refreshing={refreshing}
            nextRefreshAt={nextRefreshAt}
          />

          <div className="portfolio-workspace-main">
            <CurrentPosition position={position} cash={data.strategy_cash} />
          </div>

          <section className="portfolio-orders-section">
            <div className="portfolio-section-heading portfolio-orders-heading">
              <div><span className="panel-kicker">{tr("Activity")}</span><h2>{tr("Recent Paper Orders")}</h2></div>
              <span className="panel-count">{data.recent_orders?.length || 0} {tr("records")}</span>
            </div>
            <div className="table-wrap portfolio-orders-table-wrap compact-order-scroll">
              <table className="dashboard-table portfolio-orders-table">
                <thead><tr><th>{tr("Created")}</th><th>{tr("Asset")}</th><th>{tr("Side")}</th><th>{tr("Status")}</th><th>{tr("Quantity")}</th><th>{tr("Average Fill")}</th></tr></thead>
                <tbody>
                  {data.recent_orders?.length ? data.recent_orders.map((order, index) => (
                    <tr key={`${order.created_at || 'order'}-${order.symbol || 'asset'}-${index}`}>
                      <td>{shortDateTime(order.created_at)}</td><td>{order.symbol || '—'}</td>
                      <td><span className={`order-side ${order.side}`}>{order.side === 'buy' ? tr('Buy') : order.side === 'sell' ? tr('Sell') : String(order.side || '—').toUpperCase()}</span></td>
                      <td>{order.status ? tr(String(order.status).replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase())) : '—'}</td><td>{order.filled_quantity ?? order.quantity ?? '—'}</td><td>{order.filled_average_price ? money(order.filled_average_price) : '—'}</td>
                    </tr>
                  )) : <tr><td colSpan="6" className="empty-cell">{tr("No paper orders have been submitted yet.")}</td></tr>}
                </tbody>
              </table>
            </div>
          </section>
        </section>
      )}
    </section>
  )
}
