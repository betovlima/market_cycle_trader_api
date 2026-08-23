import { useMemo } from 'react'

import { hasCapability } from '../../auth/capabilities'
import { tr } from '../../i18n/runtime'
import { DashboardIcon, PlayIcon, ShieldIcon } from '../../shared/components/Icons'
import { money, percent, relativeTime, shortDateTime } from '../../shared/formatters'
import { DASHBOARD_HINTS } from './dashboardConfig'
import { DashboardMetric, MarketUpdateMetric } from './components/DashboardPrimitives'
import { DashboardBacktestAnalyticsSection } from './components/DashboardBacktestAnalyticsSection'
import { RotationQualityPerformanceSection } from './components/RotationQualityPerformanceSection'

export function DashboardPage({ workspace, capabilities = {}, onOpenBacktest, initialProcessingId = "" }) {
  const { dashboard, loadingDashboard, running, restoringExecution, startingBacktest, startDisabled, runBacktest } = workspace
  const best = dashboard?.best_performance
  const last = dashboard?.last_backtest
  const recentBacktests = useMemo(() => dashboard?.recent_backtests || [], [dashboard])
  const canRunBacktest = hasCapability(capabilities, 'backtest.start')
  const canViewTemporal = hasCapability(capabilities, 'temporal_intelligence.view')

  async function startBacktest() {
    const created = await runBacktest()
    if (created) onOpenBacktest()
  }

  return (
    <section className="page-stack dashboard-single-workspace">
      <section className="data-panel dashboard-workspace-panel">
        <div className="dashboard-workspace-header">
          <div className="dashboard-workspace-title">
            <div className="page-title-icon"><DashboardIcon size={21} /></div>
            <div><h2>{tr('Dashboard')}</h2></div>
          </div>
          <div className="dashboard-header-actions">
            <span className="dashboard-protected-badge"><ShieldIcon size={15} />{tr('Protected configuration')}</span>
            {canRunBacktest ? <button className="primary-action compact dashboard-start-action" type="button" disabled={startDisabled} onClick={startBacktest}><PlayIcon size={14} />{tr(restoringExecution ? 'Checking Execution' : startingBacktest ? 'Starting…' : running ? 'Simulation Running' : 'Start New Backtest')}</button> : null}
          </div>
        </div>

        <div className="dashboard-workspace-metrics">
          <DashboardMetric id="dashboard-hint-total-backtests" label={tr('Total Backtests')} value={loadingDashboard ? '…' : String(dashboard?.total_backtests ?? 0)} note={tr('{count} completed', { count: dashboard?.completed_backtests ?? 0 })} tone="green" hint={DASHBOARD_HINTS.totalBacktests} />
          <DashboardMetric id="dashboard-hint-best-performance" label={tr('Best Performance')} value={best?.metrics?.simulation_return == null ? '—' : percent(best.metrics.simulation_return)} note={best?.metrics?.ending_capital == null ? tr('No completed result') : tr('Ending capital {value}', { value: money(best.metrics.ending_capital) })} tone="gold" hint={DASHBOARD_HINTS.bestPerformance} />
          <DashboardMetric id="dashboard-hint-last-backtest" label={tr('Last Backtest')} value={last?.created_at ? relativeTime(last.created_at) : '—'} note={last?.created_at ? shortDateTime(last.created_at) : tr('No execution yet')} tone="blue" hint={DASHBOARD_HINTS.lastBacktest} />
          <MarketUpdateMetric />
        </div>

        {canViewTemporal ? <RotationQualityPerformanceSection /> : null}

        <DashboardBacktestAnalyticsSection fallbackJobs={recentBacktests} initialProcessingId={initialProcessingId} />
      </section>
    </section>
  )
}
