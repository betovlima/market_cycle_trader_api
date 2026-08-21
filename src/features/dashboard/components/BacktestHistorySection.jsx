import { tr } from '../../../i18n/runtime'

import { SearchIcon } from '../../../shared/components/Icons'
import { durationLabel, percent, shortDateTime } from '../../../shared/formatters'
import { DASHBOARD_HINTS } from '../dashboardConfig'
import { DashboardPagination, DashboardSortHeader, StatusBadge } from './DashboardPrimitives'

export function BacktestHistorySection({
  filteredRows,
  query,
  onQueryChange,
  statusFilter,
  onStatusFilterChange,
  statusCounts,
  visibleRows,
  sort,
  updateSort,
  safePage,
  pages,
  onPageChange,
}) {
  return (
<section className="dashboard-history-section">
          <div className="dashboard-section-heading"><div><span className="panel-kicker">{tr("History")}</span><h2>{tr("Recent Backtests")}</h2></div><span className="panel-count">{filteredRows.length} {tr(filteredRows.length === 1 ? "result" : "results")}</span></div>
          <div className="dashboard-history-toolbar">
            <label className="dashboard-list-search"><SearchIcon size={15} /><input type="search" value={query} onChange={(event) => onQueryChange(event.target.value)} placeholder={tr("Filter execution history")} aria-label={tr("Filter execution history")} /></label>
            <div className="dashboard-status-filters" role="group" aria-label={tr("Backtest status filter")}>
              {[['all', 'All'], ['completed', 'Completed'], ['interrupted', 'Interrupted'], ['failed', 'Failed'], ['active', 'Active']].map(([value, label]) => <button key={value} type="button" className={statusFilter === value ? 'active' : ''} onClick={() => onStatusFilterChange(value)}>{tr(label)}<span>{statusCounts[value]}</span></button>)}
            </div>
          </div>
          <div className="table-wrap dashboard-history-table-wrap">
            <table className="dashboard-table dashboard-sortable-table">
              <thead><tr>
                <DashboardSortHeader label={tr("Date")} sortKey="date" sort={sort} onSort={updateSort} hint={DASHBOARD_HINTS.date} />
                <DashboardSortHeader label={tr("Status")} sortKey="status" sort={sort} onSort={updateSort} hint={DASHBOARD_HINTS.status} />
                <DashboardSortHeader label={tr("Total Return")} sortKey="return" sort={sort} onSort={updateSort} hint={DASHBOARD_HINTS.totalReturn} />
                <DashboardSortHeader label={tr("Sharpe Ratio")} sortKey="sharpe" sort={sort} onSort={updateSort} hint={DASHBOARD_HINTS.sharpe} />
                <DashboardSortHeader label={tr("Max Drawdown")} sortKey="drawdown" sort={sort} onSort={updateSort} hint={DASHBOARD_HINTS.drawdown} />
                <DashboardSortHeader label={tr("Rotations")} sortKey="rotations" sort={sort} onSort={updateSort} hint={DASHBOARD_HINTS.rotations} />
                <DashboardSortHeader label={tr("Duration")} sortKey="duration" sort={sort} onSort={updateSort} hint={DASHBOARD_HINTS.duration} />
              </tr></thead>
              <tbody>{visibleRows.length ? visibleRows.map((item) => <tr key={item.id}>
                <td>{shortDateTime(item.created_at)}</td><td><StatusBadge status={item.status} /></td><td className={item.metrics?.simulation_return == null ? '' : Number(item.metrics.simulation_return) >= 0 ? 'positive' : 'negative'}>{percent(item.metrics?.simulation_return)}</td><td>{item.metrics?.sharpe == null ? '—' : Number(item.metrics.sharpe).toFixed(3)}</td><td className={item.metrics?.maximum_drawdown == null ? '' : 'negative'}>{percent(item.metrics?.maximum_drawdown)}</td><td>{item.metrics?.position_changes == null ? '—' : Math.round(item.metrics.position_changes)}</td><td>{durationLabel(item.duration_seconds)}</td>
              </tr>) : <tr><td colSpan="7" className="empty-cell">{tr("No backtests match the current filters.")}</td></tr>}</tbody>
            </table>
          </div>
          <DashboardPagination page={safePage} pages={pages} total={filteredRows.length} onPageChange={onPageChange} />
        </section>
  )
}
