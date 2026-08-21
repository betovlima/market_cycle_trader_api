import { tr, translatedStatus } from '../../../i18n/runtime'
import { durationLabel, shortDateTime } from '../../../shared/formatters'

function duration(run) {
  if (!run?.started_at || !run?.finished_at) return '—'
  return durationLabel(Math.max(0, (new Date(run.finished_at).getTime() - new Date(run.started_at).getTime()) / 1000))
}

export function RunHistory({ runs }) {
  return <section className="asset-discovery-runs">
    <div className="asset-discovery-section-heading">
      <div><span className="eyebrow">{tr('EXECUTION HISTORY')}</span><h3>{tr('Recent discovery runs')}</h3></div>
    </div>
    <div className="table-scroll"><table className="data-table asset-discovery-run-table"><thead><tr><th>{tr('Started')}</th><th>{tr('Source')}</th><th>{tr('Status')}</th><th>{tr('Attempted')}</th><th>{tr('Processed')}</th><th>{tr('Candidates')}</th><th>{tr('Watchlist')}</th><th>{tr('Rejected')}</th><th>{tr('Skipped')}</th><th>{tr('Failed')}</th><th>{tr('Duration')}</th></tr></thead><tbody>
      {runs.length ? runs.slice(0, 10).map((run) => <tr key={run.run_id}><td>{shortDateTime(run.started_at || run.created_at)}</td><td>{tr(run.source === 'automatic' ? 'Automatic' : 'Manual')}</td><td>{translatedStatus(run.status)}</td><td>{run.attempted_count ?? run.processed_count ?? 0}</td><td>{run.processed_count ?? 0}</td><td>{run.candidate_count ?? 0}</td><td>{run.watchlist_count ?? 0}</td><td>{run.rejected_count ?? 0}</td><td>{run.skipped_count ?? 0}</td><td>{run.failed_count ?? 0}</td><td>{duration(run)}</td></tr>) : <tr><td colSpan="11" className="empty-cell">{tr('No discovery runs have been recorded yet.')}</td></tr>}
    </tbody></table></div>
  </section>
}
