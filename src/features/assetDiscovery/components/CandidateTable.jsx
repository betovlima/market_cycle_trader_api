import { useMemo, useState } from 'react'

import { tr, translatedStatus } from '../../../i18n/runtime'
import { number, shortDateTime } from '../../../shared/formatters'
import { SearchIcon } from '../../../shared/components/Icons'

const PAGE_SIZE = 12

function candidateTone(status) {
  if (status === 'candidate') return 'candidate'
  if (status === 'watchlist') return 'watchlist'
  if (status === 'evaluating') return 'evaluating'
  if (status === 'failed') return 'failed'
  if (status === 'rejected') return 'rejected'
  if (status === 'skipped') return 'skipped'
  return 'neutral'
}

function dollarVolume(value) {
  const amount = Number(value)
  if (!Number.isFinite(amount)) return '—'
  if (amount >= 1_000_000_000) return `$${number(amount / 1_000_000_000, 2)}B`
  if (amount >= 1_000_000) return `$${number(amount / 1_000_000, 2)}M`
  if (amount >= 1_000) return `$${number(amount / 1_000, 1)}K`
  return `$${number(amount, 0)}`
}

function historyLabel(item) {
  if (!item.historical_cache_ready || !item.history_start) return tr('Not cached')
  const range = `${String(item.history_start).slice(0, 10)} → ${String(item.history_end || '').slice(0, 10)}`
  const profile = item.history_profile ? tr(String(item.history_profile).replaceAll('_', ' ')) : ''
  return profile ? `${range} · ${profile}` : range
}

export function CandidateTable({ candidates }) {
  const [query, setQuery] = useState('')
  const [status, setStatus] = useState('all')
  const [page, setPage] = useState(1)

  const filtered = useMemo(() => {
    const search = query.trim().toUpperCase()
    return candidates.filter((item) => {
      if (status !== 'all' && item.status !== status) return false
      return !search || String(item.symbol || '').includes(search)
    })
  }, [candidates, query, status])

  const pages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE))
  const safePage = Math.min(page, pages)
  const rows = filtered.slice((safePage - 1) * PAGE_SIZE, safePage * PAGE_SIZE)

  return <section className="asset-discovery-candidates">
    <div className="asset-discovery-section-heading">
      <div>
        <span className="eyebrow">{tr('CANDIDATE POOL')}</span>
        <h3>{tr('Candidate assets')}</h3>
      </div>
      <strong>{tr('{count} results', { count: filtered.length })}</strong>
    </div>

    <div className="asset-discovery-table-toolbar">
      <label className="asset-discovery-search"><SearchIcon size={16} /><input value={query} onChange={(event) => { setQuery(event.target.value); setPage(1) }} placeholder={tr('Filter symbol')} /></label>
      <select value={status} onChange={(event) => { setStatus(event.target.value); setPage(1) }}>
        <option value="all">{tr('All statuses')}</option>
        <option value="candidate">{tr('Candidate')}</option>
        <option value="watchlist">{tr('Watchlist')}</option>
        <option value="rejected">{tr('Rejected')}</option>
        <option value="skipped">{tr('Skipped')}</option>
        <option value="evaluating">{tr('Evaluating')}</option>
        <option value="failed">{tr('Failed')}</option>
      </select>
    </div>

    <div className="table-scroll asset-discovery-table-wrap">
      <table className="data-table asset-discovery-table">
        <thead><tr>
          <th>{tr('Asset')}</th><th>{tr('Status')}</th><th>{tr('History')}</th><th>{tr('Sessions')}</th><th>{tr('Last price')}</th><th>{tr('Median $ volume (63d)')}</th><th>{tr('Evaluated API')}</th><th>{tr('Last analysis')}</th><th>{tr('Indicators')}</th>
        </tr></thead>
        <tbody>
          {rows.length ? rows.map((item) => <tr key={item.symbol}>
            <td><strong>{item.symbol}</strong></td>
            <td><span className={`table-status ${candidateTone(item.status)}`}>{translatedStatus(item.status)}</span></td>
            <td>{historyLabel(item)}</td>
            <td>{item.history_sessions ?? '—'}</td>
            <td>{Number.isFinite(Number(item.latest_close)) ? `$${number(item.latest_close, 2)}` : '—'}</td>
            <td>{dollarVolume(item.median_dollar_volume_63d)}</td>
            <td>{item.last_evaluated_api_version ? `v${item.last_evaluated_api_version}` : tr('Legacy')}</td>
            <td>{shortDateTime(item.last_evaluated_at)}</td>
            <td className="asset-discovery-reasons">
              {[...(item.reason_codes || []).map((code) => tr(String(code).replaceAll('_', ' '))), item.last_error].filter(Boolean).join(' · ') || '—'}
            </td>
          </tr>) : <tr><td colSpan="9" className="empty-cell">{tr('No candidate assets match the current filters.')}</td></tr>}
        </tbody>
      </table>
    </div>

    <div className="asset-discovery-pagination">
      <span>{tr('Page {page} of {pages}', { page: safePage, pages })}</span>
      <div><button type="button" className="secondary-action compact asset-discovery-action" disabled={safePage <= 1} onClick={() => setPage((value) => Math.max(1, value - 1))}>{tr('Previous')}</button><button type="button" className="secondary-action compact asset-discovery-action" disabled={safePage >= pages} onClick={() => setPage((value) => Math.min(pages, value + 1))}>{tr('Next')}</button></div>
    </div>
  </section>
}
