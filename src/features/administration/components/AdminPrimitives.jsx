import { tr } from '../../../i18n/runtime'
import { useState } from 'react'

import { AccessLinkIcon, ChevronLeftIcon, ChevronRightIcon, SearchIcon, SortIcon } from '../../../shared/components/Icons'
import { ParameterHint } from '../../../shared/components/ParameterHint'
import { copyText, dateTime } from '../adminUtils'

export function AdminFieldLabel({ id, label, hint }) {
  return (
    <span className="admin-field-label">
      <span>{tr(label)}</span>
      <ParameterHint id={id} title={label} description={hint} />
    </span>
  )
}

export function AdminSortableTh({ label, field, sort, onSort, hint = '' }) {
  const active = sort.key === field
  return (
    <th>
      <button type="button" className={`admin-sort-header ${active ? 'active' : ''}`} onClick={() => onSort(field)} title={tr("Sort by {label}", { label: tr(label) })}>
        <span>{tr(label)}</span>
        {hint ? <ParameterHint id={`admin-column-${field}`} title={label} description={hint} /> : null}
        <SortIcon size={14} descending={active ? sort.direction === 'desc' : true} />
      </button>
    </th>
  )
}

export function AdminListToolbar({ query, onQueryChange, placeholder, count, children }) {
  return (
    <div className="admin-list-toolbar">
      <label className="admin-list-search">
        <SearchIcon size={15} />
        <input type="search" value={query} onChange={(event) => onQueryChange(event.target.value)} placeholder={tr(placeholder)} aria-label={tr(placeholder)} />
      </label>
      <div className="admin-list-filters">{children}</div>
      <span className="admin-list-count">{count} {tr(count === 1 ? "result" : "results")}</span>
    </div>
  )
}

export function AdminPagination({ page, pages, total, pageSize, onPageChange }) {
  const from = total ? ((page - 1) * pageSize) + 1 : 0
  const to = Math.min(page * pageSize, total)
  return (
    <div className="admin-pagination">
      <span>{total ? tr("{from}–{to} of {total}", { from, to, total }) : tr("0 results")}</span>
      <div>
        <button type="button" onClick={() => onPageChange(page - 1)} disabled={page <= 1} aria-label={tr("Previous page")} title={tr("Previous page")}><ChevronLeftIcon size={16} /></button>
        <strong>{tr("Page")}{' '}{page} {tr("of")}{' '}{pages}</strong>
        <button type="button" onClick={() => onPageChange(page + 1)} disabled={page >= pages} aria-label={tr("Next page")} title={tr("Next page")}><ChevronRightIcon size={16} /></button>
      </div>
    </div>
  )
}

export function AccessLinkDialog({ access, onClose, onError }) {
  const [copied, setCopied] = useState(false)

  async function copyLink() {
    try {
      await copyText(access.access_url)
      setCopied(true)
      onError('')
    } catch (copyError) {
      setCopied(false)
      onError(tr(copyError.message || 'Unable to copy the access link.'))
    }
  }

  return (
    <div className="access-link-dialog-backdrop" role="presentation" onMouseDown={onClose}>
      <section className="access-link-dialog" role="dialog" aria-modal="true" aria-labelledby="access-link-title" onMouseDown={(event) => event.stopPropagation()}>
        <div className="access-link-dialog-heading">
          <div>
            <span className="panel-kicker">{tr("ONE-TIME IDENTITY CLAIM")}</span>
            <h2 id="access-link-title">{tr("Verified access for")}{' '}{access.guest_name}</h2>
          </div>
          <button type="button" className="access-link-close" onClick={onClose} aria-label={tr("Close")}>×</button>
        </div>
        <p>
          {tr("Share this link only with")}{' '}<strong>{access.authorized_email}</strong>{tr(". The first valid claim must use that Google account. After claiming, the raw token is consumed and the authorization is bound to the verified Google identity.")}</p>
        <textarea readOnly value={access.access_url} rows="4" aria-label={tr("Generated access link")} />
        <div className="access-link-expiration">
          {tr("Expires:")}{' '}<strong>{dateTime(access.expires_at)}</strong> {tr("· Maximum active sessions:")}{' '}<strong>{access.max_active_sessions}</strong>
        </div>
        <div className="access-link-dialog-actions">
          <button type="button" className="admin-primary-button" onClick={copyLink}>
            <AccessLinkIcon size={17} />{tr(copied ? 'Link copied' : 'Copy verified link')}
          </button>
          <button type="button" className="access-link-secondary" onClick={onClose}>{tr("Close")}</button>
        </div>
      </section>
    </div>
  )
}

export function AdminSummary({ icon, label, value, tone = '', hint = '' }) {
  const hintId = `admin-summary-${label.toLowerCase().replace(/[^a-z0-9]+/g, '-')}`
  return (
    <article className={`admin-summary-card ${tone}`}>
      <div className={`admin-summary-icon ${tone}`}>{icon}</div>
      <div>
        <div className="admin-summary-label">
          <span>{tr(label)}</span>
          {hint ? <ParameterHint id={hintId} title={label} description={hint} /> : null}
        </div>
        <strong className={tone}>{value}</strong>
      </div>
    </article>
  )
}
