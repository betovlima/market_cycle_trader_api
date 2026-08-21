import { tr } from '../i18n/runtime'
import { useCallback, useEffect, useMemo, useState } from 'react'

import { ApiError, apiFetch } from '../api/http'
import { API } from '../config/env'
import {
  AccessLinkIcon,
  AccessLockIcon,
  AccessUsersIcon,
  ClockIcon,
  EyeIcon,
  ListFilterIcon,
  ShieldIcon,
} from '../shared/components/Icons'
import { ADMIN_HINTS, DEFAULT_DURATION_SECONDS, DURATION_OPTIONS, INVITATION_PAGE_SIZE, LOG_PAGE_SIZE, SESSION_OPTIONS } from './administration/adminConfig'
import { boundedPage, dateTime, defaultSessions, roleLabel, sortedRows, statusClass, statusLabel, toggledSort } from './administration/adminUtils'
import { AccessLinkDialog, AdminFieldLabel, AdminListToolbar, AdminPagination, AdminSortableTh, AdminSummary } from './administration/components/AdminPrimitives'

export function AdministrationPage({ onSessionExpired }) {
  const [invitations, setInvitations] = useState([])
  const [logs, setLogs] = useState([])
  const [loading, setLoading] = useState(true)
  const [busyId, setBusyId] = useState('')
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [form, setForm] = useState({
    guest_name: '',
    authorized_email: '',
    role: 'viewer',
    duration_seconds: DEFAULT_DURATION_SECONDS,
    max_active_sessions: defaultSessions('viewer'),
  })
  const [extendDurations, setExtendDurations] = useState({})
  const [sessionLimits, setSessionLimits] = useState({})
  const [generatedAccess, setGeneratedAccess] = useState(null)
  const [dataView, setDataView] = useState('invitations')
  const [invitationQuery, setInvitationQuery] = useState('')
  const [invitationRole, setInvitationRole] = useState('all')
  const [invitationStatus, setInvitationStatus] = useState('all')
  const [invitationSort, setInvitationSort] = useState({ key: 'guest_name', direction: 'asc' })
  const [invitationPage, setInvitationPage] = useState(1)
  const [logQuery, setLogQuery] = useState('')
  const [logRole, setLogRole] = useState('all')
  const [logResult, setLogResult] = useState('all')
  const [logSort, setLogSort] = useState({ key: 'created_at', direction: 'desc' })
  const [logPage, setLogPage] = useState(1)

  const handleError = useCallback((requestError) => {
    if (requestError instanceof ApiError && requestError.status === 401) {
      onSessionExpired()
      return
    }
    setError(tr(requestError.message || 'Unable to update access control.'))
  }, [onSessionExpired])

  const loadData = useCallback(async () => {
    setLoading(true)
    try {
      const [invitationResponse, logResponse] = await Promise.all([
        apiFetch(`${API}/admin/invitations`),
        apiFetch(`${API}/admin/access-logs?limit=100`),
      ])
      const items = invitationResponse.items || []
      setInvitations(items)
      setSessionLimits((current) => {
        const next = { ...current }
        items.forEach((item) => {
          if (!next[item.id]) next[item.id] = String(item.max_active_sessions || defaultSessions(item.role))
        })
        return next
      })
      setLogs(logResponse.items || [])
      setError('')
    } catch (requestError) {
      handleError(requestError)
    } finally {
      setLoading(false)
    }
  }, [handleError])

  useEffect(() => {
    loadData()
  }, [loadData])

  async function createInvitation(event) {
    event.preventDefault()
    if (!form.duration_seconds) {
      setError(tr('Select an access duration.'))
      return
    }

    setBusyId('create')
    setError('')
    setNotice('')
    try {
      const created = await apiFetch(`${API}/admin/invitations`, {
        method: 'POST',
        body: {
          guest_name: form.guest_name.trim(),
          authorized_email: form.authorized_email.trim().toLowerCase(),
          role: form.role,
          duration_seconds: Number(form.duration_seconds),
          max_active_sessions: Number(form.max_active_sessions),
        },
      })
      setForm({
        guest_name: '',
        authorized_email: '',
        role: 'viewer',
        duration_seconds: DEFAULT_DURATION_SECONDS,
        max_active_sessions: defaultSessions('viewer'),
      })
      setGeneratedAccess(created)
      setNotice(tr('Identity-verified access link generated for {name}.', { name: created.guest_name }))
      await loadData()
    } catch (requestError) {
      handleError(requestError)
    } finally {
      setBusyId('')
    }
  }

  async function runAction(id, action, message) {
    setBusyId(`${id}:${action}`)
    setError('')
    setNotice('')
    try {
      await apiFetch(`${API}/admin/invitations/${encodeURIComponent(id)}/${action}`, {
        method: 'POST',
      })
      setNotice(tr(message))
      await loadData()
    } catch (requestError) {
      handleError(requestError)
    } finally {
      setBusyId('')
    }
  }

  async function regenerateAccessLink(invitation) {
    const selectedDuration = extendDurations[invitation.id] ?? DEFAULT_DURATION_SECONDS
    setBusyId(`${invitation.id}:regenerate-link`)
    setError('')
    setNotice('')
    try {
      const generated = await apiFetch(
        `${API}/admin/invitations/${encodeURIComponent(invitation.id)}/regenerate-link`,
        {
          method: 'POST',
          body: { duration_seconds: Number(selectedDuration) },
        },
      )
      setGeneratedAccess(generated)
      setNotice(tr('A new identity claim link was generated for {name}. Existing sessions were ended.', { name: invitation.guest_name }))
      setExtendDurations((current) => ({
        ...current,
        [invitation.id]: DEFAULT_DURATION_SECONDS,
      }))
      await loadData()
    } catch (requestError) {
      handleError(requestError)
    } finally {
      setBusyId('')
    }
  }

  async function updateInvitation(invitation, body, message, actionName) {
    setBusyId(`${invitation.id}:${actionName}`)
    setError('')
    setNotice('')
    try {
      await apiFetch(`${API}/admin/invitations/${encodeURIComponent(invitation.id)}`, {
        method: 'PATCH',
        body,
      })
      setNotice(tr(message))
      await loadData()
    } catch (requestError) {
      handleError(requestError)
    } finally {
      setBusyId('')
    }
  }

  async function extendInvitation(invitation) {
    const selectedDuration = extendDurations[invitation.id] ?? DEFAULT_DURATION_SECONDS
    await updateInvitation(
      invitation,
      { duration_seconds: Number(selectedDuration) },
      tr('Access extended for {name}.', { name: invitation.guest_name }),
      'extend',
    )
    setExtendDurations((current) => ({ ...current, [invitation.id]: DEFAULT_DURATION_SECONDS }))
  }

  async function saveSessionLimit(invitation) {
    const limit = Number(sessionLimits[invitation.id] || invitation.max_active_sessions || 1)
    await updateInvitation(
      invitation,
      { max_active_sessions: limit },
      tr('Session limit updated for {name}.', { name: invitation.guest_name }),
      'session-limit',
    )
  }

  async function deleteInvitation(invitation) {
    if (!window.confirm(tr('Delete the access record for {name}?', { name: invitation.guest_name }))) return

    setBusyId(`${invitation.id}:delete`)
    setError('')
    setNotice('')
    try {
      await apiFetch(`${API}/admin/invitations/${encodeURIComponent(invitation.id)}`, {
        method: 'DELETE',
      })
      setNotice(tr('Access record deleted for {name}.', { name: invitation.guest_name }))
      await loadData()
    } catch (requestError) {
      handleError(requestError)
    } finally {
      setBusyId('')
    }
  }

  const counts = useMemo(() => ({
    pending: invitations.filter((item) => item.status === 'pending_verification').length,
    active: invitations.filter((item) => ['active', 'claimed'].includes(item.status)).length,
    expired: invitations.filter((item) => item.status === 'expired').length,
    restricted: invitations.filter((item) => ['revoked', 'legacy_unverified', 'blocked'].includes(item.status)).length,
  }), [invitations])

  const filteredInvitations = useMemo(() => {
    const query = invitationQuery.trim().toLowerCase()
    const rows = invitations.filter((item) => {
      const roleMatch = invitationRole === 'all' || item.role === invitationRole
      const statusMatch = invitationStatus === 'all'
        || (invitationStatus === 'active' && ['active', 'claimed'].includes(item.status))
        || (invitationStatus === 'pending' && item.status === 'pending_verification')
        || (invitationStatus === 'expired' && item.status === 'expired')
        || (invitationStatus === 'restricted' && ['revoked', 'legacy_unverified', 'blocked'].includes(item.status))
      const queryMatch = !query || [
        item.guest_name,
        item.authorized_email,
        item.claimed_email,
        roleLabel(item.role),
        statusLabel(item.status),
      ].some((value) => String(value || '').toLowerCase().includes(query))
      return roleMatch && statusMatch && queryMatch
    })
    return sortedRows(rows, invitationSort, (item, key) => {
      if (key === 'role') return roleLabel(item.role)
      if (key === 'status') return statusLabel(item.status)
      if (key === 'sessions') return Number(item.active_sessions || 0)
      if (key === 'claimed_identity') return item.claimed_email || ''
      if (key === 'expires_at') return item.expires_at ? new Date(item.expires_at).getTime() : null
      if (key === 'last_access_at') return item.last_access_at ? new Date(item.last_access_at).getTime() : null
      return item?.[key]
    })
  }, [invitations, invitationQuery, invitationRole, invitationStatus, invitationSort])

  const invitationPages = Math.max(1, Math.ceil(filteredInvitations.length / INVITATION_PAGE_SIZE))
  const currentInvitationPage = boundedPage(invitationPage, filteredInvitations.length, INVITATION_PAGE_SIZE)
  const visibleInvitations = filteredInvitations.slice((currentInvitationPage - 1) * INVITATION_PAGE_SIZE, currentInvitationPage * INVITATION_PAGE_SIZE)

  const filteredLogs = useMemo(() => {
    const query = logQuery.trim().toLowerCase()
    const rows = logs.filter((item) => {
      const roleMatch = logRole === 'all' || item.role === logRole
      const resultMatch = logResult === 'all' || (logResult === 'success' ? Boolean(item.success) : !item.success)
      const queryMatch = !query || [
        item.event,
        item.guest_name,
        item.identity_email,
        item.client_ip,
        item.role,
      ].some((value) => String(value || '').replaceAll('_', ' ').toLowerCase().includes(query))
      return roleMatch && resultMatch && queryMatch
    })
    return sortedRows(rows, logSort, (item, key) => {
      if (key === 'created_at') return item.created_at ? new Date(item.created_at).getTime() : null
      if (key === 'event') return String(item.event || '').replaceAll('_', ' ')
      if (key === 'role') return item.role ? roleLabel(item.role) : ''
      if (key === 'success') return item.success ? 1 : 0
      return item?.[key]
    })
  }, [logs, logQuery, logRole, logResult, logSort])

  const logPages = Math.max(1, Math.ceil(filteredLogs.length / LOG_PAGE_SIZE))
  const currentLogPage = boundedPage(logPage, filteredLogs.length, LOG_PAGE_SIZE)
  const visibleLogs = filteredLogs.slice((currentLogPage - 1) * LOG_PAGE_SIZE, currentLogPage * LOG_PAGE_SIZE)

  useEffect(() => {
    setInvitationPage(1)
  }, [invitationQuery, invitationRole, invitationStatus])

  useEffect(() => {
    setLogPage(1)
  }, [logQuery, logRole, logResult])

  return (
    <section className="page-stack administration-page administration-single-workspace">
      <section className="data-panel administration-workspace-panel">
        <header className="administration-workspace-header">
          <div className="administration-workspace-title">
            <div className="page-title-icon"><ShieldIcon size={22} /></div>
            <div>
              <h2>{tr("Administration")}</h2>
            </div>
          </div>
          <div className="administration-workspace-actions">
            <span>{tr(invitations.length === 1 ? '{count} access record' : '{count} access records', { count: invitations.length })}</span>
            <button type="button" className="secondary-button compact admin-refresh-button" onClick={loadData} disabled={loading || Boolean(busyId)}>{tr("Refresh")}</button>
          </div>
        </header>

        {error ? <div className="global-inline-message error-inline administration-workspace-message">{error}</div> : null}
        {notice ? <div className="global-inline-message success-inline administration-workspace-message">{notice}</div> : null}

        <section className="admin-workspace-metrics" aria-label={tr("Access summary")}>
          <AdminSummary icon={<AccessUsersIcon size={19} />} label={tr("Access records")} value={invitations.length} hint={ADMIN_HINTS.accessRecords} />
          <AdminSummary icon={<AccessLinkIcon size={19} />} label={tr("Pending")} value={counts.pending} hint={ADMIN_HINTS.pending} />
          <AdminSummary icon={<ClockIcon size={19} />} label={tr("Claimed / Active")} value={counts.active} tone="positive" hint={ADMIN_HINTS.active} />
          <AdminSummary icon={<AccessLockIcon size={19} />} label={tr("Expired / Restricted")} value={counts.expired + counts.restricted} tone="negative" hint={ADMIN_HINTS.restricted} />
        </section>

        <section className="admin-workspace-section admin-create-section">
          <div className="admin-section-heading">
            <div>
              <span className="panel-kicker">{tr("IDENTITY-VERIFIED ACCESS")}</span>
              <h2>{tr("Generate access invitation")}</h2>
            </div>
            <span className="admin-readonly-badge"><EyeIcon size={14} /> {tr("Google account required")}</span>
          </div>

          <form className="admin-invite-form identity-invite-form admin-workspace-invite-form" onSubmit={createInvitation}>
            <label>
              <AdminFieldLabel id="admin-hint-user-name" label={tr("User name")} hint={ADMIN_HINTS.guestName} />
              <input
                value={form.guest_name}
                onChange={(event) => setForm({ ...form, guest_name: event.target.value })}
                maxLength={120}
                required
              />
            </label>
            <label>
              <AdminFieldLabel id="admin-hint-authorized-email" label={tr("Authorized Google email")} hint={ADMIN_HINTS.authorizedEmail} />
              <input
                type="email"
                value={form.authorized_email}
                onChange={(event) => setForm({ ...form, authorized_email: event.target.value })}
                maxLength={254}
                autoComplete="off"
                required
              />
            </label>
            <label>
              <AdminFieldLabel id="admin-hint-role" label={tr("Role")} hint={ADMIN_HINTS.role} />
              <select
                value={form.role}
                onChange={(event) => {
                  const role = event.target.value
                  setForm({ ...form, role, max_active_sessions: defaultSessions(role) })
                }}
                required
              >
                <option value="viewer">{tr("Viewer")}</option>
                <option value="trader">{tr("Trader")}</option>
                <option value="admin">{tr("Administrator")}</option>
              </select>
            </label>
            <label>
              <AdminFieldLabel id="admin-hint-access-duration" label={tr("Access duration")} hint={ADMIN_HINTS.duration} />
              <select
                value={form.duration_seconds}
                onChange={(event) => setForm({ ...form, duration_seconds: event.target.value })}
                required
              >
                {DURATION_OPTIONS.map(([value, label]) => (
                  <option key={value} value={value}>{tr(label)}</option>
                ))}
              </select>
            </label>
            <label>
              <AdminFieldLabel id="admin-hint-maximum-sessions" label={tr("Maximum active sessions")} hint={ADMIN_HINTS.sessions} />
              <select
                value={form.max_active_sessions}
                onChange={(event) => setForm({ ...form, max_active_sessions: event.target.value })}
                required
              >
                {SESSION_OPTIONS.map((value) => <option key={value} value={value}>{value}</option>)}
              </select>
            </label>
            <button
              type="submit"
              className="admin-primary-button"
              disabled={busyId === 'create' || !form.duration_seconds}
            >
              <AccessLinkIcon size={17} />
              {tr(busyId === 'create' ? 'Generating…' : 'Generate verified link')}
            </button>
          </form>
        </section>

        <section className="admin-workspace-section admin-access-section">
          <div className="admin-section-heading admin-access-heading">
            <div>
              <span className="panel-kicker">{tr("ACCESS CONTROL & AUDIT")}</span>
              <h2>{tr("Access records")}</h2>
            </div>
            <div className="admin-data-tabs" role="tablist" aria-label={tr("Administration data view")}>
              <button type="button" role="tab" aria-selected={dataView === 'invitations'} className={dataView === 'invitations' ? 'active' : ''} onClick={() => setDataView('invitations')}>{tr("Invitations")}{' '}<span>{invitations.length}</span></button>
              <button type="button" role="tab" aria-selected={dataView === 'audit'} className={dataView === 'audit' ? 'active' : ''} onClick={() => setDataView('audit')}>{tr("Audit history")}{' '}<span>{logs.length}</span></button>
            </div>
          </div>

          {loading ? (
            <div className="admin-loading"><span className="loading-ring" />{tr("Loading administration…")}</div>
          ) : dataView === 'invitations' ? (
            <>
              <AdminListToolbar
                query={invitationQuery}
                onQueryChange={setInvitationQuery}
                placeholder={tr("Search user or Google identity")}
                count={filteredInvitations.length}
              >
                <select value={invitationStatus} onChange={(event) => setInvitationStatus(event.target.value)} aria-label={tr("Filter invitation status")}>
                  <option value="all">{tr("All statuses")}</option>
                  <option value="active">{tr("Claimed / Active")}</option>
                  <option value="pending">{tr("Pending")}</option>
                  <option value="expired">{tr("Expired")}</option>
                  <option value="restricted">{tr("Restricted")}</option>
                </select>
                <select value={invitationRole} onChange={(event) => setInvitationRole(event.target.value)} aria-label={tr("Filter invitation role")}>
                  <option value="all">{tr("All roles")}</option>
                  <option value="viewer">{tr("Viewer")}</option>
                  <option value="trader">{tr("Trader")}</option>
                  <option value="admin">{tr("Administrator")}</option>
                </select>
              </AdminListToolbar>

              <div className="admin-workspace-table-wrap">
                <table className="market-table admin-table identity-admin-table admin-sortable-table">
                  <thead>
                    <tr>
                      <AdminSortableTh label={tr("User")} field="guest_name" sort={invitationSort} onSort={(key) => setInvitationSort((current) => toggledSort(current, key))} hint={ADMIN_HINTS.guestName} />
                      <AdminSortableTh label={tr("Role")} field="role" sort={invitationSort} onSort={(key) => setInvitationSort((current) => toggledSort(current, key))} hint={ADMIN_HINTS.role} />
                      <AdminSortableTh label={tr("Status")} field="status" sort={invitationSort} onSort={(key) => setInvitationSort((current) => toggledSort(current, key))} hint={ADMIN_HINTS.status} />
                      <AdminSortableTh label={tr("Sessions")} field="sessions" sort={invitationSort} onSort={(key) => setInvitationSort((current) => toggledSort(current, key))} hint={ADMIN_HINTS.sessions} />
                      <AdminSortableTh label={tr("Claimed identity")} field="claimed_identity" sort={invitationSort} onSort={(key) => setInvitationSort((current) => toggledSort(current, key))} hint={ADMIN_HINTS.claimedIdentity} />
                      <AdminSortableTh label={tr("Expires")} field="expires_at" sort={invitationSort} onSort={(key) => setInvitationSort((current) => toggledSort(current, key))} hint={ADMIN_HINTS.expires} />
                      <AdminSortableTh label={tr("Last access")} field="last_access_at" sort={invitationSort} onSort={(key) => setInvitationSort((current) => toggledSort(current, key))} hint={ADMIN_HINTS.lastAccess} />
                      <th>{tr("Actions")}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {visibleInvitations.length === 0 ? (
                      <tr><td colSpan="8" className="empty-table-cell">{tr("No access records match the selected filters.")}</td></tr>
                    ) : visibleInvitations.map((item) => {
                      const legacy = item.status === 'legacy_unverified'
                      const primaryAdministrator = Boolean(item.primary_administrator)
                      const locked = ['revoked', 'legacy_unverified'].includes(item.status)
                      const cannotDelete = primaryAdministrator || ['pending_verification', 'claimed', 'active'].includes(item.status)
                      return (
                        <tr key={item.id}>
                          <td data-label={tr('User')}>
                            <strong>{item.guest_name}</strong>
                            <small className="admin-identity-email">{item.authorized_email || tr('No verified email')}</small>
                            {primaryAdministrator ? <small className="primary-administrator-label">{tr("Primary Google administrator")}</small> : null}
                          </td>
                          <td data-label={tr('Role')}>{roleLabel(item.role)}</td>
                          <td data-label={tr('Status')}><span className={`admin-status ${statusClass(item.status)}`}>{statusLabel(item.status)}</span></td>
                          <td data-label={tr('Sessions')}>
                            <div className="session-limit-control">
                              <strong>{item.active_sessions || 0}</strong>
                              <span>{tr("of")}</span>
                              <select
                                value={sessionLimits[item.id] ?? String(item.max_active_sessions || 1)}
                                onChange={(event) => setSessionLimits({ ...sessionLimits, [item.id]: event.target.value })}
                                disabled={locked}
                                aria-label={`Maximum sessions for ${item.guest_name}`}
                              >
                                {SESSION_OPTIONS.map((value) => <option key={value} value={value}>{value}</option>)}
                              </select>
                              <button type="button" onClick={() => saveSessionLimit(item)} disabled={Boolean(busyId) || locked}>{tr("Save")}</button>
                            </div>
                          </td>
                          <td data-label={tr('Claimed identity')}>
                            <span className="claimed-identity">{item.claimed_email || tr(legacy ? 'New invitation required' : 'Not claimed')}</span>
                            {item.claimed_at ? <small>{dateTime(item.claimed_at)}</small> : null}
                          </td>
                          <td data-label={tr('Expires')}>{dateTime(item.expires_at)}</td>
                          <td data-label={tr('Last access')}>{dateTime(item.last_access_at)}</td>
                          <td data-label={tr('Actions')}>
                            <div className="admin-row-actions identity-row-actions">
                              <button
                                type="button"
                                title={tr("End sessions, rotate the token and require a fresh Google identity claim.")}
                                onClick={() => regenerateAccessLink(item)}
                                disabled={Boolean(busyId) || locked || primaryAdministrator}
                              >
                                {tr("Generate new claim link")}</button>
                              <select
                                value={extendDurations[item.id] ?? DEFAULT_DURATION_SECONDS}
                                onChange={(event) => setExtendDurations({ ...extendDurations, [item.id]: event.target.value })}
                                disabled={locked || primaryAdministrator}
                                aria-label={`Duration for ${item.guest_name}`}
                              >
                                {DURATION_OPTIONS.map(([value, label]) => <option key={value} value={value}>+{tr(label)}</option>)}
                              </select>
                              <button type="button" onClick={() => extendInvitation(item)} disabled={Boolean(busyId) || locked || primaryAdministrator}>{tr("Extend")}</button>
                              <button
                                type="button"
                                onClick={() => runAction(item.id, 'terminate-sessions', tr('Sessions terminated for {name}.', { name: item.guest_name }))}
                                disabled={Boolean(busyId) || legacy}
                              >
                                {tr("End sessions")}</button>
                              <button
                                type="button"
                                className="danger"
                                onClick={() => runAction(item.id, 'revoke', tr('Access revoked for {name}.', { name: item.guest_name }))}
                                disabled={Boolean(busyId) || item.status === 'revoked' || legacy || primaryAdministrator}
                              >
                                {tr("Revoke")}</button>
                              <button
                                type="button"
                                className="danger ghost"
                                onClick={() => deleteInvitation(item)}
                                disabled={Boolean(busyId) || cannotDelete}
                              >
                                {tr("Delete")}</button>
                            </div>
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
              <AdminPagination page={currentInvitationPage} pages={invitationPages} total={filteredInvitations.length} pageSize={INVITATION_PAGE_SIZE} onPageChange={setInvitationPage} />
            </>
          ) : (
            <>
              <AdminListToolbar
                query={logQuery}
                onQueryChange={setLogQuery}
                placeholder={tr("Search event, user, identity or client")}
                count={filteredLogs.length}
              >
                <div className="admin-result-filters" aria-label={tr("Audit result filter")}>
                  <button type="button" className={logResult === 'all' ? 'active' : ''} onClick={() => setLogResult('all')} title={tr("Show all audit results")}><ListFilterIcon size={15} /> {tr("All")}</button>
                  <button type="button" className={`positive ${logResult === 'success' ? 'active' : ''}`} onClick={() => setLogResult('success')}>{tr("Success")}</button>
                  <button type="button" className={`negative ${logResult === 'denied' ? 'active' : ''}`} onClick={() => setLogResult('denied')}>{tr("Denied")}</button>
                </div>
                <select value={logRole} onChange={(event) => setLogRole(event.target.value)} aria-label={tr("Filter audit role")}>
                  <option value="all">{tr("All roles")}</option>
                  <option value="viewer">{tr("Viewer")}</option>
                  <option value="trader">{tr("Trader")}</option>
                  <option value="admin">{tr("Administrator")}</option>
                </select>
              </AdminListToolbar>

              <div className="admin-workspace-table-wrap">
                <table className="market-table access-log-table admin-sortable-table">
                  <thead>
                    <tr>
                      <AdminSortableTh label={tr("Time")} field="created_at" sort={logSort} onSort={(key) => setLogSort((current) => toggledSort(current, key))} hint={ADMIN_HINTS.auditTime} />
                      <AdminSortableTh label={tr("Event")} field="event" sort={logSort} onSort={(key) => setLogSort((current) => toggledSort(current, key))} hint={ADMIN_HINTS.auditEvent} />
                      <AdminSortableTh label={tr("User")} field="guest_name" sort={logSort} onSort={(key) => setLogSort((current) => toggledSort(current, key))} hint={ADMIN_HINTS.auditUser} />
                      <AdminSortableTh label={tr("Google identity")} field="identity_email" sort={logSort} onSort={(key) => setLogSort((current) => toggledSort(current, key))} hint={ADMIN_HINTS.auditIdentity} />
                      <AdminSortableTh label={tr("Role")} field="role" sort={logSort} onSort={(key) => setLogSort((current) => toggledSort(current, key))} hint={ADMIN_HINTS.auditRole} />
                      <AdminSortableTh label={tr("Result")} field="success" sort={logSort} onSort={(key) => setLogSort((current) => toggledSort(current, key))} hint={ADMIN_HINTS.auditResult} />
                      <AdminSortableTh label={tr("Client")} field="client_ip" sort={logSort} onSort={(key) => setLogSort((current) => toggledSort(current, key))} hint={ADMIN_HINTS.auditClient} />
                    </tr>
                  </thead>
                  <tbody>
                    {visibleLogs.length === 0 ? (
                      <tr><td colSpan="7" className="empty-table-cell">{tr("No access events match the selected filters.")}</td></tr>
                    ) : visibleLogs.map((item) => (
                      <tr key={item.id}>
                        <td>{dateTime(item.created_at)}</td>
                        <td>{tr(String(item.event || '').replaceAll('_', ' '))}</td>
                        <td>{item.guest_name || '—'}</td>
                        <td>{item.identity_email || '—'}</td>
                        <td>{item.role ? roleLabel(item.role) : '—'}</td>
                        <td className={item.success ? 'positive' : 'negative'}>{tr(item.success ? 'Success' : 'Denied')}</td>
                        <td>{item.client_ip}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <AdminPagination page={currentLogPage} pages={logPages} total={filteredLogs.length} pageSize={LOG_PAGE_SIZE} onPageChange={setLogPage} />
            </>
          )}
        </section>

        {generatedAccess ? (
          <AccessLinkDialog access={generatedAccess} onClose={() => setGeneratedAccess(null)} onError={setError} />
        ) : null}
      </section>
    </section>
  )
}
