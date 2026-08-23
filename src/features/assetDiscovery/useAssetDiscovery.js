import { useCallback, useEffect, useRef, useState } from 'react'

import { ApiError, apiFetch, downloadFile } from '../../api/http'
import { API, FRONT_VERSION } from '../../config/env'
import { tr } from '../../i18n/runtime'

const ACTIVE_STATUSES = new Set(['queued', 'running', 'stopping'])

export function useAssetDiscovery({ onSessionExpired }) {
  const [status, setStatus] = useState(null)
  const [candidates, setCandidates] = useState([])
  const [runs, setRuns] = useState([])
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const mountedRef = useRef(true)
  const previousActiveRef = useRef(false)

  const handleError = useCallback((requestError) => {
    if (requestError instanceof ApiError && requestError.status === 401) {
      onSessionExpired?.()
      return
    }
    setError(tr(requestError?.message || 'Unable to load Asset Discovery.'))
  }, [onSessionExpired])

  const loadStatus = useCallback(async () => {
    try {
      const response = await apiFetch(`${API}/admin/asset-discovery/status`)
      if (mountedRef.current) setStatus(response)
    } catch (requestError) {
      if (mountedRef.current) handleError(requestError)
    }
  }, [handleError])

  const load = useCallback(async ({ silent = false } = {}) => {
    if (!silent) setLoading(true)
    try {
      const [statusResponse, candidateResponse, runResponse] = await Promise.all([
        apiFetch(`${API}/admin/asset-discovery/status`),
        apiFetch(`${API}/admin/asset-discovery/candidates?limit=500`),
        apiFetch(`${API}/admin/asset-discovery/runs?limit=30`),
      ])
      if (!mountedRef.current) return
      setStatus(statusResponse)
      setCandidates(candidateResponse?.items || [])
      setRuns(runResponse?.items || [])
      setError('')
    } catch (requestError) {
      if (mountedRef.current) handleError(requestError)
    } finally {
      if (mountedRef.current && !silent) setLoading(false)
    }
  }, [handleError])

  useEffect(() => {
    mountedRef.current = true
    load()
    return () => { mountedRef.current = false }
  }, [load])

  const active = ACTIVE_STATUSES.has(String(status?.run?.status || '').toLowerCase())

  useEffect(() => {
    const refresh = () => {
      if (active) {
        load({ silent: true })
      } else {
        loadStatus()
      }
    }
    const interval = window.setInterval(refresh, active ? 2_500 : 30_000)
    return () => window.clearInterval(interval)
  }, [active, load, loadStatus])

  useEffect(() => {
    if (previousActiveRef.current && !active) load({ silent: true })
    previousActiveRef.current = active
  }, [active, load])

  const runAction = useCallback(async (action) => {
    setBusy(action)
    setError('')
    setNotice('')
    try {
      await apiFetch(`${API}/admin/asset-discovery/${action}`, { method: 'POST' })
      setNotice(tr(action === 'start' ? 'Asset Discovery started.' : 'Stop requested.'))
      await load({ silent: true })
    } catch (requestError) {
      handleError(requestError)
    } finally {
      setBusy('')
    }
  }, [handleError, load])


  const exportAnalysis = useCallback(async () => {
    setBusy('export')
    setError('')
    setNotice('')
    try {
      const fallback = `asset_discovery_analysis_${new Date().toISOString().replace(/[-:]/g, '').slice(0, 15)}Z.json`
      await downloadFile(
        `${API}/admin/asset-discovery/export?front_version=${encodeURIComponent(FRONT_VERSION)}`,
        fallback,
      )
      setNotice(tr('Asset Discovery analysis exported.'))
    } catch (requestError) {
      handleError(requestError)
    } finally {
      setBusy('')
    }
  }, [handleError])

  const saveSettings = useCallback(async (form) => {
    if (!status?.settings?.revision) return
    setBusy('settings')
    setError('')
    setNotice('')
    try {
      await apiFetch(`${API}/admin/asset-discovery/settings`, {
        method: 'PATCH',
        body: {
          expected_revision: status.settings.revision,
          reason: form.reason || 'Asset Discovery schedule updated.',
          settings: {
            automatic_enabled: form.automatic_enabled,
            batch_size: Number(form.batch_size),
            schedule_hours_et: form.schedule_hours_et,
            recheck_days: Number(form.recheck_days),
          },
        },
      })
      setNotice(tr('Asset Discovery settings saved.'))
      await load({ silent: true })
    } catch (requestError) {
      handleError(requestError)
    } finally {
      setBusy('')
    }
  }, [handleError, load, status?.settings?.revision])

  return {
    status,
    candidates,
    runs,
    loading,
    busy,
    active,
    error,
    notice,
    load,
    start: () => runAction('start'),
    stop: () => runAction('stop'),
    exportAnalysis,
    saveSettings,
  }
}
