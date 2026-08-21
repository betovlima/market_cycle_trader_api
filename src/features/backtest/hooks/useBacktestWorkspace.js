import { useCallback, useEffect, useState } from 'react'

import { apiFetch } from '../../../api/http'
import { API } from '../../../config/env'

const ACTIVE_JOB_STATUSES = new Set(['queued', 'running'])
const TERMINAL_FAILURE_STATUSES = new Set(['failed', 'interrupted'])

export function useBacktestWorkspace() {
  const [job, setJob] = useState(null)
  const [detail, setDetail] = useState(null)
  const [dashboard, setDashboard] = useState(null)
  const [error, setError] = useState('')
  const [loadingDetail, setLoadingDetail] = useState(false)
  const [loadingDashboard, setLoadingDashboard] = useState(true)
  const [restoringExecution, setRestoringExecution] = useState(true)
  const [startingBacktest, setStartingBacktest] = useState(false)
  const [apiVersion, setApiVersion] = useState('…')

  const running = Boolean(job && ACTIVE_JOB_STATUSES.has(job.status))
  const startDisabled = restoringExecution || startingBacktest || running

  const refreshDashboard = useCallback(async () => {
    setLoadingDashboard(true)
    try {
      const payload = await apiFetch(`${API}/dashboard/summary?limit=50`)
      setDashboard(payload)
      return payload
    } catch (requestError) {
      setError(requestError.message)
      return null
    } finally {
      setLoadingDashboard(false)
    }
  }, [])

  const loadDetail = useCallback(async (jobId) => {
    if (!jobId) return null
    setLoadingDetail(true)
    try {
      const payload = await apiFetch(`${API}/dashboard/jobs/${jobId}`)
      setDetail(payload)
      setJob((current) => current?.id === jobId ? { ...current, ...payload } : payload)
      return payload
    } catch (requestError) {
      setError(requestError.message)
      return null
    } finally {
      setLoadingDetail(false)
    }
  }, [])

  const loadLatestJob = useCallback(async () => {
    try {
      return await apiFetch(`${API}/jobs/latest`)
    } catch (requestError) {
      if (!String(requestError.message || '').includes('404')) {
        setError(requestError.message)
      }
      return null
    }
  }, [])

  useEffect(() => {
    let cancelled = false

    async function bootstrap() {
      setRestoringExecution(true)
      try {
        try {
          const health = await apiFetch(`${API}/health`)
          if (!cancelled) setApiVersion(health.api_version || 'unknown')
        } catch {
          if (!cancelled) setApiVersion('unavailable')
        }

        const [summary, latestJob] = await Promise.all([
          refreshDashboard(),
          loadLatestJob(),
        ])
        if (cancelled) return

        const latest = latestJob || summary?.recent_backtests?.[0] || null
        if (!latest) return

        setJob(latest)
        if (latest.status === 'completed') {
          await loadDetail(latest.id)
        } else if (!ACTIVE_JOB_STATUSES.has(latest.status)) {
          const completed = summary?.recent_backtests?.find((item) => item.status === 'completed')
          if (completed) await loadDetail(completed.id)
        }
      } finally {
        if (!cancelled) setRestoringExecution(false)
      }
    }

    bootstrap()
    return () => {
      cancelled = true
    }
  }, [loadDetail, loadLatestJob, refreshDashboard])

  useEffect(() => {
    if (!running || !job?.id) return undefined

    let cancelled = false
    let timerId = null

    async function poll() {
      try {
        const updated = await apiFetch(`${API}/jobs/${job.id}`)
        if (cancelled) return
        setJob(updated)

        if (updated.status === 'completed') {
          await loadDetail(updated.id)
          await refreshDashboard()
          return
        }

        if (TERMINAL_FAILURE_STATUSES.has(updated.status)) {
          await refreshDashboard()
          return
        }

        timerId = window.setTimeout(poll, 3000)
      } catch (requestError) {
        if (!cancelled) {
          setError(requestError.message)
          timerId = window.setTimeout(poll, 5000)
        }
      }
    }

    poll()
    return () => {
      cancelled = true
      if (timerId) window.clearTimeout(timerId)
    }
  }, [job?.id, loadDetail, refreshDashboard, running])

  async function runBacktest() {
    if (startDisabled) return null

    setError('')
    setDetail(null)
    setStartingBacktest(true)
    try {
      const latest = await loadLatestJob()
      if (latest && ACTIVE_JOB_STATUSES.has(latest.status)) {
        setJob(latest)
        return latest
      }

      const created = await apiFetch(`${API}/jobs`, { method: 'POST' })
      setJob(created)
      return created
    } catch (requestError) {
      setError(requestError.message)
      const latest = await loadLatestJob()
      if (latest && ACTIVE_JOB_STATUSES.has(latest.status)) setJob(latest)
      return null
    } finally {
      setStartingBacktest(false)
    }
  }

  return {
    job,
    detail,
    dashboard,
    error,
    setError,
    apiVersion,
    running,
    restoringExecution,
    startingBacktest,
    startDisabled,
    loadingDetail,
    loadingDashboard,
    runBacktest,
    refreshDashboard,
  }
}
