import { useEffect, useMemo, useState } from 'react'

import { apiFetch } from '../../../api/http'
import { API } from '../../../config/env'
import { tr } from '../../../i18n/runtime'
import { shortDateTime } from '../../../shared/formatters'
import { ParameterHint } from '../../../shared/components/ParameterHint'
import { MonthlyCapitalMovementHeatmap } from '../../backtest/components/RotationPanel'
import { DashboardMonthlyReturnConsistencySection } from './DashboardMonthlyReturnConsistencySection'

function completedRows(rows) {
  return (Array.isArray(rows) ? rows : [])
    .filter((item) => String(item?.status || '').toLowerCase() === 'completed')
    .sort((left, right) => String(right?.created_at || right?.finished_at || '').localeCompare(String(left?.created_at || left?.finished_at || '')))
}

function optionLabel(item) {
  const name = item?.processing_label || item?.strategy_profile_name || item?.strategy_name || item?.id || tr('Backtest')
  const model = item?.processing_kind === 'caro_champion' ? '' : (item?.research_model_label || item?.model_label || '')
  const at = item?.created_at || item?.finished_at
  return [at ? shortDateTime(at) : null, name, model].filter(Boolean).join(' · ')
}

export function DashboardBacktestAnalyticsSection({ fallbackJobs = [], initialProcessingId = "" }) {
  const [jobs, setJobs] = useState([])
  const [jobId, setJobId] = useState('')
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    let active = true
    apiFetch(`${API}/analytics/processings?limit=200`)
      .then((payload) => {
        if (!active) return
        setJobs(completedRows(payload?.items || []))
      })
      .catch(() => {
        if (active) setJobs([])
      })
    return () => { active = false }
  }, [])

  const history = useMemo(() => {
    const primary = completedRows(jobs)
    return primary.length ? primary : completedRows(fallbackJobs).map((item) => ({ ...item, processing_kind: 'backtest', processing_label: tr('Backtest') }))
  }, [fallbackJobs, jobs])

  useEffect(() => {
    if (!history.length) {
      setJobId('')
      return
    }
    const requested = String(initialProcessingId || '')
    if (requested && history.some((item) => String(item.id) === requested) && String(jobId) !== requested) {
      setJobId(requested)
      return
    }
    if (!jobId || !history.some((item) => String(item.id) === String(jobId))) {
      setJobId(String(history[0].id))
    }
  }, [history, initialProcessingId, jobId])

  useEffect(() => {
    if (!jobId) {
      setData(null)
      return undefined
    }
    let active = true
    setLoading(true)
    setError('')
    apiFetch(`${API}/analytics/processings/${encodeURIComponent(jobId)}`)
      .then((payload) => { if (active) setData(payload) })
      .catch((requestError) => {
        if (!active) return
        setData(null)
        setError(requestError.message || tr('Unable to load backtest analytics.'))
      })
      .finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [jobId])

  return (
    <section className="dashboard-analytics-hub">
      <div className="dashboard-analytics-history-head">
        <div>
          <span className="panel-kicker">{tr('BACKTEST ANALYTICS')}</span>
          <div className="dashboard-historical-performance-title">
            <h2>{tr('Historical performance')}</h2>
            <ParameterHint
              id="dashboard-historical-performance-hint"
              title={tr('Historical performance')}
              description={tr('Choose a completed Backtest or validated CARO Champion to inspect its consolidated historical performance. Changing the selection updates both charts without running new research or modifying the Strategy.')}
              details={[
                { label: tr('Default'), value: tr('Latest completed processing') },
                { label: tr('Backtest Analytics'), value: tr('Monthly capital movement') },
                { label: tr('Return and consistency'), value: tr('Monthly return heatmap') },
                { label: tr('Scope'), value: tr('Selected processing only') },
              ]}
            />
          </div>
        </div>
        <label className="dashboard-analytics-run-select">
          <span>{tr('Processing')}</span>
          <select value={jobId} onChange={(event) => setJobId(event.target.value)} disabled={!history.length || loading}>
            {history.map((item) => <option key={item.id} value={item.id}>{optionLabel(item)}</option>)}
          </select>
          <small>{tr('Latest completed processing is selected by default.')}</small>
        </label>
      </div>

      {loading ? <div className="backtest-loading-row">{tr('Loading backtest analytics…')}</div> : null}
      {error ? <div className="global-inline-message error-inline">{tr(error)}</div> : null}
      {!loading && !error && !data ? <div className="global-inline-message">{tr('No completed processing is available for analytics.')}</div> : null}

      {data ? (
        <div className="dashboard-analytics-charts">
          <MonthlyCapitalMovementHeatmap jobId={jobId} processingId={jobId} rotations={data.rotations || []} equity={data.equity || []} allowDrilldown />
          <DashboardMonthlyReturnConsistencySection data={data} />
        </div>
      ) : null}
    </section>
  )
}
