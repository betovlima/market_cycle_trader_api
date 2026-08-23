import { apiFetch } from '../../../api/http'
import { API } from '../../../config/env'

function query(processingId, startMonth, endMonth) {
  return new URLSearchParams({ processing_id: processingId, start_month: startMonth, end_month: endMonth }).toString()
}

export function fetchLatestMilpDecision(runId, processingId, startMonth, endMonth) {
  return apiFetch(`${API}/temporal-intelligence/${encodeURIComponent(runId)}/decision-optimization/latest?${query(processingId, startMonth, endMonth)}`)
}

export function runMilpDecision(runId, processingId, startMonth, endMonth) {
  return apiFetch(`${API}/temporal-intelligence/${encodeURIComponent(runId)}/decision-optimization`, {
    method: 'POST',
    body: { processing_id: processingId, start_month: startMonth, end_month: endMonth },
  })
}

export function materializeMilpDecision(runId, optimizationId) {
  return apiFetch(`${API}/temporal-intelligence/${encodeURIComponent(runId)}/decision-optimization/${encodeURIComponent(optimizationId)}/strategy`, { method: 'POST' })
}
