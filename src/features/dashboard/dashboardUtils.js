import { getIntlLocale } from '../../i18n/runtime'
import { timestampValue } from '../../shared/charts/timeSeries'
import { DAY_MS } from './dashboardConfig'

export function dashboardAxisLabel(value, visibleSpan) {
  const date = new Date(Number(value))
  if (Number.isNaN(date.getTime())) return ''
  if (visibleSpan <= DAY_MS * 2) return date.toLocaleTimeString(getIntlLocale(), { hour: '2-digit', minute: '2-digit', hour12: false })
  if (visibleSpan <= DAY_MS * 60) return date.toLocaleDateString(getIntlLocale(), { month: 'short', day: 'numeric' })
  return date.toLocaleDateString(getIntlLocale(), { month: 'short', year: '2-digit' })
}

export function dashboardSortValue(item, key) {
  if (key === 'date') return new Date(item?.created_at || 0).getTime() || 0
  if (key === 'status') return String(item?.status || '').toLocaleLowerCase()
  if (key === 'return') return Number(item?.metrics?.simulation_return ?? Number.NEGATIVE_INFINITY)
  if (key === 'sharpe') return Number(item?.metrics?.sharpe ?? Number.NEGATIVE_INFINITY)
  if (key === 'drawdown') return Number(item?.metrics?.maximum_drawdown ?? Number.NEGATIVE_INFINITY)
  if (key === 'rotations') return Number(item?.metrics?.position_changes ?? Number.NEGATIVE_INFINITY)
  if (key === 'duration') return Number(item?.duration_seconds ?? Number.NEGATIVE_INFINITY)
  return ''
}

export function statusMatchesFilter(status, filter) {
  const value = String(status || '').toLocaleLowerCase()
  if (filter === 'all') return true
  if (filter === 'active') return value === 'running' || value === 'queued'
  return value === filter
}

export function tradeTone(value) {
  const number = Number(value)
  if (!Number.isFinite(number) || number === 0) return 'flat'
  return number > 0 ? 'profit' : 'loss'
}

export function decimal(value, digits = 4) {
  const number = Number(value)
  return Number.isFinite(number) ? number.toFixed(digits) : '—'
}

export function strategyValue(value) {
  if (value === null || value === undefined) return '—'
  if (Array.isArray(value)) return value.join(', ')
  if (typeof value === 'object') return JSON.stringify(value)
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  return String(value)
}

export function buildCashIntervals(rows) {
  const intervals = []
  let start = null
  let end = null
  rows.forEach((row) => {
    const timestamp = timestampValue(row.decision_date || row.timestamp)
    const asset = String(row.final_action_asset || row.selected_asset || '').toUpperCase()
    if (timestamp === null) return
    if (asset === 'CASH') {
      if (start === null) start = timestamp
      end = timestamp
      return
    }
    if (start !== null) {
      intervals.push({ start, end: end ?? start })
      start = null
      end = null
    }
  })
  if (start !== null) intervals.push({ start, end: end ?? start })
  return intervals
}
