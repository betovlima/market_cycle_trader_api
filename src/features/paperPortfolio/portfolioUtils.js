import { getIntlLocale, tr } from '../../i18n/runtime'
import { compactDate } from '../../shared/formatters'
import { timestampValue } from '../../shared/charts/timeSeries'
import { DAY_MS } from './portfolioConfig'

export function countdownLabel(totalSeconds) {
  const safeSeconds = Math.max(0, Number(totalSeconds) || 0)
  const hours = Math.floor(safeSeconds / 3600)
  const minutes = Math.floor((safeSeconds % 3600) / 60)
  const seconds = safeSeconds % 60
  return [hours, minutes, seconds].map((value) => String(value).padStart(2, '0')).join(':')
}

export function scheduleValue(targetAt, now, running = false) {
  if (running) return tr('Running now')
  const target = timestampValue(targetAt)
  if (target === null) return tr('Pending')
  const remaining = Math.max(0, Math.ceil((target - now) / 1000))
  return remaining === 0 ? tr('Due now') : countdownLabel(remaining)
}

export function portfolioAxisLabel(value, visibleSpan) {
  const date = new Date(Number(value))
  if (Number.isNaN(date.getTime())) return ''
  if (visibleSpan <= DAY_MS * 2) {
    return date.toLocaleTimeString(getIntlLocale(), { hour: '2-digit', minute: '2-digit', hour12: false })
  }
  if (visibleSpan <= DAY_MS * 14) {
    const day = date.toLocaleDateString(getIntlLocale(), { month: 'short', day: 'numeric' })
    const time = date.toLocaleTimeString(getIntlLocale(), { hour: '2-digit', minute: '2-digit', hour12: false })
    return `${day} ${time}`
  }
  return compactDate(date)
}

export function portfolioMeasureInterval(startTimestamp, endTimestamp) {
  const elapsed = Math.abs(Number(endTimestamp) - Number(startTimestamp))
  if (!Number.isFinite(elapsed)) return '—'

  const days = Math.floor(elapsed / DAY_MS)
  const hours = Math.floor((elapsed % DAY_MS) / (60 * 60 * 1000))
  const minutes = Math.floor((elapsed % (60 * 60 * 1000)) / (60 * 1000))

  if (days > 0) return `${days}d ${hours}h`
  if (hours > 0) return `${hours}h ${minutes}m`
  return `${minutes}m`
}
