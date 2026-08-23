import { getIntlLocale } from '../../../i18n/runtime'
export const ANALYTICS_ZOOM_STEP = 0.84
export const ANALYTICS_MIN_ZOOM_POINTS = 8
export const ANALYTICS_DAY_MS = 24 * 60 * 60 * 1000

export function analyticsTimestamp(value) {
  if (!value) return null
  const parsed = new Date(value).getTime()
  return Number.isFinite(parsed) ? parsed : null
}

export function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value))
}

export function minimumAnalyticsZoomSpan(points) {
  if (points.length < 2) return 0
  const gaps = []
  for (let index = 1; index < points.length; index += 1) {
    const gap = Number(points[index].timestamp_value) - Number(points[index - 1].timestamp_value)
    if (Number.isFinite(gap) && gap > 0) gaps.push(gap)
  }
  if (!gaps.length) return 0
  gaps.sort((left, right) => left - right)
  const medianGap = gaps[Math.floor(gaps.length / 2)]
  return medianGap * Math.max(2, ANALYTICS_MIN_ZOOM_POINTS - 1)
}

export function analyticsAxisLabel(value, visibleSpan) {
  const date = new Date(Number(value))
  if (Number.isNaN(date.getTime())) return ''
  if (visibleSpan <= ANALYTICS_DAY_MS * 2) {
    return date.toLocaleTimeString(getIntlLocale(), { hour: '2-digit', minute: '2-digit', hour12: false })
  }
  if (visibleSpan <= ANALYTICS_DAY_MS * 60) {
    return date.toLocaleDateString(getIntlLocale(), { month: 'short', day: 'numeric' })
  }
  return date.toLocaleDateString(getIntlLocale(), { month: 'short', year: '2-digit' })
}

export function monthParts(value) {
  if (!value) return null
  const direct = String(value).match(/^(\d{4})[-/](\d{1,2})/)
  if (direct) return { year: Number(direct[1]), month: Number(direct[2]) }
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return null
  return { year: parsed.getFullYear(), month: parsed.getMonth() + 1 }
}

export function returnTone(value, epsilon = 1e-12) {
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return ''
  if (Math.abs(numeric) <= epsilon) return 'neutral'
  return numeric > 0 ? 'positive' : 'negative'
}
