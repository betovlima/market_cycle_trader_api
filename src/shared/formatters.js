import { getIntlLocale, tr } from '../i18n/runtime'

export function percent(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—'
  return new Intl.NumberFormat(getIntlLocale(), {
    style: 'percent',
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(Number(value))
}

export function money(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—'
  return new Intl.NumberFormat(getIntlLocale(), {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 2,
  }).format(Number(value))
}

export function number(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—'
  return new Intl.NumberFormat(getIntlLocale(), {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(Number(value))
}

function parseDate(value) {
  if (!value) return null
  if (value instanceof Date) return Number.isNaN(value.getTime()) ? null : value

  const raw = String(value).trim()
  const normalized = /^\d{4}-\d{2}-\d{2}$/.test(raw) ? `${raw}T00:00:00Z` : raw
  const date = new Date(normalized)
  return Number.isNaN(date.getTime()) ? null : date
}

export function compactDate(value) {
  const date = parseDate(value)
  if (!date) return ''
  return date.toLocaleDateString(getIntlLocale(), { month: 'short', year: '2-digit', timeZone: 'UTC' })
}

export function tradeDate(value) {
  if (!value) return '—'
  const raw = String(value).trim()
  if (/^\d{4}-\d{2}-\d{2}/.test(raw)) return raw.slice(0, 10)
  const date = parseDate(value)
  return date ? date.toISOString().slice(0, 10) : '—'
}

export function shortDateTime(value) {
  const date = parseDate(value)
  if (!date) return '—'
  return date.toLocaleString(getIntlLocale(), {
    year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', hour12: false,
  })
}

export function shortDate(value) {
  const date = parseDate(value)
  if (!date) return '—'
  return date.toLocaleDateString(getIntlLocale(), { month: 'short', day: '2-digit', timeZone: 'UTC' })
}

export function durationLabel(seconds) {
  const value = Number(seconds)
  if (!Number.isFinite(value) || value < 0) return '—'
  if (value < 60) return tr('{count} sec', { count: Math.round(value) })
  if (value < 3600) return tr('{minutes}m {seconds}s', {
    minutes: Math.floor(value / 60),
    seconds: Math.round(value % 60),
  })
  return tr('{hours}h {minutes}m', {
    hours: Math.floor(value / 3600),
    minutes: Math.round((value % 3600) / 60),
  })
}

export function relativeTime(value) {
  const date = parseDate(value)
  if (!date) return '—'
  const seconds = Math.round((date.getTime() - Date.now()) / 1000)
  const absolute = Math.abs(seconds)
  const formatter = new Intl.RelativeTimeFormat(getIntlLocale(), { numeric: 'auto' })
  if (absolute < 60) return formatter.format(seconds, 'second')
  if (absolute < 3600) return formatter.format(Math.round(seconds / 60), 'minute')
  if (absolute < 86400) return formatter.format(Math.round(seconds / 3600), 'hour')
  return formatter.format(Math.round(seconds / 86400), 'day')
}
