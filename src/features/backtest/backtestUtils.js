import { getIntlLocale } from '../../i18n/runtime'
import { DAY_MS } from './backtestConfig'

export function backtestAxisLabel(value, visibleSpan) {
  const date = new Date(Number(value))
  if (Number.isNaN(date.getTime())) return ''
  if (visibleSpan <= DAY_MS * 2) {
    return date.toLocaleTimeString(getIntlLocale(), { hour: '2-digit', minute: '2-digit', hour12: false })
  }
  if (visibleSpan <= DAY_MS * 45) {
    return date.toLocaleDateString(getIntlLocale(), { month: 'short', day: 'numeric' })
  }
  return date.toLocaleDateString(getIntlLocale(), { month: 'short', year: '2-digit' })
}

export function compareValues(left, right) {
  if (left == null && right == null) return 0
  if (left == null) return 1
  if (right == null) return -1
  if (typeof left === 'number' && typeof right === 'number') return left - right
  return String(left).localeCompare(String(right), undefined, { numeric: true, sensitivity: 'base' })
}

export function sortRows(rows, sort, accessors) {
  const accessor = accessors[sort.key] || ((item) => item?.[sort.key])
  const direction = sort.direction === 'asc' ? 1 : -1
  return [...rows].sort((left, right) => direction * compareValues(accessor(left), accessor(right)))
}

export function toggleSort(current, key) {
  if (current.key !== key) return { key, direction: 'desc' }
  return { key, direction: current.direction === 'desc' ? 'asc' : 'desc' }
}
