import { getIntlLocale, tr } from '../../i18n/runtime'

export function dateTime(value) {
  if (!value) return '—'
  return new Date(value).toLocaleString(getIntlLocale())
}

export function statusClass(status) {
  if (status === 'active' || status === 'claimed') return 'positive'
  if (status === 'pending_verification') return 'pending'
  if (status === 'legacy_unverified') return 'warning'
  return 'negative'
}

export function statusLabel(status) {
  const labels = {
    pending_verification: 'Pending verification',
    claimed: 'Claimed',
    active: 'Active',
    legacy_unverified: 'Legacy unverified',
    expired: 'Expired',
    revoked: 'Revoked',
    blocked: 'Blocked',
  }
  return tr(labels[status] || String(status || 'Unknown').replaceAll('_', ' '))
}

export function roleLabel(value) {
  if (!value) return tr('Viewer')
  return tr(value.charAt(0).toUpperCase() + value.slice(1))
}

export function defaultSessions(role) {
  return ['trader', 'admin'].includes(role) ? '1' : '2'
}

export function compareValues(left, right) {
  if (left == null && right == null) return 0
  if (left == null) return -1
  if (right == null) return 1
  const leftNumber = Number(left)
  const rightNumber = Number(right)
  if (Number.isFinite(leftNumber) && Number.isFinite(rightNumber)) return leftNumber - rightNumber
  return String(left).localeCompare(String(right), undefined, { numeric: true, sensitivity: 'base' })
}

export function sortedRows(rows, sort, valueGetter = null) {
  const multiplier = sort.direction === 'asc' ? 1 : -1
  return [...rows].sort((left, right) => {
    const leftValue = valueGetter ? valueGetter(left, sort.key) : left?.[sort.key]
    const rightValue = valueGetter ? valueGetter(right, sort.key) : right?.[sort.key]
    return compareValues(leftValue, rightValue) * multiplier
  })
}

export function toggledSort(current, key) {
  if (current.key === key) return { key, direction: current.direction === 'asc' ? 'desc' : 'asc' }
  return { key, direction: 'asc' }
}

export function boundedPage(page, total, pageSize) {
  const pages = Math.max(1, Math.ceil(total / pageSize))
  return Math.min(Math.max(1, page), pages)
}

export async function copyText(value) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(value)
    return
  }

  const textarea = document.createElement('textarea')
  textarea.value = value
  textarea.setAttribute('readonly', '')
  textarea.style.position = 'fixed'
  textarea.style.opacity = '0'
  document.body.appendChild(textarea)
  textarea.select()
  const copied = document.execCommand('copy')
  textarea.remove()
  if (!copied) throw new Error(tr('The browser did not allow copying the link.'))
}
