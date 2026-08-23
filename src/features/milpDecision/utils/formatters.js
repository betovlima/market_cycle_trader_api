export function money(value) {
  const number = Number(value)
  if (!Number.isFinite(number)) return '—'
  return new Intl.NumberFormat(undefined, { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(number)
}

export function number(value, digits = 2) {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return '—'
  return parsed.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits })
}

export function percent(value, digits = 2) {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return '—'
  return `${(parsed * 100).toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits })}%`
}

export function actionLabel(value) {
  const text = String(value || '').toUpperCase()
  return text || '—'
}
