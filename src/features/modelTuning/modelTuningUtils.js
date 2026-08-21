import { tr } from '../../i18n/runtime'

export function pct(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—'
  return `${(Number(value) * 100).toFixed(digits)}%`
}

export function money(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—'
  return new Intl.NumberFormat(undefined, { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(Number(value))
}

export function decimal(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—'
  return Number(value).toFixed(digits)
}

export function candidateLabel(candidate) {
  if (candidate.is_control) return tr('Current Strategy model')
  if (candidate.kind === 'probability_startup') return `${tr('Exploration candidate')} ${candidate.candidate_id}`
  if (candidate.kind === 'unified_exploration') return `${tr('Exploration candidate')} ${candidate.candidate_id}`
  if (candidate.kind === 'champion_probability') return `${tr('CARO candidate')} ${candidate.candidate_id}`
  return `${tr('Candidate')} ${candidate.candidate_id}`
}

export function numberOr(value, fallback = 0) {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : fallback
}
