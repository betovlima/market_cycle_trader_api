import { getIntlLocale, tr } from '../../i18n/runtime'
import { strategyParameterLabel } from '../../i18n/strategyParameters'
import { STATUS_LABELS } from './strategySettingsConfig'

export function titleFromName(name) {
  return name.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase())
}

export function dateTime(value) {
  if (!value) return '—'
  return new Date(value).toLocaleString(getIntlLocale())
}

export function parameterRelationship(name, schema, reference) {
  const details = []
  const enumValues = Array.isArray(schema?.enum) ? schema.enum : []
  if (enumValues.length) details.push(`${tr('Allowed:')} ${enumValues.join(', ')}`)
  if (schema?.minimum !== undefined) details.push(`${tr('Minimum:')} ${schema.minimum}`)
  if (schema?.exclusiveMinimum !== undefined) details.push(`${tr('Greater than:')} ${schema.exclusiveMinimum}`)
  if (schema?.maximum !== undefined) details.push(`${tr('Maximum:')} ${schema.maximum}`)
  if (schema?.exclusiveMaximum !== undefined) details.push(`${tr('Less than:')} ${schema.exclusiveMaximum}`)
  if (name === 'assets') details.push(tr('Type: ticker symbols'))
  else if (typeof reference === 'boolean') details.push(tr('Type: on/off'))
  else if (Array.isArray(reference)) details.push(tr('Type: JSON array'))
  else if (typeof reference === 'number') details.push(tr(schema?.type === 'integer' ? 'Type: integer' : 'Type: number'))
  else details.push(tr('Type: text'))
  details.push(`${tr('Technical name:')} ${name}`)
  return details.join(' · ')
}

export function statusLabel(value) {
  return tr(STATUS_LABELS[String(value || 'draft')] || titleFromName(String(value || 'draft')))
}

export function lifecycleSummary(item, isWinner, isResearch) {
  if (isWinner) return tr('Active Winner')
  if (isResearch) return tr('Active Research Strategy')
  return tr('Saved Strategy')
}

export function strategyCatalogRank(item, winnerId, latestSavedId, researchId) {
  if (item.id === winnerId) return 0
  if (item.id === latestSavedId) return 1
  if (item.id === researchId) return 2
  return 3
}


export function resolveFieldSchema(schema, name) {
  const property = schema?.properties?.[name] || {}
  const resolve = (value) => {
    if (!value?.$ref) return value || {}
    const key = value.$ref.split('/').pop()
    return schema?.$defs?.[key] || value
  }
  if (property.$ref) return resolve(property)
  if (Array.isArray(property.anyOf)) {
    const candidate = property.anyOf.find((item) => item.type !== 'null') || property.anyOf[0]
    return { ...property, ...resolve(candidate) }
  }
  return property
}

export function toEditorValues(configuration) {
  return Object.fromEntries(Object.entries(configuration || {}).map(([name, value]) => {
    if (name === 'assets' && Array.isArray(value)) return [name, value.join(', ')]
    if (Array.isArray(value)) return [name, JSON.stringify(value)]
    if (value === null || value === undefined) return [name, '']
    if (typeof value === 'number') return [name, String(value)]
    return [name, value]
  }))
}

export function parseEditorValues(values, original) {
  const configuration = {}
  let assetsInput = null
  for (const [name, raw] of Object.entries(values)) {
    const reference = original[name]
    if (name === 'assets' && Array.isArray(reference)) {
      assetsInput = String(raw || '').trim()
    } else if (Array.isArray(reference)) {
      const parsed = JSON.parse(String(raw || '[]'))
      if (!Array.isArray(parsed)) throw new Error(tr('{field} must be a JSON array.', { field: strategyParameterLabel(name, titleFromName(name)) }))
      configuration[name] = parsed
    } else if (typeof reference === 'boolean') {
      configuration[name] = Boolean(raw)
    } else if (typeof reference === 'number') {
      const parsed = Number(raw)
      if (!Number.isFinite(parsed)) throw new Error(tr('{field} must be numeric.', { field: strategyParameterLabel(name, titleFromName(name)) }))
      configuration[name] = parsed
    } else if (reference === null) {
      configuration[name] = String(raw || '').trim() || null
    } else {
      configuration[name] = String(raw)
    }
  }
  return { configuration, assetsInput }
}
