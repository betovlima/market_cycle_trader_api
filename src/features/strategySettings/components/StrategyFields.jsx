import { tr } from '../../../i18n/runtime'
import { strategyParameterLabel } from '../../../i18n/strategyParameters'
import { memo } from 'react'

import { ParameterHint } from '../../../shared/components/ParameterHint'
import { parameterRelationship, titleFromName } from '../strategySettingsUtils'

export function StrategyFieldLabel({ id, label, hint, align = 'left' }) {
  return (
    <span className="strategy-field-label-with-hint">
      <span>{tr(label)}</span>
      <ParameterHint id={id} title={tr(label)} align={align} {...hint} />
    </span>
  )
}
export const ParameterField = memo(function ParameterField({ name, value, reference, schema, hintAlign = 'left', disabled, onChange }) {
  const label = strategyParameterLabel(name, schema?.title || titleFromName(name))
  const hint = {
    description: schema?.description ? tr(schema.description) : tr('Controls the {label} value used by this protected research configuration.', { label: tr(label).toLocaleLowerCase() }),
    relationship: parameterRelationship(name, schema, reference),
  }
  const fieldHeading = (
    <span className="strategy-field-heading">
      <span className="strategy-field-label-with-hint">
        <span>{tr(label)}</span>
        <ParameterHint id={`hint-strategy-parameter-${name}`} title={tr(label)} align={hintAlign} {...hint} />
      </span>
      <code>{name}</code>
    </span>
  )
  const enumValues = Array.isArray(schema?.enum) ? schema.enum : []
  if (enumValues.length) {
    return (
      <label>
        {fieldHeading}
        <select value={value ?? ''} disabled={disabled} onChange={(event) => onChange(name, event.target.value)}>
          {enumValues.map((option) => <option key={String(option)} value={option}>{String(option)}</option>)}
        </select>
      </label>
    )
  }
  if (typeof reference === 'boolean') {
    return (
      <label className="strategy-boolean-field">
        {fieldHeading}
        <input type="checkbox" checked={Boolean(value)} disabled={disabled} onChange={(event) => onChange(name, event.target.checked)} />
      </label>
    )
  }
  if (name === 'assets' && Array.isArray(reference)) {
    return (
      <label className="strategy-asset-field">
        {fieldHeading}
        <textarea
          value={value ?? ''}
          disabled={disabled}
          rows="2"
          spellCheck="false"
          autoComplete="off"
          autoCapitalize="characters"
          placeholder={tr('NVDA, AAPL, MSFT or one symbol per line')}
          onChange={(event) => onChange(name, event.target.value)}
        />
        <small>{tr('Enter ticker symbols separated by commas, spaces, semicolons or line breaks. The API normalizes the symbols, removes duplicates and builds the final asset list.')}</small>
      </label>
    )
  }
  if (Array.isArray(reference)) {
    return (
      <label className="strategy-array-field">
        {fieldHeading}
        <textarea value={value ?? ''} disabled={disabled} rows="2" spellCheck="false" onChange={(event) => onChange(name, event.target.value)} />
      </label>
    )
  }
  if (typeof reference === 'number') {
    return (
      <label>
        {fieldHeading}
        <input
          type="number"
          value={value ?? ''}
          disabled={disabled}
          step={schema?.type === 'integer' ? '1' : 'any'}
          min={schema?.minimum}
          max={schema?.maximum}
          onChange={(event) => onChange(name, event.target.value)}
          required
        />
      </label>
    )
  }
  return (
    <label>
      {fieldHeading}
      <input value={value ?? ''} disabled={disabled} onChange={(event) => onChange(name, event.target.value)} />
    </label>
  )
})
