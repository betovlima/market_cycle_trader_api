import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { ApiError, apiFetch } from '../api/http'
import { API } from '../config/env'
import { tr } from '../i18n/runtime'
import { ParameterHint } from '../shared/components/ParameterHint'

function valuesFromModel(model) {
  return Object.fromEntries((model?.fields || []).map((field) => [field.name, field.value]))
}

function valuesEqual(left, right) {
  return JSON.stringify(left || {}) === JSON.stringify(right || {})
}

function inputValue(field, value) {
  if (field.type === 'boolean') return Boolean(value)
  if (value === null || value === undefined) return ''
  return String(value)
}

function normalizedValue(field, value) {
  if (field.type === 'boolean') return Boolean(value)
  if (field.type === 'integer') return Number.parseInt(value, 10)
  return Number(value)
}

function ModelField({ field, value, disabled, onChange }) {
  const hint = {
    description: field.description || '',
    relationship: tr('This value is saved with the selected Strategy revision and frozen into every Backtest created from it.'),
  }
  if (field.type === 'boolean') {
    return (
      <div className="model-research-field model-research-toggle-field">
        <div className="model-research-field-label">
          <span>{tr(field.label)}</span>
          {field.description ? <ParameterHint id={`model-research-${field.name}`} title={field.label} {...hint} /> : null}
        </div>
        <label className="settings-toggle-switch" htmlFor={`model-research-input-${field.name}`}>
          <input
            id={`model-research-input-${field.name}`}
            type="checkbox"
            checked={Boolean(value)}
            disabled={disabled}
            onChange={(event) => onChange(event.target.checked)}
            aria-label={tr(field.label)}
          />
          <i aria-hidden="true" />
        </label>
      </div>
    )
  }

  const min = field.min ?? field.exclusive_min
  const max = field.max ?? field.exclusive_max
  return (
    <div className="model-research-field">
      <div className="model-research-field-label">
        <span>{tr(field.label)}</span>
        {field.description ? <ParameterHint id={`model-research-${field.name}`} title={field.label} {...hint} /> : null}
      </div>
      <input
        id={`model-research-input-${field.name}`}
        type="number"
        value={inputValue(field, value)}
        min={min ?? undefined}
        max={max ?? undefined}
        step={field.type === 'integer' ? 1 : 'any'}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
        required
      />
    </div>
  )
}

export function ModelResearchSettingsPanel({
  onSessionExpired,
  embedded = false,
  strategy = null,
  onStrategyModelSaved = null,
  onDirtyChange = null,
  parameterSearch = '',
  onSearchMatchCount = null,
  readOnly = false,
}) {
  const [payload, setPayload] = useState(null)
  const [selectedModelId, setSelectedModelId] = useState('')
  const [formValues, setFormValues] = useState({})
  const [reason, setReason] = useState('')
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const initialLoadStartedRef = useRef(false)

  const savedModel = strategy?.research_model_configuration || strategy?.research_model || null
  const savedModelFamily = savedModel?.family || ''
  const savedValues = strategy?.research_model_configuration?.values || {}

  const handleError = useCallback((requestError) => {
    if (requestError instanceof ApiError && requestError.status === 401) {
      onSessionExpired()
      return
    }
    setError(tr(requestError.message || 'Unable to manage Strategy model settings.'))
  }, [onSessionExpired])

  const initializeFromPayload = useCallback((nextPayload, family = '') => {
    setPayload(nextPayload)
    const models = nextPayload?.models || []
    const targetId = family && models.some((item) => item.id === family)
      ? family
      : savedModelFamily && models.some((item) => item.id === savedModelFamily)
        ? savedModelFamily
        : models[0]?.id || ''
    setSelectedModelId(targetId)
    const target = models.find((item) => item.id === targetId)
    setFormValues(targetId === savedModelFamily && Object.keys(savedValues).length ? { ...savedValues } : valuesFromModel(target))
  }, [savedModelFamily, savedValues])

  const loadData = useCallback(async () => {
    setLoading(true)
    try {
      const settingsResponse = await apiFetch(`${API}/admin/model-research/settings`)
      initializeFromPayload(settingsResponse)
      setError('')
    } catch (requestError) {
      handleError(requestError)
    } finally {
      setLoading(false)
    }
  }, [handleError, initializeFromPayload])

  useEffect(() => {
    if (initialLoadStartedRef.current) return
    initialLoadStartedRef.current = true
    loadData()
  }, [loadData])

  useEffect(() => {
    if (!payload) return
    initializeFromPayload(payload, savedModelFamily)
    setReason('')
    setNotice('')
    setError('')
  }, [strategy?.id, strategy?.revision, savedModelFamily])

  const selectedModel = useMemo(
    () => (payload?.models || []).find((item) => item.id === selectedModelId) || null,
    [payload, selectedModelId],
  )

  const visibleModelFields = useMemo(() => {
    const fields = selectedModel?.fields || []
    const query = parameterSearch.trim().toLocaleLowerCase()
    if (!query) return fields
    return fields.filter((field) => {
      const searchableText = [
        field.name,
        field.label,
        field.description,
        selectedModel?.id,
        selectedModel?.label,
        'model parameters',
      ].filter(Boolean).join(' ').toLocaleLowerCase()
      return searchableText.includes(query)
    })
  }, [parameterSearch, selectedModel])

  useEffect(() => {
    onSearchMatchCount?.(visibleModelFields.length)
  }, [onSearchMatchCount, visibleModelFields.length])

  const dirty = Boolean(
    strategy
    && !readOnly
    && !strategy.locked
    && selectedModel
    && (selectedModelId !== savedModelFamily || !valuesEqual(formValues, savedValues)),
  )

  useEffect(() => {
    onDirtyChange?.(dirty)
  }, [dirty, onDirtyChange])

  function selectModel(modelId) {
    if (dirty && !window.confirm(tr('Discard unsaved model parameter changes?'))) return
    setSelectedModelId(modelId)
    const model = (payload?.models || []).find((item) => item.id === modelId)
    setFormValues(modelId === savedModelFamily && Object.keys(savedValues).length ? { ...savedValues } : valuesFromModel(model))
    setReason('')
    setError('')
    setNotice('')
  }

  function changeField(field, value) {
    setFormValues((current) => ({ ...current, [field.name]: normalizedValue(field, value) }))
  }

  async function save(event) {
    event.preventDefault()
    if (!strategy || readOnly || strategy.locked || !selectedModel) return
    const normalizedReason = reason.trim()
    if (normalizedReason.length < 3) {
      setError(tr('Enter a reason for this change.'))
      return
    }
    if (!dirty) {
      setNotice(tr('No model changes to save.'))
      return
    }

    setSaving(true)
    setError('')
    setNotice('')
    try {
      const updated = await apiFetch(`${API}/admin/strategies/${encodeURIComponent(strategy.id)}/model`, {
        method: 'PUT',
        body: {
          expected_strategy_revision: strategy.revision,
          model_family: selectedModel.id,
          values: formValues,
          note: normalizedReason,
        },
      })
      setReason('')
      setNotice(tr('{model} and its parameters were saved with this Strategy.', { model: selectedModel.label }))
      await onStrategyModelSaved?.(updated)
    } catch (requestError) {
      handleError(requestError)
    } finally {
      setSaving(false)
    }
  }

  if (loading) {
    const Container = embedded ? 'div' : 'section'
    return <Container className={`${embedded ? 'model-research-embedded' : 'settings-workspace-section'} model-research-settings-section`}><div className="settings-loading"><span className="loading-ring" />{tr('Loading model settings…')}</div></Container>
  }

  if (!payload) {
    const Container = embedded ? 'div' : 'section'
    return (
      <Container className={`${embedded ? 'model-research-embedded' : 'settings-workspace-section'} model-research-settings-section`}>
        <div className="global-inline-message error-inline">{error || tr('Model settings are unavailable.')}</div>
        <button type="button" className="secondary-action" onClick={loadData}>{tr('Retry')}</button>
      </Container>
    )
  }

  const Container = embedded ? 'div' : 'section'
  return (
    <Container className={`${embedded ? 'model-research-embedded' : 'settings-workspace-section'} model-research-settings-section`}>
      <div className="settings-section-heading model-research-local-heading">
        <div>
          <span className="panel-kicker">{tr('MODEL PARAMETERS')}</span>
          <h2>{tr('Model saved with this Strategy')}</h2>
        </div>
        <div className="model-research-heading-meta">
          {savedModel?.settings_hash ? <span className="settings-revision-badge">{tr('Model hash')} {savedModel.settings_hash.slice(0, 10)}…</span> : null}
          {dirty ? <span className="model-research-unsaved">{tr('Unsaved model changes')}</span> : null}
        </div>
      </div>

      {error ? <div className="global-inline-message error-inline">{error}</div> : null}
      {notice ? <div className="global-inline-message success-inline">{notice}</div> : null}

      <div className="model-research-selector-row model-research-selector-single">
        <label>
          <span>{tr('Algorithm')}</span>
          <select value={selectedModelId} onChange={(event) => selectModel(event.target.value)} disabled={readOnly || Boolean(strategy?.locked)}>
            {(payload.models || []).map((model) => <option key={model.id} value={model.id}>{model.label}</option>)}
          </select>
        </label>
        <div className="model-research-bound-state">
          <span>{tr(readOnly ? 'Saved with Strategy' : 'Saved for Backtest')}</span>
          <strong>{savedModel?.label || tr('Not saved yet')}</strong>
        </div>
      </div>

      {selectedModel ? (
        <form onSubmit={save} className="model-research-form">
          <div className="model-research-fields-grid">
            {visibleModelFields.map((field) => (
              <ModelField key={field.name} field={field} value={formValues[field.name]} disabled={readOnly || Boolean(strategy?.locked)} onChange={(value) => changeField(field, value)} />
            ))}
          </div>
          {!readOnly && !strategy?.locked ? (
            <div className="model-research-save-row">
              <label>
                <span>{tr('Change reason')}</span>
                <input value={reason} onChange={(event) => setReason(event.target.value)} maxLength={500} required placeholder={tr('Describe why this Strategy model is changing')} />
              </label>
              <button type="submit" className="admin-primary-button" disabled={saving || !dirty}>{tr(saving ? 'Saving…' : 'Save Strategy model')}</button>
            </div>
          ) : null}
        </form>
      ) : null}

    </Container>
  )
}
