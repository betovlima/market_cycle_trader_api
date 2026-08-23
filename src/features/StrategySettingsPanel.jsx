import { tr } from '../i18n/runtime'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { ApiError, apiFetch } from '../api/http'
import { API } from '../config/env'
import { ModelResearchSettingsPanel } from './ModelResearchSettingsPanel'
import { ACTIVE_JOB_STATUSES, STRATEGY_FIELD_HINTS } from './strategySettings/strategySettingsConfig'
import { dateTime, parseEditorValues, resolveFieldSchema, statusLabel, strategyCatalogRank, titleFromName, toEditorValues } from './strategySettings/strategySettingsUtils'
import { ParameterField, StrategyFieldLabel } from './strategySettings/components/StrategyFields'
import { StrategyBoundaryGrid } from './strategySettings/components/StrategyBoundaryGrid'
import { StrategyCatalog } from './strategySettings/components/StrategyCatalog'

export function StrategySettingsPanel({ onSessionExpired, onTraderWinnerChanged, embedded = false }) {
  const [catalog, setCatalog] = useState(null)
  const [selected, setSelected] = useState(null)
  const [editorValues, setEditorValues] = useState({})
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [changeNote, setChangeNote] = useState('')
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [activeJob, setActiveJob] = useState(null)
  const [modelHasUnsavedChanges, setModelHasUnsavedChanges] = useState(false)
  const [parameterSearch, setParameterSearch] = useState('')
  const [modelParameterMatchCount, setModelParameterMatchCount] = useState(0)
  const initialLoadStartedRef = useRef(false)
  const catalogLoadedRef = useRef(false)

  const baselineEditorValues = useMemo(
    () => toEditorValues(selected?.configuration || {}),
    [selected?.configuration],
  )

  const parameterSchemas = useMemo(() => {
    if (!selected?.configuration) return {}
    return Object.fromEntries(
      Object.keys(selected.configuration).map((field) => [
        field,
        resolveFieldSchema(catalog?.parameter_schema, field),
      ]),
    )
  }, [catalog?.parameter_schema, selected?.configuration])

  const updateEditorValue = useCallback((field, value) => {
    setEditorValues((current) => {
      if (Object.is(current[field], value)) return current
      return { ...current, [field]: value }
    })
  }, [])

  const handleError = useCallback((requestError) => {
    if (requestError instanceof ApiError && requestError.status === 401) {
      onSessionExpired()
      return
    }
    setError(tr(requestError.message || 'Unable to manage strategies.'))
  }, [onSessionExpired])

  const refreshActiveJob = useCallback(async () => {
    try {
      const latest = await apiFetch(`${API}/jobs/latest`)
      setActiveJob(latest && ACTIVE_JOB_STATUSES.has(latest.status) ? latest : null)
    } catch {
      setActiveJob(null)
    }
  }, [])

  const loadStrategy = useCallback(async (strategyId) => {
    const detail = await apiFetch(`${API}/admin/strategies/${encodeURIComponent(strategyId)}`)
    setSelected(detail)
    setName(detail.name || '')
    setDescription(detail.description || '')
    setEditorValues(toEditorValues(detail.configuration || {}))
    setChangeNote('')
    setModelHasUnsavedChanges(false)
    return detail
  }, [])

  const loadCatalog = useCallback(async (preferredStrategyId = '') => {
    const showBlockingLoader = !catalogLoadedRef.current
    if (showBlockingLoader) setLoading(true)
    try {
      const response = await apiFetch(`${API}/admin/strategies`)
      setCatalog(response)
      await refreshActiveJob()
      const targetId = preferredStrategyId
        || response.control?.research_strategy_id
        || response.items?.[0]?.id
      if (targetId) await loadStrategy(targetId)
      catalogLoadedRef.current = true
      setError('')
    } catch (requestError) {
      handleError(requestError)
    } finally {
      if (showBlockingLoader) setLoading(false)
    }
  }, [handleError, loadStrategy, refreshActiveJob])

  useEffect(() => {
    if (initialLoadStartedRef.current) return
    initialLoadStartedRef.current = true
    loadCatalog()
  }, [loadCatalog])

  useEffect(() => {
    if (!activeJob) return undefined
    const timerId = window.setInterval(refreshActiveJob, 5000)
    return () => window.clearInterval(timerId)
  }, [activeJob, refreshActiveJob])

  const hasUnsavedStrategyChanges = useMemo(() => {
    if (!selected || selected.locked || selected.legacy_configuration_incomplete) return false
    const editorKeys = Object.keys(editorValues)
    const baselineKeys = Object.keys(baselineEditorValues)
    if (editorKeys.length !== baselineKeys.length) return true
    return editorKeys.some((field) => !Object.is(editorValues[field], baselineEditorValues[field]))
  }, [baselineEditorValues, editorValues, selected])

  const hasUnsavedChanges = hasUnsavedStrategyChanges || modelHasUnsavedChanges

  useEffect(() => {
    if (!hasUnsavedChanges) return undefined
    const protectDraft = (event) => {
      event.preventDefault()
      event.returnValue = ''
    }
    window.addEventListener('beforeunload', protectDraft)
    return () => window.removeEventListener('beforeunload', protectDraft)
  }, [hasUnsavedChanges])

  function confirmDiscardDraft() {
    if (!hasUnsavedChanges) return true
    return window.confirm(tr('Discard the unsaved strategy changes?'))
  }

  async function selectDetail(strategyId) {
    if (selected?.id !== strategyId && !confirmDiscardDraft()) return
    setBusy(`read:${strategyId}`)
    setError('')
    try {
      await loadStrategy(strategyId)
    } catch (requestError) {
      handleError(requestError)
    } finally {
      setBusy('')
    }
  }

  async function cloneStrategy(source) {
    if (!confirmDiscardDraft()) return
    setBusy(`clone:${source.id}`)
    setError('')
    setNotice('')
    try {
      const created = await apiFetch(`${API}/admin/strategies`, {
        method: 'POST',
        body: { clone_from_strategy_id: source.id },
      })
      setNotice(tr('Strategy created with the next sequential Catalog identity.'))
      await loadCatalog(created.id)
    } catch (requestError) {
      handleError(requestError)
    } finally {
      setBusy('')
    }
  }

  async function saveStrategy(event) {
    event.preventDefault()
    if (!selected || selected.locked) return
    const note = changeNote.trim() || null
    let configuration
    let assetsInput
    try {
      const parsed = parseEditorValues(editorValues, selected.configuration || {})
      configuration = parsed.configuration
      assetsInput = parsed.assetsInput
    } catch (parseError) {
      setError(parseError.message)
      return
    }
    setBusy('save')
    setError('')
    setNotice('')
    try {
      const updated = await apiFetch(`${API}/admin/strategies/${encodeURIComponent(selected.id)}`, {
        method: 'PUT',
        body: {
          expected_revision: selected.revision,
          configuration,
          assets_input: assetsInput,
          name: selected.name || '',
          description: selected.description || '',
          note,
        },
      })
      setSelected(updated)
      setEditorValues(toEditorValues(updated.configuration || {}))
      setChangeNote('')
      setNotice(tr('Strategy saved as revision {revision}. Run a new backtest before promotion.', { revision: updated.revision }))
      const response = await apiFetch(`${API}/admin/strategies`)
      setCatalog(response)
    } catch (requestError) {
      if (requestError instanceof ApiError && requestError.status === 409) {
        await loadCatalog(selected.id)
      }
      handleError(requestError)
    } finally {
      setBusy('')
    }
  }

  async function handleStrategyModelSaved(updated) {
    setSelected(updated)
    setName(updated.name || '')
    setDescription(updated.description || '')
    setEditorValues(toEditorValues(updated.configuration || {}))
    setChangeNote('')
    setModelHasUnsavedChanges(false)
    await loadCatalog(updated.id)
  }

  async function useForStrategyResearch(strategy) {
    if (strategy.id === selected?.id && hasUnsavedChanges) {
      setError(tr('Save or discard the current strategy changes before selecting it for Strategy Research.'))
      return
    }
    if (activeJob) {
      setError(tr('Wait for the active backtest to finish before changing the selected Strategy Research baseline.'))
      return
    }
    const note = window.prompt(tr('Reason for selecting this Strategy for Strategy Research:'), `Research ${strategy.name}`)?.trim()
    if (!note) return
    setBusy(`select:${strategy.id}`)
    setError('')
    setNotice('')
    try {
      await apiFetch(`${API}/admin/strategies/${encodeURIComponent(strategy.id)}/select-for-strategy-research`, {
        method: 'POST',
        body: {
          expected_control_revision: catalog.control.revision,
          note,
        },
      })
      setNotice(tr('Strategy Research will use this Strategy for Simulation Backtest, Model Tuning and Temporal Intelligence. Trader continues using the protected Winner.'))
      await loadCatalog(strategy.id)
    } catch (requestError) {
      handleError(requestError)
    } finally {
      setBusy('')
    }
  }



  async function promoteToTrader(strategy) {
    if (strategy.id === selected?.id && hasUnsavedChanges) {
      setError(tr('Save or discard the current strategy changes before promotion.'))
      return
    }
    if (activeJob) {
      setError(tr('Wait for the active backtest to finish before promoting another Winner.'))
      return
    }
    if (catalog.control?.live_market_refresh_in_progress) {
      setError(tr('Winner promotion is temporarily unavailable while temporal market-series synchronization is running.'))
      return
    }
    const promotingResearch = strategy.id === catalog.control?.research_strategy_id
    const confirmation = window.confirm(
      tr('Promote {name} to WINNER?', { name: `"${strategy.name}"` }) + '\n\n'
      + tr(promotingResearch
        ? 'The Strategy keeps the same sequential name and technical identity. Because it is RESEARCH, the current Winner becomes the new RESEARCH Strategy. Current position, cash and scheduled pipeline are preserved.'
        : 'The Strategy keeps the same sequential name and technical identity. The current Winner becomes a saved historical Strategy. Current position, cash and scheduled pipeline are preserved.'),
    )
    if (!confirmation) return
    const note = window.prompt(tr('Promotion reason:'), tr('Promote {name} to WINNER', { name: strategy.name }))?.trim()
    if (!note) return
    setBusy(`promote:${strategy.id}`)
    setError('')
    setNotice('')
    try {
      const result = await apiFetch(`${API}/admin/strategies/${encodeURIComponent(strategy.id)}/promote-to-trader`, {
        method: 'POST',
        body: {
          confirm_promote_to_trader: true,
          confirm_temporal_series_idle: true,
          confirm_preserve_operational_state: true,
          expected_control_revision: catalog.control.revision,
          expected_strategy_revision: strategy.revision,
          note,
        },
      })
      setNotice(tr('{name} is now WINNER. Its Strategy identity and sequential name were preserved.', { name: result.winner.name }))
      await loadCatalog(strategy.id)
      onTraderWinnerChanged?.()
    } catch (requestError) {
      handleError(requestError)
    } finally {
      setBusy('')
    }
  }

  async function deleteStrategy(strategy) {
    if (strategy.id === selected?.id && hasUnsavedChanges && !confirmDiscardDraft()) return
    if (!window.confirm(tr('Delete saved Strategy "{name}"?', { name: strategy.name }))) return
    setBusy(`delete:${strategy.id}`)
    setError('')
    setNotice('')
    try {
      await apiFetch(`${API}/admin/strategies/${encodeURIComponent(strategy.id)}`, {
        method: 'DELETE',
        body: { confirm_delete: true, note: `Delete saved Strategy ${strategy.id}` },
      })
      setNotice(tr('Saved Strategy deleted.'))
      await loadCatalog()
    } catch (requestError) {
      handleError(requestError)
    } finally {
      setBusy('')
    }
  }

  const groupedParameters = useMemo(() => {
    if (!selected?.configuration) return []
    const order = catalog?.parameter_order || Object.keys(selected.configuration)
    const used = new Set()
    const groups = (catalog?.parameter_groups || []).map((group) => {
      const fields = (group.fields || []).filter((field) => selected.configuration[field] !== undefined)
      fields.forEach((field) => used.add(field))
      return { id: group.id, label: group.label, fields }
    }).filter((group) => group.fields.length)
    const other = order.filter((field) => !used.has(field) && selected.configuration[field] !== undefined)
    if (other.length) groups.push({ id: 'other', label: 'Other parameters', fields: other })

    const query = parameterSearch.trim().toLocaleLowerCase()
    if (!query) return groups

    return groups.map((group) => ({
      ...group,
      fields: group.fields.filter((field) => {
        const schema = parameterSchemas[field]
        const searchableText = [
          field,
          titleFromName(field),
          group.label,
          schema?.title,
          schema?.description,
        ].filter(Boolean).join(' ').toLocaleLowerCase()
        return searchableText.includes(query)
      }),
    })).filter((group) => group.fields.length)
  }, [catalog?.parameter_groups, catalog?.parameter_order, parameterSchemas, parameterSearch, selected])

  const visibleParameterCount = useMemo(
    () => groupedParameters.reduce((total, group) => total + group.fields.length, 0),
    [groupedParameters],
  )
  const globalVisibleParameterCount = visibleParameterCount + modelParameterMatchCount

  if (loading) {
    return <section className={`${embedded ? 'settings-workspace-section settings-strategy-section' : 'panel'} strategy-lab-panel`}><div className="settings-loading"><span className="loading-ring" />{tr("Loading strategies…")}</div></section>
  }

  if (!catalog || !selected) {
    return <section className={`${embedded ? 'settings-workspace-section settings-strategy-section' : 'panel'} strategy-lab-panel`}><div className="global-inline-message error-inline">{tr(error || 'Strategy catalog is unavailable.')}</div></section>
  }

  const researchId = catalog.control?.research_strategy_id
  const winnerId = catalog.control?.trader_winner_strategy_id
  const latestSavedId = catalog.control?.latest_saved_strategy_id
  const legacyConfigurationIncomplete = Boolean(selected.legacy_configuration_incomplete)
  const hasActiveBacktest = Boolean(activeJob)
  const temporalSeriesUpdateInProgress = Boolean(catalog.control?.live_market_refresh_in_progress)
  const isTemporalStrategy = selected.strategy_kind === 'temporal_intelligence'
  const isStatefulTemporalStrategy = isTemporalStrategy && selected.temporal_strategy_variant === 'winner_transition_stateful'
  const statefulValidation = isStatefulTemporalStrategy ? (selected.temporal_policy?.stateful_validation || {}) : {}
  const statefulCandidateMetrics = statefulValidation?.candidate_metrics || {}
  const statefulControlParityPassed = String(statefulValidation?.control_parity?.status || '').toLowerCase() === 'passed'
  const traderRuntimeReady = Boolean(selected.trader_compatibility?.eligible)
  const traderRuntimeBlockReason = tr(selected.trader_compatibility?.reason || 'This Strategy is not compatible with the installed Trader runtime.')
  const canPromote = traderRuntimeReady
    && !legacyConfigurationIncomplete
    && selected.id !== winnerId
    && !hasActiveBacktest
    && !temporalSeriesUpdateInProgress
    && (!isStatefulTemporalStrategy || statefulControlParityPassed)
  const orderedStrategies = [...catalog.items].sort((left, right) => {
    const rankDifference = strategyCatalogRank(left, winnerId, latestSavedId, researchId)
      - strategyCatalogRank(right, winnerId, latestSavedId, researchId)
    if (rankDifference !== 0) return rankDifference
    return Number(right.strategy_sequence || 0) - Number(left.strategy_sequence || 0)
  })


  return (
    <section className={`${embedded ? 'settings-workspace-section settings-strategy-section' : 'panel'} strategy-lab-panel`}>
      <div className="panel-heading strategy-lab-heading">
        <div>
          <span className="panel-kicker">{tr("STRATEGIES")}</span>
          <h2>{tr("Research strategies and Trader winner")}</h2>
        </div>
        <div className="strategy-heading-state">
          {hasUnsavedChanges ? <span className="strategy-unsaved-badge">{tr("Unsaved changes")}</span> : null}
          <span className="strategy-control-revision">{tr("Selection revision")}{' '}{catalog.control.revision}</span>
        </div>
      </div>

      {error ? <div className="global-inline-message error-inline">{error}</div> : null}
      {notice ? <div className="global-inline-message success-inline">{notice}</div> : null}
      {hasActiveBacktest ? (
        <div className="global-inline-message warning-inline">
          {tr("Backtest")}{' '}{activeJob.id} {tr("is")}{' '}{statusLabel(activeJob.status)}{tr(". You may clone and edit test strategies, but strategy selection, promotion and a new backtest remain locked until it finishes. The strategy used by the running backtest cannot be deleted until it finishes.")}</div>
      ) : null}

      <StrategyBoundaryGrid catalog={catalog} />

      {catalog.control?.paper_state_reinitialization_required ? (
        <div className="global-inline-message warning-inline">{tr("The Trader winner changed. Run the protected Paper initialization before restarting Trader.")}</div>
      ) : null}

      <div className="strategy-workspace">
        <StrategyCatalog
          catalog={catalog}
          orderedStrategies={orderedStrategies}
          selected={selected}
          busy={busy}
          researchId={researchId}
          winnerId={winnerId}
          latestSavedId={latestSavedId}
          onCloneWinner={cloneStrategy}
          onSelectDetail={selectDetail}
        />

        <div className="strategy-editor-panel">
          <div className="strategy-editor-header">
            <div>
              <span className="panel-kicker">{tr("SELECTED STRATEGY")}</span>
              <h3>{selected.name}</h3>
              {selected.description ? <small className="strategy-editor-commit">{selected.description}</small> : null}
              <p>{tr("Revision")}{' '}{selected.revision} {tr("· Hash")}{' '}{selected.configuration_hash?.slice(0, 12) || '—'}{tr("… · Source")}{' '}{selected.origin?.winner_source_file || tr('catalog snapshot')}</p>
            </div>
            <div className="strategy-editor-actions">
              <button type="button" onClick={() => cloneStrategy(selected)} disabled={Boolean(busy)}>{tr("Clone Strategy")}</button>
              {selected.id === winnerId ? null : selected.id !== researchId ? <button type="button" onClick={() => useForStrategyResearch(selected)} disabled={Boolean(busy) || hasActiveBacktest}>{tr("Mark as RESEARCH")}</button> : <button type="button" disabled>{tr("RESEARCH")}</button>}
              {selected.id !== winnerId ? <button type="button" className="promote-action" title={temporalSeriesUpdateInProgress ? tr('Winner promotion is temporarily unavailable while temporal market-series synchronization is running.') : (!traderRuntimeReady ? traderRuntimeBlockReason : '')} onClick={() => promoteToTrader(selected)} disabled={Boolean(busy) || !canPromote}>{tr("Promote to WINNER")}</button> : <button type="button" className="promote-action" disabled>{tr("WINNER")}</button>}
              {selected.id !== winnerId && selected.id !== researchId ? <button type="button" className="danger" onClick={() => deleteStrategy(selected)} disabled={Boolean(busy)}>{tr("Delete strategy")}</button> : null}
            </div>
          </div>

          {isTemporalStrategy ? (
            <div className="strategy-temporal-policy-summary">
              <span><small>{tr('Strategy type')}</small><strong>{tr(isStatefulTemporalStrategy ? 'Conservative Decision Policy' : 'Temporal Intelligence')}</strong></span>
              <span><small>{tr('Source run')}</small><strong>{selected.source_temporal_run_id || '—'}</strong></span>
              {isStatefulTemporalStrategy ? <span><small>{tr('Control parity')}</small><strong>{statefulControlParityPassed ? tr('PASSED') : '—'}</strong></span> : <span><small>{tr('Temporal experiment')}</small><strong>{selected.source_temporal_experiment || '—'}</strong></span>}
              <span><small>{tr('Base weak threshold')}</small><strong>{selected.temporal_policy?.parameters?.timing_base_weak_threshold ?? '—'}</strong></span>
              <span><small>{tr('Challenger minimum')}</small><strong>{selected.temporal_policy?.parameters?.timing_challenger_minimum ?? '—'}</strong></span>
              <span><small>{tr('Minimum advantage')}</small><strong>{selected.temporal_policy?.parameters?.timing_minimum_advantage ?? '—'}</strong></span>
              <span><small>{tr('Validated capital')}</small><strong>{(isStatefulTemporalStrategy ? statefulCandidateMetrics?.ending_capital : selected.temporal_policy?.validation?.ending_capital) != null ? `$${Number(isStatefulTemporalStrategy ? statefulCandidateMetrics.ending_capital : selected.temporal_policy.validation.ending_capital).toLocaleString(undefined, { maximumFractionDigits: 2 })}` : '—'}</strong></span>
              {isStatefulTemporalStrategy ? <span><small>{tr('Delta vs control')}</small><strong>{selected.temporal_policy?.stateful_validation?.candidate_delta_vs_control_rate != null ? `${(Number(selected.temporal_policy.stateful_validation.candidate_delta_vs_control_rate) * 100).toFixed(2)}%` : '—'}</strong></span> : null}
            </div>
          ) : null}

          {!isTemporalStrategy ? (<>
          <div className="strategy-parameter-tools strategy-parameter-tools-global">
            <label className="strategy-parameter-search">
              <StrategyFieldLabel id="hint-parameter-search" label={tr("Find a parameter")} hint={STRATEGY_FIELD_HINTS.search} />
              <div className="strategy-parameter-search-control">
                <input
                  type="search"
                  value={parameterSearch}
                  placeholder={tr("Search Strategy and selected model parameters by label or technical name")}
                  onChange={(event) => setParameterSearch(event.target.value)}
                  autoComplete="off"
                />
                {parameterSearch ? <button type="button" onClick={() => setParameterSearch('')}>{tr("Clear")}</button> : null}
              </div>
            </label>
            <small>{parameterSearch ? tr(globalVisibleParameterCount === 1 ? '{count} matching parameter' : '{count} matching parameters', { count: globalVisibleParameterCount }) : tr(globalVisibleParameterCount === 1 ? '{count} parameter available' : '{count} parameters available', { count: globalVisibleParameterCount })}</small>
          </div>
          </>) : null}


          {!legacyConfigurationIncomplete ? (
            <ModelResearchSettingsPanel
              onSessionExpired={onSessionExpired}
              embedded
              strategy={selected}
              readOnly={isTemporalStrategy}
              onStrategyModelSaved={isTemporalStrategy ? null : handleStrategyModelSaved}
              onDirtyChange={isTemporalStrategy ? null : setModelHasUnsavedChanges}
              parameterSearch={isTemporalStrategy ? '' : parameterSearch}
              onSearchMatchCount={isTemporalStrategy ? null : setModelParameterMatchCount}
            />
          ) : null}


          {!isTemporalStrategy ? (
          <form className="strategy-parameter-form" onSubmit={saveStrategy}>
            <div className="strategy-metadata-grid">
              <label>
                <StrategyFieldLabel id="hint-strategy-name" label={tr("Strategy identity")} hint={STRATEGY_FIELD_HINTS.name} />
                <input value={name} readOnly disabled />
              </label>
              <label>
                <StrategyFieldLabel id="hint-strategy-description" label={tr("Git commit description")} hint={STRATEGY_FIELD_HINTS.description} align="right" />
                <input value={description} readOnly disabled />
              </label>
            </div>

            <div className="strategy-parameter-groups">
              {groupedParameters.map((group, index) => (
                <details key={`${group.id}:${parameterSearch ? 'filtered' : 'all'}`} open={parameterSearch ? true : index === 0 || group.id === 'model'}>
                  <summary>{tr(group.label)}<span>{group.fields.length} {tr("parameters")}</span></summary>
                  <div className="strategy-parameter-grid">
                    {group.fields.map((field, fieldIndex) => (
                      <ParameterField
                        key={field}
                        name={field}
                        value={editorValues[field]}
                        reference={selected.configuration[field]}
                        schema={parameterSchemas[field]}
                        hintAlign={fieldIndex % 2 === 1 ? 'right' : 'left'}
                        disabled={selected.locked || legacyConfigurationIncomplete}
                        onChange={updateEditorValue}
                      />
                    ))}
                  </div>
                </details>
              ))}
              {parameterSearch && globalVisibleParameterCount === 0 ? (
                <div className="strategy-parameter-empty">{tr("No parameter matches “")}{parameterSearch}{tr("”. Search Strategy and selected model parameters by label, technical name or description.")}</div>
              ) : null}
            </div>

            {!selected.locked && !legacyConfigurationIncomplete ? (
              <div className="strategy-save-row">
                <label>
                  <StrategyFieldLabel id="hint-strategy-change-reason" label={tr("Change reason (optional)")} hint={STRATEGY_FIELD_HINTS.changeReason} />
                  <input value={changeNote} onChange={(event) => setChangeNote(event.target.value)} maxLength={500} placeholder={tr('Optional audit note')} />
                </label>
                <div className="strategy-save-actions">
                  <small>{tr(hasUnsavedStrategyChanges ? 'Local Strategy changes are preserved until you save or leave this Strategy.' : 'No unsaved Strategy parameter changes.')}</small>
                  <button type="submit" className="admin-primary-button" disabled={Boolean(busy) || !hasUnsavedStrategyChanges}>{tr(busy === 'save' ? 'Saving…' : 'Save test strategy')}</button>
                </div>
              </div>
            ) : null}
          </form>
          ) : null}

          <div className="strategy-last-test">
            <span>{tr(isTemporalStrategy ? 'Source validation' : 'Latest backtest')}</span>
            <strong>{isTemporalStrategy ? tr('Temporal Intelligence completed') : selected.last_backtest_status ? statusLabel(selected.last_backtest_status) : tr('Not run for this revision')}</strong>
            <small>{isTemporalStrategy ? selected.source_temporal_run_id || '—' : selected.last_backtest_id || '—'} {tr("· Updated")}{' '}{dateTime(selected.updated_at)}</small>
          </div>
        </div>
      </div>
    </section>
  )
}
