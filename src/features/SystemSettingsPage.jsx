import { getIntlLocale, tr, translatedStatus } from '../i18n/runtime'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { ApiError, apiFetch } from '../api/http'
import { API } from '../config/env'
import { SettingsIcon } from '../shared/components/Icons'
import { ParameterHint } from '../shared/components/ParameterHint'
import { StrategySettingsPanel } from './StrategySettingsPanel'

const DEFAULT_FORM = {
  enabled: true,
  automatic_training_enabled: true,
  max_concurrent_jobs: 1,
  timeout_minutes: 360,
  reason: '',
}


const PARAMETER_HINTS = {
  trainingEnabled: {
    description: 'Controls whether new model-training and backtest jobs may start. Turning it off does not forcibly terminate a job that is already running.',
    relationship: 'When disabled, the system will not start new training or backtest jobs.',
    example: 'With training disabled and 3 requested jobs, jobs_started = 0.',
  },
  automaticTraining: {
    description: 'Allows the scheduler to start the authorized pre-market training cycle automatically when an eligible market session is approaching.',
    relationship: 'Runs only for eligible pre-market sessions and only while training is enabled.',
    example: 'With 5 eligible market sessions in a week, at most 5 scheduled pre-market runs can be started.',
  },
  timeoutMinutes: {
    description: 'Maximum wall-clock duration allowed for one backtest before the API stops it.',
    relationship: 'The UI stores minutes; the API persists the equivalent duration in seconds.',
    example: '360 minutes × 60 = 21,600 seconds = 6 hours.',
  },
  changeReason: {
    description: 'Required audit note explaining why the Administrator changed the configuration. It is stored with the resulting revision.',
    relationship: 'Saving creates a new audited system-settings revision.',
    example: 'Saving revision 12 with a valid reason creates revision 13 and records the Administrator and timestamp.',
  },
}



const RUNTIME_HINTS = {
  detectedCpu: {
    description: 'Number of CPU cores detected by the API runtime for capacity awareness.',
    relationship: 'This is an observed runtime value and is not directly editable from this screen.',
  },
  traderMode: {
    description: 'Current operational mode of the Trader service.',
    relationship: 'Use Trader operation below to change the mode. This status is shown here only as a compact runtime summary.',
  },
  queue: {
    description: 'Backtests use a single execution queue so only one simulation job runs at a time.',
    relationship: 'Strategy drafts remain editable while another backtest is running.',
  },
  strategySeparation: {
    description: 'Research strategy parameters stay in the dedicated strategy workspace instead of being mixed with runtime controls.',
    relationship: 'This keeps operational settings and research configuration separated.',
  },
}


const TRADER_FIELD_HINTS = {
  status: {
    description: 'Current operational state reported by the Trader service.',
    relationship: 'This is runtime state, not a strategy parameter.',
  },
  phase: {
    description: 'Current stage of the Trader lifecycle, such as waiting, evaluating, executing or stopped.',
    relationship: 'Phase changes as the scheduled trading workflow advances.',
  },
  scheduler: {
    description: 'Shows whether the process that coordinates scheduled Trader sessions is alive.',
    relationship: 'Online means the scheduler loop is responding; it does not mean an order is being sent.',
  },
  nextSession: {
    description: 'Next market session currently scheduled for the Trader workflow.',
    relationship: 'The date is derived by the backend from the market calendar and Trader state.',
  },
  traderWinner: {
    description: 'Protected strategy revision currently assigned to the live Paper Trader.',
    relationship: 'Research edits do not change this winner until an explicit candidate promotion is completed.',
  },
}

function dateTime(value) {
  if (!value) return '—'
  return new Date(value).toLocaleString(getIntlLocale())
}

function modeLabel(value) {
  return translatedStatus(value || 'stopped')
}

function historySummary(item) {
  const training = item.training || {}
  return [
    tr(training.enabled ? 'Training on' : 'Training off'),
    tr(training.automatic_training_enabled ? 'Automatic on' : 'Automatic off'),
    tr('Strategy catalog separated from runtime controls'),
    tr('Single backtest queue'),
  ].join(' · ')
}

export function SystemSettingsPage({ onSessionExpired }) {
  const [settings, setSettings] = useState(null)
  const [history, setHistory] = useState([])
  const [traderControl, setTraderControl] = useState(null)
  const [traderHistory, setTraderHistory] = useState([])
  const [form, setForm] = useState(DEFAULT_FORM)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [traderBusy, setTraderBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const initialLoadStartedRef = useRef(false)

  const handleError = useCallback((requestError) => {
    if (requestError instanceof ApiError && requestError.status === 401) {
      onSessionExpired()
      return
    }
    setError(tr(requestError.message || 'Unable to update system settings.'))
  }, [onSessionExpired])

  const loadData = useCallback(async () => {
    setLoading(true)
    try {
      const [settingsResponse, historyResponse, traderResponse, traderHistoryResponse] = await Promise.all([
        apiFetch(`${API}/admin/system-settings`),
        apiFetch(`${API}/admin/system-settings/history?limit=20`),
        apiFetch(`${API}/admin/trader-control/status`),
        apiFetch(`${API}/admin/trader-control/history?limit=20`),
      ])
      const training = settingsResponse.training || {}
      setSettings(settingsResponse)
      setHistory(historyResponse.items || [])
      setTraderControl(traderResponse)
      setTraderHistory(traderHistoryResponse.items || [])
      setForm({
        enabled: Boolean(training.enabled),
        automatic_training_enabled: Boolean(training.automatic_training_enabled),
        max_concurrent_jobs: Number(training.max_concurrent_jobs || 1),
        timeout_minutes: Math.max(5, Math.round(Number(training.timeout_seconds || 21600) / 60)),
        reason: '',
      })
      setError('')
    } catch (requestError) {
      handleError(requestError)
    } finally {
      setLoading(false)
    }
  }, [handleError])

  useEffect(() => {
    if (initialLoadStartedRef.current) return
    initialLoadStartedRef.current = true
    loadData()
  }, [loadData])

  const refreshTraderControl = useCallback(async () => {
    try {
      const [traderResponse, traderHistoryResponse] = await Promise.all([
        apiFetch(`${API}/admin/trader-control/status`),
        apiFetch(`${API}/admin/trader-control/history?limit=20`),
      ])
      setTraderControl(traderResponse)
      setTraderHistory(traderHistoryResponse.items || [])
    } catch (requestError) {
      handleError(requestError)
    }
  }, [handleError])

  async function saveSettings(event) {
    event.preventDefault()
    const reason = form.reason.trim()
    if (reason.length < 3) {
      setError(tr('Enter a reason for this change.'))
      return
    }
    setSaving(true)
    setError('')
    setNotice('')
    try {
      const response = await apiFetch(`${API}/admin/system-settings`, {
        method: 'PATCH',
        body: {
          expected_revision: settings.revision,
          reason,
          training: {
            enabled: Boolean(form.enabled),
            automatic_training_enabled: Boolean(form.automatic_training_enabled),
            max_concurrent_jobs: Number(form.max_concurrent_jobs),
            timeout_seconds: Number(form.timeout_minutes) * 60,
          },
        },
      })
      setSettings(response)
      setForm((current) => ({ ...current, reason: '' }))
      setNotice(tr('System settings saved.'))
      const historyResponse = await apiFetch(`${API}/admin/system-settings/history?limit=20`)
      setHistory(historyResponse.items || [])
    } catch (requestError) {
      if (requestError instanceof ApiError && requestError.status === 409) {
        await loadData()
      }
      handleError(requestError)
    } finally {
      setSaving(false)
    }
  }

  async function changeTraderMode(mode) {
    const labels = {
      active: 'Start Trader',
      paused: 'Pause Trader',
      exit_only: 'Enable exit-only mode',
      stopped: 'Stop Trader',
    }
    const destructive = mode === 'stopped'
    if (destructive && !window.confirm(tr('Stop the Trader and cancel a pending non-executing run?'))) return
    setTraderBusy(true)
    setError('')
    setNotice('')
    try {
      const response = await apiFetch(`${API}/admin/trader-control/mode`, {
        method: 'POST',
        body: {
          mode,
          cancel_pending_run: destructive,
          reason: labels[mode],
        },
      })
      setTraderControl(response)
      setNotice(tr('Trader mode changed to {mode}.', { mode: modeLabel(mode) }))
      const historyResponse = await apiFetch(`${API}/admin/trader-control/history?limit=20`)
      setTraderHistory(historyResponse.items || [])
    } catch (requestError) {
      handleError(requestError)
    } finally {
      setTraderBusy(false)
    }
  }

  const timeoutHours = useMemo(() => Number(form.timeout_minutes || 0) / 60, [form.timeout_minutes])

  if (loading) {
    return <div className="settings-loading"><span className="loading-ring" />{tr("Loading system settings…")}</div>
  }

  if (!settings) {
    return <div className="page-stack system-settings-page"><div className="global-inline-message error-inline">{tr(error || 'System settings are unavailable.')}</div><button type="button" className="secondary-action" onClick={loadData}>{tr("Retry")}</button></div>
  }

  return (
    <div className="page-stack system-settings-page">
      <section className="panel settings-workspace-panel">
        <div className="settings-workspace-header">
          <div className="settings-workspace-title">
            <div className="page-title-icon"><SettingsIcon size={22} /></div>
            <div>
              <h2>{tr("System Settings")}</h2>
            </div>
          </div>
          <div className="settings-workspace-actions">
            <span className="settings-revision-badge">{tr("Revision")}{' '}{settings?.revision || 1}</span>
            <button type="button" className="secondary-action settings-refresh-button" onClick={loadData} disabled={loading}>{tr("Refresh")}</button>
          </div>
        </div>

        {error ? <div className="global-inline-message error-inline settings-workspace-message">{error}</div> : null}
        {notice ? <div className="global-inline-message success-inline settings-workspace-message">{notice}</div> : null}

        <div className="settings-runtime-overview" aria-label={tr("Runtime summary")}>
          <RuntimeFact label={tr("CPU")} value={settings?.runtime?.detected_cpu_count || '—'} hint={RUNTIME_HINTS.detectedCpu} hintId="hint-runtime-cpu" />
          <RuntimeFact label={tr("Trader")} value={modeLabel(traderControl?.control_mode)} tone={traderControl?.control_mode === 'active' ? 'positive' : 'warning'} hint={RUNTIME_HINTS.traderMode} hintId="hint-runtime-trader" />
          <RuntimeFact label={tr("Backtest limit")} value={timeoutHours >= 1 ? `${timeoutHours.toFixed(timeoutHours % 1 ? 1 : 0)} h` : `${form.timeout_minutes} min`} hint={PARAMETER_HINTS.timeoutMinutes} hintId="hint-runtime-timeout" />
          <RuntimeFact label={tr("Queue")} value={tr("Single")} hint={RUNTIME_HINTS.queue} hintId="hint-runtime-queue" />
          <RuntimeFact label={tr("Strategy parameters")} value={tr("Separated")} hint={RUNTIME_HINTS.strategySeparation} hintId="hint-runtime-strategy-separation" />
        </div>

        <section className="settings-workspace-section settings-training-section">
          <div className="settings-section-heading settings-compact-heading">
            <div>
              <span className="panel-kicker">{tr("TRAINING")}</span>
              <h2>{tr("Model execution")}</h2>
            </div>
          </div>

          <form className="settings-form settings-compact-form" onSubmit={saveSettings}>
            <div className="settings-model-execution-grid">
              <ToggleField
                id="training-enabled"
                label={tr("Training enabled")}
                hint={PARAMETER_HINTS.trainingEnabled}
                checked={form.enabled}
                onChange={(checked) => setForm({ ...form, enabled: checked })}
              />
              <ToggleField
                id="automatic-training-enabled"
                label={tr("Automatic pre-market training")}
                hint={PARAMETER_HINTS.automaticTraining}
                checked={form.automatic_training_enabled}
                disabled={!form.enabled}
                onChange={(checked) => setForm({ ...form, automatic_training_enabled: checked })}
              />
              <NumberField
                id="backtest-timeout-minutes"
                label={tr("Timeout (minutes)")}
                hint={PARAMETER_HINTS.timeoutMinutes}
                value={form.timeout_minutes}
                min="5"
                max="1440"
                step="5"
                onChange={(value) => setForm({ ...form, timeout_minutes: value })}
              />
              <div className="settings-reason-field settings-inline-reason">
                <FieldLabel label={tr("Change reason")} hint={PARAMETER_HINTS.changeReason} hintId="hint-change-reason" align="right" />
                <input id="settings-change-reason" value={form.reason} onChange={(event) => setForm({ ...form, reason: event.target.value })} maxLength={500} required />
              </div>
              <button type="submit" className="admin-primary-button settings-inline-save" disabled={saving}>{tr(saving ? 'Saving…' : 'Save settings')}</button>
            </div>
          </form>
        </section>

        <StrategySettingsPanel
          embedded
          onSessionExpired={onSessionExpired}
          onTraderWinnerChanged={refreshTraderControl}
        />

        <section className="settings-workspace-section trader-control-panel settings-trader-section">
          <div className="settings-section-heading">
            <div>
              <span className="panel-kicker">{tr("TRADER")}</span>
              <h2>{tr("Trader operation")}</h2>
            </div>
            <span className={`trader-mode-badge mode-${traderControl?.control_mode || 'stopped'}`}>{modeLabel(traderControl?.control_mode)}</span>
          </div>

          <div className="trader-control-grid settings-trader-grid">
            <div className="trader-control-status">
              <div><FieldLabel label={tr("Status")} hint={TRADER_FIELD_HINTS.status} hintId="hint-trader-status" /><strong>{tr(traderControl?.status || 'Unknown')}</strong></div>
              <div><FieldLabel label={tr("Phase")} hint={TRADER_FIELD_HINTS.phase} hintId="hint-trader-phase" /><strong>{modeLabel(traderControl?.phase || '—')}</strong></div>
              <div><FieldLabel label={tr("Scheduler")} hint={TRADER_FIELD_HINTS.scheduler} hintId="hint-trader-scheduler" /><strong>{tr(traderControl?.scheduler_alive ? 'Online' : 'Offline')}</strong></div>
              <div><FieldLabel label={tr("Next session")} hint={TRADER_FIELD_HINTS.nextSession} hintId="hint-trader-next-session" /><strong>{traderControl?.next_execution_session || '—'}</strong></div>
              <div><FieldLabel label={tr("Trader winner")} hint={TRADER_FIELD_HINTS.traderWinner} hintId="hint-trader-winner" align="right" /><strong>{traderControl?.trader_winner?.name || '—'}</strong></div>
            </div>
            <div className="trader-control-actions">
              <button type="button" title={tr("Allow the Trader to execute its normal scheduled Paper workflow.")} onClick={() => changeTraderMode('active')} disabled={traderBusy || traderControl?.control_mode === 'active'}>{tr("Start")}</button>
              <button type="button" title={tr("Pause new Trader actions while preserving the current operational state.")} onClick={() => changeTraderMode('paused')} disabled={traderBusy || traderControl?.control_mode === 'paused'}>{tr("Pause")}</button>
              <button type="button" title={tr("Allow exits from existing positions but do not open new positions.")} onClick={() => changeTraderMode('exit_only')} disabled={traderBusy || traderControl?.control_mode === 'exit_only'}>{tr("Exit only")}</button>
              <button type="button" className="danger" title={tr("Stop Trader operation and cancel a pending non-executing run after confirmation.")} onClick={() => changeTraderMode('stopped')} disabled={traderBusy || traderControl?.control_mode === 'stopped'}>{tr("Stop")}</button>
            </div>
          </div>

          {traderHistory.length ? (
            <div className="trader-control-history">
              <span>{tr("Recent changes")}</span>
              {traderHistory.slice(0, 5).map((item, index) => (
                <div key={`${item.created_at || 'event'}-${index}`}>
                  <strong>{modeLabel(item.new_mode)}</strong>
                  <small>{dateTime(item.created_at)}</small>
                </div>
              ))}
            </div>
          ) : null}
        </section>

        <section className="settings-workspace-section settings-history-section">
          <div className="settings-section-heading">
            <div>
              <span className="panel-kicker">{tr("HISTORY")}</span>
              <h2>{tr("Configuration history")}</h2>
            </div>
            <span className="settings-section-meta">{tr(history.length === 1 ? '{count} revision loaded' : '{count} revisions loaded', { count: history.length })}</span>
          </div>
          <div className="settings-history-list">
            {history.length ? history.map((item) => (
              <article key={`${item.revision}-${item.updated_at}`} className="settings-history-item">
                <div>
                  <strong>{tr("Revision")}{' '}{item.revision}</strong>
                  <span>{historySummary(item)}</span>
                </div>
                <div>
                  <strong>{item.reason}</strong>
                  <span>{tr(item.updated_by || 'Administrator')} · {dateTime(item.updated_at)}</span>
                </div>
              </article>
            )) : <div className="settings-empty-history">{tr("No settings changes recorded.")}</div>}
          </div>
        </section>
      </section>
    </div>
  )
}

function ToggleField({ id, label, hint, hintAlign = 'left', checked, disabled = false, onChange }) {
  return (
    <div className={`settings-toggle ${disabled ? 'disabled' : ''}`}>
      <FieldLabel label={label} hint={hint} hintId={`hint-${id}`} align={hintAlign} />
      <label className="settings-toggle-switch" htmlFor={id}>
        <input id={id} type="checkbox" checked={checked} disabled={disabled} onChange={(event) => onChange(event.target.checked)} aria-label={tr(label)} />
        <i aria-hidden="true" />
      </label>
    </div>
  )
}

function NumberField({ id, label, hint, hintAlign = 'left', value, onChange, ...inputProps }) {
  return (
    <div className="settings-number-field">
      <FieldLabel label={label} hint={hint} hintId={`hint-${id}`} align={hintAlign} />
      <input id={id} type="number" value={value} onChange={(event) => onChange(event.target.value)} required {...inputProps} />
    </div>
  )
}

function FieldLabel({ label, hint, hintId, align = 'left' }) {
  return (
    <div className="settings-control-label">
      <span className="settings-control-label-text">{tr(label)}</span>
      {hint ? <ParameterHint id={hintId} title={label} align={align} {...hint} /> : null}
    </div>
  )
}

function RuntimeFact({ label, value, tone = '', hint, hintId }) {
  return (
    <div className={`settings-runtime-fact ${tone}`}>
      <FieldLabel label={label} hint={hint} hintId={hintId} />
      <strong>{value}</strong>
    </div>
  )
}
