import { useEffect, useMemo, useState } from 'react'

import { tr } from '../../../i18n/runtime'
import { shortDateTime } from '../../../shared/formatters'
import { ParameterHint } from '../../../shared/components/ParameterHint'

function parseHours(value) {
  const hours = String(value || '')
    .split(',')
    .map((item) => Number(item.trim()))
    .filter((item) => Number.isInteger(item) && item >= 0 && item <= 23)
  return [...new Set(hours)].sort((left, right) => left - right)
}

export function DiscoverySettings({ settings, busy, onSave }) {
  const [automaticEnabled, setAutomaticEnabled] = useState(false)
  const [batchSize, setBatchSize] = useState('8')
  const [hours, setHours] = useState('18, 20, 22')
  const [recheckDays, setRecheckDays] = useState('30')
  const [reason, setReason] = useState('')

  useEffect(() => {
    if (!settings) return
    setAutomaticEnabled(Boolean(settings.automatic_enabled))
    setBatchSize(String(settings.batch_size ?? 8))
    setHours((settings.schedule_hours_et || []).join(', '))
    setRecheckDays(String(settings.recheck_days ?? 30))
  }, [settings])

  const parsedHours = useMemo(() => parseHours(hours), [hours])
  const valid = parsedHours.length > 0 && Number(batchSize) >= 1 && Number(recheckDays) >= 1

  function submit(event) {
    event.preventDefault()
    if (!valid) return
    onSave({
      automatic_enabled: automaticEnabled,
      batch_size: Number(batchSize),
      schedule_hours_et: parsedHours,
      recheck_days: Number(recheckDays),
      reason: reason.trim() || 'Asset Discovery schedule updated.',
    })
  }

  return <form className="asset-discovery-settings" onSubmit={submit}>
    <div className="asset-discovery-section-heading">
      <div>
        <span className="eyebrow">{tr('AUTOMATION')}</span>
        <h3>{tr('Discovery schedule')}</h3>
      </div>
      <span className="asset-discovery-next-run">{tr('Next automatic run')}: <strong>{settings?.next_scheduled_at ? shortDateTime(settings.next_scheduled_at) : tr('Disabled')}</strong></span>
    </div>

    <div className="asset-discovery-settings-grid">
      <label className="asset-discovery-toggle-field">
        <span>{tr('Automatic discovery')} <ParameterHint title="Automatic discovery" description="Allows the API scheduler to start bounded candidate batches at the configured Eastern Time hours." /></span>
        <input type="checkbox" checked={automaticEnabled} onChange={(event) => setAutomaticEnabled(event.target.checked)} />
      </label>

      <label>
        <span>{tr('Batch size')} <ParameterHint title="Batch size" description="Maximum number of symbols evaluated in one discovery cycle. Smaller batches spread the historical-data workload through the day." /></span>
        <input type="number" min="1" max="50" value={batchSize} onChange={(event) => setBatchSize(event.target.value)} />
      </label>

      <label>
        <span>{tr('Hours (ET)')} <ParameterHint title="Hours (ET)" description="Comma-separated Eastern Time hours when automatic discovery may start. Each hour runs at most once per day." /></span>
        <input value={hours} onChange={(event) => setHours(event.target.value)} placeholder="18, 20, 22" />
      </label>

      <label>
        <span>{tr('Recheck interval (days)')} <ParameterHint title="Recheck interval (days)" description="How long a previously evaluated asset waits before it becomes eligible for another lightweight discovery pass." /></span>
        <input type="number" min="1" max="365" value={recheckDays} onChange={(event) => setRecheckDays(event.target.value)} />
      </label>

      <label className="asset-discovery-reason-field">
        <span>{tr('Change reason')}</span>
        <input value={reason} onChange={(event) => setReason(event.target.value)} placeholder={tr('Optional audit note')} />
      </label>

      <button type="submit" className="primary-action compact asset-discovery-action" disabled={!valid || busy === 'settings'}>
        {busy === 'settings' ? tr('Saving…') : tr('Save schedule')}
      </button>
    </div>
  </form>
}
