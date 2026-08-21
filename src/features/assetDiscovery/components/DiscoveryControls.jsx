import { tr, translatedStatus } from '../../../i18n/runtime'
import { PlayIcon } from '../../../shared/components/Icons'

function StopIcon({ size = 16 }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden="true"><rect x="6" y="6" width="12" height="12" rx="1.5" fill="currentColor" /></svg>
}

function runMessage(run) {
  const status = String(run?.status || '').toLowerCase()
  const phase = String(run?.phase || '').toLowerCase()
  if (status === 'stopping') return tr('Stop requested.')
  if (status === 'completed') return tr('Asset Discovery batch completed.')
  if (status === 'stopped') return tr('Asset Discovery stopped safely.')
  if (status === 'failed') return tr('Asset Discovery failed.')
  if (run?.current_symbol) return tr('Evaluating {symbol}.', { symbol: run.current_symbol })
  if (phase === 'discovering') return tr('Loading the market universe.')
  if (status === 'queued') return tr('Queued')
  return tr('No Asset Discovery run is active.')
}

export function DiscoveryControls({ run, active, busy, onStart, onStop, onRefresh, onExport }) {
  const processed = Number(run?.processed_count || 0)
  const attempted = Number(run?.attempted_count || 0)
  const batchSize = Number(run?.batch_size || 0)
  const progress = batchSize > 0 ? Math.min(100, (processed / batchSize) * 100) : 0

  return <section className="asset-discovery-control-strip">
    <div className="asset-discovery-control-copy">
      <span className={`asset-discovery-status-dot ${active ? 'active' : ''}`} />
      <div>
        <strong>{translatedStatus(run?.status || 'idle')}</strong>
        <span>{runMessage(run)}</span>
      </div>
    </div>

    <div className="asset-discovery-run-progress" aria-label={tr('Current batch progress')}>
      <div className="asset-discovery-progress-labels">
        <span>{tr('Batch')} {processed}/{batchSize || '—'} · {tr('{count} attempted', { count: attempted })}</span>
        <strong>{run?.current_symbol || '—'}</strong>
      </div>
      <div className="asset-discovery-progress-track"><span style={{ width: `${progress}%` }} /></div>
    </div>

    <div className="asset-discovery-actions">
      <button type="button" className="primary-action compact asset-discovery-action" onClick={onStart} disabled={active || busy === 'start'}>
        <PlayIcon size={15} /> {tr('Start analysis')}
      </button>
      <button type="button" className="secondary-action compact asset-discovery-action danger-soft" onClick={onStop} disabled={!active || busy === 'stop'}>
        <StopIcon /> {tr('Stop analysis')}
      </button>
      <button type="button" className="secondary-action compact asset-discovery-action" onClick={() => onRefresh({ silent: true })} disabled={Boolean(busy)}>
        {tr('Refresh')}
      </button>
      <button type="button" className="secondary-action compact asset-discovery-action" onClick={onExport} disabled={active || Boolean(busy)}>
        {tr('Export analysis')}
      </button>
    </div>
  </section>
}
