import { tr } from '../../i18n/runtime'
import { SearchIcon } from '../../shared/components/Icons'
import { shortDateTime } from '../../shared/formatters'
import { CandidateTable } from './components/CandidateTable'
import { DiscoveryControls } from './components/DiscoveryControls'
import { DiscoverySettings } from './components/DiscoverySettings'
import { RunHistory } from './components/RunHistory'
import { useAssetDiscovery } from './useAssetDiscovery'
import './assetDiscovery.css'

export function AssetDiscoveryPage({ onSessionExpired }) {
  const discovery = useAssetDiscovery({ onSessionExpired })
  const counts = discovery.status?.counts || {}
  const settings = discovery.status?.settings
  const run = discovery.status?.run

  if (discovery.loading && !discovery.status) {
    return <section className="asset-discovery-page"><div className="page-loading">{tr('Loading Asset Discovery…')}</div></section>
  }

  return <section className="asset-discovery-page">
    <div className="asset-discovery-workspace">
      <header className="asset-discovery-header">
        <div className="asset-discovery-title-icon"><SearchIcon size={21} /></div>
        <div>
          <span className="eyebrow">{tr('RESEARCH UNIVERSE')}</span>
          <h2>{tr('Asset Discovery')}</h2>
        </div>
        <div className="asset-discovery-header-meta"><span>{tr('Historical base')}</span><strong>{run?.status === 'running' ? tr('Refreshing incrementally') : tr('Cached and incremental')}</strong></div>
      </header>

      {discovery.error ? <div className="inline-error">{discovery.error}</div> : null}
      {discovery.notice ? <div className="inline-notice">{discovery.notice}</div> : null}

      <DiscoveryControls run={run} active={discovery.active} busy={discovery.busy} onStart={discovery.start} onStop={discovery.stop} onRefresh={discovery.load} onExport={discovery.exportAnalysis} />

      <div className="asset-discovery-summary-strip">
        <div><span>{tr('Candidates')}</span><strong>{counts.candidate || 0}</strong></div>
        <div><span>{tr('Watchlist')}</span><strong>{counts.watchlist || 0}</strong></div>
        <div><span>{tr('Rejected')}</span><strong>{counts.rejected || 0}</strong></div>
        <div><span>{tr('Skipped')}</span><strong>{counts.skipped || 0}</strong></div>
        <div><span>{tr('Failed')}</span><strong>{counts.failed || 0}</strong></div>
        <div><span>{tr('Next automatic run')}</span><strong>{settings?.next_scheduled_at ? shortDateTime(settings.next_scheduled_at) : tr('Disabled')}</strong></div>
      </div>

      <DiscoverySettings settings={settings} busy={discovery.busy} onSave={discovery.saveSettings} />
      <CandidateTable candidates={discovery.candidates} />
      <RunHistory runs={discovery.runs} />

    </div>
  </section>
}
