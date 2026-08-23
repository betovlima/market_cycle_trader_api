import { tr } from './i18n/runtime'
import { useCallback, useEffect, useMemo, useState } from 'react'

import { apiFetch } from './api/http'
import { hasCapability } from './auth/capabilities'
import { API, FRONT_VERSION } from './config/env'
import { useI18n } from './i18n/I18nProvider'
import { AppHeader } from './features/backtest/components/AppHeader'
import { BacktestPage } from './features/backtest/components/BacktestPage'
import { StrategyResearchPage } from './features/strategyResearch/StrategyResearchPage'
import { useBacktestWorkspace } from './features/backtest/hooks/useBacktestWorkspace'
import { AdministrationPage } from './features/AdministrationPage'
import { SystemSettingsPage } from './features/SystemSettingsPage'
import { AssetDiscoveryPage } from './features/assetDiscovery/AssetDiscoveryPage'
import { DashboardPage } from './features/dashboard/DashboardPage'
import { DecisionSciencePage } from './features/decisionScience/DecisionSciencePage'
import { LoginPage } from './features/LoginPage'
import { PaperPortfolioDashboard } from './features/paperPortfolio/PaperPortfolioDashboard'

const TAB_CAPABILITIES = {
  dashboard: 'dashboard.view',
  research: 'backtest.view',
  'decision-science': 'backtest.view',
  backtest: 'backtest.view',
  'asset-discovery': 'asset_discovery.view',
  portfolio: 'portfolio.view',
  administration: 'administration.view',
  'system-settings': 'settings.view',
}

function isPermissionDeniedMessage(message) {
  const value = String(message || '').toLowerCase()
  return value.includes('administrator access required')
    || value.includes('trader or administrator access required')
    || value.includes('permission denied')
    || value.includes('forbidden')
}

function AuthenticatedApp({ session, onLogout, onSessionExpired, onSessionRefresh }) {
  const workspace = useBacktestWorkspace()
  const [activeTab, setActiveTab] = useState('dashboard')
  const [dashboardProcessingId, setDashboardProcessingId] = useState('')
  const capabilities = session?.capabilities || {}
  const allowedTabs = useMemo(
    () => Object.entries(TAB_CAPABILITIES).filter(([, capability]) => hasCapability(capabilities, capability)).map(([tab]) => tab),
    [capabilities],
  )

  useEffect(() => {
    if (workspace.running && hasCapability(capabilities, 'backtest.view') && activeTab !== 'research') setActiveTab('backtest')
  }, [activeTab, capabilities, workspace.running])

  useEffect(() => {
    const openDashboardProcessing = (event) => {
      const processingId = String(event?.detail?.processingId || '')
      if (processingId) setDashboardProcessingId(processingId)
      if (hasCapability(capabilities, 'dashboard.view')) setActiveTab('dashboard')
    }
    window.addEventListener('mct:open-dashboard-processing', openDashboardProcessing)
    return () => window.removeEventListener('mct:open-dashboard-processing', openDashboardProcessing)
  }, [capabilities])

  useEffect(() => {
    let active = true
    const refreshSession = async () => {
      try {
        const value = await apiFetch(`${API}/auth/session`)
        if (!active) return
        if (!value?.authenticated) onSessionExpired()
        else onSessionRefresh(value)
      } catch (error) {
        if (active && error?.status === 401) onSessionExpired()
      }
    }
    const interval = window.setInterval(refreshSession, 60_000)
    return () => { active = false; window.clearInterval(interval) }
  }, [onSessionExpired, onSessionRefresh])

  const idleRemaining = session.idle_expires_at
    ? Math.max(0, Math.floor((new Date(session.idle_expires_at).getTime() - Date.now()) / 1000))
    : null

  useEffect(() => {
    if (allowedTabs.includes(activeTab)) return
    setActiveTab(allowedTabs.includes('dashboard') ? 'dashboard' : (allowedTabs[0] || 'dashboard'))
  }, [activeTab, allowedTabs])

  return <div className="app-frame">
    <AppHeader activeTab={activeTab} onTabChange={setActiveTab} session={session} capabilities={capabilities} onLogout={onLogout} />
    {idleRemaining !== null && idleRemaining <= 300 ? <div className="session-expiration-warning">{tr("Your session will expire soon.")}</div> : null}
    {workspace.error && !isPermissionDeniedMessage(workspace.error) ? <div className="global-error"><strong>{tr("Unable to load data")}</strong><span>{tr(workspace.error)}</span><button type="button" onClick={() => workspace.setError('')}>×</button></div> : null}
    <main className="workspace-main">
      {activeTab === 'dashboard' && hasCapability(capabilities, 'dashboard.view') ? <DashboardPage workspace={workspace} capabilities={capabilities} onOpenBacktest={() => setActiveTab('backtest')} initialProcessingId={dashboardProcessingId} /> : null}
      {activeTab === 'research' && hasCapability(capabilities, 'backtest.view') ? <StrategyResearchPage workspace={workspace} capabilities={capabilities} onSessionExpired={onSessionExpired} /> : null}
      {activeTab === 'decision-science' && hasCapability(capabilities, 'backtest.view') ? <DecisionSciencePage capabilities={capabilities} /> : null}
      {activeTab === 'backtest' && hasCapability(capabilities, 'backtest.view') ? <BacktestPage workspace={workspace} capabilities={capabilities} onSessionExpired={onSessionExpired} /> : null}
      {activeTab === 'asset-discovery' && hasCapability(capabilities, 'asset_discovery.view') ? <AssetDiscoveryPage onSessionExpired={onSessionExpired} /> : null}
      {activeTab === 'portfolio' && hasCapability(capabilities, 'portfolio.view') ? <PaperPortfolioDashboard /> : null}
      {activeTab === 'administration' && hasCapability(capabilities, 'administration.view') ? <AdministrationPage onSessionExpired={onSessionExpired} /> : null}
      {activeTab === 'system-settings' && hasCapability(capabilities, 'settings.view') ? <SystemSettingsPage onSessionExpired={onSessionExpired} /> : null}
    </main>
    <footer className="app-footer">
      <span className="app-footer-divider" aria-hidden="true">•</span>
      <span className="app-footer-versions">
        <span>{tr("API v")}{workspace.apiVersion}</span>
        <span>{tr("Front v")}{FRONT_VERSION}</span>
      </span>
    </footer>
  </div>
}

export default function App() {
  useI18n()
  const [state, setState] = useState('checking')
  const [session, setSession] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    let active = true
    apiFetch(`${API}/auth/session`).then((value) => {
      if (!active) return
      if (value?.authenticated) { setSession(value); setState('authenticated') }
      else { setState('anonymous') }
    }).catch((e) => { if (active) { setError(e.message); setState('anonymous') } })
    return () => { active = false }
  }, [])

  const expired = useCallback(() => {
    setSession(null)
    setState('anonymous')
  }, [])

  const logout = useCallback(async () => {
    try {
      await apiFetch(`${API}/auth/logout`, { method: 'POST' })
    } finally {
      expired()
    }
  }, [expired])

  const authenticated = useCallback((value) => {
    setError('')
    setSession(value)
    setState('authenticated')
  }, [])

  if (state === 'checking') return <div className="app-loading">{tr("Checking private session…")}</div>
  if (state !== 'authenticated' || !session) return <><LoginPage onAuthenticated={authenticated} />{error ? <div className="startup-error">{tr(error)}</div> : null}</>
  return <AuthenticatedApp session={session} onLogout={logout} onSessionExpired={expired} onSessionRefresh={setSession} />
}
