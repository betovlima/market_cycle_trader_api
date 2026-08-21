import { tr } from '../../../i18n/runtime'
import { LanguageSelector } from '../../../i18n/LanguageSelector'
import { AnalyticsIcon, BacktestIcon, DashboardIcon, PortfolioIcon, SearchIcon, SettingsIcon } from '../../../shared/components/Icons'
import appLogoUrl from '../../../assets/market-cycle-trader-logo.png'

const NAV_ITEMS = [
  { id: 'dashboard', label: 'Dashboard', Icon: DashboardIcon, capability: 'dashboard.view' },
  { id: 'research', label: 'Strategy Research', Icon: AnalyticsIcon, capability: 'backtest.view' },
  { id: 'backtest', label: 'Backtest', Icon: BacktestIcon, capability: 'backtest.view' },
  { id: 'portfolio', label: 'Portfolio', Icon: PortfolioIcon, capability: 'portfolio.view' },
  { id: 'asset-discovery', label: 'Asset Discovery', Icon: SearchIcon, capability: 'asset_discovery.view' },
  { id: 'administration', label: 'Administration', Icon: DashboardIcon, capability: 'administration.view' },
  { id: 'system-settings', label: 'Settings', Icon: SettingsIcon, capability: 'settings.view' },
]

export function AppHeader({ activeTab, onTabChange, session, capabilities = {}, onLogout }) {
  const navItems = NAV_ITEMS.filter(({ capability }) => capabilities?.[capability] === true)
  return (
    <header className="app-header">
      <div className="brand-area">
        <div className="brand-logo-frame" aria-hidden="true">
          <img className="app-logo" src={appLogoUrl} alt="" width="64" height="64" decoding="async" fetchPriority="high" />
        </div>
        <div className="brand-divider" />
        <div className="brand-copy">
          <h1>{tr("Market Cycle Trader")}</h1>
          <div className="brand-subtitle">{tr("Historical Market Simulation")}</div>
        </div>
      </div>

      <div className="header-right">
        <div className="header-language-primary">
          <LanguageSelector compact />
        </div>
        <nav className="main-nav" aria-label={tr("Main navigation")}>
          {navItems.map(({ id, label, Icon }) => (
            <button key={id} type="button" className={activeTab === id ? 'active' : ''} onClick={() => onTabChange(id)}>
              <Icon size={16} />
              <span>{tr(label)}</span>
            </button>
          ))}
        </nav>
        <div className="session-controls"><span>{session?.display_name || tr('User')}</span><button type="button" onClick={onLogout}>{tr("Sign out")}</button></div>
      </div>
    </header>
  )
}
