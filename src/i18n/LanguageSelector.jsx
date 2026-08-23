import { tr } from './runtime'
import { useI18n } from './I18nProvider'

const LANGUAGES = [
  { id: 'en', label: 'English' },
  { id: 'pt', label: 'Português' },
  { id: 'es', label: 'Español' },
]

function FlagIcon({ language }) {
  if (language === 'pt') {
    return <svg viewBox="0 0 30 20" aria-hidden="true"><rect width="30" height="20" rx="2" fill="#009739"/><path d="M15 3.2 25 10 15 16.8 5 10Z" fill="#FEDD00"/><circle cx="15" cy="10" r="4" fill="#012169"/><path d="M11.5 9.4c2.5-.8 5.2-.4 7.1.7" fill="none" stroke="#fff" strokeWidth=".7"/></svg>
  }
  if (language === 'es') {
    return <svg viewBox="0 0 30 20" aria-hidden="true"><rect width="30" height="20" rx="2" fill="#AA151B"/><rect y="5" width="30" height="10" fill="#F1BF00"/><rect x="8" y="8" width="2" height="4" rx=".5" fill="#AA151B"/></svg>
  }
  return <svg viewBox="0 0 30 20" aria-hidden="true"><rect width="30" height="20" rx="2" fill="#fff"/><g fill="#B22234"><rect y="0" width="30" height="2"/><rect y="4" width="30" height="2"/><rect y="8" width="30" height="2"/><rect y="12" width="30" height="2"/><rect y="16" width="30" height="2"/></g><rect width="13" height="10.5" fill="#3C3B6E"/><g fill="#fff"><circle cx="3" cy="2.5" r=".7"/><circle cx="6.5" cy="2.5" r=".7"/><circle cx="10" cy="2.5" r=".7"/><circle cx="4.8" cy="5.4" r=".7"/><circle cx="8.3" cy="5.4" r=".7"/><circle cx="3" cy="8.2" r=".7"/><circle cx="6.5" cy="8.2" r=".7"/><circle cx="10" cy="8.2" r=".7"/></g></svg>
}

export function LanguageSelector({ compact = false }) {
  const { language, setLanguage } = useI18n()

  return <div className={`language-selector ${compact ? 'compact' : ''}`} role="group" aria-label={tr("Language")}>
    {LANGUAGES.map((item) => (
      <button
        key={item.id}
        type="button"
        className={language === item.id ? 'active' : ''}
        onClick={() => setLanguage(item.id)}
        title={item.label}
        aria-label={item.label}
        aria-pressed={language === item.id}
      >
        <FlagIcon language={item.id} />
      </button>
    ))}
  </div>
}
