import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'

import {
  DEFAULT_LANGUAGE,
  LANGUAGE_STORAGE_KEY,
  getIntlLocale,
  normalizeLanguage,
  setCurrentLanguage,
  tr,
} from './runtime'

const I18nContext = createContext(null)

function initialLanguage() {
  if (typeof window === 'undefined') return DEFAULT_LANGUAGE
  return normalizeLanguage(window.localStorage.getItem(LANGUAGE_STORAGE_KEY) || DEFAULT_LANGUAGE)
}

export function I18nProvider({ children }) {
  const [language, setLanguageState] = useState(() => {
    const resolved = initialLanguage()
    setCurrentLanguage(resolved)
    return resolved
  })

  useEffect(() => {
    if (typeof document !== 'undefined') {
      document.documentElement.lang = language === 'pt' ? 'pt-BR' : language === 'es' ? 'es' : 'en'
    }
  }, [language])

  const setLanguage = useCallback((nextLanguage) => {
    const resolved = normalizeLanguage(nextLanguage)
    setCurrentLanguage(resolved)
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(LANGUAGE_STORAGE_KEY, resolved)
      document.documentElement.lang = resolved === 'pt' ? 'pt-BR' : resolved === 'es' ? 'es' : 'en'
    }
    setLanguageState(resolved)
  }, [])

  const value = useMemo(() => ({
    language,
    locale: getIntlLocale(language),
    setLanguage,
    t: tr,
  }), [language, setLanguage])

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>
}

export function useI18n() {
  const context = useContext(I18nContext)
  if (!context) throw new Error('useI18n must be used inside I18nProvider')
  return context
}
