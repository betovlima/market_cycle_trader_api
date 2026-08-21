import { tr } from '../i18n/runtime'
import { useEffect, useRef, useState } from 'react'

import { apiFetch } from '../api/http'
import { API, GOOGLE_CLIENT_ID } from '../config/env'
import appLogoUrl from '../assets/market-cycle-trader-logo.png'
import { LanguageSelector } from '../i18n/LanguageSelector'
import { getIntlLocale } from '../i18n/runtime'

function accessFromLocation() {
  if (typeof window === 'undefined') return { invitation_id: '', token: '' }
  const hash = new URLSearchParams(window.location.hash.replace(/^#/, ''))
  const query = new URLSearchParams(window.location.search)
  return {
    invitation_id: hash.get('invitation') || query.get('invitation') || '',
    token: hash.get('token') || '',
  }
}

function keepInvitationAndRemoveToken(invitationId) {
  if (typeof window === 'undefined') return
  const url = new URL(window.location.href)
  url.hash = ''
  if (invitationId) url.searchParams.set('invitation', invitationId)
  else url.searchParams.delete('invitation')
  window.history.replaceState({}, document.title, `${url.pathname}${url.search}`)
}

function roleLabel(role) {
  const labels = {
    admin: 'Administrator',
    trader: 'Trader',
    viewer: 'Viewer',
  }
  return tr(labels[role] || 'Authorized')
}

function GoogleIdentityButton({ disabled, onCredential, onError }) {
  const buttonRef = useRef(null)
  const credentialHandlerRef = useRef(onCredential)
  const errorHandlerRef = useRef(onError)

  useEffect(() => { credentialHandlerRef.current = onCredential }, [onCredential])
  useEffect(() => { errorHandlerRef.current = onError }, [onError])

  useEffect(() => {
    if (!GOOGLE_CLIENT_ID || disabled) return undefined
    let cancelled = false
    let attempts = 0
    let timer = null

    function render() {
      if (cancelled) return
      const googleIdentity = window.google?.accounts?.id
      if (!googleIdentity) {
        attempts += 1
        if (attempts > 80) {
          errorHandlerRef.current(tr('Google Sign-In could not be loaded. Refresh the page and try again.'))
          return
        }
        timer = window.setTimeout(render, 100)
        return
      }
      try {
        window.__marketCycleGoogleCredentialHandler = (response) => {
          if (response?.credential) credentialHandlerRef.current(response.credential)
          else errorHandlerRef.current(tr('Google did not return a verified identity credential.'))
        }
        if (!window.__marketCycleGoogleInitialized) {
          googleIdentity.initialize({
            client_id: GOOGLE_CLIENT_ID,
            callback: (response) => window.__marketCycleGoogleCredentialHandler?.(response),
            auto_select: false,
            cancel_on_tap_outside: true,
            ux_mode: 'popup',
          })
          window.__marketCycleGoogleInitialized = true
        }
        if (buttonRef.current) {
          buttonRef.current.replaceChildren()
          googleIdentity.renderButton(buttonRef.current, {
            type: 'standard',
            theme: 'outline',
            size: 'large',
            text: 'continue_with',
            shape: 'rectangular',
            logo_alignment: 'left',
            width: Math.min(400, Math.max(260, buttonRef.current.clientWidth || 360)),
          })
        }
      } catch (error) {
        errorHandlerRef.current(tr(error?.message || 'Unable to initialize Google Sign-In.'))
      }
    }

    render()
    return () => {
      cancelled = true
      if (timer) window.clearTimeout(timer)
    }
  }, [disabled])

  if (!GOOGLE_CLIENT_ID) {
    return <div className="auth-error">{tr("Google access is not configured for this frontend.")}</div>
  }
  return <div className={`google-identity-button ${disabled ? 'disabled' : ''}`} ref={buttonRef} />
}

export function LoginPage({ onAuthenticated }) {
  const initialAccess = useRef(accessFromLocation())
  const [locator, setLocator] = useState(initialAccess.current)
  const [preview, setPreview] = useState(null)
  const [checkingInvitation, setCheckingInvitation] = useState(Boolean(initialAccess.current.invitation_id || initialAccess.current.token))
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  async function loadPreview(nextLocator) {
    if (!nextLocator.invitation_id) {
      setPreview(null)
      setCheckingInvitation(false)
      if (nextLocator.token) {
        setError(tr('This invitation link is incomplete. Ask the administrator for a new link.'))
      }
      keepInvitationAndRemoveToken('')
      return
    }

    setCheckingInvitation(true)
    setError('')
    try {
      const body = { invitation_id: nextLocator.invitation_id }
      if (nextLocator.token) body.token = nextLocator.token
      const value = await apiFetch(`${API}/auth/access/preview`, { method: 'POST', body })
      const normalized = {
        invitation_id: value.invitation_id,
        token: nextLocator.token || '',
      }
      setLocator(normalized)
      setPreview(value)
      keepInvitationAndRemoveToken(value.invitation_id)
    } catch (requestError) {
      setPreview(null)
      setError(tr(requestError.message || 'Unable to open this invitation.'))
      keepInvitationAndRemoveToken(nextLocator.invitation_id || '')
    } finally {
      setCheckingInvitation(false)
    }
  }

  useEffect(() => {
    if (initialAccess.current.invitation_id || initialAccess.current.token) {
      keepInvitationAndRemoveToken(initialAccess.current.invitation_id)
      loadPreview(initialAccess.current)
    }
  }, [])

  async function authenticateGoogle(credential) {
    setBusy(true)
    setError('')
    try {
      const body = { credential }
      if (locator.invitation_id) body.invitation_id = locator.invitation_id
      if (locator.token) body.token = locator.token
      const session = await apiFetch(`${API}/auth/access`, { method: 'POST', body })
      setLocator({ invitation_id: '', token: '' })
      setPreview(null)
      keepInvitationAndRemoveToken('')
      onAuthenticated(session)
    } catch (requestError) {
      setError(tr(requestError.message || 'Unable to verify the Google account.'))
    } finally {
      setBusy(false)
    }
  }

  const disabled = busy || checkingInvitation

  return (
    <main className="auth-page">
      <div className="auth-language-selector"><LanguageSelector /></div>
      <section className="auth-card unified-auth-card">
        <div className="auth-brand">
          <img src={appLogoUrl} alt="" />
          <div>
            <span>{tr("PRIVATE SIMULATION")}</span>
            <h1>{tr("Market Cycle Trader")}</h1>
            <p>{tr("Sign in with an authorized Google account.")}</p>
          </div>
        </div>

        {preview ? (
          <div className="verified-access-panel">
            <div className="verified-access-heading">
              <span>{roleLabel(preview.role)} {tr("invitation")}</span>
              <strong>{preview.guest_name}</strong>
            </div>
            <dl className="verified-access-details">
              <div><dt>{tr("Authorized Google email")}</dt><dd>{preview.masked_email}</dd></div>
              <div><dt>{tr("Access profile")}</dt><dd>{roleLabel(preview.role)}</dd></div>
              <div><dt>{tr("Invitation status")}</dt><dd>{tr(preview.status.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase()))}</dd></div>
              <div><dt>{tr("Expires")}</dt><dd>{new Date(preview.expires_at).toLocaleString(getIntlLocale())}</dd></div>
            </dl>
            <p className="verified-access-note">
              {tr("Continue with the Google account that owns the authorized email. A different account will be rejected.")}</p>
          </div>
        ) : null}

        {error ? <div className="auth-error">{tr(error)}</div> : null}
        <GoogleIdentityButton disabled={disabled} onCredential={authenticateGoogle} onError={setError} />
        {checkingInvitation ? <div className="google-verification-progress"><span className="loading-ring" />{tr("Checking invitation…")}</div> : null}
        {busy ? <div className="google-verification-progress"><span className="loading-ring" />{tr("Verifying identity…")}</div> : null}

      </section>
    </main>
  )
}
