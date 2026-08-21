export const FRONT_VERSION = '4.1.42'

const rawApiBaseUrl = import.meta.env.VITE_API_BASE_URL || ''
export const API_BASE_URL = rawApiBaseUrl.replace(/\/$/, '')
export const API = `${API_BASE_URL}/api`
export const GOOGLE_CLIENT_ID = String(import.meta.env.VITE_GOOGLE_CLIENT_ID || '').trim()


export function resolveApiResourceUrl(resourceUrl) {
  if (!resourceUrl) return ''
  if (/^https?:\/\//i.test(resourceUrl)) return resourceUrl
  if (resourceUrl.startsWith('/api/')) return `${API_BASE_URL}${resourceUrl}`
  if (resourceUrl.startsWith('/')) return `${API_BASE_URL}${resourceUrl}`
  return `${API}/${resourceUrl.replace(/^\/+/, '')}`
}
