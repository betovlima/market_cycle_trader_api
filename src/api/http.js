export class ApiError extends Error {
  constructor(message, status = 0) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

export function formatApiErrorDetail(detail, fallback) {
  if (!detail) return fallback
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) return detail.map((item) => {
    const message = item?.msg || item?.message || String(item)
    const path = Array.isArray(item?.loc)
      ? item.loc.filter((part) => part !== 'body').map(String).join('.')
      : ''
    return path ? `${path}: ${message}` : message
  }).join(' | ')
  if (typeof detail === 'object') return detail.msg || detail.message || JSON.stringify(detail)
  return String(detail)
}

export async function apiFetch(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    credentials: 'include',
    headers: {
      Accept: 'application/json',
      ...(options.body ? { 'Content-Type': 'application/json' } : {}),
      ...(options.headers || {}),
    },
    body: options.body && typeof options.body !== 'string' ? JSON.stringify(options.body) : options.body,
  })
  if (!response.ok) {
    let message = `Request failed with status ${response.status}`
    try {
      const data = await response.json()
      message = formatApiErrorDetail(data.detail, message)
    } catch {
    }
    throw new ApiError(message, response.status)
  }
  if (response.status === 204) return null
  return response.json()
}


export async function downloadFile(url, fallbackFilename = 'download.bin') {
  const response = await fetch(url, {
    credentials: 'include',
    headers: { Accept: 'application/octet-stream, application/zip, text/csv, */*' },
  })
  if (!response.ok) {
    let message = `Request failed with status ${response.status}`
    try {
      const data = await response.json()
      message = formatApiErrorDetail(data.detail, message)
    } catch {
    }
    throw new ApiError(message, response.status)
  }

  const disposition = response.headers.get('Content-Disposition') || ''
  const encodedMatch = disposition.match(/filename\*=UTF-8''([^;]+)/i)
  const plainMatch = disposition.match(/filename="?([^";]+)"?/i)
  const filename = encodedMatch
    ? decodeURIComponent(encodedMatch[1])
    : plainMatch?.[1] || fallbackFilename

  const blob = await response.blob()
  const objectUrl = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = objectUrl
  anchor.download = filename
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  URL.revokeObjectURL(objectUrl)
}
