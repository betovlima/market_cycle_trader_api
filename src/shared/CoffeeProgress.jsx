import './coffeeProgress.css'

function clampProgress(value) {
  const number = Number(value)
  return Number.isFinite(number) ? Math.max(0, Math.min(100, Math.round(number))) : null
}

export function CoffeeProgress({ progress = null, counter = null, size = 'md', label = '' }) {
  const normalized = clampProgress(progress)
  const display = counter ?? (normalized == null ? null : `${normalized}%`)

  return <span
    className={`coffee-progress coffee-progress-${size}`}
    role={label ? 'status' : undefined}
    aria-label={label || undefined}
    aria-live={label ? 'polite' : undefined}
  >
    {display != null ? <span className="coffee-progress-counter">{display}</span> : null}
  </span>
}
