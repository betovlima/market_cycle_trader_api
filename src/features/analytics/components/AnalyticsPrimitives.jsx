import { cloneElement, useLayoutEffect, useRef, useState } from 'react'
import { tr } from '../../../i18n/runtime'
import { ParameterHint } from '../../../shared/components/ParameterHint'


export function AnalyticsResponsiveContainer({ children, fallbackHeight = 260 }) {
  const hostRef = useRef(null)
  const [size, setSize] = useState({ width: 960, height: fallbackHeight })

  useLayoutEffect(() => {
    const host = hostRef.current
    if (!host) return undefined

    let animationFrame = 0
    const measure = () => {
      cancelAnimationFrame(animationFrame)
      animationFrame = requestAnimationFrame(() => {
        const rect = host.getBoundingClientRect()
        const nextWidth = Math.max(1, Math.floor(rect.width || host.clientWidth || 960))
        const nextHeight = Math.max(1, Math.floor(rect.height || host.clientHeight || fallbackHeight))
        setSize((current) => current.width === nextWidth && current.height === nextHeight
          ? current
          : { width: nextWidth, height: nextHeight })
      })
    }

    measure()
    const resizeObserver = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(measure)
    resizeObserver?.observe(host)
    window.addEventListener('resize', measure)

    return () => {
      cancelAnimationFrame(animationFrame)
      resizeObserver?.disconnect()
      window.removeEventListener('resize', measure)
    }
  }, [fallbackHeight])

  return <div
    ref={hostRef}
    className="analytics-chart-render-host"
    style={{ width: '100%', height: '100%', minWidth: 0, minHeight: 1 }}
  >
    {cloneElement(children, { width: size.width, height: size.height })}
  </div>
}

export function SectionHeading({ kicker, title, description = '', action = null, hint = null, hintId = '' }) {
  return <div className="analytics-section-heading">
    <div>
      <span className="panel-kicker">{tr(kicker)}</span>
      <div className="analytics-section-title">
        <h2>{tr(title)}</h2>
        {hint ? <ParameterHint id={hintId || `analytics-section-${String(title).toLowerCase().replace(/[^a-z0-9]+/g, '-')}`} title={tr(title)} {...hint} /> : null}
      </div>
      {description ? <p>{tr(description)}</p> : null}
    </div>
    {action}
  </div>
}

export function ChartCell({ kicker, title, children, className = '', action = null }) {
  return <div className={`analytics-chart-cell ${className}`}>
    <div className="analytics-chart-cell-heading">
      <div><span>{tr(kicker)}</span><strong>{tr(title)}</strong></div>
      {action}
    </div>
    {children}
  </div>
}

export function ChartEmpty({ children = 'Not enough observations for this chart.' }) {
  return <div className="analytics-empty">{typeof children === 'string' ? tr(children) : children}</div>
}

export function AnalyticsModeTabs({ value, onChange, items, label }) {
  return <div className="analytics-mode-tabs" role="tablist" aria-label={tr(label)}>
    {items.map((item) => <button
      key={item.value}
      type="button"
      role="tab"
      aria-selected={value === item.value}
      className={value === item.value ? 'active' : ''}
      onClick={() => onChange(item.value)}
    >{tr(item.label)}</button>)}
  </div>
}

export function AnalyticsDragHandle({ label, onDragStart, onDragEnd, onKeyDown }) {
  return <span
    className="analytics-drag-handle"
    draggable
    role="button"
    tabIndex={0}
    aria-label={tr("Move {label}. Drag to reorder or use the arrow keys.", { label: tr(label) })}
    title={tr("Drag to reorder")}
    onDragStart={onDragStart}
    onDragEnd={onDragEnd}
    onKeyDown={onKeyDown}
  ><span aria-hidden="true">⋮⋮</span></span>
}

export function AnalyticsMetric({ label, value, note, tone = '', description = '' }) {
  return <article className={`analytics-workspace-metric ${tone}`}>
    <div className="analytics-metric-label">
      <span>{tr(label)}</span>
      {description ? <ParameterHint
        id={`analytics-metric-${label.toLowerCase().replace(/[^a-z0-9]+/g, '-')}`}
        title={tr(label)}
        description={tr(description)}
      /> : null}
    </div>
    <strong>{value}</strong>
    <small>{tr(note)}</small>
  </article>
}
