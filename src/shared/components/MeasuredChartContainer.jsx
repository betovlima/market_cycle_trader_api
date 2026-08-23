import { cloneElement, useLayoutEffect, useRef, useState } from 'react'

export function MeasuredChartContainer({ children, fallbackHeight = 280, className = '' }) {
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
    className={`measured-chart-container ${className}`.trim()}
    style={{ width: '100%', height: '100%', minWidth: 0, minHeight: 1 }}
  >
    {cloneElement(children, { width: size.width, height: size.height })}
  </div>
}
