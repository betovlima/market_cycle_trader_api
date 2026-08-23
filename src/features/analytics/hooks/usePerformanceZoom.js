import { useEffect, useMemo, useRef, useState } from 'react'

import {
  ANALYTICS_ZOOM_STEP,
  clamp,
  minimumAnalyticsZoomSpan,
} from '../utils/performance'

export function usePerformanceZoom({ equityRows, jobId }) {
  const [zoomDomain, setZoomDomain] = useState(null)
  const [isPanning, setIsPanning] = useState(false)
  const chartInteractionRef = useRef(null)
  const panStateRef = useRef(null)

  const fullTimeDomain = useMemo(() => {
    if (equityRows.length < 2) return null
    const start = equityRows[0].timestamp_value
    const end = equityRows[equityRows.length - 1].timestamp_value
    return end > start ? { start, end } : null
  }, [equityRows])

  const minimumTimeSpan = useMemo(() => {
    if (!fullTimeDomain) return 0
    const fullSpan = fullTimeDomain.end - fullTimeDomain.start
    return Math.min(fullSpan, Math.max(minimumAnalyticsZoomSpan(equityRows), fullSpan / 300))
  }, [equityRows, fullTimeDomain])

  const effectiveZoomDomain = useMemo(() => {
    if (!fullTimeDomain) return null
    if (!zoomDomain) return fullTimeDomain

    const fullSpan = fullTimeDomain.end - fullTimeDomain.start
    const requestedSpan = Math.max(minimumTimeSpan, zoomDomain.end - zoomDomain.start)
    if (!Number.isFinite(requestedSpan) || requestedSpan >= fullSpan * .995) return fullTimeDomain

    let start = clamp(zoomDomain.start, fullTimeDomain.start, fullTimeDomain.end - requestedSpan)
    let end = start + requestedSpan
    if (end > fullTimeDomain.end) {
      end = fullTimeDomain.end
      start = end - requestedSpan
    }
    return { start, end }
  }, [fullTimeDomain, minimumTimeSpan, zoomDomain])

  const zoomActive = Boolean(
    fullTimeDomain && effectiveZoomDomain &&
    (effectiveZoomDomain.end - effectiveZoomDomain.start) <
      (fullTimeDomain.end - fullTimeDomain.start) * .995,
  )

  const visibleEquityRows = useMemo(() => {
    if (!zoomActive || !effectiveZoomDomain) return equityRows
    return equityRows.filter((row) =>
      row.timestamp_value >= effectiveZoomDomain.start &&
      row.timestamp_value <= effectiveZoomDomain.end)
  }, [effectiveZoomDomain, equityRows, zoomActive])

  const visibleSpan = effectiveZoomDomain
    ? Math.max(0, effectiveZoomDomain.end - effectiveZoomDomain.start)
    : 0

  const zoomLevel = useMemo(() => {
    if (!zoomActive || !fullTimeDomain || !effectiveZoomDomain) return 1
    return (fullTimeDomain.end - fullTimeDomain.start) /
      Math.max(1, effectiveZoomDomain.end - effectiveZoomDomain.start)
  }, [effectiveZoomDomain, fullTimeDomain, zoomActive])

  useEffect(() => {
    setZoomDomain(null)
    setIsPanning(false)
    panStateRef.current = null
  }, [jobId])

  useEffect(() => {
    const node = chartInteractionRef.current
    if (!node || !fullTimeDomain) return undefined

    const handleWheel = (event) => {
      if (event.deltaY === 0) return
      event.preventDefault()

      const fullSpan = fullTimeDomain.end - fullTimeDomain.start
      if (fullSpan <= 0 || minimumTimeSpan >= fullSpan) return

      const rect = node.getBoundingClientRect()
      const leftInset = Math.min(74, rect.width * .18)
      const rightInset = Math.min(24, rect.width * .08)
      const plotWidth = Math.max(1, rect.width - leftInset - rightInset)
      const ratio = clamp((event.clientX - rect.left - leftInset) / plotWidth, 0, 1)
      const current = effectiveZoomDomain || fullTimeDomain
      const currentSpan = current.end - current.start
      const nextSpan = clamp(
        event.deltaY < 0 ? currentSpan * ANALYTICS_ZOOM_STEP : currentSpan / ANALYTICS_ZOOM_STEP,
        minimumTimeSpan,
        fullSpan,
      )

      if (nextSpan >= fullSpan * .995) {
        setZoomDomain(null)
        return
      }

      const anchor = current.start + ratio * currentSpan
      const start = clamp(
        anchor - ratio * nextSpan,
        fullTimeDomain.start,
        fullTimeDomain.end - nextSpan,
      )
      setZoomDomain({ start, end: start + nextSpan })
    }

    node.addEventListener('wheel', handleWheel, { passive: false })
    return () => node.removeEventListener('wheel', handleWheel)
  }, [effectiveZoomDomain, fullTimeDomain, minimumTimeSpan])

  function beginPan(event) {
    if (!zoomActive || event.button !== 0 || !effectiveZoomDomain || !fullTimeDomain) return
    const rect = event.currentTarget.getBoundingClientRect()
    panStateRef.current = {
      pointerId: event.pointerId,
      clientX: event.clientX,
      width: Math.max(1, rect.width),
      start: effectiveZoomDomain.start,
      end: effectiveZoomDomain.end,
    }
    event.currentTarget.setPointerCapture?.(event.pointerId)
    setIsPanning(true)
  }

  function movePan(event) {
    const pan = panStateRef.current
    if (!pan || pan.pointerId !== event.pointerId || !fullTimeDomain) return
    const span = pan.end - pan.start
    const delta = -((event.clientX - pan.clientX) / pan.width) * span
    const start = clamp(pan.start + delta, fullTimeDomain.start, fullTimeDomain.end - span)
    setZoomDomain({ start, end: start + span })
  }

  function endPan(event) {
    const pan = panStateRef.current
    if (!pan || pan.pointerId !== event.pointerId) return
    panStateRef.current = null
    event.currentTarget.releasePointerCapture?.(event.pointerId)
    setIsPanning(false)
  }

  return {
    chartInteractionRef,
    effectiveZoomDomain,
    isPanning,
    visibleEquityRows,
    visibleSpan,
    zoomActive,
    zoomLevel,
    beginPan,
    movePan,
    endPan,
    resetZoom: () => setZoomDomain(null),
  }
}
