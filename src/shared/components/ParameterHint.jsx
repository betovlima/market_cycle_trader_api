import { tr } from '../../i18n/runtime'
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'

const EDGE_GAP = 12
const CARD_GAP = 8
const CLOSE_DELAY_MS = 120

export function ParameterHint({
  id,
  title,
  description,
  relationship = '',
  example = '',
  details = [],
  align = 'left',
}) {
  const triggerRef = useRef(null)
  const cardRef = useRef(null)
  const closeTimerRef = useRef(null)
  const [open, setOpen] = useState(false)
  const [position, setPosition] = useState({ left: EDGE_GAP, top: EDGE_GAP, placement: 'bottom' })

  const normalizedDetails = Array.isArray(details) ? details.filter((item) => item && (item.label || item.value)) : []
  const hasContent = Boolean(description || relationship || example || normalizedDetails.length)

  const cancelClose = useCallback(() => {
    if (closeTimerRef.current) {
      window.clearTimeout(closeTimerRef.current)
      closeTimerRef.current = null
    }
  }, [])

  const showHint = useCallback(() => {
    cancelClose()
    setOpen(true)
  }, [cancelClose])

  const scheduleClose = useCallback(() => {
    cancelClose()
    closeTimerRef.current = window.setTimeout(() => {
      closeTimerRef.current = null
      setOpen(false)
    }, CLOSE_DELAY_MS)
  }, [cancelClose])

  const updatePosition = useCallback(() => {
    const trigger = triggerRef.current
    if (!trigger || typeof window === 'undefined') return

    const triggerRect = trigger.getBoundingClientRect()
    const cardRect = cardRef.current?.getBoundingClientRect()
    const viewportWidth = window.innerWidth
    const viewportHeight = window.innerHeight
    const cardWidth = Math.min(cardRect?.width || 330, Math.max(220, viewportWidth - EDGE_GAP * 2))
    const cardHeight = cardRect?.height || 180

    let left = align === 'right' ? triggerRect.right - cardWidth : triggerRect.left
    left = Math.max(EDGE_GAP, Math.min(left, viewportWidth - cardWidth - EDGE_GAP))

    const roomBelow = viewportHeight - triggerRect.bottom
    const roomAbove = triggerRect.top
    const placeAbove = roomBelow < cardHeight + CARD_GAP + EDGE_GAP && roomAbove > roomBelow
    let top = placeAbove
      ? triggerRect.top - cardHeight - CARD_GAP
      : triggerRect.bottom + CARD_GAP

    top = Math.max(EDGE_GAP, Math.min(top, viewportHeight - cardHeight - EDGE_GAP))

    setPosition({ left, top, placement: placeAbove ? 'top' : 'bottom' })
  }, [align])

  useLayoutEffect(() => {
    if (!open) return undefined
    updatePosition()
    const frame = window.requestAnimationFrame(updatePosition)
    return () => window.cancelAnimationFrame(frame)
  }, [open, updatePosition, description, relationship, example, normalizedDetails.length])

  useEffect(() => {
    if (!open) return undefined

    const handleViewportChange = () => updatePosition()
    const handleEscape = (event) => {
      if (event.key === 'Escape') {
        cancelClose()
        setOpen(false)
        triggerRef.current?.focus()
      }
    }
    const handlePointerDown = (event) => {
      if (triggerRef.current?.contains(event.target) || cardRef.current?.contains(event.target)) return
      cancelClose()
      setOpen(false)
    }

    window.addEventListener('resize', handleViewportChange)
    window.addEventListener('scroll', handleViewportChange, true)
    document.addEventListener('keydown', handleEscape)
    document.addEventListener('pointerdown', handlePointerDown)

    return () => {
      window.removeEventListener('resize', handleViewportChange)
      window.removeEventListener('scroll', handleViewportChange, true)
      document.removeEventListener('keydown', handleEscape)
      document.removeEventListener('pointerdown', handlePointerDown)
    }
  }, [open, cancelClose, updatePosition])

  useEffect(() => () => cancelClose(), [cancelClose])

  if (!hasContent) return null

  const card = open && typeof document !== 'undefined' ? createPortal(
    <span
      ref={cardRef}
      id={id}
      role="tooltip"
      className="parameter-hint-card parameter-hint-card-portal"
      data-placement={position.placement}
      style={{ left: `${position.left}px`, top: `${position.top}px` }}
      onMouseEnter={cancelClose}
      onMouseLeave={scheduleClose}
    >
      <strong>{tr(title)}</strong>
      {description ? <span className="parameter-hint-description">{tr(description)}</span> : null}
      {normalizedDetails.length ? (
        <span className="parameter-hint-detail-list">
          {normalizedDetails.map((item, index) => (
            <span key={`${item.label || 'detail'}-${index}`} className={`parameter-hint-detail-row ${item.tone || ''}`}>
              <span>{tr(item.label || '')}</span>
              <strong>{tr(item.value ?? '')}</strong>
              {item.description ? <small>{tr(item.description)}</small> : null}
            </span>
          ))}
        </span>
      ) : null}
      {relationship ? (
        <>
          <span className="parameter-hint-section-label">{tr("Details")}</span>
          <code>{tr(relationship)}</code>
        </>
      ) : null}
      {example ? <span className="parameter-hint-example"><b>{tr("Example:")}</b> {tr(example)}</span> : null}
    </span>,
    document.body,
  ) : null

  return (
    <span className={`parameter-hint align-${align}`}>
      <button
        ref={triggerRef}
        type="button"
        className="parameter-hint-trigger"
        aria-label={tr("Help for {title}", { title: tr(title) })}
        aria-describedby={open ? id : undefined}
        aria-expanded={open}
        onMouseEnter={showHint}
        onMouseLeave={scheduleClose}
        onFocus={showHint}
        onBlur={scheduleClose}
        onClick={() => {
          cancelClose()
          setOpen(true)
        }}
      >
        ?
      </button>
      {card}
    </span>
  )
}
