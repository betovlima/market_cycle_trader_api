import { useCallback, useEffect, useRef, useState } from 'react'

function normalizeLayout(value, defaultLayout) {
  if (!Array.isArray(value)) return [...defaultLayout]
  const normalized = value.filter((item) => defaultLayout.includes(item))
  defaultLayout.forEach((item) => {
    if (!normalized.includes(item)) normalized.push(item)
  })
  return normalized.slice(0, defaultLayout.length)
}

function readLayout(storageKey, defaultLayout) {
  if (typeof window === 'undefined') return [...defaultLayout]
  try {
    return normalizeLayout(JSON.parse(window.localStorage.getItem(storageKey) || 'null'), defaultLayout)
  } catch {
    return [...defaultLayout]
  }
}

export function useReorderableCards({ storageKey, defaultLayout }) {
  const [layout, setLayout] = useState(() => readLayout(storageKey, defaultLayout))
  const [draggedId, setDraggedId] = useState('')
  const [dropTargetId, setDropTargetId] = useState('')
  const [justDroppedId, setJustDroppedId] = useState('')
  const dropAnimationTimerRef = useRef(null)

  useEffect(() => {
    if (typeof window === 'undefined') return
    try { window.localStorage.setItem(storageKey, JSON.stringify(layout)) } catch {}
  }, [layout, storageKey])

  useEffect(() => () => {
    if (dropAnimationTimerRef.current) window.clearTimeout(dropAnimationTimerRef.current)
  }, [])

  const reorder = useCallback((sourceId, targetId) => {
    if (!sourceId || !targetId || sourceId === targetId) return
    setLayout((current) => {
      const sourceIndex = current.indexOf(sourceId)
      const targetIndex = current.indexOf(targetId)
      if (sourceIndex < 0 || targetIndex < 0) return current
      const next = [...current]
      next.splice(sourceIndex, 1)
      next.splice(targetIndex, 0, sourceId)
      return next
    })
    setJustDroppedId(sourceId)
    if (dropAnimationTimerRef.current) window.clearTimeout(dropAnimationTimerRef.current)
    dropAnimationTimerRef.current = window.setTimeout(() => setJustDroppedId(''), 430)
  }, [])

  const dragHandleProps = useCallback((cardId) => ({
    onDragStart: (event) => {
      event.stopPropagation()
      setDraggedId(cardId)
      setDropTargetId('')
      event.dataTransfer.effectAllowed = 'move'
      event.dataTransfer.setData('text/plain', cardId)
    },
    onDragEnd: () => {
      setDraggedId('')
      setDropTargetId('')
    },
    onKeyDown: (event) => {
      if (!['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(event.key)) return
      event.preventDefault()
      const currentIndex = layout.indexOf(cardId)
      if (currentIndex < 0) return
      const delta = ['ArrowUp', 'ArrowLeft'].includes(event.key) ? -1 : 1
      const targetIndex = Math.min(layout.length - 1, Math.max(0, currentIndex + delta))
      if (targetIndex !== currentIndex) reorder(cardId, layout[targetIndex])
    },
  }), [layout, reorder])

  const dropZoneProps = useCallback((cardId) => ({
    onDragEnter: (event) => {
      if (!draggedId || draggedId === cardId) return
      event.preventDefault()
      event.dataTransfer.dropEffect = 'move'
      setDropTargetId(cardId)
    },
    onDragOver: (event) => {
      if (!draggedId || draggedId === cardId) return
      event.preventDefault()
      event.dataTransfer.dropEffect = 'move'
      setDropTargetId(cardId)
    },
    onDrop: (event) => {
      event.preventDefault()
      const sourceId = draggedId || event.dataTransfer.getData('text/plain')
      reorder(sourceId, cardId)
      setDraggedId('')
      setDropTargetId('')
    },
  }), [draggedId, reorder])

  const reset = useCallback(() => setLayout([...defaultLayout]), [defaultLayout])
  const customized = layout.length !== defaultLayout.length || layout.some((item, index) => item !== defaultLayout[index])

  return {
    layout,
    draggedId,
    dropTargetId,
    justDroppedId,
    customized,
    dragHandleProps,
    dropZoneProps,
    reset,
  }
}
