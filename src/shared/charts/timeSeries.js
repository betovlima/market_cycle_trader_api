export function timestampValue(value) {
  if (!value) return null
  const parsed = new Date(value).getTime()
  return Number.isFinite(parsed) ? parsed : null
}

export function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value))
}

export function minimumZoomSpan(points, timestampKey = 'timestamp_value', minimumPoints = 8) {
  if (points.length < 2) return 0
  const gaps = []
  for (let index = 1; index < points.length; index += 1) {
    const gap = Number(points[index]?.[timestampKey]) - Number(points[index - 1]?.[timestampKey])
    if (Number.isFinite(gap) && gap > 0) gaps.push(gap)
  }
  if (!gaps.length) return 0
  gaps.sort((left, right) => left - right)
  const medianGap = gaps[Math.floor(gaps.length / 2)]
  return medianGap * Math.max(2, minimumPoints - 1)
}

export function nearestTimeSeriesIndex(points, targetTimestamp, timestampKey = 'timestamp_value', valueKey = null) {
  if (!points.length || targetTimestamp === null) return -1
  let nearestIndex = -1
  let nearestDistance = Number.POSITIVE_INFINITY
  points.forEach((point, index) => {
    const timestamp = Number(point?.[timestampKey])
    if (!Number.isFinite(timestamp)) return
    if (valueKey && !Number.isFinite(Number(point?.[valueKey]))) return
    const distance = Math.abs(timestamp - targetTimestamp)
    if (distance < nearestDistance) {
      nearestIndex = index
      nearestDistance = distance
    }
  })
  return nearestIndex
}
