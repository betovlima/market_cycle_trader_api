function monthKey(value) {
  const text = String(value || '').trim()
  if (/^\d{4}-\d{2}$/.test(text)) return text
  const timestamp = Date.parse(text)
  if (!Number.isFinite(timestamp)) return ''
  const date = new Date(timestamp)
  return `${date.getUTCFullYear()}-${String(date.getUTCMonth() + 1).padStart(2, '0')}`
}

export function periodFromSnapshot(value) {
  const end = monthKey(value)
  if (!end) return { start: '', end: '' }
  const year = end.slice(0, 4)
  return { start: `${year}-01`, end }
}

export function periodIsValid(start, end) {
  if (!/^\d{4}-\d{2}$/.test(start || '') || !/^\d{4}-\d{2}$/.test(end || '')) return false
  return end >= start
}

export function filterAnalyticsByPeriod(payload, start, end) {
  const inPeriod = (value) => {
    const key = monthKey(value)
    return key && key >= start && key <= end
  }
  return {
    ...payload,
    equity: (payload?.equity || []).filter((row) => inPeriod(row.timestamp || row.recorded_at)),
    rotations: (payload?.rotations || []).filter((row) => inPeriod(row.executed_at)),
    monthly_returns: (payload?.monthly_returns || []).filter((row) => {
      const key = monthKey(row.month)
      return key && key >= start && key <= end
    }),
  }
}

export function monthsWithData(analytics) {
  const values = new Set()
  for (const row of analytics?.equity || []) {
    const key = monthKey(row.timestamp || row.recorded_at)
    if (key) values.add(key)
  }
  for (const row of analytics?.rotations || []) {
    const key = monthKey(row.executed_at)
    if (key) values.add(key)
  }
  return [...values].sort()
}

export function temporalStudyParameters(run, processing, start, end) {
  const multi = run?.result?.multi_horizon_metrics || {}
  const capital = multi?.shadow_capital || {}
  return {
    source_processing_id: processing?.id || null,
    source_strategy: processing?.strategy_profile_name || run?.strategy_profile_name || null,
    strategy_research_id: run?.strategy_profile_id || processing?.strategy_profile_id || null,
    strategy_research_name: run?.strategy_profile_name || processing?.strategy_profile_name || null,
    strategy_research_revision: run?.strategy_profile_revision ?? processing?.strategy_profile_revision ?? null,
    strategy_research_configuration_hash: run?.strategy_configuration_hash || processing?.strategy_configuration_hash || null,
    strategy_research_kind: run?.strategy_kind || null,
    strategy_research_temporal_variant: run?.temporal_strategy_variant || null,
    temporal_run_id: run?.id || null,
    experiment: run?.experiment || run?.result?.experiment || null,
    model: run?.model_label || run?.result?.model_label || null,
    horizons: run?.horizons || run?.result?.horizons || [],
    research_snapshot_cutoff: run?.research_snapshot_cutoff || run?.analysis_end_date || null,
    timing_base_weak_threshold: capital?.timing_base_weak_threshold ?? null,
    timing_challenger_minimum: capital?.timing_challenger_minimum ?? null,
    timing_minimum_advantage: capital?.timing_minimum_advantage ?? null,
    one_side_cost_rate: capital?.one_side_cost_rate ?? null,
    period_start: start,
    period_end: end,
  }
}

let crcTable = null

function crc32(bytes) {
  if (!crcTable) {
    crcTable = Array.from({ length: 256 }, (_, value) => {
      let crc = value
      for (let bit = 0; bit < 8; bit += 1) crc = (crc >>> 1) ^ (crc & 1 ? 0xedb88320 : 0)
      return crc >>> 0
    })
  }
  let crc = 0xffffffff
  for (const byte of bytes) crc = (crc >>> 8) ^ crcTable[(crc ^ byte) & 0xff]
  return (crc ^ 0xffffffff) >>> 0
}

function zipHeader(size) {
  return new Uint8Array(size)
}

function uint16(target, offset, value) {
  new DataView(target.buffer, target.byteOffset, target.byteLength).setUint16(offset, value, true)
}

function uint32(target, offset, value) {
  new DataView(target.buffer, target.byteOffset, target.byteLength).setUint32(offset, value >>> 0, true)
}

async function deflateRaw(bytes) {
  if (typeof CompressionStream !== 'function') return { method: 0, bytes }
  try {
    const stream = new Blob([bytes]).stream().pipeThrough(new CompressionStream('deflate-raw'))
    return { method: 8, bytes: new Uint8Array(await new Response(stream).arrayBuffer()) }
  } catch {
    return { method: 0, bytes }
  }
}

export async function saveJsonZip(filename, payload) {
  const jsonName = filename.toLowerCase().endsWith('.json') ? filename : `${filename}.json`
  const zipName = jsonName.replace(/\.json$/i, '.zip')
  const nameBytes = new TextEncoder().encode(jsonName)
  const source = new TextEncoder().encode(JSON.stringify(payload, null, 2))
  const compressed = await deflateRaw(source)
  const checksum = crc32(source)
  const flags = 0x0800

  const local = zipHeader(30)
  uint32(local, 0, 0x04034b50)
  uint16(local, 4, 20)
  uint16(local, 6, flags)
  uint16(local, 8, compressed.method)
  uint32(local, 14, checksum)
  uint32(local, 18, compressed.bytes.length)
  uint32(local, 22, source.length)
  uint16(local, 26, nameBytes.length)

  const central = zipHeader(46)
  uint32(central, 0, 0x02014b50)
  uint16(central, 4, 20)
  uint16(central, 6, 20)
  uint16(central, 8, flags)
  uint16(central, 10, compressed.method)
  uint32(central, 16, checksum)
  uint32(central, 20, compressed.bytes.length)
  uint32(central, 24, source.length)
  uint16(central, 28, nameBytes.length)

  const centralOffset = local.length + nameBytes.length + compressed.bytes.length
  const end = zipHeader(22)
  uint32(end, 0, 0x06054b50)
  uint16(end, 8, 1)
  uint16(end, 10, 1)
  uint32(end, 12, central.length + nameBytes.length)
  uint32(end, 16, centralOffset)

  const blob = new Blob([local, nameBytes, compressed.bytes, central, nameBytes, end], { type: 'application/zip' })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = zipName
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  URL.revokeObjectURL(url)
}


function dayKey(value) {
  const timestamp = Date.parse(value || '')
  if (!Number.isFinite(timestamp)) return ''
  return new Date(timestamp).toISOString().slice(0, 10)
}

function normalizedAsset(value) {
  const asset = String(value || 'CASH').trim().toUpperCase()
  return asset || 'CASH'
}

export function attachDecisionContexts(rotations, contexts) {
  const exact = new Map()
  const byDay = new Map()
  for (const context of contexts || []) {
    const day = dayKey(context?.execution_at)
    if (!day) continue
    const fromAsset = normalizedAsset(context?.current_symbol)
    const toAsset = normalizedAsset(context?.target_symbol)
    exact.set(`${day}|${fromAsset}|${toAsset}`, context)
    if (!byDay.has(day)) byDay.set(day, [])
    byDay.get(day).push(context)
  }
  return (rotations || []).map((rotation) => {
    const day = dayKey(rotation?.executed_at)
    const fromAsset = normalizedAsset(rotation?.from_asset)
    const toAsset = normalizedAsset(rotation?.to_asset)
    const matched = exact.get(`${day}|${fromAsset}|${toAsset}`)
      || (byDay.get(day) || []).find((context) => normalizedAsset(context?.target_symbol) === toAsset)
      || (byDay.get(day) || [])[0]
      || null
    return matched ? { ...rotation, decision_context: matched } : rotation
  })
}

export function attachMonthlyDecisionContexts(details, contexts) {
  const result = {}
  for (const [month, detail] of Object.entries(details || {})) {
    result[month] = {
      ...detail,
      movements: attachDecisionContexts(detail?.movements || [], contexts),
    }
  }
  return result
}


export function attachWinnerTransitionAttributions(rotations, attributions) {
  const exact = new Map()
  const byDay = new Map()
  for (const item of attributions || []) {
    const day = dayKey(item?.execution_at)
    if (!day) continue
    const fromAsset = normalizedAsset(item?.from_asset)
    const toAsset = normalizedAsset(item?.to_asset)
    exact.set(`${day}|${fromAsset}|${toAsset}`, item)
    if (!byDay.has(day)) byDay.set(day, [])
    byDay.get(day).push(item)
  }
  return (rotations || []).map((rotation) => {
    const day = dayKey(rotation?.executed_at)
    const fromAsset = normalizedAsset(rotation?.from_asset)
    const toAsset = normalizedAsset(rotation?.to_asset)
    const matched = exact.get(`${day}|${fromAsset}|${toAsset}`)
      || (byDay.get(day) || []).find((item) => normalizedAsset(item?.to_asset) === toAsset)
      || null
    return matched ? { ...rotation, winner_transition: matched } : rotation
  })
}


export function attachMonthlyWinnerTransitionAttributions(details, attributions) {
  const result = {}
  for (const [month, detail] of Object.entries(details || {})) {
    result[month] = {
      ...detail,
      movements: attachWinnerTransitionAttributions(detail?.movements || [], attributions),
    }
  }
  return result
}
