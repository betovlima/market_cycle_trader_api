import { getIntlLocale, tr } from '../../../i18n/runtime'
import { useCallback, useMemo, useState } from 'react'
import { createPortal } from 'react-dom'

import { percent } from '../../../shared/formatters'
import { monthParts, returnTone } from '../utils/performance'
import { ChartEmpty } from './AnalyticsPrimitives'
import { MonthlyReturnTooltip } from './MonthlyReturnTooltip'
import './monthlyReturnHeatmap.css'

function monthNames() {
  const formatter = new Intl.DateTimeFormat(getIntlLocale(), { month: 'short', timeZone: 'UTC' })
  return Array.from({ length: 12 }, (_, index) => formatter.format(new Date(Date.UTC(2024, index, 1))).replace('.', ''))
}
const TOOLTIP_WIDTH = 248
const TOOLTIP_PADDING = 12

function selectedModeLabel(mode, simulationLabel = null, referenceLabel = null, excessLabel = null) {
  if (mode === 'reference') return referenceLabel || tr('Reference')
  if (mode === 'excess') return excessLabel || 'S − R'
  return simulationLabel || tr('Simulation')
}

function mapMonthlyReturns(rows, mode) {
  const values = new Map()
  let maxAbs = 0

  ;(rows || []).forEach((row) => {
    const parts = monthParts(row.month)
    if (!parts) return

    const simulation = Number(row.simulation_return)
    const reference = Number(row.reference_return)
    if (!Number.isFinite(simulation) || !Number.isFinite(reference)) return

    const excess = simulation - reference
    const selectedValue = mode === 'reference' ? reference : mode === 'excess' ? excess : simulation

    values.set(`${parts.year}-${parts.month}`, { simulation, reference, excess, selectedValue })
    maxAbs = Math.max(maxAbs, Math.abs(selectedValue))
  })

  return { values, maxAbs }
}


function compoundReturn(values) {
  if (!values.length) return null
  return values.reduce((capital, value) => capital * (1 + value), 1) - 1
}

function average(values) {
  return values.length ? values.reduce((total, value) => total + value, 0) / values.length : null
}

function median(values) {
  if (!values.length) return null
  const ordered = [...values].sort((left, right) => left - right)
  const middle = Math.floor(ordered.length / 2)
  return ordered.length % 2 ? ordered[middle] : (ordered[middle - 1] + ordered[middle]) / 2
}

function standardDeviation(values) {
  if (values.length < 2) return null
  const mean = average(values)
  const variance = values.reduce((total, value) => total + ((value - mean) ** 2), 0) / values.length
  return Math.sqrt(variance)
}

function monthKey(row) {
  const parts = monthParts(row?.month)
  return parts ? `${parts.year}-${String(parts.month).padStart(2, '0')}` : ''
}

function selectedReturn(row, mode) {
  const simulation = Number(row?.simulation_return)
  const reference = Number(row?.reference_return)
  if (!Number.isFinite(simulation) || !Number.isFinite(reference)) return null
  if (mode === 'reference') return reference
  if (mode === 'excess') return simulation - reference
  return simulation
}

function monthlyContext(rows, target, mode) {
  const valid = (rows || []).filter((row) => selectedReturn(row, mode) != null)
  const targetKey = `${target.year}-${String(target.monthNumber).padStart(2, '0')}`
  const index = valid.findIndex((row) => monthKey(row) === targetKey)
  const selectedValues = valid.map((row) => selectedReturn(row, mode))
  const ranked = [...valid].sort((left, right) => selectedReturn(right, mode) - selectedReturn(left, mode))
  const rank = ranked.findIndex((row) => monthKey(row) === targetKey) + 1
  const currentYearRows = valid.filter((row) => monthParts(row.month)?.year === target.year)
  const ytdRows = currentYearRows.filter((row) => monthParts(row.month)?.month <= target.monthNumber)
  const rollingRows = index >= 0 ? valid.slice(Math.max(0, index - 2), index + 1) : []

  let streak = 0
  let streakTone = 'flat'
  if (index >= 0) {
    const current = Number(valid[index]?.simulation_return)
    if (Number.isFinite(current) && current !== 0) {
      streakTone = current > 0 ? 'positive' : 'negative'
      for (let cursor = index; cursor >= 0; cursor -= 1) {
        const value = Number(valid[cursor]?.simulation_return)
        if (!Number.isFinite(value) || value === 0 || (value > 0) !== (current > 0)) break
        streak += 1
      }
    }
  }

  const best = ranked[0]
  const worst = ranked[ranked.length - 1]
  return {
    rank: rank || null,
    count: valid.length,
    average: average(selectedValues),
    median: median(selectedValues),
    volatility: standardDeviation(selectedValues),
    rolling3: compoundReturn(rollingRows.map((row) => selectedReturn(row, mode)).filter(Number.isFinite)),
    ytdSimulation: compoundReturn(ytdRows.map((row) => Number(row.simulation_return)).filter(Number.isFinite)),
    ytdReference: compoundReturn(ytdRows.map((row) => Number(row.reference_return)).filter(Number.isFinite)),
    streak,
    streakTone,
    best: best ? { month: best.month, value: selectedReturn(best, mode) } : null,
    worst: worst ? { month: worst.month, value: selectedReturn(worst, mode) } : null,
  }
}

function fullMonthLabel(year, monthNumber) {
  return new Intl.DateTimeFormat(getIntlLocale(), { month: 'long', year: 'numeric', timeZone: 'UTC' })
    .format(new Date(Date.UTC(year, monthNumber - 1, 1)))
}

function compactMonthLabel(value) {
  const parts = monthParts(value)
  return parts ? fullMonthLabel(parts.year, parts.month) : '—'
}

function ReturnComparisonBar({ label, value, maxAbs, tone }) {
  const width = maxAbs > 0 ? Math.max(4, Math.min(100, Math.abs(value) / maxAbs * 100)) : 4
  return <div className="monthly-return-dialog-bar-row">
    <span>{label}</span>
    <div className="monthly-return-dialog-bar-track"><i className={tone} style={{ width: `${width}%` }} /></div>
    <strong className={tone}>{percent(value)}</strong>
  </div>
}

function MonthlyReturnDialog({ detail, rows, mode, referenceLabel = null, onClose }) {
  if (!detail) return null
  const context = monthlyContext(rows, detail, mode)
  const maxAbs = Math.max(Math.abs(detail.simulation), Math.abs(detail.reference), Math.abs(detail.excess), .000001)
  const selectedTone = returnTone(detail.selectedValue)
  const relativeTone = returnTone(detail.excess)
  const ytdExcess = context.ytdSimulation != null && context.ytdReference != null ? context.ytdSimulation - context.ytdReference : null
  const streakLabel = context.streak
    ? tr('{count} consecutive {direction} months', { count: context.streak, direction: context.streakTone === 'positive' ? tr('positive') : tr('negative') })
    : tr('No active positive or negative streak')

  if (typeof document === 'undefined') return null
  return createPortal(<div className="monthly-return-dialog-backdrop" role="presentation" onMouseDown={onClose}>
    <section className="monthly-return-dialog" role="dialog" aria-modal="true" aria-label={fullMonthLabel(detail.year, detail.monthNumber)} onMouseDown={(event) => event.stopPropagation()}>
      <header className="monthly-return-dialog-header">
        <div>
          <span className="panel-kicker">{tr('CONSISTENCY')}</span>
          <h3>{fullMonthLabel(detail.year, detail.monthNumber)}</h3>
          <p>{tr('Monthly return detail for the selected Backtest processing.')}</p>
        </div>
        <button type="button" className="monthly-return-dialog-close" onClick={onClose} aria-label={tr('Close')}>×</button>
      </header>

      <div className="monthly-return-dialog-metrics">
        <div><span>{tr('Simulation')}</span><strong className={returnTone(detail.simulation)}>{percent(detail.simulation)}</strong></div>
        <div><span>{referenceLabel || tr('Reference')}</span><strong className={returnTone(detail.reference)}>{percent(detail.reference)}</strong></div>
        <div><span>S − R</span><strong className={relativeTone}>{percent(detail.excess)}</strong></div>
        <div><span>{detail.selectedModeLabel}</span><strong className={selectedTone}>{percent(detail.selectedValue)}</strong></div>
      </div>

      <div className="monthly-return-dialog-main">
        <article className="monthly-return-dialog-card">
          <div className="monthly-return-dialog-section-title"><span>{tr('Simulation versus reference')}</span><strong className={relativeTone}>{tr(detail.relativeResult)}</strong></div>
          <ReturnComparisonBar label={tr('Simulation')} value={detail.simulation} maxAbs={maxAbs} tone={returnTone(detail.simulation)} />
          <ReturnComparisonBar label={referenceLabel || tr('Reference')} value={detail.reference} maxAbs={maxAbs} tone={returnTone(detail.reference)} />
          <ReturnComparisonBar label="S − R" value={detail.excess} maxAbs={maxAbs} tone={relativeTone} />
        </article>

        <article className="monthly-return-dialog-card">
          <div className="monthly-return-dialog-section-title"><span>{tr('Context in selected range')}</span><strong>{detail.selectedModeLabel}</strong></div>
          <div className="monthly-return-dialog-insight-grid">
            <div><span>{tr('Month rank')}</span><strong>{context.rank ? `${context.rank} / ${context.count}` : '—'}</strong></div>
            <div><span>{tr('3-month compounded')}</span><strong className={returnTone(context.rolling3)}>{context.rolling3 == null ? '—' : percent(context.rolling3)}</strong></div>
            <div><span>{tr('Average month')}</span><strong className={returnTone(context.average)}>{context.average == null ? '—' : percent(context.average)}</strong></div>
            <div><span>{tr('Median month')}</span><strong className={returnTone(context.median)}>{context.median == null ? '—' : percent(context.median)}</strong></div>
            <div><span>{tr('Monthly volatility')}</span><strong>{context.volatility == null ? '—' : percent(context.volatility)}</strong></div>
            <div><span>{tr('Current Simulation streak')}</span><strong className={context.streakTone}>{streakLabel}</strong></div>
          </div>
        </article>
      </div>

      <div className="monthly-return-dialog-secondary">
        <article>
          <span>{tr('Year-to-date Simulation')}</span>
          <strong className={returnTone(context.ytdSimulation)}>{context.ytdSimulation == null ? '—' : percent(context.ytdSimulation)}</strong>
        </article>
        <article>
          <span>{tr('Year-to-date {reference}', { reference: referenceLabel || tr('Reference') })}</span>
          <strong className={returnTone(context.ytdReference)}>{context.ytdReference == null ? '—' : percent(context.ytdReference)}</strong>
        </article>
        <article>
          <span>{tr('Year-to-date S − R')}</span>
          <strong className={returnTone(ytdExcess)}>{ytdExcess == null ? '—' : percent(ytdExcess)}</strong>
        </article>
      </div>

      <div className="monthly-return-dialog-extremes">
        <div><span>{tr('Best month in selected range')}</span><strong className="positive">{context.best ? `${compactMonthLabel(context.best.month)} · ${percent(context.best.value)}` : '—'}</strong></div>
        <div><span>{tr('Worst month in selected range')}</span><strong className="negative">{context.worst ? `${compactMonthLabel(context.worst.month)} · ${percent(context.worst.value)}` : '—'}</strong></div>
      </div>
    </section>
  </div>, document.body)
}

export function MonthlyReturnHeatmap({ rows, mode, simulationLabel = null, referenceLabel = null, excessLabel = null, onMonthSelect = null }) {
  const [tooltip, setTooltip] = useState(null)
  const [selectedMonth, setSelectedMonth] = useState(null)
  const mapped = useMemo(() => mapMonthlyReturns(rows, mode), [mode, rows])
  const years = useMemo(() => [...new Set((rows || [])
    .map((row) => monthParts(row.month)?.year)
    .filter(Number.isFinite))]
    .sort((left, right) => left - right), [rows])

  const hideTooltip = useCallback(() => setTooltip(null), [])
  const showTooltip = useCallback((event, data) => {
    if (!data || typeof window === 'undefined') return
    const rect = event.currentTarget.getBoundingClientRect()
    const preferredLeft = rect.left + rect.width / 2 - TOOLTIP_WIDTH / 2
    const left = Math.min(
      window.innerWidth - TOOLTIP_WIDTH - TOOLTIP_PADDING,
      Math.max(TOOLTIP_PADDING, preferredLeft),
    )
    const showAbove = rect.top > 210
    setTooltip({
      ...data,
      left,
      top: showAbove ? rect.top - 10 : rect.bottom + 10,
      placement: showAbove ? 'above' : 'below',
    })
  }, [])

  if (!years.length) return <ChartEmpty>{tr("No monthly return observations in the selected range.")}</ChartEmpty>

  const modeLabel = selectedModeLabel(mode, simulationLabel, referenceLabel, excessLabel)
  const months = monthNames()

  return <>
    <div className="analytics-return-heatmap" role="grid" aria-label={tr("Monthly return heatmap")} onMouseLeave={hideTooltip}>
      <div className="analytics-heatmap-head" aria-hidden="true">
        <span />
        {months.map((name) => <span key={name}>{name}</span>)}
      </div>

      {years.map((year) => <div className="analytics-heatmap-row" key={year}>
        <strong>{year}</strong>
        {months.map((monthName, index) => {
          const data = mapped.values.get(`${year}-${index + 1}`)
          const present = Boolean(data && Number.isFinite(data.selectedValue))
          if (!present) return <span key={`${year}-${index}`} role="gridcell" className="analytics-heatmap-cell empty" aria-label={tr('{month} {year}. No observation.', { month: monthName, year })}>—</span>

          const alpha = Math.min(.78, .16 + (mapped.maxAbs ? Math.abs(data.selectedValue) / mapped.maxAbs : 0) * .62)
          const differenceTone = returnTone(data.excess)
          const relativeResult = differenceTone === 'positive' ? 'Simulation outperformed' : differenceTone === 'negative' ? 'Reference outperformed' : 'Same performance'
          const tooltipData = {
            month: monthName,
            monthNumber: index + 1,
            year,
            ...data,
            selectedModeLabel: modeLabel,
            simulationLabel: simulationLabel || tr('Simulation'),
            referenceLabel: referenceLabel || tr('Reference'),
            excessLabel: excessLabel || 'S − R',
            relativeResult,
          }

          return <button
            key={`${year}-${index}`}
            type="button"
            role="gridcell"
            className={`analytics-heatmap-cell ${returnTone(data.selectedValue)}`}
            style={{ '--heat-alpha': alpha }}
            aria-label={tr('{month} {year}. {mode} {selected}. Simulation {simulation}. Reference {reference}. S − R {difference}.', { month: monthName, year, mode: modeLabel, selected: percent(data.selectedValue), simulation: percent(data.simulation), reference: percent(data.reference), difference: percent(data.excess) })}
            onMouseEnter={(event) => showTooltip(event, tooltipData)}
            onFocus={(event) => showTooltip(event, tooltipData)}
            onBlur={hideTooltip}
            onClick={() => { hideTooltip(); if (onMonthSelect) onMonthSelect(tooltipData); else setSelectedMonth(tooltipData) }}
          >{percent(data.selectedValue)}</button>
        })}
      </div>)}
    </div>

    <MonthlyReturnTooltip tooltip={tooltip} />
    {!onMonthSelect ? <MonthlyReturnDialog detail={selectedMonth} rows={rows} mode={mode} referenceLabel={referenceLabel} onClose={() => setSelectedMonth(null)} /> : null}
  </>
}
