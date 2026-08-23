import { useMemo, useState } from 'react'
import { createPortal } from 'react-dom'

import { getIntlLocale, tr } from '../../i18n/runtime'
import { number, percent } from '../../shared/formatters'
import './regimeClustering.css'

const FEATURE_LABELS = {
  universe_breadth_5: 'Breadth 5d',
  universe_breadth_20: 'Breadth 20d',
  breadth_impulse: 'Breadth impulse',
  spy_realized_volatility_20: 'SPY volatility 20d',
  spy_return_5: 'SPY return 5d',
  spy_return_20: 'SPY return 20d',
  best_vs_second_gap: 'Leader gap',
  position_drawdown_from_peak: 'Position drawdown',
  position_return_since_entry: 'Return since entry',
  score_change_from_entry: 'Score change from entry',
  incumbent_risk_health: 'Incumbent risk health',
  all_horizon_risk_safety: 'Risk safety',
  positive_score_share: 'Positive score share',
  best_score_zscore: 'Best score z-score',
  short_profit_consensus: 'Short profit consensus',
  long_profit_confirmation: 'Long profit confirmation',
  horizon_agreement: 'Horizon agreement',
  recent_rotations_10: 'Recent rotations 10d',
  healthy_leader_share: 'Healthy leader share',
  weak_relative_leader_share: 'Weak leader share',
  whipsaw_leadership_share: 'Whipsaw share',
  no_good_opportunity_share: 'No good opportunity share',
}

const RAW_NUMBER = new Set(['recent_rotations_10', 'best_score_zscore'])
const COLORS = ['cluster-a', 'cluster-b', 'cluster-c', 'cluster-d', 'cluster-e', 'cluster-f']

function fullMonthLabel(value) {
  if (!/^\d{4}-\d{2}$/.test(String(value || ''))) return String(value || '—')
  const [year, month] = String(value).split('-').map(Number)
  return new Intl.DateTimeFormat(getIntlLocale(), { month: 'long', year: 'numeric', timeZone: 'UTC' }).format(new Date(Date.UTC(year, month - 1, 1)))
}

function shortMonthNames() {
  const formatter = new Intl.DateTimeFormat(getIntlLocale(), { month: 'short', timeZone: 'UTC' })
  return Array.from({ length: 12 }, (_, index) => formatter.format(new Date(Date.UTC(2024, index, 1))).replace('.', ''))
}

function featureLabel(value) {
  return tr(FEATURE_LABELS[value] || String(value || '').replaceAll('_', ' '))
}

function featureValue(key, value) {
  if (value == null) return '—'
  return RAW_NUMBER.has(key) ? number(value, 2) : percent(value, 1)
}

function outcomeLabel(value) {
  if (value === 'severe_negative') return tr('Severe negative')
  if (value === 'negative') return tr('Negative')
  return tr('Positive')
}

function separationLabel(value) {
  const score = Number(value)
  if (!Number.isFinite(score)) return tr('Unavailable')
  if (score >= 0.5) return tr('Strong')
  if (score >= 0.25) return tr('Moderate')
  return tr('Weak')
}

function Metric({ label, value, tone = '' }) {
  return <div className="regime-clustering-metric"><span>{tr(label)}</span><strong className={tone}>{value}</strong></div>
}

function MonthDialog({ row, onClose }) {
  if (!row || typeof document === 'undefined') return null
  return createPortal(<div className="regime-clustering-dialog-backdrop" onMouseDown={onClose} role="presentation">
    <section className="regime-clustering-dialog" role="dialog" aria-modal="true" aria-label={fullMonthLabel(row.month)} onMouseDown={(event) => event.stopPropagation()}>
      <header>
        <div><span className="panel-kicker">{tr('REGIME CLUSTERING')}</span><h3>{fullMonthLabel(row.month)}</h3></div>
        <button type="button" onClick={onClose} aria-label={tr('Close')}>×</button>
      </header>
      <div className="regime-clustering-dialog-grid">
        <Metric label="Official monthly return" value={percent(row.official_return, 2)} tone={Number(row.official_return) >= 0 ? 'positive' : 'negative'} />
        <Metric label="Cluster" value={`#${Number(row.cluster_id) + 1}`} />
        <Metric label="Outcome" value={outcomeLabel(row.outcome)} />
        <Metric label="Sessions" value={String(row.sessions ?? '—')} />
      </div>
      <div className="regime-clustering-dialog-columns">
        <article>
          <h4>{tr('Monthly profile')}</h4>
          <div className="regime-clustering-dialog-profile">{Object.entries(row.features || {}).map(([key, value]) => <Metric key={key} label={featureLabel(key)} value={featureValue(key, value)} />)}</div>
        </article>
        <article>
          <h4>{tr('Most similar periods')}</h4>
          <div className="regime-clustering-similar-list">{(row.similar_months || []).map((similar) => <button type="button" key={similar.month} className="regime-clustering-similar-item">
            <span>{fullMonthLabel(similar.month)}</span><strong className={Number(similar.official_return) >= 0 ? 'positive' : 'negative'}>{percent(similar.official_return, 1)}</strong><small>{tr('Distance')} {number(similar.distance, 2)}</small>
          </button>)}</div>
        </article>
      </div>
    </section>
  </div>, document.body)
}

function ClusterDialog({ cluster, onClose }) {
  if (!cluster || typeof document === 'undefined') return null
  return createPortal(<div className="regime-clustering-dialog-backdrop" onMouseDown={onClose} role="presentation">
    <section className="regime-clustering-dialog regime-clustering-cluster-dialog" role="dialog" aria-modal="true" aria-label={`${tr('Cluster')} #${Number(cluster.cluster_id) + 1}`} onMouseDown={(event) => event.stopPropagation()}>
      <header>
        <div><span className="panel-kicker">{tr('CLUSTER DETAIL')}</span><h3>{tr('Cluster')} #{Number(cluster.cluster_id) + 1}</h3></div>
        <button type="button" onClick={onClose} aria-label={tr('Close')}>×</button>
      </header>
      <div className="regime-clustering-dialog-grid">
        <Metric label="Months" value={String(cluster.months ?? '—')} />
        <Metric label="Average return" value={percent(cluster.average_return, 1)} tone={Number(cluster.average_return) >= 0 ? 'positive' : 'negative'} />
        <Metric label="Negative rate" value={percent(cluster.negative_rate, 1)} />
        <Metric label="Severe months" value={String(cluster.severe_negative_months ?? '—')} />
      </div>
      <article className="regime-clustering-month-list"><h4>{tr('Months in this cluster')}</h4><div>{(cluster.months_list || []).map((month) => <span key={month}>{month}</span>)}</div></article>
    </section>
  </div>, document.body)
}

function ScatterPlot({ monthly, selectedMonth, onSelect }) {
  const points = monthly || []
  const width = 920
  const height = 350
  const padding = 38
  const xs = points.map((point) => Number(point.pca_x)).filter(Number.isFinite)
  const ys = points.map((point) => Number(point.pca_y)).filter(Number.isFinite)
  const minX = Math.min(...xs, -1)
  const maxX = Math.max(...xs, 1)
  const minY = Math.min(...ys, -1)
  const maxY = Math.max(...ys, 1)
  const mapX = (value) => padding + ((value - minX) / Math.max(1e-9, maxX - minX)) * (width - padding * 2)
  const mapY = (value) => height - padding - ((value - minY) / Math.max(1e-9, maxY - minY)) * (height - padding * 2)

  const centroids = useMemo(() => {
    const byCluster = new Map()
    points.forEach((point) => {
      const clusterId = Number(point.cluster_id)
      const bucket = byCluster.get(clusterId) || { cluster_id: clusterId, count: 0, x: 0, y: 0 }
      bucket.count += 1
      bucket.x += Number(point.pca_x) || 0
      bucket.y += Number(point.pca_y) || 0
      byCluster.set(clusterId, bucket)
    })
    return [...byCluster.values()].map((bucket) => ({
      cluster_id: bucket.cluster_id,
      pca_x: bucket.x / Math.max(1, bucket.count),
      pca_y: bucket.y / Math.max(1, bucket.count),
    }))
  }, [points])

  return <div className="regime-clustering-card regime-clustering-scatter-card">
    <div className="regime-clustering-section-heading"><strong>{tr('Monthly similarity map')}</strong><span>{tr('Each point is a month positioned by similarity in the clustering feature space.')}</span></div>
    <div className="regime-clustering-centroid-legend"><span className="regime-clustering-centroid-symbol" aria-hidden="true">◆</span><small>{tr('Diamond markers identify the centroid of each cluster.')}</small></div>
    <svg viewBox={`0 0 ${width} ${height}`} className="regime-clustering-scatter" role="img" aria-label={tr('Monthly similarity map')}>
      <line x1={padding} y1={height / 2} x2={width - padding} y2={height / 2} />
      <line x1={width / 2} y1={padding} x2={width / 2} y2={height - padding} />
      {centroids.map((centroid) => {
        const x = mapX(Number(centroid.pca_x) || 0)
        const y = mapY(Number(centroid.pca_y) || 0)
        const colorClass = COLORS[Number(centroid.cluster_id) % COLORS.length]
        return <g key={`centroid-${centroid.cluster_id}`} className={`regime-clustering-centroid ${colorClass}`}>
          <polygon points={`${x},${y - 10} ${x + 10},${y} ${x},${y + 10} ${x - 10},${y}`} />
          <text x={x + 13} y={y - 11}>{`${tr('Cluster')} ${Number(centroid.cluster_id) + 1}`}</text>
        </g>
      })}
      {points.map((point) => {
        const x = mapX(Number(point.pca_x) || 0)
        const y = mapY(Number(point.pca_y) || 0)
        const selected = selectedMonth?.month === point.month
        const severe = point.outcome === 'severe_negative'
        return <g
          key={point.month}
          className={`regime-clustering-point ${COLORS[Number(point.cluster_id) % COLORS.length]} ${selected ? 'selected' : ''} ${severe ? 'severe' : ''}`}
          onClick={(event) => { event.stopPropagation(); onSelect?.(point) }}
          role="button"
          tabIndex={0}
          onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') onSelect?.(point) }}
        >
          <circle cx={x} cy={y} r={selected ? 10 : severe ? 8 : 6} />
          {(selected || severe) ? <text x={x} y={y - 13} textAnchor="middle">{point.month}</text> : null}
        </g>
      })}
    </svg>
  </div>
}

function ClusteringHintDialog({ onClose }) {
  if (typeof document === 'undefined') return null
  return createPortal(<div className="regime-clustering-dialog-backdrop" onMouseDown={onClose} role="presentation">
    <section className="regime-clustering-dialog regime-clustering-hint-dialog" role="dialog" aria-modal="true" aria-label={tr('What does this clustering group?')} onMouseDown={(event) => event.stopPropagation()}>
      <header>
        <div><span className="panel-kicker">{tr('REGIME CLUSTERING')}</span><h3>{tr('What does this clustering group?')}</h3></div>
        <button type="button" onClick={onClose} aria-label={tr('Close')}>×</button>
      </header>
      <div className="regime-clustering-hint-content">
        <p>{tr('The clustering groups months that had similar market and Strategy conditions, using breadth, volatility, leadership, incumbent risk, score quality and temporal signals.')}</p>
        <p>{tr('Official monthly return is not used to create the clusters. It is shown only afterward to help interpret what happened inside each group.')}</p>
        <p>{tr('A centroid is the representative center of a cluster. Months closer to the centroid have a more typical profile for that regime.')}</p>
      </div>
    </section>
  </div>, document.body)
}

function SimilarPeriodsPanel({ row, onOpen }) {
  if (!row) return <div className="regime-clustering-card regime-clustering-selection-card regime-clustering-selection-empty"><span className="panel-kicker">{tr('SELECTED PERIOD')}</span><strong>{tr('No period selected')}</strong><span className="regime-clustering-muted">{tr('Click a point in the similarity map or a month in the timeline to inspect that period.')}</span></div>
  const features = row.features || {}
  return <div className="regime-clustering-card regime-clustering-selection-card">
    <div className="regime-clustering-selection-head"><div><span className="panel-kicker">{tr('SELECTED PERIOD')}</span><h4>{fullMonthLabel(row.month)}</h4></div><span className={`regime-clustering-cluster-pill ${COLORS[Number(row.cluster_id) % COLORS.length]}`}>#{Number(row.cluster_id) + 1}</span></div>
    <div className="regime-clustering-selection-metrics">
      <Metric label="Official monthly return" value={percent(row.official_return, 2)} tone={Number(row.official_return) >= 0 ? 'positive' : 'negative'} />
      <Metric label="Outcome" value={outcomeLabel(row.outcome)} />
      <Metric label="Sessions" value={String(row.sessions ?? '—')} />
      <Metric label="Breadth 5d" value={featureValue('universe_breadth_5', features.universe_breadth_5)} />
      <Metric label="Breadth 20d" value={featureValue('universe_breadth_20', features.universe_breadth_20)} />
      <Metric label="Position drawdown" value={featureValue('position_drawdown_from_peak', features.position_drawdown_from_peak)} />
      <Metric label="Incumbent risk health" value={featureValue('incumbent_risk_health', features.incumbent_risk_health)} />
    </div>
    <strong className="regime-clustering-subtitle">{tr('Most similar periods')}</strong>
    <div className="regime-clustering-similar-list compact">{(row.similar_months || []).slice(0, 5).map((similar) => <div key={similar.month} className="regime-clustering-similar-item">
      <span>{fullMonthLabel(similar.month)}</span><strong className={Number(similar.official_return) >= 0 ? 'positive' : 'negative'}>{percent(similar.official_return, 1)}</strong><small>{number(similar.distance, 2)}</small>
    </div>)}</div>
    <button type="button" className="regime-clustering-detail-button" onClick={() => onOpen?.(row)}>{tr('View full period details')}</button>
  </div>
}

function ClusterProfileHeatmap({ clusters, features }) {
  return <div className="regime-clustering-card regime-clustering-profile-card">
    <div className="regime-clustering-section-heading"><strong>{tr('Cluster profile heatmap')}</strong><span>{tr('Rows are clusters. Values are average standardized feature levels (z-scores).')}</span></div>
    <div className="regime-clustering-heatmap-grid">
      <div className="regime-clustering-heatmap-head"><span>{tr('Cluster')}</span>{features.map((feature) => <strong key={feature}>{featureLabel(feature)}</strong>)}</div>
      {clusters.map((cluster) => <div className="regime-clustering-heatmap-row" key={cluster.cluster_id}><span className={`regime-clustering-cluster-pill ${COLORS[Number(cluster.cluster_id) % COLORS.length]}`}>#{Number(cluster.cluster_id) + 1}</span>{features.map((feature) => {
        const value = Number(cluster?.feature_zscores?.[feature])
        const tone = value > 0.75 ? 'pos-strong' : value > 0.25 ? 'pos-soft' : value < -0.75 ? 'neg-strong' : value < -0.25 ? 'neg-soft' : 'neutral'
        return <div className={`regime-clustering-heatmap-cell ${tone}`} key={`${cluster.cluster_id}-${feature}`}>{number(value, 2)}</div>
      })}</div>)}
    </div>
  </div>
}

function Timeline({ monthly, onSelect }) {
  const years = [...new Set((monthly || []).map((row) => String(row.month || '').slice(0, 4)))].filter(Boolean)
  const byMonth = new Map((monthly || []).map((row) => [row.month, row]))
  const monthLabels = shortMonthNames()
  return <div className="regime-clustering-card regime-clustering-timeline-card">
    <div className="regime-clustering-section-heading"><strong>{tr('Cluster timeline')}</strong><span>{tr('Cluster by month. The small marker identifies negative official months.')}</span></div>
    <div className="regime-clustering-timeline-grid">
      <div className="regime-clustering-timeline-head"><span />{monthLabels.map((month) => <strong key={month}>{month}</strong>)}</div>
      {years.map((year) => <div className="regime-clustering-timeline-row" key={year}><strong>{year}</strong>{Array.from({ length: 12 }, (_, index) => {
        const key = `${year}-${String(index + 1).padStart(2, '0')}`
        const row = byMonth.get(key)
        if (!row) return <span key={key} className="regime-clustering-timeline-cell missing">—</span>
        return <button type="button" key={key} className={`regime-clustering-timeline-cell ${COLORS[Number(row.cluster_id) % COLORS.length]} ${Number(row.official_return) < 0 ? 'negative' : ''}`} onClick={() => onSelect?.(row)} aria-label={`${fullMonthLabel(row.month)} · ${tr('Cluster')} ${Number(row.cluster_id) + 1} · ${percent(row.official_return, 1)}`}>
          <span>{Number(row.cluster_id) + 1}</span>{Number(row.official_return) < 0 ? <i aria-hidden="true" /> : null}
        </button>
      })}</div>)}
    </div>
  </div>
}

export function RegimeClusteringPanel({ analysis }) {
  const [selectedMonth, setSelectedMonth] = useState(null)
  const [selectedCluster, setSelectedCluster] = useState(null)
  const [detailMonth, setDetailMonth] = useState(null)
  const [showHint, setShowHint] = useState(false)
  const monthly = analysis?.monthly || []
  const clusters = analysis?.clusters || []
  const topFeatures = useMemo(() => (analysis?.feature_importance || []).slice(0, 6).map((row) => row.feature), [analysis])
  const activeMonth = selectedMonth

  if (!analysis || String(analysis.status || '').toLowerCase() !== 'completed') {
    return <div className="regime-clustering-empty">{tr(analysis?.failure_message || 'Regime Clustering will appear after Temporal Intelligence completes.')}</div>
  }

  const silhouette = analysis?.summary?.silhouette_score

  return <section className="regime-clustering-panel">
    <div className="regime-clustering-heading">
      <div><span className="panel-kicker">{tr('EXPLORATORY STEP')}</span><h4>{tr('Regime Clustering')}</h4></div>
      <div className="regime-clustering-heading-actions"><button type="button" className="regime-clustering-hint-button" onClick={() => setShowHint(true)} aria-label={tr('What does this clustering group?')}>?</button><span className="regime-clustering-badge">{tr('Exploratory diagnostic')}</span></div>
    </div>

    <div className="regime-clustering-summary-grid">
      <Metric label="Months" value={String(analysis?.summary?.months ?? '—')} />
      <Metric label="Clusters" value={String((analysis?.summary?.cluster_count ?? 0) || '—')} />
      <Metric label="Silhouette" value={number(silhouette, 3)} />
      <Metric label="Separation quality" value={separationLabel(silhouette)} />
      <Metric label="Negative months" value={String(analysis?.summary?.negative_months ?? '—')} />
      <Metric label="Severe months ≤ -5%" value={String(analysis?.summary?.severe_negative_months ?? '—')} />
    </div>

    <div className="regime-clustering-main-grid">
      <SimilarPeriodsPanel row={activeMonth} onOpen={setDetailMonth} />
      <ScatterPlot monthly={monthly} selectedMonth={activeMonth} onSelect={setSelectedMonth} />
    </div>

    <ClusterProfileHeatmap clusters={clusters} features={topFeatures} />
    <Timeline monthly={monthly} onSelect={setSelectedMonth} />

    <div className="regime-clustering-card">
      <div className="regime-clustering-section-heading"><strong>{tr('Cluster summary')}</strong><span>{tr('Natural groups of similar months found without using the monthly return as an input.')}</span></div>
      <div className="regime-clustering-clusters">{clusters.map((cluster) => <button type="button" key={cluster.cluster_id} className="regime-clustering-cluster-card" onClick={() => setSelectedCluster(cluster)}>
        <header><span className={`regime-clustering-cluster-pill ${COLORS[Number(cluster.cluster_id) % COLORS.length]}`}>#{Number(cluster.cluster_id) + 1}</span><strong>{outcomeLabel(cluster.dominant_outcome)}</strong></header>
        <div className="regime-clustering-cluster-stats">
          <span><small>{tr('Months')}</small><strong>{cluster.months ?? '—'}</strong></span>
          <span><small>{tr('Average return')}</small><strong className={Number(cluster.average_return) >= 0 ? 'positive' : 'negative'}>{percent(cluster.average_return, 1)}</strong></span>
          <span><small>{tr('Negative rate')}</small><strong>{percent(cluster.negative_rate, 1)}</strong></span>
          <span><small>{tr('Severe months')}</small><strong>{cluster.severe_negative_months ?? '—'}</strong></span>
        </div>
        <small className="regime-clustering-open-detail">{tr('Click to inspect cluster months')}</small>
      </button>)}</div>
    </div>

    <MonthDialog row={detailMonth} onClose={() => setDetailMonth(null)} />
    <ClusterDialog cluster={selectedCluster} onClose={() => setSelectedCluster(null)} />
    {showHint ? <ClusteringHintDialog onClose={() => setShowHint(false)} /> : null}
  </section>
}
