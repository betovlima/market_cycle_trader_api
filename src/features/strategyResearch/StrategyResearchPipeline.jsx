import { tr } from '../../i18n/runtime'

const STAGES = [
  { id: 'reference', label: 'Strategy Replay', icon: '↻' },
  { id: 'temporal', label: 'Temporal Intelligence', icon: '⚙' },
  { id: 'clustering', label: 'Regime Clustering', icon: '◍' },
  { id: 'fragile_incumbent', label: 'Fragile Incumbent', icon: '◌' },
  { id: 'emerging_trend', label: 'Emerging Trend', icon: '↗' },
  { id: 'risk', label: 'Risk & Intervention', icon: '◉' },
  { id: 'confidence', label: 'Confidence Calibration', icon: '◇' },
  { id: 'stateful', label: 'Decision Policy Replay', icon: '⇄' },
  { id: 'milp', label: 'MILP Decision Optimization', icon: '◆' },
  { id: 'validation', label: 'Final Validation', icon: '✓' },
]

export const STRATEGY_RESEARCH_STAGES = STAGES

function stageStateLabel(value) {
  if (value === 'completed') return tr('Completed')
  if (value === 'running') return tr('Running')
  if (value === 'failed') return tr('Failed')
  if (value === 'paused') return tr('Paused')
  if (value === 'stopped') return tr('Stopped')
  if (value === 'skipped') return tr('Skipped')
  if (value === 'prepared') return tr('Prepared')
  return tr('Waiting')
}

export function StrategyResearchPipeline({ stageState = {}, selectedStage = 'reference', onSelect, pipelineProgress = null }) {
  return (
    <section className="strategy-research-pipeline-shell" aria-label={tr('Research Pipeline')}>
      <div className="strategy-research-pipeline-track">
        {STAGES.map((stage, index) => {
          const state = stageState[stage.id] || 'waiting'
          return (
            <div className="strategy-research-stage-wrap" key={stage.id}>
              <button
                type="button"
                className={`strategy-research-stage ${state} ${selectedStage === stage.id ? 'selected' : ''}`}
                onClick={() => onSelect?.(stage.id)}
                aria-current={selectedStage === stage.id ? 'step' : undefined}
              >
                <span className="strategy-research-stage-head">
                  <span className="strategy-research-stage-index">{String(index + 1).padStart(2, '0')}</span>
                  <span className={`strategy-research-stage-status-icon ${state}`} aria-hidden="true">
                    {state === 'running' ? <span className="strategy-research-running-gear">⚙</span> : state === 'completed' ? '' : stage.icon}
                  </span>
                </span>
                <strong>{tr(stage.label)}</strong>
                <small>{state === 'running' && Number.isFinite(Number(pipelineProgress)) ? `${stageStateLabel(state)} · ${Math.round(Math.max(0, Math.min(100, Number(pipelineProgress))))}%` : stageStateLabel(state)}</small>
              </button>
            </div>
          )
        })}
      </div>
    </section>
  )
}
