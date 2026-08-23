import { tr } from '../../../i18n/runtime'
import { CandidateCard } from './CandidateCard'
import { DecisionMap } from './DecisionMap'
import { number, percent } from '../utils/formatters'
import { ControlParity } from './ControlParity'

function statefulMetrics(stateful) {
  return stateful?.candidate_a?.analytics?.metrics || {}
}

export function DecisionCandidates({ stateful, milp, selectedCandidate, onCandidateSelect }) {
  const milpMetrics = milp?.metrics || {}
  const attribution = milp?.attribution || {}
  const solver = milp?.solver || {}
  return <section className="milp-workspace">
    <ControlParity parity={milp?.control_parity} />
    <div className="milp-candidate-grid">
      <CandidateCard
        title="Stateful Candidate"
        subtitle="Conservative Decision Policy"
        metrics={statefulMetrics(stateful)}
        selectable={Boolean(stateful?.candidate_a)}
        selected={selectedCandidate === 'stateful'}
        onSelect={() => onCandidateSelect?.('stateful')}
      />
      <CandidateCard
        title="MILP Candidate"
        subtitle="MILP Decision Optimization"
        metrics={milpMetrics}
        selectable={Boolean(milp?.id)}
        selected={selectedCandidate === 'milp'}
        onSelect={() => onCandidateSelect?.('milp')}
        extra={milp?.id ? <div className="milp-solver-strip"><span>{tr('Decisions solved')} <strong>{number(solver.decisions_solved, 0)}</strong></span><span>{tr('Average solve time')} <strong>{number(solver.average_solve_ms, 3)} ms</strong></span></div> : null}
      />
    </div>

    {milp?.id ? <>
      <div className="milp-attribution-grid">
        <div><span>{tr('Same decision')}</span><strong>{number(attribution.same_decision, 0)}</strong></div>
        <div><span>{tr('Different decision')}</span><strong>{number(attribution.different_decision, 0)}</strong></div>
        <div><span>{tr('MILP better')}</span><strong className="positive">{number(attribution.milp_better, 0)}</strong></div>
        <div><span>{tr('Control better')}</span><strong className="negative">{number(attribution.control_better, 0)}</strong></div>
        <div><span>{tr('Neutral')}</span><strong>{number(attribution.neutral, 0)}</strong></div>
      </div>
      <DecisionMap result={milp} />
      <details className="milp-config"><summary>{tr('MILP configuration')}</summary><div>{Object.entries(milp.configuration || {}).map(([key, value]) => <div key={key}><span>{key.replaceAll('_', ' ')}</span><strong>{typeof value === 'number' ? number(value, 4) : String(value)}</strong></div>)}</div></details>
      <div className="milp-research-only"><strong>{tr('Research only')}</strong><span>{tr('MILP strategies cannot be promoted to Winner until live runtime parity is implemented.')}</span></div>
    </> : <div className="milp-empty">{tr('MILP Decision Optimization results will appear here after this stage runs.')}</div>}
  </section>
}
