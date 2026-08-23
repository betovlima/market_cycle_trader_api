import { tr } from '../../../i18n/runtime'
import { CandidateCard } from './CandidateCard'
import { money, number, percent } from '../utils/formatters'
import { ControlParity } from './ControlParity'

function statefulMetrics(stateful) {
  return stateful?.candidate_a?.analytics?.metrics || {}
}

function foldMetric(row) {
  return row?.metrics || {}
}

export function FinalValidation({ control, stateful, milp, selectedCandidate, onCandidateSelect }) {
  const controlMetrics = control?.metrics || {}
  const statefulData = statefulMetrics(stateful)
  const milpMetrics = milp?.metrics || {}
  return <section className="milp-workspace milp-final-validation">
    <ControlParity parity={milp?.control_parity} />
    <div className="milp-candidate-grid">
      <CandidateCard title="Control" subtitle="Selected Strategy replay" metrics={controlMetrics} />
      <CandidateCard title="Stateful Candidate" subtitle="Conservative Decision Policy" metrics={statefulData} selectable={Boolean(stateful?.candidate_a)} selected={selectedCandidate === 'stateful'} onSelect={() => onCandidateSelect?.('stateful')} />
      <CandidateCard title="MILP Candidate" subtitle="MILP Decision Optimization" metrics={milpMetrics} selectable={Boolean(milp?.id)} selected={selectedCandidate === 'milp'} onSelect={() => onCandidateSelect?.('milp')} />
    </div>
    <div className="milp-selection-note"><strong>{selectedCandidate ? tr('Candidate selected') : tr('Select a candidate')}</strong><span>{selectedCandidate ? tr('Create Strategy will materialize only the selected candidate.') : tr('Choose Stateful or MILP before creating the next Strategy.')}</span></div>

    {milp?.folds?.length ? <section>
      <div className="milp-section-heading"><div><strong>{tr('MILP fold robustness')}</strong><span>{tr('Chronological out-of-sample validation')}</span></div></div>
      <div className="milp-fold-grid">{milp.folds.map((fold) => {
        const metrics = foldMetric(fold)
        return <div key={fold.fold_id} className="milp-fold-card"><strong>F{fold.fold_id}</strong><span>{String(fold.test_start || '').slice(0, 10)} → {String(fold.test_end || '').slice(0, 10)}</span><div><span>{tr('Ending capital')}</span><b>{money(metrics.ending_capital)}</b></div><div><span>Sharpe</span><b>{number(metrics.sharpe, 3)}</b></div><div><span>MaxDD</span><b>{percent(metrics.maximum_drawdown, 2)}</b></div></div>
      })}</div>
    </section> : null}

    {milp?.cost_stress?.length ? <section>
      <div className="milp-section-heading"><div><strong>{tr('MILP cost stress')}</strong><span>{tr('One-side transaction cost sensitivity')}</span></div></div>
      <div className="milp-cost-grid">{milp.cost_stress.map((row) => <div key={row.one_side_cost_bps}><span>{number(row.one_side_cost_bps, 0)} bps</span><strong>{money(row.ending_capital)}</strong><small>MaxDD {percent(row.maximum_drawdown, 2)}</small></div>)}</div>
    </section> : null}
  </section>
}
