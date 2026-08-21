import { ParameterHint } from '../../../shared/components/ParameterHint'

export function PerformanceDifferenceHint({ id }) {
  return <span className="analytics-difference-hint">
    <ParameterHint
      id={id}
      title="S − R"
      description="S is Simulation. R is Reference. S − R is the simulation return minus the reference return."
    />
  </span>
}
