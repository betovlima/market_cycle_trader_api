import { useCallback, useState } from 'react'

import { fetchLatestMilpDecision, materializeMilpDecision, runMilpDecision } from '../api/milpDecisionApi'

export function useMilpDecision() {
  const [result, setResult] = useState(null)
  const [selectedCandidate, setSelectedCandidate] = useState('')

  const clear = useCallback(() => {
    setResult(null)
    setSelectedCandidate('')
  }, [])

  const loadLatest = useCallback(async (runId, processingId, startMonth, endMonth) => {
    if (!runId || !processingId || !startMonth || !endMonth) {
      setResult(null)
      return null
    }
    const value = await fetchLatestMilpDecision(runId, processingId, startMonth, endMonth)
    setResult(value?.id ? value : null)
    return value?.id ? value : null
  }, [])

  const runCandidate = useCallback(async (runId, processingId, startMonth, endMonth) => {
    const value = await runMilpDecision(runId, processingId, startMonth, endMonth)
    setResult(value?.id ? value : null)
    return value
  }, [])

  const materialize = useCallback(async (runId) => {
    if (!runId || !result?.id) throw new Error('MILP Candidate is not available.')
    return materializeMilpDecision(runId, result.id)
  }, [result])

  return {
    result,
    selectedCandidate,
    setSelectedCandidate,
    clear,
    loadLatest,
    runCandidate,
    materialize,
  }
}
