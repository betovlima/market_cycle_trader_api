import { tr } from '../../../i18n/runtime'
import { useEffect, useRef } from 'react'

export function ExecutionStatus({ workspace, modelLabel = "" }) {
  const { job } = workspace
  const logRef = useRef(null)
  const logs = Array.isArray(job?.logs) ? job.logs : []

  useEffect(() => {
    const element = logRef.current
    if (element) element.scrollTop = element.scrollHeight
  }, [job?.id, logs.length])

  if (!job || !['queued', 'running', 'failed', 'interrupted'].includes(job.status)) return null

  const progress = Math.max(0, Math.min(100, Number(job.progress ?? 0)))
  const progressLabel = progress.toFixed(progress % 1 ? 1 : 0)
  const detail = job.progress_detail && typeof job.progress_detail === 'object'
    ? job.progress_detail
    : {}
  const runIndex = Number(detail.run_index || 0)
  const runCount = Number(detail.run_count || job.total_runs || 0)
  const foldIndex = Number(detail.fold_index || 0)
  const foldCount = Number(detail.fold_count || 0)
  const trainedModels = Number(detail.trained_models || 0)
  const totalModels = Number(detail.total_models || 0)
  const phase = String(detail.phase || '').trim()
  const device = String(detail.device || '').trim()

  return (
    <section className="execution-panel" aria-live="polite">
      <div className="execution-status-row">
        <div>
          <span className={`status-dot ${job.status}`} />
          <strong>{tr(job.stage || 'Preparing simulation')}</strong>
          <small>{job.completed_runs ?? 0} {tr("of")}{' '}{job.total_runs ?? 0} {tr("runs")}</small>
        </div>
        <span>{progressLabel}%</span>
      </div>
      {(runCount > 0 || foldCount > 0 || phase || modelLabel) ? (
        <div className="execution-detail-grid" aria-label={tr("Detailed training progress")}>
          {modelLabel ? <span><small>{tr("Model")}</small><strong>{modelLabel}</strong></span> : null}
          {runCount > 0 ? <span><small>{tr("Run")}</small><strong>{Math.max(0, runIndex)}/{runCount}</strong></span> : null}
          {foldCount > 0 ? <span><small>{tr("Fold")}</small><strong>{Math.max(0, foldIndex)}/{foldCount}</strong></span> : null}
          {phase ? <span className="phase"><small>{tr("Phase")}</small><strong>{tr(phase)}</strong></span> : null}
          {totalModels > 0 ? <span><small>{tr("Models")}</small><strong>{trainedModels}/{totalModels}</strong></span> : null}
          {device ? <span><small>{tr("Device")}</small><strong>{device}</strong></span> : null}
        </div>
      ) : null}
      <div
        className="progress-track"
        role="progressbar"
        aria-label={tr("Backtest execution progress")}
        aria-valuemin="0"
        aria-valuemax="100"
        aria-valuenow={progress}
      >
        <span style={{ width: `${progress}%` }} />
      </div>
      <div className="execution-log">
        <div className="execution-log-title">{tr("Execution log")}</div>
        <pre ref={logRef}>{logs.length ? logs.slice(-120).join('\n') : tr('Waiting for execution messages…')}</pre>
      </div>
    </section>
  )
}
