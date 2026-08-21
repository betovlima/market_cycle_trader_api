export const ACTIVE_JOB_STATUSES = new Set(['queued', 'running'])

export const STATUS_LABELS = {
  draft: 'Draft',
  backtest: 'Backtest',
  candidate: 'Candidate',
  superseded_candidate: 'Superseded candidate',
  promoted_candidate: 'Promoted candidate',
  winner: 'Winner',
  former_winner: 'Former winner',
}

export const STRATEGY_FIELD_HINTS = {
  name: {
    description: 'Human-readable name used to identify this research strategy in the catalog and backtest selection.',
    relationship: 'Renaming a draft does not change the protected Trader winner.',
  },
  description: {
    description: 'Short explanation of the purpose of this research revision so later comparisons remain understandable.',
    relationship: 'Use it to record the intent of the test, not confidential credentials or runtime secrets.',
  },
  search: {
    description: 'Filters the editable configuration by visible label, technical parameter name, group name or available schema metadata.',
    relationship: 'Filtering changes only what is visible in this page; it never changes the strategy configuration.',
  },
  changeReason: {
    description: 'Optional audit note explaining why this strategy revision is being changed.',
    relationship: 'Saving an editable draft revision does not require a note; provide one only when it adds useful research context.',
  },
}

export const BOUNDARY_HINTS = {
  winner: {
    description: 'Protected strategy snapshot currently used by the Trader.',
    relationship: 'Research edits remain isolated until an explicitly validated candidate is promoted.',
  },
  backtest: {
    description: 'Strategy revision selected as the shared source for Simulation Backtest, Model Tuning and Temporal Intelligence.',
    relationship: 'Selecting a Strategy Research baseline does not change the Trader winner.',
  },
  candidate: {
    description: 'Single validated strategy revision eligible for promotion to Trader winner.',
    relationship: 'A candidate represents the exact revision that completed its qualifying backtest.',
  },
  lifecycle: {
    description: 'Lifecycle protection keeps only one active Candidate, one active Promoted Candidate and one protected Trader Winner at a time.',
    relationship: 'Only the current Candidate and current Trader winner are protected from deletion. Historical candidates, promoted candidates and former winners may be deleted when no longer needed.',
  },
}
