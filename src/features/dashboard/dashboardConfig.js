export const DASHBOARD_PAGE_SIZE = 10

export const NAVIGATOR_PAGE_SIZE = 10

export const ZOOM_STEP = 0.84


export const DAY_MS = 24 * 60 * 60 * 1000

export const PROBABILITY_METHOD = 'champion_probability'

export const DASHBOARD_HINTS = {
  totalBacktests: {
    description: 'Total number of historical simulation executions currently available in the dashboard summary.',
    relationship: 'Completed executions are shown separately so interrupted or failed runs remain visible without inflating successful-run counts.',
  },
  bestPerformance: {
    description: 'Highest simulation return among the completed backtests available to the dashboard.',
    relationship: 'This is an execution result only. It does not expose model inputs, thresholds, signals or protected strategy parameters.',
  },
  lastBacktest: {
    description: 'Elapsed time since the most recent backtest execution was created.',
    relationship: 'The timestamp comes from the execution history and is independent of the next scheduled market refresh.',
  },
  nextMarketUpdate: {
    description: 'Countdown to the next whole-hour dashboard market refresh reference.',
    relationship: 'This is a display schedule indicator and does not trigger or change trading decisions.',
  },
  date: 'Creation time of the backtest execution.',
  status: 'Current or terminal execution state reported by the backend.',
  totalReturn: 'Total percentage return produced by the simulation for this completed execution.',
  sharpe: 'Risk-adjusted return metric reported by the completed backtest.',
  drawdown: 'Largest peak-to-trough decline observed during the simulation.',
  rotations: 'Number of position changes recorded by the completed simulation.',
  duration: 'Wall-clock time used by the backend to execute the backtest.',
  portfolioGrowth: 'Simulation equity compared with the reference equity for the selected completed backtest. Use the wheel to zoom and drag to pan through time.',
  navigator: 'Completed positions in chronological order. Select one position to connect its holding period with the portfolio curve and detailed view below.',
  selectedPosition: 'Strategy equity and Buy & Hold reference equity over the exact holding interval of the selected position. Switch between portfolio value and Indexed 100 to compare relative performance from the same starting point.',
}
