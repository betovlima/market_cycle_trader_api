from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FRONT = ROOT / "market_cycle_trader"


def test_tuning_history_is_removed_from_front_workspace() -> None:
    panel = (FRONT / "src" / "features" / "ModelTuningPanel.jsx").read_text(encoding="utf-8")
    styles = (FRONT / "src" / "styles.css").read_text(encoding="utf-8")
    history_component = FRONT / "src" / "features" / "modelTuning" / "components" / "ModelTuningHistory.jsx"

    assert "ModelTuningHistory" not in panel
    assert "/admin/model-tuning/history?limit=100" not in panel
    assert ".model-tuning-history" not in styles
    assert not history_component.exists()


def test_tuning_workspace_keeps_standard_inner_padding_without_history_section() -> None:
    styles = (FRONT / "src" / "styles.css").read_text(encoding="utf-8")
    assert ".model-tuning-workspace { margin-top: 0; padding: 12px 14px 16px;" in styles
    assert "grid-template-columns: repeat(4,minmax(0,1fr));" in styles


def test_cash_movements_use_a_dedicated_visual_tone() -> None:
    panel = (FRONT / "src" / "features" / "backtest" / "components" / "RotationPanel.jsx").read_text(encoding="utf-8")
    styles = (FRONT / "src" / "styles.css").read_text(encoding="utf-8")
    assert "? 'cash' : ''" in panel
    assert ".rotation-asset.cash" in styles
    assert "#c2a7ff" in styles


def test_capital_movements_breakdown_is_moved_into_existing_hint() -> None:
    panel = (FRONT / "src" / "features" / "backtest" / "components" / "RotationPanel.jsx").read_text(encoding="utf-8")
    primitives = (FRONT / "src" / "features" / "backtest" / "components" / "BacktestPrimitives.jsx").read_text(encoding="utf-8")
    hint = (FRONT / "src" / "shared" / "components" / "ParameterHint.jsx").read_text(encoding="utf-8")
    styles = (FRONT / "src" / "styles.css").read_text(encoding="utf-8")

    assert "hintDetails={[" in panel
    assert "Asset → CASH" in panel
    assert "CASH → Market" in panel
    assert "CASH Sessions" in panel
    assert "Market Exposure" in panel
    assert "note={tr('{asset} asset" not in panel
    assert "hintDetails = []" in primitives
    assert "details={hintDetails}" in primitives
    assert "parameter-hint-detail-list" in hint
    assert ".parameter-hint-detail-row.purple > strong" in styles
    assert ".parameter-hint-detail-row.green > strong" in styles


def test_all_capital_rotation_summary_metrics_use_detailed_existing_hints() -> None:
    panel = (FRONT / "src" / "features" / "backtest" / "components" / "RotationPanel.jsx").read_text(encoding="utf-8")

    for metric_id in (
        'hint-rotation-count',
        'hint-profitable-rotations',
        'hint-realized-pnl',
        'hint-average-holding',
    ):
        start = panel.index(f'id="{metric_id}"')
        end = panel.index('/>', start)
        assert 'hintDetails={[' in panel[start:end]

    assert "Profitable exit rate" in panel
    assert "Gross profitable exits" in panel
    assert "Gross losing exits" in panel
    assert "Average P/L per exit" in panel
    assert "Median holding" in panel
    assert "Shortest holding" in panel
    assert "Longest holding" in panel
    assert "Last capital movement" in panel
    assert "note={tr('{losing}" not in panel
    assert "note={tr('Fees {value}'" not in panel
    assert "note={summary.last_rotation_at" not in panel


def test_tuning_header_is_compact_and_moves_long_explanations_to_hints() -> None:
    panel = (FRONT / "src" / "features" / "ModelTuningPanel.jsx").read_text(encoding="utf-8")
    styles = (FRONT / "src" / "styles.css").read_text(encoding="utf-8")

    assert 'model-tuning-heading model-tuning-heading-compact' in panel
    assert 'model-tuning-hint-target' in panel
    assert 'model-tuning-hint-scope' in panel
    assert 'model-tuning-hint-saved-model' in panel
    assert 'model-tuning-hint-execution' in panel
    assert 'model-tuning-hint-baseline' in panel
    assert 'model-tuning-hint-method' in panel
    assert 'Certified Candidate baseline' in panel
    assert 'model-tuning-baseline-metrics-compact' in panel
    assert 'model-tuning-method-selector-compact' in panel
    assert '.model-tuning-context-grid > .model-tuning-context-card > strong' in styles
    assert 'text-overflow: ellipsis;' in styles
    assert '.model-tuning-baseline-metrics-compact' in styles
    assert 'grid-template-columns: repeat(5,minmax(0,1fr));' in styles


def test_model_tuning_uses_standard_workspace_gutter_after_compaction() -> None:
    styles = (FRONT / "src" / "styles.css").read_text(encoding="utf-8")
    assert 'padding: 14px 16px 18px;' in styles
    assert '.model-tuning-context-grid-wide {' in styles
    assert 'gap: 8px;' in styles


def test_backtest_monthly_capital_movement_heatmap_replaces_timeline_and_opens_month_detail() -> None:
    panel = (FRONT / "src" / "features" / "backtest" / "components" / "RotationPanel.jsx").read_text(encoding="utf-8")
    styles = (FRONT / "src" / "styles.css").read_text(encoding="utf-8")

    assert 'MonthlyCapitalMovementHeatmap rotations={rotations} equity={payload?.equity || []}' in panel
    assert 'Monthly Capital Movement Heatmap' in panel
    assert "['pnl', tr('Realized P/L')]" in panel
    assert "['movements', tr('Movements')]" in panel
    assert "['cash', 'CASH']" in panel
    assert "['holding', tr('Holding')]" in panel
    assert 'MonthlyMovementTooltip' in panel
    assert 'MonthlyMovementDialog' in panel
    assert 'Capital during the month' in panel
    assert 'rotation-month-equity-chart' in panel
    assert 'rotation-month-dialog-table' in panel
    assert 'CapitalPositionTimeline' not in panel
    assert '.rotation-monthly-heatmap-cell.cash' in styles
    assert '.rotation-month-dialog-backdrop' in styles
    assert '.rotation-month-equity-chart path.line' in styles


def test_model_tuning_exposes_unified_caro_without_manual_hypercube_handoff() -> None:
    panel = (FRONT / "src" / "features" / "ModelTuningPanel.jsx").read_text(encoding="utf-8")
    config = (FRONT / "src" / "features" / "modelTuning" / "modelTuningConfig.js").read_text(encoding="utf-8")
    assert "latin_hypercube_then_caro" not in config
    assert "Start Unified CARO" in panel
    assert "Research budget (trials)" in panel
    assert "body.caro_candidate_count" not in panel
    assert "Warm-up Latin Hypercube trials" not in panel



def test_candidate_ranking_uses_responsive_vertical_cards_without_horizontal_scroll() -> None:
    panel = (FRONT / "src" / "features" / "ModelTuningPanel.jsx").read_text(encoding="utf-8")
    styles = (FRONT / "src" / "styles.css").read_text(encoding="utf-8")

    assert 'model-tuning-candidate-grid' in panel
    assert 'model-tuning-candidate-card' in panel
    assert 'CandidateCardMetric' in panel
    assert 'visibleCandidates.map' in panel
    assert "candidate.status !== 'pending'" in panel
    assert "Number(candidate.candidate_id) === Number(run?.current_candidate_id)" in panel
    assert 'model-tuning-ranking-shell' not in panel
    assert 'CandidateRankingHeader' not in panel
    assert 'CandidateSecondaryMetric' not in panel
    assert '.model-tuning-candidate-grid {' in styles
    assert 'grid-template-columns: repeat(4, minmax(0, 1fr));' in styles
    assert '.model-tuning-candidate-metrics {' in styles
    assert 'overflow: visible;' in styles



def test_control_candidate_is_labeled_control_in_champion_gate() -> None:
    panel = (FRONT / "src" / "features" / "ModelTuningPanel.jsx").read_text(encoding="utf-8")
    assert "candidate.is_control\n                ? tr('Control')" in panel
    assert "candidate.is_control ? 'Baseline'" not in panel


def test_control_card_is_loaded_from_certified_backtest_before_starting_tuning() -> None:
    panel = (FRONT / "src" / "features" / "ModelTuningPanel.jsx").read_text(encoding="utf-8")
    assert "const baselineControlCandidate = useMemo(() => {" in panel
    assert "metrics: selectedBaseline.metrics || {}" in panel
    assert "baseline_preview: true" in panel
    assert "const control = activeRun ? (runControlCandidate || baselineControlCandidate) : (baselineControlCandidate || runControlCandidate)" in panel
    assert "{visibleCandidates.length ? (" in panel


def test_control_card_does_not_render_probability_comparison_metrics() -> None:
    panel = (FRONT / "src" / "features" / "ModelTuningPanel.jsx").read_text(encoding="utf-8")
    assert "candidateCardMethod === PROBABILITY_METHOD && !candidate.is_control ? <CandidateCardMetric candidateId={candidate.candidate_id} label=\"P(beat)\"" in panel
    assert "candidateCardMethod === PROBABILITY_METHOD && !candidate.is_control ? <CandidateCardMetric candidateId={candidate.candidate_id} label=\"Expected improvement\"" in panel
    assert "label=\"Champion gate\"" not in panel


def test_candidate_cards_are_compact_and_use_semantic_metric_colors() -> None:
    panel = (FRONT / "src" / "features" / "ModelTuningPanel.jsx").read_text(encoding="utf-8")
    styles = (FRONT / "src" / "styles.css").read_text(encoding="utf-8")
    assert "tone=\"capital\"" in panel
    assert "tone={signedMetricTone(metrics.cagr)}" in panel
    assert "tone=\"cash\"" in panel
    assert "min-height: 27px;" in styles
    assert ".model-tuning-candidate-metric.positive > strong" in styles
    assert ".model-tuning-candidate-metric.negative > strong" in styles
    assert ".model-tuning-candidate-metric.cash > strong" in styles
    assert ".model-tuning-candidate-metric.info > strong" in styles


def test_candidate_parameters_are_opened_in_a_dialog_instead_of_expanding_the_card() -> None:
    panel = (FRONT / "src" / "features" / "ModelTuningPanel.jsx").read_text(encoding="utf-8")
    styles = (FRONT / "src" / "styles.css").read_text(encoding="utf-8")
    assert "function CandidateParametersGrid({ settings })" in panel
    assert "const [parameterCandidateId, setParameterCandidateId] = useState(null)" in panel
    assert "onClick={() => setParameterCandidateId(candidate.candidate_id)}>{tr('Parameters')}" in panel
    assert "<CandidateParametersGrid settings={parameterCandidate.settings} />" in panel
    assert "model-tuning-parameters-overlay" in styles
    assert "model-tuning-parameters-dialog-grid" in styles
    assert "model-tuning-candidate-parameter-list" not in styles
    assert "<CandidateCardParameters settings={candidate.settings} />" not in panel
    assert "model-tuning-candidate-preflight" in styles


def test_main_screen_labels_use_readable_font_scale() -> None:
    styles = (FRONT / "src" / "styles.css").read_text(encoding="utf-8")
    assert "--ui-label-readable: .72rem;" in styles
    assert "--ui-meta-readable: .70rem;" in styles
    assert ".dashboard-workspace-metric > small" in styles
    assert ".backtest-field-label" in styles
    assert ".portfolio-workspace-metric > span" in styles
    assert ".analytics-metric-label > span" in styles
    assert ".admin-field-label > span:first-child" in styles
    assert ".model-tuning-candidate-metric-label" in styles


def test_candidate_view_opens_styled_dialog_instead_of_expanding_below_cards() -> None:
    panel = (FRONT / "src" / "features" / "ModelTuningPanel.jsx").read_text(encoding="utf-8")
    styles = (FRONT / "src" / "styles.css").read_text(encoding="utf-8")
    assert 'onClick={() => setSelectedCandidateId(candidate.candidate_id)}>{tr(\'View\')}' in panel
    assert 'className="model-tuning-candidate-detail-overlay"' in panel
    assert 'className="model-tuning-candidate-detail-dialog"' in panel
    assert 'aria-modal="true"' in panel
    assert '.model-tuning-candidate-detail-overlay {' in styles
    assert '.model-tuning-candidate-detail-dialog {' in styles
    assert 'model-tuning-candidate-detail">' not in panel


def test_front_uses_backend_capabilities_instead_of_role_authorization_rules() -> None:
    app = (FRONT / "src" / "App.jsx").read_text(encoding="utf-8")
    backtest = (FRONT / "src" / "features" / "backtest" / "components" / "BacktestPage.jsx").read_text(encoding="utf-8")
    tuning = (FRONT / "src" / "features" / "ModelTuningPanel.jsx").read_text(encoding="utf-8")
    header = (FRONT / "src" / "features" / "backtest" / "components" / "AppHeader.jsx").read_text(encoding="utf-8")
    dashboard = (FRONT / "src" / "features" / "dashboard" / "DashboardPage.jsx").read_text(encoding="utf-8")
    analytics = (FRONT / "src" / "features" / "analytics" / "AnalyticsPage.jsx").read_text(encoding="utf-8")

    assert "session.role ===" not in app
    assert "session.role !==" not in app
    assert "includes(session.role)" not in app
    assert "TAB_CAPABILITIES" in app
    assert "hasCapability(capabilities, 'backtest.view')" in app
    assert "hasCapability(capabilities, 'portfolio.view')" in app
    assert "capability: 'backtest.view'" in header
    assert "capability: 'portfolio.view'" in header
    assert "VIEWER_NAV_ITEMS" not in header
    assert "TRADER_NAV_ITEMS" not in header
    assert "ADMIN_NAV_ITEMS" not in header
    assert "hasCapability(capabilities, 'backtest.start')" in backtest
    assert "hasCapability(capabilities, 'backtest.export')" in backtest
    assert "hasCapability(capabilities, 'tuning.view')" in backtest
    assert "hasCapability(capabilities, 'tuning.start')" in tuning
    assert "hasCapability(capabilities, 'tuning.export')" in tuning
    assert "hasCapability(capabilities, 'tuning.promote')" in tuning
    assert "hasCapability(capabilities, 'dashboard.strategy_intelligence.view')" in dashboard
    assert "hasCapability(capabilities, 'portfolio.view')" in analytics


def test_viewer_read_only_actions_are_driven_by_backend_capabilities() -> None:
    backtest = (FRONT / "src" / "features" / "backtest" / "components" / "BacktestPage.jsx").read_text(encoding="utf-8")
    tuning = (FRONT / "src" / "features" / "ModelTuningPanel.jsx").read_text(encoding="utf-8")

    assert "researchWorkspaceMode === 'simulation' && canStartBacktest" in backtest
    assert "researchWorkspaceMode === 'tuning' && canViewTuning" in backtest
    assert "capabilities={capabilities}" in backtest
    assert "canStartTuning ? <button" in tuning
    assert "canStopTuning ? <button" in tuning
    assert "canExportTuning && !active" in tuning
    assert "canViewTuningLogs && !candidate.baseline_preview" in tuning
    assert "canPromoteTuning && adoptable" in tuning
    assert "readOnly" not in tuning


def test_read_only_profiles_do_not_render_permission_denied_banners() -> None:
    app = (FRONT / "src" / "App.jsx").read_text(encoding="utf-8")
    backtest = (FRONT / "src" / "features" / "backtest" / "components" / "BacktestPage.jsx").read_text(encoding="utf-8")
    tuning = (FRONT / "src" / "features" / "ModelTuningPanel.jsx").read_text(encoding="utf-8")
    assert "isPermissionDeniedMessage" in app
    assert "workspace.error && !isPermissionDeniedMessage(workspace.error)" in app
    assert "requestError?.status === 403" in backtest
    assert "requestError instanceof ApiError && requestError.status === 403" in tuning


def test_candidate_loader_keeps_visible_steam_at_compact_scale() -> None:
    styles = (FRONT / "src" / "styles.css").read_text(encoding="utf-8")
    panel = (FRONT / "src" / "features" / "ModelTuningPanel.jsx").read_text(encoding="utf-8")
    assert 'className="loader"' in panel
    assert 'const jobProgress = Math.max(0, Math.min(100, Number(candidate.job_progress || 0)))' in panel
    assert 'className="loader-percent"' in panel
    assert '.loader-percent {' in styles
    assert '--size: 0.46px;' in styles
    assert '.loader::before {' in styles
    assert 'width: 1px;' in styles
    assert '--color-5: color-mix(in srgb, var(--candidate-accent) 88%, transparent);' in styles
    assert 'animation: animloader 1s ease-in-out infinite;' in styles
