# Market Cycle Trader API v3.30.0 — Rotation Quality Research Console

## Objetivo

A Rotation Quality Research Console executa a pesquisa do Drawdown-Adaptive Rotation Quality Gate sem parâmetros de negócio fixos no frontend. A API publica defaults e limites, valida os valores recebidos, persiste cada execução e continua sendo a fonte de verdade para permissões, PASS/FAIL e resultados.

A decisão do gate usa somente informação disponível no instante da decisão:

- drawdown simulado antes da decisão;
- `entry_rank_score` do incumbent;
- `entry_rank_score` do target Temporal original.

`future_information_used_for_decision=false` permanece parte do contrato.

## Configuração da tela

`GET /api/temporal-rotation-quality-research/config`

Retorna:

- métodos de pesquisa disponíveis (`caro`, `grid`, `manual`);
- defaults editáveis para Research, Validation e Certification;
- ranges/limites aceitos pela API;
- defaults do Unified Adaptive CARO;
- critérios do research gate;
- features causais usadas na decisão.

O frontend não replica essas regras em constantes próprias.

## Research assíncrono

`POST /api/temporal-rotation-quality-research/runs`

### CARO

```json
{
  "source_run_id": "20260816T181543-temporal-a5afd924",
  "search_method": "caro",
  "focus_month": null,
  "control_tolerance_usd": 1.0,
  "caro": {
    "drawdown_trigger_min": -0.15,
    "drawdown_trigger_max": -0.01,
    "rotation_score_tolerance_min": -0.20,
    "rotation_score_tolerance_max": 0.0,
    "trials": 100,
    "seed": 42,
    "candidate_pool_size": 2048,
    "space_filling_pool_size": 1024,
    "exploration_weight": 0.15,
    "minimum_exploration_trials": null,
    "initial_exploration_fraction": 0.45,
    "minimum_exploration_fraction": 0.20,
    "stagnation_recovery_trials": 4,
    "minimum_capital_improvement": 0.0,
    "sharpe_tolerance": 0.0,
    "drawdown_tolerance": 0.0,
    "minimum_worst_fold_return": -1.0
  },
  "research_gate": {
    "minimum_capital_lift": 0.0,
    "minimum_sharpe_delta": 0.0,
    "minimum_max_drawdown_delta": 0.0,
    "required_fold_wins": null
  }
}
```

### Grid

Use `search_method="grid"` e informe `drawdown_triggers` e `rotation_score_tolerances`.

### Manual

Use `search_method="manual"` e informe `manual_candidates`, cada item contendo `drawdown_trigger` e `rotation_score_tolerance`.

A resposta inicial retorna `202 Accepted`. O progresso pode ser acompanhado por:

`GET /api/temporal-rotation-quality-research/{research_id}`

Histórico:

`GET /api/temporal-rotation-quality-research`

Candidatos:

`GET /api/temporal-rotation-quality-research/{research_id}/candidates`

## Validation e Certification

Ambas usam os parâmetros do candidato persistido no Research. Não existe retuning nesta fase.

`POST /api/temporal-rotation-quality-research/{research_id}/validate`

Exemplo de Validation:

```json
{
  "kind": "validation",
  "fold_count": 5,
  "required_fold_wins": 4,
  "candidate_ids": ["RQ-017", "RQ-053"],
  "minimum_capital_lift": 0.0,
  "minimum_sharpe_delta": 0.0,
  "minimum_max_drawdown_delta": 0.0
}
```

Exemplo de Certification:

```json
{
  "kind": "certification",
  "fold_count": 7,
  "required_fold_wins": 6,
  "candidate_ids": ["RQ-017", "RQ-053"],
  "minimum_capital_lift": 0.0,
  "minimum_sharpe_delta": 0.0,
  "minimum_max_drawdown_delta": 0.0
}
```

`fold_count`, `required_fold_wins` e os três thresholds econômicos são parâmetros do request. Os valores exibidos inicialmente pela tela são defaults retornados por `/config`, não regras do frontend.

Status/histórico:

- `GET /api/temporal-rotation-quality-research/{research_id}/validations`
- `GET /api/temporal-rotation-quality-research/{research_id}/validations/{validation_id}`

## Exportação

Research:

`GET /api/temporal-rotation-quality-research/{research_id}/export.zip`

Validation/Certification:

`GET /api/temporal-rotation-quality-research/{research_id}/validations/{validation_id}/export.zip`

O ZIP de evidência contém:

- `summary.json`
- `control.json`
- `candidates.csv`
- `folds.csv`
- `blocked_rotations.csv`
- `validation_policy.json`
- `candidate_details.json`

## Compatibilidade

Os endpoints síncronos anteriores continuam disponíveis:

- `POST /api/temporal-rotation-quality-research`
- `POST /api/temporal-rotation-quality-research/advanced`

A tela nova usa o fluxo assíncrono `/runs`.

## Diagnostics de rotações bloqueadas

O diagnóstico usa uma Validation/Certification concluída como fonte de evidência e reconstrói deterministicamente o mesmo protocolo de folds para recuperar as features Temporal disponíveis no instante de cada decisão bloqueada.

Configuração e catálogo de features vêm de:

`GET /api/temporal-rotation-quality-research/config`

Início assíncrono:

`POST /api/temporal-rotation-quality-research/{research_id}/validations/{validation_id}/diagnostics`

O request recebe `candidate_id`, `lookback_sessions`, `feature_names`, `minimum_group_samples`, `outcome_neutral_band` e `top_feature_count`. Esses valores são parâmetros do request; a tela não contém thresholds estratégicos fixos.

Histórico e acompanhamento:

- `GET /api/temporal-rotation-quality-research/{research_id}/validations/{validation_id}/diagnostics`
- `GET /api/temporal-rotation-quality-research/{research_id}/validations/{validation_id}/diagnostics/{diagnostic_id}`
- `POST /api/temporal-rotation-quality-research/{research_id}/validations/{validation_id}/diagnostics/{diagnostic_id}/stop`

Exportação:

`GET /api/temporal-rotation-quality-research/{research_id}/validations/{validation_id}/diagnostics/{diagnostic_id}/export.zip`

O ZIP contém:

- `summary.json`
- `diagnostic_policy.json`
- `blocked_rotation_diagnostics.csv`
- `feature_separation.csv`
- `fold_summary.csv`
- `metadata.json`

As features usadas para diagnosticar a decisão são exclusivamente contemporâneas ou históricas ao timestamp da decisão. O retorno do intervalo seguinte é usado somente como label posterior para classificar o bloqueio como favorável, prejudicial ou neutro; ele não participa da decisão nem das features causais.


## Strong Challenger Override

A hipótese Strong Challenger mantém o baseline de Rotation Quality congelado e permite que uma rotação originalmente bloqueada seja executada quando o `entry_rank_score` absoluto do challenger atinge o piso configurado.

No modo CARO, o request informa `strong_challenger_override=true`, `baseline_drawdown_trigger`, `baseline_rotation_score_tolerance` e o intervalo `challenger_quality_floor_min/max` dentro de `caro`. Nesse modo o CARO pesquisa somente `challenger_quality_floor`; os dois parâmetros do baseline não são retunados. Grid e Manual continuam disponíveis com parâmetros fornecidos pelo cliente.

O endpoint `/config` retorna defaults e limites para a tela. A decisão usa apenas drawdown e scores disponíveis no timestamp da decisão; retorno futuro não entra na política.

## Analytics mensais de Rotation Quality

Execuções novas concluídas persistem uma representação compactada do histórico da Strategy e do Control para uso do Dashboard.

- `GET /api/temporal-rotation-quality-research/analytics/processings`
- `GET /api/temporal-rotation-quality-research/analytics/processings/{processing_id}?candidate_id=...`
- `GET /api/temporal-rotation-quality-research/analytics/processings/{processing_id}/rotation-period?candidate_id=...&year=...&month=...`

O Dashboard usa esses contratos para montar o heatmap ano × mês em Strategy, Control ou Strategy − Control e para abrir o detalhe mensal com ativos, preços, rotações e timeline de alocação. Execuções históricas anteriores a esta versão não são retreinadas automaticamente apenas para preencher analytics.
