# TCC — reprodução da linha de ~US$ 7,38M com Alpaca direta

## Objetivo

Esta branch parte exatamente da linha que reproduziu o resultado:

- API base: `10.8.74`
- experimento base: `raw-split-soft-horizon-consensus-v2`
- branch base: `research/api-v10.8.74-soft-horizon-consensus-batched-inference`
- commit base: `05b765df0496c905a4195948027a9b3b7adf2bce`
- job base: `20260918T234903-52bd06f3`

Resultado reproduzido da 10.8.74:

- Control: US$ 5,551,143.96
- Soft Horizon Consensus: US$ 7,376,955.56
- ganho do Soft: +32.89%
- decisões base alteradas pelo Soft: 11

A 10.8.79 muda somente o transporte dos dados:

```text
10.8.74:
MongoDB frozen RAW collections
    -> mesmas barras RAW
    -> mesmas corporate actions
    -> mesma normalização local de splits
    -> mesmo modelo
    -> mesmos folds
    -> Control x Soft Horizon Consensus

10.8.79:
Alpaca API
    -> snapshot RAW local imutável
    -> mesmas corporate actions
    -> mesma normalização local de splits
    -> mesmo modelo
    -> mesmos folds
    -> Control x Soft Horizon Consensus
```

Não há tuning, CARO ou Optuna nesta branch.

## Versão

- API/pacote: `10.8.79`
- branch: `research/api-v10.8.79-soft-horizon-7m-direct-alpaca`
- runner: `scripts/research_soft_horizon_7m_direct_alpaca.py`
- config congelada: `research/soft_horizon_7m_direct_alpaca_v10_8_79.json`

## Contrato de reprodução

O runner aborta se qualquer uma destas condições mudar:

- janela: `2016-01-01 -> 2026-09-18`
- 56 ativos solicitados
- DOC presente no request antes do guard estrutural
- CLMT presente no request
- `rotation_accelerator=cpu`
- `deterministic_execution=true`
- `penalty_strength=1.0`

CPU é mantida porque esta execução tem como objetivo isolar a troca de transporte
Mongo -> Alpaca direta. A equivalência CPU/GPU já foi estudada separadamente.

O `.env` pode continuar contendo `MCT_ROTATION_ACCELERATOR=cuda`; o valor
persistido `rotation_accelerator=cpu` da configuração congelada tem prioridade
nesta reprodução.

## Dados

Barras:

- Alpaca Stocks Bars API
- feed: SIP
- timeframe: 1Day
- adjustment solicitado à Alpaca: RAW
- início: 2016-01-01
- fim: 2026-09-18, inclusivo

Corporate actions:

- Alpaca Corporate Actions API
- mesmo conjunto de tipos usado para criar
  `alpaca_corporate_actions_full_20260919`
- query start: um ano antes do início da pesquisa
- query end: 2026-09-18
- region: us
- data_quality: complete

Isso replica o recorte do downloader que criou a coleção Mongo usada pela
10.8.74.

## Sem MongoDB

O novo runner não importa nem chama:

- `mongo_repository`
- `create_client`
- `get_database`
- `_latest_job`

O request que antes vinha do job Mongo foi congelado em JSON no repositório.

## Sem alteração da estratégia

A camada de modelo continua usando a mesma configuração da 10.8.74:

- weighted multi-horizon LightGBM Utility
- horizontes: 5, 10, 20, 40, 60
- pesos: 0.10, 0.15, 0.20, 0.30, 0.25
- `n_estimators=329`
- `learning_rate=0.020731`
- `max_depth=3`
- `num_leaves=6`
- `min_child_samples=18`
- `min_child_weight=5`
- `subsample=0.85`
- `colsample_bytree=0.88067`
- `reg_alpha=0.050837`
- `reg_lambda=3.596305`
- `max_bin=255`
- `n_jobs=1`
- early stopping desativado
- Soft Horizon Consensus `penalty_strength=1.0`

Control e Soft usam os mesmos dados e os mesmos folds.

## Semântica interna preservada

Embora o transporte físico seja um snapshot local, o `base_config` entregue ao
`run_rotation_models` mantém os valores usados pela 10.8.74:

```text
market_data_provider = alpaca
alpaca_adjustment = split
research_market_data_mode = database_only
expected_market_data_signature_sha256 = None
research_market_data_snapshot_id = None
```

Esses campos são preservados para que a única mudança experimental seja o
adaptador de entrada. O snapshot local e seu SHA-256 são registrados
separadamente no resultado 10.8.79.

## Precisão RAW

Os CSVs locais são gravados com `float_format="%.17g"`.

Ao reler o snapshot, OHLCV é convertido explicitamente para `float`. Isso
mantém a semântica numérica da coleção Mongo e evita o erro de pandas ao aplicar
fatores de split que geram valores fracionários em `volume`.

## Obter a branch

```bash
git fetch origin
git switch research/api-v10.8.79-soft-horizon-7m-direct-alpaca
git pull --ff-only origin research/api-v10.8.79-soft-horizon-7m-direct-alpaca
```

## Testes

```bash
python -m pytest tests/test_soft_horizon_7m_direct_alpaca.py -q
```

O teste verifica, entre outros pontos:

- linhagem 10.8.74 e commit base;
- 56 ativos originais;
- DOC e CLMT presentes no request;
- CPU/determinismo preservados;
- penalty strength 1.0;
- ausência de dependência Mongo;
- precisão de OHLCV;
- normalização de split com volume fracionário;
- guard estrutural de DOC -> PEAK;
- estabilidade do hash canônico.

## Primeira execução

Para garantir uma nova carga completa da Alpaca:

```bash
python scripts/research_soft_horizon_7m_direct_alpaca.py --replace-snapshot
```

Output padrão:

```text
output/soft_horizon_7m_direct_alpaca_v10879/
```

O runner gera:

- `snapshot/manifest.json`
- `snapshot/raw_bars/<SYMBOL>.csv`
- `snapshot/corporate_actions.jsonl`
- `results/control_predictions.csv`
- `results/control_trades.csv`
- `results/soft_horizon_consensus_predictions.csv`
- `results/soft_horizon_consensus_trades.csv`
- `results/soft_horizon_consensus_decisions.csv`
- `results/data_diagnostics.csv`
- `results/excluded_assets.csv`
- `results/strategy_comparison.csv`
- `results/resolved_final_request.json`
- `results/frozen_research_config.json`
- `results/snapshot_manifest.json`
- `results/summary.json`

## Retomada

Se o download já tiver terminado e uma etapa posterior falhar, não baixar os
dados de novo. Depois da correção, usar:

```bash
python scripts/research_soft_horizon_7m_direct_alpaca.py --reuse-snapshot
```

O runner valida os SHA-256 de todos os arquivos antes de reutilizar o snapshot.

## Auditoria automática contra a 10.8.74

O `summary.json` registra:

- números de referência da 10.8.74;
- delta de capital do Control;
- delta de capital do Soft;
- razão do capital novo / capital de referência;
- diferença no número de decisões alteradas pelo Soft.

Referências:

```text
Control = 5,551,143.963971565
Soft    = 7,376,955.5577371465
Soft changed base actions = 11
```

Esses valores são apenas referência de reprodução. Não são gates e não devem
ser usados para ajustar parâmetros depois da execução.

## Critério científico

A pergunta desta execução é:

> Mantendo integralmente a estratégia 10.8.74, quanto do resultado é reproduzido
> quando o snapshot histórico deixa de vir do MongoDB e é reconstruído a partir
> de uma nova carga direta da Alpaca?

Qualquer diferença deve ser investigada primeiro em:

1. quantidade de candles por ativo;
2. timestamps;
3. precisão OHLCV;
4. corporate actions retornadas;
5. splits aplicados;
6. ativo excluído pelo guard estrutural;
7. folds gerados;
8. decisões do Control;
9. 11 intervenções do Soft.

Não fazer tuning para aproximar artificialmente os US$ 7.38 milhões.


## Correção 10.8.80 — `research_market_data_mode`

A primeira execução da 10.8.79 abortou antes de qualquer chamada à Alpaca com:

```text
ValidationError: research_market_data_mode
Input should be 'backtest_bootstrap_missing' or 'database_only'
input_value='standalone_snapshot'
```

Causa:

- a branch 10.8.79 deriva corretamente do código 10.8.74;
- o schema da 10.8.74 define:
  `ResearchMarketDataMode = Literal["backtest_bootstrap_missing", "database_only"]`;
- porém o JSON congelado inicial tinha herdado
  `research_market_data_mode="standalone_snapshot"` da linha 10.8.75;
- esse valor nem sequer é válido para o schema 10.8.74 e impedia o request de ser
  construído.

A 10.8.80 corrige apenas esse contrato:

```json
"research_market_data_mode": "database_only"
```

Isso também é o valor model-facing usado pelo experimento 10.8.74 de
US$ 7.376.955,56. Portanto a correção reduz, em vez de aumentar, a divergência
em relação ao experimento de referência.

Versão da correção:

- API/pacote: `10.8.80`
- runner: `soft-horizon-7m-direct-alpaca-v1.0.1`
- branch:
  `research/api-v10.8.80-soft-horizon-7m-direct-alpaca-mode-fix`
- config:
  `research/soft_horizon_7m_direct_alpaca_v10_8_80.json`
- output:
  `output/soft_horizon_7m_direct_alpaca_v10880/`

A falha aconteceu em `BacktestExecutionRequest.model_validate(...)`, antes do
download de barras e corporate actions. Portanto essa tentativa não produziu um
snapshot Alpaca final que deva ser preservado.

### Execução 10.8.80

```bash
git fetch origin
git switch research/api-v10.8.80-soft-horizon-7m-direct-alpaca-mode-fix
git pull --ff-only origin research/api-v10.8.80-soft-horizon-7m-direct-alpaca-mode-fix

python -m pytest tests/test_soft_horizon_7m_direct_alpaca.py -q

python scripts/research_soft_horizon_7m_direct_alpaca.py --replace-snapshot
```

Se o download da 10.8.80 terminar e uma etapa posterior falhar, a partir desse
momento usar `--reuse-snapshot` para preservar exatamente a nova carga.
