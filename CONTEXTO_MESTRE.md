# CONTEXTO_MESTRE

## 2026-10-07 - API v10.8.88 manual current-session control

Depois da promocao bem-sucedida da Strategy #29 para Winner, os botoes de
Manual Recovery permaneceram desabilitados quando nao existia um Paper run
agendado para a sessao regular atual.

A causa estava no contrato de `paper_market_manual_recovery_status()`:
`can_prepare` exigia simultaneamente mercado aberto e um scheduled Paper run
para a sessao corrente.

A v10.8.88 remove apenas essa dependencia do scheduler.

Novo comportamento:

- `Reanalyze Winner for today` fica disponivel durante o mercado regular
  aberto mesmo sem scheduled run;
- o runtime reconcilia Mongo x Alpaca, atualiza os dados do Winner, recalibra
  usando apenas barras diarias concluidas e cria um Paper plan para a sessao
  atual;
- se nao existir run agendado, o plano usa a abertura regular das 09:30
  America/New_York da sessao corrente;
- nenhuma ordem e enviada durante a reanalise;
- `Retry plan on Alpaca` habilita depois que existir um plano preparado ou
  uma contingencia recuperavel.

Protecoes mantidas:

- mercado regular precisa estar aberto;
- plano `executing`, `recovering` ou `executed` nao pode ser substituido;
- planos executados nao sao sobrescritos;
- execucao manual continua exigindo confirmacao explicita;
- reconciliacao Mongo x Alpaca e checagem de ordens conflitantes continuam;
- `paused`/`stopped` bloqueiam execucao normal;
- `exit_only` continua impedindo nova entrada;
- a mudanca nao altera a Strategy #29, seus parametros, a posicao XSD ou o
  backtest validado.

Branch: `feature/v10.8.88-manual-current-session-control`.


## 2026-10-07 - API v10.8.87 stale live-market refresh lock recovery

Producao confirmou que a Strategy #29 U67 ja esta compativel com Trader:

- `trader_compatibility.eligible=true`;
- `code=tcc_u67_live_runtime_ready`.

O botao Promote to WINNER continuou desabilitado porque o documento de controle
manteve um lock stale de sincronizacao:

- `live_market_refresh_in_progress=true`;
- `live_market_refresh_started_at=2026-09-28T23:21:42.578Z`;
- `live_market_refresh_source=premarket_plan_refresh`;
- `winner_promotion_guard.code=daily_calibration_data_sync`.

A v10.8.87 cria o endpoint administrativo protegido:

`POST /api/research/tcc-u67/recover-stale-live-market-lock`

Ele so libera o lock quando estiver com mais de seis horas e nao houver:

- Paper run em preparacao/calibracao;
- Paper run executando ordens;
- trade plan com status `executing`.

A recuperacao nao envia ordens, nao muda Winner, nao reinicializa Paper State e
nao altera a posicao XSD.

Fluxo apos deploy:

1. chamar o endpoint de recovery;
2. esperar `code=stale_lock_released`;
3. consultar `GET /api/admin/strategies/control`;
4. confirmar `live_market_refresh_in_progress=false`;
5. confirmar `winner_promotion_guard.available=true`;
6. atualizar o frontend e promover a Strategy #29.

Branch: `feature/v10.8.87-stale-refresh-lock-recovery`.


## 2026-10-07 - API v10.8.86 U67 protected live runtime

Branch ativa: `feature/v10.8.86-tcc-u67-live-runtime`.

Motivo:

A Strategy #29 U67 foi instalada e o backtest de producao foi concluido com o
resultado homologado, mas a promocao para Winner permanecia bloqueada pelo
gate `research_reference_engine_not_live`. A causa era intencional na
v10.8.85: o U67 possuia apenas runtime protegido de backtest.

Precedente historico:

O mesmo problema apareceu na linha TCC v1.0.6 e foi tratado na branch
`feature/v10.8.62-tcc-control-operational-runtime` / PR #31, criando um
runtime live separado do engine cientifico congelado e permitindo Trader
somente quando o contrato completo fosse compativel.

Implementacao v10.8.86:

- `COMPOUND_ROTATION_SWING_TCC_U67_V1210` deixa de ser Research-only;
- o backtest U67 continua usando o mesmo engine cientifico v1.21.0;
- novo `build_live_tcc_u67_decision()` prepara uma decisao prospectiva de
  next-open apenas com barras concluidas;
- cada preparacao live baixa novamente todo o historico Alpaca RAW/SIP,
  Corporate Actions e aplica a mesma normalizacao de splits e exclusoes do
  backtest homologado;
- o live runtime usa o mesmo calendario U56 elegivel, LightGBM, calibracao,
  minimum holding, cash threshold, expected edge e switch margin;
- a compatibilidade para Winner exige o binding
  `tcc_u67_v1210_operational_backtest`, configuracao U67 exata e snapshot
  LightGBM exato;
- qualquer drift de contrato bloqueia a promocao;
- o endpoint de instalacao U67 passa a marcar o perfil como
  `protected_live_runtime` e `live_trader_eligible=true`.

Preservacao da posicao operacional:

A promocao para Winner continua sem interacao com a corretora e sem
reinicializar Paper State. Os campos existentes permanecem:

- `broker_interaction_performed=false`;
- `operational_state_preserved=true`;
- `paper_state_reinitialization_required=false`;
- `current_position_preserved=true`.

No estado atual de producao, a conta/Mongo esta comprada em XSD. Ao promover a
Strategy #29, XSD deve permanecer intacto. Na primeira avaliacao U67, o runtime
recebe `current_asset=XSD` e o `holding_sessions` existente. XSD so pode ser
vendido por uma decisao normal da politica U67 para outro ativo ou CASH.

Auditoria de dados live:

Cada decisao U67 cria snapshot imutavel em
`dados/tcc_u67_operational/snapshots/trader-u67-<session>-<id>/` e registra no
Paper plan o cutoff, snapshot ID, SHA-256, ativos efetivos e exclusoes
estruturais.

Protecoes mantidas:

- promocao bloqueada durante execucao de ordens;
- promocao bloqueada durante calibracao/refresh critico;
- reconciliacao obrigatoria Mongo x Alpaca antes de preparar plano;
- nenhuma ordem e enviada pelo ato de promocao;
- nenhum parametro de backtest foi alterado para habilitar live;
- a main permanece intocada ate merge explicito.


## Fechamento validado 2026-10-07 - Strategy #13 reproduzida no TCC

A Strategy #13 do MCT foi reproduzida independentemente dentro do TCC com
paridade exata.

Referencia auditada:

- Strategy #13, revisao 2;
- job: `20261007T095423-60e489c0`;
- branch MCT: `feature/v10.8.85-tcc-u67-control-operational`;
- API: `10.8.85`;
- modo: `COMPOUND_ROTATION_SWING_TCC_U67_V1210`;
- 67 ativos configurados na Strategy;
- 65 ativos efetivamente entregues ao modelo;
- exclusoes de runtime observadas: DOC e CLMT;
- Alpaca 1Day, SIP, RAW, full refresh;
- capital em 2026-09-17: US$ 78.782.538,31270888;
- capital em 2026-10-06: US$ 76.927.051,38897176;
- CAGR: 322,7752607%;
- Sharpe: 2,56904268;
- MaxDD: -30,3589700%;
- pior fold: +282,5895543%.

A validacao independente foi executada na branch TCC
`research/tcc-mct-u67-parity-v1`.

Evidencias de paridade:

- 65/65 hashes RAW do OHLCV coincidiram;
- 65/65 hashes normalizados coincidiram;
- 1.560 linhas de decisoes/predictions coincidiram;
- 674 trades coincidiram;
- capital em 2026-09-17 coincidiu ate o centavo;
- capital em 2026-10-06 coincidiu ate o centavo;
- nenhum ativo, score, margem ou trade precisou ser forçado.

Margens efetivas reproduzidas pela calibracao:

- fold 1: 0,0005;
- fold 2: 0,0100;
- fold 3: 0,0005.

O engine cientifico usado pelo MCT permanece vendorizado a partir do TCC e a
auditoria anterior confirmou igualdade byte a byte dos modulos centrais de
rotacao, LightGBM, execucao e configuracao.

A divergencia observada em tentativas intermediarias de reproducao no TCC nao
veio de logica escondida no MCT. Ela foi causada pela ida e volta por CSV no
TCC: valores `float64` gravados com `%.17g` e relidos pelo parser padrao do
pandas podiam voltar com diferenca de 1 ULP. Usando
`float_precision="round_trip"`, o TCC recuperou os mesmos valores usados pelo
MCT e reproduziu integralmente hashes, calibracao, decisoes, trades e capital.

Conclusao operacional da auditoria:

> O backtest de aproximadamente US$ 78,78 milhoes em 2026-09-17 e
> US$ 76,93 milhoes em 2026-10-06 e matematicamente consistente e
> reproduzivel. Nao foi encontrada logica que force as rotacoes para produzir
> esse capital.

Importante sobre o universo:

A Strategy #13 continua configurada nominalmente com os 67 ativos U67. O
runtime desse backtest operou com 65 porque DOC e CLMT foram filtrados antes do
modelo. A exclusao de CLMT reproduz o comportamento do job auditado, mas CUSIP
isoladamente nao deve ser tratado como criterio cientifico geral para remover
ativos.

Estado para integracao:

- branch: `feature/v10.8.85-tcc-u67-control-operational`;
- PR: #34;
- resultado de backtest: validado;
- destino: `main` do MCT;
- tag de fechamento recomendada: `v10.8.85`.

A integracao desta branch na `main` nao equivale, por si so, a promover a
Strategy #13 para Winner nem a habilita para envio de ordens. O modo continua
com as protecoes de Research/backtest e os gates de live/shadow devem ser
tratados separadamente antes de qualquer promocao operacional.


## 2026-10-07 — TCC main U67 -> MCT API v10.8.85

### Objetivo

Migrar para o Market Cycle Trader a estratégia Control oficial presente na
`main` do TCC, sem alterar a `main` do TCC, a `main` do MCT, o Winner,
a carteira ou qualquer caminho de ordens.

### Fonte científica congelada

- repositório: `betovlima/tcc_mba_usp_data_science_analytics`
- branch: `main`
- commit: `4b5f16030afa8b850790747bb0e3e3063d233e79`
- reprodução: `1.21.0`
- schema: `u67-control-reproduction-v1`
- checkpoint: US$ 58.557.157,67496595
- modelo: LightGBM Control
- universo solicitado: U67

### Branch MCT

- base: `main` do MCT API
- base SHA: `b31f95dd9d641ef88542badfc90dd2360bbdaa5b`
- branch: `feature/v10.8.85-tcc-u67-control-operational`
- PR: #34 (draft)
- API: `10.8.85`

### Precedente

A migração segue o padrão já experimentado no PR #31,
`feature/v10.8.62-tcc-control-operational-runtime`, que vendorizou o engine
científico e manteve a adaptação operacional separada. O PR #31 não foi
mesclado na produção.

### Implementado

1. Engine científico da main do TCC copiado para
   `src/market_cycle_trader_api/tcc_u67_v1210_reference/`.
2. Hashes dos blobs e commit científico registrados em `contract.py`.
3. Modo MCT:
   `COMPOUND_ROTATION_SWING_TCC_U67_V1210`.
4. Modo marcado como Research-only.
5. Runtime protegido de backtest U67 conectado ao motor padrão do MCT.
6. Dados do backtest U67 são sempre baixados novamente da Alpaca:
   RAW diário, feed SIP, Corporate Actions e normalização causal de splits.
7. Snapshot imutável por job salvo em
   `dados/tcc_u67_operational/snapshots/<job_id>`.
8. Quebras estruturais de identidade são excluídas. CLMT é excluído
   explicitamente conforme a decisão científica já registrada no TCC.
9. Strategy dedicada pode ser criada/selecionada apenas para Research.
10. Backtest dedicado usa `certify_strategy=False`; não cria Candidate,
    não promove Winner e não envia ordens.

### Endpoints

- `GET /api/research/tcc-u67`
- `POST /api/research/tcc-u67/strategy`
- `POST /api/research/tcc-u67/jobs`

Após iniciar o job, acompanhar pelos endpoints normais:

- `GET /api/jobs/{job_id}`
- `GET /api/jobs/{job_id}/results`

### Regra do checkpoint

US$ 58,56 milhões é uma referência científica histórica e não um alvo de
otimização no MCT. Com dados Alpaca atuais e exclusões estruturais, o capital
pode divergir mesmo quando a política está correta.

### Próximo gate

Executar o primeiro backtest operacional U67 com dados atuais e auditar:

- universo efetivamente elegível;
- exclusões estruturais;
- snapshot/hash;
- folds e margens;
- decisões e trades;
- métricas financeiras.

Somente após essa validação será criado o shadow prospectivo. O modo continua
inelegível para o Trader até que a paridade live seja demonstrada.
