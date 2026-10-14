## 2026-10-13 — API v10.8.93 — observabilidade da máxima e devolução de lucro Paper

**Causa:** episódio da MAN (compra 07/10) expôs ausência de curva de
lucro por operação, possíveis máximas intradiárias não capturadas pela
amostragem existente, e divergência entre dias de negociação decorridos
e `holding_sessions`. Os planos de `hold` recentes ainda constavam
como `prepared`; isso demanda investigação de execução, **não**
mudança de política automática.

**Branch:** `feature/v10.8.93-position-excursion-observability`;
**Front correspondente:** `feature/v10.7.21-position-excursion-observability`.

**Implementado:** captura da posição Paper a cada cinco minutos durante
mercado aberto pelo próprio scheduler, sem depender do frontend;
amostras persistem cotação, quantidade, custo e ganho não realizado.
Uma auditoria por fill calcula MFE/MAE observados, instante do pico,
giveback do lucro da posição e do patrimônio, limitações de
amostragem e comparação do contador persistido com sessões do
calendário da Alpaca. Novos CSVs no ZIP de auditoria e gráfico no
Portfolio. Nenhum código de envio de ordens, política LightGBM,
Winner ou mínimo de dois pregões foi alterado. Falha de coleta não
impede a operação. Não adicionamos hash, gate de desempenho ou venda
automática.

**Evidência inicial (snapshot, não máximo real de bolsa):**
patrimônio durante MAN máximo observado US$ 10.733,61,
último US$ 10.159,16, diferença US$ 574,45;
pico de rentabilidade observado da MAN +2,52%, portanto
a afirmação de +20% não está comprovada pelos ZIPs
e não deve ser apresentada como resultado calculado.
O contador mostrado foi 1; calendário desde o fill
08/10, 09/10, 12/10 e 13/10: quatro pregões
decorridos na data do relatório.

**Próximos passos:** CI e coleta prospectiva em Paper,
exportar novo pacote e validar horários/quantidade/estado.
Investigar separadamente os planos hold pendentes que
podem impedir atualização do contador. Somente mediante
nova solicitação considerar mudança de regras de venda.
Documentação: `docs/changes/v10.8.93-position-excursion-observability.md`.

---

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

## 2026-10-07 — API v10.8.89 — recuperação manual U67 com sessão corrente intraday

### Causa
A v10.8.88 liberou a reanálise manual durante o pregão, porém o plano ainda era
calculado exclusivamente com candles diários concluídos. Assim, em 07/10/2026,
a tela preparou uma decisão com `decision_date=2026-10-06` (XSD -> MAN), embora
o objetivo do botão seja recalcular o Winner usando também o movimento já
observado na sessão aberta de 07/10 antes de permitir execução manual.

### Alteração
- Mantidos os endpoints existentes de manual recovery.
- O U67 baixa o snapshot histórico diário completo até a última sessão segura.
- Durante mercado aberto, agrega barras Alpaca live de 1 minuto da sessão
  corrente em um candle diário parcial.
- Esse candle parcial entra apenas como observação prospectiva de scoring.
- Treino/calibração continuam protegidos pelo purge U67 e por sessões históricas.
- O plano manual passa a usar a própria sessão corrente como `decision_date` e
  `execution_session`.
- O plano registra provenance intraday: horário da análise, cutoff histórico,
  feed, timeframe e timestamps das barras atuais.
- Falha fechada se um ativo elegível não tiver barra intraday atual.
- A reanálise continua sem enviar ordens. Somente o endpoint explícito de
  execução pode operar na Alpaca Paper.

### Versão e branch
- API `10.8.89`
- `feature/v10.8.89-current-session-intraday-recovery`

### Impacto
O botão **Reanalyze Winner for today** passa a representar o mercado observado
até o momento da sessão aberta, em vez de apenas repetir a decisão do fechamento
anterior. O botão **Retry plan on Alpaca** continua restrito ao plano
`prepared` e ao mercado regular aberto.

### Testes
- agregação OHLCV/VWAP/trade_count da sessão corrente;
- uso do feed live IEX e timeframe de 1 minuto;
- falha fechada sem barras da sessão;
- presença obrigatória do caminho intraday no runtime Paper U67.

### Próximos passos
Validar em Paper que uma reanálise durante a sessão produz
`analysis_mode=current_session_intraday`, `decision_date` igual à sessão
corrente e, após confirmação explícita, reconcilia corretamente a rotação na
Alpaca.

## 2026-10-09 — API v10.8.90 — auditoria operacional integrada ao Portfolio

### Objetivo
Exibir na própria tela do Portfolio a auditoria das operações Paper e a
explicação das decisões do dia, sem depender da exportação ZIP para análise
manual.

### Escopo
- somente leitura;
- nenhuma alteração no caminho de execução de ordens;
- nenhuma alteração em Winner, modelo, parâmetros ou política operacional;
- nenhuma alteração no scheduler.

### Dados adicionados ao Portfolio
- capital atual, pico, distância ao pico e maximum drawdown;
- histórico enriquecido com peak e drawdown;
- P/L realizado e não realizado por ativo;
- operações com filled value e realized P/L;
- partial fill com filled_quantity > 0 tratado como execução econômica mesmo
  quando o status final for canceled;
- decisão mais recente com ranking completo;
- utility atual, utility alvo e utility do líder bruto;
- deltas contra o ativo atual e o líder;
- switch margin e indicação de passagem;
- holding sessions e mínimo exigido;
- motivo de seleção/hold;
- provenance de execução.

### Novos campos persistidos
Planos criados a partir desta versão registram:
- holding_sessions_at_decision;
- minimum_holding_sessions.

### Observação operacional
A rotação XSD -> MAN executada por manual recovery durante a implantação da
versão intraday é comportamento conhecido e esperado da fase de testes/
implantação. O dashboard preserva essa provenance para auditoria, sem tratá-la
como decisão manual do modelo.

### Versão
API 10.8.90

### Branch
feature/v10.8.90-portfolio-audit-dashboard



## 2026-10-12 — API v10.8.91 — Strategy Challenger U67+10 estática

### Solicitação e decisão de escopo

O autor encerrou a etapa de descoberta do TCC e decidiu transferir para o MCT
a composição estática que atingiu US$ 587.165.941,1330212 no cutoff congelado
2026-10-06, mantendo separadas pesquisa e operação real.

Tag de congelamento do TCC:
`research-u67plus10-challenger-v1.4.1`, commit científico
`e5d0c4cc2ae92fcb994cc0006f357d775b486721`.
A tag ainda precisa ser criada/publicada pelo operador com `git tag -a` e
`git push`; não afirmar que ela já existe sem verificar o remoto.

Universo: U67 existente (mesma ordem e mesmos dados estruturais) + CRT, NBR,
CAPR, PNBK, PENN, JBL, STAG, MCHB, LAD, TOWN. São 77 solicitados e 75
efetivos na janela histórica. O TCC alerta sobre o PBO da otimização
(53,57% log-growth, 59,52% Sharpe) e ausência de validação nested completa.

### Implementação operacional

Branch: `feature/v10.8.91-u77-fixed-challenger`
API: `10.8.91`

O contrato do U67 atual bloqueia alteração da lista de 67 ativos. Portanto
foi criado um NOVO modo, contrato, binding, instalador idempotente e endpoint
de Backtest. Ele usa Alpaca RAW/SIP atualizado e exclui identidades
estruturalmente inválidas, sem reconstrução histórica manual.

```text
mode: COMPOUND_ROTATION_SWING_TCC_U77_FIXED_V141
binding: tcc_u77_fixed_challenger_backtest
GET  /api/research/tcc-u77-fixed
POST /api/research/tcc-u77-fixed/strategy
POST /api/research/tcc-u77-fixed/jobs
```

### Preservação do Winner e do estado de execução

Esta versão é **backtest-only**. O instalador seleciona somente Research,
não toca Winner, ordem Alpaca, posição corrente ou Paper State.
A promoção permanece bloqueada por ausência de runtime live U77 homologado.

A intenção do autor é executar primeiro o Backtest na infraestrutura MCT
com dados atualizados, revisar a auditoria e depois decidir se promove. O
resultado do TCC é referência histórica, não projeção garantida e não deve
ser usado para automatizar promoção.

A próxima entrega, separada e versionada, precisa implementar o runtime
prospectivo de 77 ativos com paridade entre backtest e Trader; validar
reconciliação Mongo/Alpaca, XNYS fechado para promoção, posição atual,
holding_sessions e regras de exclusão; só então habilitar Winner.
Não fazer bypass dos gates existentes.

Mais detalhes: `docs/changes/v10.8.91-u77-fixed-challenger.md`.


### 2026-10-12 — Correção de escopo do U77 antes da integração

O responsável pelo projeto recusou as condições adicionais impostas na
primeira proposta: `backtest-only`, `RESEARCH_ONLY`, veto artificial de
promoção e obrigação de estudos adicionais. O fluxo solicitado é o ciclo
normal da Strategy no MCT: criar, backtest, selecionar Candidate e promover
Winner pelo catálogo quando decidir.

A branch `feature/v10.8.91-u77-fixed-challenger` passa a usar o runtime
prospectivo do mesmo LightGBM Control com 77 ativos solicitados e contratos
específicos U77, conservando o U67 histórico. O modo U77 foi excluído da
categoria de estratégias apenas de pesquisa, a checagem de compatibilidade
com o Trader foi estendida, e a instalação do catálogo não cria mais campos
de impedimento artificial. O endpoint de backtest utiliza a certificação
normal e não promove automaticamente o Winner.

As verificações gerais e existentes de integridade dos dados, ordens e posições
não foram removidas. Esta correção também não executa operações nem altera
a Strategy Winner atual. O estado dos testes está registrado no GitHub Actions
da branch e deve ser confirmado no respectivo commit.

Documentação detalhada: `docs/changes/v10.8.91-u77-fixed-challenger.md`.


## 2026-10-12 — API v10.8.92 — Busca nativa de endpoints em /docs

Solicitação: permitir a localização rápida de endpoints no Swagger, especialmente
os de cadastro e backtest do U67+10, sem navegar por dezenas de grupos.

Implementação exclusivamente de interface de documentação:
`swagger_ui_parameters` na aplicação FastAPI com
`filter=True`, `docExpansion="none"`, `tagsSorter="alpha"`,
`operationsSorter="alpha"` e `deepLinking=True`.
O filtro nativo busca por nome de tag; não é busca de texto completo por
caminho ou descrição. Exemplos: `tcc-u77-fixed-challenger` e `jobs`.

Versão API `10.8.92`, branch
`feature/v10.8.92-swagger-endpoint-search`. Nenhuma alteração no catálogo,
rotas de execução, auth, Backtest, Candidate, Winner, modelos, dados Alpaca ou
MongoDB. A `main` permanece como referência até eventual merge.
Detalhes em `docs/changes/v10.8.92-swagger-endpoint-search.md`.


## 2026-10-12 — Correção intradiária AMS / API v10.8.93

### Evento observado

Em produção, ao solicitar a análise da sessão corrente XNYS de 2026-10-12,
o Trader reportou:
`CurrentSessionIntradayDataMissing: AMS has no IEX regular-session bars for 2026-10-12.`

O ZIP `market_cycle_trader_20261012T170547-7a034113.zip` submetido
simultaneamente é do **backtest histórico concluído**, Strategy #30,
`COMPOUND_ROTATION_SWING_TCC_U77_FIXED_V141`, 77 ativos solicitados
e 75 elegíveis, capital final US$ 605.911.159,62. O ZIP registra
`trader_winner_strategy_name_at_queue="Strategy #29"`, mas não
contém o traceback da tentativa live nem confirma que o Winner
efetivamente usado no momento da exceção era o U77.

A causa de código foi identificada em
`engine/tcc_u67_operational_market_data.py`: a função
`append_u67_current_session_intraday` exigia ao menos uma barra
1Min IEX da sessão atual para **todos** os ativos e falhava
inteiramente ao encontrar símbolo sem negociação na IEX.
A IEX não representa todo o mercado consolidado e ativos pouco
líquidos podem não negociar nessa bolsa durante a janela solicitada.

### Versão de correção

Branch `fix/v10.8.93-intraday-iex-sip-fallback`; API 10.8.93.

O loader mantém a IEX como prioridade e, se o ativo não tem barra
IEX hoje, busca barras SIP consolidadas RAW 1Min até
`timestamp_atual - 20 minutos`, respeitando planos SIP atrasados.
Se SIP fornecer barras, elas compõem o OHLCV **parcial observável** e
a fonte SIP fica registrada em metadata.

Se nenhuma fonte retorna barras, ou SIP não está disponível por
permissão, o ativo não detido é omitido **somente na decisão corrente**,
com `intraday_missing_assets` e `intraday_missing_reasons`.
Isso não remove permanentemente AMS, não altera os 77 ativos da
configuração, não inventa preços nem toca nos CSVs do TCC.

A função recebe `current_asset=state.managed_symbol` e distingue a
posição efetivamente detida: se faltar cotação para o ativo detido,
a ausência é reportada explicitamente em vez de gerar uma nova ordem
com uma observação sintética ou obsoleta. Se não houver cotação
para nenhum ativo, a execução também informa essa indisponibilidade.

Metadados de auditoria repassados no retorno do Trader:
`intraday_fallback_assets`, `intraday_missing_assets`,
`intraday_missing_reasons`, `intraday_asset_count` e
`live_feed` (incluindo `mixed_iex_sip`).

Sem alteração de Candidate, Winner, permissões, regras de promoção,
modelos, backtest histórico, universo persistido ou execução de ordens.
Esta é uma correção de disponibilidade da fonte de dados, não a
inclusão de uma trava estratégica.

### Testes e próximos passos

Testes cobrem IEX disponível, IEX vazio + SIP disponível,
ausência em ambos para ativo não detido, ausência em ambos
para ativo detido e indisponibilidade de permissão SIP.
A verificação de dados SIP reais de AMS exige execução no ambiente
com as credenciais Alpaca do operador. A aplicação da correção
em produção exige merge/deploy separado; não efetuar ordens
nem promoção como parte do patch.

Detalhamento em
`docs/changes/v10.8.93-intraday-iex-sip-fallback.md`.
