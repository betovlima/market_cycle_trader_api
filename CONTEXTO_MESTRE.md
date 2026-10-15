## 2026-10-15 — Preflight independente Winner/Live

Na branch `feature/live-winner-operational-parity-v1`, adicionado POST `/api/live-market/winner/preflight` que registra identidade da Winner, identidade da conta Alpaca Live e próximo horário de mercado na coleção `live_winner_preflight_snapshots`. Nunca lê posições Paper nem envia ordens. Requer estado Live active/exit_only; parado retorna skipped. Teste de isolamento adicionado `tests/test_live_winner_preflight.py`, ainda não executado. Esta etapa não produz decisões Winner; scheduler de trading Live e ordens reais não estão implementados. Main intocada; PR #53.

---

## 2026-10-15 — Paridade operacional Live/Winner: início de refatoração segura

Branch `feature/live-winner-operational-parity-v1`, partindo da `main`. O endpoint de portfólio Live passa a retornar os valores `buying_power` e `non_marginable_buying_power` diretamente da Alpaca (null se ausentes). Testes adicionados em `tests/test_live_buying_power.py`, ainda não executados. A preparação do plano Winner no Paper usa explicitamente o cliente Alpaca Paper e as coleções/estado Paper; portanto, **não foi habilitada ou copiada** para Live. É necessário extrair um gerador de sinal compartilhável, preservar inputs e decisões independentes, implementar idempotência e reconciliação de ordens Live, controles de risco, testes e autorização antes de enviar ordens reais. Nenhum pedido de ordem é acionado nesta branch.

---

## 2026-10-15 — Início da separação operacional Live (observador independente)

Branch `feature/live-execution-readiness-v1`. Novo `live_observation_scheduler.py`, processo de observação independente do Paper, iniciado no lifespan da API e ativado somente se Live estiver em `active` ou `exit_only`. Consulta a carteira Alpaca Live e os metadados da Winner, grava observações na coleção exclusiva `live_trader_session_observations` sem criar planos Paper nem ordens. O status Live informa o scheduler de observação. Testes de isolamento incluídos em `tests/test_live_observation_scheduler.py`, mas **não executados** nesta sessão. O scheduler ainda **não** prepara sinal Winner nem ordens Live, não tem calendário de pós-fechamento/pré-abertura e não equivale à automação Trading Paper. Não depositar dinheiro contando com execução real; os bloqueios reais permanecem ativos. Main intocada.

---

## 2026-10-15 — Verificações adicionais pré-depósito Live

Na branch `feature/live-execution-readiness-v1`, readiness passa a consultar ordens abertas via Alpaca em modo somente leitura; o dry-run rejeita o placeholder `STRING` e valida a existência/negociabilidade do ativo via get_asset. Testes unitários foram atualizados, **não executados** nesta sessão. Nenhuma ordem real é enviada, e as condições de executor, idempotência persistente, riscos completos, paridade Winner e autorização independente continuam pendentes. Não alterar os checks para true sem validação. `main` intocada.

---

## 2026-10-15 — Prévia Live hipotética (dry-run sem ordens)

**Branch:** `feature/live-execution-readiness-v1`. Rota administrativa `POST /api/live-market/execution/dry-run`, parâmetros `target` (ticker ou CASH) e `max_allocation_fraction`. Consulta a carteira real por cliente Live somente leitura e calcula alocação hipotética sem enviar ordens. Não consome automaticamente a previsão Winner nem é um plano executável. `would_submit_order=false`, `ready_for_real_orders=false`, `order_submission_enabled=false` obrigatórios. Checks de sinal, risco e reconciliação permanecem false de propósito. Testes adicionados em `tests/test_live_execution_dry_run.py` (não executados nesta sessão). **Main preservada.** Próximo passo é integrar o sinal congelado da estratégia ao simulador, validar ordens existentes, regras de risco e executar testes/CI antes de pensar em ativação real.

---

## 2026-10-15 — Pré-validação de execução Live (sem ordens reais)

**Branch:** `feature/live-execution-readiness-v1` a partir de `feature/live-paper-isolated-workspaces-v1`. **Main intocada.**

Criado `GET /api/live-market/execution/readiness` (autenticação administrativa herdada do router) para verificar, em modo de leitura, identidade da conta Alpaca Live, bloqueios informados pela corretora e estado separado do controle Live. Resposta deliberadamente `ready_for_real_orders=false` e `order_submission_enabled=false` enquanto não houver executor segregado, idempotência persistente, limites de risco, validação da decisão oficial e ativação independente. Nenhuma ordem pode ser iniciada por essa rota.

**Railway:** segredos exclusivos do serviço API: `ALPACA_LIVE_API_KEY_ID`, `ALPACA_LIVE_SECRET_KEY` e, recomendado, `ALPACA_LIVE_ACCOUNT_ID`. Nunca colocar segredos no frontend ou Git.

**Testes adicionados:** `tests/test_live_execution_readiness.py` cobre conta saudável e falha de credenciais permanecendo bloqueadas. Testes ainda não executados neste ambiente; validar na CI.

**Próximos passos:** revisar executor Paper como referência sem acoplamento; implementar gateway Live com dry-run obrigatório, ordens idempotentes, plano/sessão congelados, limites de exposição/perdas, parada de emergência, reconciliação corretora/Mongo, travas de horário e ativação administrativa dupla. Somente após testes específicos e autorização de risco ativar envio real de ordens.

---

## 2026-10-15 — API v10.8.102 — Botões Live operacionais em modo de monitoramento

**Solicitação:** disponibilizar Iniciar/Pausar/Somente saída/Parar na seção Operação do Trader · Live, sem alterar Trader Paper.

**Causa:** a v10.8.101 retornava `execution_enabled=false` por segurança, e o frontend usava esse mesmo indicador para desabilitar todo o painel Live.

**Correção:** novos estados administrativos em `live_trader_control` com log independente em `live_trader_control_history`. Endpoint autenticado `POST /api/live-market/robot/mode`, confirmação `LIVE_MONITORING_ONLY`, CAS por `revision`. Front v10.7.26 conecta os quatro botões e controla o refresh automático **da tela** Live; Pausado/Parado suspendem polling Live, sem afetar Paper.

**Limites deliberados:** estado `active` = monitoramento apenas, não autorização de negociar; `exit_only` ainda não vende; `scheduler_alive=false`, `order_submission_enabled=false` em todos os modos. Motor de ordens Live não implementado nesta versão.

**Branch:** `feature/live-paper-isolated-workspaces-v1` (mesmos PRs #51 API e #12 front), `main` preservada. **Testes:** `tests/test_live_control_modes.py`, suíte de integração e build frontend; confirmar GitHub CI. Próximo passo é executor Live segregado com testes de riscos e confirmação explícita antes de negociar.

---

## 2026-10-15 — API v10.8.101 — Live/Paper isolados, primeiro marco de leitura

**Branch:** `feature/live-paper-isolated-workspaces-v1` a partir da `main` (v10.8.100). **Situação:** PR para revisão; nenhuma implantação real, nenhum envio de ordens.

**Solicitação:** Portfolio com abas Live e Trading (Paper), mesmas telas e gráficos com fontes e auditorias próprias; Operação do Trader separada em Live/Paper; ocultar a seção visual de log de execução sem eliminar os registros persistentes.

**Causa:** a integração de negociação existente é exclusivamente Paper e utiliza `paper=True` com credenciais de ambiente Paper. Reutilização direta das chaves Live criaria erro de conta e risco de mistura de execução.

**Entrega parcial segura:** API Live somente leitura (`/api/live-market/portfolio`, `/robot/status`, `/portfolio/export.zip`), cliente `paper=False` com credenciais exclusivas `ALPACA_LIVE_*`, autenticação administrativa e histórico em `live_portfolio_snapshots` segregado por conta. Não usa nem altera o estado Paper, a Winner #30 ou ordens reais. A UI expõe os controles Live como inativos com explicação explícita até existir executor seguro. Histórico de ordens Live vem apenas da corretora; não inventar P/L realizado, drawdown ajustado por depósitos ou decisões da Winner.

**Testes:** `tests/test_live_paper_isolation.py` e CI do PR. Configuração Railway deve utilizar variáveis de segredo distintas, sem publicar credenciais.

**Próximo passo:** implementar executor Live com agendador, estados e comandos independentes (active/paused/exit_only/stopped), limites e travas de risco e autenticação reforçada antes de qualquer permissão de ordens; manter `main` intocada até revisão e aprovação.

---

## 2026-10-14 — API v10.8.100 — colisão de data da decisão Paper

**Evidência operacional:** ciclo automático `paper-market-2026-10-15-f6c3dcc8` permaneceu em `preparing`, com 22 tentativas e `E11000 duplicate key` no índice `uq_paper_plan_decision_date`, `decision_date=2026-10-14`. Já havia uma decisão manual executada na própria sessão de 14/10, MAN → CAPR, com a mesma `decision_date`. A decisão automática para a sessão **15/10** usa legitimamente o último pregão fechado, **14/10**, e não podia ser inserida devido à chave única incompleta. Antes disso, a coleta diária ainda estava atrasada em 13/10, mas a falha persistente passou a ser exclusivamente o conflito de índice.

**Branch:** `fix/v10.8.100-paper-decision-session-index`, criada a partir da main API 10.8.99. Sem alteração da estratégia congelada do TCC, política LightGBM, coleta Alpaca, execução de ordens ou estados antigos.

**Correção pontual:** migrar o índice único de `(decision_date)` para `(decision_date, execution_session)`. Criar primeiro o índice novo, depois retirar apenas a definição antiga do índice, sem excluir ou modificar documentos do Mongo. Alterar o `replace_one(..., upsert=True)` para localizar somente o mesmo par data-decisão/sessão-execução. Assim, a decisão manual `(14/10, 14/10)` e a decisão automática `(14/10, 15/10)` podem coexistir, sem sobrescrever um plano executado.

**Testes:** `tests/test_paper_plan_decision_session_key.py` exercita coexistência entre plano manual executado e plano automático preparado, replacement restrito ao par correto e migração idempotente que preserva documentos. CI antes de merge; não foram feitas ordens Alpaca nem atualizações em Mongo produção.

**Validação após deploy:** confirmar versão 10.8.100 e status `prepared` do ciclo 15/10 no `GET /api/paper-market/status`, com plano cuja `execution_session=2026-10-15` e `decision_date=2026-10-14`, sem erro E11000. Reinício/deploy é necessário para aplicar a migração no `ensure_database`; merge na main não reinicia processos. Nenhuma chamada de limpeza de plano antigo resolve esse problema de índice.

---

## 2026-10-14 — API v10.8.98 — histórico SIP atrasado para Winner manual

**Causa confirmada no ambiente operacional:** a reanálise manual
U77 de 14/10/2026 parou em `AMS` (sem barras IEX do pregão),
tentou SIP para todos os ativos, mas a conta Alpaca recusou a
primeira consulta SIP recente de `NVDA`:
`subscription does not permit querying recent SIP data`.
Nenhum plano manual nem ordem foi criado. O agendador para
15/10 permaneceu armado.

**Branch:** `fix/v10.8.98-delayed-sip-intraday`, da main
API v10.8.97. Não altera a pesquisa congelada do TCC.

**Mudança controlada:** preservar preferência IEX; quando um
ativo elegível não tiver barras IEX, tentar SIP fresco no
universo inteiro, como na v10.8.96. **Somente** para a mensagem
específica de assinatura que proíbe SIP recente, repetir
a coleta SIP de **todos os elegíveis** usando o mesmo horário
de corte, `timestamp_da_análise - 20 minutos` (15 min de
restrição + 5 min de margem). Compara cada barra com esse
corte estritamente exclusivo, impedindo vazamento de dados
posteriores. Se ainda faltar qualquer ativo, ou não houver
20 minutos de sessão, falhar fechado sem plano nem ordens.

**Metadados auditáveis:** o documento do plano persiste
`current_session_feed_mode=delayed_sip`,
`current_session_data_delay_minutes=20`,
`current_session_data_cutoff_utc` e a razão do fallback.
O retorno protegido `manual_recovery` e o acompanhamento
`manual_reanalysis` expõem o horário de corte e a defasagem,
sem mascarar a hora real da operação. O frontend
v10.7.24 destaca explicitamente o atraso antes da decisão
manual. O usuário continua obrigado a confirmar separadamente
`Executar plano atual na Alpaca`.

**Restrições:** a barra diária intradiária sintética somente
alimenta a inferência; treinamento/calibração permanecem com
sessões diárias fechadas. O sistema não mistura feeds entre
ativos; não inventa preços; não retira AMS do universo por
ausência de negócios no IEX. Não modifica ordens nem o fluxo
de seleção do Winner.

**Testes:** a suíte cobre retorno real de assinatura recusada,
retry histórico completo, timestamp exclusivo, bloqueio
antes dos 20 minutos e falha de cobertura. CI obrigatório
antes de integrar à main. Nenhuma ordem Alpaca submetida
durante esta implementação.

**Atenção:** essa mitigação altera a atualidade dos sinais
manual intraday (informações até 20 minutos anteriores),
portanto o comportamento precisa ser validado em Paper
Trading com inspeção do plano aprovado. O merge Git não
implanta automaticamente os serviços de produção.

---

## 2026-10-14 — API v10.8.97 — log operacional e status das ações Winner

**Solicitação:** disponibilizar barra de status e um histórico
cronológico na tela para **Reanalisar Winner agora** e
**Executar plano atual na Alpaca**, mantendo registro das
etapas, contagens reais, ativos, horários, erros e resultados.

**Branch:** `fix/v10.8.97-winner-operations-log`, originada na
`main` v10.8.96. **Causa operacional:** v10.8.96 registrava
apenas o progresso mais recente da reanálise
(`manual_reanalysis`), sem histórico cronológico e sem um
status equivalente para a execução manual na Alpaca.

**Mudança:** `paper_market_automation.manual_operation_log`
é um histórico persistido e limitado às últimas 90 entradas
com hora UTC, tipo de operação, etapa real, estado, contagem
de ativos e ticker. Um evento é registrado ao iniciar a
reanálise, na mudança de etapa, a cada dez ativos (ou no
conclusivo), na conclusão ou na falha. Dados sensíveis e
exceções brutas não integram o log público. Para execução
manual, `manual_execution` e os eventos associados distinguem
início, confirmação e falha / revisão obrigatória,
mantendo o caminho de ordens inalterado.

O endpoint de status do robô retorna os campos
`manual_reanalysis`, `manual_execution` e
`manual_operation_log`. A rota pública sanitiza mensagens
de exceção da corretora; administradores ainda podem
consultar o erro detalhado na rota protegida. A renderização
no Portfólio é implementada no frontend v10.7.23; o
componente dos botões do Trader não está identificado no
código da `main` do frontend disponível.

**Testes e escopo:** testar telemetria de execução,
milestones, limitação de log e sanitização dos erros.
Nenhuma ordem enviada à Alpaca; preservados regras de
negociação, treino, universo, permissões, TCC e estado
financeiro. Integrar na `main` somente após CI verde.
A implantação da API e frontend continua separada do merge.

---

## 2026-10-14 — API v10.8.96 — AMS sem barras IEX e progresso da reanálise Winner

**Incidente:** após **Reanalisar Winner agora**, o endpoint
`POST /api/admin/trader-control/manual-recovery/prepare` terminou
com `CurrentSessionIntradayDataMissing: AMS has no IEX
regular-session bars for 2026-10-14`. O plano existente de 13/10,
`lightgbm-2026-10-13-4cdbd908`, não era um plano manual de 14/10
(`manual_current_session_recovery=false`) e a API acertadamente
impediu a execução do botão seguinte.

**Branch:** `fix/v10.8.96-intraday-sip-recovery`.

**Causa:** a preparação U67+10 exigia um agregado intradiário
`1Min` de todos os ativos elegíveis usando o feed IEX configurado.
Na primeira ausência de barras de uma sessão, a operação falhava.
A ausência no IEX não é evidência de problema estrutural do ativo.

**Correção:** se e somente se uma consulta IEX não tiver barras,
refazer a coleta **de todo o universo elegível** em SIP; jamais
misturar dados IEX/SIP entre ativos na mesma decisão. Se SIP não
estiver autorizado pela conta ou não cobrir todos os ativos, falhar
explicitamente, sem inventar barras e sem enviar ordens. A
metainformação da decisão registra `live_feed=sip` e o motivo do
fallback para auditoria. Treinamento e calibração seguem nas
barras diárias concluídas, sem uso do agregado intradiário
incompleto para treinamento.

**Progresso observável:** a preparação manual continua sem
envio de ordens e informa etapas e contagens reais de
validação, coleta histórica, coleta intradiária IEX/SIP,
treinamento e persistência. O documento
`paper_market_automation.manual_reanalysis` permite acompanhar
o andamento e o erro final mesmo se o run automático da sessão
corrente já estiver `failed` e o de amanhã já `armed`.
O endpoint de status do robô repassa esse documento; o frontend
pode exibir as etapas sem inventar percentuais.

**Validação:** testes adicionados para ausência IEX em AMS,
retry SIP uniforme, SIP não autorizado, erro de IEX que não
justifica fallback, feed SIP configurado, progresso ad hoc e
exibição do erro de cobertura. Os testes devem passar no CI
antes do merge à main. Não houve ordens Alpaca nem alteração
da posição Paper no desenvolvimento.

**Nota:** a requisição de preparo permanece síncrona; o
progresso persistido é consultável por outra conexão. Um
workflow assíncrono completo para o botão (com polling no
próprio componente da tela de Trader) exige alterar o código
dessa tela, que não foi localizado no repositório frontend
consultado. Não alegar que o botão foi reabilitado sem
validação da implantação.

---

## 2026-10-14 — API v10.8.95 — continuidade do Trader após recuperação manual

**Incidente:** operação Paper MAN de 07/10/2026 foi deliberadamente
acionada pelo administrador na ordem **Reanalisar Winner agora →
Executar plano atual na Alpaca**. A auditoria de 14/10 evidenciou
ordem manual preenchida, mas ciclos automáticos subsequentes
`prepared`/`cancelled`, com divergência de contador já corrigida
na v10.8.94. O fluxo manual anterior registrava
`manual_recovery_plan_id` e `manual_recovery_executed_at`, mas
não concluía o `paper_market_runs` associado, que podia continuar
apontando para o `plan_id` substituído.

**Branch:** `fix/v10.8.95-manual-paper-scheduler-handoff`
a partir da `main` `a8322516f2220df414a69985943cb6fb3b26c59c`.

**Correção operacional:** a reanálise manual, quando existe run
agendado ativo para a sessão, reivindica o run de forma atômica
antes de executar calibração e registra seu novo `plan_id`,
ação, alvo e status `manual_pending`. A disputa pela recalibração
automática na abertura também passa a exigir reivindicação
condicional do run, sem prosseguir se a etapa manual o tomou.
Enquanto um plano manual espera confirmação, o scheduler não
envia ordens nem tenta executar a referência automática antiga.
O botão de execução manual aceita apenas um plano preparado
pela reanálise Winner na sessão atual e, se existir run ativo,
exige que os identificadores coincidam. Após execução
confirmada, encerra o run como `completed`, registra
a origem manual e o resultado e libera a chave ativa para a
próxima sessão. Erros incertos após reclamar o plano entram
em `review_required`, e um plano manual abandonado é
cancelado quando passa para a sessão seguinte, impedindo que
o robô permaneça indefinidamente preso ao run antigo.

**Calibração:** a ação manual de análise já recalcula o Winner.
O botão de execução NÃO recalibra silenciosamente, preservando
exatamente o plano analisado e aprovado. Nenhuma modificação
no modelo, hiperparâmetros, ranking, universo ou permanência
mínima de dois pregões. Correção compatível com a reconciliação
de fills reais da v10.8.94.

**Testes de regressão:** criação e vinculação do plano manual
ao run, execução única e liberação da chave ativa, rejeição
de plano antigo ou de outra sessão, falha incerta com
bloqueio para revisão, ausência de execução automática
enquanto aguarda aprovação, expiração de plano não executado,
falha de recalibração, preservação do fluxo ad hoc sem run,
e sinalização de elegibilidade no status do robô.

**Limites e validação:** o CI GitHub e o PR devem passar antes
de integrar à `main`. Nenhuma ordem foi enviada à Alpaca
durante o desenvolvimento e nenhuma coleção de produção foi
alterada. A análise histórica de motivos específicos de
`cancelled` em 08–12/10 continua dependente dos registros
de `paper_market_runs`, `paper_trade_plans` e operações
administrativas. Depois do deploy, confirmar API v10.8.95,
a versão de backend, o status e a continuidade dos próximos
pregões. A instância produtiva não é implantada só pelo merge.

---

## 2026-10-14 — API v10.8.94 — correção de permanência Paper por fills reais

**Incidente:** pacote `mct_paper_transaction_audit_20261014T152546Z.zip`
(API v10.8.93, Strategy #30 revisão 2). MAN foi comprado em
07/10/2026, sem nova venda/recompra, mas os planos posteriores
de 09, 12 e 13/10 mantiveram `holding_sessions=1`. O plano de
13/10 exibiu CAPR como melhor bruto (utilidade 0,429718), MAN
0,246669, margem aprovada, porém a decisão reteve MAN pela
restrição de dois pregões. Os quatro pregões *concluídos após
a entrada* até 13/10 são 08, 09, 12 e 13/10; na semântica
da execução Paper a compra vale a primeira sessão, logo o
contador operacional reconciliado passa de 1 para 5.

**Causa no runtime anterior:** `prepare_next_paper_plan` recebia
`state.holding_sessions` diretamente do MongoDB. O contador só
era incrementado em `execute_prepared_paper_plan` ao executar
uma ação de hold. Planos cancelados ou ainda preparados não
executavam esse caminho, congelando o contador mesmo que dias
de negociação tivessem transcorrido. O relatório de auditoria
detectava a discrepância, mas não a corrigia para a decisão.

**Branch:** `fix/v10.8.94-paper-holding-fill-reconciliation`, a
partir da `main` `c70681db91ce4d391ac9fadc0b218bd1e02e5e19`.

**Correção:** antes de gerar qualquer previsão/decisão, o
Paper consulta exclusivamente os fills efetivos registrados no
Mongo, localiza a entrada da posição contínua atual, reconcilia
a quantidade com o estado já conferido contra a Alpaca e conta
as sessões regulares de XNYS *efetivamente concluídas* desde
a compra, sem incluir pregão intradiário ainda aberto. Usa
`1 + sessões completas após o fill` para respeitar o contrato
original de incremento em cada execução (compra inicia em 1).
Corrige o contador persistido com compare-and-set, protegendo
trocas simultâneas de posição. Se não houver fill confirmado,
o calendário faltar ou o livro de ordens divergir, a preparação
falha com erro explícito e não envia ordens. Em caixa não há
ação. Não altera parâmetros, modelo, regra mínima de dois
pregões, ranking, compras/vendas ou contabilidade financeira.

**Testes:** casos de MAN com lacunas de execução, exclusão da
sessão corrente intradiária, data inicial da posição,
reentrada após liquidação, ausência de fill, conflito de
quantidade, atualização concorrente e carteira em caixa.
Os testes precisam ser aprovados no workflow GitHub antes
de levar a correção à `main`. A revisão não executa
ordens Paper nem atualiza diretamente o banco de produção.

**Observação operacional:** a correção evita bloquear decisões
futuras com um contador congelado, mas não explica por que os
planos Paper históricos ficaram `prepared`/`cancelled`.
Verificar logs e status do agendador e janela de execução
em produção. Um novo plano deve ser gerado após o deploy,
e não é seguro executar um plano antigo de CAPR apenas
porque o indicador bruto foi o maior. Não há garantia de
que CAPR passaria todos os critérios de elegibilidade
em um novo pregão.

---

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

 
## API v10.8.99: limpeza simples de planos Paper antigos (2026-10-14)

- Branch `fix/v10.8.99-paper-plan-status-reconciliation` (PR #49; `main` permanece intacta).
- Causa: dois planos históricos de `hold` em MAN ficaram com `status=prepared` para as sessões 13/10 e 14/10, embora houvesse plano manual distinto executado em CAPR em 14/10.
- Única API: `POST /api/admin/trader-control/paper-plans/close-old`. Requer a sessão administrativa existente; não requer body. Faz apenas um `update_many` na coleção MongoDB `paper_trade_plans`.
- Filtro: `status=prepared`, `execution_session <= ultima_sessao_encerrada` conforme relógio de Nova York (após 16h, inclui a data atual; antes, apenas até o dia anterior). A resposta informa `closed_count` e `closed_through_session`.
- Campos alterados: `status=cancelled`, `cancelled_at`, `cancel_reason`, `updated_at`. Sem correlação com ordens e runs antigos, sem preflight, sem consulta Alpaca; intenção é a limpeza de status remanescente de desenvolvimento, sem atribuir execução fictícia.
- O endpoint **não cancela ordens na corretora** e **não reclassifica** o registro de execução de ordens `paper_trade_orders`, pois os casos comprovados como incorretos são planos `prepared`, não ordens preenchidas. Não altera posição, capital, Winner, cotações, agendador, decisão nem regras da estratégia.
- Sem novas travas de negócio ou manutenção automática que interfira nas escolhas futuras; encerramento é apenas sob chamada do administrador e idempotente para planos já cancelados.
- CI: dois testes da chamada única e janela temporal foram adicionados em `tests/test_manual_current_session_control.py`; asserção de versão atualizada em `tests/test_tcc_v106_main_integration.py`.
