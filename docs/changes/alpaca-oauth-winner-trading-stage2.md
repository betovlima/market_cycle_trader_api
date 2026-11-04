# Etapa 2 do MCT: consentimento OAuth de negociação e pré-validação da Winner

**Data:** 03/11/2026  
**Repositório:** `betovlima/market_cycle_trader_api`  
**Branch:** `feature/alpaca-oauth-winner-trading-v1` · [PR draft #90](https://github.com/betovlima/market_cycle_trader_api/pull/90)  
**Origem:** `main` da API `v10.8.133`  
**Frontend associado:** `feature/alpaca-oauth-winner-trading-v1`, derivada de `v10.7.50`  
**Estado:** implementação experimental **sem execução de ordens OAuth**, não publicada na `main`; ainda sem tag.

## Objetivo e diferença entre permissões

A etapa 1 vinculou a conta Alpaca ao administrador autenticado no MCT, validou `account.id` e permitiu a leitura do Portfolio Live pelo token OAuth.

A etapa 2 começa a preparar a Winner para atuar em uma conta autorizada, mas o OAuth para negociação é uma **autorização diferente**. O escopo `trading` permite, na Alpaca, submeter, cancelar ou modificar ordens. Ele não é solicitado automaticamente pelo login, conexão somente leitura ou verificação da carteira.

Por padrão, **`ALPACA_OAUTH_TRADING_CONSENT_ENABLED` fica desativada**. Definir `1` na API habilita **somente o botão e a autorização para solicitar o escopo `trading`**, não inicia operações. Antes de testar o consentimento, conferir aprovação e condições do aplicativo Alpaca Connect.

## Arquitetura aprovada: Winner compartilhada e conta Alpaca individual

A regra desta etapa é **uma única estratégia Winner**, com um único catálogo de parâmetros, modelo e política de decisão para todas as contas. Não criar nem copiar uma Winner, configuração ou treinamento por usuário.

| Responsabilidade | Escopo / origem |
| --- | --- |
| Seleção da Winner e parametrização | **Global no MCT**, uma única fonte |
| Dados e preparação da análise diária | **Compartilhados**, não por conta OAuth |
| Sinal/alvo diário para a rotação | **Compartilhado**, após separar corretamente as dependências de posição no motor atual |
| Autorização OAuth e `account.id` | **Por usuário Alpaca** |
| Saldo, poder de compra, equity, posições e preços | **Alpaca autenticada daquela conta**, consultada em tempo real |
| Tradução do alvo Winner para `buy`, `hold`, `rotate`, `sell_to_cash` ou `stay_in_cash` | **Por conta**, usando a posição efetiva na Alpaca |
| Tamanho das ordens | **Por conta**, a partir do caixa/poder de compra informado pela Alpaca, conforme executor e política já existentes |
| Ordens, execução, confirmações e reconciliação | **Por conta Alpaca**, com identidade e histórico de execução separados |
| Dashboard Live | **Somente a carteira do usuário autenticado**, nunca patrimônio agregado no MCT |

**O MCT não é uma carteira financeira nem a fonte de verdade do patrimônio.** Não será criado saldo/capital interno para financiar operações. A conta Alpaca decide quais recursos estão disponíveis. Os dados monetários exibidos no frontend virão da Alpaca do usuário. O código legado `live_portfolio.py::_store_observation` já guarda **amostras históricas** do patrimônio e caixa para o gráfico e a auditoria, filtradas por `account_id`; esses registros não são um saldo mantido pelo MCT nem podem substituir uma consulta à corretora. Não remover ou reinterpretar esse histórico silenciosamente.

**A mesma estratégia não implica a mesma ordem.** Se o alvo comum for CAPR: conta em CASH compra conforme seu saldo; conta que já possui CAPR mantém; conta em MAN vende/rotaciona conforme sua posição. `account_action_for_shared_target()` modela essa tradução sem saldos, token ou escrita no Mongo.

**Dependência técnica identificada:** hoje `build_live_model_decision()` recebe `current_asset` e `holding_sessions` e sua saída pode depender da posição da conta que iniciou o cálculo. **Ainda não existe um sinal diário verdadeiramente comum entre contas.** Antes de distribuir decisões é preciso isolar avaliação/modelo compartilhado e regra de escolha do alvo sem alterar silenciosamente a estratégia Winner. O resultado dessa separação deve ser validado contra o comportamento atual.

**Sem travas operacionais extras:** manter os controles existentes, sem introduzir flags adicionais de bloqueio para as ordens. O parâmetro existente `ALPACA_OAUTH_TRADING_CONSENT_ENABLED` serve apenas para expor a solicitação de consentimento OAuth enquanto o piloto está em desenvolvimento, não controla o robô. Bancos MongoDB local e produção são independentes; Winner local permanece parada e produção ativa.

## O que foi implementado

1. `POST /api/alpaca-oauth/start-trading` exige administrador autenticado, Origin do frontend autorizado, flag `ALPACA_OAUTH_TRADING_CONSENT_ENABLED=1` e vínculo prévio da mesma conta Live. Cria um `state` aleatório de uso único (SHA-256 no Mongo, dez minutos, associado a `subject` e `requested_scope=trading`) e envia o navegador ao consentimento da Alpaca com **`scope=trading` e `env=live`**. Não transporta Client Secret na URL.
2. O callback distingue solicitações normais `read` de solicitações `trading` pelo `state` persistido, não por parâmetros livres do navegador. Se o escopo `trading` não vier na resposta à autorização de negociação, o vínculo não é atualizado. Uma autorização `read` que retorna `trading` é rejeitada.
3. No sucesso da autorização, a API valida `account.id` contra a conta vinculada e a conta Live esperada, cifra o token com Fernet e grava `trading_authorized=True`, `oauth_scope=trading` e `read_only=False` **somente como capacidade do token**. O estado `trading_enabled=False` significa que a Winner **não está habilitada para executar** por OAuth.
4. `GET /api/alpaca-oauth/status` apresenta `trading_authorized` e `trading_consent_available`, sem revelar token, segredo ou ID completo. A consulta normal de Portfolio Live continua funcionando mesmo com um token que contenha `trading`.
5. `connected_trading_client(db, identity)` resolve **apenas** uma autorização explícita do usuário atual e confere a identidade da conta na corretora. Não possui fallback para API keys.
6. `GET /api/alpaca-oauth/winner/readiness` usa esse cliente para **ler** conta e relógio Alpaca e confere a **única Winner global** pelo cadastro do MCT. Não procura um sinal Winner privado para cada usuário; o cálculo compartilhado ainda precisa ser extraído do motor existente. Responde `shared_winner=true`, `shared_signal_prepared=false`, `order_submission_enabled=false` e `oauth_winner_execution_enabled=false`. Não cria plano, altera estado, submete ou cancela ordens.

## Limites encontrados na arquitetura existente

O código atual da Winner Live ainda é single-account:

- `live_control.py` usa o documento global `live_trader_control._id="default"`.
- `live_observation_scheduler.py` executa apenas um ciclo e usa `create_live_readonly_client()`, ligado às chaves globais.
- `live_winner_session.py` cria sessões a partir do broker global; `live_winner_signal.py` e `live_order_execution.py` ainda têm consultas por `execution_session` sem `account_id` obrigatório em todos os passos.
- `live_order_execution.py` usa `create_live_execution_client()` e seu ledger/locks foram desenhados para a conta Live administrativa. A identidade do usuário OAuth ainda não atravessa toda a preparação e execução.
- O catálogo da Winner e os parâmetros são **globais por definição**. Os controles operacionais atuais permanecem como estão; a identidade da corretora deve ser aplicada apenas aos dados de conta, ordem e reconciliação.

**Por isso, é inseguro apenas substituir o TradingClient do executor pela versão OAuth.** Isso poderia misturar sinais, posição, orçamento, idempotência ou logs de usuários diferentes.

### Requisitos para permitir a execução individual (ainda pendentes)

1. Preservar **uma única Winner global** e construir uma representação verificável do resultado do pregão D−1 que possa ser reutilizada por todas as contas. Investigar a dependência atual de `current_asset` e `holding_sessions` para manter as regras reais da estratégia.
2. Criar um contexto de corretora `(mct_subject, alpaca_account_id, oauth_grant_id)` obtido pelo servidor, nunca por um ID livre do frontend. Associar somente **operações, estado de execução, posição e auditoria da conta** ao identificador Alpaca.
3. Usar um cliente OAuth por contexto, com consultas de saldo, equity, buying power e posições diretamente da Alpaca. Não criar capital inicial, saldo ou caixa contábil próprio no Mongo. A observação histórica legada é apenas telemetria, não a fonte do saldo.
4. Em cada conta, traduzir o **mesmo alvo da Winner** em ação compatível com sua posição atual; dimensionar compras com o saldo/poder de compra daquela Alpaca, mantendo a política de execução já adotada e suas verificações existentes.
5. Separar o processamento do sinal de mercado (compartilhado) do acompanhamento das contas (individual), com idempotência por `account_id+pregão+ordem`, sem recomputar a Winner por usuário nem adicionar novas travas operacionais. Preservar a execução atual via API Keys até a transição segura.
6. Obter aprovação da Alpaca aplicável a operações em contas de terceiros; validar com mocks as contas A/B com saldos e posições distintos, autorização revogada, reconciliação, falhas e repetição do mesmo pregão, sem ordens reais no ambiente local.

## O que NÃO mudou

- O scheduler atual continua utilizando o mesmo broker por API Keys e não seleciona usuários OAuth.
- O endpoint existente `POST /api/live-execution/winner/execute` continua com seu mecanismo original. **Não** é redirecionado ao token OAuth.
- Não existe endpoint de submissão de ordens por OAuth nesta entrega.
- Os parâmetros, decisão, modelos e estratégia financeira da Winner não foram alterados.
- `ALPACA_LIVE_API_KEY_ID` e `ALPACA_LIVE_SECRET_KEY` permanecem inalteradas.
- A etapa 1, que já foi validada, continua sendo o caminho estável de leitura da carteira.

## Configuração e validação da branch

Por padrão a variável abaixo deve estar **ausente** ou diferente de `1`:

```env
ALPACA_OAUTH_TRADING_CONSENT_ENABLED=0
```

Quando `1` apenas no ambiente local de testes, o botão **Autorizar permissão de negociação** pode abrir o consentimento da Alpaca para a conta Live já vinculada. Este teste não é uma ordem e não habilita o robô. Verificar `GET /api/alpaca-oauth/status` e `GET /api/alpaca-oauth/winner/readiness`. Mesmo com `trading_authorized=True`, confirmar que **`trading_enabled=False` e `order_submission_enabled=False`**. Não habilitar produção por essa configuração.

Os testes `tests/test_alpaca_oauth_winner_consent.py` validam flag desligada, autorização explícita, estado vinculado ao administrador, escopo obrigatório, cifragem, a única Winner global e a tradução do mesmo alvo para ações diferentes conforme a posição na Alpaca, sempre sem enviar ordens; os testes anteriores da etapa 1 precisam continuar aprovados. Executar `python -m unittest discover -s tests -p "test_*.py" -v`.

## Rastreabilidade

- **Branch API e frontend:** `feature/alpaca-oauth-winner-trading-v1`.
- **Base:** API `v10.8.133` e frontend `v10.7.50`.
- **Próxima release:** a definir depois da separação do sinal único e das execuções Alpaca por conta, dos testes de isolamento e da revisão. A estratégia Winner permanece única.
- **Tags:** nenhuma nova tag ou merge em `main` nesta fase experimental.

Referências externas: documentação oficial Alpaca *Using OAuth2 and Trading API*, *About Connect API* e *TradingClient (alpaca-py)*.
