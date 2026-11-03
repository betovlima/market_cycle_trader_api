# Etapa 2 do MCT: consentimento OAuth de negociação e pré-validação da Winner

**Data:** 03/11/2026  
**Repositório:** `betovlima/market_cycle_trader_api`  
**Branch:** `feature/alpaca-oauth-winner-trading-v1`  
**Origem:** `main` da API `v10.8.133`  
**Frontend associado:** `feature/alpaca-oauth-winner-trading-v1`, derivada de `v10.7.50`  
**Estado:** implementação experimental **sem execução de ordens OAuth**, não publicada na `main`; ainda sem tag.

## Objetivo e diferença entre permissões

A etapa 1 vinculou a conta Alpaca ao administrador autenticado no MCT, validou `account.id` e permitiu a leitura do Portfolio Live pelo token OAuth.

A etapa 2 começa a preparar a Winner para atuar em uma conta autorizada, mas o OAuth para negociação é uma **autorização diferente**. O escopo `trading` permite, na Alpaca, submeter, cancelar ou modificar ordens. Ele não é solicitado automaticamente pelo login, conexão somente leitura ou verificação da carteira.

Por padrão, **`ALPACA_OAUTH_TRADING_CONSENT_ENABLED` fica desativada**. Definir `1` na API habilita **somente o botão e a autorização para solicitar o escopo `trading`**, não inicia operações. Antes de testar o consentimento, conferir aprovação e condições do aplicativo Alpaca Connect.

## O que foi implementado

1. `POST /api/alpaca-oauth/start-trading` exige administrador autenticado, Origin do frontend autorizado, flag `ALPACA_OAUTH_TRADING_CONSENT_ENABLED=1` e vínculo prévio da mesma conta Live. Cria um `state` aleatório de uso único (SHA-256 no Mongo, dez minutos, associado a `subject` e `requested_scope=trading`) e envia o navegador ao consentimento da Alpaca com **`scope=trading` e `env=live`**. Não transporta Client Secret na URL.
2. O callback distingue solicitações normais `read` de solicitações `trading` pelo `state` persistido, não por parâmetros livres do navegador. Se o escopo `trading` não vier na resposta à autorização de negociação, o vínculo não é atualizado. Uma autorização `read` que retorna `trading` é rejeitada.
3. No sucesso da autorização, a API valida `account.id` contra a conta vinculada e a conta Live esperada, cifra o token com Fernet e grava `trading_authorized=True`, `oauth_scope=trading` e `read_only=False` **somente como capacidade do token**. O estado `trading_enabled=False` significa que a Winner **não está habilitada para executar** por OAuth.
4. `GET /api/alpaca-oauth/status` apresenta `trading_authorized` e `trading_consent_available`, sem revelar token, segredo ou ID completo. A consulta normal de Portfolio Live continua funcionando mesmo com um token que contenha `trading`.
5. `connected_trading_client(db, identity)` resolve **apenas** uma autorização explícita do usuário atual e confere a identidade da conta na corretora. Não possui fallback para API keys.
6. `GET /api/alpaca-oauth/winner/readiness` usa esse cliente para **ler** conta e relógio Alpaca, consulta sessão Winner por `_id=f"{account_id}:{execution_session}"` e verifica correspondência com a estratégia Winner selecionada. Não cria plano, altera estado, submete ou cancela ordens. O retorno declara `order_submission_enabled=False` e `oauth_winner_execution_enabled=False`, mesmo com consentimento concedido.

## Limites encontrados na arquitetura existente

O código atual da Winner Live ainda é single-account:

- `live_control.py` usa o documento global `live_trader_control._id="default"`.
- `live_observation_scheduler.py` executa apenas um ciclo e usa `create_live_readonly_client()`, ligado às chaves globais.
- `live_winner_session.py` cria sessões a partir do broker global; `live_winner_signal.py` e `live_order_execution.py` ainda têm consultas por `execution_session` sem `account_id` obrigatório em todos os passos.
- `live_order_execution.py` usa `create_live_execution_client()` e seu ledger/locks foram desenhados para a conta Live administrativa. A identidade do usuário OAuth ainda não atravessa toda a preparação e execução.
- A Winner e os controles de risco/ativação são globais, e não configurados por `subject+account_id`.

**Por isso, é inseguro apenas substituir o TradingClient do executor pela versão OAuth.** Isso poderia misturar sinais, posição, orçamento, idempotência ou logs de usuários diferentes.

### Requisitos para permitir a execução individual (ainda pendentes)

1. Criar um contexto de execução imutável `(mct_subject, alpaca_account_id, oauth_grant_id, session)`, obtido sempre no servidor a partir da identidade autenticada, nunca de um account_id enviado pelo navegador.
2. Refatorar todas as consultas/índices de sessões, decisões, sinais, estado, controles, ordens e logs da Winner para serem estritamente qualificadas por usuário/conta. Revisar migração de dados administrativos legados e unicidade da conta vinculada.
3. Introduzir um broker provider por contexto: OAuth verificado em toda execução, checagem do escopo `trading`, da permissão da Alpaca, do relógio, bloqueio de conta e reconciliação de posições/ordens. Nada de fallback para credenciais de outra conta.
4. Tornar o scheduler, locks distribuídos, idempotência e recuperação individualizados por conta, preservando janela XNYS, sinal imutável D-1, validações de segurança, stop/pause por conta e proteção contra duplicidade.
5. Exigir **opt-in operacional adicional** do titular após o consentimento OAuth. Obter aprovação Alpaca para negociar em contas de outros usuários e realizar testes completos em Paper e em ambiente isolado, antes de qualquer Live.
6. Acrescentar testes de concorrência e isolamento entre dois usuários, troca/revogação de token, desconexão, mudança de posição, interrupção durante sell→buy, reprocessamento do mesmo pregão e falhas do broker.

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

Os testes `tests/test_alpaca_oauth_winner_consent.py` validam flag desligada, autorização explícita, estado vinculado ao administrador, escopo obrigatório, cifragem e ausência de ordens; os testes anteriores da etapa 1 precisam continuar aprovados. Executar `python -m unittest discover -s tests -p "test_*.py" -v`.

## Rastreabilidade

- **Branch API e frontend:** `feature/alpaca-oauth-winner-trading-v1`.
- **Base:** API `v10.8.133` e frontend `v10.7.50`.
- **Próxima release:** a definir depois da conclusão da refatoração de execução individual, dos testes de isolamento e da revisão.
- **Tags:** nenhuma nova tag ou merge em `main` nesta fase experimental.

Referências externas: documentação oficial Alpaca *Using OAuth2 and Trading API*, *About Connect API* e *TradingClient (alpaca-py)*.
