# Alpaca Connect OAuth: teste inicial para o administrador atual

## Escopo

Esta entrega autoriza **somente leitura** da conta Alpaca **Live** já utilizada
pelo MCT e liga o **Portfolio Live e sua exportação** ao token OAuth
associado à sessão administrativa. Não altera a estratégia Winner, seus
agendadores nem a criação/envio de ordens. A engine continua usando
`ALPACA_LIVE_API_KEY_ID` e `ALPACA_LIVE_SECRET_KEY`.

A conexão OAuth fica no MongoDB, separada e vinculada ao subject autenticado.
Não é uma ativação do recurso multiusuário.

## Pré-requisito na Alpaca

1. Acesse o Dashboard da Alpaca e abra **Alpaca Connect > My Developed Apps**.
2. Cadastre o MCT como aplicativo OAuth e obtenha **Client ID** e **Client Secret**.
3. Informe à Alpaca o **Redirect URI exato** do backend:

   `https://marketcycletraderapi-production.up.railway.app/api/alpaca-oauth/callback`

4. Atenda às exigências de análise/aprovação aplicáveis da Alpaca. Somente após
   o cadastro e emissão das credenciais será possível fazer uma autorização
   real.

Documentação: https://docs.alpaca.markets/us/docs/registering-your-app

## Variáveis na API MCT (Railway ou ambiente local)

```env
ALPACA_OAUTH_CLIENT_ID=CLIENT_ID_DO_APLICATIVO
ALPACA_OAUTH_CLIENT_SECRET=CLIENT_SECRET_DO_APLICATIVO
ALPACA_OAUTH_REDIRECT_URI=https://marketcycletraderapi-production.up.railway.app/api/alpaca-oauth/callback
ALPACA_OAUTH_FRONTEND_URL=https://marketcycletrader-production.up.railway.app/
ALPACA_OAUTH_EXPECTED_ACCOUNT_ID=ID_DA_CONTA_LIVE_ATUAL
ALPACA_OAUTH_ENCRYPTION_KEY=CHAVE_FERNET_GERADA
```

O callback precisa ser idêntico ao cadastrado na Alpaca. O frontend deve
usar a origem configurada, sem query string. O retorno da Alpaca ao callback
não depende do cookie de sessão, pois navegadores podem não enviá-lo após
o redirecionamento externo. Em vez disso, o callback valida o `state`
aleatório, de uso único e válido por dez minutos, criado por `/start`
a partir de uma sessão administrativa autenticada. `/status`, `/start`,
`/verify` e `/disconnect` continuam exigindo sessão administrativa.

No ambiente local, `ALPACA_OAUTH_REDIRECT_URI` deve apontar para o callback
da API local (se estiver cadastrado na Alpaca), e
`ALPACA_OAUTH_FRONTEND_URL` deve apontar para a origem local do frontend.
Não misture o callback de produção com o estado criado na API local.

Gerar a chave Fernet localmente:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Guarde o valor em segredo no ambiente da API; não copie para o frontend,
GitHub, histórico de logs ou mensagens. Não altere essa chave depois
da conexão sem preparar uma migração, pois ela protege o token armazenado.
`ALPACA_OAUTH_EXPECTED_ACCOUNT_ID` pode ser omitida se
`ALPACA_LIVE_ACCOUNT_ID` já estiver configurada corretamente.

## Passos do teste

1. Faça login normalmente no MCT como administrador.
2. Entre em **Live** e procure **Conexão OAuth com a Alpaca**.
3. Clique em **Conectar com a Alpaca**.
4. Autorize na Alpaca a conta Live correta; não há solicitação de
   escopos de negociação, apenas as permissões de leitura padrão.
5. O callback troca o código por token no backend, consulta
   `GET https://api.alpaca.markets/v2/account`, confere o ID esperado
   e salva o token **criptografado**.
6. De volta ao Live, consulte o estado e use **Verificar conexão**.

Também existe **Desconectar OAuth**, que apaga o token localmente. A
revogação da autorização no lado da Alpaca pode ser feita separadamente.

## Rotas

- `GET /api/alpaca-oauth/status` consulta o estado sem revelar tokens.
- `POST /api/alpaca-oauth/start` cria `state` aleatório (10 min, uso único)
  e retorna a URL oficial de autorização.
- `GET /api/alpaca-oauth/callback` recebe e consome `state`, troca o código
  pelo token e verifica a identidade da conta.
- `GET /api/alpaca-oauth/verify` testa novamente a leitura da conta.
- `POST /api/alpaca-oauth/disconnect` elimina o token armazenado.

Todas as rotas, **exceto `/callback`**, exigem sessão de administrador atual.
O callback valida o `state` vinculado ao administrador que iniciou a conexão.
As rotas POST também conferem a origem do navegador. Tokens, segredos e códigos de
autorização nunca são retornados pela API ou gravados sem criptografia.
A URL de autorização não contém o `client_secret`.

## Etapa 1: Portfolio Live via OAuth (piloto atual)

Após conectar e verificar o aplicativo Alpaca no MCT, o **Portfolio Live
da sessão administrativa atual** passa a utilizar o token OAuth criptografado
salvo no MongoDB para consultar a API Alpaca Trading Live. Não é necessário
adicionar variáveis OAuth ao frontend.

A leitura utiliza `TradingClient(oauth_token=..., paper=False)` do
`alpaca-py` e confirma em cada nova consulta que o `account.id` retornado
corresponde ao ID persistido para o usuário e ao
`ALPACA_OAUTH_EXPECTED_ACCOUNT_ID` (ou `ALPACA_LIVE_ACCOUNT_ID`).
Antes disso, o token é descriptografado **somente dentro da API**.
O navegador nunca recebe token, Client Secret nem ID integral da conta.

- `GET /api/live-market/portfolio`: informações atuais da conta, patrimônio,
  caixa, poder de compra, posições, ordens recentes, histórico e auditoria já
  vinculados à conta verificada.
- `GET /api/live-market/portfolio/export.zip`: a mesma fonte de leitura,
  inclusive nos períodos solicitados na exportação de auditoria.
- O retorno inclui `broker_authentication: "oauth"` quando o token está ativo
  e `"api_keys"` quando ainda não foi conectada uma conta OAuth.
- Se há conexão OAuth armazenada, mas seu token expirou, está ilegível ou
  não corresponde à conta esperada, a API **não retorna dados usando
  silenciosamente as chaves Live**. O Portfolio Live apresenta erro.
- A tela identifica a fonte de leitura e atualiza após verificar ou
  desconectar a conexão. Desconectar restaura o modo legado com chaves
  administrativas, restrito ao administrador atual.

**Isolamento:** esta fase **não é multiusuário**. A conta OAuth deve ser
a mesma conta Live previamente configurada na API. Os metadados Winner
existentes continuam relacionados a essa conta. Nenhuma consulta aceita
um ID de conta escolhido pelo frontend.

**Sem alteração de ordens:** os agendadores, a engine Winner, os serviços
de execução e os métodos de emissão/cancelamento de ordens continuam usando
as credenciais Live originais e mantêm suas verificações operacionais.
O OAuth permanece somente leitura.

### Teste local

1. Atualize as branches de API e frontend e reinicie ambos.
2. Acesse MCT como administrador, abra **Live** e confirme o estado OAuth
   conectado. Caso já esteja conectado, use **Verificar conexão**.
3. Confira o indicador **Acesso à corretora: OAuth da Alpaca (somente leitura)**
   e compare patrimônio, caixa, posições e ordens com a conta Alpaca Live.
4. Faça um download de auditoria e confirme que
   `live_portfolio.json` declara `broker_authentication: "oauth"`.
5. Desconecte OAuth e confira o indicador de **Chaves da Alpaca** e o
   funcionamento legado. Reconecte para voltar ao OAuth.
6. Para testar falha segura, simule um token ilegível em teste automatizado:
   Portfolio Live retorna erro e **não** consulta a conta por API keys.

**Registro de release:** API `v10.8.133` e frontend `v10.7.50`, originados da branch `feature/alpaca-connect-oauth-current-user-v1` e destinados à `main`. Notas detalhadas em `docs/releases/v10.8.133.md`. Merge, tag e deploy no Railway são eventos diferentes; validar o ambiente remoto separadamente.

## Limites

- Não concede `trading` nem `account:write`: se a Alpaca retornar
  esses escopos no token, o callback recusa a associação.
- Não altera os endpoints de execução da Winner, o scheduler,
  o perfil de Trader ou o isolamento de usuários.
- O token salvo **não é utilizado para operar** e nunca vai ao frontend.
- O MCT ainda usa o login Google/senha existente. Esta é a autorização
  **da corretora**, não a substituição da autenticação do MCT.
- Para produção multiusuário, será necessária uma evolução específica
  da autorização, proteção CSRF geral, reconciliação por conta, controle
  operacional individual e aprovação da Alpaca para operar contas de terceiros.

Documentação de OAuth da Alpaca:
https://docs.alpaca.markets/us/docs/using-oauth2-and-trading-api
