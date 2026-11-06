# MCT Winner: homologacao da conta principal por OAuth (DEV)

Status: apenas branch de teste. Nao integrar na main nem publicar em producao antes da homologacao. Nenhuma ordem real foi enviada para construir esta branch.

## O que muda

A conta principal da Winner agora pode usar OAuth no mesmo caminho de corretora para consulta de patrimonio, posicoes, auditoria e envio de ordens. O executor principal preserva as regras existentes: verificacao de conta, janela de mercado, plano valido, posicoes conciliadas, idempotencia das ordens e Live Start/Pause/Stop. A conta principal continua excluida do fan-out de contas OAuth para evitar duplicidade.

Em modo OAuth, **nao ha fallback para API Keys**. Ausencia de token, grant sem permissao trading, grant ambiguo, token invalido ou conta divergente bloqueia a operacao. O cliente utiliza TradingClient(oauth_token=..., paper=False), verifica o ID real da conta e nao devolve credenciais para o frontend.

Os candles historicos SIP/RAW ainda podem usar as credenciais separadas de market data. O escopo desta branch e a autenticacao da conta de negociacao principal; nao houve alteracao no treinamento ou nas decisoes da Winner.

## Configuracao no servico API de DEV

- MCT_CANONICAL_LIVE_AUTH_MODE=oauth : escolhe OAuth. Valor padrao api_keys mantem o comportamento anterior.
- MCT_CANONICAL_LIVE_OAUTH_ALLOW_ORDERS=0 : PADRAO; desabilita o envio real de ordens no fluxo OAuth, independentemente do estado operacional Live. Teste primeiro somente com 0. O valor 1 permite passar para as protecoes operacionais preexistentes e **pode enviar ordens reais** quando Live esta ativo.
- ALPACA_LIVE_ACCOUNT_ID ou ALPACA_OAUTH_EXPECTED_ACCOUNT_ID : ID exato da conta Live principal. Um deles e obrigatorio no modo OAuth. Se os dois estiverem definidos, os valores precisam ser identicos. Nao utilizar o identificador de outra conta.
- ALPACA_OAUTH_ENCRYPTION_KEY : mesma chave Fernet utilizada no cadastro original do grant OAuth. Nao criar chave nova se o token ja foi criptografado.
- MCT_CANONICAL_LIVE_OAUTH_SUBJECT : opcional. Seleciona o login do administrador se existirem multiplos grants validos para a mesma conta. Sem ele, e exigido um unico grant com role admin e trading autorizado.

O grant precisa estar cadastrado na colecao alpaca_oauth_connections com subject e _id consistentes, environment live, account_id exato, oauth_scope trading, trading_authorized true e read_only false. Nunca compartilhar token ou chave em issue, print ou log.

## Roteiro recomendado

1. Usar MongoDB e API de dev isolados de producao, manter Live em STOPPED e a flag ALLOW_ORDERS em 0. O consentimento OAuth Live ainda acessa uma conta Live real, mesmo em dev.
2. Garantir que o administrador ja autorizou a conta exata na Alpaca.
3. Configurar as variaveis acima no backend de dev e reiniciar somente dev.
4. Abrir a aba Live e comparar saldo, posicao, ordens e identificador da conta com o extrato Alpaca. Exportar a auditoria, que deve indicar acesso oauth.
5. Sair da aba Live e voltar: status e vinculo Winner devem se recuperar automaticamente.
6. Testar falhas simuladas: token revogado/ilegivel, grant ausente, multiplos grants e account_id divergente. Deve bloquear sem consultar API Keys.
7. Rodar pytest -q tests/test_canonical_live_oauth_migration.py tests/test_live_control_modes.py tests/test_live_order_execution_safety.py tests/test_live_oauth_shared_scheduler.py tests/test_alpaca_oauth_winner_consent.py
8. Nao ativar ALLOW_ORDERS=1 ou integrar na main antes da revisao dos resultados e de um teste controlado de seguranca.

## Semantica do status

- OAuth conectado + Live parado = credencial presente, robo sem envio de ordens.
- OAuth conectado + Live ativo + ALLOW_ORDERS=0 = patrimonio consultavel, ordens bloqueadas explicitamente.
- OAuth conectado + Live ativo + ALLOW_ORDERS=1 = ordens podem ser submetidas, mas somente apos verificacoes da Winner e conciliacao de corretora.
- OAuth invalido = erro imediato, sem fallback para API Keys.

Nao usar este experimento para modificar o patrimonio ou ordem real sem autorizacao explicita e plano de homologacao.

## Diagnostico de conectividade (sem ordens)

No frontend da branch de dev, ao detectar modo OAuth, o administrador pode clicar em **Diagnosticar Live OAuth**. O endpoint `GET /api/live-market/oauth-diagnostic` e somente-leitura e requer sessao de administrador. Ele testa primeiro o grant e a vinculacao explicita da conta principal, depois consulta posicoes, ordens e relogio da Alpaca. Nenhuma ordem de corretora e enviada. A resposta inclui apenas um codigo seguro, sem token, account ID, credenciais ou mensagens internas do provedor.

Um `CANONICAL_ACCOUNT_ID_MISSING` indica que nenhuma das duas variaveis de account ID foi carregada pela API. `CANONICAL_ACCOUNT_ID_CONFLICT` significa que os dois IDs sao diferentes. `OAUTH_ADMIN_GRANT_MISSING` significa que nao existe um grant adequado de administrador no banco de dev. `OAUTH_BROKER_ORDERS_UNAVAILABLE` indica que a conexao foi verificada, mas a consulta de ordens falhou. Se o diagnostico retornar `ready` e o portfolio ainda estiver indisponivel, o problema pode estar em processamento ou historico/observacoes da carteira, que exigem outra investigacao.
