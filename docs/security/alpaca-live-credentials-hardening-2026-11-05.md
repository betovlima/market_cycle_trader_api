# MCT Live: credenciais Alpaca e limites de segurança (2026-11-05)

## Escopo confirmado no Railway
Projeto `fulfilling-bravery`, ambiente `production`, serviço `market_cycle_trader_api`.
Apenas **nomes**, nunca valores, foram inspecionados:
- `ALPACA_LIVE_API_KEY_ID`, `ALPACA_LIVE_SECRET_KEY`, `ALPACA_LIVE_ACCOUNT_ID`
- `ALPACA_OAUTH_CLIENT_ID`, `ALPACA_OAUTH_CLIENT_SECRET`, `ALPACA_OAUTH_ENCRYPTION_KEY`
- `ALPACA_API_KEY_ID`, `ALPACA_SECRET_KEY` (Paper), `MONGO_URL`
- `TRADER_SESSION_SECRET`, `TRADER_ADMIN_PASSWORD`
O frontend não possui variáveis sensíveis de Alpaca. No servidor, a consulta da carteira prefere o OAuth do administrador, mas o executor da conta principal usa a mesma dupla de API Key Live guardada no Railway. A função `create_live_readonly_client()` é **somente uma intenção de uso**: instancia `TradingClient` com credenciais que também permitem negociação. A verificação `ALPACA_LIVE_ACCOUNT_ID` já impede selecionar uma conta diferente em caso de rotação de chave, quando configurado.

## Controles imediatos, sem necessidade de editar código
1. Railway: serviço `market_cycle_trader_api` > Variables. Marcar **Seal** nas variáveis acima que contêm segredos (e nas do serviço `market_cycle_monitor_api`). Em cada variável: menu `...` > `Seal`. Variáveis sealed são write-only no dashboard/API/CLI, mas **continuam disponíveis para o processo e durante build**. Sealing **não** impede uso após uma invasão ao container. Sealed não pode ser desfeito. Não migrar para Shared Variables nem para o frontend.
2. Railway: habilitar MFA/passkey e auditar membros, tokens de acesso e histórico de alterações. Preferir acesso de projeto/ambiente, e RBAC de produção caso o plano suporte. Remover acessos excessivos e tokens não usados.
3. Alpaca: revisar sessões, integrações OAuth e ordens recentes. Ativar MFA e notificações de ordem/saque. Se houver suspeita de exposição, **revogar e regenerar** as API Keys Live pela interface oficial da Alpaca, sem copiar segredo em tickets, chat, GitHub ou logs. Coordenar a troca durante janela sem negociação: pausar execução, gerar nova dupla, atualizar variáveis diretamente no Railway, validar account ID/carteira, retornar o modo de execução e invalidar a dupla antiga. Regenerar antes do novo deploy pode interromper a automação.
4. Proteger separadamente `ALPACA_OAUTH_CLIENT_SECRET`, `ALPACA_OAUTH_ENCRYPTION_KEY`, `TRADER_SESSION_SECRET`. **Não alterar a encryption key OAuth isoladamente**: os tokens existentes foram criptografados com ela, e a rotação exige recriptografia/reautorização com testes e backups. Rotacionar a senha administrativa conforme política de acesso.
5. Não criar credenciais Broker API com permissões Funding/Journaling. **Não confundir** o controle fino da Broker API com API Keys da Trading API: chaves comuns de negociação Live não oferecem necessariamente a granularidade de permissões da Broker API.
6. Auditar respostas e logs para que nunca contenham chaves, tokens, headers de autorização nem parâmetros de conexão. Esta alteração reduz exposição em respostas HTTP de Portfolio Live, export Live e execução manual. Ela **não** reduz os privilégios da credencial subjacente.

## Arquitetura recomendada para remover a chave do Railway
Implementar `Trade Gateway` independente (por exemplo, AWS Lambda com role IAM + AWS Secrets Manager) e controlar estritamente o que pode enviar à Alpaca. O MCT Railway passa a calcular a decisão e solicitar **uma intenção limitada de ordem**, sem receber `ALPACA_LIVE_SECRET_KEY`. O gateway verifica autonomamente vínculo da conta, data do pregão, sessão e intenção autorizada, limite financeiro, instrumento permitido, idempotência, modo de operação e auditoria antes de negociar. O gateway precisa de autenticação forte entre serviços, autorização separada e anti-replay.

**Apenas deslocar a chave para um cofre remoto sem isolar a execução não resolve:** se o Railway receber permissão para ler livremente o segredo, uma invasão ao container pode extraí-lo; se puder enviar ordens arbitrárias ao gateway, ainda pode causar negociações indevidas. A fronteira de confiança deve ser imposta no gateway. A integração requer nova infraestrutura, testes Paper/Live e migração coordenada.

## Incidente da Alpaca
Em 2026-11-05 não havia confirmação oficial encontrada de violação de API Keys na Alpaca no histórico público de status. Os eventos de 2026-11-03 e 2026-11-04 tratavam, respectivamente, de verificação SMS e indisponibilidade de fornecedor para cripto. Segurança preventiva e rotação de credenciais suspeitas continuam apropriadas.

## Referências
- https://docs.railway.com/variables
- https://docs.railway.com/enterprise/security
- https://docs.alpaca.markets/us/docs/using-oauth2-and-trading-api
- https://docs.alpaca.markets/us/docs/authentication
- https://alpaca.statuspage.io/history
