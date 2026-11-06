# Winner Live OAuth: configuração pela interface (branch de desenvolvimento)

Este fluxo está isolado nas PRs de desenvolvimento. Não mesclar na main nem publicar em produção antes da homologação. Não foram enviadas ordens para criar ou testar o código.

## Onde fica cada configuração

As decisões operacionais estão na aba **Live > Conexão OAuth com a Alpaca > Broker da Winner · Configuração Live**. Um administrador autenticado pode:

1. Ver o transporte atual da Winner e o modo do robô.
2. **Usar minha conta OAuth na Winner**: enquanto o robô está parado, valida novamente o grant OAuth de negociação do administrador e a identidade da conta na Alpaca, vincula a conta principal ao robô e salva a configuração no MongoDB com autorização de ordens DESABILITADA.
3. **Autorizar ordens reais**: somente após o vínculo e com robô parado, exige confirmação explícita e registra a alteração. O controle do robô deve ser ativado SEPARADAMENTE; a autorização não dispara operações.
4. **Bloquear novas ordens**: retira a autorização persistida, inclusive se o robô estiver ativo. Isso não desfaz ordens já submetidas.

Essas configurações ficam no documento versionado `live_trader_control`, com `revision` e registro de alterações em `live_trader_control_history`. O cliente OAuth e o identificador da conta principal provêm do vínculo autorizado pelo administrador; não há novos parâmetros de operação no `.env`.

Os campos `MCT_CANONICAL_LIVE_AUTH_MODE`, `MCT_CANONICAL_LIVE_OAUTH_ALLOW_ORDERS` e `MCT_CANONICAL_LIVE_OAUTH_SUBJECT` **não são mais utilizados** pela API nesta branch. Remova-os da configuração de dev para evitar confusão.

Segredos de infraestrutura, como `ALPACA_OAUTH_CLIENT_SECRET`, a chave `ALPACA_OAUTH_ENCRYPTION_KEY` e credenciais técnicas do provedor de cotações, permanecem exclusivamente no backend ou no gerenciador de segredos. Nunca aparecem em campos editáveis do front.

## Segurança

- O fluxo começa no estado herdado `api_keys`, sem habilitar o modo OAuth automaticamente. O administrador precisa vincular explicitamente sua conta.
- No modo OAuth, **nunca** existe fallback silencioso para API Keys ao consultar ou enviar ordens.
- Não é possível autorizar ordens sem OAuth válido, conta correspondente e robô parado.
- A alteração usa comparação de revisão para evitar sobrescrever decisões concorrentes.
- O executor bloqueia ordens reais enquanto a autorização não estiver ativa e checa a permissão novamente antes de cada operação.
- A conta canônica é excluída do processamento paralelo OAuth dos demais traders, evitando duplicações.
- O bloqueio de novas ordens não cancela nem estorna ordens anteriores.
- O universo dos ativos do Dashboard continua vindo da configuração imutável da Winner selecionada no banco. O dev pode usar 25 ativos enquanto outra Winner possui 77; não troque a estratégia silenciosamente.

## Como testar em dev

1. Atualize as branches da API e do front e inicie ambos normalmente. Mantenha o robô Live parado.
2. Entre na aba Live com sessão administrativa e autorize sua conta Alpaca via OAuth, se necessário.
3. Na seção **Broker da Winner · Configuração Live**, clique em **Usar minha conta OAuth na Winner**. Confirme que o modo indicado se torna OAuth e que as ordens continuam bloqueadas.
4. Confira a carteira, posições, auditoria e diagnóstico. Saia e retorne à aba Live para verificar a persistência da configuração.
5. Execute o dry-run já existente para verificar elegibilidade e identidade, sem enviar ordens.
6. Não clique em **Autorizar ordens reais** durante os testes iniciais. Essa permissão é destinada à homologação operacional posterior, de forma controlada.
7. Teste erros de permissão, token inválido e conta divergente com mocks, nunca editando tokens reais.

Para os testes automatizados da API: `python -m unittest discover -s tests -p "test_*.py"`. A interface usa `npm test` e `npm run build`.

## API administrativa

- `GET /api/live-market/broker/settings`: transporte, permissão, revisão e sufixo da conta, sem segredos.
- `POST /api/live-market/broker/settings`: operação administrativa versionada, com confirmação explícita; não envia ordens.
- `GET /api/live-market/oauth-diagnostic`: diagnóstico seguro e somente leitura.
