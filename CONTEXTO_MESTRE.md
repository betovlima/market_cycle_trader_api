# CONTEXTO_MESTRE

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
