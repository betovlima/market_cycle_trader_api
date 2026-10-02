# CONTEXTO_MESTRE — Market Cycle Trader

Última atualização: 2026-10-01. Documento de continuidade; não substitui a verificação de versões realmente implantadas nem os registros por versão em `docs/changes/`.

## Separação de responsabilidades
- **TCC** (`betovlima/tcc_mba_usp_data_science_analytics`): pesquisa, comparação Control/Soft, otimização, validação cronológica e congelamento de evidências.
- **MCT**: aplicação de uso real e execução operacional da política selecionada pelo TCC. Não recriar um ambiente de experimentação ou otimização de estratégias dentro do MCT.
- O MCT usa dados de mercado atualizados da Alpaca; o TCC preserva o snapshot científico em CSV/manifesto no Git. Histórico recalculado e corporate actions precisam ser auditáveis.

## Baseline operacional confirmado pelo usuário
- API de produção: `10.8.38`.
- Front de produção: `10.7.16`.
- A evolução `v10.8.39` está em **branch de desenvolvimento**, não em produção. A constante API_VERSION foi atualizada para 10.8.39 somente na branch; produção continua API 10.8.38 e Front 10.7.16.
- Preservar Winner existente, estado da carteira e ordens. Não promover a Strategy #28 diretamente.

## Resultado científico selecionado
- Fonte de comparação: TCC congelado v1.0.6, commit `c0d71772092f0c26c9f28f0211b9933e9c396b95`, `EXPERIMENT_CHECKPOINT_V1_0_6.md`.
- Control: LightGBM Utility + política-base de rotação + switch margin, sem Soft. Capital final histórico: US$ 10.094.316,30 (inicial US$ 10.000).
- Soft: US$ 9.851.632,93 no mesmo experimento. Control é a referência de implementação operacional desta etapa, não uma previsão de retorno.
- 56 ativos solicitados e 55 elegíveis no snapshot. DOC foi excluído por mudança estrutural de identidade DOC -> PEAK.
- Parâmetros científicos: horizontes (5,10,20,40,60); pesos (0.10,0.15,0.20,0.30,0.25); purge 60; calibração 126; teste 504; min-holding 2; entry edge 0.001; switch margin base 0.0005; candidatos (0,0.0025,0.005,0.01); CPU; LightGBM 329 estimators e demais valores em config científica.
- Para qualquer ativo estruturalmente inválido: excluir com evidência no manifesto; não fazer bridge, reconstruir série ou ajustar manualmente.

## Diagnóstico de integração
- A Strategy #28 é referência de Backtest vinculada ao motor científico `tcc_v106_reference`. O instalador original cria/seleciona somente Research. A API bloqueia promoção desse binding ao Trader.
- O Trader usa outra cadeia: `services/paper_trading.py` -> `engine/live_model_signal.py` -> `engine/live_lightgbm_signal.py` -> `engine/live_policy.py`. Não assumir que copiar parâmetros já gera a mesma decisão.
- O replay conhece a barra posterior para validar execução na próxima abertura; o Trader prepara ordem antes dessa barra existir. Evitar qualquer uso de informação futura.
- Ver `docs/changes/v10.8.35-tcc-reference-main-integration.md`, `v10.8.37-tcc-cutoff-and-independent-tuning.md`, `v10.8.38-tcc-inclusive-oos-session.md`, `v10.8.39-control-operational-parity.md`.

## Trabalho em andamento
- Branch: `feature/v10.8.39-control-operational-parity` a partir de `main` API v10.8.38.
- Causa: o motor de referência científica não é o runtime de Trader e não é elegível para promoção.
- Implementado até aqui: teste isolado de paridade da regra de decisão Control vs política live, incluindo caso de ausência da barra futura; documentação do contrato de migração.
- Incremento seguinte no mesmo branch: `engine/operational_control_contract.py` usa painel do Control científico v1.0.6 em modo isolado e sem ordens. Testes verificam igualdade de features em OHLCV idêntico, calendário ancorado vs o atual calendário automático, ausência de barra futura e indisponibilidade de âncora.
- Incremento seguinte: `engine/operational_control_preview.py` monta decisão shadow de Control com treino/calibração congelados e política live, sem ordens; `scripts/control_shadow_preview.py` consome snapshot CSV SHA-verificado local, registra exclusões estruturais e produz JSON em `output/`. Testes cobrem janelas/ações/falhas e CLI. O snapshot CLI não é série atual Alpaca.
- Já implementado na branch: download atual RAW/SIP e Corporate Actions para snapshot local separado e ensaio da última sessão. Ainda não implementado: prova de identidade numérica sessão a sessão com TCC, novo binding operacional e integração à cadeia real Paper/Trader, alteração do catálogo e ativação da Strategy. Nenhum deploy ou ordem autorizado.
- Status: a CI do commit inicial `66e31f6f0be882b5bb3980776d8240c0e29a30ab` passou em Python 3.12. Após erro de importação no PyCharm/Python 3.14.2, o teste foi corrigido para inserir `<API>/src` no `sys.path` a partir de `__file__`; aguardar CI do novo commit. Não afirmar paridade operacional completa com base no teste de política.
- O novo preview inclui treino/calibração científicos e CLI de snapshot local; a CI identificou duas falhas apenas nos testes novos (índice esperado do calendário e fixture de exclusão estrutural). Ambas foram corrigidas na branch; confirmar CI do HEAD antes de merge. O CLI imprime progresso e nunca envia ordens.
- Erro local original: `ModuleNotFoundError: No module named 'market_cycle_trader_api'`, durante importação, antes de qualquer teste. Não indica divergência Control vs live; preferir Python 3.12 para ambiente de referência.

## API Swagger para ensaio Control com dados NOVOS do MCT
- Decisão corrigida pelo usuário: não usar a pasta do TCC para o ensaio do MCT. O MCT baixa sempre novos dados da Alpaca, cria `<raiz_api>/dados/control_shadow/snapshots/<job_id>` automaticamente, mantém raw CSV, Corporate Actions JSON, CSV normalizados e manifesto com SHA-256; `/dados/` consta no `.gitignore`.
- O endpoint administrativo `POST /api/admin/control-shadow/jobs` exige confirmação `REFRESH_ALPACA_CONTROL_SHADOW_NO_ORDERS`. Alimentação: RAW/SIP 1Day desde 2016-01-01 até a última sessão XNYS segura, corporate actions até a mesma sessão, normalização de splits e exclusão estrutural explícita. Sem fallback de fonte/correção manual.
- Apenas `MCT_CONTROL_SHADOW_API_ENABLED=true` no `.env` local da API. `MCT_CONTROL_SHADOW_FROZEN_ROOT` **não é utilizado pelo endpoint atualizado**; não é necessário copiar ou clonar o TCC. Os parâmetros HTTP não contêm caminho.
- MongoDB: somente a coleção isolada `control_shadow_jobs` para status/logs/resultados. Console PyCharm/Uvicorn imprime progresso por ativo e LightGBM. GET `/jobs/{job_id}` traz resultado e diretório, GET `/jobs/{job_id}/logs` mostra logs. Um job por vez.
- A política Control científica continua referência de código/parametrização, mas os dados são NOVOS; não pressupor mesmo capital histórico. `status=shadow_only`, `order_eligible=false`, `order_submission=never`. Não se conecta ao executor Paper/Trader nem altera Winner, carteira, Strategy #28 ou ordens.
- O CLI anterior com `--frozen-root` permanece como ferramenta científica auxiliar; não é o fluxo principal de /docs.
- Produção informada: API v10.8.38 / Front v10.7.16. Branch API v10.8.39 em desenvolvimento, sem merge/deploy.
- O ensaio requer credenciais Alpaca com SIP histórico; 403/429 sem dados válidos falha com log e sem decisão. A pasta é criada no servidor/processo PyCharm que executa a API.
- Uma thread não sobrevive a reinício; usar um worker sem reload enquanto ensaia. Se um job ativo ficar preso após encerramento abrupto, requer recuperação controlada, não iniciar outro job à força.
- Testes novos cobrem download simulado, metadados, hashes, exclusão estrutural, falha de provedor, segurança de endpoint e ausência de rota de ordens. Ver `docs/changes/v10.8.39-control-operational-parity.md`.

## v10.8.62 — TCC v1.0.6 Control Operational Runtime

### Decisão do usuário
- Em 2026-10-01 o objetivo operacional foi explicitamente corrigido para: **o MCT deve ter o mesmo comportamento do TCC**.
- Isto substitui, como objetivo de integração operacional, a ideia de levar a linha Liquidity-Aware/Meta-Veto ao Trader. A v10.8.57 continua congelada como melhor referência retrospectiva daquela linha (`US$ 2.405.223,65`), mas não representa o Control científico puro do TCC.
- A Strategy #28 científica continua imutável e bloqueada para Trader; não converter referência científica em produção apenas mudando metadados.

### Por que o TCC teve ~US$ 10,09M e a linha MCT ficou abaixo
Os números vieram de protocolos diferentes:
- TCC v1.0.6 Control, snapshot científico: `US$ 10.094.316,30`.
- MCT v10.8.41, núcleo científico Control sobre snapshot Alpaca/MCT atualizado: `US$ 5.887.904,38`.
- MCT v10.8.42, execução com capacidade/custos: `US$ 528.709,78`.
- MCT v10.8.44, Liquidity-Aware: `US$ 1.078.635,41`.
- MCT v10.8.57, Liquidity-Aware + Temporal Meta-Veto: `US$ 2.405.223,65`.
Conclusão: os ~US$ 2,4M não são uma reprodução pior do mesmo Control de ~US$ 10M; são outra política/protocolo sobre outro snapshot. Mesmo comportamento do TCC em dados Alpaca atuais não implica mesmo capital histórico. Reproduzir exatamente US$ 10.094.316,30 exige também o snapshot/precisão numérica da execução original.

### Nova identidade operacional
- Branch: `feature/v10.8.62-tcc-control-operational-runtime`, derivada da linha operacional v10.8.39 e não da linha experimental Meta-Veto.
- API de desenvolvimento: `10.8.62`; produção continua API `10.8.38` / Front `10.7.16`.
- Novo modo conceitual: `COMPOUND_ROTATION_SWING_TCC_CONTROL_V106`.
- Label: `TCC v1.0.6 — Control Operational`.
- Nenhum merge/deploy/promoção/ordem executado.

### Contrato implementado
- `engine/tcc_control_operational_runtime.py` concentra o contrato protegido.
- Backtest do novo modo chama diretamente o vendorizado `tcc_v106_reference.research_challengers.run_research_challenger`, com `build_control_config(TCC_CONFIG)`, fees/slippage científicos e Soft desabilitado; não usa `run_rotation_models` genérico para o sinal.
- Live: `live_model_signal` detecta o novo modo e chama `build_live_tcc_control_decision`, que reutiliza `operational_control_preview.build_control_shadow_decision` e funções científicas de fit/calibração/política; o `live_lightgbm_signal` genérico é bypassado.
- Paper/Trader, somente nesse modo, baixa um snapshot RAW/SIP atual + Corporate Actions, aplica normalização causal de splits e exclusões estruturais antes do treino. O cache genérico de market data não alimenta diretamente o modelo TCC.
- Cada plano registra `tcc_control_snapshot_directory`, `tcc_control_snapshot_sha256` e `plan_source=tcc_control_v106_operational`.
- O MCT continua responsável por estado, calendário, reconciliação e ordens; o TCC Control vendorizado é responsável pelo sinal.

### Parâmetros congelados
- 56 ativos solicitados na ordem TCC; exclusão estrutural continua explícita.
- Horizontes 5/10/20/40/60; pesos 0.10/0.15/0.20/0.30/0.25.
- min training 700; calibração 126; teste 504; min teste 126; purge 60.
- downside 0.20; drawdown 0.35; min hold 2; edge 0.001; cash threshold 0.
- switch margin 0.0005; candidatos 0/0.0025/0.005/0.01.
- LightGBM TCC: 329 trees, learning rate .020731, depth 3, leaves 6, child samples 18, child weight 5, subsample .85/freq0, colsample .88067, alpha .050837, lambda 3.596305, max_bin255, n_jobs=-1, repetitions1, seed_step1000, random_state42; CPU.
- Initial capital 10k, fractional shares no simulador científico, slippage/commission zero e taxas regulatórias congeladas.

### Proteções de catálogo/runtime
- Salvar uma Strategy no novo modo normaliza automaticamente os campos do contrato e grava snapshot LightGBM `tcc-v1.0.6-control` com source `tcc_v106_operational_contract`.
- `_trader_runtime_compatibility` só torna o modo elegível se configuração e snapshot LightGBM corresponderem ao contrato TCC.
- `paper_trading._validated_context` revalida ambos antes de preparar plano.
- Desvio de parâmetros ou modelo falha fechado.
- Strategy #28 com `backtest_engine_binding=tcc_v106_reference` permanece inelegível, preservando a referência científica.

### Precisão e expectativa de capital
- Auditorias anteriores provaram que diferenças submáquina/CSV em features podem mudar árvores LightGBM, predições e switch margin.
- Portanto "mesmos hiperparâmetros" não é suficiente. Paridade deve cobrir dados, normalização, features/targets, janelas, predições, margem, decisões e curva.
- A v10.8.62 entrega **paridade de comportamento/código**. A reprodução exata do checkpoint de US$ 10,09M precisa ser certificada usando o mesmo snapshot numérico que o produziu; o snapshot Alpaca corrente pode gerar valor diferente sem indicar divergência da política.

### Testes
- `tests/test_tcc_control_operational_runtime.py` cobre contrato congelado, rejeição de drift, snapshot LightGBM, bypass do live genérico, chamada do runner científico no backtest, compatibilidade de catálogo, RAW/SIP live e normalização automática de Strategy.
- `tests/test_tcc_control_operational_parity.py` continua garantindo paridade da regra científica vs política live nos estados sintéticos e documentando a exceção legítima da barra futura.
- CI do commit `420cf8ef695ba23b9e0be9ea7b83a19c13e404bd` passou antes da documentação final. Revalidar CI do HEAD após docs/CONTEXTO.

### Primeiro backtest real da v10.8.62
- Job `20261002T000018-d07e7c64`, status completed, Strategy #12 rev3.
- Contrato confirmado: `COMPOUND_ROTATION_SWING_TCC_CONTROL_V106`, LightGBM, `operational_contract=tcc_v1.0.6_control`, source commit `c0d71772092f0c26c9f28f0211b9933e9c396b95`, CPU, seed42, RAW/SIP, causal split normalization, DOC excluído, 55 elegíveis, 17 splits.
- Resultado: capital `US$ 5.882.637,94`; CAGR 180,16%; Sharpe 1,95472; MaxDD -36,6504%; 331 rotações; exposição 100%; buy-and-hold `US$ 39.279,28`.
- A diferença contra a referência científica MCT v10.8.41 (`US$ 5.887.904,38`) é apenas `-US$ 5.266,44` / `-0,08945%`. Isso confirma que a v10.8.62 reproduz a linha Control científica do MCT praticamente no mesmo patamar.
- Na mesma última sessão do checkpoint TCC, 2026-09-16, a curva v10.8.62 estava em `US$ 5.551.262,80`, aproximadamente -45,01% contra o checkpoint TCC `US$ 10.094.316,30`. Os dez pregões adicionais até 2026-09-30 elevaram o capital ~5,97%, portanto a extensão temporal não explica a diferença histórica.
- Snapshot atualmente versionado na `main` do TCC: 148.060 RAW rows elegíveis, 2.692 por ativo solicitado, SHA `4e2fd225cc0ea05da56dad8f0628ca989ad332812a5fa3a7dc796b2b8a6d5128`. Porém o manifesto declara `created_for_experiment_version=1.2.0-dev.1` e não existe nos caminhos atuais do commit/tag v1.0.6; não tratá-lo como prova do dataset exato dos US$ 10,09M.
- Snapshot atual v10.8.62: 148.610 RAW rows elegíveis, market-data signature `8d61930b791e8bfe350b88c4284717ed228389b689f821b40ff4276a59aa7650`, 550 linhas adicionais no universo elegível.
- O Fold1 desta execução escolheu candidate/effective switch margin `0.01/0.01`; a auditoria TCC congelada já havia mostrado que diferenças muito pequenas de features podem mudar árvores/predições e a escolha do margin. Portanto o próximo trabalho é **paridade numérica do snapshot TCC**, não alteração de política/parâmetros.
- Não perseguir os US$ 10,09M via tuning. Para reproduzir o checkpoint, usar exatamente o snapshot TCC e localizar a primeira divergência numérica/decisória; para operação real, manter dados Alpaca atuais com o comportamento TCC já recuperado.

### Próxima validação antes de promoção
1. Executar backtest do novo modo em input TCC cuja identidade de checkpoint possa ser comprovada; comparar com `US$ 10.094.316,30` sem ajustar parâmetros.
2. Executar o mesmo modo sobre o snapshot MCT atual e comparar com o Control científico equivalente (ordem de grandeza histórica v10.8.41 ~US$ 5,888M), não com a linha Liquidity/Meta.
3. Comparar decisões/previsões sessão a sessão.
4. Rodar shadow prospectivo em dados atuais.
5. Só então, mediante decisão explícita, considerar merge/deploy/promoção do novo perfil.


## Próximos passos
1. Confirmar CI de toda a branch, especialmente o novo shadow preview e CLI.
2. Isolar o contrato de dados/treinamento/decisão Control v1.0.6 e comparar com o live na mesma janela congelada.
3. Implantar modo de ensaio sem ordens, com resultado sessão a sessão e auditoria de divergência.
4. Criar novo perfil operacional somente após paridade aprovada, mantendo Strategy #28 original imutável.
5. Executar o job /docs com novos dados Alpaca em ambiente local, auditar snapshot/manifests e comparar decisões; depois testar estado da carteira e integridade de ordens antes de considerar qualquer promoção.

## Comandos Git
```powershell
git fetch origin
git switch main
git pull --ff-only origin main
git switch --track origin/feature/v10.8.39-control-operational-parity
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -p "test_tcc_control_operational_parity.py" -v
python -m unittest discover -s tests -p "test_*.py" -v
```
