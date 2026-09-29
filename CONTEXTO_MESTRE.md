# CONTEXTO_MESTRE — Market Cycle Trader

Última atualização: 2026-09-28. Documento de continuidade; não substitui a verificação de versões realmente implantadas nem os registros por versão em `docs/changes/`.

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

## v10.8.42 — Control Execution Feasibility (pesquisa, sem ordem)
### Resultado real v10.8.42 (29/09/2026, `control-execution-c38169f6c6fc4ea0`)
- ZIP original com snapshot e três relatórios conferido: manifesto + 167 SHA dos arquivos válidos, 55 ativos. 3 folds/1.554 sessões; v10.8.41 = US$ 5.887.904,38; v10.8.42 capacidade executável *somente no cenário hipotético* = US$ 528.709,78 de patrimônio marcado a mercado, CAGR 89,94%, Sharpe 1,460, MaxDD -38,70%, 291 transições executadas; ainda 2.147 AMZN e US$ 225,73 em dinheiro, sem liquidação final.
- Gargalos: 399 sessões parciais; 172 sessões com >50% do capital em CASH (171 segurando VNCE ou PXLW), fold 2 = 138 dessas sessões. 47 sessões consecutivas com >50% CASH de 24/03 a 31/05/2023. Uma ação PXLW residual reteve quase 100% em CASH em 28/08/2024. VNCE e PXLW respondem por 99,94% do total de ações solicitadas e não preenchidas em *linhas com execução* (quantidades repetidas entre dias, não demanda única). Modelo com posição real divergiu do ativo da linha original em 142 sessões, sinal divergiu em 39; logo custo de oportunidade não é apenas fee/spread.
- Custo de preço modelado US$ 65.671,66 e taxas US$ 981,85, não equivalentes à diferença composta de US$ 5,36M. Sem fills >10% de volume diário; zero-volume sem execução. Histórico detalhado e plano de comparação fatorial/participação *pré-definidos* em `docs/changes/v10.8.42-control-execution-feasibility.md`. Não mudar Control nem excluir ativos somente por liquidez sem nova campanha conceitual; não alterar Winner/produção.


- Em 29/09/2026 o usuário aprovou validação de liquidez/exequibilidade do Control após ZIP completo do snapshot e relatórios v10.8.41. Referência **numericamente reproduzida**: Control `control-validation-7821002400424ccc`, 55/55 hashes numéricos, 3 folds, US$ 5.887.904,38 (US$ 10 mil iniciais), CAGR 180,46%, Sharpe 1,956, MaxDD -36,65%, 330 rotações; resultado científico, não capital executável.
- Evidência no ZIP: simulação sem capacidade permitia compras/vendas maiores que 100% do volume da sessão (VNCE e volume zero). Branch `feature/v10.8.42-control-execution-feasibility` (base v10.8.41) adiciona replay estado-dependente com fills parciais, CASH residual, preço adverso e limite 10% do menor entre volume histórico mediano 20 dias **anteriores** e volume realizado no dia de execução. Volume do dia é usado retrospectivamente só para limitar execução, nunca para sinal ou escolha de ativo. Sem duas posições, venda incompleta bloqueia nova compra, sem dívida, sem liquidação fictícia no último dia; resultados são cenários de execução, não confirmação de corretora.
- Fonte `tcc_v106_reference` e repositório TCC imutáveis: CI detectou modificação involuntária do código vendorizado e a alteração foi revertida. Adaptador isolado do MCT injeta simulador na função-clone local sem mutação de globais do módulo científico. Não alterar baseline v10.8.41, snapshot, Winner, carteira, produção; rotas admin `POST/GET /api/admin/control-shadow/execution/jobs`; referência necessária `control-validation-7821002400424ccc` com SHA `6d9e7d69865277487a6b193adedcc1d91e428ab451921d538aa4938fe108e8f3`.
- Resultados ficam em `dados/control_shadow/snapshots/control-shadow-870fb66e1bdc4fd0/validation/v10.8.42/<job_id>` CSVs, JSON e PNGs. Documento `docs/changes/v10.8.42-control-execution-feasibility.md`. Ainda precisa executar no snapshot local e interpretar os números; **não presumir capital**, não promover Winner. Verificar CI final antes de uso local.

## v10.8.41 — integridade binária no snapshot (paridade de calibração)
- O job real `control-validation-468931e4196849fa` concluiu (3 folds, 4 margens, sem ordens) sobre `control-shadow-870fb66e1bdc4fd0`; score original -0.6868088598099447/margem 0.0025 vs validação -0.896092331947243/margem 0.01, janelas idênticas. OOS v10.8.40 = US$ 7.675.707,75 partindo de US$ 10k, CAGR 192.74%, Sharpe 2.0142, MaxDD -42.55%, 341 rotações, exposição 100%. Não promover; calibração não reproduzida.
- Hipótese testável: loader v10.8.40 usava parser decimal padrão sobre CSV `%.17g` e não verificava igualdade das entradas numéricas com hash original em memória. Branch derivada `feature/v10.8.41-control-snapshot-bitwise-parity` usa `read_csv(float_precision="round_trip")` e exige que o `normalized_history_sha256` de cada ativo coincida com o hash canônico do frame recarregado antes de qualquer treinamento. Snapshot imutável, falha explícita em divergência. Salva relatórios em `validation/v10.8.41/<id>`, preserva v10.8.40.
- Precisamos rodar validação no snapshot local, comparar `numeric_input_integrity` e `original_shadow.reproduced`; se inputs idênticos mas score diferente, auditar LightGBM por ativo sem forçar margem. CI final ainda depende de confirmação. Detalhes `docs/changes/v10.8.41-control-snapshot-bitwise-parity.md`.

## v10.8.40 — correção de origem ausente no Mongo e nova execução
- Primeiro POST de validação falhou com 404 porque `control-shadow-9602cef2679f4d2a` não foi encontrado na coleção `control_shadow_jobs` do Mongo atualmente conectado. Não é prova de perda do snapshot local. Corrigido POST de validação para aceitar `expected_snapshot_sha256` opcional, obrigatório se registro Mongo original estiver ausente; comprova manifesto e todos os arquivos SHA na pasta dados local, sem redownload. Não aceita Mongo job failed/running como snapshot completo. Exibe `source_registry_mode`; ausência da referência original para score/margem -> null, não falso `reproduced=false`.
- SHA original independente (log da primeira execução): `319e1658e24227616d2930a17515ddddc3e792cd9aa2fa9736b6922f8f66fee5`.
- Usuário iniciou novo job **de download** `control-shadow-870fb66e1bdc4fd0`, visto `running`, `market_data_download`, 19/56 e snapshot_directory null. Se concluir no mesmo Mongo, usar ID novo em POST `/api/admin/control-shadow/validation/jobs` após status completed; não precisa hash independente nesse caso. Não fazer terceiro download.
- A execução do novo job via `POST /api/admin/control-shadow/jobs` é distinta do endpoint de **validação** `POST /api/admin/control-shadow/validation/jobs`; este só analisa snapshot existente. Winner, carteira, ordens e TCC não alterados.

## v10.8.40 — validação do snapshot existente sem download
- Branch `feature/v10.8.40-control-snapshot-validation` derivada do HEAD v10.8.39 (CI verde). API_VERSION 10.8.40 **somente na branch**; base PR #7 permanece draft. Nenhum merge ou deploy para produção.
- Usuário executou `control-shadow-9602cef2679f4d2a`: status completed, RAW/SIP até 2026-09-28, 55/56 elegíveis, exclusão DOC->PEAK, decisão hipotética CASH->AVGO, utility 0.35302977890449344, margem 0.0025, calibration score -0.6868088598099447. O score é soma de recompensas ajustadas ao risco, não -68.68% de retorno.
- Nova rota admin POST `/api/admin/control-shadow/validation/jobs`: literal `VALIDATE_EXISTING_CONTROL_SNAPSHOT_NO_ORDERS` e `source_job_id`; GET status e GET logs nos mesmos paths acrescidos de `/{job_id}` e `/{job_id}/logs`.
- Valida status/identidade do job original, SHA do manifesto e de todos os arquivos de dados da própria API; não faz novo download nem lê CSVs do TCC. Comparação de quatro margens/curvas e replay cronológico OOS oficial com janelas/purge/simulador, no mesmo snapshot; outputs CSV, JSON e PNG em pasta por validation job. Status Mongo separado em `control_shadow_validation_jobs`; console logs e um job por vez. Nada de ordens, promoção ou alteração do Winner.
- Prova completa de paridade com Trader e integração operacional ainda pendentes. Ver `docs/changes/v10.8.40-control-snapshot-validation.md`.
- Em Git Bash usar `PYTHONPATH=src python -m unittest discover -s tests -p "test_control_snapshot_validation.py" -v`; depois /docs no processo local PyCharm (um worker).

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
