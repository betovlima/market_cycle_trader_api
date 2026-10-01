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


## Regra de objetivo da pesquisa — 2026-09-30
- Não existe meta fixa de capital (por exemplo US$ 2M, US$ 5M ou US$ 40M).
- Objetivo primário: maximizar o capital final obtido pelas configurações candidatas, usando como referência a melhor configuração válida disponível no momento.
- Valores históricos maiores servem como evidência e contexto, não como alvo obrigatório.
- Uma alteração só conta como avanço quando aumenta o capital final sob protocolo causal/reprodutível comparável; resultados obtidos por tuning pós-OOS, seleção retrospectiva de ativos ou mudança de janela não devem ser tratados como melhoria validada.
- Métricas de risco (Sharpe, MaxDD, custos, liquidez e estabilidade por fold) permanecem diagnósticos obrigatórios, mas não substituem o objetivo primário de capital final.
- Não criar novos endpoints por experimento. Reutilizar a rota de pesquisa existente da linha quando possível e manter versões/resultados nos artefatos e documentação.

## v10.8.54 — Capital-Weighted Meta-Veto (rejeitada)
- Execução real: job `control-meta-f62da71f186b459c`.
- Paridade v10.8.44 perfeita em `US$ 1.078.635,4115518222`; paridade v10.8.53 perfeita em `US$ 1.891.417,6670329159`.
- Fold2 candidata: BA 0,496212 / AUC 0,681818 -> gate falhou.
- Fold3 candidata: BA 0,442529 / AUC 0,498084 -> gate falhou.
- Nenhum fold candidato foi habilitado, `candidate_veto_count=0`; capital final candidato = `US$ 1.078.635,4115518222`, delta `-42,9721%` contra v10.8.53.
- Hipótese rejeitada sem tuning posterior. v10.8.53 continua sendo a melhor referência válida em `US$ 1.891.417,6670329159`.

## v10.8.60 — Unanimous Temporal Veto
- Branch `feature/v10.8.60-unanimous-temporal-veto`, derivada da linha v10.8.59; melhor referência continua v10.8.57.
- Objetivo: tentar aumentar capital final reduzindo falsos positivos da v10.8.57 sem alterar componentes, features, C, gates, maturidade, one-shot ou execução.
- Referências obrigatórias no mesmo job: v10.8.44 `US$ 1.078.635,4115518222` e v10.8.57 `US$ 2.405.223,6491167042`, tolerância absoluta 1e-6.
- Componentes: exatamente os Logit temporais da v10.8.57, um por source_fold maduro, balanced L2 C=0.25, 9 features, gate BA>=0.52/AUC>=0.52.
- Única mudança experimental: agregar com `max(P_component)`. Como o veto continua `P<=0.35`, isto equivale a exigir que TODOS os componentes habilitados estejam <=0.35 para vetar.
- Fold2 tem um único componente e deve ser exatamente idêntico à v10.8.57. Qualquer diferença de equity antes do Fold3 aborta.
- Nenhum threshold, peso, C, feature, gate ou hiperparâmetro novo.
- One-shot, Liquidity-Aware, custos, execução e snapshot permanecem idênticos.
- Hipótese exploratória/post-discovery: após min() e skill-weighted piorarem a seletividade, testar unanimidade como política conservadora. Melhora retrospectiva ainda exige validação futura independente.
- Nenhum endpoint novo. Reutilizar `POST /api/admin/control-shadow/reduced-meta-veto/jobs`.
- Candidata só conta como avanço retrospectivo se superar `US$ 2.405.223,6491167042`; caso contrário, manter v10.8.57.
- Nenhuma Strategy operacional, Winner, TCC, carteira real ou ordem é alterada.

## Resultado v10.8.59 — Skill-Weighted Temporal Ensemble (rejeitada)
- Job real `control-meta-36b5b2753c0348b0`; ZIP `dados(20261001-143554).zip`, 626 entradas, CRC válido.
- Paridade v10.8.44 perfeita em `US$ 1.078.635,4115518222`; paridade v10.8.57 perfeita em `US$ 2.405.223,6491167042`; diferença pré-Fold3 exatamente 0.
- Pesos causais calculados conforme protocolo: source Fold1 `w=0,1893939394` (normalizado 46,0590%); source Fold2 `w=0,2218045113` (normalizado 53,9410%).
- Candidata terminou em `US$ 1.826.974,8329424586`, delta `-US$ 578.248,82` / `-24,0414%` contra v10.8.57.
- CAGR 132,1089% vs 142,6615%; Sharpe 1,82577 vs 1,86988; MaxDD idêntico em -45,6664%.
- Vetos totais: 67 vs 76 na v10.8.57. Vetos alinháveis ao rollout-base: candidata 34 HOLD-better / 16 ROTATE-better em 50 eventos (68,0% de precisão; delta médio -1,9621%), contra v10.8.57 35/15 em 50 eventos (70,0%; delta médio -2,0086%).
- Em estados/propostas diretamente comparáveis, a ponderação cancelou três vetos da v10.8.57: 2025-05-05 MKSI->TSLA; 2026-05-14 GKOS->AMZN; 2026-09-10 ADI->AMZN. O caso GKOS->AMZN existe no rollout-base e era veto correto: `delta_capital_fraction=-1,49037%`; a v10.8.59 elevou P de 0,310236 (mean) para 0,407430 (skill-weighted) e liberou a rotação.
- Na composição de vetos alinháveis, a candidata perdeu o veto correto GKOS->AMZN (-1,4904%) e adicionou posteriormente um veto CEF->NVDA cujo rollout mostra ROTATE melhor (+0,8323%). Isso explica a piora de seletividade observada.
- Conclusão: skill de calibração BA/AUC não deve ser convertida diretamente em peso econômico do ensemble. Hipótese rejeitada sem tuning posterior. v10.8.57 permanece melhor referência retrospectiva em `US$ 2.405.223,6491167042`.
- Não ajustar a fórmula de peso usando este mesmo OOS.

## v10.8.59 — Skill-Weighted Temporal Ensemble Meta-Veto
- Branch `feature/v10.8.59-skill-weighted-temporal-ensemble`, derivada da linha v10.8.58, mas a melhor referência continua sendo v10.8.57.
- Objetivo: tentar aumentar capital final preservando os componentes temporais da v10.8.57 e substituindo apenas o peso 50/50 por pesos causais derivados da qualidade de calibração de cada componente.
- Referências obrigatórias no mesmo job: v10.8.44 `US$ 1.078.635,4115518222` e v10.8.57 `US$ 2.405.223,6491167042`, tolerância absoluta 1e-6.
- Cada componente permanece exatamente balanced L2 Logistic Regression C=0.25, mesmas 9 features, gate BA>=0.52 e AUC>=0.52, mesma maturidade `rollout_end_date < test_start`.
- Peso pré-declarado e sem hiperparâmetro: `w = (BA_calibration - 0.5) + (AUC_calibration - 0.5)`. Como componentes só entram após BA/AUC>=0.52, todos os pesos habilitados são positivos.
- Probabilidade candidata: `sum(w_i * P_i) / sum(w_i)`. Threshold de veto permanece exatamente 0.35.
- Fold2 possui um único componente e deve ser exatamente idêntico à v10.8.57. Qualquer diferença de equity antes do Fold3 aborta o job.
- One-shot, Liquidity-Aware, custos, execução e snapshot permanecem idênticos.
- Não há tuning de peso, expoente, threshold, C, feature ou gate. A fórmula de peso foi congelada antes da execução.
- Hipótese exploratória/post-discovery; melhora retrospectiva não equivale a validação futura independente.
- Implementação concluída: `control_temporal_ensemble_meta_veto.py` passou a armazenar o skill weight causal de cada componente e suporta agregação interna `skill_weighted`; `control_skill_weighted_temporal_ensemble_research.py` reproduz v10.8.57 com mean e compara a candidata no mesmo snapshot, exigindo paridade v10.8.44, paridade exata v10.8.57 e igualdade total pré-Fold3.
- Serviço existente reapontado para `research_runner=skill-weighted-temporal-veto-v1059`, com runtime guard `API_VERSION=10.8.59`.
- Artefatos permanecem curtos para evitar o limite de path do Windows.
- Nenhum endpoint novo. Reutilizar `POST /api/admin/control-shadow/reduced-meta-veto/jobs`.
- Candidata só conta como avanço retrospectivo se superar `US$ 2.405.223,6491167042`; caso contrário, manter v10.8.57.
- Próximo passo: confirmar CI e executar o endpoint existente; auditar pesos dos componentes, vetos alterados no Fold3 e capital final.
- Nenhuma Strategy operacional, Winner, TCC, carteira real ou ordem é alterada.

## Resultado v10.8.58 — Worst-Regime Temporal Veto (rejeitada)
- Job real `control-meta-fc55cbf05e5540d9`; ZIP `dados(20261001-133742).zip`, 611 entradas, CRC válido.
- Correção de path validada: artefatos curtos foram gravados com sucesso (`base_*`, `ref57_*`, `cand58_*`, `curves.csv`, `comparison.png`).
- Paridade v10.8.44 perfeita em `US$ 1.078.635,4115518222`; paridade v10.8.57 perfeita em `US$ 2.405.223,6491167042`; diferença pré-Fold3 exatamente 0.
- Candidata `min(P_component)` terminou em `US$ 1.769.558,7211170627`, delta `-US$ 635.664,93` / `-26,4285%` contra v10.8.57.
- CAGR 130,9137% vs 142,6615%; Sharpe 1,80466 vs 1,86988; MaxDD praticamente igual (-45,6670% vs -45,6664%).
- Vetos: 84 vs 76 na v10.8.57; Fold2 permaneceu exatamente 30, Fold3 aumentou de 46 para 54.
- Vetos alinháveis ao rollout-base: candidata 41 HOLD-better / 19 ROTATE-better em 60 eventos (68,33% de precisão; delta médio -1,9429%), contra v10.8.57 35/15 em 50 eventos (70,0%; delta médio -2,0086%). Portanto o aumento de recall veio com pior precisão econômica.
- Primeira divergência no Fold3: 2024-07-24 NFLX->AVGO. v10.8.57 mean P=0,368438 e executa ROTATE; v10.8.58 min P=0,285018 e veta. Rollout-base: `delta_capital_fraction=+1,2583%`, portanto ROTATE era melhor e o novo veto foi falso positivo.
- Entre 13 novos vetos diretamente comparáveis antes de efeitos de trajetória, 9 puderam ser ligados ao rollout-base; 5 eram HOLD-better e 4 ROTATE-better. A agressividade worst-regime não melhora a seletividade.
- Hipótese rejeitada sem ajuste de threshold/agregação. v10.8.57 permanece a melhor referência retrospectiva em `US$ 2.405.223,6491167042`.
- Não tentar calibrar um percentil entre mean e min olhando este mesmo OOS.

## v10.8.58 — Worst-Regime Temporal Veto
- Branch `feature/v10.8.58-worst-regime-temporal-veto`, derivada da nova melhor referência v10.8.57.
- Objetivo: tentar aumentar o capital final recuperando parte dos HOLD-better ainda não detectados, mantendo exatamente os componentes temporais da v10.8.57.
- Referências obrigatórias no mesmo job: v10.8.44 `US$ 1.078.635,4115518222`, v10.8.53 `US$ 1.891.417,6670329159` e v10.8.57 `US$ 2.405.223,6491167042`, tolerância absoluta 1e-6.
- Única mudança experimental: no Fold3, em vez de média aritmética das probabilidades dos componentes temporais habilitados, usar `min(P_component)`. Interpretação: como o Meta-Veto é um fail-safe one-shot, se qualquer regime histórico habilitado considerar a rotação suficientemente insegura, a decisão pode ser vetada.
- Modelo de cada componente permanece exatamente Logistic Regression balanceada L2 C=0.25, mesmas 9 features, mesmo gate BA>=0.52/AUC>=0.52 por source_fold, mesma maturidade `rollout_end_date < test_start`.
- Threshold permanece 0.35. Nenhum peso, threshold, C, feature ou gate novo.
- Fold2 continua estruturalmente idêntico à v10.8.53/v10.8.57 porque existe um único componente. A candidata deve ter equity exatamente igual à v10.8.57 antes do Fold3.
- One-shot, Liquidity-Aware, custos, execução e snapshot permanecem inalterados.
- Hipótese explicitamente exploratória/post-discovery. O motivo estrutural é o diagnóstico de recall: a v10.8.53/v10.8.57 ainda deixam casos HOLD-better não vetados; a regra worst-regime testa uma política de proteção sem tuning contínuo.
- Implementação concluída: `control_temporal_ensemble_meta_veto.py` agora suporta apenas duas agregações internas congeladas (`mean` para reproduzir v10.8.57 e `min` para a candidata); `control_worst_regime_temporal_veto_research.py` executa referência v10.8.57 e candidata v10.8.58 no mesmo snapshot, exige paridade v10.8.44, paridade exata v10.8.57 e igualdade total pré-Fold3.
- Serviço existente reapontado para `research_runner=worst-regime-temporal-veto-v1058`, com runtime guard `API_VERSION=10.8.58`.
- Nenhum endpoint novo. Reutilizar `POST /api/admin/control-shadow/reduced-meta-veto/jobs`.
- A candidata só conta como avanço retrospectivo se superar `US$ 2.405.223,6491167042`; caso contrário, manter v10.8.57 como melhor referência.
- Próximo passo: confirmar CI e executar o endpoint existente; auditar número de vetos adicionais no Fold3, precisão nos rollouts alinháveis e capital final.
- Incidente de persistência no job `control-meta-1bcc6907678b4426`: o replay chegou ao fim, mas falhou ao gravar `v1057_reference_temporal_ensemble_capital_curve.csv`. O path completo no Windows tinha 264 caracteres, ultrapassando o limite clássico de 260. Não houve falha científica ou de replay.
- Correção sem mudança de estratégia: nomes dos artefatos da v10.8.58 foram encurtados (`base_*`, `ref57_*`, `cand58_*`, `curves.csv`, `comparison.png`). O maior path equivalente cai para ~233 caracteres no ambiente reportado. Teste de regressão adicionado para impedir a reintrodução do nome longo.
- CI anterior da v10.8.58 também revelou um teste legado que ainda esperava `API_VERSION=10.8.57` / `temporal-ensemble-meta-veto-v1057`; expectativa corrigida para `10.8.58` / `worst-regime-temporal-veto-v1058`. Isso não altera runtime nem protocolo científico.
- O job `control-meta-1bcc6907678b4426` é inválido apenas como entrega de artefatos incompleta; rerodar a mesma rota após atualizar/reiniciar a API. Não alterar threshold, modelos, gates ou agregação por causa deste erro.
- Nenhuma Strategy operacional, Winner, TCC, carteira real ou ordem é alterada.

## Resultado v10.8.57 — Temporal Ensemble Meta-Veto (nova melhor referência retrospectiva)
- Job real `control-meta-c0a2875df3404aa0`; ZIP `output(20261001-103811).zip`, 593 entradas, CRC válido.
- Paridade v10.8.44 perfeita em `US$ 1.078.635,4115518222`; paridade v10.8.53 perfeita em `US$ 1.891.417,6670329159`.
- Trava pré-Fold3 passou com diferença máxima de equity exatamente 0. Toda a diferença de capital veio exclusivamente do ensemble temporal no Fold3.
- Fold3 habilitou 2 componentes: source Fold1 (75 labels; BA 0,537879 / AUC 0,651515) e source Fold2 (109 labels maduros; BA 0,593985 / AUC 0,627820). Probabilidade final = média aritmética simples.
- Capital final candidato `US$ 2.405.223,6491167042`, delta `+US$ 513.805,98` / `+27,1651%` contra v10.8.53.
- CAGR 142,6615% vs 133,4135%; Sharpe 1,86988 vs 1,81203; MaxDD -45,6664% vs -45,7576% (ligeiramente melhor). Rotações 286 vs 292.
- Vetos totais: 76 vs 63 na v10.8.53; Fold2 permaneceu exatamente 30 vetos em ambas, Fold3 passou de 33 para 46.
- Entre vetos alinháveis exatamente ao rollout-base: v10.8.57 teve 35 HOLD-better / 15 ROTATE-better em 50 eventos (70,0% de precisão), contra 33/15 em 48 eventos (68,75%) da v10.8.53. DeltaCapital_fraction médio dos vetos alinháveis melhorou de -1,7049% para -2,0086%.
- Primeira divergência de policy no Fold3: 2024-07-30 NFLX->AVGO. v10.8.53 P=0,382223 (passa); ensemble P=0,337412 (componentes Fold1=0,253547 e Fold2=0,421277) e veta. O rollout-base mostra delta -15,8789%, portanto HOLD era fortemente melhor.
- Entre 7 divergências de veto em estados/propostas diretamente comparáveis antes de efeitos de trajetória, 3 puderam ser ligados exatamente ao rollout-base: duas novas decisões de veto estavam corretas (NFLX->AVGO -15,8789%; GKOS->AMZN -1,4904%) e uma estava errada (NFLX->CLMT +4,3586%).
- Conclusão: v10.8.57 supera materialmente a melhor referência comparável e passa a ser a melhor configuração retrospectiva desta linha em `US$ 2.405.223,65`. Como a hipótese é pós-descoberta no mesmo histórico, isso NÃO é validação independente futura; preservar como nova referência de pesquisa e validar futuramente antes de promoção operacional.
- Não ajustar pesos dos componentes ou threshold 0,35 usando este OOS.

## v10.8.57 — Temporal Ensemble Meta-Veto
- Branch `feature/v10.8.57-temporal-ensemble-meta-veto`, derivada da linha v10.8.56; melhor referência continua v10.8.53.
- Objetivo: aumentar capital final atacando possível drift temporal sem alterar features, C, threshold, one-shot ou execução.
- Hipótese congelada: em vez de treinar um único Logistic em todos os folds anteriores, manter um Logistic independente por fold histórico maduro e usar a média aritmética das probabilidades dos componentes habilitados.
- Modelo de cada componente: exatamente Logistic Regression balanceada L2, C=0.25, mesmas 9 features da v10.8.53.
- Gate de cada componente: split cronológico 70/30 dentro do próprio source_fold, mínimo 20 eventos de calibração, BA>=0.52 e AUC>=0.52. Componente que falhar não entra no ensemble.
- Fold2: apenas componente Fold1, portanto estruturalmente equivalente à v10.8.53. Fold3: componentes independentes Fold1 e Fold2, usando apenas labels com `rollout_end_date < test_start`; probabilidade final = média simples dos componentes habilitados.
- Regra de veto permanece `mean P(ROTATE melhor) <= 0.35`; nenhuma nova calibração de threshold.
- One-shot, Liquidity-Aware, custos e snapshot idênticos à v10.8.53.
- Referências obrigatórias no mesmo job: v10.8.44 `US$ 1.078.635,4115518222` e v10.8.53 `US$ 1.891.417,6670329159`, tolerância absoluta 1e-6.
- A hipótese é exploratória/post-discovery; qualquer melhora retrospectiva ainda exige validação futura independente.
- Implementação concluída: `control_temporal_ensemble_meta_veto.py` treina/gateia um Logistic separado por source_fold e calcula a média simples das probabilidades habilitadas; `control_temporal_ensemble_meta_veto_research.py` reproduz v10.8.44 + v10.8.53 e exige paridade exata da candidata com v10.8.53 em toda a trajetória anterior ao Fold3.
- Serviço existente reapontado para `research_runner=temporal-ensemble-meta-veto-v1057`, com runtime guard `API_VERSION=10.8.57`.
- Nenhum endpoint novo. Reutilizar `POST /api/admin/control-shadow/reduced-meta-veto/jobs`.
- Critério do experimento: candidata só conta como avanço se superar `US$ 1.891.417,6670329159`; caso contrário, descartar v10.8.57 e manter v10.8.53.
- Próximo passo: confirmar CI do HEAD e executar o endpoint existente; auditar componentes habilitados por source_fold, paridade pré-Fold3, vetos do Fold3 e capital final.
- Nenhuma Strategy operacional, Winner, TCC, carteira real ou ordem é alterada.

## Resultado v10.8.56 — Consensus Meta-Veto (rejeitada)
- Job real `control-meta-63f2e1bb18014c91`; ZIP `dados(20261001-000909).zip`, 578 entradas, CRC válido.
- Paridade v10.8.44 perfeita em `US$ 1.078.635,4115518222`; paridade v10.8.53 perfeita em `US$ 1.891.417,6670329159`.
- Candidata terminou em `US$ 1.857.794,740041506`, delta `-US$ 33.622,93` / `-1,7777%` contra v10.8.53. Sharpe 1,80585 vs 1,81203; MaxDD praticamente igual (-45,7567% vs -45,7576%).
- A candidata manteve 63 vetos totais, mas no Fold2 cancelou 3 vetos originais da v10.8.53 e a mudança de trajetória gerou vetos diferentes depois. Fold3 caiu para classifier-only porque a Ridge falhou no gate.
- Os 3 vetos cancelados foram auditados contra `paired_rollout_labels.csv` e TODOS eram vetos corretos: 2023-09-19 LKFT->RACE delta -1,7357%; 2023-09-29 LKFT->CORT delta -2,5952%; 2024-02-20 CCK->CORT delta -5,9430%. Portanto o regressor removeu ações economicamente úteis.
- Ao final do Fold2 a candidata já estava ~5,82% abaixo da v10.8.53; o fallback no Fold3 reduziu parte da diferença, mas não recuperou o capital final.
- Hipótese de consenso classificador+Ridge rejeitada. v10.8.53 permanece a melhor referência válida em `US$ 1.891.417,6670329159`.
- Diagnóstico adicional da v10.8.53: entre 48 vetos que puderam ser alinhados exatamente ao rollout-base, 33 (68,75%) eram HOLD-better e 15 eram falsos positivos. Entre 180 decisões modeladas alinháveis, havia 56 oportunidades HOLD-better não vetadas. Há espaço de melhoria, mas não via Ridge de magnitude.

## v10.8.56 — Consensus Meta-Veto
- Branch `feature/v10.8.56-consensus-meta-veto`, derivada da v10.8.55 apenas para reutilizar o regressor já congelado; a melhor referência continua sendo v10.8.53.
- Objetivo: aumentar o capital final reduzindo falsos positivos de veto da v10.8.53 sem permitir que o regressor substitua o classificador.
- Referência obrigatória no mesmo job: v10.8.44 `US$ 1.078.635,4115518222` e v10.8.53 `US$ 1.891.417,6670329159`, tolerância absoluta 1e-6.
- Classificador primário: exatamente v10.8.53 (Logistic Regression balanceada L2 C=0.25, mesmas 9 features, gate BA>=0.52 e AUC>=0.52, veto candidato quando P(ROTATE melhor)<=0.35).
- Confirmação secundária: exatamente o regressor v10.8.55 (Ridge alpha=1.0, mesmas 9 features, gate sign BA>=0.52 e Spearman>0).
- Regra congelada: se o classificador NÃO pedir veto, executar Control. Se pedir veto e o regressor estiver habilitado no fold, só vetar quando `predicted_delta_capital_fraction<=0`. Se o regressor estiver desabilitado, preservar exatamente o veto da v10.8.53 (fallback classifier-only).
- Fold1 sem modelo; Fold2/Fold3 somente labels maduros com `rollout_end_date < test_start`. One-shot e Liquidity-Aware idênticos à v10.8.53.
- Não há threshold novo, tuning de C/alpha, seleção de features ou ajuste pós-OOS.
- Esta hipótese é exploratória e pós-descoberta: o consenso foi escolhido após observar que a regressão v10.8.55 falhou como substituta. Só deve ser tratada como melhoria retrospectiva se aumentar capital; validação futura independente continua necessária.
- Implementação concluída: `control_consensus_meta_veto.py` combina o classificador v10.8.53 com o regressor v10.8.55 sem permitir novos vetos; `control_consensus_meta_veto_research.py` reproduz v10.8.44 + v10.8.53 e compara a candidata no mesmo snapshot; o serviço existente foi reapontado para essa orquestração.
- Nenhum endpoint novo. Reutilizar `POST /api/admin/control-shadow/reduced-meta-veto/jobs`.
- Critério do experimento: candidata só conta como avanço se superar `US$ 1.891.417,6670329159`. Se não superar, descartar v10.8.56 e manter v10.8.53.
- Próximo passo: confirmar CI e executar o endpoint existente; auditar vetos confirmados, vetos cancelados, fallback classifier-only por fold e capital final.
- Incidente de runtime local em 2026-09-30: o ZIP `dados(20260930-235324).zip` não contém artefatos v10.8.56. A execução mais recente no horário do arquivo foi `control-meta-22a6b2af938f406f` sob `validation/v10.8.54`, com `research_kind=control_capital_weighted_meta_veto`; portanto o processo local que atendeu a requisição ainda estava com runner v10.8.54 carregado. Isso NÃO é resultado v10.8.56.
- Proteção adicionada ainda na v10.8.56 antes de qualquer execução válida: job público agora expõe `api_version` e `research_runner=consensus-meta-veto-v1056`; o serviço também aborta se `API_VERSION != 10.8.56`. Após trocar branch/pull, reiniciar explicitamente o processo Uvicorn antes de executar a pesquisa.
- Nenhuma Strategy operacional, Winner, TCC, carteira real ou ordem é alterada.

## Resultado v10.8.55 — Expected Advantage Regression (rejeitada)
- Job real `control-meta-46ab5ebc5a6c4a6d`; ZIP `dados(20260930-233332).zip`, 548 entradas, CRC válido.
- Paridade v10.8.44 perfeita em `US$ 1.078.635,4115518222`; paridade v10.8.53 perfeita em `US$ 1.891.417,6670329159`.
- Fold2 regressão: 75 labels maduros; calibration sign BA 0,575758; Spearman 0,356719; modelo habilitado.
- Fold3 regressão: 184 labels maduros; calibration sign BA 0,405492; Spearman -0,199658; gate falhou e modelo foi desabilitado.
- Candidata aplicou 57 vetos, todos no Fold2. Capital final `US$ 1.038.414,3657602259`, delta `-US$ 853.003,30` / `-45,0986%` contra v10.8.53. Sharpe 1,6594; MaxDD -42,61%.
- Ao fim do Fold2, antes do Fold3, a candidata já estava materialmente abaixo da referência; o problema não foi apenas a desativação do Fold3.
- Hipótese de substituir o classificador pelo regressor foi rejeitada sem tuning de alpha/threshold. v10.8.53 permanece a melhor referência válida.
- Diagnóstico para próxima hipótese: usar o regressor apenas como confirmação secundária dos vetos do classificador, nunca como substituto. No Fold2, dos 30 vetos da v10.8.53, 20 ocorreram em datas/estados nos quais o regressor também indicou vantagem econômica não positiva; o Fold3 deve cair automaticamente para a v10.8.53 porque o regressor não passou no gate.

## v10.8.55 — Expected Advantage Regression Meta-Veto
- Branch `feature/v10.8.55-expected-advantage-regression-meta-veto`, derivada diretamente da melhor referência v10.8.53; a v10.8.54 foi rejeitada e não é base desta versão.
- Objetivo: aumentar capital final modelando diretamente `delta_capital_fraction` do rollout ROTATE-vs-HOLD, em vez de classificar apenas seu sinal.
- Referências obrigatórias no mesmo job: v10.8.44 `US$ 1.078.635,4115518222` e v10.8.53 `US$ 1.891.417,6670329159`, tolerância absoluta 1e-6.
- Modelo candidato congelado antes da execução: Ridge Regression com `alpha=1.0`, `SimpleImputer(median)` + `StandardScaler`; mesmas 9 features da v10.8.53.
- Target: `delta_capital_fraction = (capital_ROTATE - capital_HOLD) / initial_equity` já congelado na v10.8.49/v10.8.48.
- Gate de calibração pré-declarado: mínimo 20 eventos, Balanced Accuracy do sinal previsto >=0.52 e Spearman(predição, delta real) > 0.0. Calibração permanece cronológica 70/30, sem peso.
- Regra de veto natural, sem threshold calibrável: se o modelo do fold estiver habilitado e `predicted_delta_capital_fraction <= 0.0`, HOLD one-shot; caso contrário, Control.
- Fold1 sem modelo; Fold2 usa somente rollouts maduros do Fold1; Fold3 somente rollouts maduros dos Folds1+2; `rollout_end_date < test_start`.
- Semântica one-shot e Liquidity-Aware permanecem idênticas à v10.8.53.
- Nenhum grid search, tuning de alpha, tuning de threshold, seleção de feature ou peso após observar o resultado.
- Implementação concluída: `control_expected_advantage_meta_veto.py` treina Ridge por fold e aplica veto no break-even econômico 0; `control_expected_advantage_meta_veto_research.py` reproduz v10.8.44 + v10.8.53 e compara a candidata no mesmo snapshot; o serviço existente `control_shadow_reduced_meta_veto_jobs.py` foi reapontado para essa orquestração.
- Nenhum endpoint novo. Reutilizar `POST /api/admin/control-shadow/reduced-meta-veto/jobs` com o mesmo payload.
- Candidata só conta como avanço se superar o capital final da referência v10.8.53 sob paridade exata.
- Próximo passo: confirmar CI do HEAD e executar o endpoint existente; auditar gates de regressão, quantidade de vetos e capital final sem ajustar parâmetros após observar o resultado.
- Nenhuma Strategy operacional, Winner, TCC, carteira real ou ordem é alterada.

## v10.8.53 — One-shot Meta-Veto + maturidade causal estrita
- Branch `fix/v10.8.53-one-shot-meta-veto-causal-maturity`, derivada da v10.8.52 após auditoria do job `control-meta-da56431b92294ebd`.
- A v10.8.52 corrigiu corretamente a paridade Liquidity-Aware: baseline reproduziu exatamente `US$ 1.078.635,4115518222` (diferença 0). Porém o Meta-Veto terminou em `US$ 109.239,83562554276`, delta `-89,8724%`, com 734 vetos em 809 decisões modeladas.
- Causa principal confirmada: o target original v10.8.48 é `HOLD uma vez, depois voltar ao Control`, mas a v10.8.51/52 permitia novo veto imediatamente na decisão seguinte. Isso gerou lock-in recursivo; no Fold 2 houve sequência de 333 vetos consecutivos e outras sequências longas.
- Correção sem tuning: após aplicar um veto, a decisão seguinte é obrigatoriamente entregue ao Control sem possibilidade de novo veto. O cooldown é consumido em exatamente uma chamada de policy. Isso implementa literalmente o branch de treino `HOLD once, then Control`.
- Segunda correção causal: Fold3 da v10.8.51 usou 191 labels anteriores, mas somente 184 tinham `rollout_end_date < test_start`. Sete rollouts do fim do Fold2 maturavam já dentro do Fold3. A v10.8.53 exige maturidade integral do rollout antes do início do fold de teste, igual ao protocolo v10.8.48.
- Implicação retroativa: a avaliação Fold3 das v10.8.49/v10.8.50 também deve ser tratada como exploratória, pois a seleção/treino por `source_fold_id` não excluía explicitamente esses 7 labels ainda não maduros. A v10.8.50 deixa de ser considerada confirmação causal independente; ela permanece como evidência exploratória que motivou a hipótese reduzida. A v10.8.53 é a primeira execução desta linha que restaura a regra de maturidade do target de 20 sessões no treino do Meta-Veto.
- Fonte de maturidade: `paired_rollout_labels.csv` da v10.8.48, unido ao dataset v10.8.49 por decision_date/incumbent/candidate. Fold2 permanece com 75 labels elegíveis; Fold3 passa de 191 para 184.
- Modelo, 9 features, C=0.25, split cronológico 70/30, gate BA>=0.52, AUC>=0.52 e veto P(ROTATE melhor)<=0.35 permanecem congelados. Nenhum threshold ou feature foi ajustado após observar o capital ruim.
- A v10.8.52 é válida para diagnóstico de falha/paridade, mas seu capital Meta-Veto NÃO representa a política-alvo devido à repetição indevida de veto e à maturidade causal incompleta.
- Implementação concluída: `control_reduced_signature_meta_veto.py` exige `rollout_end_date < test_start` e aplica estado one-shot `force_control_next`; `control_reduced_signature_meta_veto_research.py` une os 321 eventos ao `paired_rollout_labels.csv` da v10.8.48 para recuperar maturidade exata e grava artefatos sob `validation/v10.8.53/`.
- O endpoint permanece `POST /api/admin/control-shadow/reduced-meta-veto/jobs`, sem login e com o mesmo payload. Não foram adicionados parâmetros HTTP.
- Testes atualizados para exigir maturidade integral, one-shot explícito, Liquidity-Aware habilitado e API v10.8.53.
- Correção de importação em 2026-09-30: patch anterior inseriu 12 ocorrências de `\\n` literal em `control_reduced_signature_meta_veto.py`, causando `SyntaxError` no startup do Uvicorn/Python 3.14. Todas foram convertidas para quebras de linha reais no commit `5cc5e33860a288170f585df8f0ba935225a60db3`. Nenhuma regra científica foi alterada.
- Execução real v10.8.53 auditada no job `control-meta-f4ed26129d014c6f`, ZIP `output(20260930-221443).zip` com 518 entradas e CRC válido.
- Paridade v10.8.44 perfeita: esperado/reproduzido `US$ 1.078.635,4115518222`, diferença absoluta 0.
- Maturidade causal correta: Fold2 = 75 labels elegíveis; Fold3 = 184. Gate Fold2: BA 0,537879 / AUC 0,651515; Gate Fold3: BA 0,590677 / AUC 0,583653; ambos habilitados.
- Semântica one-shot validada: 63 vetos totais, 63 decisões seguintes marcadas `CONTROL_AFTER_ONE_SHOT_VETO`, nenhum veto consecutivo (streak máximo = 1), nenhum veto em transição CASH e todo veto manteve o incumbent. Foram 30 vetos no Fold2 e 33 no Fold3, 63/233 = 27,04% das decisões efetivamente modeladas.
- Resultado de capital: baseline `US$ 1.078.635,41`; Meta-Veto `US$ 1.891.417,67`; delta `+US$ 812.782,26` / `+75,3528%`. CAGR subiu de 113,15% para 133,41%; Sharpe de 1,6686 para 1,8120. MaxDD piorou de -42,61% para -45,76%.
- A vantagem surgiu em ambos os períodos com modelo ativo: ao fim do Fold2 a trajetória Meta estava 61,54% acima da baseline; no Fold3 o multiplicador relativo ainda aumentou cerca de 8,55% sobre a vantagem carregada.
- Rotations caíram de 320 para 292, mas custos/fees absolutos aumentaram com o capital maior (`modeled_price_cost` ~US$220,1k vs US$138,3k; fees ~US$2,91k vs US$1,93k).
- Conclusão científica: a v10.8.53 é o primeiro replay desta linha com paridade, one-shot literal e maturidade causal do target de 20 sessões corretamente implementadas. O resultado retrospectivo é forte, porém NÃO constitui novo OOS independente porque as mesmas janelas históricas participaram da descoberta da assinatura v10.8.49/v10.8.50. Congelar a policy v10.8.53 e exigir validação futura/independente antes de qualquer promoção operacional.
- CI dos commits de correção de sintaxe/importação passou; não criar novo endpoint para versões seguintes desta linha. Reutilizar `/api/admin/control-shadow/reduced-meta-veto/jobs`.


## v10.8.52 — Correção de paridade Liquidity-Aware no Meta-Veto
- Branch `fix/v10.8.52-reduced-meta-veto-liquidity-parity`, derivada da v10.8.51 após falha real do job `control-meta-c396c6c3d802488f`.
- Erro observado: paridade esperada v10.8.44 `US$ 1.078.635,4115518222`, mas replay reproduziu `US$ 528.709,776437140652`, exatamente o benchmark v10.8.42 execution-constrained.
- Causa raiz confirmada: `_CapitalAwareUtilityCache.get()` só aplica o overlay quando `account["enabled"]` é verdadeiro. A v10.8.51 inicializou `account = {}`, então o wrapper existia mas retornava as utilities originais, reproduzindo v10.8.42.
- Correção mínima: inicializar o contexto em `run_reduced_signature_meta_veto_pair` como `{"enabled": True, "audit": {}}`, igual ao contrato funcional necessário para o Liquidity-Aware. Nenhuma feature, C, gate, threshold, fold protocol ou regra de veto foi alterada.
- API incrementada para `10.8.52`. Teste dedicado adicionado para impedir regressão da ativação explícita do overlay.
- O job v10.8.51 que falhou NÃO produziu resultado científico válido e não deve ser usado em comparação de capital.
- Próximo passo: confirmar CI da v10.8.52 e repetir o mesmo endpoint/payload do Meta-Veto. A nova execução deve primeiro reproduzir exatamente `US$ 1.078.635,4115518222`; qualquer divergência continua abortando o experimento.

## v10.8.51 — Reduced Signature Meta-Veto (pesquisa, sem ordem)
- Branch `feature/v10.8.51-reduced-signature-meta-veto-research`, derivada da v10.8.50 confirmada.
- Objetivo: medir impacto de capital de um Meta-Veto fail-safe que usa exclusivamente a assinatura reduzida confirmada na v10.8.50 para decidir se uma rotação ativo->ativo proposta pelo Control deve ser mantida como HOLD por uma decisão.
- Fonte congelada de treino: dataset causal v10.8.49 `control-signature-5dedf5eee0f94e8a`, confirmação v10.8.50 `control-reduced-b9404f6e1f694e38`, rollout `control-rollout-03253d310ef445b2`, snapshot SHA `6d9e7d69865277487a6b193adedcc1d91e428ab451921d538aa4938fe108e8f3`.
- Modelo congelado: Logistic Regression balanceada L2, `C=0.25`, exatamente as 9 features da v10.8.50; nenhuma seleção/tuning posterior.
- Features: `incumbent__return_20`, `incumbent__return_60`, `incumbent__ema_distance_20`, `incumbent__ema_distance_50`, `incumbent__rsi_14`, `incumbent__channel_position_50`, `incumbent_capacity_equity_ratio`, `state_shares`, `candidate__channel_position_50`.
- Protocolo causal congelado: Fold1 sempre sem meta-modelo; Fold2 aprende somente com labels do Fold1; Fold3 aprende somente com labels dos Folds1+2. Dentro do histórico elegível, split cronológico 70/30 para treino/calibração; após gate, refit em todo histórico anterior.
- Gate fail-safe pré-declarado: pelo menos 20 amostras de calibração, Balanced Accuracy >= 0.52 e ROC AUC >= 0.52. Se qualquer requisito falhar, o fold inteiro executa Control puro.
- Regra de veto pré-declarada: somente rotação ativo->ativo, modelo habilitado e `P(ROTATE melhor) <= 0.35`; então HOLD do incumbent por uma decisão. CASH->ativo, ativo->CASH e HOLD nunca são alterados. Threshold não será ajustado após observar capital.
- Features durante o replay são calculadas causalmente no estado REAL da trajetória meta naquele decision_date; o dataset de treino permanece sempre o baseline v10.8.49, evitando retroalimentação de labels gerados por vetos.
- Comparação obrigatória: baseline deve reproduzir exatamente a v10.8.44 Liquidity-Aware `US$ 1.078.635,4115518222` (tolerância absoluta 1e-6). O novo resultado só é interpretável depois dessa paridade.
- Implementação concluída: `engine/control_reduced_signature_meta_veto.py` treina/gateia o Logistic reduzido por fold e calcula features sobre o estado real da trajetória meta; `engine/control_reduced_signature_meta_veto_research.py` executa baseline+meta e exige paridade v10.8.44; `services/control_shadow_reduced_meta_veto_jobs.py` cria job isolado; Swagger expõe `POST /api/admin/control-shadow/reduced-meta-veto/jobs` e GET de status/logs, sem login.
- Artefatos previstos: `summary.json`, curvas/fills baseline e meta, `meta_training_folds.csv`, `meta_veto_decisions.csv`, `aligned_capital_curves.csv` e `paired_capital.png`.
- Próximo passo: confirmar CI do HEAD, executar a v10.8.51 usando `control-reduced-b9404f6e1f694e38` e auditar gate por fold, número de vetos, paridade e impacto de capital sem alterar nenhum parâmetro após observar o resultado.
- Nenhuma Strategy operacional, Winner, TCC, carteira real ou ordem é alterada. `order_eligible=false`, `order_submission=never`.

## v10.8.50 — Reduced Rollout Signature Confirmation (diagnóstico, sem ordem)
- Branch `feature/v10.8.50-reduced-rollout-signature-research`, derivada da v10.8.49 após auditoria do primeiro sinal preditivo preliminar.
- Objetivo: verificar se o sinal da Logistic Regression v10.8.49 sobrevive com assinatura pequena, explicável e congelada, reduzindo dimensionalidade/colinearidade antes de qualquer Meta-Veto.
- Fonte congelada: v10.8.49 `control-signature-5dedf5eee0f94e8a`, baseada no rollout v10.8.48 `control-rollout-03253d310ef445b2`, snapshot SHA `6d9e7d69865277487a6b193adedcc1d91e428ab451921d538aa4938fe108e8f3`.
- Features congeladas antes do resultado: `incumbent__return_20`, `incumbent__return_60`, `incumbent__ema_distance_20`, `incumbent__ema_distance_50`, `incumbent__rsi_14`, `incumbent__channel_position_50`, `incumbent_capacity_equity_ratio`, `state_shares`, `candidate__channel_position_50`.
- Modelos fixos: Logistic Regression balanceada C=1.0; Logistic Regression L2 mais regularizada C=0.25; e nove modelos logísticos univariados (uma feature por vez). Threshold fixo 0.50; sem grid search, feature selection pós-OOS ou tuning.
- Validação temporal permanece: Fold2 <- Fold1; Fold3 <- Folds1+2.
- Critério de confirmação pré-declarado e mais rígido: o mesmo modelo multivariado reduzido precisa ter Balanced Accuracy > 0.50 E ROC AUC > 0.50 nos Folds 2 e 3. Modelos univariados são apenas explicativos e não contam isoladamente como confirmação.
- Implementação concluída: `engine/control_reduced_rollout_signature.py` avalia a assinatura congelada; `engine/control_reduced_rollout_signature_research.py` orquestra fonte/artefatos; `services/control_shadow_reduced_signature_jobs.py` cria job admin isolado; Swagger expõe `POST /api/admin/control-shadow/reduced-signature/jobs` e GET de status/logs.
- Artefatos: `summary.json`, `reduced_model_results.csv`, `reduced_logistic_coefficients.csv`, `coefficient_stability.csv`.
- Nenhuma policy é criada nesta versão. Mesmo se o sinal for confirmado, o próximo passo seria desenhar separadamente uma política fail-safe e congelar seu protocolo antes de avaliar capital.
- Execução/auditoria v10.8.50 concluída em 2026-09-30: job `control-reduced-b9404f6e1f694e38`, fonte `control-signature-5dedf5eee0f94e8a`, rollout `control-rollout-03253d310ef445b2`, snapshot `control-shadow-870fb66e1bdc4fd0`, SHA `6d9e7d69865277487a6b193adedcc1d91e428ab451921d538aa4938fe108e8f3`. ZIP `dados(10).zip` com 496 entradas e CRC válido.
- A assinatura reduzida foi CONFIRMADA segundo o critério pré-declarado. Logistic C=1.0: Fold2 balanced accuracy 0,557984 / AUC 0,606601; Fold3 balanced accuracy 0,555383 / AUC 0,590594. Logistic C=0.25: Fold2 balanced accuracy 0,594558 / AUC 0,609277; Fold3 balanced accuracy 0,542731 / AUC 0,586775. Ambos excederam 0,50 em BA e AUC nos dois testes temporais.
- O modelo C=0.25 foi mais equilibrado no Fold2 (accuracy 0,594828, precision 0,589286, recall 0,578947, predicted_positive_rate 0,482759, Brier 0,242402). No Fold3, C=1.0 teve BA levemente maior (0,555383 vs 0,542731), mas ambos mantiveram AUC ~0,59.
- Todos os 9 modelos univariados também ficaram acima de 0,50 em BA e AUC nos dois folds, embora por protocolo NÃO sejam elegíveis isoladamente para confirmar a hipótese. Os sinais univariados foram consistentes entre folds: `state_shares` negativo; as demais oito features positivas.
- Estabilidade multivariada: em C=0.25, 7/9 coeficientes preservaram sinal entre Fold2 e Fold3: `state_shares` negativo; `incumbent_capacity_equity_ratio`, `incumbent__return_60`, `incumbent__return_20`, `candidate__channel_position_50`, `incumbent__ema_distance_50`, `incumbent__rsi_14` positivos. `incumbent__ema_distance_20` e `incumbent__channel_position_50` inverteram sinal no modelo multivariado, sugerindo colinearidade/instabilidade condicional apesar do sinal univariado positivo.
- Conclusão v10.8.50: a hipótese de que existe uma assinatura preditiva pequena e explicável no estado da decisão está suportada preliminarmente por dois testes cronológicos independentes dentro do mesmo histórico OOS. Isso ainda NÃO autoriza produção; o próximo passo deve ser desenhar e pré-declarar separadamente um Meta-Veto fail-safe que use a assinatura confirmada e então medir impacto de capital, sem tuning sobre este OOS.
- Alteração de acesso em 2026-09-30: o `control_shadow.router` deixou de herdar `require_admin_session` no `main.py`. Todos os endpoints de pesquisa sob `/api/admin/control-shadow/*` passam a poder ser acessados sem login. As demais rotas administrativas da API continuam protegidas normalmente.
- A remoção de login NÃO remove as travas de pesquisa: confirmações literais, `_require_enabled`, validação de cadeia de jobs/SHA, ausência de campos de conta/Winner e contratos `order_eligible=false` / `order_submission="never"` permanecem.
- Teste `test_control_shadow_api.py` atualizado para exigir registro público do Control Shadow e impedir regressão que reanexe `dependencies=admin_required`.
- Nenhuma Strategy operacional, Winner, TCC, carteira real ou ordem é alterada.

## v10.8.49 — Rollout Decision Signature Research (diagnóstico, sem ordem)
- Branch `feature/v10.8.49-rollout-decision-signature-research`, derivada da v10.8.48.
- Objetivo: explicar os 321 eventos contrafactuais da v10.8.48 e descobrir se variáveis conhecidas no instante da decisão conseguem separar `ROTATE melhor` de `HOLD melhor`.
- Esta versão NÃO cria nova policy. É exclusivamente diagnóstica e não pode alterar Control, Liquidity-Aware, Winner, Strategy, Trader ou ordens.
- Fonte congelada: job v10.8.48 `control-rollout-03253d310ef445b2`, snapshot `control-shadow-870fb66e1bdc4fd0`, SHA `6d9e7d69865277487a6b193adedcc1d91e428ab451921d538aa4938fe108e8f3`.
- Dataset alvo: 321 linhas de `paired_rollout_labels.csv`, enriquecidas apenas com variáveis disponíveis em `decision_date`: estado da carteira, features causais incumbent/candidate, diferenças relativas, liquidez/capacidade estimada e, quando disponível sem recomputar futuro, diagnostics/utility do Control.
- Análises fixas antes de olhar resultado: Pearson/Spearman, comparação ROTATE vs HOLD, quantis, estabilidade por fold; modelos simples Logistic Regression, árvore rasa, Random Forest pequeno e LightGBM pequeno com validação temporal.
- Validação temporal: Fold 2 avaliado com treino somente em Fold 1; Fold 3 avaliado com treino somente em Folds 1+2. Fold 1 é apenas histórico/diagnóstico e nunca é usado como avaliação após treino futuro.
- Sem tuning em OOS: hiperparâmetros pequenos e fixos; nenhuma busca de grade; nenhuma alteração baseada no resultado da mesma v10.8.49.
- Implementação concluída na branch: `engine/control_rollout_decision_signature.py` monta dataset causal e análises; `engine/control_rollout_signature_research.py` orquestra fonte/artefatos; `services/control_shadow_rollout_signature_jobs.py` cria job isolado; Swagger expõe `POST /api/admin/control-shadow/rollout-signature/jobs` e GET de status/logs.
- Variáveis implementadas: estado da carteira quando recuperável na curva-base, 12 features causais do incumbent, 12 do candidate, 12 deltas relativos, preços, volume mediano anterior de 20 sessões, razão de liquidez, capacidade estimada a 10%, capacidade nocional e capacidade/equity.
- Modelos fixos sem tuning: Logistic Regression, árvore rasa, Random Forest pequeno e LightGBM pequeno. Avaliação temporal pré-declarada: Fold2 <- Fold1; Fold3 <- Folds1+2. Critério preliminar de sinal: balanced accuracy > 0,50 nos dois testes cronológicos para o mesmo modelo.
- Artefatos: `rollout_decision_dataset.csv`, `feature_correlations.csv`, `feature_class_comparison.csv`, `feature_quantiles.csv`, `fold_feature_stability.csv`, `model_results.csv`, `feature_importance.csv`, `summary.json`.
- Execução/auditoria real v10.8.49 concluída sobre job `control-signature-5dedf5eee0f94e8a`, fonte `control-rollout-03253d310ef445b2`, SHA `6d9e7d69865277487a6b193adedcc1d91e428ab451921d538aa4938fe108e8f3`. ZIP analisado com 490 entradas e CRC válido; 321 eventos, 157 ROTATE melhor e 164 HOLD melhor.
- Pela primeira vez um modelo simples cumpriu o critério predeclarado de sinal preliminar nos dois testes temporais: Logistic Regression balanced accuracy Fold2 = 0,516949 e Fold3 = 0,547267. AUC: Fold2 = 0,495837; Fold3 = 0,578658. Portanto `predictive_signal_detected=true` segundo a regra definida antes do resultado.
- O sinal é fraco e NÃO autoriza policy: no Fold2 a Logistic Regression teve recall 1,0 e precision 0,50, indicando comportamento quase sempre ROTATE; Brier = 0,483135. Fold3 foi melhor, com balanced accuracy 0,547267 e AUC 0,578658, mas ainda modesto.
- Outros modelos não mantiveram >0,50 nos dois folds: Decision Tree (0,500000 / 0,424445), Random Forest (0,538507 / 0,496419), LightGBM pequeno (0,506096 / 0,472547).
- Features descritivas com sinal Spearman consistente nos 3 folds incluem principalmente estado/força do incumbent: `state_shares` (negativo), `incumbent__return_20`, `incumbent__return_60`, `incumbent__ema_distance_20`, `incumbent__ema_distance_50`, `incumbent__rsi_14`, `incumbent__channel_position_50`, `incumbent_close`, `incumbent_capacity_notional`, `incumbent_capacity_equity_ratio`, além de `candidate__channel_position_50`.
- Nos 321 eventos, `incumbent__return_20` teve Spearman ~0,168 com DeltaCapital; `incumbent__ema_distance_50` ~0,153; `incumbent__return_60` ~0,134. ROTATE-better apresentou incumbent com momentum/posição técnica mais forte em média. Esse padrão é descritivo e precisa de confirmação causal/preditiva adicional.
- Conclusão v10.8.49: existe um primeiro sinal temporalmente transferível, mas insuficiente para criar Meta-Veto. Próxima pesquisa deve focar uma representação explicável do estado do incumbent e reduzir dimensionalidade/colinearidade, mantendo validação Fold2<-Fold1 e Fold3<-Folds1+2; não fazer tuning sobre este mesmo OOS nem promover Strategy/Winner.
- Nenhuma Strategy operacional, Winner, TCC, carteira real ou ordem é alterada.

## v10.8.48 — Control Policy Rollout Advantage (pesquisa, sem ordem)
- Branch `feature/v10.8.48-control-policy-rollout-advantage-research`, derivada da v10.8.47 após auditoria do resultado real.
- Motivação: v10.8.47 preservou corretamente a v10.8.44 porque nenhum fold atingiu skill mínima, mas o target simplificado `weighted_forward_return(candidate)-weighted_forward_return(incumbent)` não apresentou skill OOS. A arquitetura Meta-Veto é preservada; apenas o target muda.
- Hipótese congelada antes do novo resultado: aprender `DeltaCapital = capital(ROTATE agora + mesma policy depois) - capital(HOLD uma vez + mesma policy depois)`, usando dois rollouts com o MESMO estado inicial, MESMA policy Control/Liquidity-Aware, MESMAS restrições de execução e horizonte fixo de 20 sessões.
- Protocolo causal: Fold 1 não usa meta-modelo por não existir OOS anterior intocado. Fold 2 aprende somente com oportunidades da Fold 1. Fold 3 aprende somente com oportunidades das Folds 1–2. Dentro do histórico anterior, treino/calibração são separados cronologicamente. Nenhum label do fold corrente pode treinar ou habilitar o modelo desse mesmo fold.
- Somente rotações ativo->ativo propostas pelo Control geram label e podem ser vetadas. CASH->ativo, ativo->CASH e HOLD permanecem Control.
- O gate fail-safe permanece: sem amostras/skill/confiança suficiente, executar exatamente a decisão do Control. Não haverá tuning HTTP nem ajuste após olhar o novo OOS.
- Implementação concluída na branch: `simulate_feasible_control` aceita `initial_state` opcional e reconciliado; `engine/control_policy_rollout_advantage.py` gera rollouts pareados de 20 sessões; `engine/control_policy_rollout_research.py` exige paridade v10.8.44 e grava artefatos; `services/control_shadow_policy_rollout_jobs.py` cria job isolado; Swagger expõe `POST /api/admin/control-shadow/policy-rollout/jobs` e GET de status/logs.
- Cadeia real congelada para a execução: v10.8.41 `control-validation-7821002400424ccc`; v10.8.42 `control-execution-c38169f6c6fc4ea0`; v10.8.44 `control-liquidity-dbe4ec6f52c74694`; v10.8.45 `control-tcn-b7d0247118314107`; v10.8.46 `control-rank-c67e91abb3914f26`; v10.8.47 `control-advantage-53fc67bd758f443a`; SHA `6d9e7d69865277487a6b193adedcc1d91e428ab451921d538aa4938fe108e8f3`.
- Testes adicionados para estado histórico reconciliado, rejeição fail-closed, Fold 1 obrigatoriamente sem meta-modelo, exclusão de labels do fold corrente e bloqueio de parâmetros HTTP de tuning.
- Os labels de folds posteriores são sempre gerados sobre a trajetória-base v10.8.44; estados criados pelos próprios vetos v10.8.48 não retroalimentam o treinamento nesta hipótese, evitando dependência recursiva entre folds.
- Execução real v10.8.48 concluída com sucesso em 2026-09-30: job `control-rollout-03253d310ef445b2`. O log chegou até fold 3/3, replay OOS 1554 sessões, geração de rollouts pareados sobre 326 oportunidades e status final `completed`, sem tocar no caminho de ordens.
- Auditoria v10.8.48 concluída sobre o ZIP real `dados(8).zip`: 456 entradas, CRC válido e 167/167 hashes do snapshot conferidos contra o manifesto; SHA permanece `6d9e7d69865277487a6b193adedcc1d91e428ab451921d538aa4938fe108e8f3`.
- Paridade v10.8.44 perfeita: esperado e reproduzido `US$ 1.078.635,4115518222`, diferença absoluta 0.
- Foram gerados 321 labels de rollout pareado com horizonte fixo de 20 sessões; 157 (48,91%) favoreceram ROTATE e 164 favoreceram HOLD. Distribuição por fold-base: fold1 75 labels, 54,67% positivos; fold2 116, 49,14%; fold3 130, 45,38%.
- Skill do meta-modelo: Fold 1 desabilitado por protocolo; Fold 2 balanced accuracy de calibração 44,44% (30 amostras); Fold 3 46,23% (56 amostras). Nenhum fold atingiu o gate de 55%, logo `enabled_fold_count=0` e `veto_count=0`.
- Capital v10.8.48 = `US$ 1.078.635,4115518222`, exatamente igual à v10.8.44, por fallback integral ao Control. A arquitetura fail-safe voltou a funcionar corretamente.
- O target DeltaCapital por rollout é economicamente expressivo, com diferenças extremas de aproximadamente -33,25% a +30,93% da equity inicial em 20 sessões, porém a TinyTCN atual não conseguiu prever o sinal dessas diferenças de forma útil. A hipótese de que apenas substituir o label local por rollout de capital resolveria o problema preditivo está rejeitada nesta implementação.
- Próximo passo de pesquisa não deve ser tuning da mesma TinyTCN/threshold/horizonte sobre este OOS. Preservar o Control/Liquidity-Aware e investigar representação/variáveis específicas do estado de decisão ou modelos mais simples/interpretáveis sobre os 321 eventos antes de qualquer nova rede.
- Nenhuma Strategy operacional, Winner, TCC, carteira real ou ordem é alterada.

## v10.8.47 — Control Counterfactual Advantage Meta-Veto (pesquisa, sem ordem)
- Branch `feature/v10.8.47-control-counterfactual-advantage-research`, derivada diretamente da v10.8.46. O Control/Liquidity-Aware permanece a ação padrão; o novo modelo não rankeia os 55 ativos e só pode vetar rotações ativo->ativo.
- Primeiro incremento implementado em `engine/control_counterfactual_advantage.py`: target pareado candidato-vs-incumbent, horizontes 5/10/20/40/60 com pesos científicos, maturidade integral de 60 sessões, features relativas em janela causal de 40 sessões, TinyTCN binária CPU, calibração cronológica, gate de skill e fallback explícito para Control.
- Regra fail-safe congelada antes do novo OOS: pelo menos 30 amostras de calibração e balanced accuracy >= 0,55 para habilitar o modelo; somente probabilidade de vantagem <= 0,40 pode vetar. CASH->ativo, ativo->CASH, HOLD, ausência de score ou skill insuficiente executam exatamente o Control.
- O target deste primeiro incremento é vantagem local de ação por retorno futuro ponderado candidato menos incumbent. Não chamar de DeltaCapital exato de carteira; rollout estado-dependente de capital permanece extensão posterior.
- Testes em `tests/test_control_counterfactual_advantage.py` cobrem maturidade temporal, ausência de vazamento de features futuras, domínio restrito do veto e fallback Control.
- Integração concluída no branch: o mesmo treinamento LightGBM gera policies/caches usados em dois replays independentes, primeiro v10.8.44 Liquidity-Aware e depois v10.8.47 Meta-Veto. O job aborta se o baseline não reproduzir exatamente o capital persistido da v10.8.44 (tolerância absoluta US$ 1e-6).
- Endpoint administrativo: `POST /api/admin/control-shadow/counterfactual-advantage/jobs`, confirmação `RESEARCH_CONTROL_COUNTERFACTUAL_ADVANTAGE_NO_ORDERS`, exigindo IDs exatos v10.8.41/42/44/45/46 e mesmo SHA. GET de status/logs no mesmo prefixo. Sem parâmetros HTTP de tuning.
- Artefatos em `validation/v10.8.47/<job_id>`: summary, curvas/fills baseline e meta-veto, training folds, decisões/probabilidades/vetos, folds de capital, curvas alinhadas e PNG.
- CI do HEAD de código `f33b8354cac27aa15642ed206f8fec61acb8b99f` passou completa em Python 3.12 após atualizar os contratos de rotas e API_VERSION para 10.8.47. Próximo passo: executar localmente pelo /docs e analisar os artefatos; não interpretar capital v10.8.47 antes do job real. Nenhuma Strategy operacional, Winner, TCC, carteira ou ordem foi alterada.

## v10.8.46 — Deep Pairwise Ranking + Control Utility (pesquisa, sem ordem)
### Resultado real v10.8.46 — Deep Ranking falhou e destruiu o Control ao substituir sua ordem
- Job `control-rank-c67e91abb3914f26`, mesmo SHA, 55 ativos, 3 folds, 1.554 OOS; usuário enviou JSON + `dados(6).zip`. ZIP 429 entradas, CRC ok, 167/167 hashes do manifesto conferidos, 9 artefatos presentes, 85.470 scores = 55×1.554. Contabilidade equity=cash+shares×close fecha até ~US$ 3,64e-12; 248 fills, nenhum > capacidade/10% de volume, nenhum em volume zero, CASH sempre >=0. Portanto não é erro de execução.
- Capital final **US$ 1.280,80**, retorno −87,19%, CAGR −28,27%, Sharpe −0,266, MaxDD −88,57%. Folds terminam 6.359,55 / 5.813,03 / 1.280,80. CASH médio apenas 3,08%: falha é decisão/ranking, não ociosidade.
- Skill pré-OOS já fraca: calibration pairwise accuracy 50,28% / 47,92% / 51,80% (épocas 1/2/1). OOS: pairwise 50,738% em 2,22M pares, Spearman diário médio +0,0215, fold 2 negativo; top neural fica em percentil realizado médio 52,80% (~50,91% esperado por ordem aleatória entre 55) e acerta exatamente o melhor realizado em ~2,81%.
- Híbrido ficou **sticky**: v10.8.46 difere do ativo mantido pela v10.8.44 em 1.233/1.554 datas (79,34%), mas sinal final permanece no incumbent em ~92,1% das sessões vs ~79,0% na v10.8.44; rotações caem 320→119. O #1 neural coincide com #1 LightGBM raw em só ~3,08% das datas maduras e com adjusted-best v10.8.44 em ~3,02%. Regra predefinida rank neural + gate absoluto LightGBM destrói ordem do Control e ao mesmo tempo mantém posições quando o #1 neural não passa margem.
- Conclusão: segunda hipótese DL específica refutada. **Não** significa que DL não serve; significa que DL não deve substituir ranking Control sem skill robusta. Não tunar rede/loss/features pelo mesmo OOS. Próximo DL, se autorizado, deve preservar Control como ação padrão e aprender somente incremento: meta-veto aceitar/rejeitar rotação Control, residual limitado sobre utility Control, ou vantagem contextual candidato-vs-incumbent/rollout contrafactual. Override neural deve voltar ao Control quando confiança/skill de calibração for insuficiente.
- Benchmarks permanecem: Control científico **US$ 5.887.904,38** principal; Control executável US$ 528.709,78; v10.8.44 Liquidity-Aware **US$ 1.078.635,41** melhor caminho restrito histórico; v10.8.45 68.345,26; v10.8.46 1.280,80. Detalhes em `docs/changes/v10.8.46-deep-ranking-research.md`.


- Branch `feature/v10.8.46-deep-ranking-research`, derivada da v10.8.45 documentada. Hipótese nova após falha da TCN-regressão: rede aprende **ordenação cross-sectional entre ativos no mesmo dia**, não magnitude absoluta da utilidade. Loss pairwise logística sobre todos os pares; TinyTCN causal pequeno como encoder; labels de 60 sessões precisam maturar integralmente antes de calibração/OOS; scaler somente treino; arquitetura e hiperparâmetros fixos, sem input HTTP.
- Regra permanente de benchmark: **Control científico US$ 5.887.904,38 é a referência principal de modelagem**; Control v10.8.42 US$ 528.709,78 é apenas cenário restrito de execução; v10.8.44 Liquidity-Aware US$ 1.078.635,41 é melhor pesquisa histórica com execução restrita; v10.8.45 TinyTCN US$ 68.345,26 foi hipótese DL falha.
- Política híbrida v10.8.46: Deep Learning decide somente a ordem dos candidatos; LightGBM original do fold continua fornecendo escala absoluta, CASH, min edge, min hold e switch margin, depois do overlay de liquidez v10.8.44. Se o candidato #1 do ranker não passa o gate absoluto Control, não pular para um candidato neuralmente inferior apenas porque seu Q LightGBM é maior.
- Execução permanece v10.8.42/44: ações inteiras, 10% capacidade, volume realizado só ex-post, custos assumidos, CASH residual, sem liquidação terminal. Nova rota admin `/api/admin/control-shadow/deep-ranking/jobs` exige jobs v10.8.41/42/44/45 e mesmo SHA. Exporta 55×1554 scores, métricas pairwise/Spearman/top-realized-percentile, curva/fills/folds/decisions e JSON em `validation/v10.8.46/<job>`.
- OOS histórico continua exploratório porque resultado v10.8.45 motivou esta hipótese. Não escolher/tunar variante com base em capital v10.8.46 e chamar de confirmação; confirmação requer método congelado e janela futura intocada. Sem TCC, Winner, deploy ou ordens. Detalhes: `docs/changes/v10.8.46-deep-ranking-research.md`.

## v10.8.45 — Deep Learning TinyTCN causal (pesquisa, sem ordem)
### Regra de referência permanente — Control científico vs execução
- **Control científico congelado (v10.8.41 / TCC Control reproduzido): US$ 5.887.904,38** a partir de US$ 10.000, CAGR ~180,46%, Sharpe ~1,956, MaxDD ~−36,65%. Este continua sendo o **Control principal de desempenho/modelagem** e não deve ser esquecido ou substituído por resultados de viabilidade de execução.
- **Control com restrições de execução (v10.8.42): US$ 528.709,78**. Este é um cenário de exequibilidade histórica com limite de volume, custos assumidos, ações inteiras e trajetória estado-dependente; serve para medir quanto do Control científico sobrevive às hipóteses de execução.
- **Liquidity-Aware v10.8.44: US$ 1.078.635,41** sob a mesma contabilidade restrita da v10.8.42. É melhor que o Control *executável* nessa simulação, mas **não supera o Control científico de US$ 5,89M**.
- Toda nova pesquisa (Deep Learning, ranking, contrafactual etc.) deve reportar lado a lado, quando aplicável: (1) Control científico ~US$ 5,89M; (2) Control executável ~US$ 528,7k; (3) Liquidity-Aware ~US$ 1,078M. Nunca chamar US$ 528,7k de "resultado do Control" sem qualificar que é o cenário restrito de execução.


### Resultado real v10.8.45 — TinyTCN falhou como ranker nesta hipótese
- Usuário enviou `output(20260930-092311).zip`: 442 entradas, CRC válido e 167/167 SHA do manifesto original conferidos. Job `control-tcn-b7d0247118314107`; mesmos 55 ativos, 3 folds e 1.554 OOS, sem Alpaca/ordem. TCN terminou com **US$ 68.345,26**, CAGR 36,45%, Sharpe 0,791, MaxDD **−81,12%**, vs Control executável US$ 528.709,78 e v10.8.44 Liquidity-Aware US$ 1.078.635,41. Fold 1 US$ 33.964,69; fold 2 cai a US$ 22.248,55 (−34,50% no fold); fold 3 fecha US$ 68.345,26. Pico global US$ 60.901,95 em 17/01/2024 e fundo US$ 11.496,95 em 28/02/2025.
- Não foi nova armadilha de liquidez: CASH médio 0,596%; apenas 6 sessões >50% CASH; 811 fills todos <= capacidade e <=10% volume realizado; 0 fills em volume zero; 0 CASH negativo; resíduo de balanço equity−(cash+ações×close) < US$ 1,5e−11. Portanto falha principal está em **previsão/ranking**, não execução.
- Scores completos 85.470=55×1.554; 82.225 labels maturados. Métricas OOS: MAE 0,169442, RMSE 0,220872, sign accuracy 72,238%, Spearman −0,000773. Target é positivo 72,305%: baseline sempre-positivo já supera sign accuracy. Constante mediana tem MAE **0,167924** (melhor que TCN); constante média RMSE **0,219265** (melhor que TCN). Predição TCN quase colapsa para média positiva (std 0,0326 vs target 0,2193); Spearman por fold −0,0327/−0,0699/+0,0806; cross-sectional médio diário ~+0,0097. Conclusão: TinyTCN Huber/regressão não aprendeu ordenação útil.
- Viés por ativo explica trajetória: raw top VNCE 282, YANG 245, PXLW 201 datas; holdings finais de trajetória YANG 296 sessões e PXLW 171, vs v10.8.44 apenas 36 e 13. Média prevista/realizada do label: YANG 0,14888/0,00573; PXLW 0,14718/0,03348; CLMT 0,12373/0,19363; NVDA 0,12370/0,20280. A rede superestima YANG/PXLW e subestima alguns vencedores, apesar do overlay de liquidez.
- Não aumentar rede nem tunar features/loss/margem pelo mesmo OOS. A hipótese especificamente refutada é `pooled TinyTCN + Huber regression -> utility scalar`. Próximo DL, se autorizado, deve ser **novo modo/version** com objetivo de ranking (pairwise/listwise por data) ou target contextual candidato-vs-incumbent, sempre com maturidade temporal/purge, controles constantes + LightGBM e período futuro intocado para confirmação. v10.8.44 permanece melhor pesquisa executável histórica até aqui. Detalhes em `docs/changes/v10.8.45-deep-learning-tcn-research.md`.


- Nova branch `feature/v10.8.45-deep-learning-tcn-research` a partir da v10.8.44 auditada (não da main), PR draft separada. Novo modo **somente de pesquisa** `MCT_RESEARCH_TCN_LIQUIDITY_AWARE_V1`; nenhum registro operacional e nenhuma alteração em strategy 28, TCC congelado, Winner ou ordens. Hipótese de pesquisa previamente especificada antes de observar qualquer capital TCN.
- Primeira rede: TCN CPU **pooled** (um conjunto de pesos por fold para 55 ativos), 40 sessões de janela, 12 features causais, 5 blocos Conv1d residuais causal-dilatados, hidden 16, 8 épocas máximas, paciência 2, AdamW, Huber, seed fixa. Alvo supervisado científico `forward_risk_adjusted_utility` (60 sessões máximas), nunca usar target como feature. Cada fold: treino anterior à calibração com **maturidade integral do label antes da calibração**, calibração cronológica purgada decide épocas; retreino expanding até 60 sessões antes do OOS com rótulos já completos, scaler SOMENTE treinamento. Sem seleção de arquitetura por capital OOS. Inference não usa filtro futuro do cache TCC original.
- Política TCN nova reusa regras do Control (margem fixa 0,0025) e overlay de capacidade de entrada/saída conhecida da v10.8.44; execução idêntica à v10.8.42 (10% menor de mediana anterior / volume realizado ex-post, spread/impacto hipotéticos, ações inteiras, CASH residual, sem liquidação terminal). Baselines congelados v10.8.42 e v10.8.44 apenas consultados, não recalculados pela experiência DL.
- Rota admin `POST/GET /api/admin/control-shadow/deep-learning/jobs`, dependência jobs v10.8.41, 10.8.42, 10.8.44 e SHA imutável. Exporta `tcn_oos_scores.csv`, training folds, métricas preditivas (somente labels maturados), curva/fills e patrimônio OOS, JSON e PNG para `validation/v10.8.45/<job_id>`. Documentação de desenho e limites `docs/changes/v10.8.45-deep-learning-tcn-research.md`.
- O OOS antigo foi observado e motivou nova arquitetura, portanto resultado dessa janela será **exploratório**, não confirmação independente nem justificativa para Winner. Não inventar capital antes de executar a pesquisa local no snapshot. Para validação confirmatória, congelar tudo e usar novas sessões posteriores a 28/09/2026.

## v10.8.44 — MCT_RESEARCH_CONTROL_LIQUIDITY_AWARE_V1 (nova política experimental)
### Auditoria independente do ZIP v10.8.44 (dados (3).zip)
- ZIP 408 entradas, 11 artefatos do job `control-liquidity-dbe4ec6f52c74694`, CRC válido, 167/167 SHA dos arquivos da fonte iguais ao manifesto original. Curvas dos dois caminhos alinhadas nas 1.554 datas; Control congelado v10.8.44 idêntico à curva v10.8.42 (diferença máxima 0); todos os 85.470 scores de candidatos revisados independentemente.
- Recalcular a capacidade anterior para cada ativo a partir dos CSVs normalized_bars (10% × mediana 20 volumes até t × close(t)) reproduziu **85.470/85.470** valores exatamente, considerando regra explícita do incumbent já mantido: capacidade de *entrada nova* reportada 0 e fração de permanência 1. Patrimônio real de decisão, capacidade de saída do ativo atual, frações e scores Q×fração conferem até ruído de ponto flutuante. Balanço a mercado das 1.554 sessões dos dois caminhos fechou com erro absoluto menor que US$ 2,4e-10; 0 fills acima do limite/10% de volume, 0 fills em sessões de volume zero, 0 CASH negativo.
- Dos 209 dias com alvo diferente, 178 = 106 VNCE + 72 PXLW como alvo original substituído (85,17%); diferenças por fold = 27/154/28. Posição original VNCE 189 sessões, PXLW 119; nova VNCE 3, PXLW 13. Fills originais VNCE 167, PXLW 100, novo 4 e 10. CASH >50% passou de 172 dias (122 VNCE +49 PXLW +1 LKFT) para 5 dias (3 MYE, 1 VNCE, 1 LKFT), todos em venda parcial. Fold 2 CASH 24,93% → 0,284%.
- Risco concreto: novo MaxDD −42,614% entre pico 26/09/2024 (US$ 318.449,06) e fundo 07/10/2024 (US$ 182.744,66). Patrimônio final mantém 4.382 AMZN + US$ 6,11, sem vender/liquidar no final. Reportou custo de preço modelado maior no novo caminho (US$ 138.320 vs 65.672), por exposição/tamanho maiores.
- Auditoria de disponibilidade do próximo bar no cache científico TCC: 55×1554=85.470 decisões candidatas possuem a barra imediatamente seguinte e open/close válidos neste snapshot, portanto o filtro herdado não eliminou ativo aqui; **continua dependência metodológica de dado futuro e deve ser corrigida numa pesquisa separada para uso prospectivo, sem editar a fonte TCC**. 1.636 utilidades negativas tiveram magnitude alterada por Q×fração, mas 0 casos de primeiro colocado negativo nas 1.554 datas; não explica a alteração do top nesta execução. Guardar risco matemático em estudo futuro, sem ajustar com base no OOS.
- Auditoria completa em `docs/changes/v10.8.44-control-liquidity-aware-research.md`. Não houve rerun, mudança de política, merge, ordem, Winner ou TCC.

### Resultado local v10.8.44 — 29/09/2026
- Job `control-liquidity-dbe4ec6f52c74694` concluído (`error=null`), mesmo snapshot 55 ativos, 3 folds/1.554 OOS, um treino científico compartilhado, dados verificados e paridade v10.8.42 confirmada. Resultado JSON Control referência US$ 528.709,78 vs Control Liquidity-Aware experimental US$ 1.078.635,41 (+US$ 549.925,64 / +104,01% da referência); CAGR 89,94% vs 113,15%; Sharpe 1,460 vs 1,669; **MaxDD piorou** de −38,70% para −42,61%. CASH médio caiu de 10,17% para 0,45%; sessões parciais 399→241; pedidos de ações não preenchidos acumulados 1.394.869→30.418; custos de preço modelados aumentaram US$ 65.671,66→138.320,02, com mais capital exposto.
- A maior vantagem relativa surge no fold 2: 99.182,94 vs 261.131,01; CASH médio 24,93% vs 0,28%. No fold 3 o ganho percentual **foi menor** no novo modo: +313,06% vs +433,07% original, e DD −42,61% vs −35,15% nesse fold. A política altera alvo em 209/1554 datas, primeiro ranqueado em 181, com 85.470 linhas de scores de candidatos.
- Análise até agora baseada no JSON da API; requer ZIP com 11 artefatos para auditar o ledger/curvas e identificar quais ativos alteraram resultado. Fonte de lucro é trajetória contrafactual sob fills/custos hipotéticos; não é capital liquidado. Limitação da pré-computação TCC verifica disponibilidade do open/close de t+1; auditar antes de afirmar causalidade end-to-end. A transformação Q×fração reduz também magnitude de utilidade negativa, potencial efeito de ranking a conferir. Este OOS motivou a hipótese: não confirmar/promover nem usar para otimização. Não alterar TCC, Winner, Control original ou ordens.
- Histórico detalhado em `docs/changes/v10.8.44-control-liquidity-aware-research.md`; PR #12 draft. Recomendado próximo estudo audit-only de causalidade, consolidação de escolha por ativo/capacidade, e teste em janela futura genuinamente nova.


- Autorização contextual: usuário cobrou passar da auditoria v10.8.43 à implementação da política que avalia a liquidez **antes da escolha do ativo**. Branch `feature/v10.8.44-control-liquidity-aware-research` criada da v10.8.43 e documentação em `docs/changes/v10.8.44-control-liquidity-aware-research.md`. Modo **separado e somente de pesquisa**, não inscrito nos strategy_modes operacionais, nem altera strategy 28, TCC, Winner, universo, snapshots ou ordens.
- Hipótese fixa sem otimização no OOS: em t fechado, capacidade em dólares = 10% × mediana dos 20 volumes diários conhecidos até t × close(t); entrada = min(1, capacidade/patrimônio real); para trocar, limita também pela capacidade estimada de saída do incumbent; para manter incumbent, fração 1. Q_efetiva = Q_Control × fração capital estimado negociável; CASH 0; o Control original aplica minhold, threshold e switch-margin sem reescrita. Um único treino LightGBM com três folds, referências imutáveis v10.8.41/v10.8.42/v10.8.43 e duas trajetórias estado-dependentes com mesma execução limitada 10%, spread/impacto/taxas. Deve reproduzir todos os indicadores e ativo terminal v10.8.42 antes de reportar.
- Rotas admin `/api/admin/control-shadow/liquidity/jobs` (POST/GET/logs), corpo fixo com IDs anteriores e SHA; resultados próprios `validation/v10.8.44/<job>` (curvas/fills duplos, comparação por fold, auditoria por data e 55 candidatos, PNG, JSON). Sem download Alpaca, sem orders/deploy. A seleção de nova política não é decidida pelo resultado deste OOS; requer teste prospectivo.
- Cautela herdada: precomputação original do TCC também valida disponibilidade de open/close do próximo pregão antes de fornecer utilidade. O overlay de liquidez usa só dados de t fechado, mas auditar essa dependência original antes de afirmar causalidade completa para toda a política.

## v10.8.43 — sensibilidade fixa de exequibilidade (pesquisa, sem ordem)
### Auditoria dos CSVs completos v10.8.43 (ZIP subsequente)
- O usuário forneceu `dados (2).zip` com 395 entradas e os 15 arquivos v10.8.43. ZIP CRC e os **167 SHA-256 dos arquivos do manifesto** passaram; 5 curvas alinhadas nas 1.554 datas. Ledger `cap10_cost` v10.8.43 confere linha a linha nos principais campos contábeis com os 1.554 registros e 807 fills v10.8.42 (capital, CASH, quantidades, ativo/sinal, taxas).
- Comparador idealizado `no_cap_no_cost`: 41 fills superiores ao volume total da sessão (VNCE 24, PXLW 17); venda de 14.666 VNCE em 16/03/2023 com volume registrado ZERO; compra de 503.874 VNCE em 25/09/2024 ante 9.895 ações negociadas no dia (**50,92× volume diário**). Os cenários limitados não preencheram operações com volume zero nem excederam o teto definido. Isto prova a inviabilidade da contabilidade sem cap como garantia de fills, sem estabelecer o preço real dos cenários com cap.
- CASH >50% do patrimônio por cenário: sem cap 0 sessões; 10%-sem custo adicional 197; 10%-com custo 172; 5%-com custo 221; 1%-com custo 371 (e 190 sessões >90%). No cenário 10%+custo, VNCE 122 sessões + PXLW 49 = 171/172. No 1%, período contínuo de 100 sessões com >50% CASH (29/08/2023 a 22/01/2024), média 81,62%.
- Com mesmo cap 10%, preço modelado com/sem custo alterou posição efetivamente detida em só 8 datas e sinal em 2 datas, mas mudou capital composto final em US$ 416.515. Entre 5% e 10% com custo, posições divergem 34 datas, sinais 10; entre 1% e 10%, 161 datas e 52 sinais. Contrastes de patrimônio continuam **não causais/aditivos** porque preço, tamanho, saldo e decisões evoluem juntos. Consulta: `docs/changes/v10.8.43-control-execution-sensitivity.md` (seção auditoria).
- Não fazer seleção oportunista do percentual 5%/1% pelo capital OOS; preservar políticas e fontes. Próxima hipótese conceitual, em novo strategy_mode separado e pesquisa temporal, seria tornar candidatos conscientes de capacidade **prévia** usando apenas sessões completas anteriores, com cenários fixos e custos consistentes; não retirar VNCE/PXLW por suposto defeito estrutural de série.

### Resultado real v10.8.43 — 29/09/2026
- Job local `control-sensitivity-1c8b4690d80d4145` `completed`, `error=null`; 55 hashes numéricos OK, 3 folds, 1.554 sessões OOS; snapshot original inalterado, sem Alpaca/ordens. `v1042_scenario_regression=verified` reproduziu US$ 528.709,7764371407, 291 rotações executadas, posição terminal 2.147 AMZN + US$ 225,726 CASH.
- Capital final dos cinco cenários fixos: `no_cap_no_cost` US$ 5.873.701,17 (idealizado, pode exceder volume ou simular fill com volume 0, **não executável**); `cap10_no_cost` US$ 945.224,77; `cap10_cost` US$ 528.709,78; `cap05_cost` US$ 587.359,57; `cap01_cost` US$ 551.188,85. O cenário idealizado aproxima os US$ 5.887.904,38 científicos, mas não possui contabilidade idêntica (ações inteiras e sem liquidação terminal). Contrastes são trajetórias distintas, **não decomposição causal aditiva**.
- Mais restrição de participação não implica capital final menor: 5% e 1% encerraram acima do caminho 10% no período congelado; não usar isso para otimizar ou promover um limite no OOS. CASH médio nos casos 10% com custo 10,17%, 5% 13,08%, 1% 21,57%; sessões parciais 399, 443 e 565. Fold 2 CASH médio com cap 1% 44,96%. Evidência reforça armadilhas de liquidez e reação estado-dependente da política, além de spread/impacto simulados.
- Somente JSON de status/resultado da v10.8.43 enviado nessa rodada; os CSVs de operações e curvas novos ainda não foram inspecionados. Dados e caveats completos em `docs/changes/v10.8.43-control-execution-sensitivity.md`. Próxima investigação conceitual, se autorizada: seleção de candidatos com liquidez conhecida até o fechamento anterior; criar strategy_mode separado, pré-definir desenho e validação OOS; preservar Control, TCC e Winner.


- Após análise do ZIP de resultados reais da v10.8.42, implementar comparação fixa com **mesmo snapshot, 55 ativos, 3 folds, TCC Control imutável e um único treinamento LightGBM**; 5 replays de contabilidade independentes por posição real: `cap10_cost` (baseline v10.8.42), `no_cap_no_cost` (idealização, pode exceder volume e preencher dia volume 0; jamais chamar exequível), `cap10_no_cost`, `cap05_cost`, `cap01_cost`. Sem seleção de vencedora a posteriori.
- Branch `feature/v10.8.43-control-execution-sensitivity` baseada na v10.8.42; PR nova em draft. Gatilhos são os jobs concluídos `control-validation-7821002400424ccc` e `control-execution-c38169f6c6fc4ea0`, ambos vinculados a `control-shadow-870fb66e1bdc4fd0` e hash `6d9e7d69865277487a6b193adedcc1d91e428ab451921d538aa4938fe108e8f3`. Paridade numérica e repetição do cenário 10% com os números reais da v10.8.42 exigidas antes de emitir qualquer relatório; divergência -> falha, não ajuste oportunista.
- Rota admin `POST/GET /api/admin/control-shadow/sensitivity/jobs` sem download/ordem; outputs em `validation/v10.8.43/<job_id>` com 5 curvas/fills, `scenario_comparison.csv`, `fold_comparison.csv`, `aligned_capital_curves.csv`, PNG e JSON; detalhes em `docs/changes/v10.8.43-control-execution-sensitivity.md`.
- Hipóteses declaradas: participação 1%/5%/10% do menor volume mediano das 20 sessões anteriores e volume realizado no dia de execução (ex post, nunca como input do sinal); 15bps full spread assumido e 20bps coeficiente sqrt-participação para custo extra. Cenário sem limite mantém mesma contabilidade de ações inteiras e taxas, mas não é igual ao replay científico antigo (ações fracionárias/FINAL_SELL). Diferenças entre trajetórias não são decomposições causais aditivas. Preservar TCC, MCT Winner, snapshot, relatórios e Trader.

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
