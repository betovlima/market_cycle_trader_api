# PCEA v2.1.0 — correção da avaliação por episódios

Versão da API/pacote: **10.8.42**. Branch: `research/pooled-candidate-episode-advantage-v2-1`.

Base: `research/pooled-candidate-episode-advantage-v2`, commit `9dbd0f1fa06493c68e730df61e78589e3951174e`.

## O que o resultado enviado permite concluir

O modelo atual ainda não demonstrou aumento de capital. Ele já usa aprendizado estatístico: `StandardScaler + BayesianRidge`, com 64 variáveis, 26 candidatos externos e 56 ativos da estratégia de referência. A identidade do ativo não entra como variável do modelo. A versão final do modelo foi treinada com 160 episódios iniciados em 131 datas; esses episódios não constituem necessariamente observações independentes.

Auditoria do arquivo `885f678d-afa0-44e5-a084-f583069b7c5d.zip`, SHA-256 `6ffe6991d0af2301070a5b705633417cef7c244365123e60ea00b582e58da73b`:

| Medida do export original | Validação entre janelas | Período final |
|---|---:|---:|
| Grupos de início avaliados | 76 | 26 |
| Escolhas de candidato | 5 | 2 |
| Soma da vantagem em log dos episódios escolhidos | -0,4660092409 | -0,0395960290 |
| Fator `exp(soma) - 1` | -37,249851% | -3,882235% |

Esses fatores foram recalculados a partir dos CSVs de decisões. **Não são o retorno de uma carteira completa executando o filtro.** O próprio export informa `full_stateful_overlay_backtest_run=false`. O capital da referência informado para o período final é 9.716,5964, partindo de 10.000; o capital de uma carteira completa com o filtro não foi calculado nesse experimento.

As duas escolhas finais foram AGIG, com vantagem realizada de -0,1256998632 em log, e UTI, com +0,0861038342. Remover um ativo porque perdeu nesse período seria uma nova escolha baseada no resultado já observado; esta correção não acrescenta regras por ticker.

## Problema encontrado

Um episódio está censurado à direita quando a simulação termina antes de o candidato e a referência voltarem ao mesmo estado de decisão. Isso só é conhecido ao acompanhar o caminho depois da admissão.

O export contém 161 episódios anteriores ao período final, mas apenas 160 linhas de amostra: SGRY, iniciado em 2022-07-20, foi excluído por estar aberto. No período final, havia 33 episódios, mas somente 31 foram apresentados ao modelo. Foram excluídos:

| Candidato | Início | Motivo da exclusão na v2 |
|---|---|---|
| UTI | 2026-08-19 | Episódio ainda aberto no fim da janela |
| GOGO | 2026-09-02 | Episódio ainda aberto no fim da janela |

Excluir um resultado incompleto do treino supervisionado é necessário para não apresentá-lo como um resultado completo. Aplicar essa exclusão também às opções da avaliação introduz informação posterior à decisão. Além disso, o código exigia um retorno futuro finito para gerar a previsão.

O ZIP não contém as variáveis desses dois episódios excluídos nem as cotações completas para reconstruí-las. Portanto, **não é possível calcular a nova escolha ou o novo resultado financeiro apenas editando os CSVs enviados**. É necessário repetir o experimento com o mesmo cache local de mercado.

## Correções implementadas

1. Todos os inícios de episódio, inclusive os ainda abertos, recebem previsão quando suas variáveis da decisão são válidas.
2. A construção das variáveis do PCEA não calcula nem consulta o retorno da próxima sessão. A previsão não exige retorno futuro ou data de término do episódio.
3. O resultado completo fica ausente nos episódios em aberto. O retorno já observado é exportado em coluna própria e inclui os custos de liquidação terminal do replay.
4. O treino exige episódio encerrado e `label_available_at` anterior à primeira sessão de previsão. Essa data usa a sessão de execução, porque a data original do replay corresponde à decisão anterior.
5. Se um episódio escolhido continuar aberto, ele ocupa o restante da janela de avaliação. Seu resultado parcial é separado do resultado completo; não é convertido em zero. As métricas completas ficam `null` quando ainda há resultado pendente.
6. Janelas sem episódios produzem CSVs com cabeçalho e um diagnóstico de nenhuma intervenção. A numeração das janelas não depende de haver candidatos escolhidos.
7. O manifesto registra versão, código, ambiente e hashes das cotações para comparação entre execuções. O período final é identificado como comparação retrospectiva, pois já foi observado em pesquisas anteriores.
8. A leitura, escrita, criação de diretórios e verificação de existência no script legado de liderança usam o mesmo suporte a caminhos longos do Windows. Isso corrige a inconsistência presente na leitura de `intrinsic_timing_summary.csv` do traceback informado.

O modelo, suas 64 variáveis, o ponto de indiferença econômica em zero, o cache de mercado, a política da referência e seus parâmetros de execução são preservados. O processo lê o MongoDB local e não faz gravações nele nem busca novas cotações em provedores externos.

## Como executar no Windows

Na raiz de `market_cycle_trader_api`, usando o ambiente Python já configurado:

```powershell
git fetch origin
git switch research/pooled-candidate-episode-advantage-v2-1
git pull --ff-only origin research/pooled-candidate-episode-advantage-v2-1

python scripts/research_pooled_candidate_episode_advantage.py --strategy-sequence 10 --history-start 2016-01-01 --snapshot-end 2026-09-04 --validation-sessions 252 --fresh-run
```

`MONGO_DATABASE` e `MONGO_URL`/`MONGO_URI` vêm do ambiente ou `.env`. Também é possível usar `--database`, `--mongo-uri` e `--env-file`. A URI deve apontar para o MongoDB local. Não acrescente `--workers` ou `--selection-method`: esses parâmetros pertencem a outros scripts.

O novo diretório padrão é:

```text
research_output/pooled_candidate_episode_advantage_strategy_10_v2_1_2025-09-05_to_2026-09-04
```

`--fresh-run` apaga somente a saída desse experimento no diretório selecionado e protegido pelo script. O diretório padrão da v2 anterior tem outro nome e é preservado. Nenhum resultado antigo é usado para treinar o novo experimento.

Os principais arquivos para enviar após a execução são `pcea_result.json`, `experiment_manifest.json`, os CSVs `*_all_episodes.csv`, `*_episode_samples.csv`, `*_scored_episodes.csv` e `*_decisions.csv`.

## Verificação

Resultado: **29 testes aprovados** — 12 novos casos de correção, seis testes existentes do PCEA, dez testes da simulação marginal e uma integração completa.

```powershell
python -m unittest discover -s tests -p "test_pcea_v21.py" -v
python -m unittest discover -s tests -p "test_pooled_candidate_episode_advantage.py" -v
python -m unittest discover -s tests -p "test_asset_marginal_v2.py" -v
python -m unittest discover -s tests -p "test_pcea_pipeline_integration.py" -v
```

Os testes verificam a invariância da previsão e escolha inicial à mudança do resultado futuro, a elegibilidade de episódios abertos, a disponibilidade temporal dos resultados do treino, a separação de retornos parciais, o tratamento de janelas vazias e a leitura/escrita em caminhos com mais de 260 caracteres. A regressão da simulação compara posições, curva de capital, custos e rotações com o motor completo, incluindo ações inteiras e fracionárias.

A integração usa 1.040 sessões sintéticas, quatro ativos, modelos LightGBM reais, replay de contas e ajuste bayesiano entre janelas. Somente a entrada de dados Mongo é substituída. A execução foi verificada em Linux/Python 3.12; não houve execução nativa em Windows/Python 3.14. Os testes sintéticos validam o código e não demonstram rentabilidade.

## Limite e próximo experimento

A soma de episódios sem sobreposição ainda não reproduz necessariamente uma única conta: simulações com candidatos isolados podem ter caixa residual, quantidades e custos diferentes. Para verificar aumento de capital, o filtro precisa ser executado em uma carteira única, com estado contínuo e custos reais, comparada à referência no mesmo período e nas mesmas cotações.

Esta versão corrige a medição necessária antes dessa comparação. Ela não demonstra que a regressão bayesiana atual aprendeu uma vantagem econômica, nem resolve a incerteza decorrente de poucos episódios e dependência temporal. Trocar o modelo, selecionar variáveis ou ajustar cortes a partir deste período final constituiria outra hipótese, que precisaria de validação cronológica própria. A janela de 2025-09-05 a 2026-09-04 já observada permanece uma comparação retrospectiva.
