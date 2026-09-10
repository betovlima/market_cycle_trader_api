# Exact Marginal Capital Search v1.0.4 — auditoria e correção Top-K

Branch: `research/exact-marginal-capital-search-v1-0-4`.
Script: `exact-marginal-capital-search-v1.0.4`. API/pacote: `10.8.48`.
Base da correção: `a3dab1d7cc8b793ae1789193ed815612224ca433` (v1.0.3).

## Resultado da avaliação dos dois anexos

O primeiro anexo contém os resultados econômicos da v1.0.2. O segundo contém
somente os arquivos iniciais da v1.0.3. Não é possível comparar os capitais das
duas execuções nem confirmar o efeito do controle de identidade no segundo ZIP.

| Evidência | Execução anterior | Execução atual |
| --- | --- | --- |
| ZIP | `f1700e5c-2c0c-4352-ba26-05bcd1e8da20.zip` | `bdacac55-7615-4d44-9883-b5c8644c87b2.zip` |
| Versão no manifesto | v1.0.2 | v1.0.3 |
| Commit correspondente | `12a21c0d1ef70bdbb1f9aba3310f3e5091d7ef2b` | `a3dab1d7cc8b793ae1789193ed815612224ca433` |
| API registrada | 10.8.47 | 10.8.47 |
| Candidatos declarados | 4 | 5 |
| Arquivos de dados | 6 | 2 |
| Resultado base exportado | Sim | Ausente |
| Desfechos de candidatos exportados | 4, dos quais 3 econômicos | Nenhum |

Ambos usam a estratégia #10, revisão 15, os mesmos 56 ativos, início do histórico
em 2016-01-01 e corte em 2026-09-04. O hash da configuração é
`43abde0cf9d3e445cf29d0bb696a3b26c1124cec44daffed1cef544b578a379b`.
Os dois `baseline_history_integrity.csv` são idênticos byte a byte: todos os
56 ativos têm 2.684 sessões brutas e nenhuma sessão faltante.

O manifesto atual contém o hash da lista, mas não os símbolos. Recalcular o hash
de `HL|VSAT|RARE|CCS|LB`, lista documentada na v1.0.3, reproduz exatamente esse
hash. Assim, HL foi acrescentado ao conjunto anterior; uma futura comparação
deve separar os quatro controles comuns desse novo candidato.

### Execução v1.0.2: capital efetivamente exportado

| Universo | Capital final | Diferença contra a base | Queda máxima |
| --- | ---: | ---: | ---: |
| Base com 56 ativos | 23.712.122,62 | — | -28,24% |
| Base + VSAT | 22.257.978,42 | -1.454.144,20 (-6,13%) | -38,69% |
| Base + RARE | 11.650.044,08 | -12.062.078,54 (-50,87%) | -28,24% |
| Base + CCS | 23.712.122,62 | 0,00 (0,00%) | -28,24% |
| Base + LB | Não comparável | Rejeição de contexto | Não comparável |

As diferenças conferem aritmeticamente com os capitais exportados. Nenhum dos
três candidatos comparáveis aumentou o capital nesse teste. CCS foi neutro;
a igualdade do resumo não demonstra que ele nunca foi negociado. LB possui
histórico bruto completo, mas somente 817 das 1.538 sessões de decisão da base:
faltam 721. Sua rejeição é de comparabilidade, não uma classificação de perda.

A base tem 317 trocas e sessões de decisão de 2020-07-22 a 2026-09-03. O backtest
antigo dos aproximadamente 23 milhões terminava em 23.521.539,12, com 1.537
sessões até 2026-09-02. A diferença de 190.583,50 (+0,81%) e uma sessão adicional
impedem tratar os dois resultados como reprodução idêntica. O ZIP novo não
contém a curva diária e as operações necessárias para explicar toda a diferença.

Somando os tempos registrados, a base levou 8,20 minutos e os quatro replays de
candidatos levaram 35,00 minutos: 43,20 minutos, sem o treinamento inicial do
pré-seletor e outras despesas. LB consumiu 6,93 minutos de replay antes de ser
rejeitado. Esses tempos não medem necessariamente retreinamento integral de cada
modelo: o pacote instala cache de modelos de referência, e os arquivos não
registram quantos modelos foram reutilizados.

### Pré-seletor: ainda sem evidência de valor econômico

O LightGBM LambdaMART ordena ativos por um alvo de 20 sessões. A mediana de
NDCG@5 (medida de qualidade da ordenação dos cinco primeiros, não retorno
financeiro) foi 0,873 no treino e 0,477 fora do treino, contra 0,509 da ordenação
aleatória. O modelo ficou abaixo do aleatório em três das quatro janelas e
superou o comparador de momentum em apenas uma. O diagnóstico exportado é
`overfitting_risk`.

Há também desalinhamento entre objetivos: `_future_utility` usa retorno futuro
do ativo mais **0,45 vezes sua excursão adversa**, enquanto o avaliador econômico
mede a mudança no capital da estratégia completa. O coeficiente 0,45 é fixado
no código; não foi aprendido. O score calculado no fim de 2026-09-04 tampouco
constitui uma previsão feita antes do período econômico de 2020–2026.

Neste experimento os scores são diagnósticos e nenhum candidato é eliminado
por score. O executor submete os símbolos na ordem de entrada; a classificação
por score é calculada nos relatórios. Logo, a tabela Top-K avalia uma ordenação
hipotética, não prova economia de tempo na ordem real dessa execução.

Uma próxima pesquisa coerente com o objetivo seria aprender a contribuição
marginal ao capital usando resultados do avaliador exato, com atributos
disponíveis antes de cada janela e avaliação em períodos e ativos separados do
treino. Isso exige uma amostra maior, contendo contribuições positivas e
negativas. Quatro controles, com zero positivos comparáveis, não permitem
validar esse filtro. A v1.0.4 não introduz limites de score nem altera o alvo
econômico para forçar admissões.

### O que falta no resultado v1.0.3

O ZIP contém apenas `exact_search_manifest.json` e
`baseline_history_integrity.csv`. Estão ausentes:

- `exact_baseline.json`;
- `exact_candidate_evaluations.csv`;
- `preselector_recall.csv`;
- `exact_search_summary.json`.

No código, os dois primeiros arquivos são gravados antes da limpeza final dos
frames, do treinamento do pré-seletor e do replay da base. A checagem de
identidade da v1.0.3 só acontece na avaliação dos candidatos, após a base.
Esses arquivos não identificam a fase em execução quando o ZIP foi criado,
nem permitem concluir que houve exceção, travamento ou falha em LB. Para esse
diagnóstico são necessárias as últimas linhas do terminal, incluindo o
traceback, se houver, ou a pasta completa após o término.

## Correção demonstrada nesta branch

Na v1.0.2 a ordenação é VSAT (1), RARE (2), LB (3), CCS (4). O cálculo anterior
limitava K ao número de avaliações econômicas concluídas, que é três, mantendo
as posições originais. Por isso até Top 100 excluía CCS e mostrava apenas 2/3
das avaliações concluídas, com melhor diferença de capital de -6,13%.

A correção aplica K às posições originais, incluindo rejeições/falhas que
possuam score. Top 3 continua com dois resultados econômicos e uma rejeição;
Top 4 ou superior inclui os três resultados econômicos e a rejeição. A cobertura
dos resultados concluídos passa a 100% e a melhor diferença passa a 0,00%.
O recall de positivos permanece indefinido (`null`), pois não há positivos.
Candidatos positivos sem score continuam no denominador do recall.

Os novos campos distinguem tamanho da fila, tentativas no prefixo e avaliações
com resultado econômico. Ranks numéricos lidos de CSV são aceitos. Os capitais
não são recalculados pela correção. O wrapper v1.0.4 preserva a checagem de
identidade da v1.0.3 e o avaliador econômico existente.

## Comandos

Na raiz do repositório:

```powershell
git fetch origin
git switch research/exact-marginal-capital-search-v1-0-4
python -m unittest discover -s tests -p "test_exact_marginal_capital_search_offline.py" -v
```

Para corrigir o relatório anterior, sem MongoDB, Alpaca ou novos backtests:

```powershell
python scripts/audit_exact_marginal_capital_search.py --input research_output/exact_marginal_capital_search_smoke --output-dir research_output/exact_marginal_capital_search_audit_v104
```

`--input` também aceita o caminho para um ZIP. `--output-dir` deve ser uma pasta
nova ou vazia. `--cutoffs 3 5 10 20 50 100` é opcional. O auditor escreve
`exact_search_audit.json` e, quando há dados econômicos válidos,
`preselector_recall_corrected.csv`. Preserva os arquivos de origem e registra
seus hashes. Saída 0: exportação completa; 2: exportação incompleta; 1: erro ou
exportação inválida. Uma exportação completa pode conter candidatos rejeitados
ou falhos: isso não significa que o experimento tenha candidatos rentáveis.

Para auditar a pasta atual, inclusive se ainda estiver incompleta:

```powershell
python scripts/audit_exact_marginal_capital_search.py --input research_output/exact_marginal_capital_search_smoke_v103 --output-dir research_output/exact_marginal_capital_search_audit_current_v104
```

O novo runner para uma execução econômica futura é
`scripts/research_exact_marginal_capital_search_v104.py`, com os mesmos argumentos
da v1.0.3 e uma pasta de saída diferente. Não é necessário repetir os replays
para corrigir o relatório da v1.0.2. O auditor offline não resolve uma eventual
interrupção do treinamento/replay da v1.0.3.

## Validação e limites

Foram executados 14 testes: cinco do relatório original e nove de regressão e
auditoria. Incluem um positivo após candidato rejeitado, neutralidade sem
positivos, ranks de CSV, positivo sem score, exportação incompleta e CLI sem
pacotes externos (`python -S`). A auditoria foi aplicada aos dois ZIPs reais:
v1.0.2 completa, v1.0.3 incompleta. O motor econômico completo não foi executado.

Os manifestos não incluem SHA do código, hash completo dos preços nem hash da
configuração efetiva dos modelos; a associação a commits é feita pela versão
do script. As declarações de fonte de dados não registram os acessos efetivos
por candidato. O mecanismo de retomada também não compara hashes dos modelos
e preços e recalcula a base antes de recuperar resultados de candidatos.
Assim, os anexos não demonstram isolamento de um snapshot imutável nem ausência
de mistura em uma retomada; não há evidência de que isso tenha ocorrido aqui.

SHA-256 dos ZIPs avaliados:

```text
v1.0.2  4f1d9974f5a61ab0b3915fa84ad01fcc77b5c6e3433c52d209b8ff0e1473ad8a
v1.0.3  6be4f2504b74decc82e7b342a9b813db03577c83063487380fd6da02b4b43f05
```

Esta é uma branch de pesquisa. Não foi mesclada à produção e não recebeu tag.
