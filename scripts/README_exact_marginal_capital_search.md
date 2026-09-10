# Exact Marginal Capital Search v1.0.0

API/pacote: **10.8.46**. Branch: `research/exact-marginal-capital-search-v1`.
Base deliberadamente limpa: `research/asset-marginal-rotation-contribution-v2`, commit `7ec43fc227052e6e75b59e3de0426c765294fad6`.

## Objetivo

Esta pesquisa volta ao mecanismo que historicamente funcionou: cada ativo externo com histórico completo é colocado dentro da Strategy e julgado pelo motor econômico real. O pré-seletor existe apenas para medir a qualidade da fila. Ele não aceita nem rejeita candidatos.

O benchmark responde primeiro a duas perguntas mensuráveis:

1. Quanto custa uma avaliação econômica exata por candidato?
2. Quantos candidatos com `ending_capital_delta_rate > 0` aparecem nos Top 5, 10, 20, 50 e 100 do pré-seletor?

Nenhum novo target de ML, BayesianRidge, PCEA, DCCA ou CAUS participa do julgamento.

## Juiz econômico exato

O baseline é executado uma vez com o mesmo caminho usado pela Asset Discovery: `_run_rotation_replay -> run_rotation_models`.

Depois, para cada candidato com Full Strategy History:

- cria-se o universo `baseline + candidato`;
- o próprio motor treina/calibra/executa a Strategy com esse universo;
- custos, slippage, permanência, rotações e compound são produzidos pelo motor existente;
- a contribuição é `candidate_ending_capital / baseline_ending_capital - 1`;
- positivo, negativo e falha técnica são todos exportados.

Reutilizar a execução do baseline é exato porque o baseline não muda entre candidatos. A v1 **não reutiliza modelos treinados dos 56 dentro da execução do candidato**; primeiro queremos um benchmark correto antes de otimizar cálculo.

## Papel do pré-seletor

O `Learning-to-Rank` existente é treinado nos ativos da Strategy e fornece `preselector_raw_score` e `preselector_rank`. Porém:

- nenhum `rank` limita a avaliação exata;
- candidato sem score continua sendo avaliado se o histórico da Strategy estiver completo;
- `preselector_recall.csv` mede retrospectivamente quanto trabalho seria necessário para recuperar os vencedores econômicos exatos.

Isso permite responder, por exemplo, se Top 20 encontra 90% dos candidatos positivos sem transformar 20 em uma regra de negócio.

## Fonte de dados e reprodutibilidade

A execução é local-MongoDB only. Não chama Alpaca e não grava market bars. O universo externo é formado pelos símbolos já presentes em `alpaca_market_bars` com o mesmo timeframe/feed/adjustment da Strategy, excluindo os ativos baseline.

`--history-start` precisa coincidir com o início configurado na Strategy. O benchmark exige o calendário XNYS completo usado pelo baseline e rejeita candidatos com histórico insuficiente ou descontínuo.

O manifesto vincula Strategy id/revision/hash, janela e hash do universo de candidatos. Se a execução for interrompida, rode novamente **sem `--fresh-run`**: candidatos já concluídos são preservados. Se qualquer parte do contrato mudar, o resume é recusado.

## Executar

```bash
git fetch origin
git switch research/exact-marginal-capital-search-v1
git pull --ff-only origin research/exact-marginal-capital-search-v1

python -m unittest discover \
  -s tests \
  -p "test_exact_marginal_capital_search.py" \
  -v

python scripts/research_exact_marginal_capital_search.py \
  --strategy-sequence 10 \
  --history-start 2016-01-01 \
  --snapshot-end 2026-09-04 \
  --workers 1 \
  --fresh-run
```

O primeiro benchmark deve usar `--workers 1` para termos tempos por candidato interpretáveis e para reduzir competição de CPU/memória. Depois podemos repetir uma amostra explícita com mais workers para medir paralelismo sem mudar a matemática.

Para retomar uma execução interrompida:

```bash
python scripts/research_exact_marginal_capital_search.py \
  --strategy-sequence 10 \
  --history-start 2016-01-01 \
  --snapshot-end 2026-09-04 \
  --workers 1
```

Para um smoke test com símbolos conhecidos, sem transformar a lista em filtro permanente:

```bash
python scripts/research_exact_marginal_capital_search.py \
  --strategy-sequence 10 \
  --history-start 2016-01-01 \
  --snapshot-end 2026-09-04 \
  --candidate-symbols LB VSAT RARE CCS \
  --workers 1 \
  --output-dir research_output/exact_marginal_capital_search_smoke
```

Observação: `--fresh-run` só pode apagar diretórios com prefixo `exact_marginal_capital_search_` dentro de `research_output`. Para o smoke test com diretório customizado, remova manualmente o diretório se quiser reiniciar.

## Saídas principais

- `exact_candidate_evaluations.csv`: uma linha para **todo** candidato tentado, incluindo score/rank, cobertura histórica, capital, deltas, métricas e tempos.
- `preselector_recall.csv`: recall/precision dos candidatos economicamente positivos por Top-K do pré-seletor.
- `exact_baseline.json`: resultado e tempo da execução exata do baseline.
- `baseline_history_integrity.csv`: contrato de histórico do universo baseline.
- `exact_search_manifest.json`: identidade reproduzível do experimento.
- `exact_search_summary.json`: contagens, taxa de positivos, melhor delta e distribuição do custo de replay.

## Critério desta versão

A v1 não tenta provar generalização. Ela estabelece o **ground truth (referência econômica real)** e mede o custo.

Depois do ZIP desta execução, a próxima alteração só será feita a partir dos dados de tempo e equivalência:

- se o pré-seletor concentra os positivos, podemos reduzir a fila com uma meta de recall explicitamente medida;
- para acelerar o juiz, cada otimização precisa passar por parity testing (teste de equivalência) contra esta v1 em candidatos positivos e negativos conhecidos;
- somente depois construiremos o outer walk-forward (walk-forward externo) para verificar generalização da seleção.

Não fazer merge/tag de produção com esta branch de pesquisa.
