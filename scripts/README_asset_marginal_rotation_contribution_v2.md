# Contribuição marginal de ativos — v2.0.0

API/pacote: **10.8.41**. Scripts de seleção, Leadership e validação: **2.0.0**.
Branch: `research/asset-marginal-rotation-contribution-v2`.
Base: `research/asset-marginal-rotation-contribution-v1`, commit `10e2d46b71fc7051098116f6385f6561215cdaad`.

## Problema observado

O ZIP de 252 pregões, de 05/09/2025 a 04/09/2026, registra:

| Métrica | Universo original, 56 ativos | Expandido com UTI, 57 ativos |
|---|---:|---:|
| Capital inicial | 10.000,00 | 10.000,00 |
| Capital final | 9.716,60 | 9.517,34 |
| Retorno total | −2,8340% | −4,8266% |
| CAGR exportado pela v1 | −6,4670% | −8,3916% |
| CAGR com capital inicial correto | −2,8436% | −4,8428% |
| Rotações | 34 | 34 |

A diferença final é −199,26, ou −2,0507% em relação ao baseline. As posições
diferem em 20 pregões; o Top 1 coincide em 232. A auditoria reconcilia a soma das
diferenças diárias de log-retorno com `log(9517,339866762 / 9716,596436723)`.
Essas diferenças incluem custos e efeitos da posição anterior.

A v1 incluiu UTI por oito eventos, com soma marginal de rótulos de +0,813452.
`forward_net_log_return` é uma média ponderada de retornos futuros de vários
horizontes. Eventos próximos reutilizam trechos do futuro; somar esses rótulos
não representa o crescimento de uma única conta. A sequência de outubro/novembro
de 2020 ilustra a sobreposição. Esse problema existe mesmo com previsões OOS.

O CAGR tinha um erro separado: usava o primeiro saldo de fechamento, já alterado
pelo primeiro pregão, como denominador. A correção usa o capital inicial e mantém
a convenção de duração já adotada pelo motor. Ela altera a métrica reportada;
os saldos e negócios do backtest original permanecem os mesmos.

O resultado de 23,52 milhões cobre outro intervalo e outro protocolo de treino.
Ele não é o baseline de capital deste teste de um ano, que congela modelos e
calibração antes do período final. Esta versão não comprova a recuperação daquele resultado.

## Método novo

1. Leadership continua treinando cada ativo/fold uma vez com os modelos da Strategy.
   A exportação inclui score OOS, próxima abertura/fechamento, identificadores das
   sessões e última data de disponibilidade dos rótulos de treino.
2. O universo original permanece obrigatório. O grupo de candidatos continua
   sendo o de externos `leadership_qualified`, para isolar a mudança do objetivo marginal.
3. Para o universo atual e cada expansão proposta, uma simulação contábil percorre
   cronologicamente os mesmos pregões. A posição e o saldo atravessam os folds.
   Não há reinício da conta a cada evento nem soma de rótulos futuros sobrepostos.
4. As ações usam apenas os scores e o estado da conta. Executam na abertura
   seguinte e carregam a posição, inclusive entre fechamentos e aberturas.
   Compra, arredondamento de quantidade, taxas e slippage usam as funções do motor.
   A última posição é liquidada no fechamento final.
5. A contribuição de uma adição é `log(capital_expandido / capital_atual)`.
   Entra a melhor contribuição positiva; a trajetória inteira é recalculada após
   cada adição. Empates são resolvidos alfabeticamente. Se nenhuma contribuição
   é positiva, a seleção termina e pode preservar todos os 56 ativos sem adições.
6. Os dois universos congelados passam pela validação final completa, com o
   treinador e a calibração pré-validação já existentes.

O replay é uma simulação por candidato com previsões prontas. Não há retreino,
novo carregamento do MongoDB ou execução do motor completo/benchmark por candidato.
O manifesto declara `cached_score_replay_per_candidate=true` e
`model_refit_per_candidate=false`.

O replay usa a **margem configurada na Strategy**, com as regras existentes de
permanência mínima, CASH e custos. Não escolhe uma margem usando o OOS e não
acrescenta limiar de aceitação. A calibração de margem da validação completa pode
produzir outro valor; por isso o replay é uma medida de seleção sob política fixa,
e não uma reprodução de todas as calibrações possíveis do backtest final.
O ponto de indiferença da admissão permanece zero.

## Integridade e limites

- Cada resultado precisa de calendário completo para o baseline. Um candidato
  incompleto recebe `incomplete_oos_coverage` e não encurta a comparação.
- O replay rejeita duplicatas, preços inválidos, saltos de sessão, rótulos de
  treino posteriores à decisão e execuções posteriores ao corte de seleção.
- O contrato vincula configuração, snapshot e CSV por hashes. Preços de execução
  são resultados para contabilização; não entram nos scores ou na regra de ação.
- As validações geram hashes do OHLCV por ativo. A comparação recusa preços
  diferentes para ativos comuns, períodos/capitais distintos ou código/ambiente diferente.
- O universo ainda vem do cache local atual e do filtro Leadership anterior.
  Esta mudança não demonstra ausência de viés de sobrevivência nesse universo.
- Selecionar ativos pelo melhor resultado histórico continua sendo ajuste de
  pesquisa. Os folds ajudam a mostrar concentração da contribuição, mas não são
  um novo teste de significância e não criam outro gate.
- O período 05/09/2025–04/09/2026 já foi observado. Reexecutá-lo após esta alteração
  é comparação retrospectiva. Uma afirmação nova de desempenho exige outro
  período realmente reservado ou acompanhamento futuro após congelar a seleção.

## Baixar e executar

Na pasta do repositório, comandos compatíveis com PowerShell e Bash:

```bash
git fetch origin
git switch research/asset-marginal-rotation-contribution-v2
git pull --ff-only origin research/asset-marginal-rotation-contribution-v2
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p test_asset_marginal_v2.py -v
python scripts/research_asset_marginal_rotation_independent_then_validate.py --strategy-sequence 10 --history-start 2016-01-01 --snapshot-end 2026-09-04 --validation-sessions 252 --workers 4 --fresh-run
```

O último comando reproduz o intervalo já observado. Requer o MongoDB local com o
histórico completo e a Strategy existente; não baixa OHLCV externo.
`--fresh-run` apaga apenas a pasta de saída v2 deste experimento dentro de
`research_output`, descobre os símbolos do cache e recompõe todas as fases.
A pasta padrão inclui `cached_score_replay_v2` para separar execuções anteriores.
Leadership v2 exige `--no-resume`; o runner já fornece essa opção.

O ZIP v1 não contém aberturas/fechamentos de todos os ativos em todos os pregões
OOS. Ele permite auditoria e reprodução da seleção antiga, mas não permite
reconstruir a seleção v2 com fidelidade. O caminho normal é o novo `--fresh-run`.

Para auditar o ZIP antigo sem MongoDB ou treino:

```bash
python scripts/audit_asset_marginal_rotation_export.py --input "d5f78415-9ec4-4aff-bff8-0abb97ad0895.zip" --output-dir research_output/marginal_export_audit
```

Para reproduzir apenas o seletor antigo em arquivos já extraídos:

```bash
python scripts/research_asset_marginal_rotation_contribution.py --leadership-output-dir "CAMINHO_DA_PASTA_LEADERSHIP" --output-dir research_output/marginal_legacy_replay --selection-method legacy_event_sum
```

O serviço original `asset_marginal_rotation_contribution.py` foi preservado.

## Saídas para revisão

- `marginal_replay_contract.json`: política, corte, hashes e identidade da execução.
- `marginal_rotation_candidate_evaluations.csv`: capital, contribuição líquida,
  diferença de taxas/rotações e cobertura de cada tentativa.
- `marginal_rotation_fold_contributions.csv`: decomposição por fold; a conta continua entre folds.
- `marginal_rotation_selected_events.csv`: na v2, diferenças **diárias** da conta por
  adição aceita. A coluna relevante é `marginal_net_log_return`, não o rótulo antigo.
- `marginal_rotation_baseline_replay.csv` e `marginal_rotation_expanded_replay.csv`:
  trajetórias completas com saldo, posição, razão da decisão e log-retorno.
- Snapshots congelados e `marginal_rotation_independent_comparison.json`:
  resultado completo posterior, com verificação de preços comuns.

Os testes e a auditoria confirmam correção contábil e temporal. O retreino e
backtest reais de 56/57 ativos não foram executados aqui, pois os anexos não
incluem a base OHLCV completa. A melhora econômica permanece uma hipótese a testar.
