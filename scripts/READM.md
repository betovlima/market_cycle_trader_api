# Contextual Marginal Signature — pesquisa viva

Branch permanente: `research/contextual-marginal-signature-v1`

Coleta contrafactual congelada: `contextual-marginal-signature-v1.0.17.5`

Research Tournament atual: `contextual-marginal-signature-v1.0.18.0`

Este é o único documento vivo desta linha de pesquisa. O histórico do Git preserva versões anteriores; não criar READMEs por versão.

## Objetivo científico

O objetivo final continua sendo aumentar o capital composto do Market Cycle Trader.

A pergunta específica desta linha é:

> Entre ativos externos que já passaram pela pré-seleção, conseguimos aprender uma função que diga qual ativo deve ser inserido agora no universo para aumentar o capital futuro da estratégia?

O target principal volta a ser o efeito marginal direto da inserção:

`Y_direct(t,a,H) = log(W_policy_with_candidate_added / W_baseline_policy_without_candidate)`

A política continua decidindo normalmente depois que o candidato é adicionado. Não usamos identidade fixa do candidato como feature do modelo principal; a fórmula deve generalizar para um novo ativo pré-selecionado.

O antigo `action_advantage_log` permanece armazenado como diagnóstico secundário, mas não é mais o target principal do Tournament.

## Resultado da campanha v1.0.17.5

A coleta terminou com:

- 322 observações
- 23 estados temporais entre 2020 e 2026
- 2 universos: `Original25` e `Original24_MinusADM`
- 7 candidatos: `XSD`, `MKSI`, `GKOS`, `CLMT`, `CORT`, `APD`, `CCK`
- horizonte de 40 decision points / 39 transições
- Strategy #10 congelada
- MongoDB local como fonte de verdade
- switch margin congelada fold a fold
- traces e observações persistidos no MongoDB

No target correto `source_direct_delta_log_capital`:

- 73 inserções aumentaram o capital
- 65 reduziram o capital
- 184 não alteraram o capital
- em 23 dos 46 contextos havia pelo menos um candidato com efeito positivo
- em 23 dos 46 contextos a melhor decisão era não inserir nenhum candidato

No `Original25`:

- 161 observações
- 44 positivas
- 31 negativas
- 86 neutras
- 12 das 23 datas possuíam pelo menos uma inserção positiva
- 11 das 23 datas não possuíam inserção positiva

Isso demonstra oportunidade econômica retrospectiva, mas ainda não demonstra previsibilidade ex ante.

## v1.0.18.0 — Direct Marginal Capital Tournament

A v1.0.18.0 não repete os caros rollouts contrafactuais. Ela reutiliza exclusivamente as observações congeladas no MongoDB e reconstrói features disponíveis antes da decisão a partir do histórico de mercado local.

O Tournament usa como features:

- estado atual do candidato
- diferença candidato versus média do universo
- z-score do candidato dentro do universo
- percentil/rank relativo do candidato
- média e dispersão do universo
- estado de SPY como contexto de mercado

As features são derivadas das `ROTATION_FEATURES` já usadas pelo motor. A identidade do candidato não entra no modelo principal.

Modelos comparados com parâmetros fixos e modestos:

- Ridge
- Elastic Net
- LightGBM, quando disponível
- MLP compacto

Também existe um baseline simples de média histórica por candidato. O modelo contextual precisa superá-lo; caso contrário, não demonstrou que aprendeu contexto.

### Validação

Para evitar pseudo-replicação, `Original25` é o universo primário. Os 23 estados temporais, e não as 322 linhas, são tratados como a unidade temporal relevante.

- desenvolvimento: expanding chronological walk-forward até 2025
- mínimo de 8 estados anteriores antes da primeira previsão
- abstenção: inserir somente se a maior previsão de `Y_direct` for maior que zero
- seleção de modelo: somente pelo desenvolvimento cronológico
- confirmação reservada: estados de 2026
- `Original24_MinusADM`: apenas teste de robustez, não conta como novo estado temporal independente

Métrica econômica principal:

`capital_multiplier_vs_no_insert = exp(sum(realized_direct_log))`

Ela representa o multiplicador de capital marginal obtido pelas decisões do seletor em relação a não inserir candidato nos contextos avaliados.

### Regra terminal

O Tournament produz um dos quatro estados:

- `NO_CONTEXTUAL_SIGNAL`: o melhor modelo contextual não supera o baseline histórico; encerrar esta família de features/modelos.
- `DEVELOPMENT_SIGNAL_NOT_CONFIRMED`: houve sinal em desenvolvimento, mas ele falhou nos estados reservados de 2026; não promover.
- `HOLDOUT_SIGNAL_NOT_ROBUST`: passou no `Original25` de 2026, mas não sobreviveu ao universo perturbado; não promover.
- `CONFIRMED_LIMITED_NEXT_SYSTEM_BACKTEST`: passou pelos três níveis; congelar o vencedor e executar um único backtest sequencial final do sistema completo.

Não adicionar modelos indefinidamente para fabricar um vencedor.

## Como executar

A coleta antiga continua disponível pelo mesmo entrypoint:

```bat
python scripts\research_contextual_marginal_signature.py ^
  --strategy-sequence 10 ^
  --env-file .env
```

Para o Tournament v1.0.18.0, sem repetir os rollouts:

```bat
python scripts\research_contextual_marginal_signature.py ^
  --phase tournament ^
  --strategy-sequence 10 ^
  --env-file .env
```

O Tournament exige uma campanha 23 × 2 × 7 concluída no MongoDB local para a Strategy selecionada.

## Persistência

Coleções da campanha:

- `research_contextual_signature_runs`
- `research_contextual_signature_observations`
- `research_contextual_signature_trace_runs`
- `research_contextual_signature_trace_rows`

Coleção do Tournament:

- `research_contextual_signature_tournaments`

Contrato de artefatos permanece estável:

- entrypoint: `scripts/research_contextual_marginal_signature.py`
- pasta: `research_output/contextual_marginal_signature/`
- ZIP: `research_output/contextual_marginal_signature.zip`
- filesystem: somente exportação
- MongoDB local: fonte persistente de verdade

## Estrutura ativa

```text
scripts/
  READM.md
  research_contextual_marginal_signature.py
  research_contextual_signature_campaign_runner.py
  research_contextual_signature_campaign.py
  research_contextual_signature_protocol.py
  research_contextual_signature_runtime.py
  research_contextual_signature_storage.py
  research_contextual_signature_analysis.py
  research_contextual_signature_tournament.py
```

O `campaign_runner` preserva a execução científica congelada v1.0.17.5; o entrypoint estável apenas despacha entre `campaign` e `tournament`.

## Linha de chegada

Não estamos autorizados a chamar o resultado de definitivo apenas porque um modelo vence o Tournament. A evidência temporal continua limitada a 23 estados independentes e apenas dois estados reservados em 2026.

A pesquisa termina em no máximo duas etapas adicionais:

1. Direct Marginal Capital Tournament v1.0.18.0.
2. Somente se o Tournament confirmar sinal: congelar fórmula/modelo/regra de abstenção e executar um único backtest sequencial final do sistema completo, comparando capital composto contra a estratégia atual.

Se o Tournament não demonstrar sinal robusto, o resultado negativo encerra esta linha com os dados/features atuais. Se passar, o backtest final decide se a fórmula realmente aumenta o capital da estratégia inteira.
