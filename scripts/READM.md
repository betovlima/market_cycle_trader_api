# Contextual Marginal Signature — pesquisa viva

Branch permanente: `research/contextual-marginal-signature-v1`

Coleta contrafactual congelada: `contextual-marginal-signature-v1.0.17.5`

Research Tournament atual: `contextual-marginal-signature-v1.0.19.0`

Este é o único documento vivo desta linha. Não criar READMEs por versão; o histórico do Git preserva a evolução.

## Objetivo científico

O objetivo final continua sendo aumentar o capital composto do Market Cycle Trader.

A pergunta desta pesquisa é:

> Entre ativos externos que já passaram pela pré-seleção, conseguimos aprender uma função que diga qual ativo deve ser inserido agora no universo para aumentar o capital futuro da estratégia?

Target principal:

`Y_direct(t,a,H) = log(W_policy_with_candidate_added / W_baseline_policy_without_candidate)`

A política decide normalmente depois que o candidato é adicionado. A identidade fixa do candidato não entra como feature do modelo principal; a solução deve poder generalizar para um novo ativo pré-selecionado.

O antigo `action_advantage_log` permanece no MongoDB apenas como diagnóstico secundário e não é o target do Tournament.

## Campanha congelada v1.0.17.5

A coleta terminou com 322 observações, 23 estados temporais entre 2020 e 2026, dois universos (`Original25` e `Original24_MinusADM`) e sete candidatos (`XSD`, `MKSI`, `GKOS`, `CLMT`, `CORT`, `APD`, `CCK`). A Strategy #10, o snapshot de mercado e a switch margin fold a fold permanecem congelados.

No target direto:

- 73 inserções aumentaram o capital;
- 65 reduziram o capital;
- 184 não alteraram o capital;
- em 23 dos 46 contextos havia ao menos um candidato positivo;
- em 23 dos 46 contextos nenhum candidato melhorava o baseline.

No `Original25`, foram 161 observações: 44 positivas, 31 negativas e 86 neutras. Em 12 das 23 datas existia alguma inserção positiva; em 11 não existia.

Isso comprova oportunidade econômica retrospectiva, não previsibilidade ex ante.

## v1.0.18.1 — resultado do Snapshot Tournament

A v1.0.18.x reutilizou exclusivamente os labels já persistidos no MongoDB e reconstruiu features disponíveis antes da decisão. A representação foi um snapshot transversal em `t`:

- estado atual do candidato;
- diferença candidato versus média do universo;
- z-score e percentil/rank relativo;
- média e dispersão do universo;
- estado de SPY.

Os mesmos modelos fixos foram usados: Ridge, Elastic Net, LightGBM quando disponível e MLP compacto.

Resultado de desenvolvimento:

- baseline histórico por candidato: `1.1592x`;
- Ridge: `1.0399x`;
- Elastic Net: `1.5096x`;
- LightGBM: `1.4512x`;
- MLP: `1.2511x`.

O Elastic Net venceu no desenvolvimento, mas falhou no trecho reservado:

- `Original25` em 2026: `0.9790x`;
- `Original24_MinusADM` em 2026: `0.9483x`;
- decisão: `DEVELOPMENT_SIGNAL_NOT_CONFIRMED`.

Conclusão limitada da v1.0.18.1: o snapshot atual não produziu regra generalizável suficiente. Isso não demonstra que o efeito marginal seja imprevisível; deixa aberta uma única família de falha: o snapshot pode ter perdido a trajetória que levou ao estado atual.

## v1.0.19.0 — Temporal Trajectory Tournament

A v1.0.19.0 testa somente a hipótese seguinte:

> A trajetória anterior à decisão contém informação preditiva adicional que o snapshot em `t` não contém?

Ela não refaz os 322 rollouts. O target, os estados, os universos, os candidatos, o holdout de 2026 e os modelos permanecem iguais. A única mudança científica é a representação da informação pré-decisão.

### Representação

A v1.0.19.0 mantém todas as features do snapshot v1.0.18.1 e adiciona trajetória para dez fontes previamente fixadas:

- `return_5`
- `return_20`
- `return_60`
- `vol_20`
- `ema_distance_20`
- `ema_20_vs_50`
- `rsi_14`
- `atr_pct_14`
- `trend_efficiency_20`
- `momentum_acceleration_5_20`

Janelas pré-decisão fixas: 5, 20 e 60 sessões.

Para cada fonte e janela são derivados, usando somente dados até a data da decisão:

- delta do candidato;
- inclinação do candidato;
- delta candidato versus média do universo;
- inclinação relativa;
- dispersão da trajetória relativa;
- mudança de percentil/rank;
- inclinação do rank;
- delta de SPY;
- inclinação de SPY.

A implementação corta explicitamente cada série em `index <= decision` antes de calcular a janela. Nenhuma observação posterior à decisão pode entrar nas features de trajetória.

### Comparação controlada

A pergunta não é se um novo modelo qualquer consegue vencer. A comparação é:

`snapshot v1.0.18.1` versus `snapshot + trajetória v1.0.19.0`

com os mesmos quatro modelos e os mesmos splits cronológicos.

A v1.0.19.0 primeiro reproduz os logs e métricas do snapshot com os prefixos já existentes:

- `[tournament]`
- `[development]`
- `[winner]`

Depois acrescenta, sem renomear os anteriores:

- `[trajectory]`
- `[trajectory-development]`
- `[trajectory-winner]`
- `[comparison]`
- `[decision]`
- `[complete]`

### Validação

- universo primário: `Original25`;
- desenvolvimento: expanding chronological walk-forward até 2025;
- mínimo de oito estados anteriores antes da primeira previsão;
- abstenção: só inserir quando a maior previsão de `Y_direct` for positiva;
- escolha do modelo somente no desenvolvimento;
- 2026 permanece reservado;
- `Original24_MinusADM` permanece apenas como robustez, não como estado independente adicional.

Métrica econômica principal:

`capital_multiplier_vs_no_insert = exp(sum(realized_direct_log))`

### Regra terminal da trajetória

- `TRAJECTORY_ADDED_VALUE_NOT_FOUND`: a melhor representação com trajetória não supera a melhor representação snapshot no desenvolvimento.
- `TRAJECTORY_CONTEXTUAL_SIGNAL_NOT_FOUND`: a trajetória não supera o baseline histórico.
- `TRAJECTORY_SIGNAL_NOT_CONFIRMED`: melhora no desenvolvimento, mas falha no `Original25` reservado de 2026.
- `TRAJECTORY_HOLDOUT_NOT_ROBUST`: passa no `Original25` de 2026, mas falha em `Original24_MinusADM`.
- `TRAJECTORY_CONFIRMED_LIMITED_NEXT_SYSTEM_BACKTEST`: passa pelos níveis acima e ganha direito a um único backtest sequencial final do sistema completo.

Se a trajetória não for confirmada, não adicionar sucessivamente mais modelos tabulares a este mesmo dataset congelado apenas para fabricar um vencedor.

## Como executar

A coleta contrafactual antiga continua disponível pelo entrypoint estável:

```bat
python scripts\research_contextual_marginal_signature.py ^
  --strategy-sequence 10 ^
  --env-file .env
```

Tournament v1.0.19.0, sem repetir os rollouts:

```bat
python scripts\research_contextual_marginal_signature.py ^
  --phase tournament ^
  --strategy-sequence 10 ^
  --env-file .env
```

## Persistência e artefatos

MongoDB local é a fonte persistente de verdade.

Coleções da campanha:

- `research_contextual_signature_runs`
- `research_contextual_signature_observations`
- `research_contextual_signature_trace_runs`
- `research_contextual_signature_trace_rows`

Coleção dos Tournaments:

- `research_contextual_signature_tournaments`

Contrato de artefatos permanece estável:

- entrypoint: `scripts/research_contextual_marginal_signature.py`
- pasta: `research_output/contextual_marginal_signature/`
- ZIP: `research_output/contextual_marginal_signature.zip`
- filesystem: somente exportação, nunca entrada obrigatória para análise posterior.

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

Não adicionar arquivo novo para cada versão.

## Linha de chegada

A linha atual é curta:

1. executar a v1.0.19.0 e decidir se a trajetória acrescenta previsibilidade real ao snapshot;
2. somente se o status for `TRAJECTORY_CONFIRMED_LIMITED_NEXT_SYSTEM_BACKTEST`, congelar representação + modelo + parâmetros + regra de abstenção e executar um único backtest sequencial do sistema completo;
3. no backtest final, comparar diretamente o capital composto da estratégia atual contra a estratégia atual + seletor aprendido.

Se a v1.0.19.0 falhar, o resultado negativo encerra esta família de representação tabular de trajetória sobre o dataset congelado. Se passar, o capital final do sistema completo decide se a descoberta merece integração no MCT.
