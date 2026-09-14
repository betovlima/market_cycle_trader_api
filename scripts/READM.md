# Contextual Marginal Signature — pesquisa viva

Branch permanente: `research/contextual-marginal-signature-v1`

Coleta contrafactual congelada: `contextual-marginal-signature-v1.0.17.5`

Research Tournament atual: `contextual-marginal-signature-v1.0.20.0`

Este é o único documento vivo desta linha. Não criar READMEs por versão; o histórico do Git preserva a evolução.

## Objetivo científico

O objetivo final continua sendo aumentar o capital composto do Market Cycle Trader.

A pergunta central é:

> Entre ativos externos já pré-selecionados, qual candidato deve entrar no universo agora para melhorar o capital futuro sem desorganizar oportunidades que já estavam bem posicionadas?

A metáfora operacional adotada a partir da v1.0.20 é a mesa de bilhar:

- o universo atual é a mesa;
- os ativos atuais são as bolas já posicionadas;
- uma oportunidade futura é uma caçapa;
- um candidato externo é uma nova possibilidade de tacada;
- uma boa inserção pode criar um ângulo que o universo não possuía;
- uma inserção ruim pode perturbar a política antes mesmo de o candidato ser escolhido e destruir alinhamentos que já existiam.

O objetivo não é eliminar toda mudança. O objetivo é distinguir mudança produtiva de perturbação inútil.

## Campanha congelada v1.0.17.5

A coleta terminou com 322 observações:

- 23 estados temporais entre 2020 e 2026;
- 2 universos: `Original25` e `Original24_MinusADM`;
- 7 candidatos: `XSD`, `MKSI`, `GKOS`, `CLMT`, `CORT`, `APD`, `CCK`;
- horizonte contrafactual máximo de 40 sessões;
- Strategy #10 e snapshot de mercado congelados;
- MongoDB local como fonte persistente de verdade.

Target direto persistido:

`source_direct_delta_log_capital = log(W_policy_with_candidate_added / W_baseline_policy_without_candidate)`

O MongoDB também contém as traces de `baseline`, `policy` e `forced`, incluindo o caminho de ativos selecionados e a evolução do capital. A v1.0.20 reutiliza essas traces; não refaz os rollouts caros.

## v1.0.18.1 — Snapshot Tournament

A representação transversal em `t` produziu:

- baseline histórico por candidato: `1.1592x`;
- Ridge: `1.0399x`;
- Elastic Net: `1.5096x`;
- LightGBM: `1.4512x`;
- MLP: `1.2511x`.

O Elastic Net venceu no desenvolvimento, mas falhou no holdout:

- `Original25` 2026: `0.9790x`;
- `Original24_MinusADM` 2026: `0.9483x`;
- status: `DEVELOPMENT_SIGNAL_NOT_CONFIRMED`.

Conclusão: o snapshot não forneceu uma regra generalizável suficiente.

## v1.0.19.0 — Temporal Trajectory Tournament

A trajetória pré-decisão adicionou 270 features sobre o snapshot. O melhor resultado foi LightGBM com cerca de `1.1995x` no desenvolvimento, inferior ao snapshot vencedor `1.5096x`, e o holdout permaneceu negativo.

Status:

`TRAJECTORY_ADDED_VALUE_NOT_FOUND`

Conclusão: adicionar trajetória tabular à mesma formulação não resolveu o problema. Isso encerra a família snapshot + trajetória tabular como tentativa de previsão direta do efeito marginal.

## v1.0.20.0 — Opportunity Hole / Table Geometry Decomposition

A v1.0.20 não treina um novo modelo. Primeiro testa se o mecanismo que queremos realmente existe nos dados já coletados.

Pergunta:

> Quando a estratégia perde em uma janela futura, o problema era falta de oportunidade no universo ou falha da política em escolher uma oportunidade que já existia?

A análise é executada nos horizontes:

- 5 sessões;
- 10 sessões;
- 20 sessões;
- 40 sessões.

O horizonte primário é 20 sessões, aproximadamente um mês de pregão. Os demais funcionam como análise de sensibilidade.

### Classificação de cada estado

Para cada data e universo:

`COVERED`

A estratégia-base terminou positivamente na janela. A mesa conseguiu converter alguma oportunidade em capital.

`POLICY_MISS`

A estratégia-base não terminou positivamente, mas ao menos um ativo já presente no universo teve retorno futuro líquido positivo. Havia bola com ângulo; a política não aproveitou.

`UNIVERSE_HOLE`

A estratégia-base não terminou positivamente e nenhum incumbente teve retorno futuro líquido positivo. A mesa não continha uma bola com ângulo positivo para aquela janela.

Essa classificação é retrospectiva e serve como label de mecanismo, não como previsão.

### Candidato que preenche um buraco

Em um `UNIVERSE_HOLE`, um candidato é marcado como `fills_universe_hole` quando o próprio ativo possui retorno futuro líquido positivo naquela janela.

Isso ainda não é suficiente. Para ser considerado `clean_coverage`, ele precisa simultaneamente:

1. preencher um `UNIVERSE_HOLE`;
2. aumentar o capital da política contra o baseline naquela mesma janela;
3. ser efetivamente selecionado pela política;
4. não haver divergência do caminho de ativos selecionados antes da primeira seleção do candidato.

A condição 4 separa duas situações:

`entrada produtiva`

A trajetória-base permanece intacta até o momento em que o candidato realmente entra.

`perturbação indireta`

A simples presença do candidato altera a política antes de ele próprio ser escolhido.

Isso implementa a metáfora do bilhar: uma nova bola só é considerada cobertura limpa quando abre uma caçapa que estava sem ângulo sem desorganizar a mesa antes de participar da jogada.

### Efeito sobre estados já cobertos

Nos estados `COVERED`, a análise também mede:

- quantas vezes um candidato reduziu o capital;
- custo logarítmico dessa perturbação;
- ganho de cobertura limpa em buracos;
- `table_improvement_log = clean_coverage_log_gain - covered_disruption_cost_log`.

Isso é diagnóstico retrospectivo. Ainda não é uma regra de produção.

### Status possíveis

- `NO_UNIVERSE_HOLES_FOUND`
- `TESTED_CANDIDATES_DO_NOT_FILL_HOLES`
- `COVERAGE_FOUND_BUT_NOT_CLEAN`
- `CLEAN_OPPORTUNITY_COVERAGE_OBSERVED`

Somente o último demonstra que o mecanismo existe retrospectivamente.

Mesmo nesse caso, ainda não há claim preditivo. A etapa seguinte teria de ser pré-registrada separadamente:

1. prever, usando somente informação disponível em `t`, se um `UNIVERSE_HOLE` está se formando;
2. escolher qual candidato tem maior probabilidade de cobri-lo;
3. exigir baixo risco de perturbação de estados já cobertos;
4. confirmar tudo OOS (fora da amostra) antes de qualquer backtest final do sistema completo.

Se `POLICY_MISS` dominar, a prioridade deve voltar para a política de seleção, não para adicionar ativos.

## Como executar

A coleta antiga continua disponível:

```bat
python scripts\research_contextual_marginal_signature.py ^
  --strategy-sequence 10 ^
  --env-file .env
```

A decomposição v1.0.20.0 usa a mesma chamada de Tournament e lê somente MongoDB local:

```bat
python scripts\research_contextual_marginal_signature.py ^
  --phase tournament ^
  --strategy-sequence 10 ^
  --env-file .env
```

Não apagar as coleções `research_contextual_signature_*`.

## Logs da v1.0.20

A v1.0.20 acrescenta logs próprios sem renomear logs da campanha congelada:

- `[hole]`
- `[hole-state]`
- `[hole-candidate]`
- `[hole-summary]`
- `[decision]`
- `[complete]`

Exemplo conceitual:

```text
[hole-state] ... | Original25 | horizon=20 | class=UNIVERSE_HOLE | ...
[hole-candidate] ... | GKOS | future=... | direct=... | selected=yes | pre-perturb=no | clean=yes
[hole-summary] Original25 | horizon=20 | covered=... | policy_miss=... | universe_hole=... | fillable=... | clean=...
```

## Persistência

Fonte de verdade: MongoDB local.

Coleções consumidas:

- `research_contextual_signature_runs`
- `research_contextual_signature_observations`
- `research_contextual_signature_trace_runs`
- `research_contextual_signature_trace_rows`

Coleção de resultado:

- `research_contextual_signature_tournaments`

Contrato de artefatos permanece estável:

- entrypoint: `scripts/research_contextual_marginal_signature.py`
- pasta: `research_output/contextual_marginal_signature/`
- ZIP: `research_output/contextual_marginal_signature.zip`
- filesystem: somente exportação.

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

A v1.0.20 responde somente se existe retrospectivamente um mecanismo de `Opportunity Hole` e `Clean Coverage`.

Ela não autoriza promoção de modelo.

Se houver `CLEAN_OPPORTUNITY_COVERAGE_OBSERVED`, a próxima hipótese científica passa a ser previsão pré-decisão do buraco e da cobertura.

Se os estados ruins forem majoritariamente `POLICY_MISS`, a evidência aponta para melhorar a política de decisão com o universo atual antes de ampliar Asset Discovery.

A meta final continua inalterada:

`estratégia atual` versus `estratégia atual + seletor aprendido`

e a comparação final continua sendo capital composto robusto OOS.
