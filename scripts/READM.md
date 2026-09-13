# Contextual Marginal Signature — pesquisa viva

Branch permanente: `research/contextual-marginal-signature-v1`

Runtime científico atual: `contextual-marginal-signature-v1.0.17.5`

Este é o único documento vivo da linha Contextual Marginal Signature. Não criar READMEs por versão. O histórico do Git preserva implementações e documentos antigos.

## Comando atual

Na raiz do repositório:

```bat
python scripts\research_contextual_marginal_signature.py ^
  --strategy-sequence 10 ^
  --env-file .env
```

Contrato estável:

- entrypoint: `scripts/research_contextual_marginal_signature.py`
- saída: `research_output/contextual_marginal_signature/`
- ZIP: `research_output/contextual_marginal_signature.zip`
- fonte persistente de verdade: MongoDB local
- filesystem: somente exportação, nunca entrada obrigatória para análises futuras

Coleções:

- `research_contextual_signature_runs`
- `research_contextual_signature_observations`
- `research_contextual_signature_trace_runs`
- `research_contextual_signature_trace_rows`

## Pergunta científica

A pesquisa deixou de perguntar se um ativo possui uma assinatura marginal fixa e passou a formular uma decisão causal:

> Dado o estado observado antes da decisão, vale a pena forçar o candidato `a` agora ou é melhor deixar a política normal decidir?

Target atual:

`Y(t,a,H) = log(W_forced_candidate_then_policy / W_normal_policy)`

Os dois braços contêm o candidato. O braço forçado altera somente a primeira decisão causal e depois retorna à mesma política a partir do estado resultante do simulador.

A política normal é uma ação válida com vantagem zero. Portanto, o problema final inclui abstenção:

`max(0, Y_candidate_1, ..., Y_candidate_k)`

## Campanha temporal atual

- 23 estados temporais executáveis XNYS entre 2020 e 2026
- 2 universos: `Original25` e `Original24_MinusADM`
- 7 candidatos: `XSD`, `MKSI`, `GKOS`, `CLMT`, `CORT`, `APD`, `CCK`
- horizonte de 40 decision points / 39 transições
- 322 observações de vantagem de ação pareada
- Strategy #10 congelada
- market data exclusivamente do MongoDB local durante a campanha
- switch margin congelada fold a fold a partir do baseline
- checkpoint idempotente e resume automático

Datas:

- 03/08/2020
- 02/11/2020
- 01/02/2021
- 03/05/2021
- 02/08/2021
- 01/11/2021
- 01/02/2022
- 02/05/2022
- 01/08/2022
- 01/11/2022
- 01/02/2023
- 01/05/2023
- 01/08/2023
- 01/11/2023
- 01/02/2024
- 01/05/2024
- 01/08/2024
- 01/11/2024
- 03/02/2025
- 01/05/2025
- 01/08/2025
- 02/02/2026
- 01/06/2026

## Evolução científica

### v1.0.0–v1.0.5 — primeiras assinaturas contextuais

Os primeiros experimentos usaram capital marginal exato como target e conjuntos pequenos de features retrospectivas. Modelos lineares e árvores rasas mostraram ranking fraco e instável fora do tempo. Isso eliminou a hipótese simples de uma assinatura fixa do ativo.

### v1.0.6–v1.0.9 — contexto e composição do universo

Experimentos leave-one-out e universos reduzidos mostraram que a contribuição aparente de um candidato podia mudar com o universo ao redor. ADM e ADI foram casos diagnósticos importantes. O target relativo foi corrigido para impedir que mudanças no baseline fossem confundidas com efeito próprio do candidato.

### v1.0.11–v1.0.14.4 — atribuição de caminho e calibração congelada

Os traces completos revelaram que a simples presença de um candidato podia alterar a calibração da switch margin mesmo quando o candidato não era negociado. A pesquisa separou participação direta de recalibração da política e passou a congelar fold a fold a margem escolhida pelo baseline.

Na campanha de prevalência do efeito direto, 27 de 84 observações tiveram efeito direto não nulo, distribuídas entre sinais positivos e negativos e associadas à participação real do candidato.

### v1.0.13–v1.0.15 — representação do caminho

A Log-Signature de nível 2 mostrou que interações candidato × universo podem ser representadas sem cancelamento simples. Porém, o primeiro benchmark Ridge com todos os termos LogSig não produziu ranking temporal robusto. A conclusão foi de capacidade representacional, não de previsibilidade demonstrada.

### v1.0.16 — Paired Action Advantage

O target passou a comparar diretamente:

`log(W_force_candidate_once_then_policy / W_policy)`

A vantagem de ação ficou densa: 82 de 84 observações foram não nulas. A data/contexto explicou muito mais variação que a identidade do candidato. A pergunta seguinte ficou definida:

> Conseguimos prever essa vantagem antes da decisão?

### v1.0.17 — expansão temporal

A campanha aumentou de 6 para 23 estados independentes, mantendo os dois universos e os sete candidatos. O objetivo é obter diversidade temporal suficiente para avaliar previsão cronológica e ranking dentro de cada contexto.

### v1.0.17.1–v1.0.17.3 — execução resiliente

Foram incorporados:

- preflight das janelas walk-forward OOS
- serialização BSON segura para `NaT`, `NaN` e infinitos
- checkpoints idempotentes
- resume automático pelo MongoDB
- persistência de traces no MongoDB
- logs para análise científica parcial
- cache de RAM limitado ao par policy/forced

### v1.0.17.4 — aceleração guardada e refatoração final

A aceleração reutiliza apenas trabalho invariável:

- frames de features derivados do mercado
- contexto preparado do replay dentro do par
- modelos LightGBM ajustados dentro do par
- utilities/predições brutas dentro do par

Nunca são reutilizados:

- posição
- holding days
- trades
- estado do simulador
- trajetória de capital
- diagnósticos dependentes da intervenção

O primeiro par elegível executa uma referência sem cache e uma execução acelerada. A campanha só continua se capital, sessões, predictions e trades forem equivalentes com tolerância `1e-12`.

Nesta mesma versão técnica, a implementação incremental histórica foi achatada. O código ativo deixou de depender de aliases `v111`, `v1144`, `v116`, `v117`, `v1172` ou wrappers equivalentes. Os scripts de pesquisas anteriores foram removidos desta branch; o histórico permanece no Git.

### v1.0.17.5 — diagnóstico de divergência e fallback seguro

O primeiro teste real da v1.0.17.4 detectou divergência entre a referência sem cache e a execução acelerada e abortou corretamente antes de confiar no cache.

A v1.0.17.5 mantém o mesmo protocolo científico e torna esse guard mais informativo e resiliente:

- primeiro compara `uncached reference` contra `accelerated`;
- se houver divergência, executa uma segunda repetição sem cache;
- se as duas execuções sem cache forem equivalentes, a divergência é atribuída à aceleração, o cache é desativado para o processo e a campanha continua pela execução sem cache confiável;
- se as duas execuções sem cache também divergirem, a campanha para com diagnóstico explícito de não determinismo subjacente, incluindo `deterministic_execution`, `xgb_n_jobs` e `numeric_thread_limit`;
- nenhum resultado acelerado divergente é persistido como observação científica.

Assim, uma otimização de performance não pode bloquear a pesquisa quando o caminho sem cache é estável, nem mascarar um problema real de reprodutibilidade quando o próprio replay sem cache diverge.

## Estrutura atual

A pasta `scripts` desta branch contém somente a linha ativa:

```text
READM.md
research_contextual_marginal_signature.py
research_contextual_signature_protocol.py
research_contextual_signature_campaign.py
research_contextual_signature_runtime.py
research_contextual_signature_storage.py
research_contextual_signature_analysis.py
```

Responsabilidades:

- `research_contextual_marginal_signature.py`: entrypoint estável, cache e validação de equivalência
- `research_contextual_signature_protocol.py`: protocolo contrafactual, calendário, frozen margin, forced action, captura e helpers de replay
- `research_contextual_signature_campaign.py`: campanha 23 × 2 × 7, preflight, checkpoints, resume, dataset final e export
- `research_contextual_signature_runtime.py`: cache de RAM e observabilidade da aceleração
- `research_contextual_signature_storage.py`: MongoDB, serialização, retry, traces, observações e export
- `research_contextual_signature_analysis.py`: análise parcial/final e readiness gate

Não criar novos arquivos com versão no nome. Evoluções futuras modificam estes módulos estáveis.

## Gate para benchmark

O dataset fica pronto para o Research Tournament somente quando houver:

- pelo menos 280 observações
- pelo menos 20 estados temporais
- suporte não nulo para todos os 7 candidatos
- pelo menos 7 anos representados
- targets positivos e negativos
- pelo menos 3 contextos onde intervir é melhor
- pelo menos 3 contextos onde `NORMAL POLICY` é melhor

## Linha de chegada

A pesquisa tem três etapas finais:

1. terminar a coleta contrafactual dos 23 estados;
2. executar um Research Tournament integrado usando exclusivamente o dataset/traces congelados no MongoDB;
3. congelar a solução vencedora e fazer uma confirmação cronológica final realmente reservada.

O tournament deve comparar, sob os mesmos splits cronológicos e métricas:

- Ridge / Elastic Net
- LightGBM
- MLP
- modelo neural temporal compacto
- representações com e sem informação de caminho / Log-Signature

As tarefas incluem regressão da vantagem, classificação de sinal, ranking dentro do contexto, intervenção versus abstenção e utilidade econômica da decisão.

O resultado negativo também é terminal e válido:

> Não encontramos previsibilidade suficientemente robusta antes da decisão.

Não adicionar modelos indefinidamente para fabricar um vencedor.

## Regras de segurança

- Não mudar target, estados, candidatos, universos ou frozen-margin dentro de uma otimização técnica.
- Não usar arquivos locais como fonte persistente para uma análise posterior.
- Não apagar as coleções Mongo ao atualizar o código.
- Uma falha técnica pode parar o processo, mas observações concluídas devem permanecer recuperáveis.
- Antes de confiar em uma nova otimização, exigir equivalência determinística.
- Se o cache falhar e o replay sem cache for estável, continuar sem cache em vez de sacrificar a campanha.
- Os logs devem continuar permitindo análise parcial durante a campanha.
