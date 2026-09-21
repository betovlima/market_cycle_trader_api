# Execução final do TCC — config congelada 10.8.76 / runner 10.8.77

## Objetivo

Fechar a pesquisa do TCC com uma execução independente e reproduzível do
Control versus Soft Horizon Consensus usando uma carga nova e completa da
Alpaca, sem reutilizar dados de mercado do MongoDB, caches de backtests ou
resultados anteriores como entrada.

O resultado anterior de aproximadamente US$ 7,38 milhões para o Soft Horizon
Consensus é uma hipótese de reprodução, não uma meta de otimização. Esta execução
não deve fazer tuning para tentar recuperar esse valor. O valor obtido com o
snapshot congelado será registrado como resultado final.

## Versão

- API/pacote: `10.8.76`
- Script: `final-standalone-alpaca-v2`
- Branch: `research/api-v10.8.76-final-standalone-alpaca-gpu`
- Config congelada: `research/final_research_v10_8_76.json`

A branch deriva da linha final `research/api-v10.8.75-final-standalone-alpaca`,
que já implementava a pesquisa standalone sem MongoDB.

## Janela temporal fechada

- início: `2016-01-01`
- fim: `2026-09-18`
- análise: `2016-01-01 -> 2026-09-18`

Não estender a janela após a execução final sem abrir uma nova hipótese de
pesquisa.

## Fonte de dados

Fluxo obrigatório:

```text
Alpaca API
  -> barras RAW SIP
  -> corporate actions point-in-time
  -> snapshot local imutável
  -> normalização local de splits
  -> validação de identidade/histórico
  -> Control
  -> Soft Horizon Consensus
  -> comparação final
```

O script não deve ler nem escrever dados de mercado no MongoDB.

Configuração congelada:

- `market_data_provider=alpaca`
- `alpaca_historical_feed=sip`
- `alpaca_adjustment=raw`
- `research_market_data_mode=standalone_snapshot`
- `mongo_cache_enabled=false`
- tuning desativado
- Optuna desativado
- CARO desativado

## GPU

A configuração computacional deve vir do `.env`:

```env
MCT_ROTATION_ACCELERATOR=cuda
MCT_ROTATION_ALLOW_CPU_FALLBACK=false
```

A config final não fixa mais `rotation_accelerator=cpu`. O schema resolve os
valores pelo ambiente.

`deterministic_execution=false` é necessário nesta execução porque a
implementação LightGBM GPU rejeita CUDA quando o determinismo estrito está
habilitado.

O runner valida isso antes da pesquisa. Se CUDA não estiver configurado ou se
fallback para CPU estiver habilitado, a execução aborta. Se o backend GPU do
LightGBM não puder ser inicializado, a execução também deve falhar em vez de
continuar silenciosamente em CPU.

Os resultados registram:

- `requested_compute_device`
- `effective_compute_device`

No Windows, o esperado é `requested_compute_device=cuda` e backend efetivo
LightGBM `gpu`.

## Ativos com problema estrutural

A política do TCC é não reconstruir, fazer bridge entre fontes ou corrigir
manualmente ativos com quebra estrutural de identidade/histórico.

A config final contém exclusão explícita:

```text
CLMT
reason=known_structural_history_identity_discontinuity
policy=exclude_do_not_bridge_or_reconstruct
```

Além das exclusões explícitas, o pipeline continua aplicando o guard automático
de corporate actions para mudanças estruturais detectáveis. Toda exclusão deve
ser registrada em `excluded_assets.csv` e no resumo final.

## Experimento

Somente duas variantes devem ser executadas sobre o mesmo snapshot e os mesmos
folds cronológicos:

1. Control: Soft Horizon Consensus desativado.
2. Soft Horizon Consensus: `enabled=true`, `penalty_strength=1.0`.

Nenhum hiperparâmetro deve ser ajustado após a carga do snapshot final.

## Obter a branch

```bash
git fetch origin
git switch research/api-v10.8.76-final-standalone-alpaca-gpu
git pull --ff-only origin research/api-v10.8.76-final-standalone-alpaca-gpu
```

Confirmar o ambiente:

```bash
python -c "import os; print('accelerator=', os.getenv('MCT_ROTATION_ACCELERATOR')); print('fallback=', os.getenv('MCT_ROTATION_ALLOW_CPU_FALLBACK'))"
```

Executar os testes locais:

```bash
python -m pytest tests/test_final_standalone_alpaca.py -q
```

## Execução final com nova carga Alpaca

Usar `--replace-snapshot` para garantir uma carga nova:

```bash
python scripts/research_final_standalone_alpaca.py --replace-snapshot
```

Não usar `--reuse-snapshot` na primeira execução final.

O output padrão é:

```text
output/final_standalone_alpaca_20260918_v10876/
```

## Artefatos principais

Esperados no output:

- `snapshot/manifest.json`
- snapshot das barras RAW por ativo
- snapshot de corporate actions
- `results/control_predictions.csv`
- `results/control_trades.csv`
- `results/soft_horizon_consensus_predictions.csv`
- `results/soft_horizon_consensus_trades.csv`
- `results/soft_horizon_consensus_decisions.csv`
- `results/excluded_assets.csv`
- resumo final com hashes, versões, métricas e comparação

O snapshot deve registrar SHA-256 para permitir reprodução posterior.

## Critério de encerramento da pesquisa

A execução final não é aprovada ou reprovada por atingir um número específico de
capital. O objetivo é testar se o ganho previamente observado do Soft Horizon
Consensus se reproduz em uma nova carga Alpaca dentro da janela temporal já
congelada.

Registrar:

- capital final do Control;
- capital final do Soft Horizon Consensus;
- diferença absoluta e relativa;
- CAGR;
- Sharpe;
- maximum drawdown;
- pior fold;
- número e taxa de decisões alteradas pelo consenso;
- ativos excluídos;
- requested/effective compute device;
- hash do snapshot;
- hash da configuração;
- versões das bibliotecas.

Se o ganho próximo à faixa anterior de US$ 7 milhões persistir, ele pode compor
o resultado final do TCC. Se não persistir, registrar o resultado efetivamente
obtido sem novo tuning.


## Correção 10.8.77 — dtype na normalização de splits

Na primeira execução final com carga nova da Alpaca, o download e a criação do
snapshot foram concluídos. A execução falhou depois, durante a normalização local
de splits, com:

```text
pandas.errors.LossySetitemError
TypeError: Invalid value [...] for dtype 'int64'
```

A causa não foi um problema nos dados da Alpaca. A coluna `volume` do arquivo
RAW foi inferida como `int64`. Para splits cuja razão gera unidades
fracionárias, por exemplo 3:2, a normalização calcula um volume `float`.
Versões recentes do pandas recusam atribuir silenciosamente esse valor
fracionário de volta a uma coluna inteira.

A correção converte explicitamente todas as colunas OHLCV
(`open/high/low/close/volume`) para `float` antes de aplicar qualquer fator de
split. Isso preserva a precisão da normalização e evita coerção implícita
dependente da versão do pandas.

Versão da correção:

- API/pacote: `10.8.77`
- runner: `final-standalone-alpaca-v2.0.1`
- branch: `research/api-v10.8.77-final-standalone-alpaca-split-dtype`
- configuração científica permanece congelada em:
  `research/final_research_v10_8_76.json`

Foi adicionado teste de regressão com OHLCV `int64` e split 3:2 para verificar
que preços e volumes fracionários são normalizados sem erro e que todas as
colunas OHLCV resultantes são `float`.

### Retomada da execução final

Não executar `--replace-snapshot` novamente. O snapshot novo da Alpaca já foi
baixado e congelado antes da falha. A correção é apenas de processamento local.

Atualizar a branch:

```bash
git fetch origin
git switch research/api-v10.8.77-final-standalone-alpaca-split-dtype
git pull --ff-only origin research/api-v10.8.77-final-standalone-alpaca-split-dtype
```

Executar os testes:

```bash
python -m pytest tests/test_final_standalone_alpaca.py -q
```

Retomar usando exatamente o snapshot já criado:

```bash
python scripts/research_final_standalone_alpaca.py --reuse-snapshot
```

O runner verifica o hash da configuração e os hashes dos arquivos do snapshot
antes de reutilizá-los. Assim, a correção 10.8.77 não altera a carga final da
Alpaca nem a janela temporal congelada; apenas permite que o pipeline continue a
partir dos mesmos dados.
