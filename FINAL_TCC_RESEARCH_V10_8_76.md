# Execução final do TCC — API 10.8.76

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
