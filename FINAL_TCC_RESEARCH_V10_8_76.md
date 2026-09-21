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


## Correção 10.8.78 — constante OHLCV ausente

A primeira correção de dtype da 10.8.77 passou a iterar sobre `OHLCV`, mas o
runner standalone não declarava essa constante. Isso causou:

```text
NameError: name 'OHLCV' is not defined
```

A 10.8.78 declara explicitamente:

```python
OHLCV = ("open", "high", "low", "close", "volume")
```

e mantém a conversão das cinco colunas para `float` antes da normalização de
splits.

- API/pacote: `10.8.78`
- runner: `final-standalone-alpaca-v2.0.2`
- branch: `research/api-v10.8.78-final-standalone-alpaca-ohlcv-constant`
- configuração científica continua congelada em:
  `research/final_research_v10_8_76.json`
- snapshot Alpaca existente deve continuar sendo reutilizado.

Foi adicionado um teste explícito que exige a presença da constante OHLCV com as
cinco colunas esperadas.

Retomada:

```bash
git fetch origin
git switch research/api-v10.8.78-final-standalone-alpaca-ohlcv-constant
git pull --ff-only origin research/api-v10.8.78-final-standalone-alpaca-ohlcv-constant
python -m pytest tests/test_final_standalone_alpaca.py -q
python scripts/research_final_standalone_alpaca.py --reuse-snapshot
```


## Resultado da execução final — snapshot 2026-09-21

A execução final foi concluída com API 10.8.78 sobre o snapshot Alpaca
congelado.

Identidade da execução:

- snapshot SHA-256:
  `cc5c22771c7663d249e893a8507e8a8c0e4d5508fc933492ecd41e0e93b5ea37`
- config SHA-256:
  `6bcc794ff8ea374ff25038f4a02ff15d02baf429d21a07df6aaa2dd3226d71b1`
- janela: `2016-01-01 -> 2026-09-18`
- barras: Alpaca RAW SIP
- linhas RAW: 150,808
- ativos solicitados: 56
- ativos elegíveis: 54
- MongoDB: sem leitura e sem escrita
- tuning: desativado
- Optuna: não utilizado
- CARO: não utilizado
- requested compute device: `cuda`
- effective compute device: `gpu`

Exclusões estruturais:

- DOC: `structural_identity_change`, merger DOC -> PEAK em 2024-03-01.
- CLMT: `known_structural_history_identity_discontinuity`, exclusão congelada;
  não reconstruir ou fazer bridge.

### Control

- capital final: US$ 3,543,111.38
- CAGR: 159.55%
- Sharpe: 1.9534
- maximum drawdown: -31.22%
- pior fold: +189.74%

Folds:

1. US$ 10,000.00 -> US$ 28,973.98, retorno +189.74%
2. US$ 28,973.98 -> US$ 467,811.81, retorno +1,514.59%
3. US$ 467,811.81 -> US$ 3,543,111.38, retorno +657.38%

### Soft Horizon Consensus

- capital final: US$ 3,805,617.56
- CAGR: 162.58%
- Sharpe: 1.9691
- maximum drawdown: -31.22%
- pior fold: +170.08%
- decisões base alteradas: 16 de 1,547
- taxa de alteração: 1.0343%

Folds:

1. US$ 10,000.00 -> US$ 27,008.37, retorno +170.08%
2. US$ 27,008.37 -> US$ 502,463.05, retorno +1,760.40%
3. US$ 502,463.05 -> US$ 3,805,617.56, retorno +657.39%

### Comparação final

Soft Horizon Consensus versus Control:

- diferença de capital: +US$ 262,506.18
- razão de capital: 1.07409, equivalente a +7.41%
- diferença de CAGR: +3.03 pontos percentuais
- diferença de Sharpe: +0.0157
- diferença de maximum drawdown: praticamente zero
- diferença do pior fold: -19.66 pontos percentuais

Assim, no snapshot final limpo, o Soft Horizon Consensus melhora capital
agregado, CAGR e Sharpe, mas não domina o Control em robustez por fold. O ganho
concentra-se principalmente no fold 2; o fold 1 piora e o fold 3 fica
praticamente inalterado.

### Comparação com o resultado anterior de aproximadamente US$ 7.38 milhões

O experimento anterior `raw-split-soft-horizon-consensus-v2` produziu:

- Control: US$ 5,551,143.96
- Soft: US$ 7,376,955.56
- ganho relativo do Soft: +32.89%
- decisões alteradas pelo Soft: 11
- DOC excluído
- CLMT ainda presente no universo

Na execução final, DOC e CLMT estão ambos excluídos.

A comparação das decisões antigas com as decisões finais mostra divergência em
aproximadamente 8% das sessões comuns, inclusive antes de CLMT ser efetivamente
selecionado. Isso é consistente com o fato de que remover um ativo do universo
altera o treinamento, rankings relativos e decisões dos demais ativos, e não
apenas elimina os trades realizados naquele ticker.

No Soft anterior, CLMT participou de 16 posições encerradas. O produto dos
retornos dessas posições isoladamente foi aproximadamente +18.6%, com cerca de
US$ 812 mil de PnL realizado ao longo da trajetória; porém a diferença total
entre os experimentos não pode ser atribuída apenas a esses trades, porque a
remoção do ativo também altera a política aprendida.

### Auditoria RAW + split local versus Alpaca split-adjusted

Foi feita uma comparação usando os arquivos de auditoria Alpaca
`adjustment=split` já existentes no pacote de resultados e o snapshot final
RAW normalizado localmente.

Para os 54 ativos elegíveis da execução final, foram comparadas 135,772 linhas
disponíveis na auditoria anterior.

Nos preços OHLC:

- mediana da diferença relativa: 0%
- percentil 95: aproximadamente 0.0066% a 0.0070%
- percentil 99: aproximadamente 0.0205% a 0.0219%
- diferença máxima: aproximadamente 0.05%
- nenhuma linha teve diferença superior a 0.1%

Em volume:

- mediana: 0%
- percentil 99: aproximadamente 0.00365%
- nenhuma linha teve diferença superior a 1%

Portanto, para o universo elegível final, a queda de capital não é explicada por
uma discrepância material entre `RAW + normalização local de splits` e o
histórico Alpaca `split-adjusted`. A diferença principal é a composição limpa
do universo, em especial a exclusão estrutural de CLMT.

### Interpretação para o TCC

O resultado de aproximadamente US$ 7.38 milhões não deve ser forçado ou
recuperado por novo tuning. Ele dependia de um universo que ainda continha CLMT,
posteriormente classificado como estruturalmente inadequado para o estudo.

A execução final limpa e reproduzível sustenta:

1. desempenho fortemente positivo em todos os folds;
2. uso efetivo de dados Alpaca independentes de MongoDB;
3. snapshot e configuração imutáveis por hash;
4. GPU efetivamente usada no LightGBM;
5. Control final de aproximadamente US$ 3.54 milhões;
6. Soft Horizon Consensus final de aproximadamente US$ 3.81 milhões;
7. ganho agregado moderado do Soft (+7.41%), com trade-off de pior fold.

Esses valores devem ser tratados como o resultado final reproduzível desta
janela, salvo descoberta de erro metodológico independente de desempenho. Não
fazer tuning posterior para recuperar a faixa de US$ 7 milhões.
