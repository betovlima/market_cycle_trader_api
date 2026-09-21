# Git — contribuição marginal v2

API/pacote: `10.8.42`. Experimento: `asset-marginal-rotation-contribution-v2.0.1`.

## Correção 10.8.42 / 2.0.1 — Windows long path

Em 2026-09-20, a Phase 1A concluiu os 82/82 ativos e falhou imediatamente
depois ao reler `intrinsic_timing_summary.csv` com `FileNotFoundError`.
O arquivo havia sido gravado pelo helper compatível com caminhos longos do Windows,
mas o código base ainda usava `pd.read_csv(path)` diretamente nas leituras.
No caminho real do experimento, isso ultrapassou o limite tradicional do Windows.

Correção:
- nova branch: `research/asset-marginal-rotation-contribution-v2.0.1`;
- `research_asset_rotation_leadership.py` agora usa um hook `_read_csv`;
- o wrapper v2 injeta `research_windows_file_io.read_csv`, assim como já fazia
  para as escritas;
- API/pacote incrementado de `10.8.41` para `10.8.42`;
- experimento incrementado de `2.0.0` para `2.0.1`;
- teste de regressão adicionado para confirmar a instalação do I/O seguro.

O erro não indica falha no treinamento de YANG nem nos 82 ativos. O processamento
dos ativos chegou ao fim e a falha ocorreu na etapa de finalização/leitura dos
artefatos produzidos.


Branch de origem: `research/asset-marginal-rotation-contribution-v1`.
Commit de origem: `10e2d46b71fc7051098116f6385f6561215cdaad`.
Nova branch: `research/asset-marginal-rotation-contribution-v2`.

Mensagem do commit:

```text
feat(research): evaluate marginal assets with chronological score replay

Version API/package 10.8.41 and marginal research 2.0.0.
Replace overlapping-label sums with cached OOS account comparisons, verify
market snapshot identity, preserve the legacy selector, and fix CAGR basis.
```

## Obter a branch publicada

```bash
git fetch origin
git switch research/asset-marginal-rotation-contribution-v2
git pull --ff-only origin research/asset-marginal-rotation-contribution-v2
python -m unittest discover -s tests -p test_asset_marginal_v2.py -v
python scripts/research_asset_marginal_rotation_independent_then_validate.py --strategy-sequence 10 --history-start 2016-01-01 --snapshot-end 2026-09-04 --validation-sessions 252 --workers 4 --fresh-run
```

O último comando reavalia o período histórico já observado. Detalhes e limites
estão em `scripts/README_asset_marginal_rotation_contribution_v2.md`.

## Aplicar o patch em outro checkout

Use apenas se estiver importando a entrega por arquivo, sem a branch já publicada.
O patch é relativo ao commit de origem acima.

```bash
git fetch origin
git switch -c research/asset-marginal-rotation-contribution-v2-import 10e2d46b71fc7051098116f6385f6561215cdaad
git apply --check market_cycle_trader_api_v10.8.41_marginal_v2.patch
git apply market_cycle_trader_api_v10.8.41_marginal_v2.patch
git add pyproject.toml src/market_cycle_trader_api/core/config.py src/market_cycle_trader_api/engine/capital_rotation.py src/market_cycle_trader_api/services/asset_marginal_score_replay.py scripts/research_asset_marginal_rotation_contribution.py scripts/research_asset_marginal_rotation_leadership.py scripts/research_asset_marginal_rotation_validation.py scripts/research_asset_marginal_rotation_independent_then_validate.py scripts/research_marginal_reproducibility.py scripts/audit_asset_marginal_rotation_export.py scripts/README_asset_marginal_rotation_contribution.md scripts/README_asset_marginal_rotation_contribution_v2.md tests/test_asset_marginal_v2.py VALIDACAO_ASSET_MARGINAL_V2.md GIT_ASSET_MARGINAL_V2.md
python -m unittest discover -s tests -p test_asset_marginal_v2.py -v
git commit -m "feat(research): evaluate marginal assets with chronological score replay"
git push -u origin research/asset-marginal-rotation-contribution-v2-import
```

A entrega tem como destino a branch de pesquisa. Publicação dessa branch não
representa merge na principal ou implantação da API.


## Retomar após a falha 82/82

Atualize para a branch corrigida antes de repetir a execução:

```bash
git fetch origin
git switch research/asset-marginal-rotation-contribution-v2.0.1
git pull --ff-only origin research/asset-marginal-rotation-contribution-v2.0.1
python -m unittest discover -s tests -p test_asset_marginal_v2.py -v
```

O comando de pesquisa continua sendo:

```bash
python scripts/research_asset_marginal_rotation_independent_then_validate.py --strategy-sequence 10 --history-start 2016-01-01 --snapshot-end 2026-09-04 --validation-sessions 252 --workers 4 --fresh-run
```

Até que exista uma recuperação validada dos artefatos parcialmente finalizados,
não reutilizar silenciosamente a pasta da execução interrompida como se fosse um
resultado completo.
