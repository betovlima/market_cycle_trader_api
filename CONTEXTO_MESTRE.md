# CONTEXTO_MESTRE — Market Cycle Trader

Última atualização: 2026-09-28. Documento de continuidade; não substitui a verificação de versões realmente implantadas nem os registros por versão em `docs/changes/`.

## Separação de responsabilidades
- **TCC** (`betovlima/tcc_mba_usp_data_science_analytics`): pesquisa, comparação Control/Soft, otimização, validação cronológica e congelamento de evidências.
- **MCT**: aplicação de uso real e execução operacional da política selecionada pelo TCC. Não recriar um ambiente de experimentação ou otimização de estratégias dentro do MCT.
- O MCT usa dados de mercado atualizados da Alpaca; o TCC preserva o snapshot científico em CSV/manifesto no Git. Histórico recalculado e corporate actions precisam ser auditáveis.

## Baseline operacional confirmado pelo usuário
- API de produção: `10.8.38`.
- Front de produção: `10.7.16`.
- A evolução `v10.8.39` está em **branch de desenvolvimento**, não em produção.
- Preservar Winner existente, estado da carteira e ordens. Não promover a Strategy #28 diretamente.

## Resultado científico selecionado
- Fonte de comparação: TCC congelado v1.0.6, commit `c0d71772092f0c26c9f28f0211b9933e9c396b95`, `EXPERIMENT_CHECKPOINT_V1_0_6.md`.
- Control: LightGBM Utility + política-base de rotação + switch margin, sem Soft. Capital final histórico: US$ 10.094.316,30 (inicial US$ 10.000).
- Soft: US$ 9.851.632,93 no mesmo experimento. Control é a referência de implementação operacional desta etapa, não uma previsão de retorno.
- 56 ativos solicitados e 55 elegíveis no snapshot. DOC foi excluído por mudança estrutural de identidade DOC -> PEAK.
- Parâmetros científicos: horizontes (5,10,20,40,60); pesos (0.10,0.15,0.20,0.30,0.25); purge 60; calibração 126; teste 504; min-holding 2; entry edge 0.001; switch margin base 0.0005; candidatos (0,0.0025,0.005,0.01); CPU; LightGBM 329 estimators e demais valores em config científica.
- Para qualquer ativo estruturalmente inválido: excluir com evidência no manifesto; não fazer bridge, reconstruir série ou ajustar manualmente.

## Diagnóstico de integração
- A Strategy #28 é referência de Backtest vinculada ao motor científico `tcc_v106_reference`. O instalador original cria/seleciona somente Research. A API bloqueia promoção desse binding ao Trader.
- O Trader usa outra cadeia: `services/paper_trading.py` -> `engine/live_model_signal.py` -> `engine/live_lightgbm_signal.py` -> `engine/live_policy.py`. Não assumir que copiar parâmetros já gera a mesma decisão.
- O replay conhece a barra posterior para validar execução na próxima abertura; o Trader prepara ordem antes dessa barra existir. Evitar qualquer uso de informação futura.
- Ver `docs/changes/v10.8.35-tcc-reference-main-integration.md`, `v10.8.37-tcc-cutoff-and-independent-tuning.md`, `v10.8.38-tcc-inclusive-oos-session.md`, `v10.8.39-control-operational-parity.md`.

## Trabalho em andamento
- Branch: `feature/v10.8.39-control-operational-parity` a partir de `main` API v10.8.38.
- Causa: o motor de referência científica não é o runtime de Trader e não é elegível para promoção.
- Implementado até aqui: teste isolado de paridade da regra de decisão Control vs política live, incluindo caso de ausência da barra futura; documentação do contrato de migração.
- Não implementado: equivalência ponta a ponta de dados, features, modelos e treinamento, execução isolada, novo binding operacional, alteração do catálogo e ativação da Strategy. Nenhum deploy ou ordem autorizado por estes commits.
- Status: aguarda resultado da CI e demais testes de integração. Não afirmar paridade operacional completa com base no teste de política.

## Próximos passos
1. Confirmar CI do novo teste e baseline de regressão.
2. Isolar o contrato de dados/treinamento/decisão Control v1.0.6 e comparar com o live na mesma janela congelada.
3. Implantar modo de ensaio sem ordens, com resultado sessão a sessão e auditoria de divergência.
4. Criar novo perfil operacional somente após paridade aprovada, mantendo Strategy #28 original imutável.
5. Testar em dados Alpaca atualizados, estado da carteira e integridade de ordens antes de considerar qualquer promoção.

## Comandos Git
```powershell
git fetch origin
git switch main
git pull --ff-only origin main
git switch --track origin/feature/v10.8.39-control-operational-parity
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -p "test_tcc_control_operational_parity.py" -v
python -m unittest discover -s tests -p "test_*.py" -v
```
