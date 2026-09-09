# Validação técnica — API 10.8.41 / pesquisa 2.0.0

Base: `10e2d46b71fc7051098116f6385f6561215cdaad`, branch
`research/asset-marginal-rotation-contribution-v1`.
Entrega: `research/asset-marginal-rotation-contribution-v2`.

## Verificado

`python -m unittest discover -s tests -p test_asset_marginal_v2.py -v`:
**10 testes aprovados**.

- Replay versus `_simulate_exact` e `_utility_policy`: posições idênticas,
  curva de capital com tolerância relativa `1e-12`, taxas e rotações iguais.
  Cobertura de ações inteiras/fracionadas, CASH, empates, permanência mínima,
  margem, taxas, slippage e liquidação final.
- Alterar preços futuros não modifica as ações já determinadas pelos scores;
  rótulos `forward_net_log_return` não participam do novo replay.
- Rejeição de dados posteriores ao corte, labels imaturos, duplicatas,
  preços inválidos e saltos no calendário.
- Candidato incompleto não encurta o calendário obrigatório do baseline.
- Adições complementares são reavaliadas no universo expandido; a soma das
  contribuições aceitas reconcilia com o ganho entre contas inicial e final.
  Um candidato redundante não ganha crédito duplicado.
- Ausência de candidato positivo mantém a mesma conta do baseline.
- Exportação usa a abertura da sessão seguinte e registra maturidade dos labels.
- Arquivos v1 não são convertidos silenciosamente em replay v2.
- CAGR inclui a primeira variação do saldo.
- Alteração de preço em um ativo comum invalida a comparação final.

## Integração sintética

Executada com LightGBM real e os helpers de treino/Leadership da branch:
3 ativos artificiais, 940 barras brutas, 5 folds, 381 sessões OOS e 1.143 linhas
exportadas. O CLI de seleção consumiu o contrato e a fita gerada. Nesse cenário,
nenhum candidato foi admitido: capital original e expandido coincidiram.
Isso confirma o funcionamento da cadeia e não representa desempenho financeiro.

## Anexos reais

A reprodução explícita `legacy_event_sum` preservou a seleção de UTI e sua soma
marginal `0,8134522320833681` no ZIP fornecido. A auditoria independente dos
exports reconciliou os 252 pregões e a diferença final `−199,25656996074576`.
Os CAGRs corrigidos são `−0,028436281949587937` e `−0,04842768229492787`.
O saldo final e as negociações históricas não foram alterados pela auditoria.

`git diff --check` e análise sintática dos arquivos Python aprovados.

## Ambiente e alcance

Validação local: Python 3.12, pandas 2.2.3, NumPy 2.3.5, LightGBM 4.7.0.
O ambiente informado no backtest anterior usa outras versões; os manifestos v2
registram as versões efetivas e o hash do código para permitir essa comparação.

Não foi executado o retreino/backtest real de 56/57 ativos: os ZIPs são exports
de resultados, não a base OHLCV completa. Não há resultado econômico novo para
a v2 nem promessa de recuperar os 23,52 milhões. Repetir o intervalo que já foi
analisado é um teste retrospectivo; a confirmação independente exige dados
reservados para depois do congelamento da nova hipótese.
